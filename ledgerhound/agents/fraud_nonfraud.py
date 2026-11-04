"""
ledgerhound.agents.fraud_nonfraud
==============================

The end-to-end fraud x non-fraud EDA cycle (MVP spine).

This orchestrator chains the six steps the product is built around, carrying
a single shared :class:`~ledgerhound.findings.log.FindingsLog` forward so each
step compounds on the last (same pattern as
:class:`~ledgerhound.agents.two_dataset.TwoDatasetPlaybook`)::

    1. Fraud EDA                 step1_fraud_eda()
    2. Non-fraud sampling design  step2_design()      <- from step-1 findings
    3. Fetch non-fraud            step3_plan() then
                                  step3_export(path)  (human runs the SQL) OR
                                  step3_execute()     (HITL "execute button")
    4. Non-fraud EDA             step4_nonfraud_eda(data)
    5. Fraud + non-fraud joint    step5_joint_eda(agent)
    6. Fraud vs non-fraud compare step6_compare(fraud_df, nonfraud_df, feats)

Why staged (not one ``run()``)?
-------------------------------
Step 3 can hand off to a human -- they take the exported SQL, run it in
their own Trino console, and drop the parquet back.  The cycle therefore
*pauses* between step 3 and step 4, so the steps are explicit methods rather
than a single straight-through loop.  :meth:`run_until_fetch` and
:meth:`resume_after_fetch` wrap the two halves for callers that want a
two-call flow; :meth:`run_auto` does the whole thing when a Trino client is
wired and auto-execution is acceptable.

The comparative engine (:func:`compare_fraud_nonfraud`) is pure compute and
is exercised directly by tests; the EDA steps drive an ``EDAAgent`` and are
covered with a fake agent so the orchestration logic is testable offline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from ledgerhound.sampling.design import (
    NonFraudSamplingDesign,
    NonFraudSamplingDesigner,
)
from ledgerhound.trino.sampling_sql import (
    SamplingSQLConfig,
    SamplingSQLPlan,
    build_sql,
    execute as sql_execute,
    export_markdown,
)

if TYPE_CHECKING:
    from ledgerhound.agents.eda_agent import EDAAgent
    from ledgerhound.findings.log import FindingsLog
    from ledgerhound.trino.client import TrinoClient

log = logging.getLogger(__name__)


# Default questions per EDA step. Deliberately domain-general and
# sample-frame-aware; callers can override via the ``questions`` arg.
DEFAULT_QUESTIONS: dict[str, list[str]] = {
    "fraud": [
        "What is the overall fraud rate and how does it vary across the time "
        "window? Tag the finding with the appropriate sample_frame.",
        "Bucket transactions by account_age_days. Is there a fraud-rate lift "
        "in young accounts? Quantify it and bind the finding to that stratum.",
        "Which purpose_code or amount strata are over-represented in fraud? "
        "Record each as a finding bound to its stratum so it can seed the "
        "non-fraud sample.",
    ],
    "nonfraud": [
        "Profile this non-fraud sample: distributions of amount, "
        "account_age_days and purpose_code, with sample_frame tags.",
        "Within each stratum tagged by the _stratum column, summarise the "
        "key feature distributions.",
    ],
    "joint": [
        "Use list_findings to read the fraud-side findings. For each fraud "
        "stratum, contrast the fraud and non-fraud feature distributions.",
        "Which features separate fraud from non-fraud most strongly, and is "
        "the separation consistent across strata?",
    ],
}


# ---------------------------------------------------------------------------
# Step 6 comparative engine (pure compute)
# ---------------------------------------------------------------------------


@dataclass
class FeatureDivergence:
    """How strongly one feature separates fraud from non-fraud."""

    feature: str
    psi: float
    status: str
    n_fraud: int = 0
    n_nonfraud: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "status": self.status,
            "n_fraud": self.n_fraud,
            "n_nonfraud": self.n_nonfraud,
        }


@dataclass
class ComparisonResult:
    """Step-6 result: features ranked by fraud-vs-non-fraud divergence."""

    divergences: list[FeatureDivergence] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    excluded_matched: list[str] = field(default_factory=list)
    derived_features: list[str] = field(default_factory=list)

    def top(self, n: int = 10) -> list[FeatureDivergence]:
        return self.divergences[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergences": [d.to_dict() for d in self.divergences],
            "skipped": list(self.skipped),
            "excluded_matched": list(self.excluded_matched),
            "derived_features": list(self.derived_features),
        }

    def to_markdown(self) -> str:
        lines = ["## Fraud vs non-fraud feature divergence (PSI)", ""]
        lines.append("| Feature | PSI | Status | n_fraud | n_nonfraud |")
        lines.append("|---|---|---|---|---|")
        derived = set(self.derived_features)
        for d in self.divergences:
            tag = " _(derived)_" if d.feature in derived else ""
            lines.append(
                f"| `{d.feature}`{tag} | {d.psi:.4f} | {d.status} | "
                f"{d.n_fraud} | {d.n_nonfraud} |"
            )
        if self.derived_features:
            lines.append("")
            lines.append(
                "> _(derived)_ features are **EDA-derived and NOT "
                "leakage-validated** -- exploratory only. Hand them to the "
                "feature-engineering team to re-implement with point-in-time / "
                f"leakage rigor: {', '.join('`' + c + '`' for c in self.derived_features)}."
            )
        if self.excluded_matched:
            lines.append("")
            lines.append(
                "> **Excluded from the comparison (matched/overmatched):** "
                f"{', '.join('`' + c + '`' for c in self.excluded_matched)}. "
                "The non-fraud sample was matched on these columns, so they "
                "are equalised by construction -- their divergence would be "
                "meaningless. Use a representative sample to compare them."
            )
        if self.skipped:
            lines.append("")
            lines.append(f"*Skipped (missing/constant): {', '.join(self.skipped)}*")
        return "\n".join(lines)


_ID_LABEL_DENYLIST = {
    "is_fraud", "fraud", "label", "target", "y",
    "_stratum", "_rn", "_cap", "_grp", "_tot",
}


def build_joint_dataframe(
    fraud_df: Any, nonfraud_df: Any, *, group_col: str = "_group"
) -> Any:
    """Stack fraud + non-fraud into one frame for the step-5 joint EDA.

    Both sides are tagged in ``group_col`` ("fraud" / "nonfraud") and the
    result is restricted to the columns common to both, so a single EDA agent
    can ask cross-cutting questions over the combined view.
    """
    import pandas as pd

    f = fraud_df.copy()
    n = nonfraud_df.copy()
    f[group_col] = "fraud"
    n[group_col] = "nonfraud"
    common = [c for c in f.columns if c in set(n.columns)]
    return pd.concat([f[common], n[common]], ignore_index=True)


@dataclass
class SegmentedComparison:
    """Step-6 broken down by a segment column (3b: is the signal consistent?)."""

    segment_col: str
    by_segment: dict[str, "ComparisonResult"] = field(default_factory=dict)
    skipped_segments: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"### Breakdown by `{self.segment_col}`", ""]
        if not self.by_segment:
            lines.append("_(no segment had enough rows on both sides)_")
            return "\n".join(lines)
        for seg, comp in self.by_segment.items():
            top = comp.top(3)
            summary = ", ".join(f"`{d.feature}`={d.psi:.2f}" for d in top) or "-"
            lines.append(f"- **{seg}** (top: {summary})")
        if self.skipped_segments:
            lines.append("")
            lines.append(f"*Segments skipped (too few rows): "
                         f"{', '.join(self.skipped_segments)}*")
        return "\n".join(lines)


def compare_by_segment(
    fraud_df: Any,
    nonfraud_df: Any,
    numeric_features: list[str],
    segment_col: str,
    *,
    n_buckets: int = 10,
    exclude: Optional[list[str]] = None,
    derived: Optional[list[str]] = None,
    min_rows: int = 50,
) -> SegmentedComparison:
    """Run the fraud-vs-non-fraud comparison within each value of a segment.

    Reveals whether a signal is consistent across segments (e.g. amount
    diverges within high-risk purpose codes but not elsewhere). Segments with
    fewer than ``min_rows`` on either side are skipped (PSI is unreliable on
    tiny samples).
    """
    seg = SegmentedComparison(segment_col=segment_col)
    fcols = getattr(fraud_df, "columns", [])
    ncols = getattr(nonfraud_df, "columns", [])
    if segment_col not in fcols or segment_col not in ncols:
        return seg
    f_vals = set(fraud_df[segment_col].dropna().unique())
    n_vals = set(nonfraud_df[segment_col].dropna().unique())
    for v in sorted(f_vals & n_vals, key=str):
        f_sub = fraud_df[fraud_df[segment_col] == v]
        n_sub = nonfraud_df[nonfraud_df[segment_col] == v]
        if len(f_sub) < min_rows or len(n_sub) < min_rows:
            seg.skipped_segments.append(str(v))
            continue
        seg.by_segment[str(v)] = compare_fraud_nonfraud(
            f_sub, n_sub, numeric_features,
            n_buckets=n_buckets, exclude=exclude, derived=derived)
    return seg


def auto_numeric_features(fraud_df: Any, nonfraud_df: Any) -> list[str]:
    """Numeric columns present in BOTH frames, minus ids / labels / helpers.

    This is what step 6 compares when no explicit feature list is given -- it
    adapts to whatever schema the Trino extract actually returned, instead of
    you naming columns that may not exist.
    """
    import pandas as pd

    nf_cols = set(getattr(nonfraud_df, "columns", []))
    feats: list[str] = []
    for c in getattr(fraud_df, "columns", []):
        if c not in nf_cols:
            continue
        lc = str(c).lower()
        if lc in _ID_LABEL_DENYLIST or lc.endswith("_id") or lc == "id":
            continue
        if (pd.api.types.is_numeric_dtype(fraud_df[c])
                and pd.api.types.is_numeric_dtype(nonfraud_df[c])):
            feats.append(c)
    return feats


def compare_fraud_nonfraud(
    fraud_df: Any,
    nonfraud_df: Any,
    numeric_features: list[str],
    *,
    n_buckets: int = 10,
    exclude: Optional[list[str]] = None,
    derived: Optional[list[str]] = None,
) -> ComparisonResult:
    """Rank numeric features by how much non-fraud diverges from fraud.

    For each feature, bucket boundaries are computed on the *union* of both
    groups, the fraud distribution is used as the PSI reference (population),
    and the non-fraud series is scored against it.  A high PSI means the
    feature distribution differs sharply between fraud and non-fraud -- i.e.
    the feature is a strong candidate fraud signal.

    ``exclude`` lists columns the sample was matched on; they are dropped from
    the ranking (and reported separately) because matching equalises them by
    construction, so their divergence is meaningless (overmatching).
    """
    import pandas as pd  # local import: keep the module importable without pandas

    from ledgerhound.sampling.psi import (
        PSICalculator,
        compute_bucket_bounds,
        _numeric_sample_dist,
    )

    exclude_set = set(exclude or [])
    calc = PSICalculator()
    divergences: list[FeatureDivergence] = []
    skipped: list[str] = []
    excluded_matched: list[str] = []

    for feat in numeric_features:
        if feat in exclude_set:
            excluded_matched.append(feat)
            continue
        if feat not in getattr(fraud_df, "columns", []) or feat not in getattr(
            nonfraud_df, "columns", []
        ):
            skipped.append(feat)
            continue
        fraud_s = fraud_df[feat].dropna().astype(float)
        nonfraud_s = nonfraud_df[feat].dropna().astype(float)
        if fraud_s.empty or nonfraud_s.empty:
            skipped.append(feat)
            continue
        combined = pd.concat([fraud_s, nonfraud_s], ignore_index=True)
        bounds = compute_bucket_bounds(combined, n_buckets=n_buckets)
        if len(bounds) < 3:
            skipped.append(feat)
            continue
        fraud_dist = _numeric_sample_dist(fraud_s, bounds)
        fp = calc.numeric(
            feat, fraud_dist, nonfraud_df[feat], bounds, n_population=len(fraud_s)
        )
        divergences.append(
            FeatureDivergence(
                feature=feat,
                psi=fp.psi,
                status=fp.status.value,
                n_fraud=len(fraud_s),
                n_nonfraud=len(nonfraud_s),
            )
        )

    divergences.sort(key=lambda d: -d.psi)
    return ComparisonResult(divergences=divergences, skipped=skipped,
                            excluded_matched=excluded_matched,
                            derived_features=list(derived or []))


# ---------------------------------------------------------------------------
# Step results / cycle result
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """One EDA step's contribution."""

    name: str
    findings: list[dict] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    finding_warnings: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    questions_run: int = 0
    questions_succeeded: int = 0
    elapsed_ms: float = 0.0


@dataclass
class FraudNonFraudResult:
    """Aggregate of whatever steps have run so far."""

    steps: list[StepResult] = field(default_factory=list)
    design: Optional[NonFraudSamplingDesign] = None
    sql_markdown_path: Optional[str] = None
    comparison: Optional[ComparisonResult] = None

    @property
    def total_findings(self) -> int:
        return sum(len(s.finding_ids) for s in self.steps)

    def to_markdown(self) -> str:
        lines = ["# Fraud x non-fraud EDA cycle", ""]
        lines.append(f"*{self.total_findings} findings across {len(self.steps)} steps*")
        for s in self.steps:
            lines.append(f"\n## {s.name}")
            lines.append(
                f"*{s.questions_succeeded}/{s.questions_run} questions, "
                f"{len(s.finding_ids)} findings*"
            )
            for f in s.findings:
                lines.append(f"\n### Q: {f.get('question', '?')}")
                lines.append(f.get("answer", "") or f.get("error", "_(no answer)_"))
        if self.design is not None:
            lines.append("\n## Sampling design")
            lines.append(self.design.summary_line())
            if self.sql_markdown_path:
                lines.append(f"\nSQL plan exported to `{self.sql_markdown_path}`.")
        if self.comparison is not None:
            lines.append("\n" + self.comparison.to_markdown())
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FraudNonFraudCycle:
    """Staged orchestrator for the six-step fraud x non-fraud EDA cycle.

    Parameters
    ----------
    fraud_agent
        An :class:`EDAAgent` already bound to the fraud dataset and the
        shared ``findings_log``.
    findings_log
        The single log shared by every step, so findings compound.
    designer
        Configured :class:`NonFraudSamplingDesigner` for step 2.
    sql_config
        :class:`SamplingSQLConfig` (source table + non-fraud filter) for
        step 3.
    nonfraud_agent_factory
        ``callable(data_path) -> EDAAgent`` used in step 4 to build an agent
        over the freshly-fetched non-fraud data (sharing ``findings_log``).
        Optional until step 4 is reached.
    trino_client
        Optional :class:`TrinoClient` for the execute path of step 3.
    questions
        Override the per-step default question lists.
    interactive
        Stream progress to the console.
    """

    def __init__(
        self,
        fraud_agent: "EDAAgent",
        findings_log: "FindingsLog",
        *,
        designer: NonFraudSamplingDesigner,
        sql_config: SamplingSQLConfig,
        nonfraud_agent_factory: Optional[Callable[[str], "EDAAgent"]] = None,
        trino_client: Optional["TrinoClient"] = None,
        questions: Optional[dict[str, list[str]]] = None,
        interactive: bool = True,
    ) -> None:
        self.fraud_agent = fraud_agent
        self.findings_log = findings_log
        self.designer = designer
        self.sql_config = sql_config
        self.nonfraud_agent_factory = nonfraud_agent_factory
        self.trino_client = trino_client
        self.questions = questions or DEFAULT_QUESTIONS
        self.interactive = interactive

        self.result = FraudNonFraudResult()
        self.design: Optional[NonFraudSamplingDesign] = None
        self.plan: Optional[SamplingSQLPlan] = None

    # -- step 1 --------------------------------------------------------------

    def step1_fraud_eda(self) -> StepResult:
        step = self._run_eda("Step 1 - Fraud EDA", self.fraud_agent,
                             self.questions.get("fraud", []))
        self.result.steps.append(step)
        return step

    # -- step 2 --------------------------------------------------------------

    def step2_design(
        self,
        *,
        target_dataset: str = "nonfraud",
        parent_sample: str = "fraud",
        fraud_stratum_counts: Optional[dict[str, int]] = None,
    ) -> NonFraudSamplingDesign:
        """Derive the non-fraud sampling design from the shared findings log."""
        self.design = self.designer.design(
            self.findings_log,
            target_dataset=target_dataset,
            parent_sample=parent_sample,
            fraud_stratum_counts=fraud_stratum_counts,
        )
        self.result.design = self.design
        if self.interactive:
            log.info(self.design.summary_line())
            for w in self.design.warnings:
                log.warning("design: %s", w)
        return self.design

    # -- step 3 --------------------------------------------------------------

    def step3_plan(self) -> SamplingSQLPlan:
        if self.design is None:
            raise RuntimeError("Call step2_design() before step3_plan().")
        self.plan = build_sql(self.design, self.sql_config)
        return self.plan

    def step3_export(self, path: str | Path) -> Path:
        """Human path: write the strategy + SQL Markdown for manual running."""
        if self.plan is None:
            self.step3_plan()
        out = export_markdown(self.plan, path)
        self.result.sql_markdown_path = str(out)
        return out

    def step3_execute(self, *, run_now: bool = False):
        """Us path: submit the SQL through the HITL gate ("execute button").

        With ``run_now`` the approved job is executed immediately and the
        result parquet path is returned; otherwise the registered job is
        returned for the caller to execute later.
        """
        if self.trino_client is None:
            raise RuntimeError("step3_execute requires a trino_client.")
        if self.plan is None:
            self.step3_plan()
        job = sql_execute(self.plan, self.trino_client)
        if run_now:
            return self.trino_client.execute_job(job)
        return job

    # -- step 4 --------------------------------------------------------------

    def step4_nonfraud_eda(self, nonfraud_data: str | Path) -> StepResult:
        if self.nonfraud_agent_factory is None:
            raise RuntimeError("step4 needs a nonfraud_agent_factory.")
        agent = self.nonfraud_agent_factory(str(nonfraud_data))
        step = self._run_eda("Step 4 - Non-fraud EDA", agent,
                             self.questions.get("nonfraud", []))
        self.result.steps.append(step)
        return step

    # -- step 5 --------------------------------------------------------------

    def step5_joint_eda(self, joint_agent: "EDAAgent") -> StepResult:
        """Joint EDA over a fraud+non-fraud view (agent supplied by caller)."""
        step = self._run_eda("Step 5 - Joint EDA", joint_agent,
                             self.questions.get("joint", []))
        self.result.steps.append(step)
        return step

    # -- step 6 --------------------------------------------------------------

    def step6_compare(
        self,
        fraud_df: Any,
        nonfraud_df: Any,
        numeric_features: list[str],
        *,
        n_buckets: int = 10,
        exclude: Optional[list[str]] = None,
    ) -> ComparisonResult:
        # If the non-fraud sample was matched, exclude the matched columns:
        # matching equalises them, so their divergence is meaningless.
        if exclude is None and self.design is not None:
            exclude = self.design.matched_columns
        comp = compare_fraud_nonfraud(
            fraud_df, nonfraud_df, numeric_features,
            n_buckets=n_buckets, exclude=exclude,
        )
        self.result.comparison = comp
        return comp

    # -- convenience wrappers ------------------------------------------------

    def run_until_fetch(
        self,
        export_path: str | Path,
        *,
        fraud_stratum_counts: Optional[dict[str, int]] = None,
    ) -> Path:
        """Steps 1-3 (export): fraud EDA -> design -> SQL Markdown for a human."""
        self.step1_fraud_eda()
        self.step2_design(fraud_stratum_counts=fraud_stratum_counts)
        return self.step3_export(export_path)

    def resume_after_fetch(
        self,
        nonfraud_data: str | Path,
        joint_agent: "EDAAgent",
        fraud_df: Any,
        nonfraud_df: Any,
        numeric_features: list[str],
    ) -> FraudNonFraudResult:
        """Steps 4-6 once the non-fraud data has been fetched."""
        self.step4_nonfraud_eda(nonfraud_data)
        self.step5_joint_eda(joint_agent)
        self.step6_compare(fraud_df, nonfraud_df, numeric_features)
        return self.result

    # -- internals -----------------------------------------------------------

    def _run_eda(self, name: str, agent: "EDAAgent", questions: list[str]) -> StepResult:
        step = StepResult(name=name)
        t0 = time.perf_counter()
        for q in questions:
            step.questions_run += 1
            try:
                ar = agent.ask(q, interactive=self.interactive)
                step.findings.append({
                    "question": q,
                    "answer": getattr(ar, "answer", ""),
                    "finding_ids": list(getattr(ar, "finding_ids", [])),
                    "status": "success",
                })
                step.finding_ids.extend(getattr(ar, "finding_ids", []))
                step.finding_warnings.extend(getattr(ar, "finding_warnings", []))
                step.charts.extend(getattr(ar, "charts", []))
                step.questions_succeeded += 1
            except Exception as exc:  # pragma: no cover - defensive
                step.findings.append({"question": q, "error": str(exc),
                                      "status": "failed"})
                log.warning("EDA question failed: %s", exc)
        step.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return step
