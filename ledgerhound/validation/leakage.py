"""
ledgerhound.validation.leakage
============================

LeakageSentinel -- V3-018: vets a proposed feature set for target leakage
before it can enter the knowledge base as a "win". Runs as a deterministic,
rule-based check -- there is no LLM reasoning inside this module, which is
exactly what satisfies the roadmap's "runs in a SEPARATE agent context from
the feature engineer" requirement: a hard rule can't be talked out of a
verdict by the same reasoning trace that proposed the feature.

Why this matters
-----------------
A model trained on a leaky feature looks like a huge win on the split it was
built and tested on, gets written to the dataset card / knowledge base as a
"discovery", and then poisons every future study that reuses that KB entry.
Leakage Sentinel is the gate V3-020 (feature selection) and V3-021 (modeler)
sit behind -- see EPIC 5's mandatory ordering note in the v3 roadmap.

Four checks, each independent
------------------------------
1. POINT-IN-TIME VETTING -- every feature declares when it is computable
   (``pre_event`` / ``post_event`` / ``unknown``). ``post_event`` is vetoed
   outright: the feature could only exist using information not available
   at scoring time. ``unknown`` is challenged, not vetoed -- an undeclared
   spec is a process failure, not proof of a leak.
2. TARGET-LEAK LEXICON -- column name pattern matching (e.g. ``chargeback_*``,
   ``*_closed_reason``). This OVERRIDES a ``pre_event`` declaration: a name
   that screams "outcome of the fraud investigation" is vetoed even if the
   engineer marked it pre-event, on the theory that a name that suspicious is
   more likely a mislabeled spec than a false positive.
   KNOWN LIMITATION (intentional, not yet solved): this is a blunt,
   unconditional veto. A legitimate historical aggregate like
   ``chargeback_count_prior_90d`` -- genuinely knowable pre-event and often
   one of the most predictive real features in fraud modelling -- will match
   ``chargeback_*`` and get vetoed anyway. A ``pre_event`` declaration does
   NOT currently rescue a lexicon match. Left as-is per the roadmap's literal
   spec; flagged here for whoever tunes the lexicon next.
3. DEV-VS-OOT AUROC GAP -- a STUDY-level check (not per-feature): if a
   model/score scores much better on the dev split than on out-of-time data,
   that is diagnostic of an in-sample-only leak (temporal instability, ID
   memorization, dev-period-specific artifacts) -- NOT the same failure mode
   as a target leak baked identically into both splits (that one scores near-
   perfect on BOTH dev and OOT, so the gap is near zero and this check
   correctly stays silent; the lexicon/point-in-time checks are what catch
   that case). The two checks are complementary, not redundant.
4. DOMINANT-FEATURE INTERROGATION -- a single feature that accounts for an
   outsized share of total importance gets CHALLENGED (not auto-vetoed --
   a dominant feature is sometimes just a genuinely strong, clean signal).
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()


class ComputedAt:
    PRE_EVENT = "pre_event"
    POST_EVENT = "post_event"
    UNKNOWN = "unknown"


class FitScope:
    """Where a GROUP-AGGREGATE feature's statistic was fit.

    Only meaningful for features that aggregate the target/outcome by a
    grouping key (e.g. a bank- or IFSC-level fraud rate) -- a row-wise
    transform like a ratio of two columns has no "fit" scope and should
    leave `FeatureSpec.is_group_aggregate` False, which skips this check
    entirely (declaring fit_scope on a non-aggregate feature would just be
    noise). Added 2026-08-05 (V3-032): a real pilot reviewer flagged exactly
    this failure mode -- IFSC-level fraud-rate aggregates fit on the whole
    dataset instead of train-only, leaking test-set outcome into a feature.
    """
    TRAIN_ONLY = "train_only"
    FULL_DATASET = "full_dataset"
    UNKNOWN = "unknown"


class LeakVerdict:
    CLEAR = "CLEAR"
    CHALLENGED = "CHALLENGED"
    VETOED = "VETOED"


_VERDICT_RANK = {LeakVerdict.CLEAR: 0, LeakVerdict.CHALLENGED: 1, LeakVerdict.VETOED: 2}


def _worse(a: str, b: str) -> str:
    """Return whichever of two verdicts is more severe."""
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


# Column-name glob patterns that suggest the feature encodes the outcome of
# the very event being predicted, or something downstream of it. Matched
# case-insensitively against the full feature name via fnmatch.
DEFAULT_LEAK_LEXICON: tuple[str, ...] = (
    "*_closed_reason",
    "chargeback_*",
    "*_chargeback",
    "*chargeback_flag*",
    "*dispute_outcome*",
    "*dispute_resolution*",
    "*settlement_status*",
    "*writeoff*",
    "*write_off*",
    "*_after_event",
    "post_event_*",
    "*investigation_result*",
    "*case_outcome*",
    "*fraud_confirmed*",
    "*confirmed_fraud*",
    # Added 2026-08-05 from real pilot feedback (V3-032). Risk Score is a
    # downstream MODEL output in the pilot's pipeline, not an analytical
    # signal -- a reviewer said its misinterpretation "impacts one of the
    # major conclusions" of a report. `*_fraud_rate*` / `*fraud_signal*`
    # covers the specific named features the pilot flagged as unusable
    # (payer_bank_fraud_rate, payee_ifsc_bank_fraud_rate,
    # fraud_signal_composite) -- these are exactly the shape of aggregate
    # that risks train/test leakage when fit on the full dataset (see the
    # fit_scope check on FeatureSpec below for the data-level version of
    # this same concern).
    "*risk_score*",
    # Real column names seen in production do not always carry the
    # underscore -- `riskscore_float` (the actual UPI mule dataset column)
    # matched NONE of these patterns before, so the single most-called-out
    # leak class in the system prompt could not be caught by name.
    "*riskscore*",
    "*risk_scr*",
    "*model_score*",
    "*_score_v[0-9]*",
    "*propensity*",
    "*_fraud_rate*",
    "fraud_rate_*",
    "*fraud_signal*",
)


@dataclass
class FeatureSpec:
    """One feature under review. `computed_at` is the engineer's declared
    as-of-time spec; `as_of_note` is free text explaining it (e.g. "30d
    trailing window ending at txn_ts, no look-ahead").

    `inputs` (added V3-032): the raw/upstream columns this feature was
    built from, if known -- lets `vet()` catch a feature built directly
    from the label column even under a name that matches no lexicon
    pattern (a real pilot bug: features generated from `Fraud_Flag`, the
    target itself). Optional and additive -- an empty list just means this
    check doesn't apply to that spec, exactly like `computed_at=UNKNOWN`
    doesn't auto-veto.

    `is_group_aggregate` + `fit_scope` (added V3-032): declare these only
    for a feature that aggregates the target/outcome by a grouping key
    (bank/IFSC/device rate-style features). See `FitScope`.
    """
    name: str
    computed_at: str = ComputedAt.UNKNOWN
    as_of_note: str = ""
    inputs: list = field(default_factory=list)
    is_group_aggregate: bool = False
    fit_scope: str = FitScope.UNKNOWN


@dataclass
class FeatureVerdict:
    feature: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeakageReport:
    generated_at: str
    n_features: int = 0
    n_clear: int = 0
    n_challenged: int = 0
    n_vetoed: int = 0
    feature_verdicts: list[FeatureVerdict] = field(default_factory=list)
    auroc_gap: Optional[dict] = None
    overall_blocked: bool = False
    notes: str = ""

    def is_clean(self) -> bool:
        return self.n_vetoed == 0 and not self.overall_blocked

    def vetoed_features(self) -> list[str]:
        return [v.feature for v in self.feature_verdicts if v.verdict == LeakVerdict.VETOED]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def display(self) -> None:
        status = "BLOCKED" if self.overall_blocked else "CLEAR"
        color = "red" if self.overall_blocked else "green"
        console.print(f"\n[bold {color}]Leakage Sentinel -- {status}[/bold {color}]")
        console.print(
            f"  {self.n_clear} clear  {self.n_challenged} challenged  "
            f"{self.n_vetoed} vetoed\n"
        )
        t = Table("Feature", "Verdict", "Reasons", show_header=True, header_style="bold dim")
        clr = {LeakVerdict.CLEAR: "green", LeakVerdict.CHALLENGED: "yellow", LeakVerdict.VETOED: "red"}
        for v in self.feature_verdicts:
            color = clr.get(v.verdict, "")
            t.add_row(v.feature, f"[{color}]{v.verdict}[/{color}]", "; ".join(v.reasons)[:80])
        console.print(t)
        if self.auroc_gap is not None:
            g = self.auroc_gap
            console.print(
                f"\n  AUROC gap: dev={g.get('auroc_dev'):.3f} oot={g.get('auroc_oot'):.3f} "
                f"gap={g.get('gap'):.3f} blocked={g.get('blocked')}"
            )


class LeakageSentinel:
    """
    Vets a proposed feature set for target leakage.

    Parameters
    ----------
    lexicon             : Column-name glob patterns treated as target-leak
                          signatures. Defaults to DEFAULT_LEAK_LEXICON.
    auroc_gap_threshold : Dev-minus-OOT AUROC gap above which the study is
                          blocked. Default 0.05 (5 points), per the roadmap.
    dominance_threshold : Share of total feature importance above which a
                          single feature is challenged. Default 0.5.
    """

    def __init__(
        self,
        lexicon: Optional[Sequence[str]] = None,
        auroc_gap_threshold: float = 0.05,
        dominance_threshold: float = 0.5,
    ) -> None:
        self.lexicon = list(lexicon) if lexicon is not None else list(DEFAULT_LEAK_LEXICON)
        self.auroc_gap_threshold = auroc_gap_threshold
        self.dominance_threshold = dominance_threshold

    # -- public entrypoint ---------------------------------------------------

    def vet(
        self,
        specs: Sequence[FeatureSpec],
        y_dev=None,
        scores_dev=None,
        y_oot=None,
        scores_oot=None,
        feature_importances: Optional[dict] = None,
        label_column: Optional[str] = None,
    ) -> LeakageReport:
        """Run all checks and return a LeakageReport.

        `y_dev`/`scores_dev`/`y_oot`/`scores_oot` are 1-D array-likes (labels
        and model scores on the dev and out-of-time splits respectively).
        The AUROC-gap check runs only when all four are provided -- it is a
        STUDY-level check, independent of any single feature's verdict.

        `feature_importances` is an optional {feature_name: importance}
        dict driving the dominant-feature interrogation.

        `label_column` (V3-032): the target column's name. When given,
        any spec whose `inputs` contains it is vetoed outright -- a feature
        built directly from the label, regardless of what it's named.
        """
        verdicts: dict[str, FeatureVerdict] = {}
        for spec in specs:
            verdicts[spec.name] = self._vet_one(spec, label_column)

        if feature_importances:
            self._apply_dominance_check(verdicts, feature_importances)

        auroc_gap = None
        overall_blocked = any(
            v.verdict == LeakVerdict.VETOED for v in verdicts.values()
        )

        if (
            y_dev is not None
            and scores_dev is not None
            and y_oot is not None
            and scores_oot is not None
        ):
            auroc_gap = self._check_auroc_gap(y_dev, scores_dev, y_oot, scores_oot)
            if auroc_gap.get("blocked"):
                overall_blocked = True

        ordered = [verdicts[s.name] for s in specs]
        n_clear = sum(1 for v in ordered if v.verdict == LeakVerdict.CLEAR)
        n_challenged = sum(1 for v in ordered if v.verdict == LeakVerdict.CHALLENGED)
        n_vetoed = sum(1 for v in ordered if v.verdict == LeakVerdict.VETOED)

        return LeakageReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            n_features=len(ordered),
            n_clear=n_clear,
            n_challenged=n_challenged,
            n_vetoed=n_vetoed,
            feature_verdicts=ordered,
            auroc_gap=auroc_gap,
            overall_blocked=overall_blocked,
        )

    # -- per-feature checks ---------------------------------------------------

    def _vet_one(self, spec: FeatureSpec, label_column: Optional[str] = None) -> FeatureVerdict:
        verdict = LeakVerdict.CLEAR
        reasons: list[str] = []

        # 1. Point-in-time vetting
        if spec.computed_at == ComputedAt.POST_EVENT:
            verdict = _worse(verdict, LeakVerdict.VETOED)
            reasons.append(
                "declared post_event: computable only with information not "
                "available at scoring time"
            )
        elif spec.computed_at == ComputedAt.UNKNOWN:
            verdict = _worse(verdict, LeakVerdict.CHALLENGED)
            reasons.append(
                "as-of-time spec not declared; cannot confirm point-in-time safety"
            )

        # 2. Target-leak lexicon (can override a pre_event declaration)
        match = self._lexicon_match(spec.name)
        if match is not None:
            verdict = _worse(verdict, LeakVerdict.VETOED)
            reasons.append(f"column name matches target-leak lexicon pattern '{match}'")

        # 3. Direct target derivation (V3-032) -- catches a feature built
        # from the label column under a name the lexicon would never match.
        # A real pilot bug: features generated from `Fraud_Flag` itself.
        if label_column and label_column in spec.inputs:
            verdict = _worse(verdict, LeakVerdict.VETOED)
            reasons.append(
                f"built directly from the label column '{label_column}' -- "
                "a feature cannot use the target as its own input"
            )

        # 4. Group-aggregate fit scope (V3-032) -- only applies to features
        # explicitly declared as aggregating the target/outcome by a
        # grouping key (bank/IFSC-style rate features). A real pilot bug:
        # such aggregates fit on the full dataset instead of train-only.
        if spec.is_group_aggregate:
            if spec.fit_scope == FitScope.FULL_DATASET:
                verdict = _worse(verdict, LeakVerdict.VETOED)
                reasons.append(
                    "group-aggregate feature fit on the FULL dataset -- "
                    "must be fit on train only and applied to val/test"
                )
            elif spec.fit_scope == FitScope.UNKNOWN:
                verdict = _worse(verdict, LeakVerdict.CHALLENGED)
                reasons.append(
                    "group-aggregate feature with no declared fit_scope; "
                    "cannot confirm it was fit on train only"
                )

        return FeatureVerdict(
            feature=spec.name,
            verdict=verdict,
            reasons=reasons,
            detail={"computed_at": spec.computed_at, "as_of_note": spec.as_of_note,
                    "is_group_aggregate": spec.is_group_aggregate,
                    "fit_scope": spec.fit_scope if spec.is_group_aggregate else None},
        )

    def _lexicon_match(self, name: str) -> Optional[str]:
        lname = name.lower()
        for pattern in self.lexicon:
            if fnmatch.fnmatch(lname, pattern.lower()):
                return pattern
        return None

    # -- dominant-feature interrogation ---------------------------------------

    def _apply_dominance_check(
        self, verdicts: dict[str, FeatureVerdict], feature_importances: dict
    ) -> None:
        total = sum(abs(v) for v in feature_importances.values())
        if total <= 0:
            return
        top_feature, top_importance = max(
            feature_importances.items(), key=lambda kv: abs(kv[1])
        )
        share = abs(top_importance) / total
        if share > self.dominance_threshold and top_feature in verdicts:
            fv = verdicts[top_feature]
            fv.verdict = _worse(fv.verdict, LeakVerdict.CHALLENGED)
            fv.reasons.append(
                f"dominant feature: {share:.1%} of total importance "
                f"(threshold {self.dominance_threshold:.0%}) -- interrogate before trusting"
            )
            fv.detail["importance_share"] = share

    # -- study-level AUROC gap -------------------------------------------------

    def _check_auroc_gap(self, y_dev, scores_dev, y_oot, scores_oot) -> dict:
        from sklearn.metrics import roc_auc_score

        auroc_dev = float(roc_auc_score(y_dev, scores_dev))
        auroc_oot = float(roc_auc_score(y_oot, scores_oot))
        gap = auroc_dev - auroc_oot
        blocked = gap > self.auroc_gap_threshold
        message = (
            f"dev AUROC {auroc_dev:.3f} vs OOT AUROC {auroc_oot:.3f} "
            f"(gap {gap:+.3f}, threshold {self.auroc_gap_threshold:.2f}). "
        )
        if blocked:
            message += (
                "Gap exceeds threshold: performance that does not survive the "
                "temporal boundary is diagnostic of an in-sample-only leak "
                "(temporal instability, ID memorization, or a dev-period-"
                "specific artifact) -- distinct from a leak baked identically "
                "into both splits, which this check cannot see (near-zero gap)."
            )
        else:
            message += "Within threshold."
        return {
            "auroc_dev": auroc_dev,
            "auroc_oot": auroc_oot,
            "gap": gap,
            "threshold": self.auroc_gap_threshold,
            "blocked": blocked,
            "message": message,
        }
