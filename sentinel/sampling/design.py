"""
sentinel.sampling.design
=========================

Derive a NON-FRAUD sampling design *from the fraud EDA findings themselves*.

Where this sits in the end-to-end EDA cycle
--------------------------------------------
::

    1. Fraud EDA            -> FindingsLog (fraud side)
    2. THIS MODULE          -> NonFraudSamplingDesign   <-- you are here
    3. design -> Trino SQL  -> fetch non-fraud (human-approve or auto-run)
    4. Non-fraud EDA
    5. Fraud + non-fraud joint EDA
    6. Fraud vs non-fraud comparative EDA

Why derive the design from fraud findings?
------------------------------------------
A random non-fraud sample makes the comparative EDA (step 6) *confounded*:
fraud and non-fraud differ on amount, account age, purpose code and time
window, so any "fraud looks different here" claim is partly a sampling
artifact rather than fraud behaviour.

Sentinel's fraud-side findings already record **which strata the fraud
signal lives in** (``Finding.stratum``) and **which datasets a finding can
seed a sampling spec for** (``Finding.enables_sampling_for``).  This module
reads those structured hooks and turns them into a per-stratum non-fraud
sampling plan, so the non-fraud pull mirrors the fraud strata and the
step-6 comparison is defensible.

Output contract
---------------
``NonFraudSamplingDesign.to_validator_dict()`` returns exactly the
``{"strata": {...}, "iid_reservoir_exists": bool}`` shape that
``sentinel.discipline.sample_frame.SampleFrameValidator`` consumes, so the
design plugs straight into the existing sample-frame discipline without an
adapter.

This module is intentionally pure-Python (no pandas / numpy / LLM), so it is
fully testable offline and never blocks CI or an on-prem install.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from sentinel.findings.models import SampleFrame

logger = logging.getLogger(__name__)

# SQL tokens that are operators/keywords, not column names -- stripped when
# extracting the columns a stratum predicate matches on.
_SQL_KEYWORDS = {
    "and", "or", "not", "in", "between", "like", "is", "null",
    "true", "false", "as",
}


def _columns_in_predicate(predicate: Optional[str]) -> list[str]:
    """Best-effort extraction of column identifiers from a SQL predicate.

    e.g. "account_age_days < 90 AND amount > 50000" -> ["account_age_days",
    "amount"]. Used to know which columns a matched/mirror design equalises,
    so they can be excluded from the step-6 comparison (overmatching guard).
    """
    if not predicate:
        return []
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", predicate)
    out: list[str] = []
    for t in toks:
        if t.lower() in _SQL_KEYWORDS or t in out:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SamplingStrategy(str, Enum):
    """How the non-fraud sample is drawn relative to the fraud strata."""

    # One non-fraud control row per fraud row, inside the same stratum.
    # Strongest confound control; CROSS_STRATUM (or WITHIN_STRATUM if a
    # single stratum) with documented per-stratum allocation.
    MATCHED_CONTROLS = "matched_controls"

    # Non-fraud drawn so each stratum's *proportion* mirrors the fraud
    # sample (not 1:1).  Middle ground; CROSS_STRATUM with weights.
    STRATIFIED_MIRROR = "stratified_mirror"

    # i.i.d. non-fraud reservoir reflecting true base rates; ignores the
    # fraud strata for allocation.  Enables POPULATION_ESTIMATE claims.
    REPRESENTATIVE_RESERVOIR = "representative_reservoir"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StratumSpec:
    """One non-fraud stratum to pull, derived from fraud-side findings."""

    name: str
    target_count: int = 0
    source_finding_ids: list[str] = field(default_factory=list)
    predicate: Optional[str] = None        # SQL/pandas condition for the stratum
    fraud_count: Optional[int] = None       # fraud rows seen in this stratum
    confidence: float = 0.0                 # max confidence of source findings
    rationale: str = ""

    @property
    def predicate_known(self) -> bool:
        return bool(self.predicate and self.predicate.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_count": self.target_count,
            "source_finding_ids": list(self.source_finding_ids),
            "predicate": self.predicate,
            "fraud_count": self.fraud_count,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "predicate_known": self.predicate_known,
        }


@dataclass
class NonFraudSamplingDesign:
    """A complete, recordable plan for fetching the non-fraud sample."""

    target_dataset: str
    parent_sample: str                      # the fraud dataset this derives from
    strategy: SamplingStrategy
    strata: list[StratumSpec] = field(default_factory=list)
    sample_frame: SampleFrame = SampleFrame.UNSPECIFIED
    weights_documented: bool = False
    iid_reservoir_exists: bool = False
    total_target: int = 0
    base_rate_note: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    # Nuisance/time axis to stratify a representative draw on (e.g. "month").
    # Keeps the draw proportional on this axis WITHOUT matching on signal
    # features. None = plain i.i.d. draw.
    stratify_on: Optional[str] = None

    # -- discipline interop --------------------------------------------------

    def to_validator_dict(self) -> dict[str, Any]:
        """Shape consumed by ``SampleFrameValidator(sampling_design=...)``."""
        return {
            "strata": {s.name: s.to_dict() for s in self.strata},
            "iid_reservoir_exists": self.iid_reservoir_exists,
        }

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dataset": self.target_dataset,
            "parent_sample": self.parent_sample,
            "strategy": self.strategy.value,
            "sample_frame": self.sample_frame.value,
            "weights_documented": self.weights_documented,
            "iid_reservoir_exists": self.iid_reservoir_exists,
            "total_target": self.total_target,
            "base_rate_note": self.base_rate_note,
            "warnings": list(self.warnings),
            "notes": self.notes,
            "stratify_on": self.stratify_on,
            "strata": [s.to_dict() for s in self.strata],
        }

    @property
    def all_predicates_known(self) -> bool:
        return all(s.predicate_known for s in self.strata)

    @property
    def matched_columns(self) -> list[str]:
        """Columns this design equalises between fraud and non-fraud.

        For matched / stratified designs these are the columns the strata
        predicates condition on -- they are made (near-)identical across the
        two groups by construction, so comparing them in step 6 is
        meaningless (classic overmatching). Empty for a representative
        reservoir, which equalises nothing.
        """
        if self.strategy == SamplingStrategy.REPRESENTATIVE_RESERVOIR:
            return []
        cols: list[str] = []
        for s in self.strata:
            for c in _columns_in_predicate(s.predicate):
                if c not in cols:
                    cols.append(c)
        return cols

    def summary_line(self) -> str:
        return (
            f"NonFraudSamplingDesign({self.strategy.value}): "
            f"{len(self.strata)} strata, target n={self.total_target}, "
            f"frame={self.sample_frame.value}"
        )


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------


def _iter_findings(findings: Any) -> list[Any]:
    """Accept a FindingsLog (has ``.all()``) or any iterable of Finding."""
    if hasattr(findings, "all"):
        return list(findings.all())
    return list(findings)


class NonFraudSamplingDesigner:
    """Turn fraud-side findings into a non-fraud sampling design.

    Parameters
    ----------
    strategy
        How the sample is drawn (see :class:`SamplingStrategy`).
    ratio
        Non-fraud : fraud row ratio for MATCHED_CONTROLS (default 1.0).
    total_budget
        Total non-fraud rows to pull.  Required for REPRESENTATIVE_RESERVOIR;
        used as an equal-split fallback for matched/mirror designs when no
        per-stratum fraud counts are supplied.
    min_confidence
        Findings below this confidence are ignored when forming strata.
    stratum_predicates
        Optional ``{stratum_name: sql_predicate}`` map.  Strata without a
        predicate are still emitted, but flagged so a human can fill the
        condition before the SQL is generated.
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.REPRESENTATIVE_RESERVOIR,
        *,
        ratio: float = 1.0,
        total_budget: Optional[int] = None,
        min_confidence: float = 0.0,
        stratum_predicates: Optional[dict[str, str]] = None,
        stratify_on: Optional[str] = None,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0; got {ratio}")
        self.strategy = strategy
        self.ratio = ratio
        self.total_budget = total_budget
        self.min_confidence = min_confidence
        self.stratum_predicates = dict(stratum_predicates or {})
        # Nuisance/time axis for a stratified representative draw (reservoir).
        self.stratify_on = stratify_on

    # -- public API ----------------------------------------------------------

    def design(
        self,
        findings: Any,
        *,
        target_dataset: str,
        parent_sample: str,
        fraud_stratum_counts: Optional[dict[str, int]] = None,
    ) -> NonFraudSamplingDesign:
        """Build a :class:`NonFraudSamplingDesign` from fraud-side findings.

        ``fraud_stratum_counts`` maps stratum name -> number of fraud rows in
        that stratum (typically from the fraud profiler).  When present it
        drives matched/mirror allocation; when absent allocation falls back
        to an equal split of ``total_budget`` (with a warning).
        """
        if self.strategy == SamplingStrategy.REPRESENTATIVE_RESERVOIR:
            return self._design_reservoir(target_dataset, parent_sample)

        return self._design_stratified(
            findings,
            target_dataset=target_dataset,
            parent_sample=parent_sample,
            fraud_stratum_counts=fraud_stratum_counts or {},
        )

    # -- reservoir -----------------------------------------------------------

    def _design_reservoir(
        self, target_dataset: str, parent_sample: str
    ) -> NonFraudSamplingDesign:
        warnings: list[str] = []
        budget = self.total_budget or 0
        if budget <= 0:
            warnings.append(
                "REPRESENTATIVE_RESERVOIR needs total_budget > 0; "
                "target left at 0."
            )
        draw = ("i.i.d. draw reflecting true non-fraud base rate"
                if not self.stratify_on
                else f"draw stratified proportionally on '{self.stratify_on}' "
                     "(nuisance axis); signal features left free to vary")
        reservoir = StratumSpec(
            name="_iid_reservoir",
            target_count=budget,
            predicate=self.stratum_predicates.get("_iid_reservoir"),
            rationale=draw,
        )
        note = (
            "Reservoir reflects true base rate; record fraud prevalence "
            "separately so step-6 prevalence claims are calibratable."
        )
        if self.stratify_on:
            note += (f" Proportionally stratified on '{self.stratify_on}' to "
                     "avoid skew on that nuisance axis.")
        return NonFraudSamplingDesign(
            target_dataset=target_dataset,
            parent_sample=parent_sample,
            strategy=self.strategy,
            strata=[reservoir],
            sample_frame=SampleFrame.POPULATION_ESTIMATE,
            weights_documented=False,
            iid_reservoir_exists=True,
            total_target=budget,
            base_rate_note=note,
            warnings=warnings,
            stratify_on=self.stratify_on,
        )

    # -- matched / mirror ----------------------------------------------------

    def _design_stratified(
        self,
        findings: Any,
        *,
        target_dataset: str,
        parent_sample: str,
        fraud_stratum_counts: dict[str, int],
    ) -> NonFraudSamplingDesign:
        warnings: list[str] = []
        grouped = self._collect_strata(findings, target_dataset, warnings)

        if not grouped:
            warnings.append(
                "No fraud findings carry a stratum (or enables_sampling_for "
                f"'{target_dataset}'); cannot stratify. Falling back to an "
                "unstratified plan -- consider REPRESENTATIVE_RESERVOIR."
            )
            budget = self.total_budget or 0
            spec = StratumSpec(
                name="_unstratified",
                target_count=budget,
                predicate=self.stratum_predicates.get("_unstratified"),
                rationale="no discriminating strata found in fraud findings",
            )
            return NonFraudSamplingDesign(
                target_dataset=target_dataset,
                parent_sample=parent_sample,
                strategy=self.strategy,
                strata=[spec],
                sample_frame=SampleFrame.UNSPECIFIED,
                total_target=budget,
                warnings=warnings,
            )

        strata = self._build_specs(grouped, fraud_stratum_counts, warnings)
        total = sum(s.target_count for s in strata)

        # Frame: a single stratum is a clean WITHIN_STRATUM claim; multiple
        # strata mix strata, so CROSS_STRATUM with documented allocation.
        if len(strata) == 1:
            frame = SampleFrame.WITHIN_STRATUM
            weights_documented = False
        else:
            frame = SampleFrame.CROSS_STRATUM
            weights_documented = True

        missing = [s.name for s in strata if not s.predicate_known]
        if missing:
            warnings.append(
                "Strata without a SQL predicate (a human must supply the "
                f"condition before SQL generation): {missing}"
            )

        return NonFraudSamplingDesign(
            target_dataset=target_dataset,
            parent_sample=parent_sample,
            strategy=self.strategy,
            strata=strata,
            sample_frame=frame,
            weights_documented=weights_documented,
            iid_reservoir_exists=False,
            total_target=total,
            warnings=warnings,
        )

    # -- helpers -------------------------------------------------------------

    def _collect_strata(
        self, findings: Any, target_dataset: str, warnings: list[str]
    ) -> "dict[str, list[Any]]":
        """Group qualifying fraud findings by their bound stratum name."""
        grouped: dict[str, list[Any]] = {}
        seeded_no_stratum = 0
        for f in _iter_findings(findings):
            if getattr(f, "confidence", 0.0) < self.min_confidence:
                continue
            enables = set(getattr(f, "enables_sampling_for", []) or [])
            stratum = getattr(f, "stratum", None)
            # Eligible if it names a stratum, or explicitly seeds sampling for
            # the target dataset.
            if not stratum and target_dataset not in enables:
                continue
            if not stratum:
                seeded_no_stratum += 1
                continue
            grouped.setdefault(stratum, []).append(f)
        if seeded_no_stratum:
            warnings.append(
                f"{seeded_no_stratum} finding(s) enable sampling for "
                f"'{target_dataset}' but name no stratum; they were skipped "
                "(no stratum to mirror)."
            )
        return grouped

    def _build_specs(
        self,
        grouped: "dict[str, list[Any]]",
        fraud_stratum_counts: dict[str, int],
        warnings: list[str],
    ) -> list[StratumSpec]:
        equal_split = None
        if not fraud_stratum_counts:
            budget = self.total_budget or 0
            if budget > 0 and grouped:
                equal_split = budget // len(grouped)
                warnings.append(
                    "No per-stratum fraud counts supplied; allocating "
                    f"total_budget equally ({equal_split}/stratum). "
                    "Supply fraud_stratum_counts for true matched allocation."
                )
            else:
                warnings.append(
                    "No fraud_stratum_counts and no total_budget; per-stratum "
                    "target_count left at 0 (design is structural only)."
                )

        specs: list[StratumSpec] = []
        for name in sorted(grouped):
            members = grouped[name]
            fraud_n = fraud_stratum_counts.get(name)
            if fraud_n is not None:
                if self.strategy == SamplingStrategy.MATCHED_CONTROLS:
                    target = int(round(fraud_n * self.ratio))
                else:  # STRATIFIED_MIRROR -> proportional, ratio-scaled
                    target = int(round(fraud_n * self.ratio))
            elif equal_split is not None:
                target = equal_split
            else:
                target = 0
            strongest = max(members, key=lambda m: getattr(m, "confidence", 0.0))
            specs.append(
                StratumSpec(
                    name=name,
                    target_count=target,
                    source_finding_ids=[getattr(m, "id", "") for m in members],
                    predicate=self.stratum_predicates.get(name),
                    fraud_count=fraud_n,
                    confidence=getattr(strongest, "confidence", 0.0),
                    rationale=getattr(strongest, "claim", "")[:200],
                )
            )
        return specs
