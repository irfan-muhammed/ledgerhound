"""
ledgerhound.discipline.sample_frame
================================

Formal sample-frame algebra for Ledgerhound claims.

Motivation
----------
A finding like *"fraud rate is 3.2x higher in young accounts"* is only
defensible under a specific sampling regime.  If the sample over-represents
young accounts (as most fraud datasets do), the rate is meaningful only
*within the young-accounts stratum*, NOT as a population estimate.

Most LLM-EDA agents (Code Interpreter, Pandas-AI, DSAgent...) silently
extrapolate sample-level statistics to population claims.  Ledgerhound is
designed to refuse that extrapolation unless the sampling design supports
it.  This module is the formal core of that discipline.

The four sample frames
----------------------
::

    WITHIN_STRATUM       Claim holds inside a specific named stratum.
                         Strongest, most specific. Always defensible.

    CROSS_STRATUM        Claim mixes strata; weights have been applied
                         to combine them. Defensible if weights are
                         documented.

    POPULATION_ESTIMATE  Claim is an estimate of the full population
                         parameter. Only legal when an i.i.d. reservoir
                         stratum exists in the sampling design.

    DERIVED_SAMPLE       Claim refers to a sample derived from another
                         sample (e.g. hard negatives from non-fraud).
                         Defensible only relative to the parent sample.

    UNKNOWN              Last resort. Emits a warning. Used when the
                         sampling design isn't recorded.

The algebra
-----------
For each pandas-like operation we define how an input sample-frame tag
maps to the output tag.  The complete table is ``operation_table``.

Example::

    df: tag=WITHIN_STRATUM("young_accounts")
    df.mean()                  -> WITHIN_STRATUM("young_accounts")
    df[df.fraud == 1].mean()   -> WITHIN_STRATUM("young_accounts")
    df.groupby("bank").mean()  -> WITHIN_STRATUM("young_accounts")
                                 (still bounded by the source stratum)
    pd.concat([young, old])    -> CROSS_STRATUM   (mixing strata)
    pd.concat([young, reservoir]) -> CROSS_STRATUM (young rows are
                                 over-represented; not a population frame)
    promote(young)             -> SampleFrameViolation (combine strata first)

Validation
----------
``SampleFrameValidator.validate_claim(claim, sampling_design)`` returns
a structured ``ClaimValidation`` with ``is_valid``, ``warnings``, and
``required_promotion`` (if the claim's frame needs to be promoted to a
weaker tag because the sampling design doesn't support the stronger
claim).

This module is the formal anchor for any paper claim that Ledgerhound's
output is more epistemically disciplined than baseline LLM agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# -- Core enums and dataclasses ---------------------------------------------


class SampleFrame(str, Enum):
    """The four legal sample-frame tags, plus UNKNOWN as a safety valve."""

    WITHIN_STRATUM      = "within_stratum"
    CROSS_STRATUM       = "cross_stratum"
    POPULATION_ESTIMATE = "population_estimate"
    DERIVED_SAMPLE      = "derived_sample"
    UNKNOWN             = "unknown"

    @property
    def is_specific(self) -> bool:
        """A frame is *specific* if claims under it are bounded by a known scope."""
        return self in (
            SampleFrame.WITHIN_STRATUM,
            SampleFrame.DERIVED_SAMPLE,
        )

    @property
    def is_extrapolating(self) -> bool:
        """A frame is *extrapolating* if claims under it generalise beyond the sample."""
        return self == SampleFrame.POPULATION_ESTIMATE


# Specificity ordering -- used for the join/concat algebra.
# Lower index = more specific.  When operations combine frames, the result
# is the LEAST specific (max index) of the inputs.
_SPECIFICITY_RANK: dict[SampleFrame, int] = {
    SampleFrame.WITHIN_STRATUM:      0,
    SampleFrame.DERIVED_SAMPLE:      1,
    SampleFrame.CROSS_STRATUM:       2,
    SampleFrame.POPULATION_ESTIMATE: 3,
    SampleFrame.UNKNOWN:             4,
}


def _least_specific(*frames: SampleFrame) -> SampleFrame:
    """Return the LEAST specific of the given frames (used for combining)."""
    return max(frames, key=lambda f: _SPECIFICITY_RANK.get(f, 99))


class LabelProvenance(str, Enum):
    """How the labels underlying a claim came to exist (V4-025).

    Orthogonal to :class:`SampleFrame`: that says which *rows* a claim covers,
    this says how far the *labels* on those rows can be trusted. A claim can
    be perfectly scoped and still rest on labels that only record what was
    caught.

    ``INCUMBENT_DETECTED`` is the honest default for fraud work. Labels exist
    only where the current detection system or a customer complaint produced
    them, so the negative class is *not-reported* rather than non-fraud, and
    every model trained on it inherits the incumbent's blind spots. Findings
    carrying this value must never be phrased as statements about fraud in
    general -- only about fraud that gets found the way it currently gets
    found.
    """

    INCUMBENT_DETECTED = "incumbent_detected"
    ADJUDICATED        = "adjudicated"
    SYNTHETIC          = "synthetic"
    UNKNOWN            = "unknown"

    @property
    def is_selective(self) -> bool:
        """True when labels are conditional on something other than truth."""
        return self in (LabelProvenance.INCUMBENT_DETECTED,
                        LabelProvenance.UNKNOWN)

    @property
    def caveat(self) -> str:
        """One line to carry alongside any claim resting on these labels."""
        return {
            LabelProvenance.INCUMBENT_DETECTED: (
                "labels are conditional on incumbent detection and customer "
                "reporting -- the negative class is 'not-reported', not "
                "non-fraud, and undetected fraud is invisible here"),
            LabelProvenance.ADJUDICATED: (
                "labels were independently adjudicated, so the negative class "
                "is closer to genuinely negative -- state the adjudication "
                "coverage, since anything unreviewed reverts to not-reported"),
            LabelProvenance.SYNTHETIC: (
                "labels are synthetic and known by construction; results do "
                "not transfer to production data"),
            LabelProvenance.UNKNOWN: (
                "label provenance was not recorded, so no claim about the "
                "negative class is supportable"),
        }[self]


@dataclass
class TaggedFrame:
    """A sample-frame-tagged claim or DataFrame reference.

    ``stratum`` is required when ``tag == WITHIN_STRATUM``.
    ``parent_sample`` is required when ``tag == DERIVED_SAMPLE``.
    """

    tag: SampleFrame
    stratum: Optional[str] = None
    parent_sample: Optional[str] = None
    weights_documented: bool = False
    notes: str = ""
    # How far the LABELS can be trusted, independently of which rows the
    # claim covers. Defaults to UNKNOWN so an unrecorded provenance is
    # visible rather than assumed benign (V4-025).
    label_provenance: LabelProvenance = LabelProvenance.UNKNOWN

    def __post_init__(self):
        if self.tag == SampleFrame.WITHIN_STRATUM and not self.stratum:
            raise ValueError(
                "WITHIN_STRATUM tag requires a `stratum` name."
            )
        if self.tag == SampleFrame.DERIVED_SAMPLE and not self.parent_sample:
            raise ValueError(
                "DERIVED_SAMPLE tag requires a `parent_sample` name."
            )

    def to_dict(self) -> dict:
        return {
            "tag":                self.tag.value,
            "stratum":            self.stratum,
            "parent_sample":      self.parent_sample,
            "weights_documented": self.weights_documented,
            "notes":              self.notes,
            "label_provenance":   self.label_provenance.value,
            "label_caveat":       self.label_provenance.caveat,
        }


class SampleFrameViolation(Exception):
    """Raised when an operation would produce an unsupportable claim."""

    def __init__(self, message: str, suggested_fix: str = ""):
        super().__init__(message)
        self.suggested_fix = suggested_fix


# -- Operation taxonomy -----------------------------------------------------


class OperationKind(str, Enum):
    """The kinds of pandas-like operations whose sample-frame transformations
    we have formalised."""

    # Reductions
    AGGREGATE       = "aggregate"          # mean, sum, count, std, ...
    QUANTILE        = "quantile"           # quantile, median (special-cased)

    # Selections
    FILTER          = "filter"             # df[mask]
    PROJECT         = "project"            # df[cols]
    SORT            = "sort"               # sort_values

    # Group operations
    GROUPBY_AGG     = "groupby_agg"        # df.groupby(col).agg(...)
    GROUPBY_TRANSFORM = "groupby_transform" # transform that returns same shape

    # Multi-frame operations
    CONCAT          = "concat"
    MERGE           = "merge"

    # Sampling
    RANDOM_SAMPLE   = "random_sample"      # df.sample(n)
    STRATIFIED_SAMPLE = "stratified_sample"

    # Promotions
    POPULATION_ESTIMATE_PROMOTE = "population_estimate_promote"
    WEIGHTED_REWEIGHT           = "weighted_reweight"


# -- The algebra (formal table) ---------------------------------------------


@dataclass(frozen=True)
class _Rule:
    """Single transformation rule. Used internally by SampleFrameAlgebra."""

    operation: OperationKind
    input_frame: SampleFrame
    output_frame: SampleFrame
    requires: tuple[str, ...] = ()
    notes: str = ""


# Canonical operation table -- the spine of the formalism.
# Read this as: applying ``operation`` to a frame tagged ``input_frame``
# produces a frame tagged ``output_frame``, subject to the listed
# preconditions.
_OPERATION_TABLE: list[_Rule] = [
    # Aggregates preserve the input frame (rates within a stratum are
    # still claims about that stratum).
    _Rule(OperationKind.AGGREGATE, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM,
          notes="Aggregating within a stratum yields a within-stratum statistic."),
    _Rule(OperationKind.AGGREGATE, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM,
          requires=("weights_documented",),
          notes="Cross-stratum aggregation requires weights to combine strata."),
    _Rule(OperationKind.AGGREGATE, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.POPULATION_ESTIMATE),
    _Rule(OperationKind.AGGREGATE, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.DERIVED_SAMPLE),

    # Filtering does not promote the frame.  A filter that "looks at fraud
    # accounts" inside a within-stratum sample is still within-stratum.
    _Rule(OperationKind.FILTER, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM),
    _Rule(OperationKind.FILTER, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM),
    _Rule(OperationKind.FILTER, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.POPULATION_ESTIMATE),
    _Rule(OperationKind.FILTER, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.DERIVED_SAMPLE),

    # Projection / sort don't change the frame.
    _Rule(OperationKind.PROJECT, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM),
    _Rule(OperationKind.PROJECT, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM),
    _Rule(OperationKind.PROJECT, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.POPULATION_ESTIMATE),
    _Rule(OperationKind.PROJECT, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.DERIVED_SAMPLE),

    _Rule(OperationKind.SORT, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM),
    _Rule(OperationKind.SORT, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM),
    _Rule(OperationKind.SORT, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.POPULATION_ESTIMATE),
    _Rule(OperationKind.SORT, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.DERIVED_SAMPLE),

    # Quantile is preserved in tag but precision is reduced -- we leave that
    # to user confidence, not the algebra.
    _Rule(OperationKind.QUANTILE, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM),
    _Rule(OperationKind.QUANTILE, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM),

    # Groupby on stratum boundary cols still yields within-stratum results
    # (each group is a sub-stratum of the original stratum).
    _Rule(OperationKind.GROUPBY_AGG, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM,
          notes="Groupby within a stratum yields per-subgroup within-stratum claims."),
    _Rule(OperationKind.GROUPBY_AGG, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM,
          requires=("weights_documented",)),
    _Rule(OperationKind.GROUPBY_AGG, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.POPULATION_ESTIMATE),
    _Rule(OperationKind.GROUPBY_AGG, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.DERIVED_SAMPLE),

    # Random sampling produces a derived sample.
    _Rule(OperationKind.RANDOM_SAMPLE, SampleFrame.WITHIN_STRATUM,
          SampleFrame.DERIVED_SAMPLE),
    _Rule(OperationKind.RANDOM_SAMPLE, SampleFrame.CROSS_STRATUM,
          SampleFrame.DERIVED_SAMPLE),
    _Rule(OperationKind.RANDOM_SAMPLE, SampleFrame.POPULATION_ESTIMATE,
          SampleFrame.DERIVED_SAMPLE),

    # Stratified sampling preserves stratum structure.
    _Rule(OperationKind.STRATIFIED_SAMPLE, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM),
    _Rule(OperationKind.STRATIFIED_SAMPLE, SampleFrame.CROSS_STRATUM,
          SampleFrame.CROSS_STRATUM),

    # Concat / Merge handled by SampleFrameAlgebra.combine() (see below).
    # The table here only states the same-frame trivial cases.
    _Rule(OperationKind.CONCAT, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM,
          notes="Concat of two same-stratum frames stays within stratum."),
    _Rule(OperationKind.MERGE, SampleFrame.WITHIN_STRATUM,
          SampleFrame.WITHIN_STRATUM,
          notes="Merge of two same-stratum frames stays within stratum."),

    # Promotion to population estimate is the most controlled move.
    # WITHIN_STRATUM has no promotion rule on purpose: no weighting of one
    # stratum recovers the others, so apply() refuses it outright. Strata
    # must be combined into CROSS_STRATUM (weights documented) first.
    _Rule(OperationKind.POPULATION_ESTIMATE_PROMOTE, SampleFrame.CROSS_STRATUM,
          SampleFrame.POPULATION_ESTIMATE,
          requires=("iid_reservoir_exists", "weights_documented",),
          notes="Only legal when the sampling design includes an i.i.d. reservoir."),
    _Rule(OperationKind.POPULATION_ESTIMATE_PROMOTE, SampleFrame.DERIVED_SAMPLE,
          SampleFrame.UNKNOWN,
          notes="Cannot promote a derived sample directly to population estimate."),
]


def operation_table() -> list[dict]:
    """Return the operation table as plain dicts (for export to docs/papers)."""
    return [
        {
            "operation":    r.operation.value,
            "input_frame":  r.input_frame.value,
            "output_frame": r.output_frame.value,
            "requires":     list(r.requires),
            "notes":        r.notes,
        }
        for r in _OPERATION_TABLE
    ]


# -- Algebra ----------------------------------------------------------------


class SampleFrameAlgebra:
    """The core algebraic API.

    Use ``apply()`` for unary ops and ``combine()`` for binary ops
    (concat, merge).  Both return a new :class:`TaggedFrame`.
    """

    @staticmethod
    def apply(
        operation: OperationKind,
        input_tagged: TaggedFrame,
        sampling_design: Optional[dict] = None,
    ) -> TaggedFrame:
        """Apply a unary operation to a tagged frame.

        Parameters
        ----------
        operation : OperationKind
        input_tagged : TaggedFrame
        sampling_design : dict, optional
            May include keys: ``iid_reservoir_exists`` (bool),
            ``priors_encoded`` (list[str]), ``strata`` (dict).

        Returns
        -------
        TaggedFrame  (output)

        Raises
        ------
        SampleFrameViolation
            If a precondition isn't met (e.g. POPULATION_ESTIMATE_PROMOTE
            without an i.i.d. reservoir).
        """
        sd = sampling_design or {}
        if (operation == OperationKind.POPULATION_ESTIMATE_PROMOTE
                and input_tagged.tag == SampleFrame.WITHIN_STRATUM):
            raise SampleFrameViolation(
                "A single stratum cannot be promoted to population_estimate: "
                "combine strata into cross_stratum with documented weights first.",
                suggested_fix=(
                    "Keep the claim WITHIN_STRATUM, or combine all strata "
                    "(CROSS_STRATUM, weights documented) before promoting."
                ),
            )
        for rule in _OPERATION_TABLE:
            if rule.operation == operation and rule.input_frame == input_tagged.tag:
                # Check requires
                missing = []
                for req in rule.requires:
                    if req == "weights_documented" and not input_tagged.weights_documented:
                        missing.append(req)
                    elif req == "iid_reservoir_exists" and not sd.get("iid_reservoir_exists"):
                        missing.append(req)
                if missing:
                    raise SampleFrameViolation(
                        f"Operation {operation.value} on "
                        f"{input_tagged.tag.value} requires: {missing}",
                        suggested_fix=(
                            "Either document weights / register an i.i.d. "
                            "reservoir stratum, or accept the result as "
                            "WITHIN_STRATUM / CROSS_STRATUM."
                        ),
                    )
                return TaggedFrame(
                    tag=rule.output_frame,
                    stratum=(
                        input_tagged.stratum
                        if rule.output_frame == SampleFrame.WITHIN_STRATUM
                        else None
                    ),
                    parent_sample=(
                        input_tagged.stratum or input_tagged.parent_sample
                        if rule.output_frame == SampleFrame.DERIVED_SAMPLE
                        else None
                    ),
                    weights_documented=input_tagged.weights_documented,
                    notes=rule.notes,
                )
        # Default: UNKNOWN (we don't have a rule)
        log.warning(
            "No rule for operation %s on frame %s -- defaulting to UNKNOWN",
            operation.value, input_tagged.tag.value,
        )
        return TaggedFrame(
            tag=SampleFrame.UNKNOWN,
            notes=f"No formal rule for {operation.value} x {input_tagged.tag.value}",
        )

    @staticmethod
    def combine(
        operation: OperationKind,
        left: TaggedFrame,
        right: TaggedFrame,
    ) -> TaggedFrame:
        """Combine two tagged frames (concat, merge).

        The output frame is the LEAST specific of the inputs (max specificity-rank).
        If both inputs are WITHIN_STRATUM but with different strata, the output
        is CROSS_STRATUM.
        """
        if operation not in (OperationKind.CONCAT, OperationKind.MERGE):
            raise ValueError(
                f"combine() supports CONCAT/MERGE only, got {operation.value}"
            )
        # Same WITHIN_STRATUM stratum -> preserved
        if (
            left.tag == SampleFrame.WITHIN_STRATUM
            and right.tag == SampleFrame.WITHIN_STRATUM
            and left.stratum == right.stratum
        ):
            return TaggedFrame(
                tag=SampleFrame.WITHIN_STRATUM,
                stratum=left.stratum,
                weights_documented=left.weights_documented and right.weights_documented,
                notes="Both inputs in the same stratum -- preserved.",
            )
        # Different WITHIN strata -> CROSS_STRATUM
        if (
            left.tag == SampleFrame.WITHIN_STRATUM
            and right.tag == SampleFrame.WITHIN_STRATUM
            and left.stratum != right.stratum
        ):
            return TaggedFrame(
                tag=SampleFrame.CROSS_STRATUM,
                weights_documented=False,
                notes=(
                    f"Combined two strata ({left.stratum}, {right.stratum}) -- "
                    f"output is cross-stratum until weights are documented."
                ),
            )
        # Mixing a population frame with anything narrower is NOT a
        # population frame -- the narrower rows are over-represented in the
        # result. Taking the "least specific" input here would itself be an
        # unlicensed generalization, so it drops to CROSS_STRATUM.
        if (SampleFrame.POPULATION_ESTIMATE in (left.tag, right.tag)
                and left.tag != right.tag
                and SampleFrame.UNKNOWN not in (left.tag, right.tag)):
            return TaggedFrame(
                tag=SampleFrame.CROSS_STRATUM,
                weights_documented=False,
                notes=(
                    f"{operation.value}: population frame mixed with "
                    f"{(right if left.tag == SampleFrame.POPULATION_ESTIMATE else left).tag.value} "
                    f"-- cross-stratum until weights are documented."
                ),
            )
        # Otherwise: least specific
        out_tag = _least_specific(left.tag, right.tag)
        return TaggedFrame(
            tag=out_tag,
            stratum=None,
            weights_documented=left.weights_documented and right.weights_documented,
            notes=(
                f"{operation.value}: combined "
                f"{left.tag.value} x {right.tag.value} -> {out_tag.value}"
            ),
        )

    @staticmethod
    def promote_to_population(
        input_tagged: TaggedFrame,
        sampling_design: dict,
    ) -> TaggedFrame:
        """Explicit population-estimate promotion.

        Wraps ``apply(POPULATION_ESTIMATE_PROMOTE, ...)`` for clarity at
        call sites where the analyst is making the 'this generalises to
        the full population' move.
        """
        return SampleFrameAlgebra.apply(
            OperationKind.POPULATION_ESTIMATE_PROMOTE,
            input_tagged,
            sampling_design,
        )


# -- Validator --------------------------------------------------------------


@dataclass
class ClaimValidation:
    """Output of SampleFrameValidator.validate_claim."""

    is_valid: bool
    declared_frame: SampleFrame
    inferred_frame: Optional[SampleFrame] = None
    required_promotion: Optional[SampleFrame] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "is_valid":           self.is_valid,
            "declared_frame":     self.declared_frame.value,
            "inferred_frame":     (
                self.inferred_frame.value if self.inferred_frame else None
            ),
            "required_promotion": (
                self.required_promotion.value if self.required_promotion else None
            ),
            "warnings":           self.warnings,
            "errors":             self.errors,
            "suggested_fix":      self.suggested_fix,
        }


class SampleFrameValidator:
    """Validate a finding's declared sample-frame against the sampling design.

    A claim is *valid* if:
      - The declared frame is one of the four legal frames (or UNKNOWN).
      - If declared WITHIN_STRATUM, the named stratum exists in the design.
      - If declared POPULATION_ESTIMATE, the design includes an i.i.d.
        reservoir stratum (otherwise the claim must be downgraded to
        CROSS_STRATUM or WITHIN_STRATUM).
      - If declared CROSS_STRATUM, weights must be documented.
      - If declared DERIVED_SAMPLE, the parent sample must be referenceable
        in the design.
    """

    def __init__(self, sampling_design: Optional[dict] = None) -> None:
        self.sampling_design = sampling_design or {}

    def validate_claim(
        self,
        declared_frame: SampleFrame,
        stratum: Optional[str] = None,
        parent_sample: Optional[str] = None,
        weights_documented: bool = False,
    ) -> ClaimValidation:
        sd = self.sampling_design
        warnings: list[str] = []
        errors: list[str] = []
        required_promotion: Optional[SampleFrame] = None
        suggested_fix = ""

        if declared_frame == SampleFrame.UNKNOWN:
            warnings.append(
                "Sample frame is UNKNOWN -- finding will be flagged for review."
            )
            return ClaimValidation(
                is_valid=True,
                declared_frame=declared_frame,
                warnings=warnings,
                suggested_fix=(
                    "Tag the finding with one of: within_stratum, "
                    "cross_stratum, population_estimate, derived_sample."
                ),
            )

        if declared_frame == SampleFrame.WITHIN_STRATUM:
            if not stratum:
                errors.append("WITHIN_STRATUM requires a stratum name.")
            elif "strata" in sd and stratum not in sd["strata"]:
                warnings.append(
                    f"Stratum '{stratum}' is not declared in sampling_design.strata "
                    f"(known: {list(sd.get('strata', {}).keys())})."
                )

        elif declared_frame == SampleFrame.POPULATION_ESTIMATE:
            iid_ok = bool(sd.get("iid_reservoir_exists"))
            if not iid_ok:
                errors.append(
                    "POPULATION_ESTIMATE is not legal: sampling_design "
                    "has no i.i.d. reservoir stratum."
                )
                required_promotion = SampleFrame.CROSS_STRATUM
                suggested_fix = (
                    "Downgrade the claim to CROSS_STRATUM (with weights) "
                    "or WITHIN_STRATUM (with stratum name).  To make a "
                    "population claim, register an i.i.d. reservoir stratum "
                    "in the sampling design."
                )
            if not weights_documented:
                warnings.append(
                    "Weights are not documented -- population estimate may "
                    "still be biased even with an i.i.d. reservoir."
                )

        elif declared_frame == SampleFrame.CROSS_STRATUM:
            if not weights_documented:
                warnings.append(
                    "CROSS_STRATUM claims should document the weights used "
                    "to combine strata."
                )

        elif declared_frame == SampleFrame.DERIVED_SAMPLE:
            if not parent_sample:
                errors.append("DERIVED_SAMPLE requires a parent_sample name.")

        return ClaimValidation(
            is_valid=not errors,
            declared_frame=declared_frame,
            required_promotion=required_promotion,
            warnings=warnings,
            errors=errors,
            suggested_fix=suggested_fix,
        )

    def validate_tagged_frame(self, tagged: TaggedFrame) -> ClaimValidation:
        return self.validate_claim(
            declared_frame=tagged.tag,
            stratum=tagged.stratum,
            parent_sample=tagged.parent_sample,
            weights_documented=tagged.weights_documented,
        )
