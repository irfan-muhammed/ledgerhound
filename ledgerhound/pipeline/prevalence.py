"""
ledgerhound.pipeline.prevalence
=============================

**Prevalence measurement, checking, and population correction** (V4-020).

Three jobs, in order.

1. Measure what the constructed sample's fraud rate *actually is*
--------------------------------------------------------------------
Over **labelled rows only** (unlabelled rows are not denominators — see
`labeller`), and over **label-mature cohorts only**. Fraud is reported 30–90
days late, so the recent tail of any window systematically under-counts
fraud; a prevalence computed across it certifies a number that is wrong by
construction.

2. Check it against what was declared
-------------------------------------
The reported defect this exists to catch: *the agent assumes a 1:10 sample
when the real design is 1:145.* That is a 14.5x error, and it silently
rescales every precision figure downstream.

Four different ratios were live in this codebase simultaneously — 1:1
(`sampling/design.py` default), 1:10 (the agent's assumption), 1:145 (the
model story), 1:1000 (a hardcoded certificate label). This module is where
that stops being ambiguous: a measurement carries its **grain**, a
declaration carries its **grain**, and comparing across grains is refused
rather than reported as a divergence. Account- and transaction-grain
prevalence are different numbers for the same data.

3. Correct metrics back to deployment conditions
------------------------------------------------
A model evaluated at design prevalence π_s but deployed at population
prevalence π_p does not have the precision the evaluation reported.

The maths is standard and worth stating explicitly, because which metrics
move and which don't is the whole point:

* **TPR (recall) and FPR are prevalence-invariant.** They condition on the
  true class, so re-weighting the classes does not change them.
* **Precision is not.** By Bayes:

      PPV(π) = π·TPR / ( π·TPR + (1−π)·FPR )

So precision measured at 1:145 must be recomputed at the true base rate
before anyone plans capacity around it. Concretely, at TPR 0.8 / FPR 0.01,
moving from π_s = 1/145 to π_p = 1/10,000 takes precision from 0.357 to
0.0079 — a ~45x fall, and the difference between "this model is deployable"
and "every analyst drowns in false positives".

**Alert volume** follows directly and is the number an operations head
actually asks for: for N transactions scored per day,
`alerts = N·π_p·TPR + N·(1−π_p)·FPR`.

For score calibration there is the standard prior-correction offset
(King & Zeng's case-control correction): subtracting
`logit(π_s) − logit(π_p)` from a model's log-odds shifts sample-calibrated
scores onto the population scale.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from ledgerhound.pipeline.model_story import ModelStory

log = logging.getLogger(__name__)

SCHEMA = "prevalence/v1"

# Divergence bands, expressed as a ratio between measured and declared
# prevalence. Prevalence error propagates multiplicatively into precision, so
# the bands are ratio-based rather than absolute.
PASS_RATIO = 1.25
WARN_RATIO = 2.0

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_UNDECLARED = "UNDECLARED"
STATUS_INCOMPARABLE = "INCOMPARABLE"


class PrevalenceError(ValueError):
    """Prevalence cannot be measured or corrected as specified."""


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass
class PrevalenceMeasurement:
    """What the sample's fraud rate actually is, and over what."""

    grain: str
    n_fraud: int = 0
    n_legit: int = 0
    n_unlabelled: int = 0
    n_immature_excluded: int = 0
    mature_only: bool = False

    @property
    def n_labelled(self) -> int:
        return self.n_fraud + self.n_legit

    @property
    def prevalence(self) -> Optional[float]:
        """Fraud rate over labelled rows. ``None`` when nothing is labelled —
        never 0.0, which would read as a measured absence of fraud."""
        return (self.n_fraud / self.n_labelled) if self.n_labelled else None

    @property
    def ratio_text(self) -> str:
        """Human-facing ``1:N`` form, since that is how the business states it."""
        p = self.prevalence
        if p is None:
            return "n/a"
        if p <= 0:
            return "1:inf (no fraud in labelled rows)"
        return f"1:{(1.0 - p) / p:.0f}"

    def to_dict(self) -> dict:
        return {
            "grain": self.grain,
            "n_fraud": self.n_fraud,
            "n_legit": self.n_legit,
            "n_unlabelled": self.n_unlabelled,
            "n_immature_excluded": self.n_immature_excluded,
            "mature_only": self.mature_only,
            "n_labelled": self.n_labelled,
            "prevalence": self.prevalence,
            "ratio_text": self.ratio_text,
        }


def measure_prevalence(
    frame,
    *,
    grain: str,
    label_column: str = "_label",
    mature_column: Optional[str] = "_label_mature",
) -> PrevalenceMeasurement:
    """Measure prevalence over labelled, label-mature rows.

    ``mature_column`` is used when present; pass ``None`` to measure over all
    rows regardless of maturity, which is only correct if the caller has
    already excluded the immature tail.
    """
    import pandas as pd

    if label_column not in frame.columns:
        raise PrevalenceError(
            f"label column {label_column!r} not in frame; "
            f"available: {list(frame.columns)[:20]}. Run the labeller first."
        )
    if not grain.strip():
        raise PrevalenceError(
            "grain must be stated -- account- and transaction-grain "
            "prevalence are different numbers"
        )

    labels = frame[label_column]
    n_immature = 0
    mature_only = False
    if mature_column and mature_column in frame.columns:
        mature = frame[mature_column].fillna(False).astype(bool)
        n_immature = int((~mature).sum())
        labels = labels[mature]
        mature_only = True
    elif mature_column:
        log.warning(
            "measure_prevalence: no %r column -- measuring over ALL rows. If "
            "the sample includes a label-immature tail this under-counts "
            "fraud by construction.", mature_column,
        )

    return PrevalenceMeasurement(
        grain=grain,
        n_fraud=int((labels == 1).sum()),
        n_legit=int((labels == 0).sum()),
        n_unlabelled=int(pd.isna(labels).sum()),
        n_immature_excluded=n_immature,
        mature_only=mature_only,
    )


@dataclass
class PrevalenceCheck:
    """Measured vs declared, with a verdict that names the reason."""

    measurement: PrevalenceMeasurement
    declared: Optional[float] = None
    declared_grain: str = ""
    status: str = STATUS_UNDECLARED
    ratio: Optional[float] = None
    message: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_PASS

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "declared": self.declared,
            "declared_grain": self.declared_grain,
            "ratio_measured_over_declared": self.ratio,
            "message": self.message,
            "measurement": self.measurement.to_dict(),
        }

    def block(self) -> str:
        """A ``[DATA CHECK]`` block, matching `shell/data_checks.py`'s pattern
        so the shell can inject it as an observation the model cannot skip."""
        if self.status == STATUS_PASS:
            return ""
        return f"[DATA CHECK] Prevalence {self.status}:\n  - {self.message}"


def check_prevalence(
    measurement: PrevalenceMeasurement, story: ModelStory
) -> PrevalenceCheck:
    """Compare a measurement against the story's declared design prevalence."""
    declared = story.design_prevalence
    declared_grain = story.design_prevalence_grain

    if declared is None:
        return PrevalenceCheck(
            measurement=measurement, status=STATUS_UNDECLARED,
            message=(
                f"measured {measurement.ratio_text} at {measurement.grain} grain, "
                f"but the model story declares no design_prevalence -- nothing "
                f"to check against. Declare it (with its grain) so drift from "
                f"the intended design is detectable."
            ),
        )

    if declared_grain and declared_grain != measurement.grain:
        # Refusing beats reporting a divergence that is really a unit mismatch.
        return PrevalenceCheck(
            measurement=measurement, declared=declared,
            declared_grain=declared_grain, status=STATUS_INCOMPARABLE,
            message=(
                f"declared design_prevalence is at {declared_grain!r} grain but "
                f"the measurement is at {measurement.grain!r} grain. These are "
                f"different numbers for the same data (aggregation contracts "
                f"rows), so they are not comparable. Measure at "
                f"{declared_grain!r}, or declare a prevalence for "
                f"{measurement.grain!r}."
            ),
        )

    measured = measurement.prevalence
    if measured is None:
        return PrevalenceCheck(
            measurement=measurement, declared=declared,
            declared_grain=declared_grain, status=STATUS_FAIL,
            message=(
                f"no labelled rows at {measurement.grain} grain, so prevalence "
                f"is undefined -- cannot be checked against the declared "
                f"{declared:.6g}."
            ),
        )
    if measured <= 0:
        return PrevalenceCheck(
            measurement=measurement, declared=declared,
            declared_grain=declared_grain, status=STATUS_FAIL,
            ratio=0.0,
            message=(
                f"no fraud among {measurement.n_labelled:,} labelled rows at "
                f"{measurement.grain} grain, but the design declares "
                f"{declared:.6g}. Either the labelling or the draw is wrong."
            ),
        )

    ratio = measured / declared
    fold = max(ratio, 1.0 / ratio)
    if fold <= PASS_RATIO:
        status = STATUS_PASS
    elif fold <= WARN_RATIO:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL

    direction = "higher" if ratio > 1 else "lower"
    msg = (
        f"measured {measurement.ratio_text} ({measured:.6g}) at "
        f"{measurement.grain} grain vs declared "
        f"1:{(1 - declared) / declared:.0f} ({declared:.6g}) -- "
        f"{fold:.1f}x {direction} than designed."
    )
    if status != STATUS_PASS:
        msg += (
            " Prevalence error propagates multiplicatively into precision, so "
            "any precision or alert-volume figure computed from this sample is "
            "wrong by about the same factor until this is reconciled."
        )
    if not measurement.mature_only:
        msg += (
            " NOTE: measured without a label-maturity filter, so a recent "
            "unreported tail may be depressing this figure further."
        )

    return PrevalenceCheck(
        measurement=measurement, declared=declared,
        declared_grain=declared_grain, status=status, ratio=ratio, message=msg,
    )


# ---------------------------------------------------------------------------
# Population correction
# ---------------------------------------------------------------------------


def _check_prev(value: float, name: str) -> None:
    if not (0.0 < value < 1.0):
        raise PrevalenceError(
            f"{name} must be a fraction strictly between 0 and 1 "
            f"(e.g. 1/10000 = 0.0001, not 10000); got {value}"
        )


def corrected_precision(
    tpr: float, fpr: float, population_prevalence: float
) -> Optional[float]:
    """Precision at ``population_prevalence``, from prevalence-invariant rates.

    PPV(π) = π·TPR / ( π·TPR + (1−π)·FPR ).

    Returns ``None`` when the model flags nothing at all (TPR and FPR both
    zero) — precision is genuinely undefined there, not 0.
    """
    _check_prev(population_prevalence, "population_prevalence")
    for name, v in (("tpr", tpr), ("fpr", fpr)):
        if not (0.0 <= v <= 1.0):
            raise PrevalenceError(f"{name} must be in [0,1]; got {v}")

    p = population_prevalence
    tp = p * tpr
    fp = (1.0 - p) * fpr
    if tp + fp <= 0.0:
        return None
    return tp / (tp + fp)


def precision_shift(
    tpr: float, fpr: float, sample_prevalence: float, population_prevalence: float
) -> dict:
    """How precision changes moving from sample to population prevalence.

    This is the number that decides whether an evaluation result means
    anything operationally.
    """
    _check_prev(sample_prevalence, "sample_prevalence")
    at_sample = corrected_precision(tpr, fpr, sample_prevalence)
    at_population = corrected_precision(tpr, fpr, population_prevalence)
    factor = None
    if at_sample and at_population is not None and at_sample > 0:
        factor = at_population / at_sample
    return {
        "precision_at_sample_prevalence": at_sample,
        "precision_at_population_prevalence": at_population,
        "factor": factor,
        "tpr": tpr,
        "fpr": fpr,
        "sample_prevalence": sample_prevalence,
        "population_prevalence": population_prevalence,
    }


def alert_volume(
    n_scored: int, tpr: float, fpr: float, population_prevalence: float
) -> dict:
    """Expected alerts for ``n_scored`` transactions at population prevalence.

    The operational number: how many alerts land, and how many are worth
    looking at.
    """
    _check_prev(population_prevalence, "population_prevalence")
    if n_scored < 0:
        raise PrevalenceError(f"n_scored must be >= 0; got {n_scored}")

    p = population_prevalence
    expected_fraud = n_scored * p
    true_pos = expected_fraud * tpr
    false_pos = n_scored * (1.0 - p) * fpr
    total = true_pos + false_pos
    return {
        "n_scored": n_scored,
        "expected_fraud": expected_fraud,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "total_alerts": total,
        "precision": (true_pos / total) if total > 0 else None,
        "false_positives_per_true_positive": (
            (false_pos / true_pos) if true_pos > 0 else None
        ),
        "missed_fraud": expected_fraud * (1.0 - tpr),
    }


def prior_correction_offset(
    sample_prevalence: float, population_prevalence: float
) -> float:
    """Log-odds offset to move sample-calibrated scores onto the population
    scale (King & Zeng case-control correction).

    Subtract the returned value from a model's logit:
    ``logit_population = logit_sample - offset``.
    """
    _check_prev(sample_prevalence, "sample_prevalence")
    _check_prev(population_prevalence, "population_prevalence")
    return _logit(sample_prevalence) - _logit(population_prevalence)


@dataclass
class PopulationCorrection:
    """A recorded correction, for the certificate face."""

    sample_prevalence: float
    population_prevalence: float
    population_prevalence_source: str
    grain: str
    tpr: Optional[float] = None
    fpr: Optional[float] = None
    n_scored_per_period: Optional[int] = None
    period: str = "day"
    _shift: dict = field(default_factory=dict)
    _volume: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "schema": SCHEMA,
            "grain": self.grain,
            "sample_prevalence": self.sample_prevalence,
            "population_prevalence": self.population_prevalence,
            "population_prevalence_source": self.population_prevalence_source,
            "prior_correction_offset": prior_correction_offset(
                self.sample_prevalence, self.population_prevalence),
        }
        if self._shift:
            d["precision_shift"] = self._shift
        if self._volume:
            d[f"alerts_per_{self.period}"] = self._volume
        return d


def build_population_correction(
    measurement: PrevalenceMeasurement,
    story: ModelStory,
    *,
    tpr: Optional[float] = None,
    fpr: Optional[float] = None,
    n_scored_per_period: Optional[int] = None,
    period: str = "day",
) -> PopulationCorrection:
    """Assemble the correction record from a measurement plus the story.

    Refuses if the story has no sourced population prevalence: an unsourced
    base rate must not drive a capacity plan.
    """
    measured = measurement.prevalence
    if measured is None or measured <= 0:
        raise PrevalenceError(
            "cannot build a population correction: the sample has no measured "
            "fraud prevalence to correct FROM"
        )
    if story.population_prevalence is None:
        raise PrevalenceError(
            "model story has no population_prevalence -- precision and alert "
            "volume cannot be corrected to deployment conditions without the "
            "true base rate, and guessing it would be worse than omitting it"
        )
    if not story.population_prevalence_source.strip():
        raise PrevalenceError(
            "population_prevalence has no recorded source -- an unsourced base "
            "rate must not drive a precision or capacity claim"
        )
    if story.population_prevalence_grain and \
            story.population_prevalence_grain != measurement.grain:
        raise PrevalenceError(
            f"population_prevalence is at {story.population_prevalence_grain!r} "
            f"grain but the measurement is at {measurement.grain!r} grain -- "
            f"correcting across grains would silently mix two different rates"
        )

    corr = PopulationCorrection(
        sample_prevalence=measured,
        population_prevalence=story.population_prevalence,
        population_prevalence_source=story.population_prevalence_source,
        grain=measurement.grain,
        tpr=tpr, fpr=fpr,
        n_scored_per_period=n_scored_per_period, period=period,
    )
    if tpr is not None and fpr is not None:
        corr._shift = precision_shift(
            tpr, fpr, measured, story.population_prevalence)
        if n_scored_per_period is not None:
            corr._volume = alert_volume(
                n_scored_per_period, tpr, fpr, story.population_prevalence)
    return corr
