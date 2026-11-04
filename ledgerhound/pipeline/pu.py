"""
ledgerhound.pipeline.pu
=====================

**Positive-unlabelled honesty** (V4-025).

The negative class is not "non-fraud"
--------------------------------------
It is **"not-reported"**. Fraud is only ever observed through detection and
complaint, so an unflagged transaction is one of two things and we cannot
tell which: genuinely legitimate, or fraud nobody caught. Calling that class
"non-fraud" on a certificate asserts an absence we never observed — the same
overclaim `NOVELTY.md` §5 forbids elsewhere, and the reason `labeller`
returns tri-state labels.

This module supplies the vocabulary (:func:`honest_class_name`) and, more
usefully, quantifies what the contamination does.

What contamination actually does to the numbers
-----------------------------------------------
Model the negative class as a mixture: a fraction ``c`` of it is truly
positive. Two consequences follow exactly.

**Prevalence is understated.** The true rate is
``π_true = π_obs + c·(1 − π_obs)`` — every contaminated row is a positive
counted on the wrong side.

**Every measured group difference is attenuated toward zero.** If the
negative class is ``(1−c)`` legitimate plus ``c`` fraud, its mean is
``(1−c)·μ_legit + c·μ_fraud``, so

    observed_difference = (1 − c) · true_difference

The correction is therefore ``true = observed / (1 − c)``. A feature that
looks weak may simply be measured through a contaminated comparison — and at
``c = 0.5`` the observed effect is half the real one.

**This means contamination cannot manufacture a difference, only hide one.**
A feature that separates the classes *despite* contamination separates them
at least that well in truth. That asymmetry is worth stating: it makes
positive findings robust and negative findings unsafe.

Spike-in sensitivity
--------------------
The analytic correction assumes the mixture model. :func:`spike_in_sensitivity`
checks empirically instead: relabel a known fraction of confirmed fraud as
negative, recompute whatever statistic the conclusion rests on, and report
how far it moves. If a conclusion survives plausible contamination it can be
stated; if it flips, it was never safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

SCHEMA = "pu/v1"

# Canonical, honest names. The negative one is the point of this module.
POSITIVE_NAME = "reported-fraud"
NEGATIVE_NAME = "not-reported"
UNLABELLED_NAME = "unlabelled"

# Names that assert an absence of fraud nobody observed.
FORBIDDEN_NEGATIVE_NAMES = frozenset({
    "non-fraud", "nonfraud", "non_fraud", "legitimate", "genuine",
    "clean", "good", "not-fraud", "not_fraud",
})

DEFAULT_RATES = (0.001, 0.01, 0.05, 0.10)


class PUError(ValueError):
    """A positive-unlabelled analysis cannot be performed as specified."""


def honest_class_name(label: Any) -> str:
    """The name a certificate may use for a label value.

    ``0`` is **never** rendered as "non-fraud": we observed that it was not
    reported, not that it was not fraud.
    """
    import pandas as pd

    if label is None or (isinstance(label, float) and label != label) \
            or label is pd.NA:
        return UNLABELLED_NAME
    if int(label) == 1:
        return POSITIVE_NAME
    if int(label) == 0:
        return NEGATIVE_NAME
    raise PUError(f"unexpected label value {label!r}; expected 1, 0, or null")


def assert_honest_naming(text: str) -> None:
    """Raise if ``text`` uses a class name that overclaims.

    Used to keep certificate and report copy honest at the point it is
    written, rather than discovering it in a screenshot later.
    """
    low = text.lower()
    hits = sorted(n for n in FORBIDDEN_NEGATIVE_NAMES if n in low)
    if hits:
        raise PUError(
            f"text uses {hits} to describe the negative class. We observed "
            f"that those rows were NOT REPORTED, not that they were not "
            f"fraud -- use {NEGATIVE_NAME!r}."
        )


@dataclass
class ContaminationScenario:
    """What a given contamination rate implies, under the mixture model."""

    rate: float
    observed_prevalence: float

    @property
    def true_prevalence(self) -> float:
        """``π_obs + c(1 − π_obs)`` — contaminated rows are positives counted
        on the wrong side."""
        return self.observed_prevalence + self.rate * (1.0 - self.observed_prevalence)

    @property
    def prevalence_understatement(self) -> Optional[float]:
        if self.observed_prevalence <= 0:
            return None
        return self.true_prevalence / self.observed_prevalence

    @property
    def attenuation(self) -> float:
        """Factor by which a true group difference is shrunk: ``1 − c``."""
        return 1.0 - self.rate

    @property
    def effect_correction(self) -> Optional[float]:
        """Divide an observed difference by ``1 − c`` to recover the true one."""
        return (1.0 / self.attenuation) if self.attenuation > 0 else None

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "observed_prevalence": self.observed_prevalence,
            "true_prevalence": self.true_prevalence,
            "prevalence_understatement": self.prevalence_understatement,
            "attenuation": self.attenuation,
            "effect_correction": self.effect_correction,
        }


def contamination_scenarios(
    observed_prevalence: float, rates=DEFAULT_RATES
) -> list[ContaminationScenario]:
    """What a range of plausible contamination rates would imply."""
    if not (0.0 <= observed_prevalence <= 1.0):
        raise PUError(
            f"observed_prevalence must be in [0,1]; got {observed_prevalence}")
    out = []
    for r in rates:
        if not (0.0 <= r < 1.0):
            raise PUError(f"contamination rate must be in [0,1); got {r}")
        out.append(ContaminationScenario(rate=float(r),
                                         observed_prevalence=observed_prevalence))
    return out


def correct_for_contamination(observed_difference: float, rate: float) -> float:
    """Recover a true group difference from a contaminated observation.

    Contamination only ever shrinks a difference, so this only ever grows it
    in magnitude — it cannot turn a null result into a real one by itself.
    """
    if not (0.0 <= rate < 1.0):
        raise PUError(f"rate must be in [0,1); got {rate}")
    return observed_difference / (1.0 - rate)


@dataclass
class SpikeInResult:
    """How a statistic moves as known fraud is hidden in the negative class."""

    baseline: float
    rates: list[float] = field(default_factory=list)
    values: list[Optional[float]] = field(default_factory=list)
    unachievable: list[float] = field(default_factory=list)
    statistic_name: str = "statistic"

    @property
    def deltas(self) -> list[Optional[float]]:
        return [None if v is None else v - self.baseline for v in self.values]

    @property
    def relative_deltas(self) -> list[Optional[float]]:
        if self.baseline == 0:
            return [None for _ in self.values]
        return [None if v is None else (v - self.baseline) / abs(self.baseline)
                for v in self.values]

    def crosses_threshold(self, threshold: float) -> bool:
        """Does contamination move the statistic across ``threshold``?

        **This is the question that usually matters**, and it is not the same
        as "did it move a lot". Under the mixture model a contaminated
        measurement is about ``(1−c)`` times the true one, so the relative
        movement is roughly ``c`` whatever the effect size — a 5% rate moves
        almost everything by about 5%. What decides whether a conclusion
        survives is whether the statistic stays on the same side of the line
        the conclusion was drawn at (an IV cut-off, a minimum lift, zero).
        """
        base_above = self.baseline >= threshold
        return any(v is not None and (v >= threshold) != base_above
                   for v in self.values)

    def sign_flips(self) -> bool:
        """Did the statistic change sign? The ``threshold=0`` special case."""
        import math
        base = self.baseline
        if base == 0 or math.isnan(base):
            return False
        return any(v is not None and v != 0 and (v > 0) != (base > 0)
                   for v in self.values)

    def is_robust(self, tolerance: float = 0.20,
                  threshold: Optional[float] = None) -> bool:
        """Whether the conclusion survives the tested contamination.

        With ``threshold`` supplied this asks the useful question — does the
        statistic stay on the same side of the decision line? Without one it
        falls back to a relative-movement bound, which is a weaker check:
        expected movement is already about ``c``, so a tolerance below the
        largest tested rate will fire on well-behaved statistics.
        """
        if self.sign_flips():
            return False
        if threshold is not None:
            return not self.crosses_threshold(threshold)
        return all(d is None or abs(d) <= tolerance
                   for d in self.relative_deltas)

    def summary(self, tolerance: float = 0.20) -> str:
        lines = [
            f"Contamination sensitivity ({SCHEMA}) -- "
            f"{self.statistic_name} = {self.baseline:.6g} as measured",
        ]
        for r, v, d in zip(self.rates, self.values, self.relative_deltas):
            if v is None:
                lines.append(f"  c={r:.1%}: not achievable with the available positives")
            else:
                lines.append(f"  c={r:.1%}: {v:.6g} ({d:+.1%})")
        verdict = "ROBUST" if self.is_robust(tolerance) else "SENSITIVE"
        lines.append(f"  verdict: {verdict} to contamination up to "
                     f"{max(self.rates) if self.rates else 0:.1%}")
        if self.sign_flips():
            lines.append(
                "  WARNING: the statistic CHANGES SIGN under plausible "
                "contamination -- the conclusion it supports is not safe to "
                "state.")
        if self.unachievable:
            lines.append(
                f"  note: rates {self.unachievable} need more positives than "
                f"the sample contains, so they were not tested")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "statistic": self.statistic_name,
            "baseline": self.baseline,
            "rates": list(self.rates),
            "values": list(self.values),
            "relative_deltas": list(self.relative_deltas),
            "sign_flips": self.sign_flips(),
            "is_robust": self.is_robust(),
            "unachievable_rates": list(self.unachievable),
        }


def spike_in_sensitivity(
    frame,
    statistic: Callable[[Any], float],
    *,
    label_column: str = "_label",
    rates=DEFAULT_RATES,
    seed: int = 42,
    statistic_name: str = "statistic",
) -> SpikeInResult:
    """Hide known fraud in the negative class and see whether the conclusion holds.

    For each rate ``c``, relabels enough confirmed positives as negative that
    they make up ``c`` of the negative class, then recomputes ``statistic``.
    This simulates fraud that occurred but was never reported.

    ``statistic`` receives the relabelled frame and returns a float — pass
    whatever number the conclusion actually rests on (a group difference, an
    IV, an AUC).
    """
    import numpy as np

    if label_column not in frame.columns:
        raise PUError(
            f"label column {label_column!r} not in frame; "
            f"available: {list(frame.columns)[:20]}")

    labels = frame[label_column]
    pos_idx = list(frame.index[labels == 1])
    n_neg = int((labels == 0).sum())
    if not pos_idx:
        raise PUError(
            "no positive rows to spike in -- contamination sensitivity needs "
            "confirmed fraud to hide")
    if n_neg == 0:
        raise PUError("no negative rows -- nothing to contaminate")

    baseline = float(statistic(frame))
    rng = np.random.default_rng(seed)

    values: list[Optional[float]] = []
    unachievable: list[float] = []
    for c in rates:
        if not (0.0 <= c < 1.0):
            raise PUError(f"contamination rate must be in [0,1); got {c}")
        # n_move / (n_neg + n_move) = c
        n_move = int(round(c * n_neg / (1.0 - c))) if c > 0 else 0
        if n_move > len(pos_idx):
            values.append(None)
            unachievable.append(float(c))
            continue
        chosen = rng.choice(pos_idx, size=n_move, replace=False) if n_move else []
        spiked = frame.copy()
        if len(chosen):
            spiked.loc[list(chosen), label_column] = 0
        try:
            values.append(float(statistic(spiked)))
        except Exception as e:  # noqa: BLE001
            log.warning("spike_in_sensitivity: statistic failed at c=%s: %s", c, e)
            values.append(None)

    return SpikeInResult(
        baseline=baseline, rates=[float(r) for r in rates], values=values,
        unachievable=unachievable, statistic_name=statistic_name,
    )
