"""
ledgerhound.analysis.multiplicity
==============================

V5-001. Multiple-comparisons correction — the missing floor under every
per-feature p-value this package produces.

Before this module, nothing in `ledgerhound/` corrected for multiplicity: the
words ``bonferroni``, ``benjamini``, ``hochberg``, ``fdr`` and
``family_wise`` matched zero files. Two shipped paths tested every feature at
a raw threshold:

  * ``analysis/selection.py`` -- Boruta's binomial test per feature at
    alpha=0.05. The original Boruta algorithm corrects at exactly this step
    (BorutaPy calls ``multipletests``); our native Tree-SHAP reimplementation
    dropped it. With 200 candidates that is ~10 pure-noise features
    "confirmed" by chance, every run, silently.
  * ``analysis/signal_decay.py`` -- a KS p-value per feature, stored raw.

Why FDR rather than family-wise error. Feature screening is a *discovery*
problem: we want the expected proportion of false positives among the
selected set bounded, not the probability of a single false positive across
the whole family. Bonferroni at 200 features would reject almost nothing and
push analysts to turn the correction off, which is worse than a correction
tuned to the job. Benjamini-Hochberg is the default; Bonferroni is available
for the rare case where one false positive is genuinely unacceptable.

No new dependency: ``scipy.stats.false_discovery_control`` has existed since
scipy 1.11, which is already this project's declared floor.

**A correction that is not recorded is not a correction.** Every result here
carries the method, alpha and family size, and every call site is expected to
put ``to_dict()`` into whatever it persists, so a certificate can cite what
was actually applied rather than what someone assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

# "none" is a real, selectable method rather than an absence. Reproducing an
# older stored result means reproducing what it actually did, and that is only
# possible if "no correction" is a value the record can hold.
VALID_METHODS = ("fdr_bh", "fdr_by", "bonferroni", "none")

DEFAULT_METHOD = "fdr_bh"
DEFAULT_ALPHA = 0.05


@dataclass
class MultiplicityResult:
    """What was corrected, how, and over how large a family."""

    method: str
    alpha: float
    n_tests: int                                   # family size actually corrected
    adjusted: dict[str, float] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    # Tests that carried no usable p-value (NaN/None/out of range). Excluded
    # from the family rather than coerced: a categorical feature scored by
    # Cramer's V is not a failed test, it is not a test at all, and counting
    # it would inflate the denominator and weaken every real correction.
    skipped: list[str] = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    def is_rejected(self, name: str) -> bool:
        """True only for a test that was in the family AND survived."""
        return name in self._rejected_set

    def __post_init__(self) -> None:
        self._rejected_set = set(self.rejected)

    def q(self, name: str) -> float:
        """Adjusted p-value, or nan for a skipped test."""
        return self.adjusted.get(name, float("nan"))

    def summary(self) -> str:
        if self.method == "none":
            return (f"NO multiplicity correction applied "
                    f"({self.n_tests} tests at raw alpha={self.alpha})")
        line = (f"{self.method} correction at alpha={self.alpha} over a family "
                f"of {self.n_tests}: {len(self.rejected)} significant")
        if self.skipped:
            line += f" ({self.n_skipped} had no p-value and were excluded)"
        return line

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_rejected": len(self.rejected),
            "n_skipped": self.n_skipped,
            "skipped": list(self.skipped),
            "adjusted": dict(self.adjusted),
        }


def _usable(p) -> bool:
    """A p-value we can correct. Rejects None, NaN and anything outside
    [0, 1] -- signal_decay emits float('nan') for categorical features, and a
    silently-coerced NaN would become a spurious rejection."""
    if p is None:
        return False
    try:
        v = float(p)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v) and 0.0 <= v <= 1.0


def _bh_adjust(pvalues: list[float], *, by: bool = False) -> list[float]:
    """Benjamini-Hochberg (or Benjamini-Yekutieli) adjusted p-values.

    Implemented here rather than called from scipy so the arithmetic is
    pinned and testable in isolation; `test_multiplicity` cross-checks it
    against `scipy.stats.false_discovery_control` where that is available, so
    a divergence is caught rather than assumed away.
    """
    n = len(pvalues)
    if n == 0:
        return []
    # c(n) = 1 for BH (independence / positive dependence);
    # c(n) = sum(1/i) for BY (arbitrary dependence).
    c = sum(1.0 / i for i in range(1, n + 1)) if by else 1.0

    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 1.0
    # Walk from the largest p-value down, keeping the running minimum: this is
    # the step-up enforcement that keeps adjusted values monotone.
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = n - rank_from_end            # 1-based rank of this p-value
        val = pvalues[idx] * n * c / rank
        running = min(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def adjust(pvalues: Mapping[str, float], *,
           method: str = DEFAULT_METHOD,
           alpha: float = DEFAULT_ALPHA) -> MultiplicityResult:
    """Correct `pvalues` (name -> raw p) for multiple comparisons.

    The family is every entry with a usable p-value. Entries without one are
    reported in `skipped` and never rejected.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    names = list(pvalues)
    usable = [n for n in names if _usable(pvalues[n])]
    skipped = [n for n in names if n not in set(usable)]
    raw = [float(pvalues[n]) for n in usable]
    n_tests = len(usable)

    if n_tests == 0:
        return MultiplicityResult(method=method, alpha=alpha, n_tests=0,
                                  skipped=skipped)

    if method == "none":
        adj = raw[:]
    elif method == "bonferroni":
        adj = [min(1.0, p * n_tests) for p in raw]
    else:
        adj = _bh_adjust(raw, by=(method == "fdr_by"))

    adjusted = dict(zip(usable, adj))
    rejected = [n for n in usable if adjusted[n] <= alpha]
    return MultiplicityResult(method=method, alpha=alpha, n_tests=n_tests,
                              adjusted=adjusted, rejected=rejected,
                              skipped=skipped)
