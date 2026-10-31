"""
sentinel.pipeline.cluster_psi
==============================

**Cluster-aware PSI assessment** (V4-024) — fixes a live defect in the
shipped certificate.

The defect
----------
`sentinel/sampling/psi.py` grades PSI against fixed folklore thresholds:
``<0.10 PASS``, ``0.10–0.20 WARNING``, ``>0.20 FAIL``. Those numbers do not
depend on sample size at all. Two consequences, in opposite directions:

* **At 8 crore rows**, sampling noise puts PSI far below 0.10 even under a
  real shift, so the threshold is enormously insensitive — it will **pass a
  bad sample**.
* **On cluster-sampled data**, the noise floor is *higher* than the row count
  suggests, so a threshold calibrated for i.i.d. rows can **fail a good
  sample**.

The fraud pipeline draws **payees and then takes their entire history**
(`history_fetch`). That is cluster sampling: rows within one payee are
strongly correlated, so 550k rows carry nowhere near 550k rows' worth of
independent information. The certificate — the artifact the whole product
claim rests on — is currently computing at the wrong effective sample size
and reporting a verdict as if it weren't.

What this module does
---------------------
1. **Estimates the intra-cluster correlation (ICC)** by one-way random-effects
   ANOVA over entities.
2. **Computes the design effect** ``deff = 1 + (m̄ − 1)·ICC`` and the effective
   sample size ``n_eff = n / deff``.
3. **Derives a size-aware critical value.** Under the null of no shift,
   ``n·PSI`` is asymptotically ``χ²(B−1)`` (Yurdakul 2018), so a level-α
   critical value is ``χ²_{B−1,1−α} / n_eff``.
4. **Reports both verdicts and flags disagreement.** The folklore verdict is
   kept — it is what people expect and what the incumbent uses — but shown
   next to the size-aware one, so "these disagree" becomes visible instead
   of invisible.

Honest limits
-------------
* The ``n·PSI ~ χ²(B−1)`` result is asymptotic and assumes the reference
  distribution is known rather than estimated. With a large reference (the
  population side of a PSI) that is reasonable; with two small samples it is
  optimistic.
* Applying one feature-level ICC to every bin is standard survey practice but
  an approximation: strictly, each bin proportion has its own design effect.
* ICC is estimated, not known, and is unstable with few clusters.

Where an exact answer is available, take it: :func:`entity_level_frame`
collapses to one row per entity so PSI can be computed with **no clustering
at all**. It answers a different question — how entity-level summaries are
distributed, not rows — but it answers it without approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

SCHEMA = "cluster_psi/v1"

DEFAULT_ALPHA = 0.05

# Folklore bands, mirrored from sampling/psi.py so both verdicts are visible.
FOLKLORE_PASS = 0.10
FOLKLORE_FAIL = 0.20

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_UNKNOWN = "UNKNOWN"


class ClusterPSIError(ValueError):
    """The cluster-aware assessment cannot be performed as specified."""


@dataclass
class ICCEstimate:
    """Intra-cluster correlation, by one-way random-effects ANOVA."""

    icc: Optional[float]
    n_rows: int
    n_clusters: int
    mean_cluster_size: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.icc is not None

    def to_dict(self) -> dict:
        return {
            "icc": self.icc, "n_rows": self.n_rows,
            "n_clusters": self.n_clusters,
            "mean_cluster_size": self.mean_cluster_size,
            "reason": self.reason,
        }


def estimate_icc(values, entity_ids) -> ICCEstimate:
    """One-way random-effects ICC for ``values`` grouped by ``entity_ids``.

    ``ICC = (MSB − MSW) / (MSB + (m₀ − 1)·MSW)`` with the unequal-cluster-size
    correction ``m₀ = (N − Σmᵢ²/N) / (k − 1)``. Clamped to ``[0, 1]``: a
    negative estimate means clusters are *less* alike than chance, which for
    a design effect is indistinguishable from no clustering.
    """
    import numpy as np
    import pandas as pd

    s = pd.Series(np.asarray(values, dtype="float64"))
    g = pd.Series(list(entity_ids), index=s.index)
    ok = s.notna()
    s, g = s[ok], g[ok]

    n = len(s)
    if n == 0:
        return ICCEstimate(None, 0, 0, 0.0, "no non-null values")

    sizes = g.value_counts()
    k = len(sizes)
    mean_size = float(n / k) if k else 0.0

    if k < 2:
        return ICCEstimate(None, n, k, mean_size,
                           "fewer than 2 clusters -- ICC is undefined")
    if n == k:
        return ICCEstimate(0.0, n, k, mean_size,
                           "every cluster has one row -- no within-cluster "
                           "variance, so no clustering effect")

    grand = float(s.mean())
    means = s.groupby(g).mean()
    counts = s.groupby(g).count()

    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(((s - g.map(means)) ** 2).sum())

    df_b, df_w = k - 1, n - k
    if df_w <= 0:
        return ICCEstimate(0.0, n, k, mean_size, "no within-cluster degrees of freedom")

    ms_b = ss_between / df_b
    ms_w = ss_within / df_w

    m0 = (n - float((sizes ** 2).sum()) / n) / (k - 1)
    denom = ms_b + (m0 - 1.0) * ms_w
    if denom <= 0 or not np.isfinite(denom):
        return ICCEstimate(0.0, n, k, mean_size, "degenerate variance decomposition")

    icc = (ms_b - ms_w) / denom
    if not np.isfinite(icc):
        return ICCEstimate(0.0, n, k, mean_size, "non-finite ICC estimate")
    return ICCEstimate(float(min(max(icc, 0.0), 1.0)), n, k, mean_size)


@dataclass
class ClusterDiagnostics:
    """Design effect and effective sample size for a clustered draw."""

    icc_estimate: ICCEstimate
    design_effect: float
    n_effective: float

    @property
    def inflation(self) -> float:
        """How many real rows one 'effective' row is worth."""
        return self.design_effect

    def to_dict(self) -> dict:
        return {
            "design_effect": self.design_effect,
            "n_effective": self.n_effective,
            "icc": self.icc_estimate.to_dict(),
        }


def design_effect(icc: float, mean_cluster_size: float) -> float:
    """``deff = 1 + (m̄ − 1)·ICC``. Never below 1 — clustering cannot add
    information."""
    return max(1.0, 1.0 + (mean_cluster_size - 1.0) * icc)


def cluster_diagnostics(values, entity_ids) -> ClusterDiagnostics:
    est = estimate_icc(values, entity_ids)
    if not est.usable:
        # Unknown ICC: assume no clustering rather than invent one, and let
        # the caller see the reason.
        return ClusterDiagnostics(est, 1.0, float(est.n_rows))
    deff = design_effect(est.icc, est.mean_cluster_size)
    return ClusterDiagnostics(est, deff, est.n_rows / deff if deff else est.n_rows)


def psi_critical_value(
    n_effective: float, n_buckets: int, alpha: float = DEFAULT_ALPHA
) -> Optional[float]:
    """Level-α critical PSI under the null of no shift.

    ``n·PSI ~ χ²(B−1)`` asymptotically, so the critical value is
    ``χ²_{B−1,1−α} / n_eff``.
    """
    from scipy.stats import chi2

    if n_buckets < 2:
        return None
    if n_effective <= 0:
        return None
    if not (0.0 < alpha < 1.0):
        raise ClusterPSIError(f"alpha must be in (0,1); got {alpha}")
    return float(chi2.ppf(1.0 - alpha, df=n_buckets - 1) / n_effective)


def _folklore_status(psi: float) -> str:
    if psi < FOLKLORE_PASS:
        return STATUS_PASS
    if psi < FOLKLORE_FAIL:
        return STATUS_WARNING
    return STATUS_FAIL


@dataclass
class ClusterAwarePSI:
    """A PSI value assessed both ways, with the disagreement made explicit."""

    feature: str
    psi: float
    n_rows: int
    n_buckets: int
    diagnostics: ClusterDiagnostics
    critical_value: Optional[float]
    alpha: float = DEFAULT_ALPHA
    computation_grain: str = "row"

    @property
    def folklore_status(self) -> str:
        return _folklore_status(self.psi)

    @property
    def size_aware_status(self) -> str:
        if self.critical_value is None:
            return STATUS_UNKNOWN
        return STATUS_FAIL if self.psi > self.critical_value else STATUS_PASS

    @property
    def verdicts_disagree(self) -> bool:
        sa = self.size_aware_status
        if sa == STATUS_UNKNOWN:
            return False
        fk = self.folklore_status
        return (fk == STATUS_PASS) != (sa == STATUS_PASS)

    @property
    def message(self) -> str:
        d = self.diagnostics
        base = (
            f"{self.feature}: PSI={self.psi:.4f} over {self.n_rows:,} rows in "
            f"{d.icc_estimate.n_clusters:,} entities "
            f"(mean {d.icc_estimate.mean_cluster_size:.1f} rows each). "
            f"ICC={d.icc_estimate.icc if d.icc_estimate.icc is not None else float('nan'):.3f} "
            f"-> design effect {d.design_effect:.1f}x, so the effective sample "
            f"size is {d.n_effective:,.0f}, not {self.n_rows:,}."
        )
        if self.critical_value is None:
            return base + " No size-aware critical value could be computed."
        base += (
            f" Size-aware critical PSI at alpha={self.alpha} is "
            f"{self.critical_value:.4f} -> {self.size_aware_status}; "
            f"fixed-threshold verdict is {self.folklore_status}."
        )
        if self.verdicts_disagree:
            if self.folklore_status == STATUS_PASS:
                base += (
                    " THEY DISAGREE: the fixed 0.10/0.20 thresholds pass this "
                    "feature, but at the true effective sample size the shift "
                    "is statistically significant. The fixed thresholds are "
                    "insensitive at this scale."
                )
            else:
                base += (
                    " THEY DISAGREE: the fixed thresholds flag this feature, "
                    "but at the true effective sample size the difference is "
                    "within sampling noise -- clustering makes the noise floor "
                    "higher than the raw row count suggests."
                )
        return base

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "feature": self.feature,
            "psi": self.psi,
            "n_rows": self.n_rows,
            "n_buckets": self.n_buckets,
            "computation_grain": self.computation_grain,
            "alpha": self.alpha,
            "critical_value": self.critical_value,
            "folklore_status": self.folklore_status,
            "size_aware_status": self.size_aware_status,
            "verdicts_disagree": self.verdicts_disagree,
            "diagnostics": self.diagnostics.to_dict(),
            "message": self.message,
        }


def assess_psi(
    *,
    feature: str,
    psi: float,
    frame,
    entity_keys: list[str],
    value_column: str = "",
    n_buckets: int = 10,
    alpha: float = DEFAULT_ALPHA,
) -> ClusterAwarePSI:
    """Re-assess an already-computed PSI accounting for cluster sampling.

    Deliberately takes ``psi`` rather than recomputing it — `sampling/psi.py`
    owns that calculation and this must not fork it.
    """
    missing = [c for c in entity_keys if c not in frame.columns]
    if missing:
        raise ClusterPSIError(
            f"entity keys {missing} not in frame; available: "
            f"{list(frame.columns)[:20]}")
    col = value_column or feature
    if col not in frame.columns:
        raise ClusterPSIError(
            f"value column {col!r} not in frame -- the ICC is estimated from "
            f"the feature's own values, so it must be present")

    ents = list(frame[entity_keys].itertuples(index=False, name=None))
    diag = cluster_diagnostics(frame[col], ents)
    crit = psi_critical_value(diag.n_effective, n_buckets, alpha)
    return ClusterAwarePSI(
        feature=feature, psi=psi, n_rows=len(frame), n_buckets=n_buckets,
        diagnostics=diag, critical_value=crit, alpha=alpha,
    )


def entity_level_frame(frame, entity_keys: list[str], value_columns: list[str],
                       how: str = "mean"):
    """Collapse to one row per entity so PSI can be computed without clustering.

    The exact alternative to a design-effect correction: it answers a
    different question — how entity-level summaries are distributed rather
    than rows — but answers it with no approximation, and ``n`` is then
    genuinely the number of independent units.
    """
    missing = [c for c in list(entity_keys) + list(value_columns)
               if c not in frame.columns]
    if missing:
        raise ClusterPSIError(f"columns {missing} not in frame")
    if how not in {"mean", "median", "max", "min", "sum"}:
        raise ClusterPSIError(
            f"how must be one of mean|median|max|min|sum; got {how!r}")
    return (frame.groupby(list(entity_keys), dropna=False)[list(value_columns)]
            .agg(how).reset_index())
