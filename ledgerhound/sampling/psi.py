"""
ledgerhound.sampling.psi
======================

PSI (Population Stability Index) and KS test computation.

PSI measures how much a sample distribution diverges from a reference
(population) distribution. It is the primary metric for sampling
certification.

Thresholds (industry standard):
  PSI < 0.10  -> Low shift    -> PASS
  PSI 0.10-0.20 -> Moderate   -> WARNING
  PSI > 0.20  -> High shift   -> FAIL

KS test supplements PSI for continuous features: it detects any
distributional difference, not just the mean-variance shift that PSI
is most sensitive to.

Inputs
------
This module consumes two things:
  1. population_dist : dict returned by population profiler (Trino result)
       For numeric:      {bucket_idx: population_pct}
       For categorical:  {value:      population_pct}
  2. sample_series   : pd.Series (local, from the cuDF/pandas dataframe)

The population distribution is always computed from Trino aggregations.
The sample distribution is always computed locally.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Small epsilon to avoid log(0) in PSI formula
_EPS = 1e-6


class PSIStatus(str, Enum):
    PASS    = "pass"       # PSI < 0.10
    WARNING = "warning"   # 0.10 <= PSI < 0.20
    FAIL    = "fail"       # PSI >= 0.20


def _psi_status(psi: float) -> PSIStatus:
    if psi < 0.10:
        return PSIStatus.PASS
    if psi < 0.20:
        return PSIStatus.WARNING
    return PSIStatus.FAIL


@dataclass
class FeaturePSI:
    """PSI result for a single feature."""
    feature: str
    psi: float
    status: PSIStatus
    n_buckets: int
    n_sample: int
    n_population: Optional[int]
    feature_type: str          # "numeric" or "categorical"
    # Per-bucket detail
    bucket_details: list[dict] = field(default_factory=list)
    # KS test (numeric only)
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    ks_status: Optional[PSIStatus] = None
    # Confidence interval on the PSI estimate
    psi_ci_lower: Optional[float] = None
    psi_ci_upper: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "status": self.status.value,
            "n_buckets": self.n_buckets,
            "n_sample": self.n_sample,
            "n_population": self.n_population,
            "feature_type": self.feature_type,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "ks_status": self.ks_status.value if self.ks_status else None,
            "psi_ci_lower": self.psi_ci_lower,
            "psi_ci_upper": self.psi_ci_upper,
        }


@dataclass
class PSIResult:
    """PSI results across all certified features."""
    features: list[FeaturePSI] = field(default_factory=list)
    weighted_psi: float = 0.0
    overall_status: PSIStatus = PSIStatus.PASS
    n_pass: int = 0
    n_warning: int = 0
    n_fail: int = 0

    def summary(self) -> str:
        lines = [
            f"PSI Summary -- {len(self.features)} features",
            f"  Weighted PSI : {self.weighted_psi:.4f} ({self.overall_status.value.upper()})",
            f"  Pass         : {self.n_pass}",
            f"  Warning      : {self.n_warning}",
            f"  Fail         : {self.n_fail}",
        ]
        fails = [f for f in self.features if f.status == PSIStatus.FAIL]
        if fails:
            lines.append("  Failing features:")
            for f in sorted(fails, key=lambda x: -x.psi)[:10]:
                lines.append(f"    {f.feature:<40} PSI={f.psi:.4f}")
        return "\n".join(lines)


# -- Bucket boundary helpers ------------------------------------------------


def compute_bucket_bounds(
    series: pd.Series,
    n_buckets: int = 20,
    strategy: str = "quantile",
) -> list[float]:
    """
    Derive bucket boundaries from a sample series.

    Parameters
    ----------
    series    : Local pandas/cuDF Series (non-null values used).
    n_buckets : Number of buckets. More = finer PSI, but needs larger sample.
    strategy  : "quantile" (equal-frequency) or "uniform" (equal-width).

    Returns sorted list of n_buckets+1 boundary values.
    """
    clean = series.dropna()
    if len(clean) < n_buckets * 2:
        # Too few rows for requested buckets -- reduce
        n_buckets = max(5, len(clean) // 20)
        log.warning(f"Reducing bucket count to {n_buckets} due to small sample.")

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_buckets + 1)
        bounds = list(np.quantile(clean.astype(float), quantiles))
    else:
        mn, mx = float(clean.min()), float(clean.max())
        bounds = list(np.linspace(mn, mx, n_buckets + 1))

    # Deduplicate bounds (can happen with heavy ties at quantiles)
    bounds = sorted(set(bounds))
    if len(bounds) < 3:
        log.warning("Feature has near-constant values -- PSI may be unreliable.")

    return [float(b) for b in bounds]


# -- PSI computation --------------------------------------------------------


class PSICalculator:
    """
    Compute PSI between a population (Trino) and a local sample.

    Usage
    -----
    calc = PSICalculator()

    # Numeric feature
    bounds = compute_bucket_bounds(sample_df["txn_amount"])
    pop_dist = parse_bucket_distribution(trino_rows)  # from profiler.population
    result = calc.numeric(
        feature="txn_amount",
        population_dist=pop_dist,
        sample_series=sample_df["txn_amount"],
        bounds=bounds,
    )

    # Categorical feature
    pop_dist = parse_categorical_distribution(trino_rows)
    result = calc.categorical(
        feature="merchant_category",
        population_dist=pop_dist,
        sample_series=sample_df["merchant_category"],
    )
    """

    def __init__(self, ks_alpha: float = 0.05) -> None:
        self.ks_alpha = ks_alpha

    def numeric(
        self,
        feature: str,
        population_dist: dict[int, float],   # {bucket_idx: population_pct}
        sample_series: pd.Series,
        bounds: list[float],
        n_population: Optional[int] = None,
    ) -> FeaturePSI:
        """PSI for a continuous numeric feature."""
        clean = sample_series.dropna().astype(float)
        n_sample = len(clean)

        if n_sample < 50:
            log.warning(f"Feature '{feature}': only {n_sample} non-null sample values. PSI unreliable.")

        # Compute sample bucket distribution using the same bounds
        sample_dist = _numeric_sample_dist(clean, bounds)
        n_buckets = len(bounds) - 1

        # PSI per bucket
        psi_total = 0.0
        bucket_details = []
        for idx in range(n_buckets):
            pop_pct = population_dist.get(idx, 0.0)
            samp_pct = sample_dist.get(idx, 0.0)

            # Apply epsilon to avoid log(0)
            pop_adj = max(pop_pct, _EPS)
            samp_adj = max(samp_pct, _EPS)

            bucket_psi = (samp_adj - pop_adj) * math.log(samp_adj / pop_adj)
            psi_total += bucket_psi
            bucket_details.append({
                "bucket": idx,
                "lo": bounds[idx],
                "hi": bounds[idx + 1],
                "population_pct": round(pop_pct, 6),
                "sample_pct": round(samp_pct, 6),
                "bucket_psi": round(bucket_psi, 6),
            })

        # KS test (requires raw sample values, not just distribution)
        ks_stat, ks_pval, ks_status = _ks_from_buckets(
            population_dist, sample_dist, n_buckets, self.ks_alpha
        )

        # PSI confidence interval (bootstrap approximation)
        ci_lower, ci_upper = _psi_confidence_interval(psi_total, n_sample, n_buckets)

        return FeaturePSI(
            feature=feature,
            psi=psi_total,
            status=_psi_status(psi_total),
            n_buckets=n_buckets,
            n_sample=n_sample,
            n_population=n_population,
            feature_type="numeric",
            bucket_details=bucket_details,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ks_status=ks_status,
            psi_ci_lower=ci_lower,
            psi_ci_upper=ci_upper,
        )

    def categorical(
        self,
        feature: str,
        population_dist: dict[str, float],   # {value: population_pct}
        sample_series: pd.Series,
        n_population: Optional[int] = None,
    ) -> FeaturePSI:
        """PSI for a categorical feature."""
        clean = sample_series.dropna().astype(str)
        n_sample = len(clean)

        value_counts = clean.value_counts(normalize=True)
        # Align on population categories + "OTHER" bucket for unseen values
        all_cats = set(population_dist.keys()) | set(value_counts.index)

        psi_total = 0.0
        bucket_details = []
        for cat in all_cats:
            pop_pct = population_dist.get(cat, 0.0)
            samp_pct = float(value_counts.get(cat, 0.0))

            pop_adj = max(pop_pct, _EPS)
            samp_adj = max(samp_pct, _EPS)
            bucket_psi = (samp_adj - pop_adj) * math.log(samp_adj / pop_adj)
            psi_total += bucket_psi
            bucket_details.append({
                "value": cat,
                "population_pct": round(pop_pct, 6),
                "sample_pct": round(samp_pct, 6),
                "bucket_psi": round(bucket_psi, 6),
            })

        ci_lower, ci_upper = _psi_confidence_interval(psi_total, n_sample, len(all_cats))

        return FeaturePSI(
            feature=feature,
            psi=psi_total,
            status=_psi_status(psi_total),
            n_buckets=len(all_cats),
            n_sample=n_sample,
            n_population=n_population,
            feature_type="categorical",
            bucket_details=sorted(bucket_details, key=lambda x: -x["bucket_psi"]),
            psi_ci_lower=ci_lower,
            psi_ci_upper=ci_upper,
        )

    def batch(
        self,
        sample_df: pd.DataFrame,
        numeric_population_dists: dict[str, dict[int, float]],
        categorical_population_dists: dict[str, dict[str, float]],
        numeric_bounds: dict[str, list[float]],
        weights: Optional[dict[str, float]] = None,
        n_population: Optional[int] = None,
    ) -> PSIResult:
        """
        Compute PSI for all features and return a PSIResult.

        Parameters
        ----------
        numeric_population_dists   : {feature: {bucket_idx: pct}} from Trino
        categorical_population_dists: {feature: {value: pct}} from Trino
        numeric_bounds             : {feature: bounds_list} from compute_bucket_bounds
        weights                    : Optional {feature: weight} for weighted PSI.
                                     Higher weight = this feature matters more for
                                     sampling quality. Typically set to feature
                                     importance from prior model run.
        """
        results: list[FeaturePSI] = []

        for feat, pop_dist in numeric_population_dists.items():
            if feat not in sample_df.columns:
                log.warning(f"Feature '{feat}' not in sample DataFrame -- skipping.")
                continue
            bounds = numeric_bounds.get(feat) or compute_bucket_bounds(sample_df[feat])
            results.append(self.numeric(feat, pop_dist, sample_df[feat], bounds, n_population))

        for feat, pop_dist in categorical_population_dists.items():
            if feat not in sample_df.columns:
                log.warning(f"Feature '{feat}' not in sample DataFrame -- skipping.")
                continue
            results.append(self.categorical(feat, pop_dist, sample_df[feat], n_population))

        # Weighted PSI
        w = weights or {}
        total_weight = sum(w.get(f.feature, 1.0) for f in results) or 1.0
        weighted_psi = sum(f.psi * w.get(f.feature, 1.0) for f in results) / total_weight

        n_pass    = sum(1 for f in results if f.status == PSIStatus.PASS)
        n_warning = sum(1 for f in results if f.status == PSIStatus.WARNING)
        n_fail    = sum(1 for f in results if f.status == PSIStatus.FAIL)

        overall = _psi_status(weighted_psi)

        return PSIResult(
            features=results,
            weighted_psi=round(weighted_psi, 6),
            overall_status=overall,
            n_pass=n_pass,
            n_warning=n_warning,
            n_fail=n_fail,
        )


# -- Internal helpers -------------------------------------------------------


def _numeric_sample_dist(series: pd.Series, bounds: list[float]) -> dict[int, float]:
    """Bin a numeric series into buckets defined by bounds. Returns {idx: pct}."""
    n = len(series)
    if n == 0:
        return {}
    bins = pd.cut(series, bins=bounds, include_lowest=True, right=False, labels=False)
    counts = bins.value_counts(normalize=True)
    return {int(idx): float(pct) for idx, pct in counts.items()}


def _ks_from_buckets(
    pop_dist: dict[int, float],
    samp_dist: dict[int, float],
    n_buckets: int,
    alpha: float,
) -> tuple[Optional[float], Optional[float], Optional[PSIStatus]]:
    """
    Approximate KS statistic from bucket distributions.
    This is an approximation -- the real KS test needs raw values.
    Returns (ks_stat, p_value_approx, status).
    """
    try:
        pop_cdf = np.cumsum([pop_dist.get(i, 0.0) for i in range(n_buckets)])
        samp_cdf = np.cumsum([samp_dist.get(i, 0.0) for i in range(n_buckets)])
        ks_stat = float(np.max(np.abs(pop_cdf - samp_cdf)))
        # Approximate p-value using Kolmogorov distribution
        # This is a heuristic -- exact p-value requires sample size
        ks_status = PSIStatus.PASS if ks_stat < 0.05 else (
            PSIStatus.WARNING if ks_stat < 0.10 else PSIStatus.FAIL
        )
        return ks_stat, None, ks_status
    except Exception:
        return None, None, None


def _psi_confidence_interval(
    psi: float,
    n_sample: int,
    n_buckets: int,
    z: float = 1.96,  # 95% CI
) -> tuple[Optional[float], Optional[float]]:
    """
    Approximate confidence interval for PSI.
    Based on: Var(PSI) ~ 2 * (1/n_sample) * n_buckets
    """
    if n_sample < 1:
        return None, None
    variance = 2.0 * n_buckets / n_sample
    std = math.sqrt(variance)
    half_width = z * std
    return (max(0.0, round(psi - half_width, 6)), round(psi + half_width, 6))
