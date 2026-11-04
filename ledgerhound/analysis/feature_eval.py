"""
ledgerhound.analysis.feature_eval
==============================

Evaluate candidate fraud features against a binary target (rank 9).

Unlike the EDA-cycle ``--derive`` hook (exploratory, distribution-only), this
is the *modelling-grade* evaluation the Feature-Engineering agent uses to
decide whether a feature earns a place in a recipe:

- **Information Value (IV)** via WOE binning -- the standard credit/fraud
  feature-strength metric.
- **KS statistic** -- max separation of the good/bad cumulative distributions.
- **AUC** (Mann-Whitney) -- rank-based separability; direction-free strength
  reported alongside.
- **Correlation pruning** -- keep the strongest feature from each correlated
  cluster so a recipe is not padded with redundant signals.

These are pure numpy/pandas computations (no LLM), so they are fully testable
and deterministic. Leakage / point-in-time correctness is NOT decided here --
it is a property of how the feature is *built*; this module only measures
predictive strength and records a leakage caveat for the FE reviewer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _iv_band(iv: float) -> str:
    a = abs(iv)
    if a < 0.02:
        return "useless"
    if a < 0.1:
        return "weak"
    if a < 0.3:
        return "medium"
    if a < 0.5:
        return "strong"
    return "suspicious"  # >0.5 often signals leakage / target contamination


@dataclass
class FeatureEval:
    """Predictive-strength evaluation of one candidate feature."""

    feature: str
    iv: float = 0.0
    iv_band: str = "useless"
    ks: float = 0.0
    auc: float = 0.5
    strength: float = 0.0       # direction-free: 2*|auc-0.5| (Gini magnitude)
    n: int = 0
    missing_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature, "iv": round(self.iv, 5),
            "iv_band": self.iv_band, "ks": round(self.ks, 5),
            "auc": round(self.auc, 5), "strength": round(self.strength, 5),
            "n": self.n, "missing_rate": round(self.missing_rate, 5),
        }


@dataclass
class FeatureEvalReport:
    evaluations: list[FeatureEval] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)   # dropped as redundant
    skipped: list[str] = field(default_factory=list)

    def top(self, n: int = 10) -> list[FeatureEval]:
        return self.evaluations[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluations": [e.to_dict() for e in self.evaluations],
            "pruned": list(self.pruned),
            "skipped": list(self.skipped),
        }


def information_value(x: Any, y: Any, *, n_bins: int = 10) -> float:
    """IV of feature ``x`` against binary target ``y`` (1 = event/fraud)."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return 0.0
    tot_bad = float(df["y"].sum())
    tot_good = float(len(df) - tot_bad)
    if tot_bad == 0 or tot_good == 0:
        return 0.0
    if pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > n_bins:
        df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    else:
        df["bin"] = df["x"].astype(str)
    iv = 0.0
    for _, g in df.groupby("bin", observed=True):
        bad = float(g["y"].sum())
        good = float(len(g) - bad)
        dist_bad = bad / tot_bad
        dist_good = good / tot_good
        woe = np.log((dist_good + _EPS) / (dist_bad + _EPS))
        iv += (dist_good - dist_bad) * woe
    return float(iv)


def ks_statistic(score: Any, y: Any) -> float:
    """Max gap between the bad/good cumulative distributions over ``score``."""
    import pandas as pd

    df = pd.DataFrame({"s": score, "y": y}).dropna().sort_values("s")
    tot_bad = float(df["y"].sum())
    tot_good = float(len(df) - tot_bad)
    if tot_bad == 0 or tot_good == 0:
        return 0.0
    cum_bad = df["y"].cumsum() / tot_bad
    cum_good = (1 - df["y"]).cumsum() / tot_good
    return float((cum_bad - cum_good).abs().max())


def auc_score(score: Any, y: Any) -> float:
    """Rank-based AUC (Mann-Whitney). 0.5 = no separation."""
    import pandas as pd

    df = pd.DataFrame({"s": score, "y": y}).dropna()
    n_pos = float(df["y"].sum())
    n_neg = float(len(df) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = df["s"].rank()
    sum_pos = float(ranks[df["y"] == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def js_divergence(a: Any, b: Any, *, n_bins: int = 10) -> float:
    """Jensen-Shannon divergence between two samples of one feature.

    Bounded [0, ln 2], symmetric, and finite even when a bin is empty on
    one side -- unlike the IV/PSI log-ratio family. NOTE: IV against a
    binary target and PSI between the two classes are the SAME statistic
    (both sum (p1-p2)*ln(p1/p2) over shared bins); JS is genuinely
    complementary evidence, not a rebranded copy.
    """
    import numpy as np
    import pandas as pd

    xa = pd.Series(a).dropna().astype(float)
    xb = pd.Series(b).dropna().astype(float)
    if xa.empty or xb.empty:
        return 0.0
    # shared quantile bins from the pooled sample
    pooled = pd.concat([xa, xb])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    pa, _ = np.histogram(xa, bins=edges)
    pb, _ = np.histogram(xb, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p: "np.ndarray", q: "np.ndarray") -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def wasserstein_std(a: Any, b: Any) -> float:
    """Wasserstein-1 distance between two samples, in pooled-std units.

    Geometry-aware: measures HOW FAR apart the distributions sit, which
    bin-ratio metrics (IV/PSI/JS) cannot see once bin memberships match.
    W1 = integral of |CDF_a - CDF_b|, computed exactly from the sorted
    samples; standardised by the pooled std so it is scale-free.
    """
    import numpy as np
    import pandas as pd

    xa = np.sort(pd.Series(a).dropna().astype(float).to_numpy())
    xb = np.sort(pd.Series(b).dropna().astype(float).to_numpy())
    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    # exact W1 for empirical CDFs: integrate |F_a - F_b| over the merged grid
    grid = np.concatenate([xa, xb])
    grid.sort(kind="mergesort")
    cdf_a = np.searchsorted(xa, grid, side="right") / len(xa)
    cdf_b = np.searchsorted(xb, grid, side="right") / len(xb)
    w1 = float(np.sum(np.abs(cdf_a - cdf_b)[:-1] * np.diff(grid)))
    pooled_std = float(np.sqrt((xa.var() * len(xa) + xb.var() * len(xb))
                               / (len(xa) + len(xb))))
    return w1 / pooled_std if pooled_std > 0 else 0.0


PERCENTILE_GRID = (
    "average", "p1", "p5", "p10", "p20", "p30", "p40", "p50",
    "p60", "p70", "p80", "p90", "p95", "p99",
)


def percentile_table(a: Any, b: Any, *, grid: tuple = PERCENTILE_GRID) -> dict:
    """Side-by-side percentile table for one feature, two populations.

    IV/KS/AUC collapse an entire distribution into a single scalar, which
    hides two failure modes a scalar cannot show: a feature whose two
    distributions REVERSE order in the tail (no single threshold works even
    though a summary statistic looks fine), and a feature that only
    separates in one tail (median/IV say "no signal", p95+ says otherwise).
    Denser at both tails than the middle -- that is where separation and
    pruning thresholds actually live.

    Deliberately the same two-sample ``(a, b)`` signature as
    ``js_divergence``/``wasserstein_std`` above -- this is the quantile-level
    sibling of those two scalar summaries, not a replacement for IV/KS/AUC.

    Returns ``{"a": {label: value, ...}, "b": {...}}``. Empty side -> {}.
    """
    import numpy as np
    import pandas as pd

    def _stats(x: Any) -> dict:
        s = pd.Series(x).dropna().astype(float)
        if s.empty:
            return {}
        out = {"average": float(s.mean())}
        for label in grid:
            if label == "average":
                continue
            q = int(label[1:]) / 100.0
            out[label] = float(np.quantile(s, q))
        return out

    return {"a": _stats(a), "b": _stats(b)}


def separation_callout(table: dict, *, tol: float = 1e-9) -> dict:
    """Mechanically derive where two populations diverge, overlap, or cross,
    from a ``percentile_table`` result -- not LLM-narrated, so it is
    reproducible from the same numbers every time.

    - ``crosses``: True if, at any two grid points, the ranking of a vs b
      flips (a > b at one quantile, a < b at another) -- the "no single
      threshold works" case a scalar IV cannot show.
    - ``diverges_from`` / ``overlaps_through``: the first grid label (in
      quantile order) where |a - b| exceeds a small fraction of the pooled
      spread, and the run of labels before it where they track together.
    - ``pruning_threshold``: value at the tail label where the two
      populations differ, so a caller can act on it (e.g. "b's p95 = 15;
      values above that are atypical for b").
    """
    a, b = table.get("a") or {}, table.get("b") or {}
    labels = [l for l in PERCENTILE_GRID if l != "average" and l in a and l in b]
    if not labels:
        return {"crosses": False, "diverges_from": None,
                "overlaps_through": None, "pruning_threshold": None}

    diffs = [a[l] - b[l] for l in labels]
    spread = max(abs(v) for v in (list(a.values()) + list(b.values()))) or 1.0
    signs = [1 if d > tol else (-1 if d < -tol else 0) for d in diffs]
    nonzero = [s for s in signs if s != 0]
    crosses = len(set(nonzero)) > 1

    overlaps_through = None
    diverges_from = None
    for label, d in zip(labels, diffs):
        if abs(d) / spread < 0.02:
            overlaps_through = label
        elif diverges_from is None:
            diverges_from = label

    pruning_label = labels[-1] if diverges_from else None
    pruning_threshold = b.get(pruning_label) if pruning_label else None

    return {
        "crosses": crosses,
        "diverges_from": diverges_from,
        "overlaps_through": overlaps_through,
        "pruning_threshold": pruning_threshold,
    }


def compare_percentile_tables(table_a: dict, table_b: dict, *,
                              rel_tol: float = 0.15) -> dict:
    """V3-055: cross-PERIOD comparison -- same population, two points in
    time (a 0.5.0 next-step: the Librarian's story model makes two data
    versions on one book directly diffable, but nothing automated the
    diff itself). Different from `separation_callout` above, which
    compares two DIFFERENT populations (fraud vs non-fraud) at one point
    in time -- this compares the SAME quantity's flat {label: value}
    percentile dict (e.g. ``percentile_table(...)["a"]``) across two
    stories.

    A label is "shifted" when its relative change exceeds `rel_tol`
    (default 15%), measured against the larger of the two values so a
    near-zero baseline does not blow up the ratio. Returns:

    - ``stable``: True iff every shared grid label is within tolerance.
    - ``shifted_labels``: which labels moved, worst first.
    - ``deltas``: {label: relative_delta} for every shared label.
    - ``compared_labels``: how many grid points had data on both sides
      (0 means nothing was actually comparable -- report, don't guess).
    """
    labels = [l for l in PERCENTILE_GRID if l in table_a and l in table_b]
    deltas: dict = {}
    for label in labels:
        va, vb = table_a[label], table_b[label]
        denom = max(abs(va), abs(vb), 1e-9)
        deltas[label] = (vb - va) / denom
    shifted = sorted((l for l in labels if abs(deltas[l]) > rel_tol),
                     key=lambda l: -abs(deltas[l]))
    return {
        "stable": len(shifted) == 0 and len(labels) > 0,
        "shifted_labels": shifted,
        "deltas": deltas,
        "compared_labels": len(labels),
    }


def evaluate_feature(df: Any, feature: str, target: str, *,
                     n_bins: int = 10) -> Optional[FeatureEval]:
    """Full evaluation of one feature; None if the column is unusable."""
    import pandas as pd

    if feature not in df.columns or target not in df.columns:
        return None
    y = df[target]
    x = df[feature]
    n = int(x.notna().sum())
    if n == 0:
        return None
    missing = float(x.isna().mean())
    iv = information_value(x, y, n_bins=n_bins)
    if pd.api.types.is_numeric_dtype(x):
        ks = ks_statistic(x, y)
        auc = auc_score(x, y)
    else:
        ks, auc = 0.0, 0.5
    return FeatureEval(
        feature=feature, iv=iv, iv_band=_iv_band(iv), ks=ks, auc=auc,
        strength=abs(auc - 0.5) * 2, n=n, missing_rate=missing)


def evaluate_features(
    df: Any,
    features: list[str],
    target: str,
    *,
    n_bins: int = 10,
    corr_threshold: float = 0.9,
) -> FeatureEvalReport:
    """Evaluate + rank features by IV, then prune correlated redundancies.

    Features are ranked by IV (desc). Correlation pruning keeps the
    higher-IV feature from each pair whose absolute Pearson correlation
    exceeds ``corr_threshold`` (numeric features only).
    """
    import pandas as pd

    evals: list[FeatureEval] = []
    skipped: list[str] = []
    for f in features:
        if f == target:
            continue
        ev = evaluate_feature(df, f, target, n_bins=n_bins)
        if ev is None:
            skipped.append(f)
        else:
            evals.append(ev)
    evals.sort(key=lambda e: -abs(e.iv))

    # correlation pruning among numeric survivors, strongest-IV first
    pruned: list[str] = []
    kept: list[str] = []
    numeric = [e.feature for e in evals
               if pd.api.types.is_numeric_dtype(df[e.feature])]
    for f in [e.feature for e in evals]:
        if f in pruned:
            continue
        kept.append(f)
        if f not in numeric:
            continue
        for g in numeric:
            if g == f or g in pruned or g in kept:
                continue
            try:
                c = df[[f, g]].dropna().corr().iloc[0, 1]
            except Exception:  # noqa: BLE001
                continue
            if c is not None and abs(c) >= corr_threshold:
                pruned.append(g)

    ranked = [e for e in evals if e.feature not in pruned]
    return FeatureEvalReport(evaluations=ranked, pruned=pruned, skipped=skipped)
