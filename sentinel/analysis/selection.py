"""
sentinel.analysis.selection
=============================

V3-020: feature selection stack for the modeling core. Stage order per the
roadmap: near-zero-variance filter -> |corr|>0.95 filter -> mRMR -> Boruta
shadow-feature test -> cross-time stability check. Each stage narrows the
candidate feature set; the final report records what was dropped at each
stage and why, plus any feature that is predictive in one time slice but
not another (flagged, not silently dropped).

Disclosed deviation: BorutaShap -> shap_boruta_filter (native tree SHAP,
no `shap`/`numba` package)
-----------------------------------------------------------------------
The roadmap names ``BorutaShap`` specifically. In this environment (and,
per the user, on the actual target deployment server: pip-only, no sudo),
the `shap` PyPI package fails to *import*: its `numba` dependency hits a
blocked native DLL (locally, reported by Windows as an "Application
Control policy" block). Investigating *why* `shap` needs `numba` at all
turned up a fix that avoids the dependency entirely rather than working
around the block: `shap`'s own `explainers/_tree.py` (the TreeExplainer
BorutaShap actually uses) has NO numba reference -- it calls a compiled
Cython extension (`_cext`). `numba` is only pulled in because `shap/
__init__.py` unconditionally imports unrelated modules (`links.py`,
`_partition.py`, etc.) that do `import numba` at their top level. So
`import shap` can't be avoided, but the underlying algorithm (Tree SHAP)
can: LightGBM (`pred_contrib=True`) and XGBoost (`pred_contribs=True`)
compute the SAME Tree SHAP values NATIVELY, with zero extra dependencies
beyond the boosting library already required for V3-021. `shap_boruta_
filter()` below reimplements Boruta's shadow-feature permutation test
using those native SHAP contributions as the importance signal -- this is
MORE faithful to "BorutaShap" than a RandomForest-importance substitute
would be (real SHAP values, not a different importance metric), and it is
guaranteed pip-installable with no native-extension risk, since lightgbm/
xgboost's own compiled wheels are already required and already working.

`boruta_shadow_filter()` (BorutaPy, RandomForest `feature_importances_`)
is kept as a fallback for callers without lightgbm/xgboost available; it
was the first substitution attempted, superseded as the default by
`shap_boruta_filter()`.

Depends on the wheelhouse batch (V3-029): this module imports `mrmr` and
`boruta` at call time (not at module import time), matching the existing
codebase convention of deferring heavy/optional imports so the module can
still be imported (and its pure-Python filters used) in environments
without them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SelectionStageResult:
    stage: str
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectionReport:
    generated_at: str
    stages: list[SelectionStageResult] = field(default_factory=list)
    final_features: list[str] = field(default_factory=list)
    unstable_features: list[str] = field(default_factory=list)
    stability_detail: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# -- Stage 1: near-zero-variance filter --------------------------------------

def near_zero_variance_filter(
    X: pd.DataFrame,
    variance_threshold: float = 1e-5,
    unique_ratio_threshold: float = 0.01,
) -> SelectionStageResult:
    """Drop columns that are (near-)constant: variance at or below
    `variance_threshold`, OR the ratio of distinct values to row count is
    at or below `unique_ratio_threshold` (catches near-constant columns
    whose raw variance is nonzero but dominated by one value)."""
    n = len(X)
    kept, dropped, detail = [], [], {}
    for col in X.columns:
        series = X[col]
        var = float(series.var()) if pd.api.types.is_numeric_dtype(series) else None
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n if n else 0.0
        is_nzv = (var is not None and var <= variance_threshold) or (unique_ratio <= unique_ratio_threshold)
        if is_nzv:
            dropped.append(col)
            detail[col] = {"variance": var, "unique_ratio": unique_ratio}
        else:
            kept.append(col)
    return SelectionStageResult(
        stage="near_zero_variance", kept=kept, dropped=dropped,
        detail={"threshold_variance": variance_threshold,
                "threshold_unique_ratio": unique_ratio_threshold,
                "dropped_detail": detail},
    )


# -- Stage 2: |corr| > threshold filter --------------------------------------

def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> SelectionStageResult:
    """Greedily collapse groups of columns pairwise-correlated above
    `threshold`. Within a correlated pair, the column later in `X.columns`
    order is dropped (deterministic tie-break, no relevance signal used at
    this stage -- relevance-aware pruning is mRMR's job, next stage)."""
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    dropped: list[str] = []
    dropped_because: dict = {}
    if len(numeric_cols) >= 2:
        corr = X[numeric_cols].corr().abs()
        for i, col_i in enumerate(numeric_cols):
            if col_i in dropped:
                continue
            for col_j in numeric_cols[i + 1:]:
                if col_j in dropped:
                    continue
                c = corr.loc[col_i, col_j]
                if pd.notna(c) and c > threshold:
                    dropped.append(col_j)
                    dropped_because[col_j] = {"correlated_with": col_i, "abs_corr": float(c)}
    kept = [c for c in X.columns if c not in dropped]
    return SelectionStageResult(
        stage="correlation_filter", kept=kept, dropped=dropped,
        detail={"threshold": threshold, "dropped_detail": dropped_because},
    )


# -- Stage 3: mRMR ------------------------------------------------------------

def mrmr_select(X: pd.DataFrame, y: pd.Series, k: int) -> SelectionStageResult:
    """Minimum-Redundancy-Maximum-Relevance selection via the `mrmr`
    package. `k` is capped to the number of available columns."""
    from mrmr import mrmr_classif

    k = min(k, len(X.columns))
    if k <= 0:
        return SelectionStageResult(stage="mrmr", kept=[], dropped=list(X.columns),
                                     detail={"k": k})
    selected = mrmr_classif(X=X, y=y, K=k, show_progress=False)
    dropped = [c for c in X.columns if c not in selected]
    return SelectionStageResult(stage="mrmr", kept=list(selected), dropped=dropped, detail={"k": k})


# -- Stage 4: Boruta shadow-feature filter ------------------------------------

def boruta_shadow_filter(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_iter: int = 50,
    n_estimators: int = 200,
) -> SelectionStageResult:
    """A feature must beat a randomized shadow copy of itself (Boruta's
    permutation test) to survive. See module docstring: uses BorutaPy with
    a RandomForest's feature_importances_ rather than SHAP values (the
    roadmap's literal BorutaShap is blocked in this environment)."""
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    if X.empty or len(X.columns) == 0:
        return SelectionStageResult(stage="boruta_shadow", kept=[], dropped=[], detail={})

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                 max_depth=5, random_state=random_state)
    selector = BorutaPy(rf, n_estimators="auto", max_iter=max_iter,
                         random_state=random_state, verbose=0)
    selector.fit(X.values, y.values)

    confirmed = [c for c, s in zip(X.columns, selector.support_) if s]
    tentative = [c for c, s in zip(X.columns, selector.support_weak_) if s]
    rejected = [c for c in X.columns if c not in confirmed and c not in tentative]

    return SelectionStageResult(
        stage="boruta_shadow", kept=confirmed, dropped=rejected,
        detail={"tentative": tentative, "implementation": "boruta.BorutaPy (RandomForest importances)",
                "note": "roadmap names BorutaShap; substituted -- see module docstring"},
    )


def _native_tree_shap_importance(X: pd.DataFrame, y: pd.Series, backend: str,
                                  random_state: int, n_estimators: int) -> np.ndarray:
    """Train a small booster and return mean(|SHAP contribution|) per
    column of X, using the boosting library's OWN Tree SHAP implementation
    (LightGBM `pred_contrib`, XGBoost `pred_contribs`) -- the same
    algorithm `shap.TreeExplainer` uses, no `shap`/`numba` import."""
    if backend == "lightgbm":
        import lightgbm as lgb

        params = {"objective": "binary", "is_unbalance": True, "num_threads": 1,
                  "deterministic": True, "seed": random_state, "verbosity": -1,
                  "max_depth": 3, "num_leaves": 7, "min_child_samples": 20}
        booster = lgb.train(params, lgb.Dataset(X, label=y, free_raw_data=False),
                            num_boost_round=n_estimators)
        contrib = booster.predict(X, pred_contrib=True)  # (n_samples, n_features + 1)
        return np.abs(contrib[:, :-1]).mean(axis=0)

    if backend == "xgboost":
        import xgboost as xgb

        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        params = {"objective": "binary:logistic", "scale_pos_weight": n_neg / n_pos,
                  "nthread": 1, "seed": random_state, "tree_method": "hist",
                  "max_depth": 3, "min_child_weight": 5}
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
        contrib = booster.predict(dtrain, pred_contribs=True)  # (n_samples, n_features + 1)
        return np.abs(contrib[:, :-1]).mean(axis=0)

    raise ValueError(f"Unknown backend '{backend}'. Choose 'lightgbm' or 'xgboost'.")


def shap_boruta_filter(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_iterations: int = 20,
    backend: str = "lightgbm",
    n_estimators: int = 30,
    alpha: float = 0.05,   # V5-001a: ACCEPTED BUT UNUSED -- see below
) -> SelectionStageResult:
    """A feature must beat a randomized shadow copy of itself, using REAL
    Tree SHAP values (see module docstring for why this needs neither
    `shap` nor `numba`) as the importance signal -- the roadmap's literal
    BorutaShap, minus the blocked dependency.

    Each iteration: shuffle a fresh shadow copy of every column, train a
    small booster on [real | shadow], take mean(|SHAP|) per column, and
    record a "hit" for each real feature whose importance exceeds the max
    shadow importance that iteration. After `n_iterations`, a two-sided
    binomial test (p=0.5, matching Boruta's own statistical test) on each
    feature's hit count decides CONFIRMED / REJECTED / TENTATIVE.

    `n_estimators` defaults deliberately low (30) and the booster is kept
    shallow (max_depth=3). Found empirically: with more capacity (100
    rounds, depth 4), a pure-noise column can still get picked up by ~half
    the iterations purely from overfitting to sample-specific artifacts in
    that particular noise draw (the real data doesn't change between
    iterations -- only the shadow columns are re-shuffled -- so a model
    with enough capacity to memorize a spurious pattern in one noise column
    will do so consistently). Shallower trees make the shadow test
    meaningful; this is also why standard Boruta implementations use
    shallow/regularized trees rather than a full-capacity model.
    """

    if X.empty or len(X.columns) == 0:
        return SelectionStageResult(stage="shap_boruta", kept=[], dropped=[], detail={})

    rng = np.random.default_rng(random_state)
    columns = list(X.columns)
    hit_counts = {c: 0 for c in columns}

    for i in range(n_iterations):
        shadow = X.apply(lambda col: rng.permutation(col.values))
        shadow.columns = [f"shadow__{c}" for c in columns]
        combined = pd.concat([X.reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)

        importance = _native_tree_shap_importance(
            combined, y.reset_index(drop=True), backend, random_state + i, n_estimators,
        )
        real_importance = dict(zip(columns, importance[: len(columns)]))
        shadow_importance = importance[len(columns):]
        shadow_max = float(shadow_importance.max()) if len(shadow_importance) else 0.0

        for c in columns:
            if real_importance[c] > shadow_max:
                hit_counts[c] += 1

    # V5-001a. What used to be here: binomtest(hits, n_iterations, p=0.5), and
    # the decision gated on its p-value.
    #
    # That null is wrong, and measuring it is what showed the shape of the
    # problem. A feature scores a hit by beating `shadow_max` -- the maximum
    # over EVERY shadow column that iteration, not its own shadow. Beating a
    # max-of-N gets harder as N grows, so the chance hit-rate is nowhere near
    # 0.5; measured on this implementation it runs 0.517 at 3 candidates,
    # 0.073 at 20, and 0.0004 at 120. With the null that far off, every
    # feature tests "significant" at any alpha (122 of 122 in a 120-noise
    # screen), the p-value carries no information, and the hit-majority split
    # below was silently making 100% of the decisions anyway.
    #
    # The decision rule is therefore stated directly instead of being dressed
    # in a test that does not hold. Behaviour is unchanged -- this is the rule
    # that was already operating -- but the result no longer reports a
    # significance claim it cannot support.
    #
    # Note also what max-shadow already IS: a max-statistic comparison against
    # the whole shadow family, i.e. multiplicity control built into Boruta's
    # design rather than bolted on afterwards. That is why the chance hit-rate
    # falls as the candidate count rises. The screen was never unprotected;
    # it was mis-described.
    confirm_at = n_iterations / 2

    # Reported, not gated: under exchangeability a pure-noise feature is the
    # largest of itself plus N shadows with probability 1/(N+1). Carried so a
    # reviewer can see the margin between a kept feature's hit-rate and chance
    # without re-deriving it. The observed rate runs below this (the boosters
    # concentrate importance on real signal), so it is a conservative bound.
    null_hit_rate = 1.0 / (len(columns) + 1)

    confirmed, rejected, tentative = [], [], []
    per_feature_detail = {}
    for c in columns:
        hits = hit_counts[c]
        per_feature_detail[c] = {
            "hits": hits,
            "n_iterations": n_iterations,
            "hit_rate": hits / n_iterations if n_iterations else 0.0,
        }
        if hits > confirm_at:
            confirmed.append(c)
        elif hits < confirm_at:
            rejected.append(c)
        else:
            tentative.append(c)   # exactly half -- genuinely undecided

    return SelectionStageResult(
        stage="shap_boruta", kept=confirmed, dropped=rejected,
        detail={"tentative": tentative, "per_feature": per_feature_detail,
                "implementation": f"native {backend} Tree SHAP (pred_contrib), no shap/numba",
                "n_iterations": n_iterations,
                # Stated so a certificate can quote the rule verbatim rather
                # than a reader inferring it from the code.
                "decision_rule": (
                    f"CONFIRMED if the feature beat the maximum of all "
                    f"{len(columns)} shadow importances in more than "
                    f"{confirm_at:g} of {n_iterations} independent trials; "
                    f"REJECTED if fewer; TENTATIVE if exactly."),
                "null_hit_rate": null_hit_rate,
                "multiplicity": (
                    "implicit: each trial compares against the MAX over the "
                    "whole shadow family, which is a max-statistic control "
                    "over all candidates. No separate p-value correction is "
                    "applied because no p-value is computed."),
                "significance_test": "none (see V5-001a in docs/v5.0/next_steps_v5.md)"},
    )


# -- Stage 5: cross-time stability --------------------------------------------

def cross_time_stability(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: list[str],
    label_col: str,
    n_folds: int = 3,
    k: Optional[int] = None,
) -> tuple[list[str], list[str], dict]:
    """Re-run mRMR selection independently on `n_folds` time-ordered slices
    of the data. A feature selected in every fold is stable; a feature
    selected in some folds but not others is flagged unstable (returned,
    not silently dropped from the caller's final feature set -- the caller
    decides whether to keep a stability-flagged feature).

    Returns (stable_features, unstable_features, per_fold_selected).
    """
    from mrmr import mrmr_classif

    times = pd.to_datetime(df[time_col])
    order = times.sort_values().index
    fold_edges = np.array_split(order, n_folds)

    k = k or min(10, len(feature_cols))
    per_fold_selected: dict[str, list[str]] = {}
    for i, fold_idx in enumerate(fold_edges):
        fold_df = df.loc[fold_idx]
        X_fold = fold_df[feature_cols]
        y_fold = fold_df[label_col]
        if y_fold.nunique() < 2 or len(fold_df) < 2:
            per_fold_selected[f"fold_{i}"] = []
            continue
        fold_k = min(k, len(feature_cols))
        selected = mrmr_classif(X=X_fold, y=y_fold, K=fold_k, show_progress=False)
        per_fold_selected[f"fold_{i}"] = list(selected)

    all_selected_sets = [set(v) for v in per_fold_selected.values() if v]
    if not all_selected_sets:
        return [], list(feature_cols), per_fold_selected

    ever_selected = set.union(*all_selected_sets)
    stable = sorted(f for f in ever_selected if all(f in s for s in all_selected_sets))
    unstable = sorted(f for f in ever_selected if f not in stable)

    return stable, unstable, per_fold_selected


# -- Orchestrator --------------------------------------------------------------

class FeatureSelector:
    """Runs the full V3-020 stage pipeline and returns a SelectionReport.

    Parameters
    ----------
    nzv_variance_threshold, nzv_unique_ratio_threshold : near-zero-variance
        thresholds, see `near_zero_variance_filter`.
    corr_threshold      : |corr| above which one of a pair is dropped.
    mrmr_k              : number of features mRMR should retain.
    use_boruta          : whether to run the Boruta shadow-feature stage.
    boruta_backend      : "shap" (default) -- `shap_boruta_filter`, native
                          Tree SHAP via lightgbm/xgboost, no `shap`/`numba`
                          package required (see module docstring); or
                          "sklearn_rf" -- `boruta_shadow_filter`
                          (BorutaPy + RandomForest importances), kept as a
                          fallback for callers without a boosting library.
    boruta_max_iter     : max iterations for the "sklearn_rf" backend.
    boruta_n_iterations : shadow-permutation rounds for the "shap" backend.
    random_state        : seed for the Boruta stage.
    """

    def __init__(
        self,
        nzv_variance_threshold: float = 1e-5,
        nzv_unique_ratio_threshold: float = 0.01,
        corr_threshold: float = 0.95,
        mrmr_k: int = 20,
        use_boruta: bool = True,
        boruta_backend: str = "shap",
        boruta_max_iter: int = 50,
        boruta_n_iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.nzv_variance_threshold = nzv_variance_threshold
        self.nzv_unique_ratio_threshold = nzv_unique_ratio_threshold
        self.corr_threshold = corr_threshold
        self.mrmr_k = mrmr_k
        self.use_boruta = use_boruta
        self.boruta_backend = boruta_backend
        self.boruta_max_iter = boruta_max_iter
        self.boruta_n_iterations = boruta_n_iterations
        self.random_state = random_state

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: Optional[str] = None,
        df_full: Optional[pd.DataFrame] = None,
        n_stability_folds: int = 3,
    ) -> SelectionReport:
        stages: list[SelectionStageResult] = []

        nzv = near_zero_variance_filter(X, self.nzv_variance_threshold, self.nzv_unique_ratio_threshold)
        stages.append(nzv)
        X1 = X[nzv.kept]

        corr = correlation_filter(X1, self.corr_threshold)
        stages.append(corr)
        X2 = X1[corr.kept]

        mrmr_res = mrmr_select(X2, y, self.mrmr_k)
        stages.append(mrmr_res)
        final_features = list(mrmr_res.kept)

        if self.use_boruta and final_features:
            X3 = X2[final_features]
            if self.boruta_backend == "shap":
                boruta_res = shap_boruta_filter(X3, y, self.random_state, self.boruta_n_iterations)
            elif self.boruta_backend == "sklearn_rf":
                boruta_res = boruta_shadow_filter(X3, y, self.random_state, self.boruta_max_iter)
            else:
                raise ValueError(f"Unknown boruta_backend '{self.boruta_backend}'. "
                                 f"Choose 'shap' or 'sklearn_rf'.")
            stages.append(boruta_res)
            if boruta_res.kept:
                final_features = boruta_res.kept

        unstable_features: list[str] = []
        stability_detail: dict = {}
        if time_col is not None and df_full is not None and final_features:
            label_col = y.name or "label"
            df_for_stability = df_full.copy()
            if label_col not in df_for_stability.columns:
                df_for_stability[label_col] = y.values
            # df_full is caller-provided (typically just carries time_col);
            # the surviving feature columns live in X, not df_full -- both
            # share X's index, so align them in before slicing into folds.
            for col in final_features:
                if col not in df_for_stability.columns:
                    df_for_stability[col] = X[col].reindex(df_for_stability.index).values
            _, unstable_features, stability_detail = cross_time_stability(
                df_for_stability, time_col, final_features, label_col, n_folds=n_stability_folds,
            )

        return SelectionReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            stages=stages,
            final_features=final_features,
            unstable_features=unstable_features,
            stability_detail=stability_detail,
        )
