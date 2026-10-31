"""
sentinel.modeling.gbm
=======================

V3-021: the boring GBM baseline that wins. This is the integration
keystone of Epic 5, not a standalone trainer -- `train_gbm_baseline()` is
the first real caller of three things three prior items built and
deliberately deferred integrating:

  - V3-019's `require_temporal_split()` typed guard -- called FIRST, before
    any data is touched. Handing this function anything other than a real
    `TemporalSplit` (a `train_test_split` tuple, a K-fold generator) raises
    `NotATemporalSplitError` here, for real, not just in a toy test.
  - V3-018's `LeakageSentinel` -- run at TWO points, matching the two
    failure modes the sentinel's own docstring distinguishes:
      1. PRE-FIT: point-in-time + lexicon checks on the declared
         `feature_specs`. A VETOED feature refuses to train -- this run
         genuinely does NOT produce a model, closing V3-018's deferred
         "pipeline blocks... does NOT produce a model" integration claim.
      2. POST-FIT: the dev-vs-OOT AUROC gap check, which can only run once
         real scores exist. A blocked gap does not un-train the model (the
         artifact already exists), but marks the result `leakage_blocked`
         so it must not be recorded as a knowledge-base "win" -- this is
         the literal mechanism behind the epic's "leaky wins poison the KB"
         warning.
  - V3-020's `FeatureSelector` output -- accepted as `feature_cols` input
    (light coupling: this module does not re-run selection itself).

Design choices, each disclosed rather than silently made
-----------------------------------------------------------
* **Class weights, NOT SMOTE** -- per the roadmap's explicit instruction.
  `scale_pos_weight` (XGBoost) / `is_unbalance` (LightGBM) reweight the
  existing minority class rather than synthesizing new minority rows,
  which can manufacture near-duplicate synthetic fraud patterns a temporal
  split can't meaningfully separate into train vs OOT.
* **Threshold calibration** sweeps candidate thresholds against OOT scores
  and picks the one minimizing `fp_cost * FP + fn_cost * FN` from the
  supplied cost matrix (roadmap: "the L0 cost matrix" -- V3-023, which
  would author this value, is unbuilt; accepted as a parameter here with a
  documented default of `{"fp_cost": 1.0, "fn_cost": 25.0}`, a directional
  placeholder reflecting that missing real fraud is normally far costlier
  than a false alarm, NOT a calibrated real business figure).
* **Staged Optuna tuning**: four sequential studies (structure -> sampling
  -> regularization -> lr+estimators), each freezing the previous stage's
  best params before the next runs, using a seeded `TPESampler` for
  reproducibility. Tuned ONLY against a validation slice carved from the
  time-ordered TRAIN side (never OOT -- tuning against OOT would leak model
  *selection* information into the "held out" evaluation).
  **ASHA vs SuccessiveHalvingPruner**: the roadmap says ASHA
  (Asynchronous Successive Halving, for distributed/parallel trials).
  This module runs single-worker/sequential, so it uses Optuna's
  `SuccessiveHalvingPruner` -- the synchronous single-worker analog of the
  same early-stopping idea, not the distributed-async variant. Disclosed
  substitution, same shape as V3-020's BorutaShap->boruta swap.
* **Determinism for `sentinel replay`**: single-threaded
  (`num_threads=1`/`nthread=1`) and seeded. Multi-threaded histogram
  building in LightGBM/XGBoost drifts low-order bits across runs; a
  training run that isn't bit-identical on re-execution cannot pass
  `sentinel replay`'s output-hash comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from sentinel.validation.temporal_split import TemporalSplit, require_temporal_split

log = logging.getLogger(__name__)

DEFAULT_COST_MATRIX = {"fp_cost": 1.0, "fn_cost": 25.0}


class TrainingRefused(RuntimeError):
    """Raised when leakage vetting blocks training before any model is fit.
    The whole point: this run genuinely produces no model."""


@dataclass
class TrainReport:
    generated_at: str
    backend: str
    params: dict
    random_state: int
    split_id: str
    feature_cols: list[str]
    n_train: int
    n_oot: int
    metrics: dict                  # {"train": {...}, "oot": {...}}
    threshold: Optional[float]
    cost_matrix: dict
    leakage_pre_fit: dict          # LeakageReport.to_dict() summary
    leakage_post_fit: Optional[dict]  # AUROC-gap dict, or None if not run
    leakage_blocked: bool          # True if the post-fit gap check fired
    wall_clock_seconds: float
    tuning: Optional[dict] = None  # staged Optuna results, if tuning ran
    # V3-023: "default" (no real L0 framing intake -- DEFAULT_COST_MATRIX
    # used as a documented placeholder) or "captured" (a real study-start
    # framing supplied the cost matrix). Read by ModelValidator to cite
    # provenance in the model card's Threshold Rationale section.
    framing_source: str = "default"
    findings: list = field(default_factory=list)
    # The fitted booster (lightgbm.Booster / xgboost.Booster) -- kept
    # in-memory so an independent validator (V3-022) can re-score it
    # itself rather than trusting this report's self-reported metrics.
    # Excluded from to_dict()/save() (not JSON-serialisable, and a report
    # artifact shouldn't try to embed a model binary); compare=False since
    # boosters don't support meaningful equality.
    booster: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "booster"}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _split_id(split: TemporalSplit) -> str:
    """Stable short hash of the split's boundaries (not its full index
    lists) -- cheap to embed in an event payload or report."""
    key = f"{split.time_col}|{split.train_end}|{split.embargo_gap_days}|{split.oot_start}"
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


# -- backend abstraction -------------------------------------------------------

def _fit_lightgbm(X_tr, y_tr, params: dict, random_state: int, X_val=None, y_val=None,
                   num_boost_round: int = 200, early_stopping_rounds: Optional[int] = None):
    import lightgbm as lgb

    full_params = {
        "objective": "binary",
        "is_unbalance": True,             # class weighting, not SMOTE
        "num_threads": 1,                 # determinism for sentinel replay
        "deterministic": True,
        "seed": random_state,
        "verbosity": -1,
        **params,
    }
    train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    callbacks = []
    valid_sets = None
    if X_val is not None:
        valid_sets = [lgb.Dataset(X_val, label=y_val, reference=train_set)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(full_params, train_set, num_boost_round=num_boost_round,
                         valid_sets=valid_sets, callbacks=callbacks or None)
    return booster


def _predict_lightgbm(booster, X):
    return booster.predict(X)


def _fit_xgboost(X_tr, y_tr, params: dict, random_state: int, X_val=None, y_val=None,
                  num_boost_round: int = 200, early_stopping_rounds: Optional[int] = None):
    import xgboost as xgb

    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": n_neg / n_pos,   # class weighting, not SMOTE
        "nthread": 1,                        # determinism for sentinel replay
        "seed": random_state,
        "tree_method": "hist",
        **params,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    evals = []
    if X_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dval, "valid")]
    booster = xgb.train(full_params, dtrain, num_boost_round=num_boost_round,
                         evals=evals or None,
                         early_stopping_rounds=early_stopping_rounds if evals else None,
                         verbose_eval=False)
    return booster


def _predict_xgboost(booster, X):
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(X))


_BACKENDS = {
    "lightgbm": (_fit_lightgbm, _predict_lightgbm),
    "xgboost": (_fit_xgboost, _predict_xgboost),
}


# -- threshold calibration ------------------------------------------------------

def calibrate_threshold(y_true, scores, cost_matrix: dict) -> float:
    """Sweep candidate thresholds (unique score values) and return the one
    minimizing fp_cost*FP + fn_cost*FN."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.unique(scores)
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))

    best_threshold, best_cost = 0.5, float("inf")
    for t in candidates:
        pred = (scores >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = cost_matrix["fp_cost"] * fp + cost_matrix["fn_cost"] * fn
        if cost < best_cost:
            best_cost, best_threshold = cost, float(t)
    return best_threshold


# -- staged Optuna tuning --------------------------------------------------------

def _staged_tune(X_tr, y_tr, X_val, y_val, backend: str, random_state: int,
                  n_trials_per_stage: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.metrics import average_precision_score

    fit_fn, predict_fn = _BACKENDS[backend]
    params: dict = {}
    stage_results = {}

    def _objective_factory(space_fn):
        def objective(trial):
            trial_params = {**params, **space_fn(trial)}
            booster = fit_fn(X_tr, y_tr, trial_params, random_state,
                              X_val=X_val, y_val=y_val,
                              num_boost_round=100, early_stopping_rounds=20)
            preds = predict_fn(booster, X_val)
            return average_precision_score(y_val, preds)
        return objective

    stages = {
        "structure": (
            lambda t: {"num_leaves": t.suggest_int("num_leaves", 7, 63),
                       "max_depth": t.suggest_int("max_depth", 3, 8)}
            if backend == "lightgbm" else
            {"max_depth": t.suggest_int("max_depth", 3, 8),
             "min_child_weight": t.suggest_int("min_child_weight", 1, 10)}
        ),
        "sampling": (
            lambda t: {"feature_fraction": t.suggest_float("feature_fraction", 0.6, 1.0),
                       "bagging_fraction": t.suggest_float("bagging_fraction", 0.6, 1.0),
                       "bagging_freq": 1}
            if backend == "lightgbm" else
            {"subsample": t.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0)}
        ),
        "regularization": (
            lambda t: {"lambda_l1": t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                       "lambda_l2": t.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                       "min_child_samples": t.suggest_int("min_child_samples", 5, 50)}
            if backend == "lightgbm" else
            {"reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
             "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
        ),
        "lr_and_estimators": (
            lambda t: {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True)}
        ),
    }

    for stage_name, space_fn in stages.items():
        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(_objective_factory(space_fn), n_trials=n_trials_per_stage,
                       show_progress_bar=False)
        params.update(study.best_params)
        stage_results[stage_name] = {"best_value": study.best_value, "best_params": study.best_params}

    return {"final_params": params, "stages": stage_results}


# -- main entrypoint --------------------------------------------------------------

def train_gbm_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    split,
    feature_specs: Optional[list] = None,
    cost_matrix: Optional[dict] = None,
    framing=None,
    backend: str = "lightgbm",
    params: Optional[dict] = None,
    random_state: int = 42,
    optuna_trials_per_stage: int = 0,
    auroc_gap_threshold: float = 0.05,
    n_estimators: int = 200,
    event_stream=None,
) -> TrainReport:
    """Train a GBM fraud baseline against an authoritative TemporalSplit.

    Refuses to train (raises `TrainingRefused`, produces no model) if a
    pre-fit leakage check vetoes a feature. After fitting, runs the
    post-fit AUROC-gap check and marks the result `leakage_blocked` if it
    fires -- the model artifact still exists (it already trained), but
    must not be recorded as a knowledge-base "win".

    `framing` (V3-023, `sentinel.framing.l0.L0Framing`): when given with
    `source="captured"`, its cost matrix OVERRIDES `cost_matrix` and
    `report.framing_source` is "captured" -- a real study-start intake
    happened. Otherwise (`framing` omitted, or `source="default"`),
    `report.framing_source` is "default" and a warning is appended to
    `report.findings`: the calibrated threshold reflects a documented
    placeholder, not a confirmed business cost tradeoff.
    """
    t0 = time.time()
    split = require_temporal_split(split)   # V3-019 integration: real caller
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(_BACKENDS)}")

    findings: list = []
    if framing is not None and getattr(framing, "source", "default") == "captured":
        cost_matrix = dict(framing.cost_matrix)
        framing_source = "captured"
    else:
        _caller_supplied_matrix = cost_matrix is not None
        cost_matrix = cost_matrix or dict(DEFAULT_COST_MATRIX)
        framing_source = "default"
        if _caller_supplied_matrix:
            findings.append(
                "No L0 framing captured for this study -- a cost matrix "
                f"was supplied directly by the caller ({cost_matrix}), not "
                "via sentinel.framing.l0.L0Framing.capture(), so its "
                "business provenance is unconfirmed. The calibrated "
                "threshold below reflects this unconfirmed input, not a "
                "documented default (see V3-023)."
            )
        else:
            findings.append(
                "No L0 framing captured for this study -- using a documented "
                f"default cost matrix ({cost_matrix}). The calibrated threshold "
                "below reflects this placeholder, not a confirmed business "
                "cost tradeoff (see V3-023)."
            )
    params = dict(params or {})

    # -- PRE-FIT leakage gate (V3-018 integration, real caller) --------------
    from sentinel.validation.leakage import FeatureSpec, LeakageSentinel

    specs = feature_specs or [FeatureSpec(name=c, computed_at="unknown") for c in feature_cols]
    pre_sentinel = LeakageSentinel(auroc_gap_threshold=auroc_gap_threshold)
    pre_report = pre_sentinel.vet(specs)
    if pre_report.overall_blocked:
        if event_stream is not None:
            event_stream.append("model_train_refused",
                                reason="leakage_pre_fit_blocked",
                                vetoed_features=pre_report.vetoed_features())
        raise TrainingRefused(
            f"Training refused: leakage sentinel vetoed {pre_report.vetoed_features()}. "
            f"No model was produced."
        )

    # -- data slices via the authoritative split ------------------------------
    train_df, oot_df = split.apply(df)
    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_oot, y_oot = oot_df[feature_cols], oot_df[label_col]

    tuning_result = None
    if optuna_trials_per_stage > 0:
        # carve a validation slice from the TAIL of train (time-ordered) --
        # tuning never touches OOT.
        tune_cut = int(len(train_df) * 0.8)
        X_tune_tr, X_tune_val = X_train.iloc[:tune_cut], X_train.iloc[tune_cut:]
        y_tune_tr, y_tune_val = y_train.iloc[:tune_cut], y_train.iloc[tune_cut:]
        tuning_result = _staged_tune(X_tune_tr, y_tune_tr, X_tune_val, y_tune_val,
                                      backend, random_state, optuna_trials_per_stage)
        params.update(tuning_result["final_params"])

    fit_fn, predict_fn = _BACKENDS[backend]
    booster = fit_fn(X_train, y_train, params, random_state, num_boost_round=n_estimators)

    scores_train = predict_fn(booster, X_train)
    scores_oot = predict_fn(booster, X_oot)

    from sklearn.metrics import average_precision_score, roc_auc_score

    metrics = {
        "train": {"pr_auc": float(average_precision_score(y_train, scores_train)),
                  "roc_auc": float(roc_auc_score(y_train, scores_train))},
        "oot": {"pr_auc": float(average_precision_score(y_oot, scores_oot)),
               "roc_auc": float(roc_auc_score(y_oot, scores_oot))},
    }

    threshold = calibrate_threshold(y_oot, scores_oot, cost_matrix)

    # -- POST-FIT leakage gate: AUROC gap (V3-018 integration, 2nd call site) -
    post_sentinel = LeakageSentinel(auroc_gap_threshold=auroc_gap_threshold)
    post_report = post_sentinel.vet(
        specs, y_dev=y_train, scores_dev=scores_train, y_oot=y_oot, scores_oot=scores_oot,
    )
    leakage_blocked = bool(post_report.overall_blocked)

    # n_estimators is a separate function argument, not part of `params`
    # (it's not passed to the booster's own params dict -- LightGBM/XGBoost
    # take it as num_boost_round instead). Recorded here for the REPORT
    # only, so Seeds & Versions / the model card can actually reproduce
    # this run; `params` itself (what fit_fn/tuning actually used) is
    # untouched.
    report_params = {**params, "n_estimators": n_estimators}

    report = TrainReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        params=report_params,
        random_state=random_state,
        split_id=_split_id(split),
        feature_cols=list(feature_cols),
        n_train=len(train_df),
        n_oot=len(oot_df),
        metrics=metrics,
        threshold=threshold,
        cost_matrix=cost_matrix,
        leakage_pre_fit={"overall_blocked": pre_report.overall_blocked,
                         "n_vetoed": pre_report.n_vetoed},
        leakage_post_fit=post_report.auroc_gap,
        leakage_blocked=leakage_blocked,
        wall_clock_seconds=time.time() - t0,
        tuning=tuning_result,
        booster=booster,
        framing_source=framing_source,
        findings=findings,
    )

    if event_stream is not None:
        event_stream.append(
            "model_train",
            backend=backend, params=params, random_state=random_state,
            split_id=report.split_id, metrics=metrics, threshold=threshold,
            leakage_blocked=leakage_blocked, wall_clock_seconds=report.wall_clock_seconds,
        )

    if leakage_blocked:
        log.warning("Model trained but POST-FIT leakage gate blocked it (AUROC gap %.3f > %.3f) "
                    "-- do not record this as a knowledge-base win.",
                    post_report.auroc_gap["gap"], auroc_gap_threshold)

    return report
