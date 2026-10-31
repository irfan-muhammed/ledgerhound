"""
sentinel.sampling.hard_negatives
==================================

HardNegativeMiner -- identifies non-fraud samples that are closest to
the fraud cluster. These are the cases where models fail, where random
sampling completely misses, and where training signal is most valuable.

Three strategies (all run on local H100/cuDF or CPU):

1. ISOLATION_FOREST
   Train an Isolation Forest on fraud features. Score non-fraud samples
   using the same model. Non-fraud with the highest anomaly score
   (most "fraud-like") are the hard negatives.

2. CENTROID_DISTANCE
   Compute the centroid of the fraud cluster. Rank non-fraud samples by
   distance to this centroid. Closest = hardest negatives.
   Fast and interpretable. Works well when fraud forms a tight cluster.

3. LINEAR_BOUNDARY
   Train a logistic regression on combined fraud + random non-fraud.
   Re-score all non-fraud by predicted fraud probability. Highest
   probability = closest to the decision boundary = hard negatives.
   Most theoretically sound but requires a reasonably balanced seed sample.

All strategies return the same output: a scored non-fraud DataFrame
with a `hard_negative_score` column (higher = more fraud-like) and a
boolean `is_hard_negative` column marking the top-K selection.

Design notes
------------
- Works on pandas DataFrames. Pass cuDF DataFrames by converting to pandas
  first: df.to_pandas(). The scoring is not the bottleneck.
- Features should be pre-cleaned (no nulls, no string columns).
  Categorical features must be encoded before passing.
- The Isolation Forest is trained ONLY on fraud data -- not on non-fraud.
  This is intentional: we want to find non-fraud that looks like fraud,
  not non-fraud that is generally anomalous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class HardNegativeStrategy(str, Enum):
    ISOLATION_FOREST  = "isolation_forest"
    CENTROID_DISTANCE = "centroid_distance"
    LINEAR_BOUNDARY   = "linear_boundary"


@dataclass
class HardNegativeResult:
    """Result of hard negative mining."""
    strategy: str
    n_fraud: int
    n_nonfr_input: int
    n_hard_negatives: int
    inclusion_rate: float         # hard_negatives / n_nonfr_input
    top_k: int
    score_threshold: float        # score cutoff used to select top_k
    feature_importances: dict[str, float] = field(default_factory=dict)
    boundary_density: Optional[float] = None  # % of non-fraud within 1 of fraud
    notes: str = ""

    def summary(self) -> str:
        return (
            f"HardNegativeMiner [{self.strategy}]\n"
            f"  Fraud rows          : {self.n_fraud:,}\n"
            f"  Non-fraud input     : {self.n_nonfr_input:,}\n"
            f"  Hard negatives      : {self.n_hard_negatives:,} ({self.inclusion_rate:.2%})\n"
            f"  Score threshold     : {self.score_threshold:.4f}\n"
            + (f"  Boundary density    : {self.boundary_density:.2%}\n"
               if self.boundary_density is not None else "")
        )


class HardNegativeMiner:
    """
    Mine hard negative non-fraud samples from a large non-fraud DataFrame.

    Parameters
    ----------
    strategy  : Which scoring algorithm to use.
    top_k     : Number of hard negatives to return. If None, uses top_k_ratio.
    top_k_ratio: Return this fraction of the non-fraud input as hard negatives.
                 Ignored if top_k is set.
    random_state: For reproducibility.
    """

    def __init__(
        self,
        strategy: HardNegativeStrategy = HardNegativeStrategy.ISOLATION_FOREST,
        top_k: Optional[int] = None,
        top_k_ratio: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.top_k = top_k
        self.top_k_ratio = top_k_ratio
        self.random_state = random_state

    def mine(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
    ) -> tuple[pd.DataFrame, HardNegativeResult]:
        """
        Score non-fraud samples and return the top-K hard negatives.

        Parameters
        ----------
        fraud_df  : Confirmed fraud DataFrame (all rows used for fitting).
        nonfr_df  : Non-fraud DataFrame to score (can be large sample from Trino).
        features  : Numeric feature columns to use for scoring.
                    Must exist in both DataFrames. Must be non-null.

        Returns
        -------
        hard_neg_df : Non-fraud rows that are hard negatives, with added columns:
                      `hard_negative_score` (float, higher = more fraud-like)
                      `is_hard_negative` (True for all returned rows)
        result      : HardNegativeResult with metrics and metadata.
        """
        fraud_X = _prep(fraud_df, features)
        nonfr_X = _prep(nonfr_df, features)

        if len(fraud_X) < 10:
            raise ValueError(f"Need at least 10 fraud samples. Got {len(fraud_X)}.")
        if len(nonfr_X) == 0:
            raise ValueError("Non-fraud DataFrame is empty.")

        log.info(
            f"Mining hard negatives: {len(fraud_X):,} fraud, "
            f"{len(nonfr_X):,} non-fraud, strategy={self.strategy.value}"
        )

        if self.strategy == HardNegativeStrategy.ISOLATION_FOREST:
            scores, importances = self._isolation_forest(fraud_X, nonfr_X)
        elif self.strategy == HardNegativeStrategy.CENTROID_DISTANCE:
            scores, importances = self._centroid_distance(fraud_X, nonfr_X)
        elif self.strategy == HardNegativeStrategy.LINEAR_BOUNDARY:
            scores, importances = self._linear_boundary(fraud_X, nonfr_X)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Compute k
        k = self.top_k or max(1, int(len(nonfr_X) * self.top_k_ratio))
        k = min(k, len(nonfr_X))

        # Rank and select -- use argsort to guarantee exactly top_k rows
        top_indices = np.argsort(scores)[::-1][:k]
        threshold = float(scores[top_indices[-1]])

        hard_neg_df = nonfr_df.iloc[top_indices].copy()
        hard_neg_df["hard_negative_score"] = scores[top_indices]
        hard_neg_df["is_hard_negative"] = True

        # Boundary density: % of ALL non-fraud within 1 stddev of top-k mean score
        hn_scores = scores[top_indices]
        hn_mean = float(np.mean(hn_scores))
        hn_std  = float(np.std(hn_scores)) or 1.0
        boundary_density = float(np.mean(
            (scores >= hn_mean - hn_std) & (scores <= hn_mean + hn_std)
        ))

        result = HardNegativeResult(
            strategy=self.strategy.value,
            n_fraud=len(fraud_X),
            n_nonfr_input=len(nonfr_X),
            n_hard_negatives=len(top_indices),
            inclusion_rate=len(top_indices) / len(nonfr_X),
            top_k=k,
            score_threshold=threshold,
            feature_importances={f: float(v) for f, v in zip(features, importances)},
            boundary_density=boundary_density,
        )

        log.info(result.summary())
        return hard_neg_df, result

    # -- Scoring strategies -------------------------------------------------

    def _isolation_forest(
        self, fraud_X: np.ndarray, nonfr_X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.ensemble import IsolationForest

        # Train ONLY on fraud -- score non-fraud relative to fraud manifold
        clf = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(fraud_X)

        # decision_function: higher = more like the fitted distribution (fraud)
        scores = clf.decision_function(nonfr_X)
        # Flip sign so "higher = more fraud-like"
        scores = -scores

        # Feature importances: mean |feature contribution| across trees
        # Approximated by feature variance in fraud vs non-fraud
        importances = _variance_importance(fraud_X, nonfr_X)
        return scores, importances

    def _centroid_distance(
        self, fraud_X: np.ndarray, nonfr_X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        fraud_scaled = scaler.fit_transform(fraud_X)
        nonfr_scaled = scaler.transform(nonfr_X)

        centroid = fraud_scaled.mean(axis=0)
        # Euclidean distance to fraud centroid; negate so higher = closer
        distances = np.linalg.norm(nonfr_scaled - centroid, axis=1)
        scores = -distances

        # Importance: how much each feature contributes to centroid distance
        importances = np.abs(centroid) / (np.abs(centroid).sum() + 1e-8)
        return scores, importances

    def _linear_boundary(
        self, fraud_X: np.ndarray, nonfr_X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Build balanced training set: all fraud + random sample of non-fraud
        n_train_nonfr = min(len(nonfr_X), len(fraud_X) * 10)
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(nonfr_X), n_train_nonfr, replace=False)

        X_train = np.vstack([fraud_X, nonfr_X[idx]])
        y_train = np.array([1] * len(fraud_X) + [0] * n_train_nonfr)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        nonfr_scaled = scaler.transform(nonfr_X)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=self.random_state,
        )
        clf.fit(X_train_s, y_train)

        scores = clf.predict_proba(nonfr_scaled)[:, 1]  # P(fraud)
        importances = np.abs(clf.coef_[0]) / (np.abs(clf.coef_[0]).sum() + 1e-8)
        return scores, importances


# -- Helpers ----------------------------------------------------------------


def _prep(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Extract and clean feature matrix. Fills nulls with median."""
    X = df[features].copy()
    for col in features:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X.astype(float).values


def _variance_importance(fraud_X: np.ndarray, nonfr_X: np.ndarray) -> np.ndarray:
    """Feature importance proxy: normalized absolute mean difference."""
    diff = np.abs(fraud_X.mean(axis=0) - nonfr_X.mean(axis=0))
    std  = nonfr_X.std(axis=0) + 1e-8
    imp  = diff / std
    return imp / (imp.sum() + 1e-8)
