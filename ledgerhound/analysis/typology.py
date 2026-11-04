"""
ledgerhound.analysis.typology
===========================

TypologyAnalyzer -- discovers distinct fraud types within the confirmed
fraud sample by clustering on feature vectors.

Why this matters
----------------
The confirmed fraud set is NOT a single homogeneous population. SIM-swap
fraud, social engineering fraud, account takeover, and fake merchant fraud
each have different feature signatures. A single model trained on all
mixed typologies is trying to learn multiple decision boundaries at once --
it will underperform on all of them.

The typology analysis answers:
  - How many distinct fraud types exist in this dataset?
  - What are the distinguishing features of each type?
  - Are any types growing or shrinking over time?
  - Does the current non-fraud sample cover all typologies' hard-negative
    regions, or only some?

Output
------
For each cluster (typology), the analyzer produces:
  - Feature means and their deviation from the overall fraud mean
  - Top-K distinguishing features (most different from other clusters)
  - Cluster size and temporal distribution
  - A human-readable label (auto-generated from distinguishing features)

This feeds directly into:
  - Sampling strategy: each typology may need its own hard-negative pool
  - Feature signal decay: track signal per typology, not just overall
  - FE team guidance: which features matter for which fraud type

Clustering methods
------------------
  KMEANS   : Fast, stable, interpretable. Use when n_clusters is known.
  DBSCAN   : Density-based -- discovers number of clusters automatically.
             Good when fraud types form tight clusters of varying size.
  GMIX     : Gaussian Mixture Model -- soft assignments, handles
             overlapping typologies. More realistic for fraud data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()


class ClusteringMethod(str, Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMIX   = "gmix"


@dataclass
class FraudTypology:
    """One discovered fraud typology (cluster)."""

    cluster_id: int
    n_cases: int
    pct_of_fraud: float                    # this cluster / total fraud
    feature_means: dict[str, float]        # mean per feature in this cluster
    feature_zscores: dict[str, float]      # z-score vs overall fraud mean
    top_features: list[str]                # top-K most distinguishing features
    label: str = ""                        # auto-generated or analyst-supplied
    temporal_distribution: dict[str, int] = field(default_factory=dict)  # {YYYY-MM: count}
    is_growing: Optional[bool] = None      # True if cluster is growing over time
    hard_negative_affinity: Optional[float] = None  # 0-1 how well sampled

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"Typology {self.cluster_id}: {self.label or 'unlabeled'} "
            f"({self.n_cases:,} cases, {self.pct_of_fraud:.1%} of fraud)",
            f"  Top features: {', '.join(self.top_features[:5])}",
        ]
        for f in self.top_features[:5]:
            z = self.feature_zscores.get(f, 0.0)
            mu = self.feature_means.get(f, 0.0)
            lines.append(f"    {f:<40} mean={mu:.3f}  z={z:+.2f}")
        return "\n".join(lines)


@dataclass
class TypologyReport:
    """Complete typology analysis result."""

    generated_at: str
    method: str
    n_clusters: int
    n_fraud_cases: int
    features_used: list[str]
    typologies: list[FraudTypology]
    silhouette_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    inertia: Optional[float] = None
    cluster_assignments: Optional[list[int]] = None  # per-row cluster ID
    notes: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "cluster_assignments"}
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def display(self) -> None:
        console.print(f"\n[bold]Fraud Typology Analysis[/bold] -- {self.method.upper()} "
                      f"| {self.n_clusters} clusters | {self.n_fraud_cases:,} fraud cases")
        if self.silhouette_score is not None:
            console.print(f"  Silhouette: {self.silhouette_score:.3f}  "
                          f"Davies-Bouldin: {self.davies_bouldin_score:.3f}")
        console.print()

        t = Table("Cluster", "Label", "Cases", "% Fraud", "Top Features", show_header=True,
                  header_style="bold dim")
        for typ in sorted(self.typologies, key=lambda x: -x.n_cases):
            top = ", ".join(typ.top_features[:3])
            t.add_row(
                str(typ.cluster_id),
                typ.label or "--",
                f"{typ.n_cases:,}",
                f"{typ.pct_of_fraud:.1%}",
                top[:55],
            )
        console.print(t)
        console.print()
        for typ in self.typologies:
            console.print(typ.describe())
            console.print()


class TypologyAnalyzer:
    """
    Cluster confirmed fraud cases to discover distinct fraud typologies.

    Parameters
    ----------
    method         : Clustering algorithm.
    n_clusters     : For KMEANS/GMIX. Use None for DBSCAN (auto-discovers).
    max_clusters   : When n_clusters is None, try range(2, max_clusters+1)
                     and pick the best by silhouette score.
    n_top_features : Number of distinguishing features to report per cluster.
    random_state   : Reproducibility seed.
    """

    def __init__(
        self,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        n_clusters: Optional[int] = None,
        max_clusters: int = 8,
        n_top_features: int = 10,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.n_top_features = n_top_features
        self.random_state = random_state

    def analyze(
        self,
        fraud_df: pd.DataFrame,
        features: list[str],
        date_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> TypologyReport:
        """
        Cluster fraud cases and produce a TypologyReport.

        Parameters
        ----------
        fraud_df  : Confirmed fraud DataFrame (all rows = confirmed fraud).
        features  : Numeric feature columns to cluster on.
        date_col  : Optional date column for temporal distribution per cluster.
        label_col : Optional column with known fraud labels (e.g. fraud type code).
                    If provided, used for cluster labeling validation.
        """
        if len(fraud_df) < 10:
            raise ValueError(f"Need at least 10 fraud cases for typology analysis. Got {len(fraud_df)}.")

        available = [f for f in features if f in fraud_df.columns]
        if not available:
            raise ValueError("None of the provided features exist in fraud_df.")
        if len(available) < len(features):
            missing = set(features) - set(available)
            log.warning(f"Features not found in fraud_df, skipping: {missing}")

        X = _prep(fraud_df, available)
        X_scaled, scaler = _scale(X)

        # Fit clusters
        if self.method == ClusteringMethod.KMEANS:
            labels, k, inertia = self._fit_kmeans(X_scaled)
        elif self.method == ClusteringMethod.DBSCAN:
            labels, k, inertia = self._fit_dbscan(X_scaled)
        else:
            labels, k, inertia = self._fit_gmix(X_scaled)

        # Evaluate quality
        sil, db = _cluster_quality(X_scaled, labels)

        # Build typology objects
        overall_means = X.mean(axis=0)
        overall_stds  = X.std(axis=0) + 1e-8
        typologies = []

        unique_labels = sorted(set(labels[labels >= 0]))
        for cid in unique_labels:
            mask = labels == cid
            cluster_X = X[mask]
            n_cases = int(mask.sum())
            pct = n_cases / len(X)

            cluster_means = cluster_X.mean(axis=0)
            zscores = (cluster_means - overall_means) / overall_stds

            # Top distinguishing features = highest absolute z-score
            top_idx = np.argsort(np.abs(zscores))[::-1][:self.n_top_features]
            top_features = [available[i] for i in top_idx]

            # Auto-label from top feature names + direction
            label = _auto_label(top_features[:3], zscores, available)

            # Temporal distribution
            temp_dist: dict[str, int] = {}
            is_growing = None
            if date_col and date_col in fraud_df.columns:
                temp_dist, is_growing = _temporal_dist(fraud_df[mask], date_col)

            typologies.append(FraudTypology(
                cluster_id=cid,
                n_cases=n_cases,
                pct_of_fraud=round(pct, 4),
                feature_means={f: round(float(v), 4) for f, v in zip(available, cluster_means)},
                feature_zscores={f: round(float(v), 4) for f, v in zip(available, zscores)},
                top_features=top_features,
                label=label,
                temporal_distribution=temp_dist,
                is_growing=is_growing,
            ))

        return TypologyReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            method=self.method.value,
            n_clusters=len(unique_labels),
            n_fraud_cases=len(fraud_df),
            features_used=available,
            typologies=typologies,
            silhouette_score=round(sil, 4) if sil is not None else None,
            davies_bouldin_score=round(db, 4) if db is not None else None,
            inertia=round(inertia, 2) if inertia is not None else None,
            cluster_assignments=labels.tolist(),
        )

    # -- Fitting methods ----------------------------------------------------

    def _fit_kmeans(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, int, Optional[float]]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if self.n_clusters:
            k = self.n_clusters
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            return labels, k, float(km.inertia_)

        # Auto-select k by silhouette
        best_k, best_sil, best_labels, best_inertia = 2, -1.0, None, None
        for k in range(2, min(self.max_clusters + 1, len(X))):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            lbl = km.fit_predict(X)
            if len(set(lbl)) < 2:
                continue
            sil = silhouette_score(X, lbl, sample_size=min(5000, len(X)))
            if sil > best_sil:
                best_k, best_sil, best_labels = k, sil, lbl
                best_inertia = float(km.inertia_)
        return best_labels, best_k, best_inertia

    def _fit_dbscan(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, int, None]:
        from sklearn.cluster import DBSCAN
        eps = _estimate_eps(X)
        db = DBSCAN(eps=eps, min_samples=max(5, len(X) // 100), n_jobs=-1)
        labels = db.fit_predict(X)
        k = len(set(labels[labels >= 0]))
        log.info(f"DBSCAN: eps={eps:.3f}, {k} clusters, "
                 f"{(labels == -1).sum()} noise points")
        return labels, k, None

    def _fit_gmix(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, int, None]:
        from sklearn.mixture import GaussianMixture

        if self.n_clusters:
            gm = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                n_init=3,
            )
            labels = gm.fit_predict(X)
            return labels, self.n_clusters, None

        # Auto-select by BIC
        best_k, best_bic, best_labels = 2, np.inf, None
        for k in range(2, min(self.max_clusters + 1, len(X))):
            gm = GaussianMixture(n_components=k, random_state=self.random_state)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_labels = gm.predict(X)
        return best_labels, best_k, None


# -- Helpers ----------------------------------------------------------------


def _prep(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    for col in features:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X.astype(float).values


def _scale(X: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def _cluster_quality(
    X: np.ndarray, labels: np.ndarray
) -> tuple[Optional[float], Optional[float]]:
    unique = set(labels[labels >= 0])
    if len(unique) < 2:
        return None, None
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        mask = labels >= 0
        sil = silhouette_score(X[mask], labels[mask], sample_size=min(5000, mask.sum()))
        db  = davies_bouldin_score(X[mask], labels[mask])
        return sil, db
    except Exception:
        return None, None


def _estimate_eps(X: np.ndarray, k: int = 4) -> float:
    """Estimate DBSCAN eps using k-nearest-neighbor distances."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    knn_dists = np.sort(dists[:, -1])
    # Elbow heuristic: use 90th percentile of k-NN distances
    return float(np.percentile(knn_dists, 90))


def _auto_label(
    top_features: list[str],
    zscores: np.ndarray,
    all_features: list[str],
) -> str:
    """Generate a descriptive label from top distinguishing features."""
    parts = []
    for feat in top_features[:2]:
        if feat in all_features:
            idx = all_features.index(feat)
            direction = "high" if zscores[idx] > 0 else "low"
            short_name = feat.replace("_", " ").split()[-1]  # last word
            parts.append(f"{direction}-{short_name}")
    return " / ".join(parts) if parts else "cluster"


def _temporal_dist(
    cluster_df: pd.DataFrame, date_col: str
) -> tuple[dict[str, int], Optional[bool]]:
    """Monthly count and growth trend."""
    try:
        dates = pd.to_datetime(cluster_df[date_col].dropna())
        monthly = dates.dt.to_period("M").value_counts().sort_index()
        dist = {str(k): int(v) for k, v in monthly.items()}

        # Simple growth: last 3 months vs previous 3 months
        if len(monthly) >= 6:
            last3 = monthly.iloc[-3:].sum()
            prev3 = monthly.iloc[-6:-3].sum()
            is_growing = bool(last3 > prev3)
        else:
            is_growing = None

        return dist, is_growing
    except Exception:
        return {}, None
