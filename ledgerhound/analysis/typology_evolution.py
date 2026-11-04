"""
ledgerhound.analysis.typology_evolution
=======================================

TypologyEvolutionTracker -- tracks how fraud cluster sizes and feature
signatures change across time periods.

Why this matters
----------------
Static clustering tells you what fraud looks like today. Evolution tracking
tells you what it's becoming. A cluster growing 40% quarter-over-quarter is
an emerging attack vector -- the model hasn't seen enough of it to learn,
the feature team hasn't engineered for it yet, and the ops team doesn't know
it exists. Ledgerhound is the first to detect it.

Key questions answered
----------------------
  "Which fraud type is accelerating?"   -> growing cluster
  "Which is being neutralised?"         -> collapsing cluster
  "Is there a new attack vector?"       -> cluster present now but not before
  "Is the feature signature shifting?"  -> centroid drift within a cluster

The hardest problem: cluster identity across periods
----------------------------------------------------
KMeans re-labels clusters arbitrarily each run. "Cluster 2" in Q1 may be
completely unrelated to "Cluster 2" in Q2. We solve this with centroid
matching: after each clustering run, we align new clusters to the previous
period's clusters by minimising cosine distance between normalised centroids.
This gives each cluster a stable identity across the entire history.

If the number of clusters changes, or a new cluster has no match within
`match_threshold`, it is declared EMERGING.

Alert thresholds
----------------
Growth rate (quarter-over-quarter % change in cluster size):
  > +25%    WARNING -- cluster is growing faster than fraud overall
  > +50%    FAIL    -- rapid emergence, likely new attack vector

Decline rate:
  < -30%    INFO    -- cluster being neutralised (good news, but track it)

New cluster (no match in previous period):
  Any size  EMERGING -- always flag, analyst must review

Centroid drift (cosine distance between matched centroids):
  > 0.20    WARNING -- feature signature is shifting within this typology

History format (typology_evolution_path)
-----------------------------------------
{
  "periods": ["2024-Q1", "2024-Q2", ...],
  "clusters": {
    "sim_swap": [
      {"period": "2024-Q1", "n_cases": 125, "pct_of_fraud": 0.25,
       "centroid": [...], "top_features": [...], "label": "high-sim / ..."},
      {"period": "2024-Q2", "n_cases": 162, "pct_of_fraud": 0.28, ...}
    ]
  }
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ledgerhound.workspace.layout import WorkspaceLayout
    from ledgerhound.memory.knowledge import DatasetKnowledge

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Alert thresholds ---------------------------------------------------------

_GROWTH_WARN  = 0.25   # +25% cluster size growth -> WARNING
_GROWTH_FAIL  = 0.50   # +50% cluster size growth -> FAIL
_DECLINE_INFO = -0.30  # -30% decline -> INFO (being neutralised)
_DRIFT_WARN   = 0.20   # cosine distance > 0.20 -> centroid drifting
_MATCH_THRESH = 0.60   # cosine similarity < 0.60 -> treat as new cluster


# -- Data classes -------------------------------------------------------------

@dataclass
class ClusterSnapshot:
    """State of one matched cluster at one time period."""
    cluster_key: str          # stable identity (e.g. "cluster_0", "sim_swap")
    period: str
    n_cases: int
    pct_of_fraud: float
    centroid: list[float]     # normalised feature centroid
    top_features: list[str]
    label: str
    original_cluster_id: int  # raw id from the clustering run

    # Ordered feature list the centroid was built from.  Critical for cross-
    # period matching when feature lists differ between runs (e.g. playbook
    # uses 12 features, monitor uses 8 -- the matcher projects onto the
    # intersection and re-normalises).  Defaults to [] for backwards-compat
    # with snapshots written before this field existed.
    features_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = [round(v, 6) for v in d["centroid"]]
        return d


@dataclass
class ClusterTrajectory:
    """Multi-period trajectory for one matched cluster identity."""
    cluster_key: str
    snapshots: list[ClusterSnapshot]        # chronological
    growth_rates: list[float]               # QoQ growth rate per consecutive pair
    avg_growth_rate: float                  # mean of growth_rates
    trend: str                              # "GROWING" | "STABLE" | "DECLINING" | "VOLATILE"
    latest_label: str
    latest_n_cases: int
    latest_pct_of_fraud: float
    alert_level: str                        # "NONE" | "INFO" | "WARNING" | "FAIL"
    alert_reason: str
    centroid_drift: Optional[float] = None  # cosine dist between first and last centroid

    def to_dict(self) -> dict:
        return {
            "cluster_key":        self.cluster_key,
            "snapshots":          [s.to_dict() for s in self.snapshots],
            "growth_rates":       self.growth_rates,
            "avg_growth_rate":    self.avg_growth_rate,
            "trend":              self.trend,
            "latest_label":       self.latest_label,
            "latest_n_cases":     self.latest_n_cases,
            "latest_pct_of_fraud":self.latest_pct_of_fraud,
            "alert_level":        self.alert_level,
            "alert_reason":       self.alert_reason,
            "centroid_drift":     self.centroid_drift,
        }


@dataclass
class TypologyEvolutionResult:
    """Full output from one evolution tracking run."""
    period: str
    generated_at: str
    n_fraud_cases: int
    n_clusters_current: int
    n_clusters_historical: int
    features_used: list[str]
    trajectories: list[ClusterTrajectory]
    new_clusters: list[ClusterSnapshot]       # appeared this period with no prior match
    disappeared_clusters: list[str]           # cluster_keys not seen this period
    n_growing: int
    n_declining: int
    n_stable: int
    n_emerging: int
    overall_alert: str                        # highest alert level across all clusters
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "period":                 self.period,
            "generated_at":          self.generated_at,
            "n_fraud_cases":         self.n_fraud_cases,
            "n_clusters_current":    self.n_clusters_current,
            "n_clusters_historical": self.n_clusters_historical,
            "features_used":         self.features_used,
            "trajectories":          [t.to_dict() for t in self.trajectories],
            "new_clusters":          [c.to_dict() for c in self.new_clusters],
            "disappeared_clusters":  self.disappeared_clusters,
            "n_growing":             self.n_growing,
            "n_declining":           self.n_declining,
            "n_stable":              self.n_stable,
            "n_emerging":            self.n_emerging,
            "overall_alert":         self.overall_alert,
            "notes":                 self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def top_alerts(self) -> list[ClusterTrajectory]:
        order = {"FAIL": 0, "WARNING": 1, "INFO": 2, "NONE": 3}
        return sorted(
            [t for t in self.trajectories if t.alert_level != "NONE"],
            key=lambda t: (order.get(t.alert_level, 9), -t.avg_growth_rate),
        )

    def display(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule

        console = Console()
        console.print(Rule(f"[bold]Typology Evolution -- period: {self.period}[/bold]"))
        console.print(
            f"  Fraud cases    : {self.n_fraud_cases:,}\n"
            f"  Clusters now   : {self.n_clusters_current}\n"
            f"  Growing        : [red]{self.n_growing}[/red]\n"
            f"  Declining      : [cyan]{self.n_declining}[/cyan]\n"
            f"  Stable         : [green]{self.n_stable}[/green]\n"
            + (f"  Emerging       : [bold red]{self.n_emerging}[/bold red]\n"
               if self.n_emerging else "")
            + (f"  Disappeared    : [dim]{len(self.disappeared_clusters)}[/dim]\n"
               if self.disappeared_clusters else "")
        )

        t = Table("Cluster", "Label", "Cases", "% Fraud", "Avg Growth", "Alert",
                  show_header=True, header_style="bold dim")
        alert_colors = {"FAIL": "red", "WARNING": "yellow", "INFO": "cyan", "NONE": "green"}
        trend_icons  = {"GROWING": "", "DECLINING": "", "STABLE": "->", "VOLATILE": "~"}
        for traj in sorted(self.trajectories, key=lambda x: -x.latest_n_cases):
            ac = alert_colors.get(traj.alert_level, "white")
            ti = trend_icons.get(traj.trend, "")
            growth_str = (
                f"{traj.avg_growth_rate:+.1%}" if traj.avg_growth_rate != 0.0 else "--"
            )
            t.add_row(
                traj.cluster_key,
                f"{ti} {traj.latest_label[:30]}",
                f"{traj.latest_n_cases:,}",
                f"{traj.latest_pct_of_fraud:.1%}",
                growth_str,
                f"[{ac}]{traj.alert_level}[/{ac}]",
            )
        console.print(t)

        if self.new_clusters:
            console.print("\n[bold red]Emerging clusters (no prior match):[/bold red]")
            for c in self.new_clusters:
                console.print(
                    f"  [red]NEW[/red] {c.cluster_key}: {c.n_cases} cases "
                    f"({c.pct_of_fraud:.1%} of fraud) | {c.label}"
                )

        alerts = self.top_alerts()
        if alerts:
            console.print("\n[bold]Active alerts:[/bold]")
            for traj in alerts:
                ac = alert_colors.get(traj.alert_level, "white")
                console.print(
                    f"  [{ac}]{traj.alert_level}[/{ac}] "
                    f"{traj.cluster_key}: {traj.alert_reason}"
                )
        console.print()


# -- Main class ----------------------------------------------------------------

class TypologyEvolutionTracker:
    """
    Track how fraud typology clusters evolve across time periods.

    Parameters
    ----------
    n_clusters      : Fixed number of clusters per period. None = auto (by silhouette).
    match_threshold : Minimum cosine similarity for two centroids to be considered
                      the same cluster across periods. Default 0.60.
    growth_warn     : QoQ fractional growth rate that triggers WARNING. Default 0.25.
    growth_fail     : QoQ fractional growth rate that triggers FAIL. Default 0.50.
    random_state    : Reproducibility seed.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        match_threshold: float = _MATCH_THRESH,
        growth_warn: float = _GROWTH_WARN,
        growth_fail: float = _GROWTH_FAIL,
        random_state: int = 42,
    ) -> None:
        self.n_clusters      = n_clusters
        self.match_threshold = match_threshold
        self.growth_warn     = growth_warn
        self.growth_fail     = growth_fail
        self.random_state    = random_state

    # -- Public API -------------------------------------------------------------

    def track(
        self,
        fraud_df: pd.DataFrame,
        features: list[str],
        period: str,
        workspace: Optional["WorkspaceLayout"] = None,
        dataset_knowledge: Optional["DatasetKnowledge"] = None,
        session_id: str = "unknown",
    ) -> TypologyEvolutionResult:
        """
        Cluster current fraud data, match to history, compute trajectories.

        Parameters
        ----------
        fraud_df           : Confirmed fraud DataFrame.
        features           : Numeric feature columns to cluster on.
        period             : Time period label (e.g. "2024-Q2").
        workspace          : If provided, evolution history is persisted.
        dataset_knowledge  : If provided, significant changes are deposited.
        session_id         : For knowledge entries.
        """
        avail = [f for f in features if f in fraud_df.columns
                 and pd.api.types.is_numeric_dtype(fraud_df[f])]
        if not avail:
            raise ValueError("No numeric features found in fraud_df.")

        # 1. Cluster current period
        current_snapshots, X_scaled, scaler = self._cluster_period(
            fraud_df, avail, period
        )

        # 2. Load history
        history = self._load_history(workspace) if workspace else {}

        # 3. Match clusters to historical identities
        if history.get("clusters"):
            prev_snapshots = self._latest_snapshots(history)
            matched, new_clusters = self._match_clusters(
                current_snapshots, prev_snapshots
            )
        else:
            # First period -- assign stable keys directly
            matched = {f"cluster_{s.original_cluster_id}": s for s in current_snapshots}
            for snap in current_snapshots:
                snap.cluster_key = f"cluster_{snap.original_cluster_id}"
            new_clusters = []

        # 4. Update history
        all_hist_keys = set(history.get("clusters", {}).keys())
        disappeared = [k for k in all_hist_keys if k not in matched and k != "__meta__"]

        new_history = self._build_updated_history(history, matched, period)

        # 5. Compute trajectories
        trajectories = self._compute_trajectories(new_history, matched)

        # 6. Persist
        if workspace:
            self._save_history(workspace, new_history)
            self._append_drift_events(
                workspace=workspace,
                trajectories=self._compute_trajectories(new_history, matched),
                new_clusters=new_clusters,
                disappeared=disappeared,
                period=period,
                session_id=session_id,
            )

        # 7. Summarise
        n_growing  = sum(1 for t in trajectories if t.trend == "GROWING")
        n_declining= sum(1 for t in trajectories if t.trend == "DECLINING")
        n_stable   = sum(1 for t in trajectories if t.trend == "STABLE")
        n_volatile = sum(1 for t in trajectories if t.trend == "VOLATILE")

        alert_order = {"FAIL": 0, "WARNING": 1, "INFO": 2, "NONE": 3}
        overall = min(
            [t.alert_level for t in trajectories] + ["NONE" for _ in new_clusters],
            key=lambda a: alert_order.get(a, 9),
        ) if trajectories else ("FAIL" if new_clusters else "NONE")

        result = TypologyEvolutionResult(
            period=period,
            generated_at=_now(),
            n_fraud_cases=len(fraud_df),
            n_clusters_current=len(current_snapshots),
            n_clusters_historical=len(all_hist_keys),
            features_used=avail,
            trajectories=trajectories,
            new_clusters=new_clusters,
            disappeared_clusters=disappeared,
            n_growing=n_growing,
            n_declining=n_declining,
            n_stable=n_stable + n_volatile,
            n_emerging=len(new_clusters),
            overall_alert=overall,
        )

        # 8. Deposit knowledge
        if dataset_knowledge:
            self._deposit_knowledge(result, dataset_knowledge, session_id)

        return result

    # -- Clustering ------------------------------------------------------------

    def _cluster_period(
        self,
        fraud_df: pd.DataFrame,
        features: list[str],
        period: str,
    ) -> tuple[list[ClusterSnapshot], np.ndarray, object]:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        X = fraud_df[features].copy()
        for col in features:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        X_arr = X.astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        if self.n_clusters:
            k = self.n_clusters
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
        else:
            # Auto-select k by silhouette
            best_k, best_sil, best_labels = 2, -1.0, None
            for k in range(2, min(9, len(X_scaled))):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                lbl = km.fit_predict(X_scaled)
                if len(set(lbl)) < 2:
                    continue
                sil = silhouette_score(X_scaled, lbl, sample_size=min(5000, len(X_scaled)))
                if sil > best_sil:
                    best_k, best_sil, best_labels = k, sil, lbl
            labels = best_labels
            km = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
            km.fit(X_scaled)

        # Build snapshots
        overall_means = X_arr.mean(axis=0)
        overall_stds  = X_arr.std(axis=0) + 1e-8
        snapshots: list[ClusterSnapshot] = []
        n_total = len(X_arr)

        for cid in sorted(set(labels)):
            mask = labels == cid
            cluster_X = X_arr[mask]
            cluster_scaled = X_scaled[mask]

            cluster_means = cluster_X.mean(axis=0)
            zscores = (cluster_means - overall_means) / overall_stds

            # Top features by absolute z-score
            top_idx = np.argsort(np.abs(zscores))[::-1][:5]
            top_feats = [features[i] for i in top_idx]

            # Auto-label
            label = self._auto_label(top_feats[:2], zscores, features)

            # Normalised centroid for matching (from scaled space)
            centroid = cluster_scaled.mean(axis=0)
            norm = np.linalg.norm(centroid) + 1e-8
            centroid_norm = (centroid / norm).tolist()

            snapshots.append(ClusterSnapshot(
                cluster_key=f"cluster_{cid}",     # temporary -- overwritten after matching
                period=period,
                n_cases=int(mask.sum()),
                pct_of_fraud=round(float(mask.sum()) / n_total, 4),
                centroid=centroid_norm,
                top_features=top_feats,
                label=label,
                original_cluster_id=int(cid),
                features_used=list(features),
            ))

        return snapshots, X_scaled, scaler

    # -- Cluster matching ------------------------------------------------------

    def _match_clusters(
        self,
        current: list[ClusterSnapshot],
        prev: dict[str, ClusterSnapshot],
    ) -> tuple[dict[str, ClusterSnapshot], list[ClusterSnapshot]]:
        """
        Match current clusters to previous-period clusters by cosine similarity.

        Feature-aware: each ClusterSnapshot carries ``features_used``.  When a
        previous-period snapshot was built from a *different* feature set
        than the current period (e.g. the playbook used 12 features but the
        monitor cycle uses 8), we project both centroids onto the shared
        features and re-normalise before computing cosine similarity.

        Backwards-compat: if ``prev.features_used`` is empty (legacy snapshot
        written before the feature_used field existed) we attempt strict
        dimension matching; if dimensions differ we set sim = -1 for that
        pair so the cluster is treated as unmatched.

        Returns (matched: {cluster_key -> snapshot}, new_clusters: [snapshot]).
        """
        if not prev:
            # First period with a real history -- assign keys directly
            matched: dict[str, ClusterSnapshot] = {
                f"cluster_{s.original_cluster_id}": s for s in current
            }
            for snap in current:
                snap.cluster_key = f"cluster_{snap.original_cluster_id}"
            return matched, []

        prev_keys = list(prev.keys())

        # Compute similarity per (current, prev) pair so each pair can use
        # its own projection onto shared features.
        sim = np.full((len(current), len(prev_keys)), -1.0, dtype=float)
        for ci, curr_snap in enumerate(current):
            for pi, key in enumerate(prev_keys):
                prev_snap = prev[key]
                s = self._pairwise_centroid_similarity(curr_snap, prev_snap)
                sim[ci, pi] = s

        # Annotated once above (in the early-return branch); re-annotating
        # here would be a Cython redeclare error.
        matched = {}
        new_clusters: list[ClusterSnapshot] = []
        used_prev: set[int] = set()

        # Greedy matching: assign each current cluster to best available prev cluster
        # Sort current clusters by their best similarity (highest first) for priority
        order = np.argsort(-sim.max(axis=1))
        for ci in order:
            best_pi = int(np.argmax(sim[ci]))
            best_sim = float(sim[ci, best_pi])
            if best_sim >= self.match_threshold and best_pi not in used_prev:
                key = prev_keys[best_pi]
                current[ci].cluster_key = key
                matched[key] = current[ci]
                used_prev.add(best_pi)
            else:
                # No good match -> new/emerging cluster
                new_key = f"cluster_new_{current[ci].original_cluster_id}_{current[ci].period}"
                current[ci].cluster_key = new_key
                new_clusters.append(current[ci])

        return matched, new_clusters

    @staticmethod
    def _snapshot_drift(
        first: dict | "ClusterSnapshot",
        last:  dict | "ClusterSnapshot",
    ) -> Optional[float]:
        """Cosine distance between two snapshot centroids, projecting onto
        the shared feature subset when feature lists differ.

        Accepts both raw dicts (as stored in history) and ClusterSnapshot
        objects.  Returns ``None`` when the centroids can't be aligned at
        all (no shared features and legacy mismatched dims).
        """
        def _f(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        f_feats = list(_f(first, "features_used") or [])
        l_feats = list(_f(last,  "features_used") or [])
        f_cent  = list(_f(first, "centroid") or [])
        l_cent  = list(_f(last,  "centroid") or [])

        if f_feats == l_feats and len(f_cent) == len(l_cent):
            a = np.asarray(f_cent, dtype=float)
            b = np.asarray(l_cent, dtype=float)
            return 1.0 - float(np.dot(a, b))

        # Legacy: at least one snapshot has no features_used
        if not f_feats or not l_feats:
            if len(f_cent) == len(l_cent):
                a = np.asarray(f_cent, dtype=float)
                b = np.asarray(l_cent, dtype=float)
                return 1.0 - float(np.dot(a, b))
            return None

        # Project both onto shared features, re-normalise, compute cosine dist.
        l_idx = {f: i for i, f in enumerate(l_feats)}
        shared = [(i, l_idx[f]) for i, f in enumerate(f_feats) if f in l_idx]
        if not shared:
            return None
        a = np.array([f_cent[i] for i, _ in shared], dtype=float)
        b = np.array([l_cent[j] for _, j in shared], dtype=float)
        a /= float(np.linalg.norm(a)) + 1e-8
        b /= float(np.linalg.norm(b)) + 1e-8
        return 1.0 - float(np.dot(a, b))

    @staticmethod
    def _pairwise_centroid_similarity(
        curr_snap: ClusterSnapshot,
        prev_snap: ClusterSnapshot,
    ) -> float:
        """Cosine similarity between two cluster centroids, projecting onto
        the shared feature subset when feature lists differ.

        Returns ``-1.0`` if the centroids can't be meaningfully compared
        (no overlap or legacy snapshot with mismatched dimensions).
        """
        curr_feats = curr_snap.features_used
        prev_feats = prev_snap.features_used
        curr_cent = curr_snap.centroid
        prev_cent = prev_snap.centroid

        # Fast path: identical feature lists OR both legacy with matching dims
        if curr_feats == prev_feats and len(curr_cent) == len(prev_cent):
            a = np.asarray(curr_cent, dtype=float)
            b = np.asarray(prev_cent, dtype=float)
            return float(np.dot(a, b))

        # Legacy: prev has no features_used (old data on disk).  Only safe
        # if dimensions match exactly -- otherwise we have no way to align.
        if not prev_feats:
            if len(curr_cent) == len(prev_cent):
                a = np.asarray(curr_cent, dtype=float)
                b = np.asarray(prev_cent, dtype=float)
                return float(np.dot(a, b))
            return -1.0

        # Symmetric: current has no features_used (shouldn't happen since
        # the new code populates it, but defend anyway).
        if not curr_feats:
            return -1.0

        # General case: project both onto shared features, re-normalise.
        prev_idx = {f: i for i, f in enumerate(prev_feats)}
        shared_idx_pairs = [
            (i, prev_idx[f]) for i, f in enumerate(curr_feats) if f in prev_idx
        ]
        if not shared_idx_pairs:
            return -1.0
        curr_proj = np.array([curr_cent[ci] for ci, _ in shared_idx_pairs], dtype=float)
        prev_proj = np.array([prev_cent[pi] for _, pi in shared_idx_pairs], dtype=float)
        # Re-normalise on the shared subspace
        curr_norm = float(np.linalg.norm(curr_proj)) + 1e-8
        prev_norm = float(np.linalg.norm(prev_proj)) + 1e-8
        return float(np.dot(curr_proj / curr_norm, prev_proj / prev_norm))

    # -- History management ----------------------------------------------------

    @staticmethod
    def _latest_snapshots(history: dict) -> dict[str, ClusterSnapshot]:
        """Extract the most recent snapshot per cluster_key from history.

        ``features_used`` defaults to ``[]`` when a legacy snapshot lacks
        it -- :meth:`_match_clusters` then falls back to strict-dimension
        matching for that snapshot.
        """
        result = {}
        for key, snaps in history.get("clusters", {}).items():
            if snaps:
                last = snaps[-1]
                result[key] = ClusterSnapshot(
                    cluster_key=key,
                    period=last["period"],
                    n_cases=last["n_cases"],
                    pct_of_fraud=last["pct_of_fraud"],
                    centroid=last["centroid"],
                    top_features=last["top_features"],
                    label=last["label"],
                    original_cluster_id=last.get("original_cluster_id", -1),
                    features_used=list(last.get("features_used") or []),
                )
        return result

    @staticmethod
    def _build_updated_history(
        history: dict,
        matched: dict[str, ClusterSnapshot],
        period: str,
    ) -> dict:
        """Append current snapshots to history."""
        h = {
            "periods": list(history.get("periods", [])),
            "clusters": {k: list(v) for k, v in history.get("clusters", {}).items()},
        }
        if period not in h["periods"]:
            h["periods"].append(period)
        for key, snap in matched.items():
            h["clusters"].setdefault(key, [])
            # Avoid duplicate period
            if not any(s.get("period") == period for s in h["clusters"][key]):
                h["clusters"][key].append(snap.to_dict())
        return h

    @staticmethod
    def _load_history(workspace: "WorkspaceLayout") -> dict:
        p = workspace.typology_evolution_path
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load typology evolution history: %s", exc)
            return {}

    @staticmethod
    def _append_drift_events(
        workspace: "WorkspaceLayout",
        trajectories: list["ClusterTrajectory"],
        new_clusters: list["ClusterSnapshot"],
        disappeared: list[str],
        period: str,
        session_id: str = "unknown",
    ) -> None:
        """Emit drift events into the workspace's unified drift_events.jsonl."""
        from ledgerhound.analysis.drift_writer import DriftEventWriter, DriftEvent

        events: list[DriftEvent] = []
        for traj in trajectories:
            if traj.alert_level == "NONE":
                continue
            event_type = (
                "cluster_growth"   if traj.trend == "GROWING" else
                "cluster_decline"  if traj.trend == "DECLINING" else
                "centroid_drift"
            )
            events.append(DriftEvent(
                event_type=event_type,
                period=period,
                severity=traj.alert_level,
                claim=(
                    f"Cluster '{traj.cluster_key}' [{traj.alert_level}] -- "
                    f"{traj.alert_reason[:120]}"
                ),
                cluster_key=traj.cluster_key,
                session_id=session_id,
                details={
                    "trend":            traj.trend,
                    "avg_growth_rate":  round(traj.avg_growth_rate, 4),
                    "centroid_drift":   traj.centroid_drift,
                    "latest_n_cases":   traj.latest_n_cases,
                    "latest_pct_of_fraud": traj.latest_pct_of_fraud,
                    "latest_label":     traj.latest_label,
                },
            ))
        for nc in new_clusters:
            events.append(DriftEvent(
                event_type="cluster_emerging",
                period=period,
                severity="WARNING",  # emerging cluster always merits review
                claim=(
                    f"NEW cluster '{nc.cluster_key}' emerged in {period} -- "
                    f"{nc.n_cases} cases ({nc.pct_of_fraud:.1%} of fraud)"
                ),
                cluster_key=nc.cluster_key,
                session_id=session_id,
                details={
                    "n_cases":       nc.n_cases,
                    "pct_of_fraud":  round(nc.pct_of_fraud, 4),
                    "label":         nc.label,
                    "top_features":  nc.top_features[:5],
                },
            ))
        for key in disappeared:
            events.append(DriftEvent(
                event_type="cluster_disappeared",
                period=period,
                severity="INFO",
                claim=f"Cluster '{key}' disappeared in {period}",
                cluster_key=key,
                session_id=session_id,
                details={},
            ))
        if events:
            DriftEventWriter(workspace).append_many(events)

    @staticmethod
    def _save_history(workspace: "WorkspaceLayout", history: dict) -> None:
        p = workspace.typology_evolution_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    # -- Trajectory computation ------------------------------------------------

    def _compute_trajectories(
        self,
        history: dict,
        matched: dict[str, ClusterSnapshot],
    ) -> list[ClusterTrajectory]:
        trajectories = []

        for key, snaps_raw in history.get("clusters", {}).items():
            if key not in matched:
                continue  # disappeared -- don't compute trajectory

            snaps = sorted(snaps_raw, key=lambda s: s["period"])
            if len(snaps) < 2:
                # Only one period -- no trajectory yet
                s = snaps[-1]
                trajectories.append(ClusterTrajectory(
                    cluster_key=key,
                    snapshots=[ClusterSnapshot(**{**s, "cluster_key": key})],
                    growth_rates=[],
                    avg_growth_rate=0.0,
                    trend="STABLE",
                    latest_label=s["label"],
                    latest_n_cases=s["n_cases"],
                    latest_pct_of_fraud=s["pct_of_fraud"],
                    alert_level="NONE",
                    alert_reason="Insufficient history (1 period)",
                ))
                continue

            # Growth rates between consecutive periods
            case_counts = [s["n_cases"] for s in snaps]
            growth_rates = []
            for i in range(1, len(case_counts)):
                prev = case_counts[i - 1]
                curr = case_counts[i]
                if prev > 0:
                    growth_rates.append((curr - prev) / prev)
                else:
                    growth_rates.append(0.0)

            avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0
            last_growth = growth_rates[-1] if growth_rates else 0.0

            # Trend classification
            if avg_growth > 0.10:
                trend = "GROWING"
            elif avg_growth < -0.10:
                trend = "DECLINING"
            elif np.std(growth_rates) > 0.20:
                trend = "VOLATILE"
            else:
                trend = "STABLE"

            # Alert level -- based on most recent growth rate (not average)
            alert_level = "NONE"
            alert_reason = ""
            if last_growth >= self.growth_fail:
                alert_level = "FAIL"
                alert_reason = (
                    f"Rapid growth {last_growth:+.1%} last period -- "
                    f"possible emerging attack vector"
                )
            elif last_growth >= self.growth_warn:
                alert_level = "WARNING"
                alert_reason = (
                    f"Cluster growing {last_growth:+.1%} last period -- "
                    f"monitor closely"
                )
            elif last_growth <= _DECLINE_INFO:
                alert_level = "INFO"
                alert_reason = (
                    f"Cluster declining {last_growth:+.1%} -- "
                    f"may be neutralised"
                )

            # Centroid drift (cosine distance between first and last centroid).
            # If the two snapshots were built from different feature lists
            # we project onto the shared subset before computing distance
            # (same logic as _pairwise_centroid_similarity).
            centroid_drift = None
            if len(snaps) >= 2:
                drift = self._snapshot_drift(snaps[0], snaps[-1])
                if drift is not None:
                    centroid_drift = round(drift, 4)
                    if drift > _DRIFT_WARN and alert_level == "NONE":
                        alert_level = "WARNING"
                        alert_reason = (
                            f"Feature signature drifting (centroid distance={drift:.3f}) -- "
                            f"attack pattern may be evolving"
                        )

            snap_objs = [
                ClusterSnapshot(**{**s, "cluster_key": key}) for s in snaps
            ]
            last = snaps[-1]
            trajectories.append(ClusterTrajectory(
                cluster_key=key,
                snapshots=snap_objs,
                growth_rates=[round(r, 4) for r in growth_rates],
                avg_growth_rate=round(avg_growth, 4),
                trend=trend,
                latest_label=last["label"],
                latest_n_cases=last["n_cases"],
                latest_pct_of_fraud=last["pct_of_fraud"],
                alert_level=alert_level,
                alert_reason=alert_reason,
                centroid_drift=centroid_drift,
            ))

        return sorted(trajectories, key=lambda t: -t.latest_n_cases)

    # -- Knowledge deposit -----------------------------------------------------

    @staticmethod
    def _deposit_knowledge(
        result: TypologyEvolutionResult,
        knowledge: "DatasetKnowledge",
        session_id: str,
    ) -> None:
        from ledgerhound.memory.knowledge import KnowledgeEntry, KnowledgeType

        for traj in result.top_alerts():
            kt = (KnowledgeType.SIGNAL_EMERGING
                  if traj.alert_level in ("FAIL", "WARNING") and traj.trend == "GROWING"
                  else KnowledgeType.ANOMALY)
            entry = KnowledgeEntry(
                session_id=session_id,
                entry_type=kt,
                feature="",
                claim=(
                    f"Typology '{traj.cluster_key}' ({traj.latest_label}): "
                    f"{traj.alert_level} -- {traj.alert_reason[:100]}"
                ),
                detail=(
                    f"Avg growth rate: {traj.avg_growth_rate:+.1%}. "
                    f"Cases: {traj.latest_n_cases}. "
                    f"Trend: {traj.trend}. "
                    f"Centroid drift: {traj.centroid_drift}"
                ),
                confidence=0.85 if traj.alert_level == "FAIL" else 0.70,
                period=result.period,
                tags=["typology_evolution", traj.alert_level.lower(), traj.trend.lower()],
            )
            try:
                knowledge.add(entry)
            except Exception as exc:
                log.warning("Could not deposit evolution knowledge: %s", exc)

        for nc in result.new_clusters:
            entry = KnowledgeEntry(
                session_id=session_id,
                entry_type=KnowledgeType.SIGNAL_EMERGING,
                feature="",
                claim=(
                    f"NEW fraud typology cluster '{nc.cluster_key}' appeared "
                    f"in {result.period} -- {nc.n_cases} cases ({nc.pct_of_fraud:.1%})"
                ),
                detail=f"Label: {nc.label}. Top features: {', '.join(nc.top_features[:3])}",
                confidence=0.80,
                period=result.period,
                tags=["typology_evolution", "emerging", "new_cluster"],
            )
            try:
                knowledge.add(entry)
            except Exception as exc:
                log.warning("Could not deposit emerging cluster knowledge: %s", exc)

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _auto_label(top_features: list[str], zscores: np.ndarray, all_features: list[str]) -> str:
        parts = []
        for feat in top_features[:2]:
            if feat in all_features:
                idx = all_features.index(feat)
                direction = "high" if zscores[idx] > 0 else "low"
                short = feat.replace("_", " ").split()[-1]
                parts.append(f"{direction}-{short}")
        return " / ".join(parts) if parts else "cluster"
