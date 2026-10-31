"""
sentinel.analysis.signal_decay
================================

SignalDecayTracker -- monitors how feature discriminative power changes over
time and fires alerts when features start losing their ability to separate
fraud from non-fraud.

Why this matters
----------------
Feature signal is not permanent. A SIM-swap feature that was highly
discriminative in 2023 may become irrelevant if the attack pattern
evolves. A feature built on merchant categories may decay after a UPI
guideline change. Models trained on stale features silently underperform.

Sentinel detects this BEFORE retraining cycles by:
1. Measuring each feature's current discriminative power
2. Comparing to the earliest clean baseline in feature history
3. Firing a decay event when power drops past a threshold
4. Depositing institutional memory entries automatically

This is the foundation of the Longitudinal Benchmark:
  "How many weeks before model degradation does Sentinel detect the signal?"

Metrics used
------------
KS statistic    scipy.stats.ks_2samp -- max distributional divergence between
                fraud and non-fraud. Range 0-1, higher = better discrimination.

AUC (Mann-Whitney)  scipy.stats.mannwhitneyu normalised to AUC equivalent.
                Range 0.5-1.0. 0.5 = random, 1.0 = perfect separation.

Mutual Information  sklearn.feature_selection.mutual_info_classif
                Bits of information about fraud label carried by feature.
                Normalised to [0, 1] by dividing by max in the batch.

For categorical features: Cramer's V (normalised chi-squared).

Decay thresholds
----------------
KS drop > 15%   WARNING
KS drop > 30%   FAIL

AUC drop > 0.05 WARNING
AUC drop > 0.10 FAIL

MI drop > 20%   WARNING
MI drop > 40%   FAIL

History format (feature_signal_history_path)
--------------------------------------------
{
  "feature_name": [
    {
      "period": "2024-Q1",
      "snapshot_ts": "ISO",
      "ks_statistic": 0.85,
      "auc": 0.92,
      "mutual_info": 0.61,
      ...
    }
  ]
}

Drift events (drift_events_path)  -- newline-delimited JSON
{"feature": ..., "period": ..., "severity": "WARNING"|"FAIL", ...}
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from sentinel.analysis.multiplicity import adjust as adjust_pvalues

if TYPE_CHECKING:
    from sentinel.workspace.layout import WorkspaceLayout
    from sentinel.memory.knowledge import DatasetKnowledge

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Decay thresholds ---------------------------------------------------------

_KS_WARN_DROP  = 0.15   # 15% relative drop in KS statistic
_KS_FAIL_DROP  = 0.30
_AUC_WARN_DROP = 0.05   # absolute drop in AUC
_AUC_FAIL_DROP = 0.10
_MI_WARN_DROP  = 0.20   # 20% relative drop in mutual info
_MI_FAIL_DROP  = 0.40


# -- Data classes -------------------------------------------------------------

@dataclass
class FeatureSignalMeasure:
    """Signal strength measurement for one feature at one time period."""
    feature: str
    period: str
    snapshot_ts: str
    feature_type: str          # "numeric" | "categorical"
    n_fraud: int
    n_nonfr: int

    # Discrimination metrics
    ks_statistic: float        # 0-1, higher = better separation
    ks_pvalue: float
    auc: float                 # Mann-Whitney AUC 0.5-1.0
    mutual_info: float         # normalised 0-1
    cramers_v: Optional[float] # categorical only

    # V5-001: ks_pvalue is a RAW per-feature p-value and one of these is
    # computed for every feature in the sweep, so it is not the quantity to
    # threshold -- at 200 features, ~10 cross alpha=0.05 under the null.
    # `measure()` fills this in with the family-corrected value; it stays nan
    # for categorical features, which are scored by Cramer's V and never
    # entered a test at all.
    ks_qvalue: float = float("nan")

    # Distribution summary
    fraud_mean: Optional[float] = None
    nonfr_mean: Optional[float] = None
    fraud_std:  Optional[float] = None
    nonfr_std:  Optional[float] = None
    mean_ratio: Optional[float] = None  # fraud_mean / nonfr_mean (>1 = fraud higher)

    def to_dict(self) -> dict:
        return asdict(self)

    def signal_grade(self) -> str:
        """A/B/C/D/F grade based on KS statistic alone."""
        if self.ks_statistic >= 0.60:   return "A"
        if self.ks_statistic >= 0.40:   return "B"
        if self.ks_statistic >= 0.25:   return "C"
        if self.ks_statistic >= 0.10:   return "D"
        return "F"


@dataclass
class SignalDecayEvent:
    """Fired when a feature's discriminative power drops past a threshold."""
    feature: str
    period: str
    baseline_period: str
    primary_metric: str          # "ks_statistic" | "auc" | "mutual_info"
    baseline_value: float
    current_value: float
    absolute_change: float       # current - baseline (negative = decay)
    pct_change: float            # (current - baseline) / baseline * 100
    severity: str                # "WARNING" | "FAIL"
    event_ts: str                = field(default_factory=_now)
    notes: str                   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def human_summary(self) -> str:
        direction = "dropped" if self.pct_change < 0 else "improved"
        return (
            f"[{self.severity}] {self.feature}: {self.primary_metric} {direction} "
            f"{abs(self.pct_change):.1f}% "
            f"({self.baseline_value:.3f} -> {self.current_value:.3f}) "
            f"since {self.baseline_period}"
        )


@dataclass
class SignalDecayResult:
    """Complete output from one signal decay analysis run."""
    period: str
    generated_at: str
    n_features: int
    features_used: list[str]
    measures: list[FeatureSignalMeasure]
    decay_events: list[SignalDecayEvent]
    n_stable: int
    n_warning: int
    n_fail: int
    n_improved: int
    baseline_period: Optional[str]
    notes: str = ""

    # Per-feature lookup (populated after __post_init__)
    _measure_map: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self._measure_map = {m.feature: m for m in self.measures}

    def get(self, feature: str) -> Optional[FeatureSignalMeasure]:
        return self._measure_map.get(feature)

    def top_features(self, n: int = 10) -> list[FeatureSignalMeasure]:
        """Features ranked by KS statistic, descending."""
        return sorted(self.measures, key=lambda m: -m.ks_statistic)[:n]

    def decaying_features(self) -> list[SignalDecayEvent]:
        return sorted(self.decay_events, key=lambda e: e.pct_change)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_measure_map", None)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def summary_table(self) -> list[dict]:
        """Compact per-feature table for display."""
        rows = []
        decay_by_feat = {e.feature: e for e in self.decay_events}
        for m in sorted(self.measures, key=lambda x: -x.ks_statistic):
            ev = decay_by_feat.get(m.feature)
            rows.append({
                "feature":      m.feature,
                "grade":        m.signal_grade(),
                "ks":           round(m.ks_statistic, 3),
                "auc":          round(m.auc, 3),
                "mi":           round(m.mutual_info, 3),
                "status":       ev.severity if ev else "STABLE",
                "decay_pct":    round(ev.pct_change, 1) if ev else None,
            })
        return rows


# -- Main class ----------------------------------------------------------------

class SignalDecayTracker:
    """
    Track feature discriminative power over time and detect decay.

    Parameters
    ----------
    ks_warn_drop   : Relative KS drop fraction that triggers WARNING (default 0.15)
    ks_fail_drop   : Relative KS drop fraction that triggers FAIL    (default 0.30)
    auc_warn_drop  : Absolute AUC drop that triggers WARNING         (default 0.05)
    auc_fail_drop  : Absolute AUC drop that triggers FAIL            (default 0.10)
    mi_warn_drop   : Relative MI drop fraction that triggers WARNING  (default 0.20)
    mi_fail_drop   : Relative MI drop fraction that triggers FAIL     (default 0.40)
    random_state   : Reproducibility seed for MI estimation.
    """

    def __init__(
        self,
        ks_warn_drop:  float = _KS_WARN_DROP,
        ks_fail_drop:  float = _KS_FAIL_DROP,
        auc_warn_drop: float = _AUC_WARN_DROP,
        auc_fail_drop: float = _AUC_FAIL_DROP,
        mi_warn_drop:  float = _MI_WARN_DROP,
        mi_fail_drop:  float = _MI_FAIL_DROP,
        random_state:  int   = 42,
    ) -> None:
        self.ks_warn_drop  = ks_warn_drop
        self.ks_fail_drop  = ks_fail_drop
        self.auc_warn_drop = auc_warn_drop
        self.auc_fail_drop = auc_fail_drop
        self.mi_warn_drop  = mi_warn_drop
        self.mi_fail_drop  = mi_fail_drop
        self.random_state  = random_state

    # -- Public API ------------------------------------------------------------

    def measure(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
        period: str,
    ) -> list[FeatureSignalMeasure]:
        """
        Measure discriminative power for each feature in the current period.

        Parameters
        ----------
        fraud_df  : Confirmed fraud DataFrame.
        nonfr_df  : Non-fraud DataFrame (can be a sample).
        features  : Feature columns to measure. Numeric and categorical handled.
        period    : Label for this snapshot (e.g. "2024-Q2", "2024-W12").

        Returns
        -------
        List of FeatureSignalMeasure -- one per feature that could be measured.
        """
        measures = []
        ts = _now()

        # Build binary label arrays for MI calculation
        fraud_vals_all  = fraud_df[features].copy() if all(f in fraud_df.columns for f in features) else None
        nonfr_vals_all  = nonfr_df[features].copy() if all(f in nonfr_df.columns for f in features) else None

        # Prepare combined df for MI
        avail = [f for f in features if f in fraud_df.columns and f in nonfr_df.columns]
        if not avail:
            log.warning("No common features found in both fraud_df and nonfr_df")
            return measures

        combined_df = pd.concat([
            fraud_df[avail].assign(_is_fraud=1),
            nonfr_df[avail].assign(_is_fraud=0),
        ], ignore_index=True)

        # Compute mutual information for numeric features in batch
        numeric_feats = [
            f for f in avail
            if pd.api.types.is_numeric_dtype(combined_df[f])
        ]
        mi_scores = self._compute_mi(combined_df, numeric_feats)

        for feat in avail:
            try:
                m = self._measure_feature(
                    fraud_df=fraud_df,
                    nonfr_df=nonfr_df,
                    feature=feat,
                    period=period,
                    snapshot_ts=ts,
                    mi_score=mi_scores.get(feat, 0.0),
                )
                if m is not None:
                    measures.append(m)
            except Exception as exc:
                log.warning("Could not measure signal for '%s': %s", feat, exc)

        # V5-001: one KS test per feature means the sweep is a family, and the
        # raw p-values above are not comparable to a fixed alpha. Correct
        # across the family that was actually tested -- categorical features
        # carry nan and are excluded rather than coerced, since Cramer's V is
        # not a hypothesis test.
        mult = adjust_pvalues({m.feature: m.ks_pvalue for m in measures})
        for m in measures:
            m.ks_qvalue = mult.q(m.feature)
        if mult.n_tests:
            log.info("Signal measured: %d features for period '%s' -- %s",
                     len(measures), period, mult.summary())
        else:
            log.info("Signal measured: %d features for period '%s'",
                     len(measures), period)
        return measures

    def detect_decay(
        self,
        current_measures: list[FeatureSignalMeasure],
        history: dict[str, list[dict]],
    ) -> list[SignalDecayEvent]:
        """
        Compare current measures to historical baselines and return decay events.

        Parameters
        ----------
        current_measures : Output from measure().
        history          : Feature history dict loaded from workspace.
                           Format: {feature_name: [snapshot_dict, ...]}

        Returns
        -------
        List of SignalDecayEvent for any feature whose signal has degraded.
        """
        events: list[SignalDecayEvent] = []

        for measure in current_measures:
            feat_hist = history.get(measure.feature, [])
            if not feat_hist:
                continue  # no baseline yet

            # Use the earliest snapshot as baseline (most conservative)
            baseline = feat_hist[0]
            baseline_period = baseline.get("period", "unknown")

            evt = self._check_decay(measure, baseline, baseline_period)
            if evt:
                events.append(evt)

        return events

    def track(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
        period: str,
        workspace: Optional["WorkspaceLayout"] = None,
        dataset_knowledge: Optional["DatasetKnowledge"] = None,
        session_id: str = "unknown",
    ) -> SignalDecayResult:
        """
        Full pipeline: measure -> compare -> write history -> fire events -> deposit memory.

        Parameters
        ----------
        fraud_df           : Confirmed fraud DataFrame.
        nonfr_df           : Non-fraud DataFrame.
        features           : Feature columns to analyze.
        period             : Time-period label for this snapshot.
        workspace          : If provided, history and drift events are persisted.
        dataset_knowledge  : If provided, decay events are deposited as knowledge entries.
        session_id         : Session ID for knowledge entries.

        Returns
        -------
        SignalDecayResult with full measures and any decay events.
        """
        # 1. Measure current signal
        measures = self.measure(fraud_df, nonfr_df, features, period)

        # 2. Load history
        history: dict[str, list[dict]] = {}
        if workspace:
            history = self._load_history(workspace)

        # 3. Detect decay against baselines
        events = self.detect_decay(measures, history)

        # 4. Classify features
        decay_feats = {e.feature for e in events}
        warn_feats = {e.feature for e in events if e.severity == "WARNING"}
        fail_feats = {e.feature for e in events if e.severity == "FAIL"}
        # Improved = current KS higher than baseline
        improved_feats: set[str] = set()
        for m in measures:
            hist = history.get(m.feature, [])
            if hist and m.ks_statistic > hist[0].get("ks_statistic", 0.0) * 1.05:
                improved_feats.add(m.feature)

        baseline_period: Optional[str] = None
        if history:
            # Use earliest period from any feature
            all_periods = [
                snap["period"]
                for snaps in history.values()
                for snap in snaps
                if "period" in snap
            ]
            if all_periods:
                baseline_period = all_periods[0]

        result = SignalDecayResult(
            period=period,
            generated_at=_now(),
            n_features=len(measures),
            features_used=[m.feature for m in measures],
            measures=measures,
            decay_events=events,
            n_stable=len(measures) - len(decay_feats) - len(improved_feats),
            n_warning=len(warn_feats),
            n_fail=len(fail_feats),
            n_improved=len(improved_feats),
            baseline_period=baseline_period,
        )

        # 5. Persist history
        if workspace:
            self._append_history(workspace, measures)
            self._append_drift_events(workspace, events, session_id=session_id)
            log.info(
                "Signal history updated: period=%s features=%d events=%d",
                period, len(measures), len(events),
            )

        # 6. Deposit knowledge entries for significant events
        if dataset_knowledge and events:
            self._deposit_decay_knowledge(
                knowledge=dataset_knowledge,
                events=events,
                period=period,
                session_id=session_id,
            )

        return result

    # -- Feature-level measurement ---------------------------------------------

    def _measure_feature(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        feature: str,
        period: str,
        snapshot_ts: str,
        mi_score: float,
    ) -> Optional[FeatureSignalMeasure]:
        from scipy import stats

        if feature not in fraud_df.columns or feature not in nonfr_df.columns:
            return None

        fraud_series = fraud_df[feature].dropna()
        nonfr_series = nonfr_df[feature].dropna()

        if len(fraud_series) < 5 or len(nonfr_series) < 5:
            return None

        is_numeric = pd.api.types.is_numeric_dtype(fraud_series)

        if is_numeric:
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(
                fraud_series.astype(float).values,
                nonfr_series.astype(float).values,
            )

            # Mann-Whitney AUC
            auc = self._mannwhitney_auc(fraud_series, nonfr_series)

            # Distribution stats
            f_mean = float(fraud_series.mean())
            n_mean = float(nonfr_series.mean())
            f_std  = float(fraud_series.std())
            n_std  = float(nonfr_series.std())
            mean_ratio = round(f_mean / n_mean, 4) if abs(n_mean) > 1e-8 else None

            return FeatureSignalMeasure(
                feature=feature,
                period=period,
                snapshot_ts=snapshot_ts,
                feature_type="numeric",
                n_fraud=len(fraud_series),
                n_nonfr=len(nonfr_series),
                ks_statistic=round(ks_stat, 6),
                ks_pvalue=round(ks_pval, 6),
                auc=round(auc, 6),
                mutual_info=round(mi_score, 6),
                cramers_v=None,
                fraud_mean=round(f_mean, 4),
                nonfr_mean=round(n_mean, 4),
                fraud_std=round(f_std, 4),
                nonfr_std=round(n_std, 4),
                mean_ratio=mean_ratio,
            )

        else:
            # Categorical: Cramer's V
            cramers = self._cramers_v(fraud_series, nonfr_series)
            # Use Cramer's V as a proxy for KS (same 0-1 scale)
            return FeatureSignalMeasure(
                feature=feature,
                period=period,
                snapshot_ts=snapshot_ts,
                feature_type="categorical",
                n_fraud=len(fraud_series),
                n_nonfr=len(nonfr_series),
                ks_statistic=round(cramers, 6),  # Cramer's V as discrimination proxy
                ks_pvalue=float("nan"),
                auc=0.5 + cramers / 2,            # rough AUC proxy from Cramer's V
                mutual_info=round(mi_score, 6),
                cramers_v=round(cramers, 6),
            )

    # -- Decay detection --------------------------------------------------------

    def _check_decay(
        self,
        current: FeatureSignalMeasure,
        baseline: dict,
        baseline_period: str,
    ) -> Optional[SignalDecayEvent]:
        """Return a decay event if any metric has degraded past threshold, else None."""
        events_candidates: list[SignalDecayEvent] = []

        # KS statistic check (relative drop)
        base_ks = baseline.get("ks_statistic", 0.0)
        if base_ks > 0.01:  # only check if baseline had signal
            ks_drop = (base_ks - current.ks_statistic) / base_ks
            if ks_drop >= self.ks_fail_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="ks_statistic",
                    baseline_value=base_ks,
                    current_value=current.ks_statistic,
                    absolute_change=current.ks_statistic - base_ks,
                    pct_change=(current.ks_statistic - base_ks) / base_ks * 100,
                    severity="FAIL",
                ))
            elif ks_drop >= self.ks_warn_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="ks_statistic",
                    baseline_value=base_ks,
                    current_value=current.ks_statistic,
                    absolute_change=current.ks_statistic - base_ks,
                    pct_change=(current.ks_statistic - base_ks) / base_ks * 100,
                    severity="WARNING",
                ))

        # AUC check (absolute drop)
        base_auc = baseline.get("auc", 0.5)
        if base_auc > 0.51:
            auc_drop = base_auc - current.auc
            if auc_drop >= self.auc_fail_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="auc",
                    baseline_value=base_auc,
                    current_value=current.auc,
                    absolute_change=current.auc - base_auc,
                    pct_change=(current.auc - base_auc) / (base_auc - 0.5 + 1e-8) * 100,
                    severity="FAIL",
                ))
            elif auc_drop >= self.auc_warn_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="auc",
                    baseline_value=base_auc,
                    current_value=current.auc,
                    absolute_change=current.auc - base_auc,
                    pct_change=(current.auc - base_auc) / (base_auc - 0.5 + 1e-8) * 100,
                    severity="WARNING",
                ))

        # MI check (relative drop)
        base_mi = baseline.get("mutual_info", 0.0)
        if base_mi > 0.01:
            mi_drop = (base_mi - current.mutual_info) / base_mi
            if mi_drop >= self.mi_fail_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="mutual_info",
                    baseline_value=base_mi,
                    current_value=current.mutual_info,
                    absolute_change=current.mutual_info - base_mi,
                    pct_change=(current.mutual_info - base_mi) / base_mi * 100,
                    severity="FAIL",
                ))
            elif mi_drop >= self.mi_warn_drop:
                events_candidates.append(SignalDecayEvent(
                    feature=current.feature,
                    period=current.period,
                    baseline_period=baseline_period,
                    primary_metric="mutual_info",
                    baseline_value=base_mi,
                    current_value=current.mutual_info,
                    absolute_change=current.mutual_info - base_mi,
                    pct_change=(current.mutual_info - base_mi) / base_mi * 100,
                    severity="WARNING",
                ))

        if not events_candidates:
            return None

        # Return the worst event (FAIL before WARNING, then largest absolute change)
        return sorted(
            events_candidates,
            key=lambda e: (0 if e.severity == "FAIL" else 1, e.absolute_change),
        )[0]

    # -- Stats helpers ---------------------------------------------------------

    def _compute_mi(
        self, combined_df: pd.DataFrame, numeric_features: list[str]
    ) -> dict[str, float]:
        """Compute normalised mutual information for a batch of numeric features."""
        if not numeric_features:
            return {}
        try:
            from sklearn.feature_selection import mutual_info_classif

            X = combined_df[numeric_features].copy()
            y = combined_df["_is_fraud"].values

            # Fill nulls with median per column
            for col in numeric_features:
                if X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())

            scores = mutual_info_classif(
                X.astype(float).values,
                y,
                discrete_features=False,
                random_state=self.random_state,
            )
            max_score = max(scores.max(), 1e-8)
            return {f: float(s / max_score) for f, s in zip(numeric_features, scores)}
        except Exception as exc:
            log.warning("MI computation failed: %s", exc)
            return {}

    @staticmethod
    def _mannwhitney_auc(
        fraud_series: pd.Series, nonfr_series: pd.Series
    ) -> float:
        """Compute AUC equivalent from Mann-Whitney U statistic."""
        from scipy import stats
        try:
            u_stat, _ = stats.mannwhitneyu(
                fraud_series.astype(float).values,
                nonfr_series.astype(float).values,
                alternative="two-sided",
            )
            n1 = len(fraud_series)
            n2 = len(nonfr_series)
            auc = u_stat / (n1 * n2)
            # Ensure AUC >= 0.5 (flip if model predicts in wrong direction)
            return float(max(auc, 1.0 - auc))
        except Exception:
            return 0.5

    @staticmethod
    def _cramers_v(s1: pd.Series, s2: pd.Series) -> float:
        """Cramer's V between fraud and non-fraud value distributions."""
        from scipy import stats
        try:
            combined = pd.concat([
                s1.astype(str).value_counts().rename("fraud"),
                s2.astype(str).value_counts().rename("nonfr"),
            ], axis=1).fillna(0)
            chi2, _, _, _ = stats.chi2_contingency(combined.T)
            n = combined.sum().sum()
            k = max(combined.shape) - 1
            return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0
        except Exception:
            return 0.0

    # -- History I/O ------------------------------------------------------------

    @staticmethod
    def _load_history(workspace: "WorkspaceLayout") -> dict[str, list[dict]]:
        p = workspace.feature_signal_history_path
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load signal history: %s", exc)
            return {}

    @staticmethod
    def _append_history(
        workspace: "WorkspaceLayout",
        measures: list[FeatureSignalMeasure],
    ) -> None:
        history = SignalDecayTracker._load_history(workspace)
        for m in measures:
            history.setdefault(m.feature, [])
            # Avoid duplicate period entries
            existing_periods = {s.get("period") for s in history[m.feature]}
            if m.period not in existing_periods:
                history[m.feature].append(m.to_dict())
        p = workspace.feature_signal_history_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _append_drift_events(
        workspace: "WorkspaceLayout",
        events: list[SignalDecayEvent],
        session_id: str = "unknown",
    ) -> None:
        if not events:
            return
        # Use unified DriftEventWriter (schema = drift_event/v1)
        from sentinel.analysis.drift_writer import DriftEventWriter, DriftEvent

        writer = DriftEventWriter(workspace)
        unified = [
            DriftEvent(
                event_type="signal_decay" if ev.severity in ("WARN", "WARNING", "FAIL")
                              else "signal_emerging",
                period=ev.period,
                severity=("FAIL" if ev.severity == "FAIL"
                          else "WARNING" if ev.severity in ("WARN", "WARNING")
                          else "INFO"),
                claim=ev.human_summary() if hasattr(ev, "human_summary") else (
                    f"{ev.feature}: {ev.primary_metric} "
                    f"{ev.pct_change:+.1f}% vs {ev.baseline_period}"
                ),
                feature=ev.feature,
                session_id=session_id,
                details=ev.to_dict(),
            )
            for ev in events
        ]
        writer.append_many(unified)

    @staticmethod
    def _deposit_decay_knowledge(
        knowledge: "DatasetKnowledge",
        events: list[SignalDecayEvent],
        period: str,
        session_id: str,
    ) -> None:
        from sentinel.memory.knowledge import KnowledgeEntry, KnowledgeType

        for ev in events:
            entry = KnowledgeEntry(
                session_id=session_id,
                entry_type=KnowledgeType.SIGNAL_DECAY,
                feature=ev.feature,
                claim=(
                    f"{ev.feature} signal decayed {abs(ev.pct_change):.1f}% "
                    f"({ev.primary_metric}: {ev.baseline_value:.3f} -> {ev.current_value:.3f})"
                ),
                detail=(
                    f"Decay detected in period '{period}' vs baseline '{ev.baseline_period}'. "
                    f"Severity: {ev.severity}. Primary metric: {ev.primary_metric}. "
                    f"{ev.notes}"
                ),
                confidence=0.9 if ev.severity == "FAIL" else 0.75,
                period=period,
                tags=["signal_decay", ev.severity.lower(), ev.primary_metric],
            )
            try:
                knowledge.add(entry)
            except Exception as exc:
                log.warning("Could not deposit decay knowledge entry: %s", exc)


# -- Convenience display --------------------------------------------------------

def display_signal_result(result: SignalDecayResult, top_n: int = 15) -> None:
    """Pretty-print a SignalDecayResult using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule

    console = Console()
    console.print(Rule(f"[bold]Feature Signal Health -- period: {result.period}[/bold]"))
    console.print(
        f"  Features measured : {result.n_features}\n"
        f"  Stable            : [green]{result.n_stable}[/green]\n"
        f"  Warning           : [yellow]{result.n_warning}[/yellow]\n"
        f"  Fail              : [red]{result.n_fail}[/red]\n"
        f"  Improved          : [cyan]{result.n_improved}[/cyan]\n"
        + (f"  Baseline period   : {result.baseline_period}\n" if result.baseline_period else "")
    )

    t = Table("Feature", "Grade", "KS", "AUC", "MI", "Status", "%",
              show_header=True, header_style="bold dim")
    status_colors = {"STABLE": "green", "WARNING": "yellow", "FAIL": "red", None: "dim"}
    for row in result.summary_table()[:top_n]:
        sc = status_colors.get(row["status"], "white")
        delta_str = f"{row['decay_pct']:+.1f}%" if row["decay_pct"] is not None else "--"
        t.add_row(
            row["feature"],
            row["grade"],
            str(row["ks"]),
            str(row["auc"]),
            str(row["mi"]),
            f"[{sc}]{row['status']}[/{sc}]",
            delta_str,
        )
    console.print(t)

    if result.decay_events:
        console.print("\n[bold red]Decay events:[/bold red]")
        for ev in result.decaying_features():
            console.print(f"  {ev.human_summary()}")
    console.print()
