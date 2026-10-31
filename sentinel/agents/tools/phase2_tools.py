"""
sentinel.agents.tools.phase2_tools
====================================

Phase 2 tool dispatch wrappers for the ReAct agent.

Each public method accepts a plain dict (parsed from the agent's JSON action)
and returns a JSON string -- exactly the same contract as the existing
dict_tools and findings_tools in eda_agent.py.

Available tools
---------------
psi_certify          Run PSI calculation comparing a local sample to
                     population distributions stored in workspace memory.

mine_hard_negatives  Find non-fraud accounts closest to the fraud cluster
                     using one of three strategies (isolation_forest,
                     centroid_distance, linear_boundary).

cluster_typologies   Cluster confirmed fraud cases to discover distinct fraud
                     typology groups (KMeans / DBSCAN / GMix).

validate_consistency Cross-dataset consistency checks between summary table,
                     raw transactions, and fraud labels.

check_leakage        Leakage Sentinel (V3-018): point-in-time vetting,
                     target-leak lexicon, dev-vs-OOT AUROC gap, dominant-
                     feature interrogation. Blocking verdicts.

tier_check           Ask TierClassifier which compute tier (LOCAL vs TRINO)
                     should handle a given dataset/operation.

deposit_insight      Write a structured knowledge entry to the DatasetKnowledge
                     store for the given dataset -- permanent institutional memory.

Design rules
-----------
- All methods MUST return a JSON-serialisable string (never raise to caller).
- Errors are returned as {"ok": false, "error": "..."} -- the agent sees them
  as an Observation and can retry or report to the user.
- Heavy imports (sklearn etc.) are deferred inside methods so that the class
  can be imported even in environments without those deps.
- These tools NEVER call Trino -- they work on DataFrames already in memory.
  The tier_check tool tells the agent which tier to use; it does not execute.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.workspace.layout import WorkspaceLayout
    from sentinel.memory.knowledge import DatasetKnowledge
    from sentinel.compute.tier import TierClassifier
    from sentinel.compute.context import ComputeContext
    from sentinel.compute.handle import DataFrameHandle
    from sentinel.narrative.op_log import OperationLog

log = logging.getLogger(__name__)

# -- Numeric features used by default when none are specified ------------------
_DEFAULT_NUMERIC_FEATURES = [
    "account_age_days",
    "total_txn_count",
    "total_txn_amount",
    "avg_txn_amount",
    "max_txn_amount",
    "unique_beneficiaries",
    "new_benef_rate",
    "night_txn_rate",
    "weekend_txn_rate",
    "p2p_ratio",
    "txn_velocity_7d",
    "avg_txn_gap_hours",
    "device_change_count",
    "sim_change_count",
    "failed_txn_rate",
]


def _safe_json(obj: Any) -> str:
    """Dump *obj* to JSON, replacing un-serialisable values with their str()."""
    return json.dumps(obj, indent=2, default=str)


def _error(msg: str) -> str:
    return _safe_json({"ok": False, "error": msg})


def _jsonable_subset(d: dict, *, max_keys: int = 20, max_value_chars: int = 200) -> dict:
    """Best-effort: keep small JSON-serialisable arg values for the narrative.
    Large nested objects (population distributions, weight dicts) are summarised
    rather than copied verbatim so the op_log stays cheap to load."""
    out: dict = {}
    for k, v in list(d.items())[:max_keys]:
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                if isinstance(v, str) and len(v) > max_value_chars:
                    out[k] = v[: max_value_chars - 3] + "..."
                else:
                    out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = (f"<list len={len(v)}>"
                          if len(v) > 10 else list(v)[:10])
            elif isinstance(v, dict):
                out[k] = f"<dict keys={len(v)}>"
            else:
                out[k] = f"<{type(v).__name__}>"
        except Exception:
            out[k] = "<unrepr>"
    return out


def _track_columns(tool_name: str, columns) -> None:
    """Best-effort coverage telemetry -- never raises."""
    try:
        from sentinel.instrumentation import record_columns as _rc
        _rc(tool_name, columns)
    except Exception:
        pass


class Phase2Tools:
    """
    Callable tool-set for the Phase 2 ReAct agent loop.

    Parameters
    ----------
    workspace      : WorkspaceLayout -- root filesystem layout.
    datasets       : Dict of name -> pd.DataFrame currently loaded in memory.
                     Keys are canonical names: "fraud", "nonfr", "summary",
                     "raw_txn", etc.
    knowledge      : Optional DatasetKnowledge for the active dataset.
    tier_classifier: Optional TierClassifier for tier_check tool.
    session_id     : Current session ID (for knowledge entries).
    """

    # Registry of all tool names exposed to the agent
    TOOL_NAMES = frozenset({
        "psi_certify",
        "mine_hard_negatives",
        "cluster_typologies",
        "validate_consistency",
        "check_leakage",
        "tier_check",
        "deposit_insight",
        "measure_signal_decay",
        "track_typology_evolution",
        "detect_mule_motifs",
        "compute_account_centrality",
        "find_mule_rings",
        "trace_money_flow",
        "fingerprint_accounts",
    })

    def __init__(
        self,
        workspace: "WorkspaceLayout | None" = None,
        datasets: dict | None = None,
        knowledge: "DatasetKnowledge | None" = None,
        tier_classifier: "TierClassifier | None" = None,
        session_id: str = "unknown",
        ctx: "ComputeContext | None" = None,
        op_log: "OperationLog | None" = None,
    ) -> None:
        self.workspace = workspace
        self.datasets: dict = datasets or {}
        self.knowledge = knowledge
        self.tier_classifier = tier_classifier
        self.session_id = session_id
        self.ctx = ctx
        self.op_log = op_log
        # Map of dataset key -> DataFrameHandle, populated lazily on first
        # tool dispatch. Only used when ctx + op_log are both attached.
        self._handles: dict[str, "DataFrameHandle"] = {}

    # -- Handle registration (Layer C: data I/O boundary) ---------------------

    def _ensure_handle(self, key: str) -> "DataFrameHandle | None":
        """If a frame is loaded under `key` and ctx is attached, register it
        once and cache the resulting handle. Returns None when not applicable."""
        if self.ctx is None or key not in self.datasets:
            return None
        if key in self._handles:
            return self._handles[key]
        try:
            h = self.ctx.load(self.datasets[key], name=key, op="load")
            self._handles[key] = h
            return h
        except Exception:
            return None

    def _handles_for_args(self, args: dict) -> list[str]:
        """Best-effort: discover which loaded frames a tool will likely touch
        based on common arg key names. Used only for narrative attribution."""
        if self.ctx is None:
            return []
        candidates = [
            args.get("sample_key"),
            args.get("fraud_key"),
            args.get("nonfr_key"),
            args.get("frame_key"),
            args.get("dataset"),
        ]
        ids: list[str] = []
        for key in candidates:
            if not key:
                continue
            h = self._ensure_handle(str(key))
            if h is not None:
                ids.append(h.id)
        return ids

    # -- Tool lookup (mirrors dict_tools / findings_tools pattern) -------------

    def as_tool_map(self) -> dict[str, Any]:
        """Return {tool_name: callable} for use in the agent dispatch loop.

        When an OperationLog is attached, each tool call is wrapped so that
        invocation, params, and result status all land in the narrative log.
        The wrapper is a thin shell -- it does not alter the tool body or
        return contract (still returns the JSON-string payload)."""
        raw = {name: getattr(self, name) for name in self.TOOL_NAMES}
        if self.op_log is None:
            return raw
        return {name: self._wrap_tool_for_op_log(name, fn) for name, fn in raw.items()}

    def _wrap_tool_for_op_log(self, name: str, fn):
        """Build an op-log-wrapped version of a bound tool method."""
        op_log = self.op_log

        def wrapped(args: dict) -> str:
            input_ids = self._handles_for_args(args or {})
            safe_params = _jsonable_subset(args or {})
            with op_log.step(
                kind="tool",
                op=name,
                inputs=input_ids,
                params=safe_params,
                tags=["tool"],
            ) as step:
                result = fn(args)
                # Peek at the JSON payload for status/error so the narrative
                # can flag failed tools without re-parsing later.
                try:
                    payload = json.loads(result)
                    if isinstance(payload, dict):
                        if payload.get("ok") is False:
                            step.add_tag("error")
                            err = str(payload.get("error", ""))[:200]
                            step.set_notes(err)
                        else:
                            status = payload.get("status") or payload.get("verdict")
                            if status:
                                step.add_param("status", str(status))
                except Exception:
                    pass
                return result

        return wrapped

    # -- PSI CERTIFY -----------------------------------------------------------

    def psi_certify(self, args: dict) -> str:
        """
        Compute PSI between a local sample and the population distribution.

        Expected args
        -------------
        features        : list[str]  -- numeric features to check (optional,
                          defaults to standard UPI feature list)
        sample_key      : str        -- key into self.datasets for the local
                          non-fraud sample (default: "nonfr")
        n_buckets       : int        -- number of PSI buckets (default: 20)
        pop_dists       : dict       -- precomputed population distributions
                          {feature: {bucket_label: fraction}}. If omitted,
                          the sample distribution is used as its own baseline
                          (useful for smoke-testing the pipeline).
        n_population    : int        -- total population count (default: 1_000_000)
        weights         : dict       -- feature importance weights for weighted PSI
        """
        try:
            import pandas as pd
            from sentinel.sampling.psi import PSICalculator, compute_bucket_bounds, PSIStatus

            features   = args.get("features") or _DEFAULT_NUMERIC_FEATURES
            sample_key = args.get("sample_key", "nonfr")
            n_buckets  = int(args.get("n_buckets", 20))
            n_pop      = int(args.get("n_population", 1_000_000))
            weights    = args.get("weights") or {}
            pop_dists  = args.get("pop_dists") or {}

            sample_df = self._require_df(sample_key)
            if isinstance(sample_df, str):  # error string
                return sample_df

            available = [f for f in features if f in sample_df.columns]
            if not available:
                return _error(
                    f"None of the requested features found in '{sample_key}'. "
                    f"Available columns: {list(sample_df.columns)[:20]}"
                )
            _track_columns("psi_certify", available)

            calc = PSICalculator()

            # Build bounds from sample, compute per-feature PSI
            numeric_bounds: dict = {}
            numeric_pop: dict = {}

            for feat in available:
                series = sample_df[feat].dropna()
                if series.empty:
                    continue
                bounds = compute_bucket_bounds(series, n_buckets=n_buckets)
                numeric_bounds[feat] = bounds

                if feat in pop_dists:
                    # Caller provided actual population distribution
                    numeric_pop[feat] = pop_dists[feat]
                else:
                    # Use sample as its own baseline (PSI = 0 -- confirms pipeline works)
                    from sentinel.sampling.psi import _numeric_sample_dist
                    dist = _numeric_sample_dist(series, bounds)
                    numeric_pop[feat] = dist

            psi_result = calc.batch(
                sample_df=sample_df[available],
                numeric_population_dists=numeric_pop,
                categorical_population_dists={},
                numeric_bounds=numeric_bounds,
                weights={f: weights.get(f, 1.0) for f in available},
                n_population=n_pop,
            )

            # Save certificate if workspace is available
            cert_path = None
            if self.workspace:
                from sentinel.sampling.certificate import CertificateGenerator
                gen = CertificateGenerator(weights=weights or None)
                cert = gen.generate(
                    psi_result=psi_result,
                    sample_name=sample_key,
                    sample_version="v1",
                    fraud_dataset_key="fraud",
                    nonfr_dataset_key=sample_key,
                    fraud_rows=len(self.datasets.get("fraud", pd.DataFrame())),
                    nonfr_rows=len(sample_df),
                    sampling_strategy="manual",
                )
                cert_dir = self.workspace.samples_dir / sample_key
                cert_dir.mkdir(parents=True, exist_ok=True)
                cert_path_obj = cert_dir / "sampling_certificate.json"
                cert.save(cert_path_obj)
                cert_path = str(cert_path_obj)

            feature_results = []
            for fr in psi_result.features:
                feature_results.append({
                    "feature": fr.feature,
                    "psi": round(fr.psi, 4),
                    "status": fr.status.value,
                    "ks_statistic": round(fr.ks_statistic, 4) if fr.ks_statistic else None,
                    "ks_status": fr.ks_status if fr.ks_status else None,
                })

            return _safe_json({
                "ok": True,
                "tool": "psi_certify",
                "sample_key": sample_key,
                "n_sample": len(sample_df),
                "n_features_checked": len(available),
                "weighted_psi": round(psi_result.weighted_psi, 4),
                "overall_status": psi_result.overall_status.value,
                "n_pass": psi_result.n_pass,
                "n_warning": psi_result.n_warning,
                "n_fail": psi_result.n_fail,
                "feature_results": feature_results,
                "certificate_path": cert_path,
                "note": (
                    "PSI computed against sample baseline (no population distributions "
                    "provided). Pass pop_dists arg with Trino-queried bucket distributions "
                    "for a meaningful representativeness check."
                ) if not pop_dists else "",
            })

        except Exception:
            return _error(f"psi_certify failed:\n{traceback.format_exc()[-600:]}")

    # -- MINE HARD NEGATIVES ---------------------------------------------------

    def mine_hard_negatives(self, args: dict) -> str:
        """
        Find non-fraud accounts most similar to the fraud cluster.

        Expected args
        -------------
        features    : list[str]  -- numeric feature columns (required)
        strategy    : str        -- isolation_forest | centroid_distance |
                                   linear_boundary  (default: isolation_forest)
        top_k       : int        -- exact number of hard negatives to return
                                   (default: 10% of non-fraud input)
        top_k_ratio : float      -- fraction when top_k not set (default: 0.10)
        fraud_key   : str        -- dataset key for fraud data (default: "fraud")
        nonfr_key   : str        -- dataset key for non-fraud data (default: "nonfr")
        save_result : bool       -- whether to save hard_neg_df to workspace
        """
        try:
            from sentinel.sampling.hard_negatives import (
                HardNegativeMiner, HardNegativeStrategy,
            )

            features    = args.get("features") or _DEFAULT_NUMERIC_FEATURES
            strategy_s  = args.get("strategy", "isolation_forest")
            top_k       = args.get("top_k")
            top_k_ratio = float(args.get("top_k_ratio", 0.10))
            fraud_key   = args.get("fraud_key", "fraud")
            nonfr_key   = args.get("nonfr_key", "nonfr")
            save_result = bool(args.get("save_result", False))

            fraud_df = self._require_df(fraud_key)
            if isinstance(fraud_df, str):
                return fraud_df

            nonfr_df = self._require_df(nonfr_key)
            if isinstance(nonfr_df, str):
                return nonfr_df

            try:
                strategy = HardNegativeStrategy(strategy_s)
            except ValueError:
                return _error(
                    f"Unknown strategy '{strategy_s}'. "
                    f"Choose from: isolation_forest, centroid_distance, linear_boundary"
                )

            # Filter to available features
            fraud_avail = [f for f in features if f in fraud_df.columns]
            nonfr_avail = [f for f in features if f in nonfr_df.columns]
            avail = list(set(fraud_avail) & set(nonfr_avail))
            if not avail:
                return _error(
                    f"No common numeric features found between '{fraud_key}' and "
                    f"'{nonfr_key}'. Features requested: {features[:10]}"
                )
            _track_columns("mine_hard_negatives", avail)

            miner = HardNegativeMiner(
                strategy=strategy,
                top_k=int(top_k) if top_k is not None else None,
                top_k_ratio=top_k_ratio,
            )
            hn_df, result = miner.mine(fraud_df, nonfr_df, avail)

            # Optionally persist to workspace
            hn_path = None
            if save_result and self.workspace:
                hn_dir = self.workspace.samples_dir / "hard_negatives"
                hn_dir.mkdir(parents=True, exist_ok=True)
                hn_file = hn_dir / f"hard_negatives_{strategy_s}.parquet"
                hn_df.to_parquet(hn_file, index=False)
                hn_path = str(hn_file)
                log.info("Hard negatives saved: %s", hn_file)

            # Top feature importances
            top_feats = sorted(
                result.feature_importances.items(),
                key=lambda x: -x[1],
            )[:10]

            return _safe_json({
                "ok": True,
                "tool": "mine_hard_negatives",
                "strategy": result.strategy,
                "n_fraud": result.n_fraud,
                "n_nonfr_input": result.n_nonfr_input,
                "n_hard_negatives": result.n_hard_negatives,
                "inclusion_rate": round(result.inclusion_rate, 4),
                "score_threshold": round(result.score_threshold, 6),
                "boundary_density": round(result.boundary_density, 4) if result.boundary_density else None,
                "top_discriminating_features": [
                    {"feature": f, "importance": round(v, 4)} for f, v in top_feats
                ],
                "features_used": avail,
                "saved_path": hn_path,
                "interpretation": (
                    f"Found {result.n_hard_negatives:,} hard negative non-fraud accounts "
                    f"({result.inclusion_rate:.1%} of non-fraud input) that are most similar "
                    f"to the fraud cluster. These should be included in the non-fraud sample "
                    f"for model training to reduce false negatives."
                ),
            })

        except Exception:
            return _error(f"mine_hard_negatives failed:\n{traceback.format_exc()[-600:]}")

    # -- CLUSTER TYPOLOGIES ----------------------------------------------------

    def cluster_typologies(self, args: dict) -> str:
        """
        Cluster confirmed fraud cases to discover distinct typologies.

        Expected args
        -------------
        features    : list[str]   -- numeric features to cluster on
        method      : str         -- kmeans | dbscan | gmix (default: kmeans)
        n_clusters  : int | null  -- number of clusters (None = auto-select)
        fraud_key   : str         -- dataset key for fraud data (default: "fraud")
        date_col    : str | null  -- date column for temporal analysis
        label_col   : str | null  -- known fraud type column for validation
        save_report : bool        -- whether to save the report JSON
        """
        try:
            from sentinel.analysis.typology import TypologyAnalyzer, ClusteringMethod

            features   = args.get("features") or _DEFAULT_NUMERIC_FEATURES
            method_s   = args.get("method", "kmeans")
            n_clusters = args.get("n_clusters")
            fraud_key  = args.get("fraud_key", "fraud")
            date_col   = args.get("date_col")
            label_col  = args.get("label_col")
            save_rep   = bool(args.get("save_report", False))

            fraud_df = self._require_df(fraud_key)
            if isinstance(fraud_df, str):
                return fraud_df

            try:
                method = ClusteringMethod(method_s)
            except ValueError:
                return _error(
                    f"Unknown clustering method '{method_s}'. "
                    f"Choose from: kmeans, dbscan, gmix"
                )

            avail = [f for f in features if f in fraud_df.columns]
            if not avail:
                return _error(
                    f"No features found in '{fraud_key}'. "
                    f"Requested: {features[:10]}"
                )
            _track_columns("cluster_typologies", avail)

            analyzer = TypologyAnalyzer(
                method=method,
                n_clusters=int(n_clusters) if n_clusters else None,
            )
            report = analyzer.analyze(
                fraud_df=fraud_df,
                features=avail,
                date_col=date_col,
                label_col=label_col,
            )

            # Save if requested
            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "typology"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / "typology_report.json"
                report.save(rfile)
                report_path = str(rfile)
                log.info("Typology report saved: %s", rfile)

            typology_summaries = []
            for t in report.typologies:
                typology_summaries.append({
                    "cluster_id": t.cluster_id,
                    "label": t.label,
                    "n_cases": t.n_cases,
                    "pct_of_fraud": round(t.pct_of_fraud, 4),
                    "top_features": t.top_features[:5],
                    "is_growing": t.is_growing,
                    "feature_zscores_top3": {
                        f: round(t.feature_zscores.get(f, 0.0), 3)
                        for f in t.top_features[:3]
                    },
                })

            return _safe_json({
                "ok": True,
                "tool": "cluster_typologies",
                "method": report.method,
                "n_clusters": report.n_clusters,
                "n_fraud_cases": report.n_fraud_cases,
                "silhouette_score": report.silhouette_score,
                "davies_bouldin_score": report.davies_bouldin_score,
                "typologies": typology_summaries,
                "features_used": avail,
                "report_path": report_path,
                "interpretation": (
                    f"Discovered {report.n_clusters} distinct fraud typologies from "
                    f"{report.n_fraud_cases:,} confirmed fraud cases. "
                    + (f"Silhouette score {report.silhouette_score:.3f} -- "
                       + ("good separation." if (report.silhouette_score or 0) > 0.5
                          else "moderate overlap, consider reviewing cluster count.")
                       if report.silhouette_score else "")
                ),
            })

        except Exception:
            return _error(f"cluster_typologies failed:\n{traceback.format_exc()[-600:]}")

    # -- VALIDATE CONSISTENCY --------------------------------------------------

    def validate_consistency(self, args: dict) -> str:
        """
        Run cross-dataset consistency checks.

        Expected args
        -------------
        summary_key    : str  -- dataset key for summary table (default: "summary")
        raw_txn_key    : str  -- dataset key for raw transactions (default: "raw_txn")
        fraud_label_key: str  -- dataset key for fraud labels (default: "fraud_labels")
        join_key       : str  -- account ID column (default: "account_id")
        txn_count_col  : str  -- txn count column in summary (default: "total_txn_count")
        txn_amount_col : str  -- txn amount column in summary (default: "total_txn_amount")
        txn_date_col   : str  -- transaction date column in raw (default: "txn_date")
        save_report    : bool -- whether to save the report
        """
        try:
            from sentinel.validation.consistency import CrossDatasetValidator

            summary_key   = args.get("summary_key", "summary")
            raw_key       = args.get("raw_txn_key", "raw_txn")
            labels_key    = args.get("fraud_label_key", "fraud_labels")
            join_key      = args.get("join_key", "account_id")
            txn_count_col = args.get("txn_count_col", "total_txn_count")
            txn_amount_col= args.get("txn_amount_col", "total_txn_amount")
            txn_date_col  = args.get("txn_date_col", "txn_date")
            save_rep      = bool(args.get("save_report", False))

            summary_df = self._require_df(summary_key)
            if isinstance(summary_df, str):
                return summary_df

            raw_df = self._require_df(raw_key)
            if isinstance(raw_df, str):
                return raw_df

            # Fraud label IDs -- optional
            fraud_label_ids = None
            labels_df = self.datasets.get(labels_key)
            if labels_df is not None and join_key in labels_df.columns:
                fraud_label_ids = set(labels_df[join_key].tolist())

            import pandas as pd
            validator = CrossDatasetValidator()
            report = validator.validate(
                summary_df=summary_df,
                raw_df=raw_df,
                join_key=join_key,
                fraud_label_ids=(
                    pd.Series(list(fraud_label_ids)) if fraud_label_ids else None
                ),
                summary_txn_count_col=txn_count_col,
                summary_avg_amount_col=txn_amount_col,
                raw_date_col=txn_date_col,
            )

            # Save if requested
            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "consistency"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / "consistency_report.json"
                report.save(rfile)
                report_path = str(rfile)

            check_results = []
            for c in report.checks:
                check_results.append({
                    "check": c.check_name,
                    "status": c.status,
                    "message": c.message,
                    "metric": c.metric,
                    "threshold": c.threshold,
                })

            return _safe_json({
                "ok": True,
                "tool": "validate_consistency",
                "overall_status": report.overall_status,
                "n_pass": report.n_pass,
                "n_warning": report.n_warning,
                "n_fail": report.n_fail,
                "n_skipped": report.n_skipped,
                "checks": check_results,
                "report_path": report_path,
                "interpretation": (
                    f"Cross-dataset validation: {report.n_pass} PASS, "
                    f"{report.n_warning} WARNING, {report.n_fail} FAIL. "
                    + ("All checks passed -- data is consistent." if report.n_fail == 0 and report.n_warning == 0
                       else f"Review {report.n_fail} failed and {report.n_warning} warning checks above.")
                ),
            })

        except Exception:
            return _error(f"validate_consistency failed:\n{traceback.format_exc()[-600:]}")

    # -- CHECK LEAKAGE (V3-018) -------------------------------------------------

    def check_leakage(self, args: dict) -> str:
        """
        Vet a proposed feature set for target leakage (Leakage Sentinel).

        Runs as a deterministic, rule-based check -- no LLM reasoning inside
        it -- which is what gives it a genuinely separate "agent context"
        from whatever proposed the features: a hard rule can't be talked out
        of a verdict.

        Expected args
        -------------
        feature_specs   : list[dict] -- REQUIRED. Each dict:
                          {"name": str, "computed_at": "pre_event"|"post_event"|"unknown",
                           "as_of_note": str (optional),
                           "inputs": list[str] (optional -- raw columns this
                             feature was built from; catches a feature built
                             directly from the label under any name),
                           "is_group_aggregate": bool (optional -- set True
                             for a feature that aggregates the target/outcome
                             by a grouping key, e.g. a bank/IFSC fraud rate),
                           "fit_scope": "train_only"|"full_dataset"|"unknown"
                             (optional -- only checked when is_group_aggregate
                             is True; full_dataset is vetoed)}
        dev_key         : str  -- dataset key holding dev-split label+score columns
                          (optional; AUROC-gap check skipped if omitted)
        oot_key         : str  -- dataset key holding OOT-split label+score columns
        label_col       : str  -- label column name. Used both for the dev/oot
                          AUROC-gap check AND (new, V3-032) to veto any
                          feature_spec whose "inputs" contains this column --
                          a feature cannot be built from the target itself.
                          Default "label".
        score_col       : str  -- score column name in dev_key/oot_key (default: "score")
        feature_importances : dict[str, float] -- optional, drives dominant-feature check
        auroc_gap_threshold  : float (default 0.05)
        dominance_threshold  : float (default 0.5)
        lexicon         : list[str] -- optional override of the target-leak lexicon
        save_report     : bool -- whether to save the report under
                          workspace.reports_dir/leakage/
        """
        try:
            from sentinel.validation.leakage import FeatureSpec, LeakageSentinel

            raw_specs = args.get("feature_specs")
            if not raw_specs:
                return _error("feature_specs is required: list of {name, computed_at, as_of_note}")

            specs = [
                FeatureSpec(
                    name=s["name"],
                    computed_at=s.get("computed_at", "unknown"),
                    as_of_note=s.get("as_of_note", ""),
                    inputs=list(s.get("inputs", []) or []),
                    is_group_aggregate=bool(s.get("is_group_aggregate", False)),
                    fit_scope=s.get("fit_scope", "unknown"),
                )
                for s in raw_specs
            ]

            dev_key = args.get("dev_key")
            oot_key = args.get("oot_key")
            label_col = args.get("label_col", "label")
            score_col = args.get("score_col", "score")

            y_dev = scores_dev = y_oot = scores_oot = None
            if dev_key and oot_key:
                dev_df = self._require_df(dev_key)
                if isinstance(dev_df, str):
                    return dev_df
                oot_df = self._require_df(oot_key)
                if isinstance(oot_df, str):
                    return oot_df
                missing = [c for c in (label_col, score_col)
                           if c not in dev_df.columns or c not in oot_df.columns]
                if missing:
                    return _error(
                        f"label/score columns {missing} not found in both "
                        f"'{dev_key}' and '{oot_key}'"
                    )
                y_dev, scores_dev = dev_df[label_col], dev_df[score_col]
                y_oot, scores_oot = oot_df[label_col], oot_df[score_col]

            sentinel_ = LeakageSentinel(
                lexicon=args.get("lexicon"),
                auroc_gap_threshold=float(args.get("auroc_gap_threshold", 0.05)),
                dominance_threshold=float(args.get("dominance_threshold", 0.5)),
            )
            report = sentinel_.vet(
                specs,
                y_dev=y_dev, scores_dev=scores_dev,
                y_oot=y_oot, scores_oot=scores_oot,
                feature_importances=args.get("feature_importances"),
                label_column=label_col,
            )

            save_rep = bool(args.get("save_report", False))
            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "leakage"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / "leakage_report.json"
                report.save(rfile)
                report_path = str(rfile)

            return _safe_json({
                "ok": True,
                "tool": "check_leakage",
                "overall_blocked": report.overall_blocked,
                "n_clear": report.n_clear,
                "n_challenged": report.n_challenged,
                "n_vetoed": report.n_vetoed,
                "vetoed_features": report.vetoed_features(),
                "feature_verdicts": [v.to_dict() for v in report.feature_verdicts],
                "auroc_gap": report.auroc_gap,
                "report_path": report_path,
                "interpretation": (
                    f"{'BLOCKED' if report.overall_blocked else 'CLEAR'}: "
                    f"{report.n_vetoed} vetoed, {report.n_challenged} challenged, "
                    f"{report.n_clear} clear out of {report.n_features} features."
                    + (f" Vetoed: {', '.join(report.vetoed_features())}."
                       if report.vetoed_features() else "")
                ),
            })

        except Exception:
            return _error(f"check_leakage failed:\n{traceback.format_exc()[-600:]}")

    # -- TIER CHECK ------------------------------------------------------------

    def tier_check(self, args: dict) -> str:
        """
        Ask TierClassifier whether to run an operation locally or via Trino.

        Expected args
        -------------
        dataset    : str           -- canonical dataset name
        operation  : str           -- e.g. "population_profile", "hard_negative_boundary"
        n_rows     : int | null    -- estimated row count (helps the classifier)
        """
        try:
            dataset   = args.get("dataset", "")
            operation = args.get("operation", "")
            n_rows    = args.get("n_rows")

            if not dataset and not operation:
                return _error("Provide at least 'dataset' or 'operation'.")

            if self.tier_classifier is None:
                # Build a default tier classifier
                from sentinel.compute.tier import TierClassifier
                self.tier_classifier = TierClassifier(use_trino=True)

            if operation:
                decision = self.tier_classifier.classify_operation(
                    operation=operation,
                    dataset_name=dataset or None,
                    n_rows=int(n_rows) if n_rows else None,
                )
            else:
                decision = self.tier_classifier.classify_dataset(
                    name=dataset,
                    n_rows=int(n_rows) if n_rows else None,
                )

            return _safe_json({
                "ok": True,
                "tool": "tier_check",
                "dataset": dataset,
                "operation": operation,
                "tier": decision.tier.value,
                "reason": decision.reason,
                "local_feasible": decision.local_feasible,
                "warning": decision.warning,
                "recommendation": (
                    "Run this operation locally using in-memory DataFrames."
                    if decision.tier.value == "LOCAL"
                    else "This operation requires Trino. Use submit_trino_job with HITL."
                ),
            })

        except Exception:
            return _error(f"tier_check failed:\n{traceback.format_exc()[-600:]}")

    # -- MEASURE SIGNAL DECAY -------------------------------------------------

    def measure_signal_decay(self, args: dict) -> str:
        """
        Measure feature discriminative power and detect decay vs historical baseline.

        Expected args
        -------------
        features        : list[str]  -- feature columns to analyze
        period          : str        -- label for this snapshot (e.g. "2024-Q2")
        fraud_key       : str        -- dataset key for fraud data (default: "fraud")
        nonfr_key       : str        -- dataset key for non-fraud (default: "nonfr")
        save_history    : bool       -- persist to workspace feature_signal_history (default: True)
        deposit_to_memory: bool      -- deposit decay events to DatasetKnowledge (default: True)
        dataset_name    : str        -- dataset name for knowledge entries (default: "default")
        top_n           : int        -- how many top features to return in summary (default: 15)
        """
        try:
            from sentinel.analysis.signal_decay import SignalDecayTracker

            features    = args.get("features") or _DEFAULT_NUMERIC_FEATURES
            period      = args.get("period", "current")
            fraud_key   = args.get("fraud_key", "fraud")
            nonfr_key   = args.get("nonfr_key", "nonfr")
            save_hist   = bool(args.get("save_history", True))
            deposit_mem = bool(args.get("deposit_to_memory", True))
            dataset_name= args.get("dataset_name", "default")
            top_n       = int(args.get("top_n", 15))
            with_graph_evidence = bool(args.get("with_graph_evidence", False))
            raw_txn_key = args.get("raw_txn_key", "raw_txn")

            fraud_df = self._require_df(fraud_key)
            if isinstance(fraud_df, str):
                return fraud_df

            nonfr_df = self._require_df(nonfr_key)
            if isinstance(nonfr_df, str):
                return nonfr_df

            # Resolve DatasetKnowledge for the target dataset
            dk = None
            if deposit_mem:
                dk = self.knowledge
                if dk is None and self.workspace is not None:
                    try:
                        from sentinel.memory.knowledge import DatasetKnowledge
                        dk = DatasetKnowledge.load(self.workspace, dataset_name)
                    except Exception:
                        pass

            _track_columns("measure_signal_decay",
                           [f for f in features if f in fraud_df.columns])

            tracker = SignalDecayTracker()
            result = tracker.track(
                fraud_df=fraud_df,
                nonfr_df=nonfr_df,
                features=features,
                period=period,
                workspace=self.workspace if save_hist else None,
                dataset_knowledge=dk,
                session_id=self.session_id,
            )

            # Build compact output
            top_rows = result.summary_table()[:top_n]
            decay_summaries = [
                {
                    "feature":        ev.feature,
                    "severity":       ev.severity,
                    "metric":         ev.primary_metric,
                    "baseline_value": round(ev.baseline_value, 4),
                    "current_value":  round(ev.current_value, 4),
                    "pct_change":     round(ev.pct_change, 2),
                    "baseline_period": ev.baseline_period,
                }
                for ev in result.decaying_features()
            ]

            interpretation = (
                f"Measured signal strength for {result.n_features} features in period '{period}'. "
            )
            if result.baseline_period:
                interpretation += (
                    f"Compared to baseline '{result.baseline_period}': "
                    f"{result.n_stable} stable, {result.n_warning} WARNING, "
                    f"{result.n_fail} FAIL, {result.n_improved} improved."
                )
            else:
                interpretation += (
                    "No baseline yet -- this snapshot is now recorded as the baseline. "
                    "Run again in a future period to detect decay."
                )

            # Optional graph evidence -- when decay events fire, attach a
            # quick fingerprint summary on raw_txn so deposits can be richer.
            graph_evidence = None
            if (
                with_graph_evidence
                and decay_summaries
                and raw_txn_key in self.datasets
            ):
                try:
                    from sentinel.graph.fingerprint import BehavioralFingerprinter
                    fp_er = BehavioralFingerprinter(min_txns=3)
                    fp_res = fp_er.fingerprint_all(
                        self.datasets[raw_txn_key], period=period,
                    )
                    graph_evidence = {
                        "n_accounts_fingerprinted": fp_res.n_accounts,
                        "n_alerts": fp_res.n_alerts,
                        "n_warnings": fp_res.n_warnings,
                        "top_5_mule_likelihood": [
                            {"account_id": f.account_id,
                             "mule_likelihood": round(f.mule_likelihood, 3),
                             "severity": f.severity}
                            for f in fp_res.top_by_mule_likelihood(5)
                        ],
                        "note": (
                            "Behavioural-feature decay corroborated by "
                            f"{fp_res.n_alerts} ALERT fingerprints."
                            if fp_res.n_alerts > 0
                            else "No fingerprint ALERTs -- decay may be label noise."
                        ),
                    }
                except Exception as exc:
                    graph_evidence = {"error": str(exc)[:200]}

            return _safe_json({
                "ok": True,
                "tool": "measure_signal_decay",
                "period": period,
                "baseline_period": result.baseline_period,
                "n_features": result.n_features,
                "n_stable": result.n_stable,
                "n_warning": result.n_warning,
                "n_fail": result.n_fail,
                "n_improved": result.n_improved,
                "top_features": top_rows,
                "decay_events": decay_summaries,
                "history_saved": save_hist and self.workspace is not None,
                "graph_evidence": graph_evidence,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"measure_signal_decay failed:\n{traceback.format_exc()[-600:]}")

    # -- TRACK TYPOLOGY EVOLUTION ----------------------------------------------

    def track_typology_evolution(self, args: dict) -> str:
        """
        Cluster fraud data for the current period and compare to historical
        cluster identities to detect growing, declining, or new attack typologies.

        Expected args
        -------------
        features        : list[str]  -- numeric feature columns to cluster on
        period          : str        -- period label (e.g. "2024-Q2")
        fraud_key       : str        -- dataset key for fraud data (default: "fraud")
        n_clusters      : int | null -- fixed cluster count (None = auto)
        match_threshold : float      -- cosine similarity threshold for identity
                                       matching across periods (default: 0.60)
        save_history    : bool       -- persist evolution history to workspace
        deposit_to_memory: bool      -- deposit alert events to DatasetKnowledge
        dataset_name    : str        -- dataset name for knowledge entries
        top_n_alerts    : int        -- number of top alerts to include in response
        """
        try:
            from sentinel.analysis.typology_evolution import TypologyEvolutionTracker

            features         = args.get("features") or _DEFAULT_NUMERIC_FEATURES
            period           = args.get("period", "current")
            fraud_key        = args.get("fraud_key", "fraud")
            n_clusters       = args.get("n_clusters")
            match_threshold  = float(args.get("match_threshold", 0.60))
            save_hist        = bool(args.get("save_history", True))
            deposit_mem      = bool(args.get("deposit_to_memory", True))
            dataset_name     = args.get("dataset_name", "default")
            top_n_alerts     = int(args.get("top_n_alerts", 5))
            with_graph_evidence = bool(args.get("with_graph_evidence", False))
            raw_txn_key      = args.get("raw_txn_key", "raw_txn")

            fraud_df = self._require_df(fraud_key)
            if isinstance(fraud_df, str):
                return fraud_df

            # Resolve DatasetKnowledge
            dk = None
            if deposit_mem:
                dk = self.knowledge
                if dk is None and self.workspace is not None:
                    try:
                        from sentinel.memory.knowledge import DatasetKnowledge
                        dk = DatasetKnowledge.load(self.workspace, dataset_name)
                    except Exception:
                        pass

            _track_columns("track_typology_evolution",
                           [f for f in features if f in fraud_df.columns])

            tracker = TypologyEvolutionTracker(
                n_clusters=int(n_clusters) if n_clusters else None,
                match_threshold=match_threshold,
            )
            result = tracker.track(
                fraud_df=fraud_df,
                features=features,
                period=period,
                workspace=self.workspace if save_hist else None,
                dataset_knowledge=dk,
                session_id=self.session_id,
            )

            # Compact trajectory summaries
            traj_summaries = []
            for t in result.trajectories:
                traj_summaries.append({
                    "cluster_key":       t.cluster_key,
                    "label":             t.latest_label,
                    "n_cases":           t.latest_n_cases,
                    "pct_of_fraud":      round(t.latest_pct_of_fraud, 4),
                    "trend":             t.trend,
                    "avg_growth_rate":   round(t.avg_growth_rate, 4),
                    "growth_rates":      [round(r, 4) for r in t.growth_rates],
                    "alert_level":       t.alert_level,
                    "alert_reason":      t.alert_reason,
                    "centroid_drift":    t.centroid_drift,
                    "n_periods_tracked": len(t.snapshots),
                })

            new_cluster_summaries = [
                {
                    "cluster_key":  nc.cluster_key,
                    "label":        nc.label,
                    "n_cases":      nc.n_cases,
                    "pct_of_fraud": round(nc.pct_of_fraud, 4),
                    "top_features": nc.top_features[:4],
                }
                for nc in result.new_clusters
            ]

            # Top alerts only
            top_alerts = [
                {
                    "cluster_key":  t.cluster_key,
                    "alert_level":  t.alert_level,
                    "alert_reason": t.alert_reason,
                    "trend":        t.trend,
                    "last_growth":  round(t.growth_rates[-1], 4) if t.growth_rates else None,
                }
                for t in result.top_alerts()[:top_n_alerts]
            ]

            # Build interpretation
            interpretation = (
                f"Typology evolution for period '{period}': "
                f"{result.n_clusters_current} active clusters from "
                f"{result.n_fraud_cases:,} fraud cases. "
            )
            if result.n_emerging:
                interpretation += (
                    f"{result.n_emerging} NEW cluster(s) with no historical match -- "
                    f"potential emerging attack vectors. "
                )
            if result.n_growing:
                interpretation += f"{result.n_growing} growing cluster(s). "
            if result.n_declining:
                interpretation += f"{result.n_declining} declining cluster(s). "
            if result.disappeared_clusters:
                interpretation += (
                    f"{len(result.disappeared_clusters)} cluster(s) disappeared "
                    f"(neutralised?): {result.disappeared_clusters}. "
                )
            if result.overall_alert == "NONE":
                interpretation += "No alerts -- typology landscape is stable."
            else:
                interpretation += f"Overall alert: {result.overall_alert}."

            # Optional graph evidence -- when emerging/alert clusters appear,
            # corroborate with motif + ring detection on raw_txn.
            graph_evidence = None
            if (
                with_graph_evidence
                and (result.n_emerging > 0 or result.overall_alert in ("WARNING", "FAIL"))
                and raw_txn_key in self.datasets
            ):
                try:
                    from sentinel.graph.builder import AccountGraphBuilder
                    from sentinel.graph.motifs import MotifDetector
                    from sentinel.graph.community import CommunityFinder

                    builder = AccountGraphBuilder()
                    txn_graph = builder.build(self.datasets[raw_txn_key], period=period)
                    motif_res = MotifDetector().detect(txn_graph)
                    ring_res = CommunityFinder().find(txn_graph)
                    graph_evidence = {
                        "n_motif_alerts":  motif_res.n_alerts,
                        "n_motif_warnings": motif_res.n_warnings,
                        "matches_by_pattern": motif_res.matches_by_pattern,
                        "n_ring_alerts":   len(ring_res.alerts()),
                        "n_communities_kept": ring_res.n_communities_kept,
                        "modularity":     (round(ring_res.modularity, 4)
                                           if ring_res.modularity is not None else None),
                        "note": (
                            "Emerging cluster CORROBORATED by graph evidence -- "
                            f"{motif_res.n_alerts} motif alerts, "
                            f"{len(ring_res.alerts())} ring alerts."
                            if motif_res.n_alerts + len(ring_res.alerts()) > 0
                            else "No corroborating graph alerts -- emerging "
                                 "cluster may be a sampling artifact, lower confidence."
                        ),
                    }
                except Exception as exc:
                    graph_evidence = {"error": str(exc)[:200]}

            return _safe_json({
                "ok": True,
                "tool": "track_typology_evolution",
                "period": period,
                "n_fraud_cases": result.n_fraud_cases,
                "n_clusters_current": result.n_clusters_current,
                "n_clusters_historical": result.n_clusters_historical,
                "features_used": result.features_used,
                "overall_alert": result.overall_alert,
                "n_growing": result.n_growing,
                "n_declining": result.n_declining,
                "n_stable": result.n_stable,
                "n_emerging": result.n_emerging,
                "disappeared_clusters": result.disappeared_clusters,
                "trajectories": traj_summaries,
                "new_clusters": new_cluster_summaries,
                "top_alerts": top_alerts,
                "history_saved": save_hist and self.workspace is not None,
                "graph_evidence": graph_evidence,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"track_typology_evolution failed:\n{traceback.format_exc()[-600:]}")

    # -- DEPOSIT INSIGHT -------------------------------------------------------

    def deposit_insight(self, args: dict) -> str:
        """
        Write a structured insight to the DatasetKnowledge store.

        Expected args
        -------------
        dataset     : str   -- dataset name this insight belongs to
        entry_type  : str   -- knowledge type constant (e.g. "quality_flag",
                              "drift_event", "signal_decay", "analyst_note")
        feature     : str   -- feature this insight is about (or "" for global)
        claim       : str   -- the insight claim (short, <=120 chars)
        detail      : str   -- extended explanation
        confidence  : float -- 0.0-1.0 (default: 0.8)
        period      : str   -- time period this applies to (e.g. "2024-Q1")
        evidence_ref: str   -- reference to evidence (e.g. report path, PSI result)
        tags        : list  -- free-form tags
        """
        try:
            from sentinel.memory.knowledge import DatasetKnowledge, KnowledgeEntry, KnowledgeType

            dataset     = args.get("dataset", "default")
            entry_type  = args.get("entry_type", KnowledgeType.ANALYST_NOTE)
            feature     = args.get("feature", "")
            claim       = args.get("claim", "")
            detail      = args.get("detail", "")
            confidence  = float(args.get("confidence", 0.8))
            period      = args.get("period", "")
            evidence_ref= args.get("evidence_ref", "")
            tags        = list(args.get("tags") or [])

            if not claim:
                return _error("'claim' is required and must be a non-empty string.")

            # Validate entry type
            valid_types = {
                v for k, v in vars(KnowledgeType).items()
                if not k.startswith("_")
            }
            if entry_type not in valid_types:
                return _error(
                    f"Unknown entry_type '{entry_type}'. "
                    f"Valid types: {sorted(valid_types)}"
                )

            # Get or create DatasetKnowledge
            dk = self.knowledge
            if dk is None or getattr(dk, "dataset_name", None) != dataset:
                if self.workspace is not None:
                    dk = DatasetKnowledge.load(self.workspace, dataset)
                else:
                    return _error(
                        "No workspace configured. Cannot persist knowledge entry. "
                        "Pass workspace= to Phase2Tools constructor."
                    )

            entry = KnowledgeEntry(
                session_id=self.session_id,
                entry_type=entry_type,
                feature=feature,
                claim=claim,
                detail=detail,
                confidence=confidence,
                period=period,
                evidence_ref=evidence_ref,
                tags=tags,
            )
            dk.add(entry)

            return _safe_json({
                "ok": True,
                "tool": "deposit_insight",
                "entry_id": entry.entry_id,
                "dataset": dataset,
                "entry_type": entry_type,
                "feature": feature,
                "claim": claim,
                "confidence": confidence,
                "message": (
                    f"Insight recorded as {entry.entry_id}. It will be available "
                    f"in future sessions as context for '{dataset}'."
                ),
            })

        except Exception:
            return _error(f"deposit_insight failed:\n{traceback.format_exc()[-600:]}")

    # -- DETECT MULE MOTIFS ---------------------------------------------------

    def detect_mule_motifs(self, args: dict) -> str:
        """
        Build a transaction graph and run the 6-pattern mule motif catalogue.

        Patterns: fan_in_collector, fan_out_distributor, layering_chain,
        cyclic_ring, velocity_hub, pass_through.

        Expected args
        -------------
        raw_txn_key       : str       -- dataset key for txn table (default: "raw_txn")
        period            : str       -- period label (default: "current")
        payer_col         : str       -- default: "account_id"
        payee_col         : str       -- default: "beneficiary_id"
        amount_col        : str       -- default: "txn_amount"
        timestamp_col     : str       -- default: "txn_date"
        status_col        : str       -- default: "status" (None to disable filter)
        success_only      : bool      -- default: True
        patterns          : list[str] -- subset of patterns; default = all
        save_report       : bool      -- default: False
        deposit_to_memory : bool      -- deposit ALERT motifs to DatasetKnowledge
        dataset_name      : str       -- for knowledge entries (default "default")
        top_n             : int       -- top matches to include in response (default 25)
        """
        try:
            from sentinel.graph.builder import AccountGraphBuilder
            from sentinel.graph.motifs import (
                MotifDetector, FanInCollector, FanOutDistributor,
                LayeringChain, CyclicRing, VelocityHub, PassThrough,
            )

            raw_key       = args.get("raw_txn_key", "raw_txn")
            period        = args.get("period", "current")
            payer_col     = args.get("payer_col", "account_id")
            payee_col     = args.get("payee_col", "beneficiary_id")
            amount_col    = args.get("amount_col", "txn_amount")
            timestamp_col = args.get("timestamp_col", "txn_date")
            status_col    = args.get("status_col", "status")
            success_only  = bool(args.get("success_only", True))
            patterns      = args.get("patterns") or None
            save_rep      = bool(args.get("save_report", False))
            deposit_mem   = bool(args.get("deposit_to_memory", True))
            dataset_name  = args.get("dataset_name", "default")
            top_n         = int(args.get("top_n", 25))

            txn_df = self._require_df(raw_key)
            if isinstance(txn_df, str):
                return txn_df

            _track_columns("detect_mule_motifs",
                           [payer_col, payee_col, amount_col, timestamp_col, status_col])

            builder = AccountGraphBuilder(
                payer_col=payer_col,
                payee_col=payee_col,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                status_col=status_col,
                success_values=("SUCCESS",) if success_only else (),
            )
            graph = builder.build(txn_df, period=period)

            # Subset patterns if requested
            if patterns:
                pat_map = {
                    "fan_in_collector":     FanInCollector,
                    "fan_out_distributor":  FanOutDistributor,
                    "layering_chain":       LayeringChain,
                    "cyclic_ring":          CyclicRing,
                    "velocity_hub":         VelocityHub,
                    "pass_through":         PassThrough,
                }
                use = []
                for name in patterns:
                    if name in pat_map:
                        use.append(pat_map[name]())
                if not use:
                    return _error(
                        f"No valid pattern names in {patterns}. "
                        f"Choose from: {sorted(pat_map.keys())}"
                    )
                detector = MotifDetector(patterns=use)
            else:
                detector = MotifDetector()

            result = detector.detect(graph)

            # Save if requested
            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "graph"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / f"motifs_{period}.json"
                result.save(rfile)
                report_path = str(rfile)

            # Auto-deposit ALERT motifs to memory
            entries_deposited = 0
            if deposit_mem and result.alerts():
                dk = self.knowledge
                if dk is None and self.workspace is not None:
                    try:
                        from sentinel.memory.knowledge import (
                            DatasetKnowledge, KnowledgeEntry, KnowledgeType,
                        )
                        dk = DatasetKnowledge.load(self.workspace, dataset_name)
                    except Exception:
                        dk = None

                if dk is not None:
                    from sentinel.memory.knowledge import KnowledgeEntry, KnowledgeType
                    for m in result.alerts()[:10]:
                        try:
                            entry = KnowledgeEntry(
                                session_id=self.session_id,
                                entry_type=KnowledgeType.ANOMALY,
                                feature="",
                                claim=(
                                    f"Mule motif {m.pattern_name} [ALERT] "
                                    f"score={m.score:.2f} -- {m.metadata.get('highlight', '')[:80]}"
                                ),
                                detail=(
                                    f"Pattern: {m.pattern_name}. Nodes: {len(m.nodes)}. "
                                    f"Txns: {m.n_transactions}. "
                                    f"Total Rs : {m.total_amount:,.0f}. "
                                    f"Window: {m.time_window_start} -> {m.time_window_end}. "
                                    f"Metadata: {m.metadata}"
                                ),
                                confidence=min(0.95, 0.60 + 0.35 * m.score),
                                period=period,
                                tags=["graph", "motif", m.pattern_name, "alert"],
                            )
                            dk.add(entry)
                            entries_deposited += 1
                        except Exception:
                            pass

            top_matches = [m.to_dict() for m in result.top_by_score(top_n)]
            interpretation = (
                f"Built account graph from {result.n_transactions:,} txns "
                f"({result.n_nodes:,} nodes / {result.n_edges:,} edges). "
                f"Found {len(result.matches)} motif match(es): "
                f"{result.n_alerts} ALERT, {result.n_warnings} WARNING, "
                f"{result.n_info} INFO. "
            )
            if result.n_alerts:
                interpretation += (
                    f"! {result.n_alerts} ALERT-level motifs require analyst review. "
                )
            if entries_deposited:
                interpretation += f"Deposited {entries_deposited} ALERT motifs to memory."

            return _safe_json({
                "ok": True,
                "tool": "detect_mule_motifs",
                "period": period,
                "n_nodes": result.n_nodes,
                "n_edges": result.n_edges,
                "n_transactions": result.n_transactions,
                "total_amount": round(result.total_amount, 2),
                "n_matches": len(result.matches),
                "n_alerts": result.n_alerts,
                "n_warnings": result.n_warnings,
                "n_info": result.n_info,
                "matches_by_pattern": result.matches_by_pattern,
                "top_matches": top_matches,
                "report_path": report_path,
                "entries_deposited": entries_deposited,
                "elapsed_ms": result.elapsed_ms,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"detect_mule_motifs failed:\n{traceback.format_exc()[-600:]}")

    # -- COMPUTE ACCOUNT CENTRALITY -------------------------------------------

    def compute_account_centrality(self, args: dict) -> str:
        """
        Compute centrality measures (PageRank, betweenness, k-core) for every
        account in the transaction graph. Identifies money-flow conduits and
        pass-through mule candidates.

        Expected args
        -------------
        raw_txn_key      : str        -- dataset key for txn table (default: "raw_txn")
        period           : str        -- period label (default: "current")
        payer_col, payee_col, amount_col, timestamp_col, status_col, success_only
            -- same as detect_mule_motifs
        measures         : list[str]  -- subset of {pagerank, betweenness, eigenvector, k_core}
        betweenness_k    : int | null -- k-sample size for approximate betweenness
        save_report      : bool       -- default: False
        top_n            : int        -- top accounts to return (default 25)
        """
        try:
            from sentinel.graph.builder import AccountGraphBuilder
            from sentinel.graph.centrality import CentralityComputer

            raw_key       = args.get("raw_txn_key", "raw_txn")
            period        = args.get("period", "current")
            payer_col     = args.get("payer_col", "account_id")
            payee_col     = args.get("payee_col", "beneficiary_id")
            amount_col    = args.get("amount_col", "txn_amount")
            timestamp_col = args.get("timestamp_col", "txn_date")
            status_col    = args.get("status_col", "status")
            success_only  = bool(args.get("success_only", True))
            measures      = tuple(args.get("measures") or ("pagerank", "betweenness", "k_core"))
            bk            = args.get("betweenness_k")
            save_rep      = bool(args.get("save_report", False))
            top_n         = int(args.get("top_n", 25))

            txn_df = self._require_df(raw_key)
            if isinstance(txn_df, str):
                return txn_df

            _track_columns("compute_account_centrality",
                           [payer_col, payee_col, amount_col, timestamp_col, status_col])

            builder = AccountGraphBuilder(
                payer_col=payer_col,
                payee_col=payee_col,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                status_col=status_col,
                success_values=("SUCCESS",) if success_only else (),
            )
            graph = builder.build(txn_df, period=period)

            comp = CentralityComputer(
                measures=measures,
                betweenness_k=int(bk) if bk else None,
            )
            result = comp.compute(graph)

            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "graph"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / f"centrality_{period}.json"
                result.save(rfile)
                report_path = str(rfile)

            top_pr = [a.to_dict() for a in result.top_pagerank(top_n)]
            top_bt = [a.to_dict() for a in result.top_betweenness(top_n)]
            passthroughs = [a.to_dict() for a in result.passthrough_candidates()[:top_n]]

            interpretation = (
                f"Computed {', '.join(result.measures_computed)} for "
                f"{result.n_nodes:,} nodes / {result.n_edges:,} edges. "
                f"Found {len(passthroughs)} pass-through mule candidates "
                "(in~out, >=3 sources)."
            )

            return _safe_json({
                "ok": True,
                "tool": "compute_account_centrality",
                "period": period,
                "n_nodes": result.n_nodes,
                "n_edges": result.n_edges,
                "measures_computed": result.measures_computed,
                "top_pagerank": top_pr,
                "top_betweenness": top_bt,
                "passthrough_candidates": passthroughs,
                "n_passthrough_candidates": len(passthroughs),
                "elapsed_ms": result.elapsed_ms,
                "report_path": report_path,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"compute_account_centrality failed:\n{traceback.format_exc()[-600:]}")

    # -- FIND MULE RINGS ------------------------------------------------------

    def find_mule_rings(self, args: dict) -> str:
        """
        Run Louvain community detection on the account graph and identify
        candidate mule rings -- small, dense, internally-balanced communities.

        Expected args
        -------------
        raw_txn_key       : str    -- txn dataset key (default "raw_txn")
        period            : str    -- period label
        payer_col, payee_col, amount_col, timestamp_col, status_col, success_only
            -- same as detect_mule_motifs
        min_size, max_size       : int   -- community size bounds (default 3, 50)
        min_density              : float -- directed density threshold (default 0.10)
        min_internal_flow_ratio  : float -- internal-edge fraction (default 0.30)
        min_mule_score           : float -- composite score threshold (default 0.30)
        resolution               : float -- Louvain resolution (default 1.0)
        save_report              : bool
        deposit_to_memory        : bool  -- auto-deposit ALERT rings to knowledge
        dataset_name             : str
        top_n                    : int   -- communities to return (default 25)
        """
        try:
            from sentinel.graph.builder import AccountGraphBuilder
            from sentinel.graph.community import CommunityFinder

            raw_key       = args.get("raw_txn_key", "raw_txn")
            period        = args.get("period", "current")
            payer_col     = args.get("payer_col", "account_id")
            payee_col     = args.get("payee_col", "beneficiary_id")
            amount_col    = args.get("amount_col", "txn_amount")
            timestamp_col = args.get("timestamp_col", "txn_date")
            status_col    = args.get("status_col", "status")
            success_only  = bool(args.get("success_only", True))

            min_size      = int(args.get("min_size", 3))
            max_size      = int(args.get("max_size", 50))
            min_density   = float(args.get("min_density", 0.10))
            min_flow      = float(args.get("min_internal_flow_ratio", 0.30))
            min_score     = float(args.get("min_mule_score", 0.30))
            resolution    = float(args.get("resolution", 1.0))

            save_rep      = bool(args.get("save_report", False))
            deposit_mem   = bool(args.get("deposit_to_memory", True))
            dataset_name  = args.get("dataset_name", "default")
            top_n         = int(args.get("top_n", 25))

            txn_df = self._require_df(raw_key)
            if isinstance(txn_df, str):
                return txn_df

            builder = AccountGraphBuilder(
                payer_col=payer_col,
                payee_col=payee_col,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                status_col=status_col,
                success_values=("SUCCESS",) if success_only else (),
            )
            graph = builder.build(txn_df, period=period)

            finder = CommunityFinder(
                min_size=min_size,
                max_size=max_size,
                min_density=min_density,
                min_internal_flow_ratio=min_flow,
                min_mule_score=min_score,
                resolution=resolution,
            )
            result = finder.find(graph)

            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "graph"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / f"mule_rings_{period}.json"
                result.save(rfile)
                report_path = str(rfile)

            # Auto-deposit ALERT rings
            entries_deposited = 0
            if deposit_mem and result.alerts():
                dk = self.knowledge
                if dk is None and self.workspace is not None:
                    try:
                        from sentinel.memory.knowledge import DatasetKnowledge
                        dk = DatasetKnowledge.load(self.workspace, dataset_name)
                    except Exception:
                        dk = None
                if dk is not None:
                    from sentinel.memory.knowledge import KnowledgeEntry, KnowledgeType
                    for com in result.alerts()[:10]:
                        try:
                            entry = KnowledgeEntry(
                                session_id=self.session_id,
                                entry_type=KnowledgeType.ANOMALY,
                                feature="",
                                claim=(
                                    f"Mule ring [ALERT] community_id={com.community_id} "
                                    f"score={com.mule_score:.2f} -- {com.label}"
                                ),
                                detail=(
                                    f"Nodes ({com.n_nodes}): {com.nodes[:10]}"
                                    f"{'...' if len(com.nodes) > 10 else ''}. "
                                    f"Density: {com.density:.3f}. "
                                    f"Internal Rs : {com.total_internal_amount:,.0f}. "
                                    f"Internal flow ratio: {com.internal_flow_ratio:.0%}. "
                                    f"Top nodes by PageRank: {com.top_nodes_by_pagerank}"
                                ),
                                confidence=min(0.95, 0.60 + 0.35 * com.mule_score),
                                period=period,
                                tags=["graph", "community", "mule_ring", "alert"],
                            )
                            dk.add(entry)
                            entries_deposited += 1
                        except Exception:
                            pass

            top_coms = [c.to_dict() for c in result.top_by_score(top_n)]
            interpretation = (
                f"Built account graph from {graph.n_transactions:,} txns "
                f"({graph.n_nodes:,} nodes). Louvain found "
                f"{result.n_communities_total} total communities; "
                f"{result.n_communities_kept} passed mule-ring filters. "
            )
            if result.alerts():
                interpretation += (
                    f"! {len(result.alerts())} ALERT-level rings need analyst review. "
                )
            if entries_deposited:
                interpretation += f"Deposited {entries_deposited} alert rings to memory."

            return _safe_json({
                "ok": True,
                "tool": "find_mule_rings",
                "period": period,
                "n_nodes": result.n_nodes,
                "n_edges": result.n_edges,
                "n_communities_total": result.n_communities_total,
                "n_communities_kept": result.n_communities_kept,
                "modularity": (round(result.modularity, 4)
                               if result.modularity is not None else None),
                "n_alerts": len(result.alerts()),
                "n_warnings": len(result.warnings()),
                "top_communities": top_coms,
                "report_path": report_path,
                "entries_deposited": entries_deposited,
                "elapsed_ms": result.elapsed_ms,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"find_mule_rings failed:\n{traceback.format_exc()[-600:]}")

    # -- TRACE MONEY FLOW -----------------------------------------------------

    def trace_money_flow(self, args: dict) -> str:
        """
        Trace money flow downstream from a source, upstream from a sink, or
        find the k-shortest paths from one account to another.

        Expected args
        -------------
        raw_txn_key       : str    -- txn dataset key
        period            : str
        source            : str    -- account_id to trace from (downstream/path)
        sink              : str    -- account_id to trace to (upstream/path)
                                     If both source and sink are set, returns
                                     k-shortest source->sink paths.
                                     If only source: downstream BFS.
                                     If only sink: upstream BFS.
        max_hops          : int    -- BFS depth cap (default 6)
        max_paths         : int    -- k-shortest paths cap (default 20)
        min_amount        : float  -- drop edges below this (default 0)
        enforce_time_order: bool   -- only follow ts > parent (default True)
        save_report       : bool
        """
        try:
            from sentinel.graph.builder import AccountGraphBuilder
            from sentinel.graph.flow import MoneyFlowTracer

            raw_key       = args.get("raw_txn_key", "raw_txn")
            period        = args.get("period", "current")
            payer_col     = args.get("payer_col", "account_id")
            payee_col     = args.get("payee_col", "beneficiary_id")
            amount_col    = args.get("amount_col", "txn_amount")
            timestamp_col = args.get("timestamp_col", "txn_date")
            status_col    = args.get("status_col", "status")
            success_only  = bool(args.get("success_only", True))
            source        = args.get("source")
            sink          = args.get("sink")
            max_hops      = int(args.get("max_hops", 6))
            max_paths     = int(args.get("max_paths", 20))
            min_amount    = float(args.get("min_amount", 0.0))
            enforce_time  = bool(args.get("enforce_time_order", True))
            save_rep      = bool(args.get("save_report", False))

            if not source and not sink:
                return _error("trace_money_flow: provide at least 'source' or 'sink'.")

            txn_df = self._require_df(raw_key)
            if isinstance(txn_df, str):
                return txn_df

            builder = AccountGraphBuilder(
                payer_col=payer_col,
                payee_col=payee_col,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                status_col=status_col,
                success_values=("SUCCESS",) if success_only else (),
            )
            graph = builder.build(txn_df, period=period)

            tracer = MoneyFlowTracer(
                max_hops=max_hops,
                max_paths=max_paths,
                min_amount=min_amount,
                enforce_time_order=enforce_time,
            )

            if source and sink:
                result = tracer.shortest_paths(graph, source, sink)
            elif source:
                result = tracer.trace_downstream(graph, source, max_hops=max_hops)
            else:
                result = tracer.trace_upstream(graph, sink, max_hops=max_hops)

            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "graph"
                rdir.mkdir(parents=True, exist_ok=True)
                tag = source or sink
                rfile = rdir / f"flow_{result.direction}_{tag}_{period}.json"
                result.save(rfile)
                report_path = str(rfile)

            interpretation = (
                f"Trace [{result.direction}] from "
                f"{result.source}{' to ' + result.sink if result.sink else ''} "
                f"in period {period}: "
                f"{result.n_reachable} reachable accounts, "
                f"{len(result.paths)} paths returned, "
                f"Rs {result.total_amount_traced:,.0f} traced."
            )
            if result.notes:
                interpretation += f" Note: {result.notes}"

            paths_dict = [p.to_dict() for p in result.paths]
            nodes_by_distance = {
                str(k): v[:50] for k, v in result.nodes_by_distance.items()
            }

            return _safe_json({
                "ok": True,
                "tool": "trace_money_flow",
                "direction": result.direction,
                "source": result.source,
                "sink": result.sink,
                "period": period,
                "n_reachable": result.n_reachable,
                "nodes_by_distance": nodes_by_distance,
                "n_paths": len(result.paths),
                "paths": paths_dict,
                "total_amount_traced": round(result.total_amount_traced, 2),
                "report_path": report_path,
                "elapsed_ms": result.elapsed_ms,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"trace_money_flow failed:\n{traceback.format_exc()[-600:]}")

    # -- FINGERPRINT ACCOUNTS -------------------------------------------------

    def fingerprint_accounts(self, args: dict) -> str:
        """
        Compute per-account behavioral fingerprints and rank by mule likelihood.

        Captures: time-of-day skew, inter-arrival CV, burst rate, round-amount
        ratio, beneficiary diversity, device diversity. No graph required.

        Expected args
        -------------
        raw_txn_key       : str
        period            : str
        account_col, beneficiary_col, amount_col, timestamp_col, device_col,
        status_col        : column overrides
        success_only      : bool -- drop non-SUCCESS txns (default True)
        min_txns          : int  -- drop accounts with fewer txns (default 5)
        focus_account     : str  -- fingerprint a single account if given
        save_report       : bool
        deposit_to_memory : bool -- deposit ALERT fingerprints
        dataset_name      : str
        top_n             : int  -- top fingerprints to return
        """
        try:
            from sentinel.graph.fingerprint import BehavioralFingerprinter

            raw_key         = args.get("raw_txn_key", "raw_txn")
            period          = args.get("period", "current")
            account_col     = args.get("account_col", "account_id")
            beneficiary_col = args.get("beneficiary_col", "beneficiary_id")
            amount_col      = args.get("amount_col", "txn_amount")
            timestamp_col   = args.get("timestamp_col", "txn_date")
            device_col      = args.get("device_col", "device_id")
            status_col      = args.get("status_col", "status")
            success_only    = bool(args.get("success_only", True))
            min_txns        = int(args.get("min_txns", 5))
            focus           = args.get("focus_account")
            save_rep        = bool(args.get("save_report", False))
            deposit_mem     = bool(args.get("deposit_to_memory", True))
            dataset_name    = args.get("dataset_name", "default")
            top_n           = int(args.get("top_n", 25))

            txn_df = self._require_df(raw_key)
            if isinstance(txn_df, str):
                return txn_df

            fp_er = BehavioralFingerprinter(
                account_col=account_col,
                beneficiary_col=beneficiary_col,
                amount_col=amount_col,
                timestamp_col=timestamp_col,
                device_col=device_col,
                status_col=status_col,
                success_only=success_only,
                min_txns=min_txns,
            )

            if focus:
                fp = fp_er.fingerprint_account(txn_df, focus)
                if fp is None:
                    return _error(
                        f"Account '{focus}' has fewer than {min_txns} txns "
                        f"or doesn't exist."
                    )
                return _safe_json({
                    "ok": True,
                    "tool": "fingerprint_accounts",
                    "mode": "focus",
                    "period": period,
                    "fingerprint": fp.to_dict(),
                    "interpretation": (
                        f"Fingerprint for account {focus}: "
                        f"mule_likelihood={fp.mule_likelihood:.2f} "
                        f"({fp.severity}). {fp.n_txns} txns, "
                        f"night_ratio={fp.night_ratio:.0%}, "
                        f"median_gap={fp.median_gap_minutes:.1f} min, "
                        f"unique_benef={fp.unique_beneficiaries}."
                    ),
                })

            result = fp_er.fingerprint_all(txn_df, period=period)

            report_path = None
            if save_rep and self.workspace:
                rdir = self.workspace.reports_dir / "graph"
                rdir.mkdir(parents=True, exist_ok=True)
                rfile = rdir / f"fingerprints_{period}.json"
                result.save(rfile)
                report_path = str(rfile)

            entries_deposited = 0
            if deposit_mem and result.alerts():
                dk = self.knowledge
                if dk is None and self.workspace is not None:
                    try:
                        from sentinel.memory.knowledge import DatasetKnowledge
                        dk = DatasetKnowledge.load(self.workspace, dataset_name)
                    except Exception:
                        dk = None
                if dk is not None:
                    from sentinel.memory.knowledge import KnowledgeEntry, KnowledgeType
                    for fp in result.alerts()[:10]:
                        try:
                            entry = KnowledgeEntry(
                                session_id=self.session_id,
                                entry_type=KnowledgeType.ANOMALY,
                                feature="",
                                claim=(
                                    f"Account {fp.account_id} mule_likelihood="
                                    f"{fp.mule_likelihood:.2f} [ALERT] -- "
                                    f"behavioral fingerprint flagged"
                                ),
                                detail=(
                                    f"{fp.n_txns} txns, "
                                    f"night={fp.night_ratio:.0%}, "
                                    f"median_gap={fp.median_gap_minutes:.1f}min, "
                                    f"burst5={fp.burst_max_5min}, "
                                    f"round_amt={fp.round_amount_ratio:.0%}, "
                                    f"unique_benef={fp.unique_beneficiaries}, "
                                    f"unique_dev={fp.unique_devices}"
                                ),
                                confidence=min(0.95, 0.60 + 0.35 * fp.mule_likelihood),
                                period=period,
                                tags=["graph", "fingerprint", "mule_candidate", "alert"],
                            )
                            dk.add(entry)
                            entries_deposited += 1
                        except Exception:
                            pass

            top_fps = [f.to_dict() for f in result.top_by_mule_likelihood(top_n)]
            interpretation = (
                f"Fingerprinted {result.n_accounts:,} accounts in {period}. "
                f"{result.n_alerts} ALERT, {result.n_warnings} WARNING."
            )
            if entries_deposited:
                interpretation += f" Deposited {entries_deposited} alerts to memory."

            return _safe_json({
                "ok": True,
                "tool": "fingerprint_accounts",
                "mode": "all",
                "period": period,
                "n_accounts": result.n_accounts,
                "n_alerts": result.n_alerts,
                "n_warnings": result.n_warnings,
                "top_fingerprints": top_fps,
                "report_path": report_path,
                "entries_deposited": entries_deposited,
                "elapsed_ms": result.elapsed_ms,
                "interpretation": interpretation,
            })

        except Exception:
            return _error(f"fingerprint_accounts failed:\n{traceback.format_exc()[-600:]}")

    # -- Helpers ---------------------------------------------------------------

    def _require_df(self, key: str):
        """
        Look up a DataFrame from self.datasets.

        Returns the DataFrame if found, or an error JSON string if not.
        """
        df = self.datasets.get(key)
        if df is None:
            available = list(self.datasets.keys())
            return _error(
                f"Dataset '{key}' not found in memory. "
                f"Available: {available}. "
                f"Load it first and pass via the datasets= argument of Phase2Tools."
            )
        return df

    def register_dataset(self, key: str, df) -> None:
        """Add or replace a DataFrame in the in-memory registry."""
        self.datasets[key] = df
        log.debug("Phase2Tools: registered dataset '%s' (%d rows)", key, len(df))

    def dataset_keys(self) -> list[str]:
        """Return the list of currently loaded dataset keys."""
        return list(self.datasets.keys())


# -- Convenience factory -------------------------------------------------------


def make_phase2_tools(
    workspace=None,
    datasets: dict | None = None,
    knowledge=None,
    tier_classifier=None,
    session_id: str = "unknown",
) -> dict[str, Any]:
    """
    Build a Phase2Tools instance and return its tool map.

    Returns
    -------
    dict[tool_name, callable]  -- same shape as make_dictionary_tools()
    and make_findings_tools() so the agent can treat all tool sets uniformly.
    """
    p2 = Phase2Tools(
        workspace=workspace,
        datasets=datasets,
        knowledge=knowledge,
        tier_classifier=tier_classifier,
        session_id=session_id,
    )
    return p2.as_tool_map(), p2
