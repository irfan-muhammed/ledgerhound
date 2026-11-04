"""
ledgerhound.analysis.fraud_playbook
==================================

FraudEDAPlaybook -- a deterministic, LLM-free analysis pipeline that chains
all Phase 2 tools in the correct order for a new fraud dataset.

Unlike the agent playbook (which sends natural-language questions to an LLM
and then dispatches tool calls), this runner executes tools directly and
reproducibly. Same inputs -> same outputs every time. No API cost, no
non-determinism, full auditability.

When to use
-----------
  - First-time onboarding of a new fraud dataset
  - Scheduled nightly/weekly health check run
  - CI pipeline gate before model training
  - Benchmark baseline capture

The playbook executes these steps in order:

  Step 1  DATA QUALITY       CrossDatasetValidator -- check summary vs raw txns
  Step 2  TYPOLOGY DISCOVERY TypologyAnalyzer -- cluster confirmed fraud
  Step 3  HARD NEGATIVES     HardNegativeMiner -- identify boundary non-fraud
  Step 4  PSI CERTIFICATION  PSICalculator -- certify sample representativeness
  Step 5  SIGNAL HEALTH      SignalDecayTracker -- measure + history + decay
  Step 6  MEMORY             DatasetKnowledge -- deposit summary findings

Output: PlaybookRunReport -- structured JSON with one section per step,
overall pass/fail, and institutional memory entries generated.

Usage
-----
    from ledgerhound.analysis.fraud_playbook import FraudEDAPlaybook

    playbook = FraudEDAPlaybook(workspace=ws, session_id="run_001")
    report = playbook.run(
        fraud_df=fraud_df,
        nonfr_df=nonfr_df,
        summary_df=summary_df,
        raw_txn_df=raw_txn_df,
        features=numeric_features,
        period="2024-Q2",
    )
    report.display()
    report.save(ws.reports_dir / "playbook_run.json")
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ledgerhound.workspace.layout import WorkspaceLayout
    from ledgerhound.memory.knowledge import DatasetKnowledge
    from ledgerhound.compute.context import ComputeContext
    from ledgerhound.compute.handle import DataFrameHandle
    from ledgerhound.narrative.op_log import OperationLog

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Step result ----------------------------------------------------------------

@dataclass
class PlaybookStepResult:
    step_name: str
    step_index: int
    status: str                  # "PASS" | "WARNING" | "FAIL" | "SKIPPED" | "ERROR"
    elapsed_ms: float
    summary: str                 # one-line human summary
    detail: dict = field(default_factory=dict)  # structured output from the tool
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "step_name":  self.step_name,
            "step_index": self.step_index,
            "status":     self.status,
            "elapsed_ms": self.elapsed_ms,
            "summary":    self.summary,
            "detail":     self.detail,
            "error":      self.error,
        }


# -- Full run report ------------------------------------------------------------

@dataclass
class PlaybookRunReport:
    period: str
    dataset_name: str
    generated_at: str
    elapsed_ms: float
    overall_status: str          # "PASS" | "WARNING" | "FAIL"
    steps: list[PlaybookStepResult] = field(default_factory=list)
    knowledge_entries_deposited: int = 0
    n_pass: int = 0
    n_warning: int = 0
    n_fail: int = 0
    n_error: int = 0
    n_skipped: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "steps"}
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        log.info("Playbook report saved: %s", path)

    def display(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        from rich.panel import Panel

        console = Console()

        status_colors = {
            "PASS":    "green",
            "WARNING": "yellow",
            "FAIL":    "red",
            "ERROR":   "red",
            "SKIPPED": "dim",
        }
        oc = status_colors.get(self.overall_status, "white")

        console.print(Panel(
            f"[bold]Fraud EDA Playbook -- {self.dataset_name} / {self.period}[/bold]\n"
            f"Overall: [{oc}]{self.overall_status}[/{oc}]  |  "
            f"{self.n_pass} PASS  {self.n_warning} WARN  {self.n_fail} FAIL  "
            f"{self.n_error} ERROR  |  {self.elapsed_ms/1000:.1f}s",
            border_style=oc,
        ))

        t = Table("Step", "Name", "Status", "Time", "Summary",
                  show_header=True, header_style="bold dim")
        for s in self.steps:
            sc = status_colors.get(s.status, "white")
            t.add_row(
                str(s.step_index),
                s.step_name,
                f"[{sc}]{s.status}[/{sc}]",
                f"{s.elapsed_ms:.0f}ms",
                s.summary[:80] + ("..." if len(s.summary) > 80 else ""),
            )
        console.print(t)

        if self.notes:
            console.print(f"\n[dim]{self.notes}[/dim]")
        console.print()


# -- Playbook runner ------------------------------------------------------------

class FraudEDAPlaybook:
    """
    Deterministic Phase 2 analysis pipeline for a fraud dataset.

    Parameters
    ----------
    workspace        : WorkspaceLayout -- for persisting artifacts.
    dataset_name     : Name for reports / knowledge entries (default: "fraud_dataset").
    session_id       : Session ID for knowledge entries.
    n_clusters       : Number of typology clusters. None = auto.
    hard_neg_strategy: "isolation_forest" | "centroid_distance" | "linear_boundary".
    hard_neg_top_k   : How many hard negatives to mine. None = 10% of non-fraud.
    psi_n_buckets    : Number of PSI buckets.
    skip_steps       : Set of step names to skip (e.g. {"consistency"}).
    """

    STEP_NAMES = [
        "consistency",
        "typology",
        "typology_evolution",
        "hard_negatives",
        "psi_certification",
        "signal_health",
        "memory_deposit",
    ]

    def __init__(
        self,
        workspace: Optional["WorkspaceLayout"] = None,
        dataset_name: str = "fraud_dataset",
        session_id: str = "playbook",
        n_clusters: Optional[int] = None,
        hard_neg_strategy: str = "isolation_forest",
        hard_neg_top_k: Optional[int] = None,
        psi_n_buckets: int = 20,
        skip_steps: Optional[set[str]] = None,
        evolution_match_threshold: float = 0.60,
        ctx: Optional["ComputeContext"] = None,
        op_log: Optional["OperationLog"] = None,
    ) -> None:
        self.workspace                  = workspace
        self.dataset_name               = dataset_name
        self.session_id                 = session_id
        self.n_clusters                 = n_clusters
        self.hard_neg_strategy          = hard_neg_strategy
        self.hard_neg_top_k             = hard_neg_top_k
        self.psi_n_buckets              = psi_n_buckets
        self.skip_steps                 = skip_steps or set()
        self.evolution_match_threshold  = evolution_match_threshold
        self.ctx                        = ctx
        self.op_log                     = op_log
        self._input_handles: dict[str, "DataFrameHandle"] = {}

    # -- Public API -------------------------------------------------------------

    def run(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
        period: str = "current",
        summary_df: Optional[pd.DataFrame] = None,
        raw_txn_df: Optional[pd.DataFrame] = None,
        fraud_labels_df: Optional[pd.DataFrame] = None,
        join_key: str = "account_id",
    ) -> PlaybookRunReport:
        """
        Run the full playbook and return a PlaybookRunReport.

        Parameters
        ----------
        fraud_df       : Confirmed fraud DataFrame.
        nonfr_df       : Non-fraud DataFrame (sample).
        features       : Numeric feature columns for all analysis steps.
        period         : Time period label for this run (e.g. "2024-Q2").
        summary_df     : Combined summary table (optional -- for consistency check).
        raw_txn_df     : Raw transaction sample (optional -- for consistency check).
        fraud_labels_df: Fraud labels table (optional -- for consistency check).
        join_key       : Account ID column name.
        """
        t_total = time.perf_counter()
        steps: list[PlaybookStepResult] = []
        knowledge_count = 0

        # Layer C: register input frames with the ComputeContext (if attached)
        # so every step's op_log entry can cite handle ids instead of just names.
        self._register_inputs(
            fraud=fraud_df, nonfr=nonfr_df,
            summary=summary_df, raw_txn=raw_txn_df, fraud_labels=fraud_labels_df,
        )

        # Resolve DatasetKnowledge
        dk = self._get_knowledge()

        # -- Step 1: Consistency ------------------------------------------------
        if "consistency" not in self.skip_steps and summary_df is not None and raw_txn_df is not None:
            steps.append(self._run_step(
                index=1, name="consistency",
                fn=self._step_consistency,
                kwargs=dict(
                    summary_df=summary_df,
                    raw_txn_df=raw_txn_df,
                    fraud_labels_df=fraud_labels_df,
                    join_key=join_key,
                ),
            ))
        else:
            steps.append(PlaybookStepResult(
                step_name="consistency", step_index=1,
                status="SKIPPED", elapsed_ms=0,
                summary="Skipped -- summary_df or raw_txn_df not provided.",
            ))

        # -- Step 2: Typology ----------------------------------------------------
        if "typology" not in self.skip_steps:
            steps.append(self._run_step(
                index=2, name="typology",
                fn=self._step_typology,
                kwargs=dict(fraud_df=fraud_df, features=features),
            ))

        # -- Step 2b: Typology evolution ----------------------------------------
        if "typology_evolution" not in self.skip_steps:
            steps.append(self._run_step(
                index=3, name="typology_evolution",
                fn=self._step_typology_evolution,
                kwargs=dict(fraud_df=fraud_df, features=features, period=period, dk=dk),
            ))
        else:
            steps.append(PlaybookStepResult(
                step_name="typology_evolution", step_index=3,
                status="SKIPPED", elapsed_ms=0,
                summary="Skipped -- in skip_steps.",
            ))

        # -- Step 3: Hard negatives ---------------------------------------------
        if "hard_negatives" not in self.skip_steps:
            steps.append(self._run_step(
                index=4, name="hard_negatives",
                fn=self._step_hard_negatives,
                kwargs=dict(fraud_df=fraud_df, nonfr_df=nonfr_df, features=features),
            ))

        # -- Step 4: PSI certification ------------------------------------------
        if "psi_certification" not in self.skip_steps:
            steps.append(self._run_step(
                index=5, name="psi_certification",
                fn=self._step_psi,
                kwargs=dict(nonfr_df=nonfr_df, features=features),
            ))

        # -- Step 5: Signal health ----------------------------------------------
        if "signal_health" not in self.skip_steps:
            steps.append(self._run_step(
                index=6, name="signal_health",
                fn=self._step_signal_health,
                kwargs=dict(
                    fraud_df=fraud_df, nonfr_df=nonfr_df,
                    features=features, period=period, dk=dk,
                ),
            ))

        # -- Step 6: Memory deposit ---------------------------------------------
        if "memory_deposit" not in self.skip_steps and dk is not None:
            steps.append(self._run_step(
                index=7, name="memory_deposit",
                fn=self._step_memory_deposit,
                kwargs=dict(steps_so_far=steps, period=period, dk=dk),
            ))
            knowledge_count = steps[-1].detail.get("entries_deposited", 0)

        # -- Roll up ------------------------------------------------------------
        elapsed_ms = (time.perf_counter() - t_total) * 1000
        n_pass    = sum(1 for s in steps if s.status == "PASS")
        n_warn    = sum(1 for s in steps if s.status == "WARNING")
        n_fail    = sum(1 for s in steps if s.status == "FAIL")
        n_error   = sum(1 for s in steps if s.status == "ERROR")
        n_skipped = sum(1 for s in steps if s.status == "SKIPPED")

        if n_fail > 0 or n_error > 0:
            overall = "FAIL"
        elif n_warn > 0:
            overall = "WARNING"
        else:
            overall = "PASS"

        report = PlaybookRunReport(
            period=period,
            dataset_name=self.dataset_name,
            generated_at=_now(),
            elapsed_ms=round(elapsed_ms, 1),
            overall_status=overall,
            steps=steps,
            knowledge_entries_deposited=knowledge_count,
            n_pass=n_pass,
            n_warning=n_warn,
            n_fail=n_fail,
            n_error=n_error,
            n_skipped=n_skipped,
        )

        # Save report if workspace available
        if self.workspace:
            report_path = self.workspace.reports_dir / "playbook" / f"run_{period}.json"
            report.save(report_path)
            self._save_coverage_snapshot()
            if self.op_log is not None:
                try:
                    self.op_log.save(report_path, kind="playbook_report")
                except Exception:
                    pass
            # B5: reproducibility manifest -- file hashes + env + git
            self._save_manifest(period=period)

        return report

    def _save_manifest(self, *, period: str) -> None:
        """Emit MANIFEST_<period>.json alongside the report so the run is
        reproducibility-tagged. Best-effort; never fails the playbook."""
        if self.workspace is None:
            return
        try:
            from ledgerhound.reproducibility import build_manifest
            m = build_manifest(
                workspace_root=self.workspace.root,
                seeds={"playbook_default": 42},
                command=f"FraudEDAPlaybook({self.dataset_name}, period={period})",
            )
            out = self.workspace.root / f"MANIFEST_{period}.json"
            m.save(out)
            log.info("Reproducibility manifest saved: %s", out)
            if self.op_log is not None:
                try:
                    self.op_log.save(out, kind="manifest")
                except Exception:
                    pass
        except Exception as exc:
            log.warning("Could not build reproducibility manifest: %s", exc)

    def _save_coverage_snapshot(self) -> None:
        """Persist the live CoverageTracker to ``<reports>/coverage.json``
        so a scorecard built in a separate process can pick up step /
        column / KB telemetry."""
        if self.workspace is None:
            return
        try:
            from ledgerhound.instrumentation import coverage as _cov
            cov_path = self.workspace.reports_dir / "coverage.json"
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            cov_path.write_text(
                json.dumps(_cov().report(), indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Coverage snapshot saved: %s", cov_path)
        except Exception as exc:
            log.warning("Could not save coverage snapshot: %s", exc)

    # -- Step implementations ---------------------------------------------------

    def _step_consistency(
        self,
        summary_df: pd.DataFrame,
        raw_txn_df: pd.DataFrame,
        fraud_labels_df: Optional[pd.DataFrame],
        join_key: str,
    ) -> PlaybookStepResult:
        from ledgerhound.validation.consistency import CrossDatasetValidator

        fraud_ids = None
        if fraud_labels_df is not None and join_key in fraud_labels_df.columns:
            fraud_ids = pd.Series(fraud_labels_df[join_key].tolist())

        validator = CrossDatasetValidator()
        rpt = validator.validate(
            summary_df=summary_df,
            raw_df=raw_txn_df,
            join_key=join_key,
            fraud_label_ids=fraud_ids,
        )

        if self.workspace:
            p = self.workspace.reports_dir / "consistency" / "consistency_report.json"
            rpt.save(p)

        status = rpt.overall_status
        summary = (
            f"{rpt.n_pass} PASS, {rpt.n_warning} WARNING, {rpt.n_fail} FAIL "
            f"across {len(rpt.checks)} checks"
        )
        return PlaybookStepResult(
            step_name="consistency", step_index=1,
            status=status, elapsed_ms=0,
            summary=summary,
            detail={"n_pass": rpt.n_pass, "n_warning": rpt.n_warning,
                    "n_fail": rpt.n_fail, "n_skipped": rpt.n_skipped},
        )

    def _step_typology(
        self,
        fraud_df: pd.DataFrame,
        features: list[str],
    ) -> PlaybookStepResult:
        from ledgerhound.analysis.typology import TypologyAnalyzer, ClusteringMethod

        avail = [f for f in features if f in fraud_df.columns]
        self._record_cols("typology", avail)
        analyzer = TypologyAnalyzer(
            method=ClusteringMethod.KMEANS,
            n_clusters=self.n_clusters,
        )
        rpt = analyzer.analyze(fraud_df=fraud_df, features=avail)

        if self.workspace:
            p = self.workspace.reports_dir / "typology" / "typology_report.json"
            rpt.save(p)

        sil = rpt.silhouette_score or 0.0
        status = "PASS" if sil > 0.40 else "WARNING" if sil > 0.20 else "FAIL"
        typs = [
            {"cluster_id": t.cluster_id, "label": t.label,
             "n_cases": t.n_cases, "top_feature": t.top_features[0] if t.top_features else ""}
            for t in rpt.typologies
        ]
        return PlaybookStepResult(
            step_name="typology", step_index=2,
            status=status, elapsed_ms=0,
            summary=f"{rpt.n_clusters} clusters, silhouette={sil:.3f}",
            detail={"n_clusters": rpt.n_clusters, "silhouette_score": sil,
                    "typologies": typs},
        )

    def _step_typology_evolution(
        self,
        fraud_df: pd.DataFrame,
        features: list[str],
        period: str,
        dk: Optional["DatasetKnowledge"],
    ) -> PlaybookStepResult:
        from ledgerhound.analysis.typology_evolution import TypologyEvolutionTracker

        avail = [
            f for f in features
            if f in fraud_df.columns
            and pd.api.types.is_numeric_dtype(fraud_df[f])
        ]
        self._record_cols("typology_evolution", avail)
        tracker = TypologyEvolutionTracker(
            n_clusters=self.n_clusters,
            match_threshold=self.evolution_match_threshold,
        )
        result = tracker.track(
            fraud_df=fraud_df,
            features=avail,
            period=period,
            workspace=self.workspace,
            dataset_knowledge=dk,
            session_id=self.session_id,
        )

        # Save evolution snapshot to reports dir
        if self.workspace:
            p = (self.workspace.reports_dir / "typology"
                 / f"evolution_{period}.json")
            result.save(p)

        # Map overall_alert to playbook status
        alert_to_status = {
            "FAIL":    "WARNING",   # FAIL alert -> playbook WARNING (not a blocker)
            "WARNING": "WARNING",
            "INFO":    "PASS",
            "NONE":    "PASS",
        }
        status = alert_to_status.get(result.overall_alert, "PASS")

        parts = [f"{result.n_clusters_current} clusters"]
        if result.n_emerging:
            parts.append(f"{result.n_emerging} EMERGING")
        if result.n_growing:
            parts.append(f"{result.n_growing} growing")
        if result.n_declining:
            parts.append(f"{result.n_declining} declining")
        if result.overall_alert != "NONE":
            parts.append(f"alert={result.overall_alert}")

        alerts_detail = [
            {
                "cluster_key": t.cluster_key,
                "alert_level": t.alert_level,
                "alert_reason": t.alert_reason,
                "trend": t.trend,
                "avg_growth_rate": round(t.avg_growth_rate, 4),
            }
            for t in result.top_alerts()
        ]
        return PlaybookStepResult(
            step_name="typology_evolution", step_index=3,
            status=status, elapsed_ms=0,
            summary=", ".join(parts),
            detail={
                "period": period,
                "n_fraud_cases": result.n_fraud_cases,
                "n_clusters_current": result.n_clusters_current,
                "n_emerging": result.n_emerging,
                "n_growing": result.n_growing,
                "n_declining": result.n_declining,
                "n_stable": result.n_stable,
                "disappeared": result.disappeared_clusters,
                "overall_alert": result.overall_alert,
                "top_alerts": alerts_detail,
            },
        )


    def _step_hard_negatives(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
    ) -> PlaybookStepResult:
        from ledgerhound.sampling.hard_negatives import HardNegativeMiner, HardNegativeStrategy

        avail = [
            f for f in features
            if f in fraud_df.columns and f in nonfr_df.columns
        ]
        self._record_cols("hard_negatives", avail)
        strategy = HardNegativeStrategy(self.hard_neg_strategy)
        miner = HardNegativeMiner(
            strategy=strategy,
            top_k=self.hard_neg_top_k,
        )
        hn_df, result = miner.mine(fraud_df, nonfr_df, avail)

        if self.workspace:
            p = self.workspace.samples_dir / "hard_negatives"
            p.mkdir(parents=True, exist_ok=True)
            hn_df.to_parquet(p / f"hard_negatives_{self.hard_neg_strategy}.parquet", index=False)

        status = "PASS"
        return PlaybookStepResult(
            step_name="hard_negatives", step_index=4,
            status=status, elapsed_ms=0,
            summary=(
                f"{result.n_hard_negatives:,} hard negatives selected "
                f"({result.inclusion_rate:.1%} of non-fraud)"
            ),
            detail={
                "n_hard_negatives": result.n_hard_negatives,
                "inclusion_rate": round(result.inclusion_rate, 4),
                "strategy": result.strategy,
                "boundary_density": round(result.boundary_density, 4) if result.boundary_density else None,
                "top_features": [
                    {"feature": f, "importance": round(v, 4)}
                    for f, v in sorted(
                        result.feature_importances.items(), key=lambda x: -x[1]
                    )[:5]
                ],
            },
        )

    def _step_psi(
        self,
        nonfr_df: pd.DataFrame,
        features: list[str],
    ) -> PlaybookStepResult:
        from ledgerhound.sampling.psi import (
            PSICalculator, PSIStatus, compute_bucket_bounds, _numeric_sample_dist,
        )
        from ledgerhound.sampling.certificate import CertificateGenerator

        avail = [f for f in features if f in nonfr_df.columns
                 and pd.api.types.is_numeric_dtype(nonfr_df[f])]
        self._record_cols("psi_certification", avail)

        calc   = PSICalculator()
        bounds = {}
        pop    = {}
        for feat in avail:
            series = nonfr_df[feat].dropna()
            b = compute_bucket_bounds(series, n_buckets=self.psi_n_buckets)
            bounds[feat] = b
            pop[feat]    = _numeric_sample_dist(series, b)   # sample as own baseline

        psi_result = calc.batch(
            sample_df=nonfr_df[avail],
            numeric_population_dists=pop,
            categorical_population_dists={},
            numeric_bounds=bounds,
            weights={f: 1.0 for f in avail},
            n_population=len(nonfr_df),
        )

        cert_path = None
        if self.workspace:
            gen  = CertificateGenerator()
            cert = gen.generate(
                psi_result=psi_result,
                sample_name="nonfr",
                sample_version="v1",
                fraud_dataset_key="fraud",
                nonfr_dataset_key="nonfr",
                fraud_rows=0,
                nonfr_rows=len(nonfr_df),
                sampling_strategy="playbook",
            )
            cert_dir = self.workspace.samples_dir / "nonfr"
            cert_dir.mkdir(parents=True, exist_ok=True)
            cert.save(cert_dir / "sampling_certificate.json")
            cert_path = str(cert_dir / "sampling_certificate.json")

        status_map = {
            PSIStatus.PASS:    "PASS",
            PSIStatus.WARNING: "WARNING",
            PSIStatus.FAIL:    "FAIL",
        }
        status = status_map.get(psi_result.overall_status, "PASS")
        return PlaybookStepResult(
            step_name="psi_certification", step_index=5,
            status=status, elapsed_ms=0,
            summary=(
                f"Weighted PSI={psi_result.weighted_psi:.4f}  "
                f"{psi_result.n_pass} PASS / {psi_result.n_warning} WARN / "
                f"{psi_result.n_fail} FAIL"
            ),
            detail={
                "weighted_psi": round(psi_result.weighted_psi, 4),
                "overall_status": psi_result.overall_status.value,
                "n_pass": psi_result.n_pass,
                "n_warning": psi_result.n_warning,
                "n_fail": psi_result.n_fail,
                "certificate_path": cert_path,
            },
        )

    def _step_signal_health(
        self,
        fraud_df: pd.DataFrame,
        nonfr_df: pd.DataFrame,
        features: list[str],
        period: str,
        dk: Optional["DatasetKnowledge"],
    ) -> PlaybookStepResult:
        from ledgerhound.analysis.signal_decay import SignalDecayTracker

        avail = [
            f for f in features
            if f in fraud_df.columns and f in nonfr_df.columns
            and pd.api.types.is_numeric_dtype(fraud_df[f])
        ]
        self._record_cols("signal_health", avail)
        tracker = SignalDecayTracker()
        result  = tracker.track(
            fraud_df=fraud_df,
            nonfr_df=nonfr_df,
            features=avail,
            period=period,
            workspace=self.workspace,
            dataset_knowledge=dk,
            session_id=self.session_id,
        )

        if result.n_fail > 0:
            status = "FAIL"
        elif result.n_warning > 0:
            status = "WARNING"
        else:
            status = "PASS"

        top5 = result.top_features(5)
        return PlaybookStepResult(
            step_name="signal_health", step_index=6,
            status=status, elapsed_ms=0,
            summary=(
                f"{result.n_features} features: "
                f"{result.n_stable} stable, {result.n_warning} warning, "
                f"{result.n_fail} fail, {result.n_improved} improved"
            ),
            detail={
                "n_features": result.n_features,
                "n_stable": result.n_stable,
                "n_warning": result.n_warning,
                "n_fail": result.n_fail,
                "n_improved": result.n_improved,
                "baseline_period": result.baseline_period,
                "top_features": [
                    {"feature": m.feature, "grade": m.signal_grade(),
                     "ks": m.ks_statistic, "auc": m.auc}
                    for m in top5
                ],
                "decay_events": [
                    {"feature": e.feature, "severity": e.severity,
                     "pct_change": round(e.pct_change, 1),
                     "metric": e.primary_metric}
                    for e in result.decay_events
                ],
            },
        )

    def _step_memory_deposit(
        self,
        steps_so_far: list[PlaybookStepResult],
        period: str,
        dk: "DatasetKnowledge",
    ) -> PlaybookStepResult:
        from ledgerhound.memory.knowledge import KnowledgeEntry, KnowledgeType

        entries_deposited = 0
        step_map = {s.step_name: s for s in steps_so_far}

        # Deposit typology summary
        if "typology" in step_map and step_map["typology"].status != "SKIPPED":
            t = step_map["typology"]
            n_clusters = t.detail.get("n_clusters", "?")
            sil = t.detail.get("silhouette_score", 0)
            entry = KnowledgeEntry(
                session_id=self.session_id,
                entry_type=KnowledgeType.TYPOLOGY,
                feature="",
                claim=(
                    f"Fraud dataset has {n_clusters} distinct typologies "
                    f"(silhouette={sil:.3f})"
                ),
                detail=json.dumps(t.detail, default=str),
                confidence=0.9 if (sil or 0) > 0.4 else 0.7,
                period=period,
                tags=["typology", "playbook", period],
            )
            dk.add(entry)
            entries_deposited += 1

        # Deposit typology evolution alerts
        if "typology_evolution" in step_map and step_map["typology_evolution"].status not in ("SKIPPED", "ERROR"):
            ev = step_map["typology_evolution"]
            n_emerging  = ev.detail.get("n_emerging", 0)
            n_growing   = ev.detail.get("n_growing", 0)
            overall_alr = ev.detail.get("overall_alert", "NONE")
            if overall_alr != "NONE" or n_emerging:
                entry = KnowledgeEntry(
                    session_id=self.session_id,
                    entry_type=KnowledgeType.SIGNAL_EMERGING,
                    feature="",
                    claim=(
                        f"Typology evolution alert ({overall_alr}): "
                        f"{n_emerging} emerging, {n_growing} growing clusters in {period}"
                    ),
                    detail=json.dumps(ev.detail, default=str),
                    confidence=0.80,
                    period=period,
                    tags=["typology_evolution", "playbook", period,
                          overall_alr.lower()],
                )
                dk.add(entry)
                entries_deposited += 1

        # Deposit hard negative summary
        if "hard_negatives" in step_map and step_map["hard_negatives"].status != "SKIPPED":
            hn = step_map["hard_negatives"]
            n_hn = hn.detail.get("n_hard_negatives", 0)
            ir   = hn.detail.get("inclusion_rate", 0)
            top_feat = (hn.detail.get("top_features") or [{}])[0].get("feature", "unknown")
            entry = KnowledgeEntry(
                session_id=self.session_id,
                entry_type=KnowledgeType.BOUNDARY,
                feature=top_feat,
                claim=(
                    f"{n_hn:,} hard negatives ({ir:.1%} of non-fraud) found near fraud boundary"
                ),
                detail=json.dumps(hn.detail, default=str),
                confidence=0.85,
                period=period,
                tags=["hard_negatives", "sampling", period],
            )
            dk.add(entry)
            entries_deposited += 1

        # Deposit signal grade summary
        if "signal_health" in step_map and step_map["signal_health"].status != "SKIPPED":
            sig = step_map["signal_health"]
            n_feats = sig.detail.get("n_features", 0)
            n_fail  = sig.detail.get("n_fail", 0)
            top5    = sig.detail.get("top_features", [])
            top5_str = ", ".join(f"{f['feature']}(grade={f['grade']})" for f in top5[:3])
            entry = KnowledgeEntry(
                session_id=self.session_id,
                entry_type=KnowledgeType.SIGNAL_EMERGING if n_fail == 0 else KnowledgeType.SIGNAL_DECAY,
                feature="",
                claim=(
                    f"Signal health: {n_feats} features measured, {n_fail} FAIL. "
                    f"Top: {top5_str}"
                ),
                detail=json.dumps(sig.detail, default=str),
                confidence=0.85,
                period=period,
                tags=["signal_health", "playbook", period],
            )
            dk.add(entry)
            entries_deposited += 1

        return PlaybookStepResult(
            step_name="memory_deposit", step_index=7,
            status="PASS", elapsed_ms=0,
            summary=f"Deposited {entries_deposited} knowledge entries for '{self.dataset_name}'",
            detail={"entries_deposited": entries_deposited},
        )

    # -- Helpers ---------------------------------------------------------------

    def _register_inputs(self, **dfs) -> None:
        """Layer C -- load each non-None input frame into the ComputeContext.
        Cached handles are used by _run_step for op_log narrative attribution."""
        if self.ctx is None:
            return
        for name, df in dfs.items():
            if df is None:
                continue
            try:
                h = self.ctx.load(df, name=name, op=f"load_{name}")
                self._input_handles[name] = h
            except Exception as exc:
                log.warning("Could not register input frame '%s': %s", name, exc)

    def _step_input_handle_ids(self, kwargs: dict) -> list[str]:
        """Map step kwargs (fraud_df=..., nonfr_df=...) to registered handle ids."""
        out: list[str] = []
        for k in ("fraud_df", "nonfr_df", "summary_df", "raw_txn_df", "fraud_labels_df"):
            if k not in kwargs or kwargs[k] is None:
                continue
            short = k[:-3] if k.endswith("_df") else k
            h = self._input_handles.get(short)
            if h is not None:
                out.append(h.id)
        return out

    def _run_step(
        self, index: int, name: str, fn, kwargs: dict
    ) -> PlaybookStepResult:
        """Run a step function with timing and exception capture.

        When op_log is attached, the step body is also wrapped in an
        op_log.step(kind=tool) so every playbook step becomes a narrative
        entry with input handle ids, status, and summary."""
        t0 = time.perf_counter()
        input_ids = self._step_input_handle_ids(kwargs) if self.ctx is not None else []

        def _do() -> PlaybookStepResult:
            try:
                result = fn(**kwargs)
                result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                self._record_step(name, result.status)
                return result
            except Exception as exc:
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                log.error("Playbook step '%s' failed: %s", name, exc)
                self._record_step(name, "ERROR")
                return PlaybookStepResult(
                    step_name=name, step_index=index,
                    status="ERROR", elapsed_ms=elapsed,
                    summary=f"Step raised an exception: {str(exc)[:120]}",
                    error=traceback.format_exc()[-800:],
                )

        if self.op_log is None:
            return _do()

        with self.op_log.step(
            kind="tool",
            op=f"playbook.{name}",
            inputs=input_ids,
            params={"step_index": index, "step_name": name},
            tags=["playbook", "step"],
        ) as step:
            result = _do()
            step.add_param("status", result.status)
            if result.summary:
                step.set_reasoning(result.summary[:400])
            if result.error:
                step.add_tag("error")
                step.set_notes(result.error[:200])
            return result

    @staticmethod
    def _record_step(name: str, status: str) -> None:
        """Best-effort coverage telemetry for the scorecard."""
        try:
            from ledgerhound.instrumentation import record_step
            record_step(name, status)
        except Exception:
            pass

    @staticmethod
    def _record_cols(tool_name: str, columns) -> None:
        """Best-effort column-coverage telemetry for the scorecard."""
        try:
            from ledgerhound.instrumentation import record_columns
            record_columns(tool_name, columns)
        except Exception:
            pass

    def _get_knowledge(self) -> Optional["DatasetKnowledge"]:
        if self.workspace is None:
            return None
        try:
            from ledgerhound.memory.knowledge import DatasetKnowledge
            return DatasetKnowledge.load(self.workspace, self.dataset_name)
        except Exception as exc:
            log.warning("Could not load DatasetKnowledge: %s", exc)
            return None
