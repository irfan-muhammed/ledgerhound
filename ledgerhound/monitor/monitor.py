"""
ledgerhound.monitor.monitor
========================

The DriftMonitor -- Ledgerhound's scheduled-watcher loop.

One *cycle* does:

  1. Load fraud + non-fraud + raw-txn DataFrames from configured paths
  2. Run SignalDecayTracker -> appends signal_decay / signal_emerging
     events into ``workspace.drift_events_path``
  3. Run TypologyEvolutionTracker -> appends cluster_growth /
     cluster_decline / cluster_emerging / cluster_disappeared events
  4. Re-read the drift events file for entries from this cycle (by
     period + cycle start time) and select those crossing configured
     severity thresholds -> these become AlertEvents
  5. Fan AlertEvents out to all matching webhooks via WebhookEmitter
  6. Build a reproducibility-manifest snapshot for the workspace
  7. Persist the MonitorRunResult to ``<workspace>/monitor/runs.jsonl``

Daemon mode (``run_daemon``) loops the cycle on a fixed interval until
interrupted.  For cron-style invocation, use ``run_once`` from the CLI.

Failure handling
----------------
Any unexpected exception inside ``run_cycle`` is caught, recorded on the
MonitorRunResult (``status="ERROR"``, ``error=<traceback>``), and the
cycle still terminates with a logged run.  This is critical for daemon
mode -- one bad cycle must never silently break the loop.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ledgerhound.monitor.config import DriftMonitorConfig, derive_period
from ledgerhound.monitor.result import AlertEvent, MonitorRunResult
from ledgerhound.monitor.schedule import compute_next_run, seconds_until
from ledgerhound.monitor.webhook import WebhookEmitter

if TYPE_CHECKING:
    from ledgerhound.workspace.layout import WorkspaceLayout
    from ledgerhound.trino.config import TrinoConfig

log = logging.getLogger(__name__)


_SEV_RANK = {"INFO": 0, "WARNING": 1, "FAIL": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_now() -> float:
    return time.time()


class DriftMonitor:
    """Scheduled watcher around SignalDecayTracker + TypologyEvolutionTracker."""

    def __init__(
        self,
        config: DriftMonitorConfig,
        workspace: "WorkspaceLayout",
        webhook_emitter: Optional[WebhookEmitter] = None,
        trino_config: Optional["TrinoConfig"] = None,
        llm=None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.webhook_emitter = webhook_emitter or WebhookEmitter()
        # Only constructed lazily (via TrinoConfig.from_env()) when a cycle
        # actually needs it, so monitors that don't use Trino never require
        # LEDGERHOUND_TRINO_* env vars to be set.
        self._trino_config = trino_config
        # Only required when config.auto_investigate=True; a real LLM
        # client (see ledgerhound.monitor.investigate).
        self.llm = llm

    # -- Public API -----------------------------------------------------

    @property
    def runs_log_path(self) -> Path:
        return Path(self.workspace.root) / "monitor" / "runs.jsonl"

    def run_cycle(
        self,
        period: Optional[str] = None,
        emit_webhooks: bool = True,
        build_manifest: bool = True,
    ) -> MonitorRunResult:
        """Execute one monitor cycle and return its MonitorRunResult."""
        t_start = time.perf_counter()
        cycle_start_wall = _ts_now()

        # Resolve period
        period = (
            period
            or self.config.fixed_period
            or derive_period(self.config.period_granularity)
        )

        result = MonitorRunResult(
            dataset_name=self.config.dataset_name,
            period=period,
            workspace_root=str(self.workspace.root),
        )
        log.info("DriftMonitor cycle start: %s/%s", result.dataset_name, period)

        try:
            # 1. Load data
            fraud_df, nonfr_df, raw_txn_df = self._load_dataframes()

            # 2. Signal decay
            n_decay = self._run_signal_decay(
                fraud_df=fraud_df,
                nonfr_df=nonfr_df,
                period=period,
                session_id=result.cycle_id,
            )
            result.n_signal_decay_events = n_decay

            # 3. Typology evolution (only if we have raw_txn or fraud_df features)
            n_evo = self._run_typology_evolution(
                fraud_df=fraud_df,
                period=period,
                session_id=result.cycle_id,
            )
            result.n_evolution_events = n_evo

            # 4. Collect alerts from drift_events.jsonl matching this cycle
            alerts = self._collect_alerts(
                period=period,
                cycle_start_wall=cycle_start_wall,
                cycle_id=result.cycle_id,
            )
            result.alerts = alerts
            result.n_alerts_emitted = len(alerts)
            result.n_drift_events_total = n_decay + n_evo

            # 5. Determine overall severity
            result.overall_alert = self._overall_severity(alerts)

            # 6. Emit webhooks
            if emit_webhooks and alerts and self.config.webhooks:
                stats = self.webhook_emitter.emit_many(
                    webhooks=self.config.webhooks,
                    alerts=alerts,
                )
                result.n_webhooks_attempted = stats["attempted"]
                result.n_webhooks_succeeded = stats["succeeded"]
                result.n_webhooks_failed    = stats["failed"]

            # 6b. Auto-investigate the worst alert (opt-in, real LLM call)
            if self.config.auto_investigate and alerts:
                if self.config.trino.enabled:
                    log.warning(
                        "auto_investigate=True with trino.enabled=True -- the "
                        "Trino-sourced slice for this cycle will be sent to "
                        "whatever LLM endpoint this DriftMonitor was built "
                        "with. Confirm that endpoint is the internal/regulated "
                        "gateway, not a public API -- sending real transaction "
                        "data to a public LLM is exactly what the zone "
                        "boundary (docs/v3.0/ZONING.md) forbids."
                    )
                if self.llm is None:
                    log.warning(
                        "auto_investigate=True but no llm was passed to "
                        "DriftMonitor(...) -- skipping investigation"
                    )
                else:
                    from ledgerhound.monitor.investigate import investigate_worst_alert
                    investigation = investigate_worst_alert(
                        alerts=alerts, fraud_df=fraud_df, nonfr_df=nonfr_df,
                        llm=self.llm, workspace=self.workspace,
                        dataset_name=self.config.dataset_name,
                    )
                    result.investigation = investigation

            # 7. Build manifest for this cycle
            if build_manifest:
                try:
                    from ledgerhound.reproducibility import build_manifest as _bm
                    m = _bm(
                        self.workspace.root,
                        seeds={"period": period},
                        command=f"monitor.run_cycle({period})",
                    )
                    mpath = Path(self.workspace.root) / f"MANIFEST_monitor_{result.cycle_id}.json"
                    m.save(mpath)
                    result.manifest_path = str(mpath)
                except Exception as exc:
                    log.warning("manifest build failed: %s", exc)

            result.status = "OK"

        except Exception as exc:
            result.status = "ERROR"
            result.error  = traceback.format_exc()[-1500:]
            log.error("DriftMonitor cycle FAILED: %s", exc)

        # Finalise + persist
        result.finished_at = _now()
        result.elapsed_ms  = round((time.perf_counter() - t_start) * 1000, 1)
        try:
            result.save_to_log(self.runs_log_path)
        except Exception as exc:
            log.warning("could not persist run log: %s", exc)

        log.info("DriftMonitor cycle done: %s", result.summary())
        return result

    def run_daemon(
        self,
        interval_seconds: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> int:
        """Loop ``run_cycle`` forever (or ``max_cycles`` times), sleeping
        between cycles according to ``config.schedule``.

        ``interval_seconds``, if passed, overrides the config and forces
        plain interval mode (back-compat with the original signature) --
        calendar-mode scheduling (specific days + time-of-day) is driven
        entirely by ``config.schedule`` when no override is given.

        Returns the number of cycles executed. Catches KeyboardInterrupt
        cleanly so Ctrl+C produces a graceful shutdown summary.
        """
        from ledgerhound.monitor.schedule import ScheduleConfig

        if interval_seconds is not None:
            schedule = ScheduleConfig(mode="interval", interval_seconds=int(interval_seconds))
        else:
            schedule = self.config.schedule

        n_done = 0
        try:
            while True:
                self.run_cycle()
                n_done += 1
                if max_cycles is not None and n_done >= max_cycles:
                    break
                wait_s = seconds_until(schedule)
                log.info("DriftMonitor next cycle at %s (%.0fs from now)",
                          compute_next_run(schedule).isoformat(), wait_s)
                time.sleep(wait_s)
        except KeyboardInterrupt:
            log.info("DriftMonitor interrupted by user after %d cycle(s)", n_done)
        return n_done

    # -- Internals -----------------------------------------------------

    def _load_dataframes(self):
        if self.config.trino.enabled:
            return self._load_dataframes_trino()
        return self._load_dataframes_files()

    def _load_dataframes_files(self):
        import pandas as pd

        def _load(path: str):
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Data file not found: {p}")
            return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)

        fraud_df = _load(self.config.fraud_data)
        nonfr_df = _load(self.config.nonfr_data)
        raw_txn_df = _load(self.config.raw_txn_data) if self.config.raw_txn_data else None
        return fraud_df, nonfr_df, raw_txn_df

    def _load_dataframes_trino(self):
        from ledgerhound.trino.config import TrinoConfig
        from ledgerhound.monitor.trino_source import TrinoMonitorSource

        trino_cfg = self._trino_config or TrinoConfig.from_env()
        source = TrinoMonitorSource(trino_cfg, self.config.trino)
        return source.load()

    def _run_signal_decay(self, fraud_df, nonfr_df, period: str, session_id: str) -> int:
        from ledgerhound.analysis.signal_decay import SignalDecayTracker
        import pandas as pd

        avail = [
            f for f in self.config.features
            if f in fraud_df.columns and f in nonfr_df.columns
            and pd.api.types.is_numeric_dtype(fraud_df[f])
        ]
        if not avail:
            log.warning("No numeric features available for signal decay; skipping")
            return 0
        tracker = SignalDecayTracker()
        res = tracker.track(
            fraud_df=fraud_df,
            nonfr_df=nonfr_df,
            features=avail,
            period=period,
            workspace=self.workspace,
            session_id=session_id,
        )
        return len(res.decay_events)

    def _run_typology_evolution(self, fraud_df, period: str, session_id: str) -> int:
        from ledgerhound.analysis.typology_evolution import TypologyEvolutionTracker
        import pandas as pd

        avail = [
            f for f in self.config.features
            if f in fraud_df.columns
            and pd.api.types.is_numeric_dtype(fraud_df[f])
        ]
        if len(avail) < 2:
            log.warning("Need >=2 numeric features for typology evolution; skipping")
            return 0
        tracker = TypologyEvolutionTracker(
            match_threshold=self.config.match_threshold,
        )
        res = tracker.track(
            fraud_df=fraud_df,
            features=avail,
            period=period,
            workspace=self.workspace,
            session_id=session_id,
        )
        # We don't have a direct event count from the tracker; the drift writer
        # appended events into drift_events.jsonl. We'll count them from the
        # alert collection step instead. Return a coarse proxy here.
        return (
            res.n_emerging
            + sum(1 for t in res.trajectories if t.alert_level != "NONE")
            + len(res.disappeared_clusters)
        )

    def _collect_alerts(
        self,
        period: str,
        cycle_start_wall: float,
        cycle_id: str,
    ) -> list[AlertEvent]:
        """Re-read drift_events.jsonl and pick this cycle's events that
        cross configured severity thresholds."""
        from ledgerhound.analysis.drift_writer import DriftEventWriter

        thresholds = self.config.alerts
        sev_floor = min(
            _SEV_RANK.get(thresholds.signal_decay_severity, 1),
            _SEV_RANK.get(thresholds.evolution_severity, 1),
        )

        writer = DriftEventWriter(self.workspace)
        if not writer.path.exists():
            return []

        alerts: list[AlertEvent] = []
        # Best-effort: filter by period + cycle_id; cycle_id is stamped as
        # session_id when SignalDecay/TypologyEvolution call the writer.
        for d in writer.iter_events(period=period):
            sev = d.get("severity", "INFO")
            if _SEV_RANK.get(sev, 0) < sev_floor:
                continue
            if not thresholds.emit_info_events and sev == "INFO":
                continue
            # Only include events from this cycle
            if d.get("session_id") not in (cycle_id, None):
                continue
            alerts.append(AlertEvent(
                severity=sev,
                event_type=d.get("event_type", "unknown"),
                period=d.get("period", period),
                claim=d.get("claim", ""),
                feature=d.get("feature"),
                cluster_key=d.get("cluster_key"),
                dataset_name=self.config.dataset_name,
                cycle_id=cycle_id,
                details=d.get("details") or {},
            ))
        return alerts

    @staticmethod
    def _overall_severity(alerts: list[AlertEvent]) -> str:
        if not alerts:
            return "NONE"
        worst = max(alerts, key=lambda a: _SEV_RANK.get(a.severity, 0))
        return worst.severity

    # -- Reading runs --------------------------------------------------

    def history(self, limit: Optional[int] = None) -> list[MonitorRunResult]:
        """Read past cycles from runs.jsonl, newest first."""
        import json
        p = self.runs_log_path
        if not p.exists():
            return []
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rows.append(MonitorRunResult.from_dict(d))
        rows.reverse()
        if limit is not None:
            rows = rows[:limit]
        return rows

    def latest(self) -> Optional[MonitorRunResult]:
        h = self.history(limit=1)
        return h[0] if h else None
