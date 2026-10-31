"""
sentinel.monitor
================

Scheduled drift monitoring for Sentinel.

The monitor runs the signal-decay and typology-evolution trackers on a
fresh slice of data every cycle, persists the result to a chronological
log, and emits webhook alerts when FAIL- or WARNING-level events fire.

Typical usage
-------------
    from sentinel.monitor import DriftMonitor, DriftMonitorConfig

    cfg = DriftMonitorConfig.load(workspace)
    mon = DriftMonitor(cfg, workspace=workspace)
    result = mon.run_cycle(period="2026-W19")
    # OR
    mon.run_daemon(interval_seconds=3600)
"""

from sentinel.monitor.config import (
    DriftMonitorConfig,
    WebhookConfig,
    AlertThresholds,
)
from sentinel.monitor.result import (
    MonitorRunResult,
    AlertEvent,
)
from sentinel.monitor.monitor import DriftMonitor
from sentinel.monitor.webhook import WebhookEmitter
from sentinel.monitor.drift_sentinel import (
    DriftSentinel,
    DriftEpisode,
    capture_baseline,
)
from sentinel.monitor.schedule import ScheduleConfig, compute_next_run, seconds_until
from sentinel.monitor.trino_source import (
    TrinoSourceConfig,
    TrinoMonitorSource,
    TrinoSourceNotApproved,
)
from sentinel.monitor.investigate import investigate_alert, investigate_worst_alert

__all__ = [
    "DriftMonitor",
    "DriftMonitorConfig",
    "WebhookConfig",
    "AlertThresholds",
    "MonitorRunResult",
    "AlertEvent",
    "WebhookEmitter",
    "DriftSentinel",
    "DriftEpisode",
    "capture_baseline",
    "ScheduleConfig",
    "compute_next_run",
    "seconds_until",
    "TrinoSourceConfig",
    "TrinoMonitorSource",
    "TrinoSourceNotApproved",
    "investigate_alert",
    "investigate_worst_alert",
]
