"""
sentinel.monitor.result
=======================

Result objects for a single Drift Monitor cycle.

Each cycle produces:
  - An ``AlertEvent`` list (subset of drift events that crossed configured
    severity thresholds during this cycle).
  - A ``MonitorRunResult`` with timing, alert counts, webhook delivery
    status, and the reproducibility-manifest path produced for this cycle.

All records persist to ``<workspace>/monitor/runs.jsonl`` as the
chronological cycle log.  The same JSONL is what ``sentinel monitor
history`` reads.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _sentinel_version() -> str:
    """Read the canonical version at instantiation time.

    These records stamp WHICH Sentinel produced a saved artifact, so a value
    that silently lags a release mislabels provenance. It was hardcoded in
    three separate dataclasses and had to be hand-edited every bump; imported
    lazily here because `sentinel/__init__` imports `sentinel.core`, so a
    module-level import would risk a cycle.
    """
    from sentinel import __version__
    return __version__



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cycle_id() -> str:
    return f"cyc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass
class AlertEvent:
    """One drift event that crossed configured severity thresholds.

    This is the *delivered* unit -- what gets POSTed to a webhook and
    written into the cycle's alerts list.  Distinct from
    ``DriftEvent`` (the raw underlying record on disk) because we
    enrich with monitor-cycle context: ``cycle_id``, ``dataset_name``,
    ``delivered_to`` webhook URLs.
    """

    severity: str                  # "INFO" | "WARNING" | "FAIL"
    event_type: str                # signal_decay, cluster_emerging, etc.
    period: str
    claim: str
    feature: Optional[str] = None
    cluster_key: Optional[str] = None
    dataset_name: str = ""
    cycle_id: str = ""
    triggered_at: str = field(default_factory=_now)
    details: dict[str, Any] = field(default_factory=dict)
    delivered_to: list[str] = field(default_factory=list)   # webhook URLs that ACKed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorRunResult:
    """One Drift Monitor cycle's outcome."""

    cycle_id: str = field(default_factory=_cycle_id)
    dataset_name: str = ""
    period: str = ""
    started_at: str = field(default_factory=_now)
    finished_at: Optional[str] = None
    elapsed_ms: float = 0.0

    # Drift detector outputs
    n_signal_decay_events:  int = 0
    n_evolution_events:     int = 0
    n_drift_events_total:   int = 0

    # Alert delivery
    overall_alert: str = "NONE"          # "NONE" | "INFO" | "WARNING" | "FAIL"
    alerts: list[AlertEvent] = field(default_factory=list)
    n_alerts_emitted: int = 0
    n_webhooks_attempted: int = 0
    n_webhooks_succeeded: int = 0
    n_webhooks_failed: int = 0

    # Auto-investigation (opt-in, see sentinel.monitor.investigate) --
    # None when auto_investigate is off, no alerts fired, or the
    # investigation itself failed (never raises into the cycle).
    investigation: Optional[dict[str, Any]] = None

    # Provenance
    manifest_path: Optional[str] = None
    workspace_root: str = ""
    sentinel_version: str = field(default_factory=_sentinel_version)

    # Error capture (best-effort: a failed cycle still produces a record)
    status: str = "OK"                   # "OK" | "ERROR"
    error: Optional[str] = None

    # -- Convenience -----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alerts"] = [a.to_dict() if hasattr(a, "to_dict") else a for a in self.alerts]
        return d

    def save_to_log(self, log_path: Path) -> None:
        """Append this result to the cycle log (newline-delimited JSON)."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorRunResult":
        # Rehydrate alerts to AlertEvent objects
        alerts = []
        for a in d.get("alerts") or []:
            if isinstance(a, dict):
                alerts.append(AlertEvent(**{
                    k: v for k, v in a.items()
                    if k in AlertEvent.__dataclass_fields__
                }))
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        kw["alerts"] = alerts
        return cls(**kw)

    def summary(self) -> str:
        return (
            f"[{self.cycle_id}] {self.dataset_name}/{self.period} -> "
            f"{self.overall_alert} "
            f"({self.n_signal_decay_events} decay, "
            f"{self.n_evolution_events} evo, "
            f"{self.n_alerts_emitted} alerts emitted, "
            f"{self.n_webhooks_succeeded}/{self.n_webhooks_attempted} webhooks OK, "
            f"{self.elapsed_ms:.0f}ms)"
        )
