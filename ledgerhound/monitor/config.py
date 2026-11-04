"""
ledgerhound.monitor.config
=======================

Drift Monitor configuration: YAML-backed schema for what to watch, how
often, and where to send alerts.

Config file location
--------------------
``<workspace>/monitor/config.yaml``

Example
-------
```yaml
dataset_name: upi_fraud
fraud_data:  data/fraud.parquet
nonfr_data:  data/nonfr.parquet
raw_txn_data: data/raw_txn.parquet   # optional, only for typology evolution
features:
  - sim_change_count
  - device_change_count
  - night_txn_rate
  - failed_txn_rate
  - new_benef_rate
  - txn_velocity_7d
period_granularity: quarterly        # quarterly | monthly | weekly | daily
match_threshold: 0.60
schedule_interval_seconds: 3600
alerts:
  signal_decay_severity: WARNING     # any decay >= this -> alert
  evolution_severity:    WARNING
  emit_info_events:      false
webhooks:
  - url:  https://hooks.slack.com/services/...
    headers:
      Authorization: "Bearer $SLACK_TOKEN"
    on_severity: [WARNING, FAIL]
    on_event_types: []               # empty = all
    timeout: 10.0
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ledgerhound.monitor.schedule import ScheduleConfig
from ledgerhound.monitor.trino_source import TrinoSourceConfig

log = logging.getLogger(__name__)


# -- Period granularity helper ----------------------------------------------


_GRANULARITIES = {"quarterly", "monthly", "weekly", "daily"}


def derive_period(granularity: str, dt: Optional[datetime] = None) -> str:
    """Convert a datetime into a period label for the given granularity.

    quarterly  -> ``YYYY-Q1`` ... ``YYYY-Q4``
    monthly    -> ``YYYY-MM``
    weekly     -> ``YYYY-Www``  (ISO week)
    daily      -> ``YYYY-MM-DD``
    """
    if granularity not in _GRANULARITIES:
        raise ValueError(
            f"Unknown granularity {granularity!r}; expected one of "
            f"{sorted(_GRANULARITIES)}"
        )
    dt = dt or datetime.now(timezone.utc)
    if granularity == "quarterly":
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"
    if granularity == "monthly":
        return f"{dt.year}-{dt.month:02d}"
    if granularity == "weekly":
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if granularity == "daily":
        return dt.strftime("%Y-%m-%d")
    raise AssertionError("unreachable")


# -- Dataclasses ------------------------------------------------------------


@dataclass
class WebhookConfig:
    """One webhook target.

    The emitter expands ``$VARNAME`` references inside ``headers`` against
    the calling process's environment, so secrets aren't checked into YAML.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    on_severity: list[str] = field(default_factory=lambda: ["WARNING", "FAIL"])
    on_event_types: list[str] = field(default_factory=list)   # empty = all
    timeout: float = 10.0
    name: str = ""                                            # optional label

    @classmethod
    def from_dict(cls, d: dict) -> "WebhookConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AlertThresholds:
    """Severity floors for the cycle-level overall_alert decision."""

    signal_decay_severity: str = "WARNING"
    evolution_severity:    str = "WARNING"
    emit_info_events:      bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "AlertThresholds":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DriftMonitorConfig:
    """Full config for one Drift Monitor on one dataset."""

    dataset_name:               str
    fraud_data:                 str
    nonfr_data:                 str
    raw_txn_data:               Optional[str] = None
    features:                   list[str] = field(default_factory=list)

    period_granularity:         str = "quarterly"        # see derive_period
    match_threshold:            float = 0.60
    # Deprecated (kept for backward-compat with existing config.yaml files):
    # a bare interval, now folded into `schedule` (ScheduleConfig) on load
    # when no explicit `schedule` block is present. New configs should set
    # `schedule` directly -- see ledgerhound.monitor.schedule.ScheduleConfig
    # for interval vs. calendar (specific days + time-of-day) modes.
    schedule_interval_seconds:  int   = 3600
    schedule:                   ScheduleConfig = field(default_factory=ScheduleConfig)

    # Optional: pull fraud/nonfr/raw_txn straight from Trino instead of
    # the file paths below. See ledgerhound.monitor.trino_source for the
    # pre-approved/hash-pinned SQL safety model this requires.
    trino:                      TrinoSourceConfig = field(default_factory=TrinoSourceConfig)

    # Optional: automatically investigate the worst alert of a cycle with
    # a real, autonomous LLM session and record a grounded finding. Off
    # by default -- see ledgerhound.monitor.investigate's module docstring.
    auto_investigate:           bool = False

    alerts:                     AlertThresholds = field(default_factory=AlertThresholds)
    webhooks:                   list[WebhookConfig] = field(default_factory=list)

    # Used by the monitor to seed the period when none is provided
    fixed_period:               Optional[str] = None

    # -- Serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DriftMonitorConfig":
        alerts = d.get("alerts") or {}
        if isinstance(alerts, dict):
            alerts = AlertThresholds.from_dict(alerts)
        webhooks = []
        for w in d.get("webhooks") or []:
            if isinstance(w, dict):
                webhooks.append(WebhookConfig.from_dict(w))
            else:
                webhooks.append(w)

        if "schedule" in d and d["schedule"]:
            schedule = ScheduleConfig.from_dict(d["schedule"])
        else:
            # Backward compat: no explicit `schedule` block -- fold the
            # legacy bare interval into an interval-mode ScheduleConfig.
            schedule = ScheduleConfig(
                mode="interval",
                interval_seconds=d.get("schedule_interval_seconds", 3600),
            )

        trino = TrinoSourceConfig.from_dict(d.get("trino"))

        kw = {k: v for k, v in d.items()
              if k in cls.__dataclass_fields__
              and k not in ("alerts", "webhooks", "schedule", "trino")}
        return cls(alerts=alerts, webhooks=webhooks, schedule=schedule, trino=trino, **kw)

    # -- Workspace I/O -------------------------------------------------

    @staticmethod
    def config_path(workspace) -> Path:
        return Path(workspace.root) / "monitor" / "config.yaml"

    @classmethod
    def load(cls, workspace) -> "DriftMonitorConfig":
        import yaml
        p = cls.config_path(workspace)
        if not p.exists():
            raise FileNotFoundError(
                f"No monitor config at {p}. "
                f"Run `ledgerhound monitor init` first."
            )
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.from_dict(data or {})

    def save(self, workspace) -> Path:
        import yaml
        p = self.config_path(workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return p

    # -- Validation ----------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of issues; empty list means OK."""
        issues: list[str] = []
        if not self.dataset_name:
            issues.append("dataset_name is required")
        if self.trino.enabled:
            # Trino-sourced: fraud_data/nonfr_data file paths aren't used.
            issues.extend(self.trino.validate())
        else:
            if not self.fraud_data:
                issues.append("fraud_data path is required (or set trino.enabled=true)")
            if not self.nonfr_data:
                issues.append("nonfr_data path is required (or set trino.enabled=true)")
        if self.period_granularity not in _GRANULARITIES:
            issues.append(
                f"period_granularity={self.period_granularity!r} "
                f"must be one of {sorted(_GRANULARITIES)}"
            )
        issues.extend(self.schedule.validate())
        if not self.features:
            issues.append(
                "features list is empty -- at least one feature is required "
                "for signal-decay tracking"
            )
        for i, w in enumerate(self.webhooks):
            if not w.url:
                issues.append(f"webhook[{i}] has no url")
            for sev in w.on_severity:
                if sev not in ("INFO", "WARNING", "FAIL"):
                    issues.append(
                        f"webhook[{i}] on_severity={sev!r} is not a valid severity"
                    )
        return issues
