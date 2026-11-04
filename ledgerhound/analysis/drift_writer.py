"""
ledgerhound.analysis.drift_writer
==============================

Append drift events to the workspace's unified ``drift_events.jsonl``.

A *drift event* is any time-bounded change Ledgerhound detects in the data
that an analyst should be aware of:

  - **signal_decay**       a feature lost discriminative power vs baseline
  - **signal_emerging**    a feature gained discriminative power
  - **cluster_growth**     a typology cluster grew past WARN/FAIL threshold
  - **cluster_emerging**   a brand-new typology cluster appeared
  - **cluster_decline**    a typology cluster shrank significantly
  - **cluster_disappeared** a previously-tracked cluster vanished
  - **centroid_drift**     a cluster's feature signature shifted

Format: one JSON object per line, schema versioned via ``schema``.

    {"schema": "drift_event/v1",
     "event_type": "signal_decay",
     "period": "2024-Q2",
     "severity": "FAIL",
     "feature": "sim_change_count",
     "claim": "KS dropped 32% vs 2024-Q1 baseline",
     "details": {...},
     "session_id": "playbook_2024Q2",
     "generated_at": "2026-..."}

This is the chronological audit trail the Drift Monitor (future C1) will
poll and the Scorecard's data_quality component will roll up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ledgerhound.workspace.layout import WorkspaceLayout

log = logging.getLogger(__name__)


_SCHEMA = "drift_event/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DriftEvent:
    """One drift event in the unified schema."""

    event_type: str               # see module docstring for valid values
    period: str
    severity: str                 # "INFO" | "WARNING" | "FAIL"
    claim: str                    # one-line summary
    feature: Optional[str] = None
    cluster_key: Optional[str] = None
    session_id: str = "unknown"
    generated_at: str = field(default_factory=_now)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = _SCHEMA
        # reorder: schema first
        return {"schema": _SCHEMA, **{k: v for k, v in d.items() if k != "schema"}}


class DriftEventWriter:
    """Append-only JSONL writer for drift events.

    Usage::

        writer = DriftEventWriter(workspace)
        writer.append(DriftEvent(
            event_type="signal_decay",
            period="2024-Q2",
            severity="FAIL",
            claim="...",
            feature="sim_change_count",
            session_id="run_001",
        ))
        writer.append_many([...])
    """

    def __init__(self, workspace: "WorkspaceLayout") -> None:
        self.path = Path(workspace.drift_events_path)

    def append(self, event: DriftEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def append_many(self, events: Iterable[DriftEvent]) -> int:
        events_list = list(events)
        if not events_list:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for ev in events_list:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")
        return len(events_list)

    # -- Read helpers ---------------------------------------------------

    def iter_events(
        self,
        event_type: Optional[str] = None,
        period: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> Iterable[dict]:
        if not self.path.exists():
            return
        sev_order = {"INFO": 0, "WARNING": 1, "FAIL": 2}
        min_rank = sev_order.get(min_severity, -1) if min_severity else -1
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_type and d.get("event_type") != event_type:
                    continue
                if period and d.get("period") != period:
                    continue
                if min_severity and sev_order.get(
                    d.get("severity", "INFO"), 0,
                ) < min_rank:
                    continue
                yield d

    def all_events(self, **kw) -> list[dict]:
        return list(self.iter_events(**kw))

    def count(self, **kw) -> int:
        return sum(1 for _ in self.iter_events(**kw))
