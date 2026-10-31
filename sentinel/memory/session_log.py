"""
sentinel.memory.session_log
============================

SessionLog -- append-only history of every Sentinel analysis session.

Every session records:
  - which datasets were touched
  - how many findings were produced
  - how many Trino jobs were submitted
  - a one-line summary of what was done

This creates an auditable timeline of analytical activity. When a model
degrades in production, the session log is the first place to check:
which session last touched the training data? Were there drift warnings?
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"sess_{ts}_{uuid.uuid4().hex[:6]}"


@dataclass
class SessionRecord:
    """One completed or in-progress analysis session."""

    session_id: str = field(default_factory=_session_id)
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    datasets_used: list[str] = field(default_factory=list)
    questions_asked: int = 0
    findings_produced: int = 0
    trino_jobs_submitted: int = 0
    trino_jobs_completed: int = 0
    summary: str = ""           # one-line description of what the session did
    model_track: str = "fraud"  # "fraud" | "mule" | "both"
    tags: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.ended_at is not None

    @property
    def duration_display(self) -> str:
        if not self.ended_at:
            return "in progress"
        from datetime import datetime as dt
        start = dt.fromisoformat(self.started_at)
        end = dt.fromisoformat(self.ended_at)
        secs = int((end - start).total_seconds())
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs // 3600}h {(secs % 3600) // 60}m"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SessionLog:
    """
    Append-only log of all analysis sessions.

    Backed by a JSONL file -- one record per line, one line per session.
    Sessions are written on start and updated (line rewrite) on end.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._sessions: dict[str, SessionRecord] = {}
        self._load()

    # -- Session lifecycle --------------------------------------------------

    def start(
        self,
        datasets: Optional[list[str]] = None,
        model_track: str = "fraud",
        tags: Optional[list[str]] = None,
    ) -> SessionRecord:
        """Open a new session. Returns the SessionRecord with its ID."""
        record = SessionRecord(
            datasets_used=datasets or [],
            model_track=model_track,
            tags=tags or [],
        )
        self._sessions[record.session_id] = record
        self._save()
        return record

    def end(
        self,
        session_id: str,
        summary: str,
        findings_produced: int = 0,
        trino_jobs_submitted: int = 0,
        trino_jobs_completed: int = 0,
        questions_asked: int = 0,
    ) -> SessionRecord:
        """Close a session with final stats."""
        record = self._get_or_raise(session_id)
        record.ended_at = _now()
        record.summary = summary
        record.findings_produced = findings_produced
        record.trino_jobs_submitted = trino_jobs_submitted
        record.trino_jobs_completed = trino_jobs_completed
        record.questions_asked = questions_asked
        self._save()
        return record

    def touch(self, session_id: str, **kwargs) -> None:
        """Update in-progress session stats mid-session."""
        record = self._get_or_raise(session_id)
        for k, v in kwargs.items():
            if hasattr(record, k):
                setattr(record, k, v)
        self._save()

    # -- Query --------------------------------------------------------------

    def get(self, session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(session_id)

    def recent(self, n: int = 20) -> list[SessionRecord]:
        return sorted(
            self._sessions.values(),
            key=lambda s: s.started_at,
            reverse=True,
        )[:n]

    def for_dataset(self, dataset_name: str) -> list[SessionRecord]:
        return [
            s for s in self._sessions.values()
            if dataset_name in s.datasets_used
        ]

    def all(self) -> list[SessionRecord]:
        return sorted(self._sessions.values(), key=lambda s: s.started_at)

    def context_for_agent(self, dataset_name: Optional[str] = None, n: int = 5) -> str:
        """
        One-paragraph summary of recent session activity for agent context.
        Optionally filtered to sessions that touched a specific dataset.
        """
        sessions = self.for_dataset(dataset_name) if dataset_name else self.recent(n)
        sessions = sorted(sessions, key=lambda s: s.started_at, reverse=True)[:n]

        if not sessions:
            return "No prior sessions recorded."

        lines = ["### Recent analysis sessions\n"]
        for s in sessions:
            lines.append(
                f"- **{s.started_at[:10]}** ({s.duration_display}) -- "
                f"{s.findings_produced} findings, {s.trino_jobs_submitted} Trino jobs. "
                f"{s.summary or 'No summary recorded.'}"
            )
        return "\n".join(lines)

    # -- Persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = SessionRecord.from_dict(json.loads(line))
                    self._sessions[r.session_id] = r
                except Exception:
                    pass

    def _save(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(s.to_dict()) for s in self._sessions.values()]
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _get_or_raise(self, session_id: str) -> SessionRecord:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        return self._sessions[session_id]

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"SessionLog(path={self.log_path}, sessions={len(self._sessions)})"
