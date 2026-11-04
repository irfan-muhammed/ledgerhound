"""
ledgerhound.trino.jobs
===================

AsyncJobRegistry -- tracks every Trino job submitted in this workspace.

The async pattern:
  1. Agent generates SQL and submits via TrinoClient.submit_async()
  2. Job is registered here with status=pending
  3. Agent checkpoints its state (findings log, session context)
  4. Agent returns control -- user can close the terminal
  5. Hours later: user runs `ledgerhound resume --job <job_id>`
  6. TrinoClient.execute_job(job) runs and writes result parquet
  7. Agent reloads checkpoint and continues from where it left off

The registry is a single JSON file -- one entry per job. It is the
source of truth for what jobs exist, their status, and where results
landed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ledgerhound.workspace.layout import WorkspaceLayout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id() -> str:
    return f"job_{uuid.uuid4().hex[:8]}"


@dataclass
class TrinoJob:
    """One async Trino job. Persisted to the job registry."""

    job_id: str = field(default_factory=_job_id)
    sql: str = ""
    session_id: str = ""
    dataset_context: str = ""
    status: str = "pending"             # pending | running | done | failed | cancelled
    approved_by_user: bool = False
    submitted_at: str = field(default_factory=_now)
    completed_at: Optional[str] = None
    estimated_rows: Optional[int] = None
    estimated_seconds_min: Optional[int] = None
    estimated_seconds_max: Optional[int] = None
    actual_runtime_seconds: Optional[float] = None
    sql_path: Optional[str] = None      # relative path to persisted query.sql
    result_path: Optional[str] = None   # relative path to result parquet
    error: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def is_pending(self) -> bool:
        return self.status in ("pending", "running")

    @property
    def estimated_runtime_display(self) -> str:
        if self.estimated_seconds_min is None:
            return "unknown"
        mn, mx = self.estimated_seconds_min, self.estimated_seconds_max
        return f"{_fmt(mn)} - {_fmt(mx)}"


class AsyncJobRegistry:
    """
    Append-friendly registry of all Trino jobs.

    Backed by a single JSON file at `registry_path`. Load once, mutate
    in memory, persist on every change. Safe for single-process use.
    """

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._jobs: dict[str, TrinoJob] = {}
        self._load()

    def submit(
        self,
        sql: str,
        session_id: str,
        dataset_context: str,
        estimated_rows: Optional[int] = None,
        estimated_seconds_min: Optional[int] = None,
        estimated_seconds_max: Optional[int] = None,
        workspace: Optional["WorkspaceLayout"] = None,
    ) -> TrinoJob:
        """Register a new job and persist its SQL to disk."""
        job = TrinoJob(
            sql=sql,
            session_id=session_id,
            dataset_context=dataset_context,
            status="pending",
            approved_by_user=True,
            estimated_rows=estimated_rows,
            estimated_seconds_min=estimated_seconds_min,
            estimated_seconds_max=estimated_seconds_max,
        )

        # Persist the SQL file so it can be inspected or replayed later
        if workspace is not None:
            sql_dir = workspace.trino_job_dir(job.job_id)
            sql_dir.mkdir(parents=True, exist_ok=True)
            sql_file = sql_dir / "query.sql"
            sql_file.write_text(sql, encoding="utf-8")
            job.sql_path = str(sql_file.relative_to(workspace.root))

            # Write job metadata alongside the SQL
            meta_file = sql_dir / "metadata.json"
            meta_file.write_text(
                json.dumps(
                    {
                        "job_id": job.job_id,
                        "session_id": session_id,
                        "dataset_context": dataset_context,
                        "submitted_at": job.submitted_at,
                        "estimated_rows": estimated_rows,
                        "estimated_seconds_min": estimated_seconds_min,
                        "estimated_seconds_max": estimated_seconds_max,
                        "status": "pending",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        self._jobs[job.job_id] = job
        self._save()
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        actual_runtime_seconds: Optional[float] = None,
    ) -> TrinoJob:
        job = self._get_or_raise(job_id)
        job.status = status
        if result_path:
            job.result_path = result_path
        if error:
            job.error = error
        if actual_runtime_seconds is not None:
            job.actual_runtime_seconds = actual_runtime_seconds
        if status in ("done", "failed", "cancelled"):
            job.completed_at = _now()
        self._save()
        return job

    def get(self, job_id: str) -> Optional[TrinoJob]:
        return self._jobs.get(job_id)

    def pending(self) -> list[TrinoJob]:
        return [j for j in self._jobs.values() if j.status == "pending"]

    def for_session(self, session_id: str) -> list[TrinoJob]:
        return [j for j in self._jobs.values() if j.session_id == session_id]

    def all(self) -> list[TrinoJob]:
        return list(self._jobs.values())

    def summary_table(self) -> list[dict]:
        """Return a lightweight summary of all jobs for display."""
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "context": j.dataset_context[:50],
                "submitted": j.submitted_at[:10],
                "runtime": (
                    f"{j.actual_runtime_seconds:.0f}s"
                    if j.actual_runtime_seconds
                    else j.estimated_runtime_display
                ),
            }
            for j in sorted(self._jobs.values(), key=lambda j: j.submitted_at)
        ]

    def _get_or_raise(self, job_id: str) -> TrinoJob:
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found in registry at {self.registry_path}")
        return self._jobs[job_id]

    def _load(self) -> None:
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for jid, jdata in raw.items():
                self._jobs[jid] = TrinoJob(**jdata)

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(
                {jid: asdict(job) for jid, job in self._jobs.items()},
                indent=2,
            ),
            encoding="utf-8",
        )


def _fmt(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h {(s % 3600) // 60}m"
