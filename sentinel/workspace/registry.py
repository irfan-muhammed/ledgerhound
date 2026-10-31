"""
sentinel.workspace.registry
============================

Four registries, one design:
  DatasetRegistry  -- every dataset version ever registered
  SampleRegistry   -- every certified sample ever produced
  ReportRegistry   -- every EDA report ever written
  JobRegistry      -- every Trino job ever submitted

Each registry is a single JSON file. Entries are appended; nothing is
deleted. The registries are the source of truth for what exists in the
workspace and how artifacts relate to each other.
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


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


# -- Dataset Registry -------------------------------------------------------


@dataclass
class DatasetEntry:
    """One registered dataset version."""

    name: str
    version: str                        # "1", "2", "2b_cleaned", etc.
    source: str                         # "trino_pull", "local_file", "manual"
    path: str                           # relative path from workspace root
    registered_at: str = field(default_factory=_now)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    size_bytes: Optional[int] = None
    schema_hash: Optional[str] = None  # hash of column names + dtypes
    parent_name: Optional[str] = None
    parent_version: Optional[str] = None
    transformation: Optional[str] = None  # what was applied to produce this version
    trino_job_id: Optional[str] = None    # if pulled via Trino
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}:v{self.version}"


@dataclass
class DatasetRegistry:
    path: Path
    _entries: dict[str, DatasetEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def register(self, entry: DatasetEntry) -> DatasetEntry:
        """Register a new dataset version. Raises if key already exists."""
        if entry.key in self._entries:
            raise ValueError(f"Dataset {entry.key} already registered. Bump the version.")
        self._entries[entry.key] = entry
        self._save()
        return entry

    def get(self, name: str, version: str) -> Optional[DatasetEntry]:
        return self._entries.get(f"{name}:v{version}")

    def latest(self, name: str) -> Optional[DatasetEntry]:
        """Return the most recently registered version for a dataset name."""
        candidates = [e for e in self._entries.values() if e.name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.registered_at)

    def list_versions(self, name: str) -> list[DatasetEntry]:
        return sorted(
            [e for e in self._entries.values() if e.name == name],
            key=lambda e: e.registered_at,
        )

    def all(self) -> list[DatasetEntry]:
        return list(self._entries.values())

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: DatasetEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2),
            encoding="utf-8",
        )


# -- Sample Registry --------------------------------------------------------


@dataclass
class SampleEntry:
    """One certified sample."""

    name: str
    version: str
    path: str                           # relative path from workspace root
    created_at: str = field(default_factory=_now)
    fraud_dataset_key: Optional[str] = None    # DatasetEntry.key
    nonfr_dataset_key: Optional[str] = None    # DatasetEntry.key
    sampling_strategy: str = ""                # "hard_negative", "stratified", "random"
    fraud_rows: Optional[int] = None
    nonfr_rows: Optional[int] = None
    ratio: Optional[float] = None              # nonfr / fraud
    certified: bool = False                    # True once certificate.json written
    certificate_path: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}:v{self.version}"


@dataclass
class SampleRegistry:
    path: Path
    _entries: dict[str, SampleEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def register(self, entry: SampleEntry) -> SampleEntry:
        if entry.key in self._entries:
            raise ValueError(f"Sample {entry.key} already registered.")
        self._entries[entry.key] = entry
        self._save()
        return entry

    def certify(self, name: str, version: str, certificate_path: str) -> None:
        key = f"{name}:v{version}"
        if key not in self._entries:
            raise KeyError(f"Sample {key} not found.")
        self._entries[key].certified = True
        self._entries[key].certificate_path = certificate_path
        self._save()

    def get(self, name: str, version: str) -> Optional[SampleEntry]:
        return self._entries.get(f"{name}:v{version}")

    def latest(self, name: str) -> Optional[SampleEntry]:
        candidates = [e for e in self._entries.values() if e.name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.created_at)

    def all(self) -> list[SampleEntry]:
        return list(self._entries.values())

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: SampleEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2),
            encoding="utf-8",
        )


# -- Report Registry --------------------------------------------------------


@dataclass
class ReportEntry:
    """One EDA report."""

    report_id: str = field(default_factory=lambda: f"report_{_short_id()}")
    title: str = ""
    path: str = ""
    created_at: str = field(default_factory=_now)
    session_id: Optional[str] = None
    datasets_used: list[str] = field(default_factory=list)
    sample_keys: list[str] = field(default_factory=list)
    finding_count: int = 0
    chart_count: int = 0
    has_notebook: bool = False
    has_certificate: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class ReportRegistry:
    path: Path
    _entries: dict[str, ReportEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def register(self, entry: ReportEntry) -> ReportEntry:
        self._entries[entry.report_id] = entry
        self._save()
        return entry

    def update(self, report_id: str, **kwargs) -> None:
        if report_id not in self._entries:
            raise KeyError(f"Report {report_id} not found.")
        for k, v in kwargs.items():
            setattr(self._entries[report_id], k, v)
        self._save()

    def get(self, report_id: str) -> Optional[ReportEntry]:
        return self._entries.get(report_id)

    def recent(self, n: int = 10) -> list[ReportEntry]:
        return sorted(self._entries.values(), key=lambda e: e.created_at, reverse=True)[:n]

    def all(self) -> list[ReportEntry]:
        return list(self._entries.values())

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: ReportEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2),
            encoding="utf-8",
        )


# -- Job Registry -----------------------------------------------------------


@dataclass
class JobEntry:
    """One async Trino job."""

    job_id: str = field(default_factory=lambda: f"job_{_short_id()}")
    sql_path: str = ""                  # relative path to query.sql
    result_path: Optional[str] = None  # relative path to result parquet
    submitted_at: str = field(default_factory=_now)
    completed_at: Optional[str] = None
    status: str = "pending"            # pending | running | done | failed | cancelled
    estimated_rows: Optional[int] = None
    estimated_seconds_min: Optional[int] = None
    estimated_seconds_max: Optional[int] = None
    actual_runtime_seconds: Optional[float] = None
    session_id: Optional[str] = None
    dataset_context: str = ""          # which dataset/question this serves
    approved_by_user: bool = False
    error: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class JobRegistry:
    path: Path
    _entries: dict[str, JobEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def register(self, entry: JobEntry) -> JobEntry:
        self._entries[entry.job_id] = entry
        self._save()
        return entry

    def update_status(
        self,
        job_id: str,
        status: str,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        actual_runtime_seconds: Optional[float] = None,
    ) -> None:
        if job_id not in self._entries:
            raise KeyError(f"Job {job_id} not found.")
        entry = self._entries[job_id]
        entry.status = status
        if result_path:
            entry.result_path = result_path
        if error:
            entry.error = error
        if actual_runtime_seconds is not None:
            entry.actual_runtime_seconds = actual_runtime_seconds
        if status in ("done", "failed", "cancelled"):
            entry.completed_at = _now()
        self._save()

    def get(self, job_id: str) -> Optional[JobEntry]:
        return self._entries.get(job_id)

    def pending(self) -> list[JobEntry]:
        return [e for e in self._entries.values() if e.status == "pending"]

    def for_session(self, session_id: str) -> list[JobEntry]:
        return [e for e in self._entries.values() if e.session_id == session_id]

    def all(self) -> list[JobEntry]:
        return list(self._entries.values())

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: JobEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2),
            encoding="utf-8",
        )
