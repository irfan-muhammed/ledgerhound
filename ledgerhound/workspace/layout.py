"""
ledgerhound.workspace.layout
=========================

WorkspaceLayout -- the filesystem spine of every Ledgerhound run.

All artifacts (datasets, samples, reports, Trino jobs, memory) live
under one root directory. Nothing escapes the workspace. Every path is
derived from this single source of truth.

Directory contract
------------------
ledgerhound_workspace/
+-- datasets/           registered datasets (immutable raw + metadata)
+-- samples/            certified samples (data + validation certificate)
+-- reports/            EDA reports (findings, charts, notebooks, digests)
+-- trino_jobs/         async Trino job state (SQL, metadata, results)
+-- memory/
    +-- datasets/       per-dataset knowledge files (.md, living documents)
    +-- knowledge/      cross-session intelligence (signal history, drift events)
    +-- sessions/       session log and per-session context snapshots
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspaceLayout:
    root: Path

    # -- Top-level directories ----------------------------------------------

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def trino_jobs_dir(self) -> Path:
        return self.root / "trino_jobs"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def books_dir(self) -> Path:
        return self.root / "books"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def memory_datasets_dir(self) -> Path:
        return self.memory_dir / "datasets"

    @property
    def memory_knowledge_dir(self) -> Path:
        return self.memory_dir / "knowledge"

    @property
    def memory_sessions_dir(self) -> Path:
        return self.memory_dir / "sessions"

    # -- Registry paths -----------------------------------------------------

    @property
    def dataset_registry_path(self) -> Path:
        return self.datasets_dir / "registry.json"

    @property
    def sample_registry_path(self) -> Path:
        return self.samples_dir / "registry.json"

    @property
    def report_registry_path(self) -> Path:
        return self.reports_dir / "registry.json"

    @property
    def job_registry_path(self) -> Path:
        return self.trino_jobs_dir / "registry.json"

    @property
    def memory_index_path(self) -> Path:
        return self.memory_dir / "index.json"

    @property
    def sessions_log_path(self) -> Path:
        return self.memory_sessions_dir / "sessions.jsonl"

    # -- Per-artifact directories -------------------------------------------

    def dataset_dir(self, name: str, version: str) -> Path:
        """ledgerhound_workspace/datasets/{name}_v{version}/"""
        return self.datasets_dir / f"{name}_v{version}"

    def dataset_raw_dir(self, name: str, version: str) -> Path:
        return self.dataset_dir(name, version) / "raw"

    def sample_dir(self, name: str, version: str) -> Path:
        """ledgerhound_workspace/samples/{name}_v{version}/"""
        return self.samples_dir / f"{name}_v{version}"

    def sample_data_dir(self, name: str, version: str) -> Path:
        return self.sample_dir(name, version) / "data"

    def sample_validation_dir(self, name: str, version: str) -> Path:
        return self.sample_dir(name, version) / "validation"

    def report_dir(self, report_id: str) -> Path:
        """ledgerhound_workspace/reports/{report_id}/"""
        return self.reports_dir / report_id

    def report_findings_dir(self, report_id: str) -> Path:
        return self.report_dir(report_id) / "findings"

    def report_charts_dir(self, report_id: str) -> Path:
        return self.report_dir(report_id) / "charts"

    def report_notebooks_dir(self, report_id: str) -> Path:
        return self.report_dir(report_id) / "notebooks"

    def report_certificates_dir(self, report_id: str) -> Path:
        return self.report_dir(report_id) / "certificates"

    def trino_job_dir(self, job_id: str) -> Path:
        """ledgerhound_workspace/trino_jobs/{job_id}/"""
        return self.trino_jobs_dir / job_id

    def trino_job_result_dir(self, job_id: str) -> Path:
        return self.trino_job_dir(job_id) / "result"

    def book_features_dir(self, book_key: str) -> Path:
        """ledgerhound_workspace/features/{book_key}/ -- the ONE place a
        generated feature table is allowed to land (V3-049). Keyed the
        same way a book is (ledgerhound.knowledge.dataset_card.card_key), so
        every feature export from every story/chapter on one dataset
        collects in one folder instead of scattering across run folders
        and launch-cwd writes."""
        return self.features_dir / book_key

    def book_dir(self, book_key: str) -> Path:
        """ledgerhound_workspace/books/{book_key}/"""
        return self.books_dir / book_key

    # -- Memory paths --------------------------------------------------------

    def dataset_knowledge_path(self, dataset_name: str) -> Path:
        """Living markdown knowledge file for a dataset."""
        return self.memory_datasets_dir / f"{dataset_name}.md"

    def dataset_knowledge_entries_path(self, dataset_name: str) -> Path:
        """Structured JSONL entries backing the markdown knowledge file."""
        return self.memory_datasets_dir / f"{dataset_name}.jsonl"

    def session_context_path(self, session_id: str) -> Path:
        return self.memory_sessions_dir / f"{session_id}.json"

    @property
    def feature_signal_history_path(self) -> Path:
        return self.memory_knowledge_dir / "feature_signal_history.json"

    @property
    def drift_events_path(self) -> Path:
        return self.memory_knowledge_dir / "drift_events.json"

    @property
    def typology_evolution_path(self) -> Path:
        return self.memory_knowledge_dir / "typology_evolution.json"

    # -- Initialization -----------------------------------------------------

    def initialize(self) -> None:
        """Create the full directory tree. Safe to call multiple times."""
        dirs = [
            self.datasets_dir,
            self.samples_dir,
            self.reports_dir,
            self.trino_jobs_dir,
            self.features_dir,
            self.books_dir,
            self.memory_datasets_dir,
            self.memory_knowledge_dir,
            self.memory_sessions_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Seed empty index if not present
        if not self.memory_index_path.exists():
            self.memory_index_path.write_text(
                json.dumps({"datasets": {}, "features": {}, "created_at": _now()}, indent=2),
                encoding="utf-8",
            )

    # Convenience alias used by CLI
    def init(self) -> None:
        """Alias for initialize() -- creates the workspace directory tree."""
        self.initialize()

    def exists(self) -> bool:
        return self.root.exists()

    # -- Factories ----------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path) -> "WorkspaceLayout":
        return cls(root=Path(path).resolve())

    @classmethod
    def from_cwd(cls, name: str = "ledgerhound_workspace") -> "WorkspaceLayout":
        return cls(root=Path.cwd() / name)

    def __repr__(self) -> str:
        return f"WorkspaceLayout(root={self.root})"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
