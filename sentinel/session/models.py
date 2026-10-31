"""
sentinel.session.models
=======================

Session -- the single source of truth for one experiment.

Persists to two files inside workspace_dir/:
  session.json      -- metadata, datasets, goal, status, checkpoint
  messages.jsonl    -- append-only full conversation log (one JSON object per line)

The messages log is the source of truth for resuming context. The
ContextCompressor reads it and produces a compressed_context.json when
the raw log grows too large for the LLM's context window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


GoalType  = Literal["eda", "fraud_modelling", "mule_graph", "sampling", "custom"]
ReportFmt = Literal["html", "jupyter", "both", "markdown"]


@dataclass
class DatasetEntry:
    """One dataset file registered in the session."""
    path: str
    name: str
    rows: int | None = None
    cols: int | None = None
    format: str = ""
    digest_path: str = ""
    description: str = ""        # LLM-enriched + user-confirmed description
    clarifications: dict = field(default_factory=dict)  # Q->A pairs from enrichment


@dataclass
class StepCheckpoint:
    """Tracks which auto-run steps are complete so resume can skip them."""
    step_index: int       # last completed step (1-based)
    step_title: str
    output_summary: str   # compressed output for context rebuild
    completed_at: str


@dataclass
class Session:
    id: str
    name: str
    created_at: str
    workspace_dir: str

    # User-supplied + LLM-enriched context
    description: str = ""
    llm_enrichment: str = ""     # what the LLM said about the data
    goal: GoalType | str = "eda"
    goal_detail: str = ""
    report_format: ReportFmt = "both"

    # Data
    datasets: list[DatasetEntry] = field(default_factory=list)

    # Lifecycle
    status: str = "setup"        # setup | profiling | running | paused | done | error
    last_run_id: str = ""
    llm_config: str = ""

    # Checkpoint -- which step we paused at
    checkpoints: list[StepCheckpoint] = field(default_factory=list)
    last_completed_step: int = 0  # 0 = not started

    # Context compression metadata
    context_compressed_at: int = 0   # message count at last compression
    compressed_context_path: str = ""

    # GPU allocation plan, populated on first run by GPUPlanner.
    # Empty dict => not planned yet; runner will detect at start.
    gpu_plan: dict = field(default_factory=dict)

    # Execution plan, frozen on first auto-run so resume reuses the SAME
    # steps instead of regenerating a (non-deterministic) new plan that would
    # misalign with saved checkpoints. Empty => not yet planned.
    plan: list = field(default_factory=list)

    # ------------------------------------------------------------------ #

    @classmethod
    def new(cls, name: str, workspace_root: str = "sentinel_workspace") -> "Session":
        ts = datetime.now(timezone.utc)
        slug = _slugify(name)
        sid = f"{slug}-{ts.strftime('%Y%m%d-%H%M%S')}"
        ws = Path(workspace_root) / "sessions" / sid
        ws.mkdir(parents=True, exist_ok=True)
        for sub in ("digest", "outputs", "reports", "context"):
            (ws / sub).mkdir(exist_ok=True)
        return cls(
            id=sid,
            name=name,
            created_at=ts.isoformat(),
            workspace_dir=str(ws.resolve()),
        )

    def save(self) -> None:
        data = asdict(self)
        p = Path(self.workspace_dir) / "session.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, session_dir: Path) -> "Session":
        data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        # Forward-compat: old sessions lack newer fields -- supply defaults
        raw_datasets = data.pop("datasets", [])
        datasets = []
        for d in raw_datasets:
            d.setdefault("clarifications", {})   # added in wizard update
            d.setdefault("description", "")
            datasets.append(DatasetEntry(**d))
        checkpoints = [StepCheckpoint(**c) for c in data.pop("checkpoints", [])]
        # Strip unknown top-level keys so old sessions load cleanly
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        obj = cls(**data)
        obj.datasets = datasets
        obj.checkpoints = checkpoints
        return obj

    # -- Message log -----------------------------------------------------

    def append_message(self, role: str, content: str, tag: str = "") -> None:
        """Append one message to messages.jsonl. Never overwrites."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,           # user | assistant | system | tool
            "content": content,
            "tag": tag,             # wizard | step | finding | summary | error
        }
        p = Path(self.workspace_dir) / "messages.jsonl"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_messages(self) -> list[dict]:
        p = Path(self.workspace_dir) / "messages.jsonl"
        if not p.exists():
            return []
        msgs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return msgs

    def message_count(self) -> int:
        p = Path(self.workspace_dir) / "messages.jsonl"
        if not p.exists():
            return 0
        return sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())

    # -- Checkpoint ------------------------------------------------------

    def save_checkpoint(self, step_index: int, step_title: str, output_summary: str) -> None:
        cp = StepCheckpoint(
            step_index=step_index,
            step_title=step_title,
            output_summary=output_summary,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        # Replace existing checkpoint for this step if present
        self.checkpoints = [c for c in self.checkpoints if c.step_index != step_index]
        self.checkpoints.append(cp)
        self.last_completed_step = max(self.last_completed_step, step_index)
        self.save()

    def completed_step_indices(self) -> set[int]:
        return {c.step_index for c in self.checkpoints}

    # -- Convenience -----------------------------------------------------

    def add_dataset(self, path: str, rows: int | None = None,
                    cols: int | None = None) -> DatasetEntry:
        p = Path(path)
        entry = DatasetEntry(
            path=str(p.resolve()),
            name=p.stem,
            rows=rows,
            cols=cols,
            format=p.suffix.lstrip("."),
        )
        self.datasets.append(entry)
        self.save()
        return entry

    def goal_label(self) -> str:
        labels = {
            "eda":              "Exploratory Data Analysis",
            "fraud_modelling":  "Fraud Modelling (sample + features)",
            "mule_graph":       "Mule Detection (graph analysis)",
            "sampling":         "Data Sampling / PSI Certification",
            "custom":           self.goal_detail or "Custom",
        }
        return labels.get(self.goal, str(self.goal))

    def digest_dir(self) -> Path:
        return Path(self.workspace_dir) / "digest"

    def outputs_dir(self) -> Path:
        return Path(self.workspace_dir) / "outputs"

    def reports_dir(self) -> Path:
        return Path(self.workspace_dir) / "reports"

    def context_dir(self) -> Path:
        return Path(self.workspace_dir) / "context"


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]
