"""
sentinel.memory.knowledge
=========================

DatasetKnowledge -- the living institutional memory for one dataset.

Every session that touches a dataset can deposit insights here. The
knowledge accumulates indefinitely. Future sessions start with this
context pre-loaded, so the agent never rediscovers what is already known.

Two layers:
  1. Structured entries (JSONL) -- machine-readable, searchable by
     feature / type / date. Used to build agent context automatically.

  2. Markdown document (.md) -- human-readable summary rebuilt from
     entries. Useful for onboarding new team members.

This is distinct from DatasetMemory (sentinel.memory.models), which
records sampling design metadata. DatasetKnowledge records accumulated
EDA insights: quality issues, drift events, signal decay observations,
known anomalies, and analyst notes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.workspace.layout import WorkspaceLayout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_id() -> str:
    return f"k_{uuid.uuid4().hex[:8]}"


# -- Entry types ------------------------------------------------------------

class KnowledgeType:
    QUALITY_FLAG    = "quality_flag"      # null spike, encoding issue, truncation
    DRIFT_EVENT     = "drift_event"       # distribution shift detected
    SIGNAL_DECAY    = "signal_decay"      # feature losing discriminative power
    SIGNAL_EMERGING = "signal_emerging"   # feature gaining discriminative power
    ANOMALY         = "anomaly"           # unexpected value or pattern
    SAMPLING_NOTE   = "sampling_note"     # observation about sampling behavior
    PIPELINE_ISSUE  = "pipeline_issue"    # upstream data pipeline problem
    ANALYST_NOTE    = "analyst_note"      # free-form analyst observation
    TYPOLOGY        = "typology"          # fraud cluster / typology observation
    BOUNDARY        = "boundary"          # hard negative / decision boundary insight


@dataclass
class KnowledgeEntry:
    """One insight about a dataset, deposited by a session."""

    entry_id: str = field(default_factory=_entry_id)
    timestamp: str = field(default_factory=_now)
    session_id: str = ""
    entry_type: str = KnowledgeType.ANALYST_NOTE
    feature: Optional[str] = None       # which column/feature, if applicable
    claim: str = ""                     # the insight in one sentence
    detail: str = ""                    # supporting detail, stats, or context
    evidence_ref: Optional[str] = None  # finding_id or job_id that backs this
    confidence: float = 0.8             # 0.0-1.0
    period: Optional[str] = None        # data period this applies to, e.g. "2025-Q1"
    tags: list[str] = field(default_factory=list)
    still_valid: bool = True            # set False when superseded

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# -- Knowledge base ---------------------------------------------------------

class DatasetKnowledge:
    """
    Living knowledge base for one dataset.

    Usage
    -----
    kb = DatasetKnowledge.load(workspace, "upi_summary")
    kb.add(KnowledgeEntry(
        session_id="sess_abc",
        entry_type=KnowledgeType.QUALITY_FLAG,
        feature="merchant_category",
        claim="Null rate spikes to 18% after 2025-03-15 due to pipeline change.",
        confidence=0.95,
    ))
    # Inject into agent context at session start:
    context_str = kb.for_session_context()
    """

    def __init__(self, dataset_name: str, entries_path: Path, md_path: Path) -> None:
        self.dataset_name = dataset_name
        self.entries_path = entries_path
        self.md_path = md_path
        self._entries: list[KnowledgeEntry] = []
        self._load()

    # -- Factory -----------------------------------------------------------

    @classmethod
    def load(cls, workspace: "WorkspaceLayout", dataset_name: str) -> "DatasetKnowledge":
        return cls(
            dataset_name=dataset_name,
            entries_path=workspace.dataset_knowledge_entries_path(dataset_name),
            md_path=workspace.dataset_knowledge_path(dataset_name),
        )

    # -- Write --------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Deposit a new insight. Appends to JSONL and rebuilds the markdown."""
        self._entries.append(entry)
        self._append_jsonl(entry)
        self._rebuild_markdown()
        return entry

    def invalidate(self, entry_id: str, reason: str = "") -> None:
        """Mark an entry as no longer valid (superseded by newer finding)."""
        for e in self._entries:
            if e.entry_id == entry_id:
                e.still_valid = False
                break
        self._rewrite_jsonl()
        self._rebuild_markdown()

    # -- Read / Search ------------------------------------------------------

    def search(
        self,
        feature: Optional[str] = None,
        entry_type: Optional[str] = None,
        since: Optional[str] = None,
        valid_only: bool = True,
    ) -> list[KnowledgeEntry]:
        results = self._entries
        if valid_only:
            results = [e for e in results if e.still_valid]
        if feature:
            results = [e for e in results if e.feature == feature]
        if entry_type:
            results = [e for e in results if e.entry_type == entry_type]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results

    def for_session_context(self, max_entries: int = 30) -> str:
        """
        Compact knowledge summary for injection into agent system prompt.

        Returns a markdown block the agent sees at the start of every
        session. Prioritises: quality flags > drift events > signal decay
        > other. Caps at max_entries to stay within context limits.
        """
        valid = [e for e in self._entries if e.still_valid]
        if not valid:
            return f"No prior knowledge recorded for dataset '{self.dataset_name}'."

        # Priority ordering
        priority_order = [
            KnowledgeType.QUALITY_FLAG,
            KnowledgeType.PIPELINE_ISSUE,
            KnowledgeType.DRIFT_EVENT,
            KnowledgeType.SIGNAL_DECAY,
            KnowledgeType.SIGNAL_EMERGING,
            KnowledgeType.BOUNDARY,
            KnowledgeType.TYPOLOGY,
            KnowledgeType.SAMPLING_NOTE,
            KnowledgeType.ANOMALY,
            KnowledgeType.ANALYST_NOTE,
        ]
        priority = {t: i for i, t in enumerate(priority_order)}
        sorted_entries = sorted(valid, key=lambda e: (priority.get(e.entry_type, 99), e.timestamp))
        top = sorted_entries[:max_entries]

        lines = [f"### Known facts about dataset: `{self.dataset_name}`\n"]
        for e in top:
            feature_tag = f" [`{e.feature}`]" if e.feature else ""
            period_tag = f" ({e.period})" if e.period else ""
            lines.append(
                f"- **[{e.entry_type}]{feature_tag}{period_tag}** "
                f"(conf={e.confidence:.0%}): {e.claim}"
            )
            if e.detail:
                lines.append(f"  > {e.detail}")

        if len(valid) > max_entries:
            lines.append(f"\n*({len(valid) - max_entries} more entries not shown -- search the full knowledge base)*")

        return "\n".join(lines)

    def summary(self) -> str:
        """Full human-readable markdown summary (for the .md file)."""
        valid = [e for e in self._entries if e.still_valid]
        invalid = [e for e in self._entries if not e.still_valid]

        lines = [
            f"# Dataset Knowledge: {self.dataset_name}",
            f"\n*{len(valid)} active entries, {len(invalid)} superseded.*\n",
        ]

        # Group by type
        by_type: dict[str, list[KnowledgeEntry]] = {}
        for e in valid:
            by_type.setdefault(e.entry_type, []).append(e)

        for etype, entries in sorted(by_type.items()):
            lines.append(f"\n## {etype.replace('_', ' ').title()}\n")
            for e in sorted(entries, key=lambda x: x.timestamp, reverse=True):
                feature_tag = f" [`{e.feature}`]" if e.feature else ""
                period_tag = f" -- {e.period}" if e.period else ""
                lines.append(f"**{e.timestamp[:10]}{feature_tag}{period_tag}** (conf={e.confidence:.0%})")
                lines.append(f"> {e.claim}")
                if e.detail:
                    lines.append(f"> {e.detail}")
                if e.evidence_ref:
                    lines.append(f"> *Evidence: {e.evidence_ref}*")
                lines.append("")

        return "\n".join(lines)

    # -- Persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.entries_path.exists():
            return
        for line in self.entries_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    self._entries.append(KnowledgeEntry.from_dict(json.loads(line)))
                except Exception:
                    pass  # corrupt line -- skip silently

    def _append_jsonl(self, entry: KnowledgeEntry) -> None:
        self.entries_path.parent.mkdir(parents=True, exist_ok=True)
        with self.entries_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _rewrite_jsonl(self) -> None:
        self.entries_path.parent.mkdir(parents=True, exist_ok=True)
        with self.entries_path.open("w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e.to_dict()) + "\n")

    def _rebuild_markdown(self) -> None:
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.write_text(self.summary(), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"DatasetKnowledge(dataset={self.dataset_name!r}, entries={len(self._entries)})"
