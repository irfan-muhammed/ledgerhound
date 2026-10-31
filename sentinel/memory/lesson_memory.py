"""
sentinel.memory.lesson_memory
=============================

LessonMemory -- procedural / "how to analyse THIS data" memory.

FactMemory remembers *what is true* (numbers). LessonMemory remembers *what to
do* -- the procedural instinct a tenured analyst builds:

  "Use IQR not z-score for outliers on this long-tailed amount column."
  "Collect-request fraud needs a month-over-month trend, not a point estimate."
  "payee_device_id is unreliable -- don't build device features from it."

Lessons are textual and inherently interpretive, so they are treated as
**advisory hints** injected at code-gen, never as load-bearing facts. Trust is
managed by recurrence: a lesson's confidence grows only when it is independently
re-observed across distinct runs. Each lesson keeps provenance (run ids + source).

Storage: JSONL (human-readable, diff-able, dependency-free), keyed per dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    """Normalised key for dedup (lowercase, collapse whitespace/punct)."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


@dataclass
class Lesson:
    id: str
    text: str
    dataset: str
    tags: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)   # distinct confirmations
    source: str = "derived"        # derived | llm | manual
    created_at: str = ""
    last_seen: str = ""

    @property
    def confidence(self) -> float:
        n = len(set(self.run_ids))
        return round(1.0 - 1.0 / (n + 1), 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Lesson":
        known = {"id", "text", "dataset", "tags", "run_ids", "source",
                 "created_at", "last_seen"}
        return cls(**{k: v for k, v in d.items() if k in known})


class LessonMemory:
    """Append-only store of procedural lessons, deduped by normalised text."""

    def __init__(self, store_path: str | Path):
        self.path = Path(store_path)
        self._by_key: dict[str, Lesson] = {}
        self._load()

    def add(self, text: str, dataset: str, run_id: str,
            tags: list[str] | None = None, source: str = "derived") -> Lesson:
        """Add a lesson, or confirm an existing one (raising its confidence)."""
        text = text.strip()
        if not text:
            raise ValueError("empty lesson text")
        key = f"{dataset}::{_norm(text)}"
        existing = self._by_key.get(key)
        if existing:
            if run_id and run_id not in existing.run_ids:
                existing.run_ids.append(run_id)
            existing.last_seen = _now()
            for t in (tags or []):
                if t not in existing.tags:
                    existing.tags.append(t)
            return existing
        lesson = Lesson(
            id=f"L{abs(hash(key)) % 10_000_000:07d}",
            text=text, dataset=dataset, tags=list(tags or []),
            run_ids=[run_id] if run_id else [], source=source,
            created_at=_now(), last_seen=_now(),
        )
        self._by_key[key] = lesson
        return lesson

    def merge_lesson(self, lesson: Lesson) -> Lesson:
        """Merge a foreign lesson (union run_ids). Used for private->shared
        promotion by the Librarian. Idempotent on repeated merges."""
        key = f"{lesson.dataset}::{_norm(lesson.text)}"
        existing = self._by_key.get(key)
        if existing is None:
            self._by_key[key] = Lesson.from_dict(lesson.to_dict())
            return self._by_key[key]
        for r in lesson.run_ids:
            if r not in existing.run_ids:
                existing.run_ids.append(r)
        for t in lesson.tags:
            if t not in existing.tags:
                existing.tags.append(t)
        existing.last_seen = _now()
        return existing

    def lessons_for(self, dataset: str) -> list[Lesson]:
        items = [l for l in self._by_key.values() if l.dataset == dataset]
        return sorted(items, key=lambda l: (-l.confidence, -len(l.run_ids)))

    def find(self, dataset: str, query: str, max_items: int = 8) -> list[Lesson]:
        """Keyword overlap retrieval (cheap, deterministic)."""
        q = set(_norm(query).split())
        scored = []
        for l in self.lessons_for(dataset):
            words = set(_norm(l.text).split()) | set(t.lower() for t in l.tags)
            overlap = len(q & words)
            if overlap:
                scored.append((overlap, l))
        scored.sort(key=lambda x: (-x[0], -x[1].confidence))
        return [l for _, l in scored[:max_items]]

    def block(self, dataset: str, query: str = "", max_items: int = 8) -> str:
        """Prompt-injectable 'lessons learned' block for code-gen."""
        lessons = self.find(dataset, query, max_items) if query else \
            self.lessons_for(dataset)[:max_items]
        if not lessons:
            return ""
        lines = ["=== LESSONS LEARNED (procedural memory, advisory) ==="]
        for l in lessons:
            conf = f" (seen {len(set(l.run_ids))}x)" if l.run_ids else ""
            lines.append(f"  - {l.text}{conf}")
        lines.append("These are hints from past analyses of this data; apply "
                     "when relevant. Verify against the actual data.")
        lines.append("=== END LESSONS ===")
        return "\n".join(lines)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for l in self._by_key.values():
                f.write(json.dumps(l.to_dict(), default=str) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                l = Lesson.from_dict(json.loads(line))
                self._by_key[f"{l.dataset}::{_norm(l.text)}"] = l
            except Exception:
                continue

    def stats(self) -> dict:
        return {
            "lessons": len(self._by_key),
            "datasets": len({l.dataset for l in self._by_key.values()}),
            "high_confidence": sum(1 for l in self._by_key.values()
                                   if l.confidence >= 0.75),
        }
