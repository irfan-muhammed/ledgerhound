"""
ledgerhound.analyst.books
======================

Phase 5: the Librarian's "books". At the end of a session the Analyst offers to
save the whole understanding -- goal, data digest, findings, report -- into a
**book**: a durable per-project document whose first section is a SUMMARY used
as the retrieval index. On opening a project, matching books are recalled ("we
have seen this dataset before...") so understanding compounds across sessions.

Books live under ``.ledgerhound/books/`` (one ``.md`` + ``.json`` per book). This
is deliberately lightweight and project-local; promoting a book's findings into
the shared Librarian moat is a separate, optional step.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BOOKS_DIR = "books"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _tokens(s: str) -> set[str]:
    return {t for t in ("".join(c if c.isalnum() else " " for c in s.lower())).split()
            if len(t) >= 3}


@dataclass
class Book:
    """One saved unit of understanding. ``summary`` is the retrieval index."""
    title: str
    summary: str = ""
    goal: str = ""
    data_ref: str = ""
    digest_summary: str = ""
    finding_ids: list[str] = field(default_factory=list)
    report_path: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Book":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def relevance(self, query: str) -> float:
        q = _tokens(query)
        if not q:
            return 0.0
        mine = _tokens(f"{self.title} {self.summary} {self.goal} "
                       f"{self.data_ref} {' '.join(self.tags)}")
        return len(q & mine) / len(q) if mine else 0.0

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines += ["## Summary (index)", "", self.summary.strip() or "(none)", ""]
        lines += [f"- Goal: {self.goal or '?'}",
                  f"- Data: {self.data_ref or '?'}",
                  f"- Created: {self.created_at or '?'}",
                  f"- Report: {self.report_path or '(none)'}", ""]
        if self.digest_summary.strip():
            lines += ["## Data digest", "", self.digest_summary.strip(), ""]
        if self.finding_ids:
            lines += ["## Findings", ""]
            lines += [f"- {fid}" for fid in self.finding_ids]
            lines += [""]
        return "\n".join(lines)


class BookShelf:
    """Save / list / recall books under a project's ``.ledgerhound/books``."""

    def __init__(self, project):
        self.project = project
        self.dir = Path(project.dotdir) / BOOKS_DIR

    def save(self, book: Book) -> Path:
        if not book.created_at:
            book.created_at = _now()
        if not book.id:
            book.id = f"book_{_stamp()}_{uuid.uuid4().hex[:6]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{book.id}.json").write_text(
            json.dumps(book.to_dict(), indent=2), encoding="utf-8")
        (self.dir / f"{book.id}.md").write_text(
            book.to_markdown(), encoding="utf-8")
        return self.dir / f"{book.id}.md"

    def list(self) -> list[Book]:
        if not self.dir.exists():
            return []
        out = []
        for f in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                out.append(Book.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return out

    def recall(self, query: str, *, k: int = 3, min_score: float = 0.01
               ) -> list[Book]:
        scored = [(b.relevance(query), b) for b in self.list()]
        scored = [(s, b) for s, b in scored if s > min_score]
        scored.sort(key=lambda t: (t[0], t[1].created_at), reverse=True)
        return [b for _, b in scored[:k]]

    def recall_block(self, query: str, *, k: int = 3) -> str:
        """Prompt-ready 'seen before' block for project open, or '' if none."""
        hits = self.recall(query, k=k)
        if not hits:
            return ""
        lines = ["# Prior understanding (books)", ""]
        for b in hits:
            lines.append(f"- {b.title}: {b.summary.strip()[:160]}")
        return "\n".join(lines) + "\n"


def make_book(analyst, *, findings: list[str] | None = None,
              report_path: str = "", summary: str = "",
              digest_summary: str = "", tags: list[str] | None = None) -> Book:
    """Build a Book from the current Analyst session state."""
    g = analyst.grounding
    title = (g.goal.strip()[:60] or "session") + f" -- {_stamp()}"
    if not summary:
        summary = (f"Goal '{g.goal}' on data '{g.data}'. "
                   f"{len(findings or [])} finding(s) recorded.")
    return Book(title=title, summary=summary, goal=g.goal, data_ref=g.data,
                digest_summary=digest_summary, finding_ids=list(findings or []),
                report_path=report_path, tags=list(tags or []))
