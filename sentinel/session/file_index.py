"""
sentinel.session.file_index
============================

FileIndex -- append-only index of every file Sentinel creates.

Every time Sentinel writes a file it calls ``file_index.register()``.
The index writes one line to ``file_index.jsonl`` and keeps an
in-memory dict keyed by absolute path.

On demand -- before every LLM code-generation call -- the runner asks
``file_index.prompt_block()`` which returns a compact, categorised list
of known files. This block is injected into the prompt so the LLM never
has to guess where a digest, chart, or script lives.

Naming note: this used to be ``sentinel.session.librarian.FileLibrarian``.
It was renamed ``FileIndex`` to remove the collision with the memory
curator ``sentinel.memory.librarian.Librarian`` (the moat gateway). The
old name/module remain importable as a thin deprecation shim.

Categories
----------
  digest      profile YAML produced by sentinel profile
  script      auto-generated Python step scripts
  output      sandbox artefacts: charts, CSVs, JSON dumps
  report      final HTML / ipynb / markdown deliverables
  trace       trace.jsonl run logs
  patch       patch candidate diffs and metadata
  context     compressed context snapshots
  session     session.json itself
  data        input dataset files (registered but not created by Sentinel)

File index on disk
------------------
  <session_workspace>/file_index.jsonl   -- append-only, one entry per line

  Each entry:
    { "path": "/abs/path/file.yaml",
      "category": "digest",
      "description": "profile of fraud_2026_03 dataset",
      "created_at": "2026-05-26T14:32:01Z",
      "session_id": "fraud-eda-20260526-143201" }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Category = Literal[
    "digest", "script", "output", "report",
    "trace", "patch", "context", "session", "data",
]

_CATEGORY_LABELS = {
    "digest":  "Data Profiles (digests)",
    "script":  "Generated Step Scripts",
    "output":  "Analysis Outputs (charts, CSVs, JSON)",
    "report":  "Final Reports",
    "trace":   "Run Logs",
    "patch":   "Patch Candidates",
    "context": "Context Snapshots",
    "session": "Session Metadata",
    "data":    "Input Datasets",
}


class FileEntry:
    __slots__ = ("path", "category", "description", "created_at", "session_id")

    def __init__(self, path: str, category: str, description: str,
                 created_at: str, session_id: str):
        self.path        = path
        self.category    = category
        self.description = description
        self.created_at  = created_at
        self.session_id  = session_id

    def to_dict(self) -> dict:
        return {
            "path":        self.path,
            "category":    self.category,
            "description": self.description,
            "created_at":  self.created_at,
            "session_id":  self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileEntry":
        return cls(**d)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def rel(self, base: Path) -> str:
        try:
            return str(Path(self.path).relative_to(base))
        except ValueError:
            return self.path


class FileIndex:
    """Maintains a categorised index of all files in a session workspace.

    Usage
    -----
    >>> idx = FileIndex(session)
    >>> idx.register("/abs/path/digest.yaml", "digest", "profile of fraud dataset")
    >>> block = idx.prompt_block()   # inject into LLM prompt
    >>> results = idx.find("digest") # keyword search
    """

    def __init__(self, session):
        self._session    = session
        self._index_path = Path(session.workspace_dir) / "file_index.jsonl"
        self._ws         = Path(session.workspace_dir)
        self._entries: dict[str, FileEntry] = {}
        self._load()

    # -- Write ----------------------------------------------------------

    def register(
        self,
        path: str | Path,
        category: Category,
        description: str = "",
    ) -> FileEntry:
        """Register a file. Safe to call multiple times for the same path."""
        abs_path = str(Path(path).resolve())
        if abs_path in self._entries:
            # Update description if richer one provided
            if description:
                self._entries[abs_path].description = description
            return self._entries[abs_path]

        entry = FileEntry(
            path=abs_path,
            category=category,
            description=description or Path(abs_path).name,
            created_at=datetime.now(timezone.utc).isoformat(),
            session_id=self._session.id,
        )
        self._entries[abs_path] = entry
        self._append(entry)
        return entry

    def register_many(self, paths: list[str | Path], category: Category,
                      description_prefix: str = "") -> None:
        for p in paths:
            desc = f"{description_prefix} {Path(p).name}".strip()
            self.register(p, category, desc)

    # -- Read -----------------------------------------------------------

    def find(self, keyword: str) -> list[FileEntry]:
        """Case-insensitive keyword search on path + description + category."""
        kw = keyword.lower()
        return [
            e for e in self._entries.values()
            if kw in e.path.lower()
            or kw in e.description.lower()
            or kw in e.category.lower()
        ]

    def list(self, category: str | None = None) -> list[FileEntry]:
        entries = list(self._entries.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: e.created_at)

    def latest(self, category: str) -> FileEntry | None:
        entries = self.list(category)
        return entries[-1] if entries else None

    # -- LLM prompt block -----------------------------------------------

    def prompt_block(self) -> str:
        """Return a compact multi-line block injected into every LLM prompt.

        Format (relative paths, grouped by category):

          === KNOWN FILES (File Index) ===
          [digest]  digest/fraud_2026_03.digest.yaml
                    > profile of fraud_2026_03 dataset
          [script]  auto_runs/run_.../step_01_load_and_inspect.py
                    > Step 1: Load and inspect data
          [output]  outputs/step1_distribution.png
                    > chart from step 1
          === END FILE INDEX ===
        """
        if not self._entries:
            return ""

        lines = ["=== KNOWN FILES (File Index) ==="]

        by_cat: dict[str, list[FileEntry]] = {}
        for e in sorted(self._entries.values(), key=lambda x: x.created_at):
            by_cat.setdefault(e.category, []).append(e)

        # Print in a sensible order
        order = ["data", "digest", "script", "output", "report",
                 "trace", "patch", "context", "session"]
        shown_cats = order + [c for c in by_cat if c not in order]

        for cat in shown_cats:
            if cat not in by_cat:
                continue
            label = _CATEGORY_LABELS.get(cat, cat)
            lines.append(f"\n  {label}:")
            for e in by_cat[cat]:
                rel = e.rel(self._ws)
                exists_marker = "" if e.exists() else " [MISSING]"
                lines.append(f"    [{cat}]  {rel}{exists_marker}")
                if e.description and e.description != Path(e.path).name:
                    lines.append(f"             > {e.description}")

        lines.append("\n=== END FILE INDEX ===")
        return "\n".join(lines)

    def summary_line(self) -> str:
        """One-line summary, e.g. '3 digests | 12 outputs | 2 reports'"""
        from collections import Counter
        counts = Counter(e.category for e in self._entries.values())
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        return " | ".join(parts) if parts else "no files registered yet"

    # -- Persistence ----------------------------------------------------

    def _append(self, entry: FileEntry) -> None:
        with self._index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = FileEntry.from_dict(json.loads(line))
                self._entries[e.path] = e
            except Exception:
                pass


# Backward-compatible alias for the pre-rename class name.
FileLibrarian = FileIndex
