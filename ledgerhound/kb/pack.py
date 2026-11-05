"""
ledgerhound.kb.pack
===================

The Knowledge Pack: three scopes, most specific wins (design doc):

    builtin  shipped with Ledgerhound (ledgerhound/kb/builtin/*.yaml)
    team     a shared folder (paths.team_kb_dir / LEDGERHOUND_TEAM_KB_DIR);
             append-only. Object-store sync is deferred (Phase 3 decision).
    project  <cwd>/.ledgerhound/kb, or LEDGERHOUND_KB_DIR

A writable scope is a folder holding `entries.jsonl` (append-only; the
latest line per (type, id) wins) and `docs/` (uploaded documents plus their
extracted text). Nothing is ever executed or followed from a document: it is
data, quoted with its source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ledgerhound.kb.schema import now, slug, validate

SCOPES = ("builtin", "team", "project")
WRITABLE = ("team", "project")
DOC_EXTS = (".md", ".markdown", ".txt", ".pdf")
MAX_DOC_BYTES = 25 * 1024 * 1024
BUILTIN_DIR = Path(__file__).parent / "builtin"


class KBError(ValueError):
    """A refused Knowledge Pack operation; the message says how to fix it."""


def team_dir() -> Optional[Path]:
    v = os.environ.get("LEDGERHOUND_TEAM_KB_DIR", "").strip()
    if not v:
        try:
            from ledgerhound.paths import _explicit
            v = _explicit("team_kb_dir")
        except Exception:  # noqa: BLE001
            v = ""
    return Path(v).expanduser() if v else None


def project_dir() -> Path:
    v = os.environ.get("LEDGERHOUND_KB_DIR", "").strip()
    return Path(v).expanduser() if v else Path.cwd() / ".ledgerhound" / "kb"


@dataclass
class Entry:
    data: dict
    scope: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.data["type"], self.data["id"])

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)


class KnowledgePack:
    def __init__(self, *, team: Optional[Path] = None, project: Optional[Path] = None,
                 builtin: Optional[Path] = BUILTIN_DIR):
        self.dirs: dict[str, Optional[Path]] = {
            "builtin": builtin,
            "team": team if team is not None else team_dir(),
            "project": project if project is not None else project_dir(),
        }
        self._cache: Optional[list[Entry]] = None
        self.load_errors: list[str] = []

    # -- reading -------------------------------------------------------------
    def _scope_entries(self, scope: str) -> Iterable[dict]:
        d = self.dirs.get(scope)
        if d is None or not d.exists():
            return []
        if scope == "builtin":
            import yaml
            out = []
            for p in sorted(d.glob("*.yaml")):
                docs = yaml.safe_load(p.read_text(encoding="utf-8")) or []
                out.extend(docs if isinstance(docs, list) else [docs])
            return out
        f = d / "entries.jsonl"
        if not f.exists():
            return []
        out = []
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                self.load_errors.append(f"{f}:{i}: not valid JSON, skipped")
        return out

    def entries(self, *, scope: Optional[str] = None, type: Optional[str] = None,
                effective: bool = True) -> list[Entry]:
        """Every entry; with effective=True, one per (type, id) -- the most
        specific scope wins, then the latest line within a scope."""
        if self._cache is None:
            self.load_errors = []
            all_: list[Entry] = []
            for sc in SCOPES:
                for raw in self._scope_entries(sc):
                    e, errs = validate(raw)
                    if errs:
                        self.load_errors.append(f"[{sc}] {e.get('id', '?')}: " + "; ".join(errs))
                        continue
                    e["scope"] = sc
                    all_.append(Entry(e, sc))
            self._cache = all_
        out = self._cache
        if effective:
            last: dict[tuple, Entry] = {}
            for en in out:                       # SCOPES order + file order = precedence
                last[en.key] = en
            out = list(last.values())
        if scope:
            out = [e for e in out if e.scope == scope]
        if type:
            out = [e for e in out if e.data["type"] == type]
        return out

    def get(self, type: str, id: str) -> Optional[Entry]:
        for e in self.entries(type=type):
            if e.data["id"] == id:
                return e
        return None

    def doc_text(self, entry: Entry) -> str:
        d = self.dirs.get(entry.scope)
        if d is None:
            return ""
        p = d / "docs" / f"{entry['id']}.txt"
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""

    # -- writing -------------------------------------------------------------
    def _writable(self, scope: str) -> Path:
        if scope not in WRITABLE:
            raise KBError(f"scope must be one of {', '.join(WRITABLE)}; the builtin pack is read-only")
        d = self.dirs.get(scope)
        if d is None:
            raise KBError("no team folder is configured: set LEDGERHOUND_TEAM_KB_DIR "
                          "or paths.team_kb_dir in main_config.yaml")
        return d

    def init(self, scope: str = "project") -> Path:
        d = self._writable(scope)
        (d / "docs").mkdir(parents=True, exist_ok=True)
        (d / "entries.jsonl").touch()
        return d

    def add(self, entry: dict, *, scope: str = "project", added_by: str = "") -> tuple[Entry, bool]:
        """Validate and append. Returns (entry, replaced_existing)."""
        d = self._writable(scope)
        e, errs = validate(entry)
        if errs:
            raise KBError("entry not added: " + "; ".join(errs))
        if e["type"] == "document":
            raise KBError("add documents with add_document(path), not as an entry")
        e.pop("scope", None)
        e["added_by"] = added_by or e.get("added_by") or "unknown"
        e["added_at"] = now()
        existed = self.get(e["type"], e["id"]) is not None
        self.init(scope)
        with open(d / "entries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        self._cache = None
        return Entry({**e, "scope": scope}, scope), existed

    def add_document(self, path: str | Path, *, scope: str = "project", title: str = "",
                     source: str = "", added_by: str = "") -> Entry:
        d = self._writable(scope)
        p = Path(path).expanduser()
        if not p.is_file():
            raise KBError(f"no file at {p}")
        if p.suffix.lower() not in DOC_EXTS:
            raise KBError(f"documents must be {', '.join(DOC_EXTS)}; got {p.suffix or 'no extension'}")
        if p.stat().st_size > MAX_DOC_BYTES:
            raise KBError(f"{p.name} is over {MAX_DOC_BYTES // (1024 * 1024)} MB")
        text = extract_text(p)
        if not text.strip():
            raise KBError(f"no text could be extracted from {p.name} (scanned PDF?)")
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        title = title or p.stem
        doc_id = slug(f"{title}_{sha[:8]}")
        self.init(scope)
        shutil.copyfile(p, d / "docs" / f"{doc_id}{p.suffix.lower()}")
        (d / "docs" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        entry = {"type": "document", "id": doc_id, "title": title,
                 "file": f"{doc_id}{p.suffix.lower()}", "source": source or p.name,
                 "sha256": sha, "chars": len(text),
                 "added_by": added_by or "unknown", "added_at": now()}
        e, errs = validate(entry)
        if errs:                               # cannot happen; never store a bad line
            raise KBError("; ".join(errs))
        with open(d / "entries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        self._cache = None
        return Entry({**e, "scope": scope}, scope)

    # -- validation of files before adding ----------------------------------
    @staticmethod
    def read_entry_file(path: str | Path) -> list[dict]:
        """Entries from a .yaml/.yml/.json/.jsonl file (one or a list)."""
        p = Path(path).expanduser()
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".jsonl":
            return [json.loads(x) for x in text.splitlines() if x.strip()]
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text)
        if data is None:
            return []
        return data if isinstance(data, list) else [data]


def extract_text(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise KBError("PDF documents need the docs extra: pip install 'ledgerhound[docs]'") from e
        reader = PdfReader(str(p))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="replace")
