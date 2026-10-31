"""
sentinel.project.project
========================

``Project`` -- one Sentinel project rooted at a folder's ``.sentinel/``.

Backward-compatible with the existing workspace/session layout: sessions still
live under ``.sentinel/sessions`` and are read by the existing ``SessionStore``.
The new pieces are ``project.json`` (machine state) and ``SENTINEL.md`` (the
human-editable project memory, auto-injected into every session -- Sentinel's
CLAUDE.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DOTDIR = ".sentinel"
CONFIG_NAME = "project.json"
MEMORY_NAME = "SENTINEL.md"
SESSIONS_DIR = "sessions"

# Save formats this server can actually produce (Phase 0 decision: default .md).
SAVE_FORMATS = ("md", "ipynb", "pdf", "html")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SENTINEL_MD_TEMPLATE = """# {name}

> SENTINEL.md -- project memory. Auto-injected into every session's context.
> Edit freely: dataset notes, domain facts, team conventions, prior knowledge.

## Goal

(what this project is trying to achieve)

## Data

(where the data lives, what each file is, known quirks)

## Domain notes

(fraud/mule knowledge, label definitions, conventions)

## Conventions

- Default save format: {save_format}
"""


@dataclass
class ProjectConfig:
    """Machine state persisted to .sentinel/project.json."""
    name: str
    created_at: str = ""
    owner: str = ""
    save_format: str = "md"          # Phase 0 decision: default .md
    model_config_ref: str = ""       # e.g. configs/deepseek.yaml (set via /model)
    schema: str = "project/v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class Project:
    """A project rooted at ``root`` (the folder that contains ``.sentinel/``)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dotdir = self.root / DOTDIR

    # -- identity ----------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.dotdir / CONFIG_NAME

    @property
    def memory_path(self) -> Path:
        return self.dotdir / MEMORY_NAME

    @property
    def sessions_root(self) -> Path:
        return self.dotdir / SESSIONS_DIR

    def exists(self) -> bool:
        return self.config_path.exists()

    # -- create / load -----------------------------------------------------
    @classmethod
    def init(cls, root: str | Path, name: str = "", owner: str = "",
             save_format: str = "md") -> "Project":
        """Create ``.sentinel/`` (idempotent) and return the Project."""
        p = cls(root)
        if not name:
            name = p.root.resolve().name or "project"
        if save_format not in SAVE_FORMATS:
            save_format = "md"
        p.dotdir.mkdir(parents=True, exist_ok=True)
        p.sessions_root.mkdir(parents=True, exist_ok=True)
        if not p.config_path.exists():
            cfg = ProjectConfig(name=name, created_at=_now(), owner=owner,
                                save_format=save_format)
            p.save_config(cfg)
        if not p.memory_path.exists():
            p.memory_path.write_text(
                _SENTINEL_MD_TEMPLATE.format(name=name, save_format=save_format),
                encoding="utf-8")
        return p

    def config(self) -> ProjectConfig:
        """The project's saved settings. A damaged project.json (a half-
        written file, a hand edit gone wrong) used to crash `sentinel shell`
        at startup with a raw JSONDecodeError. It is now moved aside as
        project.corrupt-<time>.json, a fresh config is written in its place
        (named after the folder), and `recovered_from` records the backup so
        the shell can say what happened. Only the settings file is replaced
        -- SENTINEL.md, sessions and dataset cards are untouched."""
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("project.json is not a JSON object")
            data.setdefault("name", self.root.resolve().name or "project")
            return ProjectConfig.from_dict(data)
        except FileNotFoundError:
            raise
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            return self._recover_config(e)

    # Set when config() had to replace a damaged project.json.
    recovered_from: str = ""
    recovered_reason: str = ""

    def _recover_config(self, error: Exception) -> ProjectConfig:
        import uuid
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = self.config_path.with_name(
            f"project.corrupt-{stamp}-{uuid.uuid4().hex[:6]}.json")
        try:
            self.config_path.replace(backup)
            self.recovered_from = str(backup)
        except OSError:
            self.recovered_from = ""
        self.recovered_reason = f"{type(error).__name__}: {error}"
        cfg = ProjectConfig(name=self.root.resolve().name or "project",
                            created_at=_now())
        try:
            self.save_config(cfg)
        except OSError:
            pass
        return cfg

    def save_config(self, cfg: ProjectConfig) -> None:
        self.dotdir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    # -- SENTINEL.md (the CLAUDE.md analog) --------------------------------
    def memory(self) -> str:
        """The SENTINEL.md text, auto-injected into session context. '' if none."""
        if self.memory_path.exists():
            return self.memory_path.read_text(encoding="utf-8")
        return ""

    def memory_block(self) -> str:
        """A prompt-ready block, or '' when the memory is empty or still the
        untouched template (so placeholder prose never enters the context)."""
        text = self.memory().strip()
        if not text or self._is_pristine_template(text):
            return ""
        return f"# Project memory (SENTINEL.md)\n\n{text}\n"

    def _is_pristine_template(self, text: str) -> bool:
        """True if SENTINEL.md is byte-equal to the freshly generated template
        (i.e. the user has not added real content yet)."""
        try:
            cfg = self.config()
            tmpl = _SENTINEL_MD_TEMPLATE.format(
                name=cfg.name, save_format=cfg.save_format).strip()
        except Exception:
            return False
        return text == tmpl
