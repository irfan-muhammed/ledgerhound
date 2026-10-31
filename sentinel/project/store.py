"""
sentinel.project.store
======================

``ProjectStore`` -- discover, create, and open projects under a parent folder
(the "projects home"). "New project / continue / list" is the shell's on-ramp,
mirroring how Claude Code lists prior projects.
"""

from __future__ import annotations

from pathlib import Path

from sentinel.project.project import CONFIG_NAME, DOTDIR, Project


class ProjectStore:
    """Lists and creates projects under ``home`` (each project is a subfolder)."""

    def __init__(self, home: str | Path = "."):
        self.home = Path(home)

    def list(self) -> list[Project]:
        """Every immediate subfolder that is a Sentinel project, newest first."""
        if not self.home.exists():
            return []
        found: list[tuple[float, Project]] = []
        for d in self.home.iterdir():
            if not d.is_dir():
                continue
            cfg = d / DOTDIR / CONFIG_NAME
            if cfg.exists():
                found.append((cfg.stat().st_mtime, Project(d)))
        found.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in found]

    def get(self, name: str) -> Project | None:
        for p in self.list():
            if p.config().name == name or p.root.name == name:
                return p
        return None

    def new(self, name: str, *, owner: str = "", save_format: str = "md") -> Project:
        """Create ``home/<name>/.sentinel/`` and return the Project."""
        root = self.home / name
        root.mkdir(parents=True, exist_ok=True)
        return Project.init(root, name=name, owner=owner, save_format=save_format)
