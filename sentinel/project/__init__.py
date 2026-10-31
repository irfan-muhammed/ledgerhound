"""
sentinel.project
================

The project layer (Phase 1 of the agentic redesign) -- Sentinel's analog of
Claude Code's ``.claude/`` folder. A *project* is any directory containing a
``.sentinel/`` subfolder with:

    .sentinel/
      project.json     machine state: name, owner, default save format, model ref
      SENTINEL.md      human-editable project memory, auto-injected every session
      sessions/        per-iteration session store (existing SessionStore root)

``ProjectStore`` discovers/creates projects; ``Project`` loads one.
"""

from sentinel.project.project import Project, ProjectConfig
from sentinel.project.store import ProjectStore

__all__ = ["Project", "ProjectConfig", "ProjectStore"]
