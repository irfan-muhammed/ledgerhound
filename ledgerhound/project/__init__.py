"""
ledgerhound.project
================

The project layer (Phase 1 of the agentic redesign) -- Ledgerhound's analog of
Claude Code's ``.claude/`` folder. A *project* is any directory containing a
``.ledgerhound/`` subfolder with:

    .ledgerhound/
      project.json     machine state: name, owner, default save format, model ref
      LEDGERHOUND.md      human-editable project memory, auto-injected every session
      sessions/        per-iteration session store (existing SessionStore root)

``ProjectStore`` discovers/creates projects; ``Project`` loads one.
"""

from ledgerhound.project.project import Project, ProjectConfig
from ledgerhound.project.store import ProjectStore

__all__ = ["Project", "ProjectConfig", "ProjectStore"]
