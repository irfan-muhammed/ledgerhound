"""
sentinel.session.store
======================

SessionStore -- find, list, and load sessions from the workspace.
"""

from __future__ import annotations

from pathlib import Path
from sentinel.session.models import Session


class SessionStore:
    def __init__(self, workspace_root: str = "sentinel_workspace"):
        self._root = Path(workspace_root) / "sessions"

    def list(self) -> list[Session]:
        if not self._root.exists():
            return []
        sessions = []
        for d in sorted(self._root.iterdir(), reverse=True):
            if d.is_dir() and (d / "session.json").exists():
                try:
                    sessions.append(Session.load(d))
                except Exception:
                    pass
        return sessions

    def get(self, session_id: str) -> Session | None:
        d = self._root / session_id
        if d.exists() and (d / "session.json").exists():
            return Session.load(d)
        return None
