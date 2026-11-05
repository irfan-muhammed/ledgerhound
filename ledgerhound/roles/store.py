"""
ledgerhound.roles.store
=======================

Confirmed roles, kept per dataset CONTENT (sha256), so the same data is
confirmed once and every later session -- CLI or MCP, with or without a
project -- starts from the confirmed map. Lives under the library dir
(ledgerhound.paths.library_dir), next to the Librarian's books.

A stored map is reused only if its columns are exactly the loaded columns;
anything else is re-inferred, because a renamed or added column is exactly
where a stale role would do harm.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ledgerhound.roles.model import RoleMap

ROLES_DIR = "column_roles"


class RoleStore:
    def __init__(self, root: Optional[Path] = None):
        if root is None:
            from ledgerhound.paths import library_dir
            root = library_dir()
        self.dir = Path(root) / ROLES_DIR

    def _path(self, data_sha256: str) -> Path:
        safe = "".join(c for c in data_sha256 if c.isalnum())[:64]
        return self.dir / f"{safe}.json"

    def load(self, data_sha256: str, columns: list[str]) -> Optional[RoleMap]:
        if not data_sha256:
            return None
        p = self._path(data_sha256)
        try:
            rm = RoleMap.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
        if list(rm.columns) != [str(c) for c in columns] or not rm.confirmed:
            return None
        return rm

    def save(self, rm: RoleMap) -> Optional[Path]:
        if not rm.data_sha256:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(rm.data_sha256)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rm.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, p)
        return p
