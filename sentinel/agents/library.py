"""
sentinel.agents.library
=======================

The shared agent library (rank 6): push user-defined agents to MinIO so other
analysts can pull them. This distributes the moat -- a good custom agent
written once becomes available team-wide.

Layout (mirrors the memory moat's owner-namespacing):

    {stream}://agents/{owner}/<name>.yaml

Push uploads this owner's user-tier agents; pull materialises agents from the
library into the local user tier (``~/.sentinel/agents``), where the
:class:`~sentinel.agents.registry.AgentRegistry` picks them up on next load.

Like ``memory.sync``, this is the only networked piece; the registry itself
stays filesystem-only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from sentinel.agents.spec import AgentSpec

logger = logging.getLogger(__name__)

_PREFIX = "agents"
_CT_YAML = "application/x-yaml"


class AgentLibrary:
    """Push/pull user agents to/from the MinIO shared library."""

    def __init__(self, store: Any, *, stream: str = "data",
                 user_dir: Optional[Path] = None):
        self.store = store
        self.stream = stream
        self.user_dir = Path(user_dir) if user_dir is not None else (
            Path.home() / ".sentinel" / "agents")

    def _key(self, owner: str, name: str) -> str:
        return f"{_PREFIX}/{owner}/{name}.yaml"

    def push(self, owner: str, names: Optional[list[str]] = None) -> list[str]:
        """Upload this owner's user agents. Returns the keys written."""
        if not self.user_dir.exists():
            return []
        written: list[str] = []
        for path in sorted(self.user_dir.glob("*.yaml")):
            stem = path.stem
            if names is not None and stem not in names:
                continue
            # validate before publishing so we never share a broken agent
            try:
                AgentSpec.from_yaml(path, allow_unknown_tools=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid agent %s: %s", path.name, exc)
                continue
            key = self._key(owner, stem)
            self.store.put_file(self.stream, key, path, content_type=_CT_YAML)
            written.append(key)
        return written

    def pull(self, owner: Optional[str] = None,
             *, overwrite: bool = False) -> list[str]:
        """Download agents from the library into the local user tier.

        ``owner=None`` pulls every owner's agents; otherwise just that owner.
        Returns the local paths written. Existing files are skipped unless
        ``overwrite=True`` (so a pull never silently clobbers your own agent).
        """
        prefix = f"{_PREFIX}/{owner}/" if owner else f"{_PREFIX}/"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        for key in self.store.list_keys(self.stream, prefix=prefix):
            if not key.endswith(".yaml"):
                continue
            name = key.rsplit("/", 1)[-1]
            dest = self.user_dir / name
            if dest.exists() and not overwrite:
                logger.info("pull: keeping existing %s (use overwrite)", name)
                continue
            dest.write_bytes(self.store.get_bytes(self.stream, key))
            out.append(str(dest))
        return out

    def list_remote(self, owner: Optional[str] = None) -> list[str]:
        """List agent keys available in the library."""
        prefix = f"{_PREFIX}/{owner}/" if owner else f"{_PREFIX}/"
        return [k for k in self.store.list_keys(self.stream, prefix=prefix)
                if k.endswith(".yaml")]
