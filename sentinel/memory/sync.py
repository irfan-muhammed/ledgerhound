"""
sentinel.memory.sync
=====================

MemorySync -- durability for the moat. Pushes/pulls the Librarian's on-disk
memory tiers (facts, lessons, vectors) to MinIO so the accumulated library
survives a wiped workspace and is shareable across machines/pods.

Why this exists
---------------
The Librarian writes everything under a workspace-local ``root`` (AUD-142):

    root/{owner}/facts.json   lessons.jsonl   vectors/{text,centroid,fingerprint}/...
    root/_shared/facts.json   lessons.jsonl   vectors/...

That is ephemeral. The *shared* tier in particular -- facts corroborated
across runs and promoted through the deterministic gate (AUD-143) -- is the
compounding asset. MemorySync mirrors these directories to object storage:

    {stream}://{prefix}/{tier}/{relpath}      e.g. data://memory/_shared/facts.json

Object storage is immutable per-key, so each push overwrites whole small
files (facts.json, lessons.jsonl) and individual vector-store parts. This is
snapshot-style mirroring, not the append-only one-object-per-record pattern
used for eval records -- memory tiers are small, owner/shared scoped, and read
back wholesale on pull.

Scope (matches the Librarian): by default we sync the caller's PRIVATE tier and
the SHARED tier. Promotion private->shared still happens locally through the
gate; this layer only moves bytes.

MinIO is optional: callers guard on a configured store, exactly like the
eval-record publish path in the runner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

SHARED = "_shared"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix == ".npy":
        return "application/octet-stream"
    return "application/octet-stream"


class MemorySync:
    """Mirror local Librarian memory tiers to/from a MinIO stream.

    Parameters
    ----------
    store : MinIOStore
        A configured store (its ``data`` bucket is used by default).
    root : Path
        The Librarian's memory root (the dir that holds ``{owner}/`` and
        ``_shared/``).
    stream : str
        Logical MinIO stream/bucket key. Default "data".
    prefix : str
        Key prefix under the bucket. Default "memory".
    """

    def __init__(
        self,
        store,
        root: str | Path,
        *,
        stream: str = "data",
        prefix: str = "memory",
    ) -> None:
        self.store = store
        self.root = Path(root)
        self.stream = stream
        self.prefix = prefix.strip("/")

    # -- key <-> path mapping -----------------------------------------------

    def _tier_key_prefix(self, tier: str) -> str:
        return f"{self.prefix}/{tier}/"

    def _iter_tier_files(self, tier: str) -> list[Path]:
        """All existing files under a tier dir, recursively."""
        tier_dir = self.root / tier
        if not tier_dir.is_dir():
            return []
        return [p for p in tier_dir.rglob("*") if p.is_file()]

    def _key_for(self, tier: str, path: Path) -> str:
        rel = path.relative_to(self.root / tier).as_posix()
        return f"{self._tier_key_prefix(tier)}{rel}"

    # -- push (local -> MinIO) ----------------------------------------------

    def push(self, tiers: Optional[Iterable[str]] = None) -> list[str]:
        """Upload each tier's files. Returns the list of object keys written."""
        written: list[str] = []
        for tier in self._resolve_tiers(tiers):
            for path in self._iter_tier_files(tier):
                key = self._key_for(tier, path)
                self.store.put_file(self.stream, key, path,
                                    content_type=_content_type(path))
                written.append(key)
        log.info("MemorySync.push mirrored %d files to %s://%s",
                 len(written), self.stream, self.prefix)
        return written

    # -- pull (MinIO -> local) ----------------------------------------------

    def pull(self, tiers: Optional[Iterable[str]] = None) -> list[str]:
        """Download each tier's objects into the local root. Returns local
        paths written. Existing local files are overwritten (object storage is
        the source of truth on pull)."""
        out: list[str] = []
        for tier in self._resolve_tiers(tiers):
            key_prefix = self._tier_key_prefix(tier)
            for key in self.store.list_keys(self.stream, prefix=key_prefix):
                rel = key[len(key_prefix):]
                if not rel:
                    continue
                dest = self.root / tier / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(self.store.get_bytes(self.stream, key))
                out.append(str(dest))
        log.info("MemorySync.pull restored %d files from %s://%s",
                 len(out), self.stream, self.prefix)
        return out

    # -- helpers ------------------------------------------------------------

    def _resolve_tiers(self, tiers: Optional[Iterable[str]]) -> list[str]:
        if tiers is not None:
            return list(tiers)
        # Default: every tier dir that exists locally, plus _shared even if
        # empty so a fresh machine can pull the shared moat.
        found = [p.name for p in self.root.iterdir() if p.is_dir()] \
            if self.root.is_dir() else []
        if SHARED not in found:
            found.append(SHARED)
        return found
