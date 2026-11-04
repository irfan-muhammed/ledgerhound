"""
ledgerhound.runtime.session_collector
====================================

Incremental pull of per-install session_log files (ledgerhound.runtime.
session_log) into ONE analyst's own archive -- for the deployment shape
where Ledgerhound is installed separately in every person's own Kubeflow
notebook, each producing its own isolated ``~/.ledgerhound/sessions.jsonl``,
with no shared filesystem between them. Getting a copy of each person's
file onto the analysing machine is someone else's problem (kubectl cp,
MinIO, a shared bucket, whatever) -- this module only handles what happens
once copies land in a staging folder: pulling in only what's NEW since the
last pull, so re-running collection after someone's file has grown further
doesn't re-read or re-append what was already taken.

Mechanism: session_log files are strictly append-only (one line per turn,
never rewritten), so "what's new" is exactly "the bytes after wherever we
stopped reading last time" -- a byte-offset cursor per source filename,
persisted at ``~/.ledgerhound/collector_state.json``. No content hashing or
diffing needed. The one edge case handled explicitly: if a source file is
ever SMALLER than the stored offset (rotated, reset, replaced with a fresh
export), that isn't a continuation of the same stream -- re-ingest it from
byte 0 rather than seeking past EOF or silently dropping data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_STATE_PATH = Path.home() / ".ledgerhound" / "collector_state.json"


def load_state(state_path: Path = DEFAULT_STATE_PATH) -> dict:
    try:
        return json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- no state yet is not an error
        return {}


def save_state(state: dict, state_path: Path = DEFAULT_STATE_PATH) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _new_bytes(source: Path, offset: int) -> tuple[bytes, int]:
    """Read only what's beyond `offset`. Returns (new_bytes, new_offset).
    A source smaller than `offset` has been rotated/replaced -- treated as
    a fresh stream, read from 0."""
    size = source.stat().st_size
    start = offset if size >= offset else 0
    with open(source, "rb") as f:
        f.seek(start)
        data = f.read()
    return data, start + len(data)


def sync_one(source: Path, archive_path: Path, state: dict) -> int:
    """Append whatever is new in `source` (since this source's stored
    cursor) onto `archive_path`, and advance the cursor in `state` (which
    the caller persists). Returns the number of new lines appended."""
    key = source.name
    prev_offset = int((state.get(key) or {}).get("byte_offset", 0))
    data, new_offset = _new_bytes(source, prev_offset)
    if not data:
        return 0
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "ab") as f:
        f.write(data)
    state[key] = {"byte_offset": new_offset}
    return data.count(b"\n")


def collect(staging_dir: Path, archive_dir: Path,
           state_path: Path = DEFAULT_STATE_PATH,
           pattern: str = "*.jsonl") -> dict:
    """Sync every `pattern`-matching file in `staging_dir` into its own
    same-named archive under `archive_dir`, only appending what's new since
    the last call. Returns {filename: n_new_lines} for whatever changed."""
    staging_dir = Path(staging_dir)
    archive_dir = Path(archive_dir)
    state = load_state(state_path)
    result = {}
    for source in sorted(staging_dir.glob(pattern)):
        n = sync_one(source, archive_dir / source.name, state)
        if n:
            result[source.name] = n
    save_state(state, state_path)
    return result
