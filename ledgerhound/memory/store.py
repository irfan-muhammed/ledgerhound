"""
ledgerhound.memory.store
=====================

YAML persistence for Dataset Memory.

Path convention
---------------
Default location: ``./.ledgerhound/memory/<dataset_name>.yaml``

This puts memory next to the user's working directory by default --
out of the way, but shareable (it's just a YAML file in a folder).
Override via the ``path`` argument when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .models import DatasetMemory

log = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path(".ledgerhound") / "memory"


def memory_path(dataset_name: str, base_dir: str | Path = DEFAULT_MEMORY_DIR) -> Path:
    """Resolve the canonical YAML path for a given dataset name."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in dataset_name)
    return Path(base_dir) / f"{safe}.yaml"


def save_memory(
    mem: DatasetMemory,
    path: str | Path | None = None,
    base_dir: str | Path = DEFAULT_MEMORY_DIR,
) -> Path:
    """Persist a DatasetMemory to YAML.

    Returns the path written. Touches ``updated_at`` automatically.
    """
    mem.touch()
    target = Path(path) if path else memory_path(mem.dataset_name, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(mem.to_dict(), f, sort_keys=False, allow_unicode=True)
    log.info("Saved DatasetMemory for %r to %s", mem.dataset_name, target)
    return target


def load_memory(
    dataset_name: str | None = None,
    path: str | Path | None = None,
    base_dir: str | Path = DEFAULT_MEMORY_DIR,
) -> DatasetMemory | None:
    """Load a DatasetMemory by dataset name or explicit path.

    Returns None if the file doesn't exist (caller decides whether to
    create a fresh memory or fail).
    """
    if path is not None:
        target = Path(path)
    elif dataset_name is not None:
        target = memory_path(dataset_name, base_dir)
    else:
        raise ValueError("Pass either dataset_name or path.")

    if not target.exists():
        return None

    with open(target, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DatasetMemory.from_dict(data)
