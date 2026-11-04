"""
ledgerhound._compat
===================

Ledgerhound was called Sentinel until v0.7.9. This module keeps setups made
under the old name working:

* Environment variables: every ``SENTINEL_*`` variable is copied to its
  ``LEDGERHOUND_*`` name when the new one is not already set, so the many
  places that read config from the environment (including subprocess
  kernels, which inherit ``os.environ``) need no dual lookups. One
  deprecation warning names the variables to rename.
* Folders: ``.sentinel/`` (home, project and working folders),
  ``SENTINEL.md`` project memory, and ``sentinel_workspace/`` are renamed
  to their new names the first time Ledgerhound sees them -- only when the
  new name does not exist yet, so nothing is ever overwritten.

Everything here is best effort: a failure leaves the old files in place and
never stops the program.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

OLD_ENV_PREFIX = "SENTINEL_"
NEW_ENV_PREFIX = "LEDGERHOUND_"

OLD_DOTDIR = ".sentinel"
NEW_DOTDIR = ".ledgerhound"
OLD_MEMORY = "SENTINEL.md"
NEW_MEMORY = "LEDGERHOUND.md"
OLD_WORKSPACE = "sentinel_workspace"
NEW_WORKSPACE = "ledgerhound_workspace"


def adopt_legacy_env(environ=None) -> list[str]:
    """Copy SENTINEL_* variables to LEDGERHOUND_* (new names win). Returns
    the old names that were adopted."""
    env = os.environ if environ is None else environ
    adopted = []
    for key in list(env):
        if not key.startswith(OLD_ENV_PREFIX):
            continue
        new_key = NEW_ENV_PREFIX + key[len(OLD_ENV_PREFIX):]
        if new_key not in env:
            env[new_key] = env[key]
            adopted.append(key)
    if adopted and environ is None:
        warnings.warn(
            "Sentinel was renamed to Ledgerhound; using legacy environment "
            f"variable(s) {', '.join(sorted(adopted))}. Rename them to "
            f"{NEW_ENV_PREFIX}* -- the old names will stop working in a "
            "future release.", DeprecationWarning, stacklevel=2)
    return adopted


def _move(old: Path, new: Path) -> bool:
    try:
        if old.exists() and not new.exists():
            old.rename(new)
            print(f"ledgerhound: renamed {old} -> {new} (Sentinel is now Ledgerhound)",
                  file=sys.stderr)
            return True
    except OSError:
        pass
    return False


def migrate_dotdir(root: str | Path) -> None:
    """``<root>/.sentinel`` -> ``<root>/.ledgerhound``, and the project
    memory file inside it. Safe to call on every project open."""
    root = Path(root)
    _move(root / OLD_DOTDIR, root / NEW_DOTDIR)
    _move(root / NEW_DOTDIR / OLD_MEMORY, root / NEW_DOTDIR / NEW_MEMORY)


def migrate_home() -> None:
    try:
        migrate_dotdir(Path.home())
    except RuntimeError:   # no resolvable home directory
        pass


def migrate_cwd() -> None:
    """The working folder's state: ./.sentinel and ./sentinel_workspace."""
    cwd = Path.cwd()
    migrate_dotdir(cwd)
    _move(cwd / OLD_WORKSPACE, cwd / NEW_WORKSPACE)
