"""
ledgerhound.paths
================

Where Ledgerhound writes, by default -- one place, per user:

    <parent of the launch folder>/Agentic_eda_results/<username>/
        run_outputs/   one dated folder per session: reports, findings,
                       events, transcripts, feature tables, paths.json
        data_cache/    local copies of /s3 data, downloaded tokenizers
        library/       the project: dataset cards, books, settings,
                       institutional knowledge (the Librarian)

Nothing lands anywhere else unless it is configured explicitly
(main_config.yaml `paths.*`, or `--project` / `--out` on the command line).
Machine-level secrets and state (~/.ledgerhound: saved keys, your login, the
team-knowledge upload queue) stay in the home folder.

Resolved lazily, after login, so <username> is the logged-in user.
"""

from __future__ import annotations

import os
from pathlib import Path

RESULTS_DIRNAME = "Agentic_eda_results"


def results_root() -> Path:
    """The Agentic_eda_results folder: `paths.results_root` (or
    $LEDGERHOUND_RESULTS_ROOT), else one level above the launch folder."""
    from ledgerhound.config import get_str
    explicit = (os.environ.get("LEDGERHOUND_RESULTS_ROOT")
                or get_str("paths", "results_root")).strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.cwd().resolve().parent / RESULTS_DIRNAME


def user_dir() -> Path:
    from ledgerhound.identity import current_user
    return results_root() / current_user()


def _explicit(key: str, env: str = "") -> str:
    from ledgerhound.config import get_str
    return (get_str("paths", key, env=env) if env else get_str("paths", key)).strip()


def run_outputs_dir() -> Path:
    v = _explicit("run_outputs_dir")
    return Path(v).expanduser() if v else user_dir() / "run_outputs"


def data_cache_dir() -> Path:
    v = _explicit("data_cache_dir", env="LEDGERHOUND_DATA_CACHE_DIR")
    return Path(v).expanduser() if v else user_dir() / "data_cache"


def library_dir() -> Path:
    v = _explicit("workspace_dir")
    return Path(v).expanduser() if v else user_dir() / "library"


def layout() -> dict[str, Path]:
    """All three, for status lines and doctor."""
    return {"results": user_dir(), "run_outputs": run_outputs_dir(),
            "data_cache": data_cache_dir(), "library": library_dir()}
