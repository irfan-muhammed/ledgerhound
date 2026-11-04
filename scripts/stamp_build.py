"""Write ledgerhound/_build.py with the current git commit + UTC build time.

Run right before zipping/packaging so the deployed build is identifiable:

    python scripts/stamp_build.py
    # then build the zip

`ledgerhound version` then shows the commit, so "I unzipped a new build but
the new command is still missing" is immediately diagnosable (the stamp
will read the OLD commit if the install is pointing at stale code).
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "ledgerhound" / "_build.py"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    commit = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # Read the version from the source file rather than importing ledgerhound:
    # stamping must work even when the package cannot import (missing deps on
    # a build box), and importing would pick up an INSTALLED ledgerhound rather
    # than the checkout being stamped.
    version = "unknown"
    try:
        init = (OUT.parent / "__init__.py").read_text(encoding="utf-8")
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
        if m:
            version = m.group(1)
    except Exception:  # noqa: BLE001 -- a stamp must never block a build
        pass
    build = {
        "version": version,
        "commit": commit + ("-dirty" if dirty else ""),
        "branch": branch,
        "built_at": built_at,
    }
    OUT.write_text(
        '"""Auto-generated build stamp (scripts/stamp_build.py). Do not edit."""\n'
        f"BUILD = {build!r}\n",
        encoding="utf-8",
    )
    print(f"Stamped {OUT} -> {build}")


if __name__ == "__main__":
    main()
