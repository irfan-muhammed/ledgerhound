#!/usr/bin/env python
"""
scripts/verify_install.py
============================

Companion to scripts/install.py. A single `pip install '.[gpu,dev,...]'`
can report success while one or two wheels inside it silently failed to
actually become importable (a common shape for the "wheel error" reports
this pair of scripts exists for) -- this checks every declared dependency
by actually importing it, not by trusting pip's exit code, then offers to
retry just the ones that are broken instead of re-running the whole install.

Reads the dependency lists straight from pyproject.toml (single source of
truth -- nothing here is a second, driftable copy of that list).

Usage:
    python scripts/verify_install.py                 # base deps only
    python scripts/verify_install.py --group gpu      # + a specific group
    python scripts/verify_install.py --all-groups     # every declared group
    python scripts/verify_install.py --fix            # retry broken ones
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# pip distribution name -> the name you actually `import`. Only listed
# where they differ; anything not here falls back to a normalized guess.
_IMPORT_NAME = {
    "typer": "typer",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "fpdf2": "fpdf",
    "cudf-cu13": "cudf",   # the only CUDA toolkit this project installs
    "mrmr-selection": "mrmr",
    "jupyter_client": "jupyter_client",
}
# Packages that are primarily a CLI, not meant to be imported -- verified
# by running `--version` instead.
_CLI_ONLY = {"ruff"}


def _load_pyproject() -> dict:
    try:
        import tomllib  # py3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            print("Neither tomllib (py3.11+) nor tomli is available to parse "
                  "pyproject.toml -- install tomli (`pip install tomli`) or "
                  "run this on Python 3.11+.", file=sys.stderr)
            raise SystemExit(1)
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _dist_to_import_name(spec: str) -> str:
    """'typer[all]>=0.9.0' -> 'typer'; 'pyyaml>=6.0' -> 'yaml'."""
    name = spec.split("[")[0]
    for sep in (">=", "==", "<=", "~=", ">", "<"):
        name = name.split(sep)[0]
    name = name.strip()
    key = name.lower()
    return _IMPORT_NAME.get(key, key.replace("-", "_"))


def collect_specs(groups: list[str]) -> dict[str, list[str]]:
    """{group_name: [dist_spec, ...]} for 'base' plus each requested extra."""
    data = _load_pyproject()
    project = data["project"]
    out = {"base": list(project.get("dependencies", []))}
    optional = project.get("optional-dependencies", {})
    for g in groups:
        if g not in optional:
            print(f"  [warn] no such extras group '{g}' in pyproject.toml -- skipping")
            continue
        out[g] = list(optional[g])
    return out


def check_one(dist_spec: str) -> tuple[bool, str]:
    dist_name = dist_spec.split("[")[0]
    for sep in (">=", "==", "<=", "~=", ">", "<"):
        dist_name = dist_name.split(sep)[0]
    dist_name = dist_name.strip()
    if dist_name.lower() in _CLI_ONLY:
        r = subprocess.run([sys.executable, "-m", dist_name.lower(), "--version"],
                           capture_output=True, text=True)
        return r.returncode == 0, dist_name
    import_name = _dist_to_import_name(dist_spec)
    r = subprocess.run(
        [sys.executable, "-c", f"import {import_name}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0, dist_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", dest="groups", action="append", default=[],
                    help="an extras group to also verify (repeatable)")
    ap.add_argument("--all-groups", action="store_true",
                    help="verify every extras group declared in pyproject.toml")
    ap.add_argument("--fix", action="store_true",
                    help="retry a targeted, cache-free reinstall for anything broken")
    args = ap.parse_args()

    groups = args.groups
    if args.all_groups:
        data = _load_pyproject()
        groups = list(data["project"].get("optional-dependencies", {}).keys())

    print("=== Checking declared dependencies actually import ===")
    by_group = collect_specs(groups)
    broken: list[str] = []
    for group, specs in by_group.items():
        print(f"\n[{group}]")
        for spec in specs:
            ok, dist_name = check_one(spec)
            print(f"  {'OK  ' if ok else 'FAIL'}  {spec}")
            if not ok:
                broken.append(dist_name)

    if not broken:
        print("\nAll checked dependencies import cleanly.")
        return 0

    print(f"\n{len(broken)} package(s) are declared but not importable: "
          f"{', '.join(broken)}")

    if not args.fix:
        print("Re-run with --fix to attempt a targeted, cache-free reinstall "
              "of just these (not the whole environment).")
        return 1

    print("\n=== Retrying broken packages individually (--force-reinstall "
          "--no-cache-dir) ===")
    still_broken = []
    for dist_name in broken:
        print(f"  $ pip install --force-reinstall --no-cache-dir {dist_name}")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-cache-dir", dist_name],
        )
        if r.returncode != 0:
            still_broken.append(dist_name)
            continue
        spec = next((s for g in by_group.values() for s in g
                    if s.split("[")[0].split(">=")[0].split("==")[0].strip() == dist_name), dist_name)
        ok, _ = check_one(spec)
        if not ok:
            still_broken.append(dist_name)

    if still_broken:
        print(f"\nStill broken after retry: {', '.join(still_broken)}")
        print("These likely need a platform-specific fix (wrong CUDA index, "
              "no wheel for this OS/Python combo, missing system compiler) "
              "rather than a simple reinstall -- see scripts/install.py's "
              "docstring and scripts/diagnose_gpu.py for GPU-specific ones.")
        return 1
    print("\nAll previously-broken packages now import cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
