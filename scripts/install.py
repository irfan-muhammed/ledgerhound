#!/usr/bin/env python
"""
scripts/install.py
====================

One installer that detects the machine BEFORE touching pip, so the common
failure mode -- a multi-GB GPU wheel (cudf-cu13, torch) failing halfway
through a download, or a package with no wheel for this OS/Python at all
-- shows up as a clear message here instead of an opaque pip traceback.

CUDA 13 is the only toolkit this project installs -- the org's fleet is CUDA
13 only, and RAPIDS wheels are CUDA-major-version specific (a -cu12 wheel
does not import on a CUDA 13 box, and vice versa), so there is no
version-branching here to get wrong: `install_group("gpu", ...)` always
installs the cu13 wheels and only WARNS if `nvidia-smi` reports a driver
that is not CUDA 13.x, it never picks a different wheel to match it.

What it detects and why it matters:
  * OS -- cudf-cu13/RAPIDS ships NO native Windows wheels (Linux or WSL2
    only). Attempting it on native Windows is a guaranteed wheel error;
    this script skips it there and says exactly why, instead of letting
    pip fail loudly on something it can never resolve.
  * GPU + CUDA major version (via `nvidia-smi`, so no GPU python package
    needs to be installed first) -- used only to WARN if the driver isn't
    CUDA 13.x; the wheel choice itself is fixed (cudf-cu13), not derived
    from what nvidia-smi reports.
  * Free disk space -- the gpu+local extras (cudf, torch, transformers)
    are several GB combined; a download that dies at 90% from a full disk
    looks exactly like a corrupt-wheel error. Checked and refused up front
    with a clear message instead.

Each extras group is installed as its OWN pip invocation (not one giant
`pip install '.[gpu,dev,wheelhouse]'` command) specifically so one bad
wheel in one group can't abort every other group -- this was the actual
shape of "lots of installation related wheel errors": one failure poisoned
the whole run. Failures are collected and reported at the end; nothing
here reraises and kills the script over a single optional group.

Usage:
    python scripts/install.py                 # detect + install base, test,
                                                # and gpu (if applicable)
    python scripts/install.py --minimal        # base only, nothing optional
    python scripts/install.py --all            # every extras group
    python scripts/install.py --no-gpu         # skip gpu extras even if a
                                                # GPU is detected
    python scripts/install.py --dev --wheelhouse   # pick specific groups
    python scripts/install.py -y               # skip the confirm prompt
                                                # (CI / remote box)

Companion: scripts/verify_install.py checks the result and retries just
the packages that silently failed to import.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Rough, deliberately conservative free-space budgets (GB) per extras group
# -- covers the downloaded wheel(s) plus their unpacked/cached size, not
# just the final installed footprint.
_SPACE_BUDGET_GB = {
    "base": 2,
    "test": 1,
    "gpu": 12,     # cudf-cu13 + its own CUDA runtime deps are large
    "local": 8,    # torch (cpu or cuda) + transformers + accelerate
    "wheelhouse": 3,
    "dev": 1,
    "sql": 1,
}

# The only CUDA toolkit this project installs -- see the module docstring.
_CUDA_MAJOR = "13"
_TORCH_INDEX = f"https://download.pytorch.org/whl/cu{_CUDA_MAJOR}"


def _warn_if_driver_is_not_cu13(cuda_major: str | None) -> None:
    if cuda_major and cuda_major != _CUDA_MAJOR:
        print(f"  [warn] nvidia-smi reports CUDA {cuda_major}.x, but this project only "
              f"installs CUDA {_CUDA_MAJOR} wheels (cudf-cu{_CUDA_MAJOR}, torch from "
              f"{_TORCH_INDEX}). A mismatched wheel installs but fails to import -- "
              "if this box genuinely has an older driver, upgrade it, or install "
              "matching wheels yourself outside this script.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT)


def detect_os() -> str:
    """'windows' | 'linux' | 'wsl' | 'macos' | 'other'."""
    system = platform.system().lower()
    if system == "linux":
        try:
            if "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower():
                return "wsl"
        except Exception:
            pass
        return "linux"
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    return "other"


def detect_gpu() -> tuple[int, str | None]:
    """(gpu_count, cuda_major_version_or_None) via nvidia-smi -- no GPU
    python package needs to be installed yet for this to work, unlike the
    ComputeRouter probes the running app uses later."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0, None
    if out.returncode != 0:
        return 0, None
    text = out.stdout
    n = len(re.findall(r"^\|\s*\d+\s+NVIDIA", text, re.MULTILINE))
    m = re.search(r"CUDA Version:\s*(\d+)\.\d+", text)
    cuda_major = m.group(1) if m else None
    return max(n, 1 if "CUDA Version" in text else 0), cuda_major


def free_space_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def plan_groups(args, os_kind: str, gpu_count: int) -> list[str]:
    if args.minimal:
        return []
    if args.all:
        groups = ["test", "gpu", "local", "wheelhouse", "dev", "sql"]
    else:
        groups = list(args.groups) if args.groups else ["test"]
        want_gpu = gpu_count > 0 and not args.no_gpu
        if want_gpu and "gpu" not in groups:
            groups.append("gpu")
    if "gpu" in groups and os_kind == "windows":
        print("  [skip] 'gpu' extras (cudf-cu13/RAPIDS) has no native Windows "
              "wheels -- Linux or WSL2 only. Run this script inside WSL2 to "
              "get cuDF; skipping it here so pip doesn't fail on it.")
        groups = [g for g in groups if g != "gpu"]
    return groups


def check_disk_space(groups: list[str]) -> bool:
    need = _SPACE_BUDGET_GB["base"] + sum(_SPACE_BUDGET_GB.get(g, 1) for g in groups)
    have = free_space_gb(REPO_ROOT)
    print(f"  disk space: {have:.1f} GB free, ~{need} GB estimated for this install")
    if have < need:
        print(f"  [FAIL] not enough free disk space (need ~{need} GB, have "
              f"{have:.1f} GB). A GPU/torch wheel dying mid-download from a "
              "full disk shows up as a confusing 'bad wheel' error -- this "
              "check exists so you see the real cause instead. Free up space "
              "or pass --minimal / drop groups and retry.")
        return False
    return True


def install_base() -> tuple[bool, str]:
    r = _run([sys.executable, "-m", "pip", "install", "-e", "."])
    return r.returncode == 0, "base (pip install -e .)"


def install_group(group: str, cuda_major: str | None, os_kind: str) -> tuple[bool, str]:
    label = f"[{group}]"
    if group == "gpu":
        # Only CUDA 13 wheels exist for this project (see _CUDA_MAJOR); warn
        # rather than branch if the driver reports something else -- a
        # different wheel for a different major would install and then fail
        # to import, which is a worse failure mode than a loud warning here.
        _warn_if_driver_is_not_cu13(cuda_major)
        r = _run([sys.executable, "-m", "pip", "install", f"cudf-cu{_CUDA_MAJOR}",
                  "--extra-index-url=https://pypi.nvidia.com"])
        ok_cudf = r.returncode == 0
        r2 = _run([sys.executable, "-m", "pip", "install", "torch>=2.0",
                  "--index-url", _TORCH_INDEX])
        return (ok_cudf and r2.returncode == 0), label
    if group == "local":
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[local]"]
        if cuda_major:  # GPU host but not going through the 'gpu'/cudf path --
            # still want a CUDA-matched torch build, not an accidental CPU one.
            _warn_if_driver_is_not_cu13(cuda_major)
            r = _run([sys.executable, "-m", "pip", "install", "torch>=2.0",
                      "--index-url", _TORCH_INDEX])
            if r.returncode != 0:
                return False, label
        r = _run(cmd)
        return r.returncode == 0, label
    r = _run([sys.executable, "-m", "pip", "install", "-e", f".[{group}]"])
    return r.returncode == 0, label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minimal", action="store_true", help="base package only")
    ap.add_argument("--all", action="store_true", help="every extras group")
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip gpu extras even if a GPU is detected")
    ap.add_argument("--dev", dest="groups", action="append_const", const="dev")
    ap.add_argument("--sql", dest="groups", action="append_const", const="sql")
    ap.add_argument("--wheelhouse", dest="groups", action="append_const", const="wheelhouse")
    ap.add_argument("--gpu", dest="groups", action="append_const", const="gpu")
    ap.add_argument("--local", dest="groups", action="append_const", const="local")
    ap.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompt")
    args = ap.parse_args()

    print("=== 1. Detecting environment ===")
    os_kind = detect_os()
    py = platform.python_version()
    gpu_count, cuda_major = detect_gpu()
    print(f"  OS: {os_kind}   Python: {py}")
    print(f"  GPU: {gpu_count} detected" + (f" (CUDA {cuda_major}.x)" if cuda_major else ""))

    print("\n=== 2. Install plan ===")
    groups = plan_groups(args, os_kind, gpu_count)
    print(f"  base (always) + {groups or '(no optional extras)'}")

    print("\n=== 3. Disk space check ===")
    if not check_disk_space(groups):
        return 1

    if not args.yes:
        try:
            resp = input("\nProceed with this install plan? [Y/n] ").strip().lower()
        except EOFError:
            resp = "y"
        if resp not in ("", "y", "yes"):
            print("Aborted.")
            return 1

    print("\n=== 4. Installing ===")
    results: list[tuple[bool, str]] = [install_base()]
    for g in groups:
        results.append(install_group(g, cuda_major, os_kind))

    print("\n=== 5. Summary ===")
    failed = [label for ok, label in results if not ok]
    for ok, label in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
    if failed:
        print(f"\n{len(failed)} group(s) failed: {', '.join(failed)}")
        print("Run: python scripts/verify_install.py   to see exactly what's "
              "missing and retry just those packages.")
        return 1
    print("\nAll requested groups installed.")
    if "gpu" in groups or gpu_count > 0:
        print("Run: python scripts/diagnose_gpu.py   to confirm cuDF is "
              "actually usable end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
