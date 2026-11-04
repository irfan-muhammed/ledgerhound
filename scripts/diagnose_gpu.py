#!/usr/bin/env python
"""Why is Ledgerhound running on CPU?  Run this ON THE GPU BOX.

    python scripts/diagnose_gpu.py /path/to/your/data.parquet

Prints, in order: what each GPU probe sees, how the router sizes the data,
what the router decides, and -- the part that matters -- what the shell
actually hands to the kernel. Read-only; changes nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledgerhound.backends.compute_router import (  # noqa: E402
    ComputeRouter,
    _dask_cudf_available,
)

BAR = "=" * 68

# Set by _check_build(); section 5 mirrors whichever mapping this build has.
_HAS_V3_044 = False


def _check_build() -> None:
    """Is this copy of Ledgerhound new enough for the GPU flags to do anything?

    Before V3-044 the shell collapsed the router's ``dask_cudf`` decision to
    ``"pandas"``, so LEDGERHOUND_ENABLE_DASK_CUDF was inert no matter how it was
    set. Diagnosing a GPU problem against a build that predates the fix
    wastes a whole cycle, so establish which code is running before anything
    else.
    """
    import inspect

    import ledgerhound
    from ledgerhound.shell.conversation import ConversationalAnalyst

    print(BAR)
    print("0. BUILD AND EXECUTION BACKEND")
    print(BAR)
    print(f"  package    {Path(ledgerhound.__file__).parent}")
    print(f"  version    {getattr(ledgerhound, '__version__', 'unknown')}")

    # The exec sandbox (ledgerhound/auto/sandbox.py) is pandas-only -- it has
    # no cuDF branch whatsoever. If kernel mode is off, NOTHING below
    # matters: no GPU code path exists to reach. jupyter_client/ipykernel
    # live in the [test] extra, not the base install, so a perfectly good
    # RAPIDS box can land here purely for want of two pure-Python packages.
    try:
        from ledgerhound.auto.kernel import kernels_available
        ok = kernels_available()
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  kernel deps  import failed: {type(e).__name__}: {e}")
    print(f"  kernel mode  {'AVAILABLE' if ok else '*** UNAVAILABLE ***'}"
          "  (jupyter_client + ipykernel)")
    if not ok:
        print()
        print("  >>> STOP. Without kernel mode the shell uses the exec")
        print("      sandbox, which is pandas-only and has NO GPU path at")
        print("      all. Fix this first:  pip install jupyter_client ipykernel")
        print("      (or: pip install -e '.[test]')")
    else:
        try:
            from ledgerhound.cli.main import _auto_detect_use_kernel
            on, why = _auto_detect_use_kernel()
            print(f"  auto-detect  kernel mode {'ON' if on else 'OFF'} -- {why}")
            if not on:
                print("               override with: ledgerhound shell --use-kernel")
        except Exception as e:  # noqa: BLE001
            print(f"  auto-detect  could not evaluate ({type(e).__name__}: {e})")

    try:
        src = inspect.getsource(ConversationalAnalyst._resolve_compute_engine)
    except Exception as e:  # noqa: BLE001
        print(f"  V3-044     could not read source ({type(e).__name__}: {e})")
        return

    global _HAS_V3_044
    has_fix = 'in ("cudf", "dask_cudf")' in src
    _HAS_V3_044 = has_fix
    print(f"  V3-044     {'PRESENT' if has_fix else '*** MISSING ***'}"
          "  (router decision reaches the kernel)")
    if not has_fix:
        print()
        print("  >>> STOP. This build still rewrites the router's dask_cudf")
        print("      decision to 'pandas'. LEDGERHOUND_ENABLE_DASK_CUDF cannot")
        print("      do anything here -- the flag is read, then the decision")
        print("      it enables is discarded one layer up. Update this")
        print("      checkout before diagnosing anything below.")
    print()


def _report_cuda_version():
    """Print every CUDA version we can see; return the driver's major, or None.

    Three different numbers matter and are routinely confused: the driver's
    maximum supported CUDA (what nvidia-smi prints), the toolkit torch was
    built against, and cupy's runtime. The RAPIDS wheel suffix has to match
    the toolkit actually installed, so print all three and let the mismatch
    be obvious rather than inferred.
    """
    major = None
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           timeout=10)
        for line in r.stdout.splitlines():
            if "CUDA Version" in line:
                part = line.split("CUDA Version:")[1].strip().strip("| ")
                ver = part.split()[0]
                print(f"  driver supports CUDA   {ver}")
                major = ver.split(".")[0]
                break
    except Exception as e:  # noqa: BLE001
        print(f"  nvidia-smi             unavailable ({type(e).__name__})")

    try:
        import torch
        print(f"  torch built for CUDA   {torch.version.cuda}")
    except Exception:  # noqa: BLE001
        print("  torch                  not installed")

    try:
        import cupy
        rt = cupy.cuda.runtime.runtimeGetVersion()
        print(f"  cupy CUDA runtime      {rt // 1000}.{(rt % 1000) // 10}")
    except Exception:  # noqa: BLE001
        print("  cupy                   not installed")
    print()
    return major


def main() -> int:
    data_path = sys.argv[1] if len(sys.argv) > 1 else None

    _check_build()

    print(BAR)
    print("1. GPU PROBES  (the router takes the FIRST one that sees a GPU)")
    print(BAR)
    router = ComputeRouter()
    for name, probe in (("torch", router._gpus_via_torch),
                        ("cupy", router._gpus_via_cupy),
                        ("nvidia-smi", router._gpus_via_nvidia_smi)):
        try:
            n, names, mems, free = probe()
            if n:
                print(f"  {name:<12} {n} GPU(s)")
                for i, (nm, m, f) in enumerate(zip(names, mems, free)):
                    print(f"               [{i}] {nm}  total {m}GB  free {f}GB")
            else:
                print(f"  {name:<12} reported 0 GPUs")
        except ImportError:
            print(f"  {name:<12} NOT INSTALLED")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<12} FAILED: {type(e).__name__}: {e}")

    print()
    print(BAR)
    print("2. CUDA TOOLKIT vs INSTALLED WHEELS")
    print(BAR)
    print("  RAPIDS wheels are CUDA-major specific. A -cu12 wheel on a")
    print("  CUDA 13 box fails to import, and the kernel then falls back to")
    print("  pandas with a one-line warning -- a slow run, not an error.")
    print()
    cuda_major = _report_cuda_version()
    for mod in ("cudf", "cupy", "rmm", "dask_cudf", "dask_cuda"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12} OK  {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<12} UNAVAILABLE: {type(e).__name__}: {e}")

    if cuda_major:
        print(f"\n  Detected CUDA major version: {cuda_major}")
        print("  Install the matching wheels, e.g.:")
        print("    pip install --extra-index-url=https://pypi.nvidia.com \\")
        print(f"        \".[gpu-cu{cuda_major},multigpu-cu{cuda_major}]\"")

    flag = os.environ.get("LEDGERHOUND_ENABLE_DASK_CUDF")
    shown = repr(flag) if flag is not None else "unset (defaults to ON)"
    print(f"\n  LEDGERHOUND_ENABLE_DASK_CUDF = {shown}"
          f"  -> multi-GPU {'ENABLED' if _dask_cudf_available() else 'OFF'}")
    if not _dask_cudf_available() and (flag or "1").lower() not in (
            "0", "false", "no", "off"):
        print("     (wanted ON, but the dask_cudf wheel is missing above --")
        print("      install it or multi-GPU stays inert)")

    if not data_path:
        print("\n(no data path given -- pass one to see the routing decision)")
        return 0

    print()
    print(BAR)
    print("3. DATA SIZING")
    print(BAR)
    p = Path(data_path)
    print(f"  path          {p}")
    print(f"  exists        {p.exists()}   is_dir={p.is_dir()}")
    est = ComputeRouter._estimate_size(p)
    print(f"  ESTIMATED     {est} GB in memory")
    if p.is_file():
        print(f"  on disk       {p.stat().st_size / 1024**3:.2f} GB")
    print("  note: parquet is sized from the footer's uncompressed column")
    print("        sizes x1.5, NOT from on-disk x4 (V3-044). That factor is")
    print("        uncalibrated -- compare ESTIMATED against what a real")
    print("        cuDF load consumes (nvidia-smi before/after) and tell us.")
    print("        A directory with no .parquet/.csv/.tsv inside returns")
    print("        1e6 GB on purpose, which forces the CPU fallback.")

    print()
    print(BAR)
    print("4. ROUTER DECISION")
    print(BAR)
    profile = router.detect(str(p), llm_is_local=False)
    print(profile.summary())

    print()
    print(BAR)
    print("5. WHAT THE SHELL ACTUALLY USES")
    print(BAR)
    # Mirror whichever mapping THIS build actually has, rather than assuming
    # one. Hardcoding the pre-V3-044 rule here would make a fixed build
    # report a bug it no longer has, which is its own kind of wasted cycle.
    if _HAS_V3_044:
        engine = (profile.backend_choice
                  if profile.backend_choice in ("cudf", "dask_cudf")
                  else "pandas")
    else:
        engine = "cudf" if profile.backend_choice == "cudf" else "pandas"

    print(f"  router decided       : {profile.backend_choice}")
    print(f"  kernel receives      : {engine}")
    print(f"  GPU actually used    : {'YES' if engine != 'pandas' else 'NO'}")

    if profile.backend_choice != "pandas" and engine == "pandas":
        print()
        print("  >>> The router chose a GPU backend and the shell discarded")
        print("      it. That is the pre-V3-044 defect -- see section 0.")
    elif profile.backend_choice == "pandas":
        print()
        print("  >>> The ROUTER itself chose CPU, so nothing downstream can")
        print("      put this on a GPU. The reason string in section 4 says")
        print("      why: no GPU detected, or the estimate exceeded the")
        print("      ceiling. Multi-GPU is ON by default now, so if it is")
        print("      the ceiling and this box has several cards, the thing")
        print("      still missing is the dask-cudf wheel (section 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
