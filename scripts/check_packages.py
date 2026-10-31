#!/usr/bin/env python
"""What is Sentinel missing on THIS machine?  Run it on the target box.

    python scripts/check_packages.py

Imports every package Sentinel can use, groups them by what breaks without
each, and prints a ready-to-paste install command for whatever is absent.
The RAPIDS wheel names are chosen from the CUDA major version the driver
reports, because a -cu12 wheel on a CUDA 13 box installs and then fails to
import.

Read-only. Imports things; installs nothing.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NVIDIA_INDEX = "https://pypi.nvidia.com"


def cuda_major() -> str | None:
    """CUDA major from nvidia-smi, or None if it cannot be determined."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           timeout=10)
        for line in r.stdout.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].strip().strip(
                    "| ").split()[0].split(".")[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def groups(cu: str):
    """(title, needs_nvidia_index, [(import_name, pip_name, what breaks)])."""
    return [
        ("REQUIRED -- Sentinel will not run without these", False, [
            ("typer", "typer[all]", "the CLI itself"),
            ("rich", "rich", "all terminal output"),
            ("pandas", "pandas", "every dataframe path"),
            ("numpy", "numpy", "numerics throughout"),
            ("pyarrow", "pyarrow", "parquet reading"),
            ("yaml", "pyyaml", "configs, methodologies, domain profiles"),
            ("scipy", "scipy", "statistics in the profiler"),
            ("httpx", "httpx", "LLM gateway calls"),
            ("matplotlib", "matplotlib", "every chart"),
            ("fpdf", "fpdf2", "PDF reports"),
            ("duckdb", "duckdb", "embedded SQL engine"),
            ("nbformat", "nbformat", "notebook export"),
            ("trino", "trino", "Trino data-platform reads"),
            ("sklearn", "scikit-learn", "metrics and splits"),
            ("networkx", "networkx", "the graph layer"),
            ("minio", "minio", "artifact mirroring"),
        ]),
        ("KERNEL MODE -- without these there is NO GPU path at all", False, [
            ("jupyter_client", "jupyter_client",
             "kernel mode; the exec sandbox is pandas-only"),
            ("ipykernel", "ipykernel", "the analysis kernel process"),
            ("nbconvert", "nbconvert", "executing the exported notebook"),
        ]),
        (f"GPU (CUDA {cu}) -- the dataframe stack", True, [
            ("cudf", f"cudf-cu{cu}", "ALL GPU dataframe work"),
            ("cupy", f"cupy-cuda{cu}x", "array backend + a detection probe"),
            ("rmm", f"rmm-cu{cu}",
             "memory pool + managed memory; without it oversubscription OOMs"),
        ]),
        ("MULTI-GPU -- spreading a frame across cards", True, [
            ("dask_cudf", f"dask-cudf-cu{cu}", "using more than one GPU"),
            ("dask_cuda", "dask-cuda", "LocalCUDACluster"),
        ]),
        ("MODELLING -- selection and model phases", False, [
            ("xgboost", "xgboost", "GBM baseline and native Tree SHAP"),
            ("lightgbm", "lightgbm", "second GBM backend"),
            ("optuna", "optuna", "hyper-parameter search"),
            ("boruta", "boruta", "fallback selection backend"),
            ("mrmr", "mrmr-selection", "redundancy-aware selection"),
        ]),
        ("WEB UI -- only if used on this host", False, [
            ("fastapi", "fastapi", "the UI server"),
            ("uvicorn", "uvicorn[standard]", "serving it"),
            ("websockets", "websockets", "live streaming to the browser"),
            ("psutil", "psutil", "the CPU/RAM badge"),
        ]),
        ("OPTIONAL -- each degrades gracefully", False, [
            ("torch", "torch", "a GPU-detection probe (cupy also works)"),
            ("sentence_transformers", "sentence-transformers",
             "real embeddings; falls back to a weak hash embedder"),
            ("faiss", "faiss-cpu", "fast vector search; falls back to brute force"),
            ("seaborn", "seaborn", "nicer charts; matplotlib still works"),
            ("pynvml", "pynvml", "GPU probe; falls back to nvidia-smi"),
            ("sqlalchemy", "sqlalchemy", "the optional sql extra"),
        ]),
    ]


def main() -> int:
    cu = cuda_major()
    if cu is None:
        cu = "13"
        print("! nvidia-smi did not report a CUDA version; assuming CUDA 13 "
              "for the RAPIDS wheel names. Check before installing.\n")
    else:
        print(f"Detected CUDA major version {cu} -- RAPIDS wheels below are "
              f"named -cu{cu} to match.\n")

    missing_plain: list[str] = []
    missing_nvidia: list[str] = []

    for title, needs_index, items in groups(cu):
        print("=" * 70)
        print(title)
        print("=" * 70)
        for mod, pip_name, why in items:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "")
                print(f"  [ok]      {mod:<22} {ver}")
            except Exception:  # noqa: BLE001
                print(f"  [MISSING] {mod:<22} pip: {pip_name}")
                print(f"            without it: {why}")
                (missing_nvidia if needs_index else missing_plain).append(
                    pip_name)
        print()

    print("=" * 70)
    print("INSTALL WHAT IS MISSING")
    print("=" * 70)
    if not missing_plain and not missing_nvidia:
        print("  Nothing missing. Every package Sentinel can use is present.")
        return 0
    if missing_nvidia:
        print("\n# RAPIDS / GPU stack (needs NVIDIA's index):")
        print(f"pip install --extra-index-url={NVIDIA_INDEX} \\")
        print("    " + " ".join(sorted(set(missing_nvidia))))
    if missing_plain:
        print("\n# everything else (normal PyPI):")
        print("pip install " + " ".join(sorted(set(missing_plain))))
    print("\n# then re-check:")
    print("python scripts/check_packages.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
