"""
ledgerhound.doctor.profile
=======================

`ledgerhound doctor` -- one-shot prerun benchmark that decides per-machine:

  * Which dataframe backend Ledgerhound should use for groupby-heavy work
    (pandas vs cuDF).
  * Which vector-store backend should hold longitudinal memory (NumpyBruteForce
    vs FaissCPU vs CuVS).

Both choices are written to a `compute_profile.json` file. The compute router
(ledgerhound.compute.router, built in M6) reads this file once at session start
and exposes `preferred_dataframe_backend()` / `preferred_vector_backend()` so
the rest of the code never re-decides.

The doctor is deliberately empirical: measure on this machine, with this
hardware and these installed libraries, and let the numbers pick the backend.
Heuristics (>1M rows -> GPU) lie; benchmarks don't.

What it measures
----------------
1. **Dataframe groupby**: build a 1M-row toy frame with `account` + `amount`,
   run `df.groupby("account")["amount"].sum()` on pandas; if cuDF available,
   run the same on cuDF. Record wall-clock milliseconds.
2. **Vector recall**: build a 100K x 384 random matrix, normalise rows,
   construct each backend's store, run a single k=10 search. Record ms.

The benchmark scales are intentionally modest so doctor runs in under
~30 seconds on any machine -- this is a prerun check, not a stress test.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

log = logging.getLogger(__name__)


# -- Result models ----------------------------------------------------------


@dataclass
class BackendTiming:
    backend: str            # e.g. "pandas", "cudf", "numpy", "faiss", "cuvs"
    available: bool
    ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComputeProfile:
    """The decision artifact persisted to disk."""
    generated_at: float
    machine: dict[str, Any]
    dataframe_results: list[BackendTiming]
    vector_results: list[BackendTiming]
    preferred_dataframe: str       # "pandas" | "cudf"
    preferred_vector: str          # "numpy" | "faiss" | "cuvs"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "machine": self.machine,
            "dataframe_results": [t.to_dict() for t in self.dataframe_results],
            "vector_results": [t.to_dict() for t in self.vector_results],
            "preferred_dataframe": self.preferred_dataframe,
            "preferred_vector": self.preferred_vector,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComputeProfile":
        return cls(
            generated_at=d["generated_at"],
            machine=d.get("machine", {}),
            dataframe_results=[BackendTiming(**t) for t in d.get("dataframe_results", [])],
            vector_results=[BackendTiming(**t) for t in d.get("vector_results", [])],
            preferred_dataframe=d.get("preferred_dataframe", "pandas"),
            preferred_vector=d.get("preferred_vector", "numpy"),
            notes=list(d.get("notes", [])),
        )


@dataclass
class DoctorReport:
    """In-memory wrapper returned from run_doctor() for display."""
    profile: ComputeProfile

    def display(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            console = Console()
            p = self.profile
            console.print(Panel.fit(
                f"[bold]ledgerhound doctor -- compute profile[/bold]\n"
                f"machine: {p.machine.get('python')} | "
                f"{p.machine.get('platform')}\n"
                f"preferred dataframe : [cyan]{p.preferred_dataframe}[/cyan]\n"
                f"preferred vector    : [cyan]{p.preferred_vector}[/cyan]",
                border_style="cyan",
            ))
            t = Table("backend", "available", "ms", "error", title="Dataframe groupby (1M rows)")
            for r in p.dataframe_results:
                t.add_row(r.backend, str(r.available),
                          f"{r.ms:.1f}" if r.ms is not None else "-",
                          (r.error or "")[:60])
            console.print(t)
            t2 = Table("backend", "available", "ms", "error", title="Vector recall (100K x 384, k=10)")
            for r in p.vector_results:
                t2.add_row(r.backend, str(r.available),
                           f"{r.ms:.2f}" if r.ms is not None else "-",
                           (r.error or "")[:60])
            console.print(t2)
            if p.notes:
                console.print("\n[dim]" + "\n".join(p.notes) + "[/dim]")
        except Exception:
            # Rich missing -> plain text
            print(json.dumps(self.profile.to_dict(), indent=2))


# -- Path conventions -------------------------------------------------------


def default_profile_path(workspace_root: Optional[Union[str, Path]] = None) -> Path:
    """Where compute_profile.json lives. Inside a workspace if one is given,
    else under the user's home (~/.ledgerhound/compute_profile.json)."""
    if workspace_root is not None:
        return Path(workspace_root) / "compute_profile.json"
    return Path.home() / ".ledgerhound" / "compute_profile.json"


def save_profile(profile: ComputeProfile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load_profile(path: Path) -> Optional[ComputeProfile]:
    if not path.exists():
        return None
    try:
        return ComputeProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning("Could not load compute_profile.json: %s", exc)
        return None


# -- Benchmarks -------------------------------------------------------------


def _bench_dataframe(n_rows: int = 1_000_000) -> list[BackendTiming]:
    """Time a groupby-sum on pandas, and on cuDF if available."""
    results: list[BackendTiming] = []

    import pandas as pd
    rng = np.random.default_rng(42)
    n_groups = max(1000, n_rows // 1000)
    df_pd = pd.DataFrame({
        "account": rng.integers(0, n_groups, size=n_rows),
        "amount": rng.random(n_rows, dtype=np.float32) * 1000.0,
    })

    # pandas
    t0 = time.perf_counter()
    _ = df_pd.groupby("account")["amount"].sum()
    pandas_ms = (time.perf_counter() - t0) * 1000.0
    results.append(BackendTiming(backend="pandas", available=True, ms=pandas_ms))

    # cuDF
    try:
        import cudf  # type: ignore[import-not-found]
        df_cu = cudf.DataFrame.from_pandas(df_pd)
        t0 = time.perf_counter()
        _ = df_cu.groupby("account")["amount"].sum()
        cudf_ms = (time.perf_counter() - t0) * 1000.0
        results.append(BackendTiming(backend="cudf", available=True, ms=cudf_ms))
    except Exception as exc:
        results.append(BackendTiming(
            backend="cudf", available=False, ms=None,
            error=f"{type(exc).__name__}: {str(exc)[:100]}",
        ))

    return results


def _bench_vector(n_vec: int = 100_000, dim: int = 384, k: int = 10) -> list[BackendTiming]:
    """Time a k=10 nearest-neighbour search on each available vector backend."""
    from ledgerhound.memory.vector_store import (
        NumpyBruteForceStore, FaissCpuStore, VectorRecord,
        _have_faiss, _have_cuvs,
    )
    results: list[BackendTiming] = []
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_vec, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    query = mat[0].copy()
    records = [VectorRecord(id=f"v{i}", vector=mat[i], metadata={}) for i in range(n_vec)]

    # numpy
    try:
        store = NumpyBruteForceStore(dim)
        store.add(records)
        t0 = time.perf_counter()
        _ = store.search(query, k=k)
        ms = (time.perf_counter() - t0) * 1000.0
        results.append(BackendTiming(backend="numpy", available=True, ms=ms))
    except Exception as exc:
        results.append(BackendTiming(
            backend="numpy", available=True, ms=None,
            error=f"{type(exc).__name__}: {str(exc)[:100]}",
        ))

    # faiss
    if _have_faiss():
        try:
            store = FaissCpuStore(dim)
            store.add(records)
            t0 = time.perf_counter()
            _ = store.search(query, k=k)
            ms = (time.perf_counter() - t0) * 1000.0
            results.append(BackendTiming(backend="faiss", available=True, ms=ms))
        except Exception as exc:
            results.append(BackendTiming(
                backend="faiss", available=True, ms=None,
                error=f"{type(exc).__name__}: {str(exc)[:100]}",
            ))
    else:
        results.append(BackendTiming(
            backend="faiss", available=False, ms=None,
            error="faiss-cpu not installed",
        ))

    # cuvs
    if _have_cuvs():
        results.append(BackendTiming(
            backend="cuvs", available=True, ms=None,
            error="cuVS backend stub: timing not implemented",
        ))
    else:
        results.append(BackendTiming(
            backend="cuvs", available=False, ms=None,
            error="cuvs not installed",
        ))

    return results


# -- Decision rules ---------------------------------------------------------


def _pick_dataframe(results: list[BackendTiming]) -> tuple[str, list[str]]:
    notes: list[str] = []
    pandas = next((r for r in results if r.backend == "pandas"), None)
    cudf = next((r for r in results if r.backend == "cudf"), None)
    if cudf is None or not cudf.available or cudf.ms is None:
        notes.append("cuDF unavailable -- defaulting to pandas")
        return "pandas", notes
    if pandas is None or pandas.ms is None:
        return "cudf", notes
    # Only prefer cuDF if it's at least ~30% faster (transfer + memory overhead
    # rarely worth a marginal win).
    if cudf.ms < 0.7 * pandas.ms:
        notes.append(f"cuDF is {pandas.ms / cudf.ms:.2f}x faster than pandas at 1M-row groupby")
        return "cudf", notes
    notes.append(f"cuDF not decisively faster (pandas={pandas.ms:.1f}ms, cudf={cudf.ms:.1f}ms) -- staying on pandas")
    return "pandas", notes


def _pick_vector(results: list[BackendTiming]) -> tuple[str, list[str]]:
    notes: list[str] = []
    timed = [r for r in results if r.available and r.ms is not None]
    if not timed:
        notes.append("no vector backend benchmarked -- defaulting to numpy")
        return "numpy", notes
    fastest = min(timed, key=lambda r: r.ms)  # type: ignore[arg-type]
    notes.append(f"fastest vector backend: {fastest.backend} ({fastest.ms:.2f}ms for 100K x 384, k=10)")
    return fastest.backend, notes


# -- Public entrypoint ------------------------------------------------------


def run_doctor(
    *,
    workspace_root: Optional[Union[str, Path]] = None,
    save: bool = True,
    n_rows: int = 1_000_000,
    n_vec: int = 100_000,
    vector_dim: int = 384,
) -> DoctorReport:
    """Run the prerun benchmark and (optionally) persist compute_profile.json.

    Parameters
    ----------
    workspace_root : per-workspace location. If None, profile goes to
                     ~/.ledgerhound/compute_profile.json (machine-global).
    save           : write the profile to disk
    n_rows         : groupby benchmark size; lower for quick runs in CI
    n_vec          : vector benchmark size
    vector_dim     : embedding dim used in vector benchmark

    Returns
    -------
    DoctorReport -- call .display() for a Rich table, or .profile.to_dict()
    for the JSON.
    """
    log.info("Running compute benchmarks (this should take ~30s)...")
    df_results = _bench_dataframe(n_rows=n_rows)
    vec_results = _bench_vector(n_vec=n_vec, dim=vector_dim)
    pref_df, notes_df = _pick_dataframe(df_results)
    pref_vec, notes_vec = _pick_vector(vec_results)

    machine = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    profile = ComputeProfile(
        generated_at=time.time(),
        machine=machine,
        dataframe_results=df_results,
        vector_results=vec_results,
        preferred_dataframe=pref_df,
        preferred_vector=pref_vec,
        notes=notes_df + notes_vec,
    )

    if save:
        path = default_profile_path(workspace_root)
        save_profile(profile, path)
        log.info("Saved compute profile: %s", path)

    return DoctorReport(profile=profile)
