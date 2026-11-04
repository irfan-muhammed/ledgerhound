"""
ledgerhound.profiler.parallel
=============================

V3-027 -- column-group profiling fan-out over a process pool.

Where this fits in ProfilerEngine.run()'s 5-step pipeline (classify ->
quality -> statistics -> priority -> assemble): ONLY the statistics step
is embarrassingly parallel per column (confirmed before writing any code
-- StatisticalProfiler.profile()'s per-column loop touches only that
column's + the target column's data, self-contained ColumnStats out).
Quality (cross-column broken-join/duplicate checks) and priority (ranks
columns RELATIVE to each other, needs every column's stats already
computed) are NOT column-independent and stay sequential, unchanged.

Why this is the roadmap's actual "first target," not a generic
Orchestrator.run() process-pool: confirmed by reading Orchestrator.run()
and every shipped playbook before writing code -- every playbook today is
a strictly linear StudyTask chain (one task per dependency level), so
"same-level tasks execute concurrently" has no real multi-task level to
exercise ANYWHERE in this codebase yet. Worse, Orchestrator's
default_agent_factory returns a CLOSURE (captures an engine cache) --
closures are not picklable, so a Windows spawn'd ProcessPoolExecutor
cannot run it at all without a rewrite of agent-task dispatch this
closure doesn't attempt. Column-group profiling is the one place real
parallelism was both possible AND had a real, close-out-testable target
(a wide table) -- so that's what's built. Orchestrator.run()'s sequential
StudyTask loop is UNCHANGED.

Windows/spawn constraints (this dev environment: sys.platform=='win32',
multiprocessing default start method is 'spawn', not 'fork' -- confirmed
before writing code): every object crossing the pool boundary must be
picklable. `_profile_column_group` is a MODULE-LEVEL function (required
for pickling under spawn -- a closure or bound method would not survive).
A plain `PandasBackend`-wrapped `pd.DataFrame` subset has no live
connections/closures and pickles fine. The cuDF/GPU backend pins a CUDA
device via a live context that does NOT survive a spawn'd subprocess
boundary the way it unsafely might under POSIX fork -- `
profile_statistics_parallel` therefore only pools when `backend.df` is a
genuine `pd.DataFrame` (confirmed via isinstance, not duck-typing), and
falls back to the existing sequential path for anything else (cuDF,
Trino-backed, or any future backend without a live in-process
DataFrame).

Two determinism traps that matter for "identical card content" and
apply to the CALLER, not this module (confirmed before writing code):
  1. `DigestFormatter.format()` stamps `meta.generated` with
     `datetime.now()` -- a sequential-vs-parallel comparison must exclude
     that one field, or it will never match regardless of correctness.
  2. Column order: this module reassembles `StatsReport.columns` from
     the ORIGINAL `columns` dict's key order (`col_names`), never from
     worker-completion order -- required for a byte-identical
     `column_profiles` ordering in the assembled Digest, since Digest.save()
     writes YAML with `sort_keys=False`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import pandas as pd

from ledgerhound.profiler.detectors.statistics import StatisticalProfiler, StatsReport


def _profile_column_group(
    df_subset: pd.DataFrame,
    columns: dict[str, Any],
    target_col: Optional[str],
) -> dict[str, Any]:
    """Module-level (picklable) worker -- profiles ONE column group in a
    subprocess. Re-imports PandasBackend inside the function body so
    nothing from the parent's live state crosses the pickle boundary
    except plain data: a DataFrame subset, ColumnSchema-like dataclasses,
    and a string. Returns a plain dict[str, ColumnStats] -- ColumnStats
    is itself a plain dataclass tree, trivially picklable back to the
    parent."""
    from ledgerhound.backends.pandas_backend import PandasBackend
    backend = PandasBackend(df_subset)
    report = StatisticalProfiler().profile(backend, columns, target_col)
    return report.columns


def _split_round_robin(items: list[str], n_groups: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, item in enumerate(items):
        groups[i % n_groups].append(item)
    return groups


def profile_statistics_parallel(
    backend: Any,
    columns: dict[str, Any],
    target_col: Optional[str] = None,
    n_workers: int = 3,
) -> StatsReport:
    """Fan out StatisticalProfiler.profile() across up to n_workers
    processes, grouped by column (round-robin), then merge results back
    in ORIGINAL column order. Falls back to the plain sequential path --
    same StatisticalProfiler().profile() call ProfilerEngine.run() has
    always made -- when there's nothing to gain or nothing safe to pool:
    n_workers<=1, <=1 column, or backend.df isn't a real pd.DataFrame
    (see module docstring re: cuDF/CUDA context)."""
    col_names = list(columns.keys())
    df = getattr(backend, "df", None)
    if n_workers <= 1 or len(col_names) <= 1 or not isinstance(df, pd.DataFrame):
        return StatisticalProfiler().profile(backend, columns, target_col)

    groups = [g for g in _split_round_robin(col_names, n_workers) if g]

    merged: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=len(groups)) as ex:
        futures = []
        for group_cols in groups:
            sub_cols = {c: columns[c] for c in group_cols}
            # target_col must ride along in EVERY worker's subset, not
            # just workers whose own group happens to omit it: by_target/
            # ks_test need the target column's data regardless of which
            # group a given feature column landed in via round-robin (a
            # real bug caught by test_parallel_and_sequential_produce_
            # identical_column_stats -- the target column being one of
            # `columns`' own keys does NOT mean it's in THIS group).
            keep = list(group_cols)
            if target_col and target_col not in group_cols and target_col in df.columns:
                keep.append(target_col)
            df_subset = df[keep].copy()
            futures.append(ex.submit(_profile_column_group, df_subset, sub_cols, target_col))
        for fut in futures:
            merged.update(fut.result())

    report = StatsReport()
    report.columns = {c: merged[c] for c in col_names if c in merged}
    return report
