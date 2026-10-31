"""
sentinel.auto.kernel
====================

KernelExecutor -- SOTA notebook-centric execution backend.

Instead of emulating a Python environment with a restricted ``exec`` (which
breaks whenever generated code uses a language feature we didn't whitelist),
this drives a REAL persistent IPython kernel via ``jupyter_client``:

  * Full Python -- all builtins, classes, decorators, imports. Nothing to break.
  * Persistent state -- ``df`` is loaded ONCE and stays in kernel memory across
    every step (no per-step reload; one CUDA context for the whole run).
  * Real isolation -- code runs in a separate process; timeouts interrupt it.
  * Accident-prevention -- a ``scratch``-only write guard is injected into the
    kernel namespace, so generated code still cannot write outside its run dir.

Interface-compatible with ScratchExecutor: exposes ``_df`` (a cheap metadata
proxy), ``_scratch``, and ``execute(code) -> ExecResult`` so the runner needs
no changes. Falls back to ScratchExecutor automatically when jupyter_client /
ipykernel are unavailable.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from sentinel.auto.sandbox import ExecResult
from sentinel.shell.write_guard import link_features_into

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Run INSIDE the kernel immediately before `import cudf`. Two jobs:
#
#   pool_allocator  -- cuDF's default allocator calls cudaMalloc per
#                      allocation, which synchronises the device. On a
#                      groupby-heavy EDA session that alone is a large
#                      fraction of runtime. A pool allocates once up front.
#   managed_memory  -- CUDA unified memory. Allocations may EXCEED the
#                      card and page to host instead of raising OOM. This
#                      is what turns "dataset is 8 GB over the ceiling" from
#                      a fall-back-to-CPU event into a slightly slower GPU
#                      run, and it is the direct answer to "use all the
#                      memory available".
#
# Entirely best-effort: rmm is part of the RAPIDS install, but if it is
# missing or reinitialize() fails we leave cuDF's default allocator alone
# and carry on. Never fatal -- a slow GPU beats no GPU.
_RMM_INIT = r'''
_rmm_status = "not attempted"
try:
    import rmm, cupy as _cp
    from rmm.allocators.cupy import rmm_cupy_allocator
    _free, _total = _cp.cuda.runtime.memGetInfo()
    # Claim half of what is free as the initial pool; managed memory lets it
    # grow past the card if the data turns out bigger than we sized for.
    _pool = int(_free * 0.5) // (256 * 1024 * 1024) * (256 * 1024 * 1024)
    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
        initial_pool_size=_pool if _pool > 0 else None,
    )
    _cp.cuda.set_allocator(rmm_cupy_allocator)
    _rmm_status = ("pool %.0fGB, managed memory ON (can oversubscribe past "
                   "the card)" % (_pool / 1024**3))
except Exception as _e:
    _rmm_status = "unavailable (%s: %s) -- default cuDF allocator" % (
        type(_e).__name__, _e)
print("RMM:", _rmm_status)
'''

# Run in the kernel to hand GPU memory back. Ordering matters: drop Python
# references first, collect, then release the pools -- freeing a pool while a
# frame still points into it is what leaves memory stranded.
_GPU_RELEASE = r'''
_released = []
try:
    import gc
    for _n in [n for n in dir() if not n.startswith('_')]:
        try:
            _o = eval(_n)
        except Exception:
            continue
        if hasattr(_o, 'columns') and hasattr(_o, 'dtypes'):
            del globals()[_n]
            _released.append(_n)
    gc.collect()
except Exception:
    pass
try:
    import cupy as _cp
    _cp.get_default_memory_pool().free_all_blocks()
    _cp.get_default_pinned_memory_pool().free_all_blocks()
except Exception:
    pass
try:                      # dask-cudf: stop the workers or the pools persist
    _client.close(); _cluster.close()
except Exception:
    pass
try:
    import cupy as _cp2
    _f, _t = _cp2.cuda.runtime.memGetInfo()
    print("GPU released: dropped %s, free %.1f/%.1f GB"
          % (_released or 'nothing', _f / 1024**3, _t / 1024**3))
except Exception:
    print("GPU released: dropped", _released or 'nothing')
'''


# Appended to every cell when a GPU engine is live. Any large pandas frame
# the cell left behind is moved onto the GPU before the next cell runs, so
# one CPU-loading mistake does not silently make the rest of the session
# CPU-bound.
#
# The row threshold is the whole safety story. `to_pdf(df)` exists so the
# analyst can hand a CPU frame to scipy/sklearn/matplotlib, and dragging
# THAT back to the GPU would break the very call it was made for. Analysis
# frames at crore scale are far above the threshold; plotting and stats
# extracts are far below it. So: convert the big ones, never touch the
# small ones.
_FORCE_GPU_FRAMES = r'''
try:
    import cudf as _cudf_f, pandas as _pd_f
    _moved = []
    for _n in [n for n in dir() if not n.startswith('_')]:
        try:
            _o = eval(_n)
        except Exception:
            continue
        if isinstance(_o, _pd_f.DataFrame) and len(_o) >= __FORCE_MIN_ROWS__:
            try:
                globals()[_n] = _cudf_f.from_pandas(_o)
                _moved.append('%s (%d rows)' % (_n, len(_o)))
            except Exception as _e:
                print('GPU-FORCE: could not move %s to GPU (%s: %s) -- left '
                      'on CPU' % (_n, type(_e).__name__, _e))
    if _moved:
        print('GPU-FORCE: moved to GPU ->', ', '.join(_moved))
        print('GPU-FORCE: (a cell built these on CPU; they are cuDF now. '
              'Use to_pdf(x) if you deliberately need a pandas frame.)')
except Exception:
    pass
'''


def _force_gpu_frames_enabled() -> bool:
    """On by default when a GPU engine is live; gpu.force_gpu_frames=false
    in main_config.yaml (or SENTINEL_FORCE_GPU_FRAMES=0) opts out.

    Deliberately NOT tied to gpu.enable_dask_cudf: that switch controls
    *multi-GPU*, and someone who turns it off is asking for single-GPU cuDF,
    not for CPU. Tying the two would silently stop pulling frames onto the
    GPU for exactly the users who just said they still want one GPU.
    """
    from sentinel.config import get_bool
    return get_bool("gpu", "force_gpu_frames", env="SENTINEL_FORCE_GPU_FRAMES",
                    default=True)


def _force_gpu_min_rows() -> int:
    """Row count above which a pandas frame is considered analysis data."""
    from sentinel.config import get_int
    v = get_int("gpu", "force_gpu_min_rows", env="SENTINEL_FORCE_GPU_MIN_ROWS",
               default=1_000_000)
    return max(1, v)


def _cudf_pandas_enabled() -> bool:
    """cudf.pandas is on by default on a GPU box; gpu.cudf_pandas=false in
    main_config.yaml (or SENTINEL_CUDF_PANDAS=0) opts out.

    Default-on because it fixes the silent-CPU path that no prompt wording
    reliably prevents. A kill switch exists because it is the change with the
    widest blast radius here: it replaces the `pd` every generated cell uses.
    """
    from sentinel.config import get_bool
    return get_bool("gpu", "cudf_pandas", env="SENTINEL_CUDF_PANDAS", default=True)

# Injected into every kernel: robust helpers that paper over the common
# cuDF<->pandas API gaps that crash generated code (merge key-dtype mismatch,
# strict to_datetime parsing, GPU-OOM parquet writes). Generated code is told
# to prefer these.
_SAFE_HELPERS = r'''
def safe_to_datetime(s, fmt=None):
    """Parse to datetime robustly. cuDF raises on bad/mixed formats; on any
    failure we round-trip through pandas with errors='coerce'."""
    try:
        import pandas as _pd
        try:
            import cudf as _cudf
            _is_cudf = hasattr(s, "to_pandas")
        except Exception:
            _cudf = None; _is_cudf = False
        if _is_cudf:
            try:
                return _cudf.to_datetime(s, format=fmt) if fmt else _cudf.to_datetime(s)
            except Exception:
                _p = _pd.to_datetime(s.to_pandas(), format=fmt, errors="coerce")
                return _cudf.from_pandas(_p)
        return _pd.to_datetime(s, format=fmt, errors="coerce")
    except Exception:
        return s

def safe_merge(left, right, on=None, how="inner", **kw):
    """Merge that first aligns join-key dtypes (cuDF errors on mismatched key
    dtypes, e.g. int64 vs object). Casts keys to string when they differ."""
    keys = on if isinstance(on, (list, tuple)) else ([on] if on else [])
    L, R = left.copy(), right.copy()
    for k in keys:
        if k in L.columns and k in R.columns and str(L[k].dtype) != str(R[k].dtype):
            try:
                L[k] = L[k].astype("str"); R[k] = R[k].astype("str")
            except Exception:
                pass
    try:
        return L.merge(R, on=on, how=how, **kw)
    except Exception:
        # last resort: do it in pandas, return same engine as `left`
        Lp = L.to_pandas() if hasattr(L, "to_pandas") else L
        Rp = R.to_pandas() if hasattr(R, "to_pandas") else R
        out = Lp.merge(Rp, on=on, how=how, **kw)
        if hasattr(left, "to_pandas"):
            try:
                import cudf as _cudf
                return _cudf.from_pandas(out)
            except Exception:
                return out
        return out

def safe_to_parquet(df, path, row_group_size=500000):
    """Write parquet without GPU OOM: convert cuDF->pandas, write via pyarrow
    in row-group batches."""
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    try:
        import pyarrow as _pa, pyarrow.parquet as _pq
        tbl = _pa.Table.from_pandas(pdf)
        _pq.write_table(tbl, str(path), row_group_size=row_group_size)
    except Exception:
        pdf.to_parquet(str(path))

def to_pdf(df):
    """Return a pandas view (no-op for pandas, .to_pandas() for cuDF)."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df
'''


def kernels_available() -> bool:
    """True if a real IPython kernel backend can be started."""
    try:
        import ipykernel  # noqa: F401
        import jupyter_client  # noqa: F401
        return True
    except Exception:
        return False


class _DFMeta:
    """Lightweight dataframe metadata proxy.

    The full dataframe lives inside the kernel; the agent process only needs
    columns / shape / dtypes for prompts and knowledge retrieval. We read those
    cheaply from the file schema instead of loading the data twice.
    """

    __slots__ = ("columns", "shape", "dtypes")

    def __init__(self, columns: list[str], n_rows: int, dtypes: dict[str, str]):
        self.columns = columns
        self.shape = (n_rows, len(columns))
        self.dtypes = dtypes

    @classmethod
    def from_path(cls, path: Path) -> "_DFMeta | None":
        try:
            if path.is_dir():
                import pyarrow.dataset as pads
                ds = pads.dataset(str(path), format="parquet", partitioning="hive")
                cols = list(ds.schema.names)
                dtypes = {n: str(ds.schema.field(n).type) for n in cols}
                return cls(cols, int(ds.count_rows()), dtypes)
            if path.suffix in (".parquet", ".pq"):
                import pyarrow.parquet as pq
                schema = pq.read_schema(path)
                meta = pq.read_metadata(path)
                cols = list(schema.names)
                dtypes = {n: str(schema.field(n).type) for n in cols}
                return cls(cols, meta.num_rows, dtypes)
            if path.suffix in (".csv", ".tsv"):
                import pandas as pd
                sep = "\t" if path.suffix == ".tsv" else ","
                head = pd.read_csv(path, sep=sep, nrows=200)
                # cheap row count
                with path.open("rb") as f:
                    n = sum(1 for _ in f) - 1
                cols = list(head.columns)
                dtypes = {c: str(head[c].dtype) for c in cols}
                return cls(cols, max(n, 0), dtypes)
        except Exception:
            return None
        return None


class KernelExecutor:
    """Execute generated code in a persistent real IPython kernel."""

    def __init__(self, scratch_dir: Path, data_path: str | None = None,
                 engine: str = "pandas", timeout: int = 120,
                 features_dir: Path | None = None):
        self._scratch = Path(scratch_dir)
        self._scratch.mkdir(parents=True, exist_ok=True)
        # V3-049: the kernel process's cwd is already `scratch` (see
        # start_kernel below), so relative writes never escaped here the
        # way they did in exec mode -- but the model still had no dedicated
        # place for FEATURE tables specifically. Defaults to scratch when
        # the caller has nowhere better (no project to key a shared
        # workspace off of).
        self._features_dir = Path(features_dir) if features_dir else self._scratch
        self._features_dir.mkdir(parents=True, exist_ok=True)
        # The kernel's cwd is scratch, so a relative `features/x.parquet`
        # (read OR write) means scratch/features. Make that the real
        # features folder, so what the model saves is where it reads it back.
        link_features_into(self._scratch, self._features_dir)
        self._timeout = timeout
        self._engine = engine
        self._data_path = data_path
        self._df = _DFMeta.from_path(Path(data_path)) if data_path else None
        # V3-047: every GPU->CPU downgrade during startup, with an RCA.
        # KernelExecutor has no EventStream of its own, so it accumulates
        # here and `_start_kernel_for` drains it into the analyst's log.
        self.gpu_fallbacks: list[dict] = []
        self._rmm_status = ""
        self._requested_engine = engine

        from rich.console import Console
        self._console = Console()

        from jupyter_client import KernelManager
        with self._console.status("[magenta]Starting analysis kernel...[/magenta]"):
            self._km = KernelManager(kernel_name="python3")
            self._km.start_kernel(cwd=str(self._scratch))
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=60)

        self._run_setup()

    # ------------------------------------------------------------------ #

    def _record_gpu_fallback(self, stage: str, error_text: str,
                             fell_back_to: str) -> dict:
        """Classify a GPU downgrade, print the RCA, and remember it.

        Printed immediately rather than at the end: the pandas retry after a
        failed load can run for many minutes on an 8-crore frame, and the
        operator should know WHY they are waiting while they wait, not
        afterwards.
        """
        from sentinel.backends.compute_router import classify_gpu_error
        error_class, remedy = classify_gpu_error(error_text, self._rmm_status)
        note = {
            "stage": stage,
            "error_class": error_class,
            "error_text": (error_text or "")[:2000],
            "remedy": remedy,
            "fell_back_to": fell_back_to,
            "rmm_status": self._rmm_status,
        }
        self.gpu_fallbacks.append(note)
        self._console.print(
            f"\n  [yellow]GPU UNAVAILABLE at '{stage}' -> using "
            f"{fell_back_to}.[/yellow]\n"
            f"  [dim]cause : {error_class}\n"
            f"  fix   : {remedy}[/dim]\n")
        return note

    def _run_setup(self) -> None:
        """Phased, visible startup so a slow phase never looks like a hang.

        Phase 1: lightweight imports + scratch + write-guard  (fast)
        Phase 2: dataframe engine import (cuDF CUDA init can be slow)
        Phase 3: one-time data preload                        (df persists)
        """
        # -- Phase 0a: RMM, before ANY import that touches CUDA --
        # An allocator can only be swapped before the first device
        # allocation. `import cudf.pandas` (Phase 0b) imports cudf, which
        # creates the CUDA context and installs RMM's default per-cudaMalloc
        # resource -- so RMM has to be configured ahead of it, not in Phase 2
        # where the cuDF import lives. Getting this backwards would leave
        # reinitialize() firing against a live context holding device
        # buffers, which is the ordering RAPIDS documents as unsafe.
        self._rmm_note = ""
        if self._engine in ("cudf", "dask_cudf"):
            rmm_res = self._exec_raw(_RMM_INIT, timeout=120)
            for line in (getattr(rmm_res, "output", "") or "").splitlines():
                if line.startswith("RMM:"):
                    self._rmm_note = line[4:].strip()
            # Kept separately from the display note: classify_gpu_error reads
            # it to tell "OOM because the allocator never came up" apart from
            # "OOM because the data really is too big".
            self._rmm_status = self._rmm_note

        # -- Phase 0b: transparent GPU pandas (must precede `import pandas`) --
        # cudf.pandas rewrites `pd` into a proxy that runs each operation on
        # the GPU and falls back to CPU per-operation when cuDF cannot do it
        # or the data will not fit. That covers the failure the system prompt
        # can only ask about: a cell that does `pd.read_parquet(...)` itself
        # now lands on the GPU regardless of what the model wrote.
        #
        # install() MUST run before pandas is imported anywhere in the kernel,
        # hence Phase 0. Only attempted when the router picked a GPU engine --
        # on a CPU box it would be pure overhead. Kill switch:
        # SENTINEL_CUDF_PANDAS=0.
        if self._engine in ("cudf", "dask_cudf") and _cudf_pandas_enabled():
            r0 = self._exec_raw(
                "_cudf_pandas = 'off'\n"
                "try:\n"
                "    import cudf.pandas\n"
                "    cudf.pandas.install()\n"
                "    _cudf_pandas = 'on'\n"
                "except Exception as _e:\n"
                "    _cudf_pandas = 'unavailable (%s)' % type(_e).__name__\n"
                "print('cudf.pandas:', _cudf_pandas)\n", timeout=180)
            out = (getattr(r0, "output", "") or "")
            if "cudf.pandas: on" in out:
                self._console.print("  [dim]cudf.pandas active -- plain "
                                    "`pd` code runs on GPU with per-op CPU "
                                    "fallback.[/dim]")
            elif "unavailable" in out:
                self._console.print("  [yellow]cudf.pandas unavailable -- "
                                    "`pd` stays CPU-only.[/yellow]")

        # -- Phase 1: lightweight environment (kernel ready quickly) --
        base = (
            "import pandas as pd, numpy as np, json, math, re\n"
            "from pathlib import Path\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "try:\n    import seaborn as sns\nexcept Exception:\n    sns = None\n"
            f"scratch = Path(r'''{self._scratch}''')\n"
            f"features_dir = Path(r'''{self._features_dir}''')\n"
            "import builtins as _bi\n"
            "_real_open = _bi.open\n"
            "def _wg_notice(_orig, _redirected):\n"
            "    print('WRITE REDIRECTED: \\'%s\\' is outside the workspace -> \\'%s\\'' % (_orig, _redirected))\n"
            "try:\n"
            "    from sentinel.shell.write_guard import resolve_write_path as _wg_resolve\n"
            "except Exception:\n"
            "    _wg_resolve = None\n"
            "def open(file, mode='r', *a, **k):\n"
            "    # Same rule as every other writer (sentinel.shell.write_guard):\n"
            "    # 'features/...' means features_dir, parents are created, and a\n"
            "    # write outside the workspace is redirected, not refused.\n"
            "    if isinstance(file, (str, Path)) and not Path(file).is_absolute():\n"
            "        _parts = Path(file).parts\n"
            "        if _parts and _parts[0] in ('features', 'features_dir'):\n"
            "            file = features_dir.joinpath(*_parts[1:])\n"
            "    if _wg_resolve is not None and any(_m in mode for _m in ('w', 'a', 'x', '+')):\n"
            "        file = _wg_resolve(file, [scratch, features_dir], features_dir, _wg_notice)\n"
            "    return _real_open(file, mode, *a, **k)\n"
            # V3-063: the open() guard above only catches direct open()
            # calls -- df.to_csv()/to_parquet()/plt.savefig() open their
            # own handles through pandas/matplotlib internals and never
            # touch this. This kernel process is DEDICATED to running
            # generated cells (nothing else shares it), so installing the
            # write guard for the kernel's whole lifetime is safe -- unlike
            # exec mode, which shares an interpreter with the rest of
            # Sentinel and must bracket it per-cell instead.
            "try:\n"
            "    from sentinel.shell.write_guard import install as _install_write_guard\n"
            "    _uninstall_write_guard = _install_write_guard(\n"
            "        lambda: [scratch, features_dir], lambda: features_dir, _wg_notice)\n"
            "except Exception as _wg_e:\n"
            "    print('write guard not installed:', type(_wg_e).__name__, _wg_e)\n"
        )
        self._exec_raw(base, timeout=60)
        self._console.print("  [dim]Kernel ready (imports + write-guard).[/dim]")

        # -- Phase 2: engine import (isolate the slow cuDF/CUDA init) --
        if self._engine == "dask_cudf":
            # Multi-GPU: the dataset exceeds one card but fits the pool.
            # rmm_managed_memory + jit_unspill let every worker oversubscribe
            # its card and spill to host rather than dying on OOM; without
            # them a LocalCUDACluster is no more capacious than one GPU.
            with self._console.status("[magenta]Starting dask-cudf cluster...[/magenta]"):
                t = time.perf_counter()
                r = self._exec_raw(
                    "from dask_cuda import LocalCUDACluster\n"
                    "from dask.distributed import Client\n"
                    "import dask_cudf as _xdf\n"
                    "_cluster = LocalCUDACluster(\n"
                    "    rmm_pool_size='auto',\n"
                    "    rmm_managed_memory=True,\n"
                    "    jit_unspill=True,\n"
                    ")\n"
                    "_client = Client(_cluster)\n"
                    "print('dask-cudf workers:', len(_client.scheduler_info()"
                    "['workers']))\n", timeout=300)
                dt = time.perf_counter() - t
            if r.success:
                self._console.print(f"  [dim]dask-cudf ready ({dt:.0f}s) -- "
                                    f"multi-GPU, managed memory.[/dim]")
            else:
                self._engine = "cudf"
                from sentinel.backends.compute_router import DASK_CUDF_OFF_HINT
                note = self._record_gpu_fallback(
                    "dask_cudf_cluster_start",
                    getattr(r, "error", "") or getattr(r, "output", "")
                    or "LocalCUDACluster failed to start",
                    "single-GPU cuDF")
                note["remedy"] += " " + DASK_CUDF_OFF_HINT
        if self._engine == "cudf":
            with self._console.status("[magenta]Initialising cuDF / CUDA context...[/magenta]"):
                t = time.perf_counter()
                # RMM was configured in Phase 0a, before anything touched
                # CUDA. Do NOT reinitialize here -- cudf is live by now.
                r = self._exec_raw("import cudf as _xdf", timeout=180)
                dt = time.perf_counter() - t
            if r.success:
                note = f" -- {self._rmm_note}" if self._rmm_note else ""
                self._console.print(f"  [dim]cuDF ready ({dt:.0f}s){note}.[/dim]")
            else:
                self._engine = "pandas"
                self._exec_raw("import pandas as _xdf", timeout=30)
                self._record_gpu_fallback(
                    "cudf_import",
                    getattr(r, "error", "") or getattr(r, "output", "")
                    or "import cudf failed with no captured output",
                    "CPU pandas")
        elif self._engine == "pandas":
            self._exec_raw("import pandas as _xdf", timeout=30)

        # -- Phase 2b: inject cuDF-safe helpers (robust merge/datetime/write) --
        self._exec_raw(_SAFE_HELPERS, timeout=60)

        # -- Phase 3: one-time data preload (skipped for directories) --
        if self._data_path:
            self._preload_data()

    def _preload_data(self) -> None:
        sp = str(self._data_path)
        is_dir = Path(sp).is_dir()
        parquet_dir = is_dir and any(
            f.suffix.lower() in (".parquet", ".pq") for f in Path(sp).rglob("*"))
        if is_dir and not parquet_dir:
            # A directory/glob is the plan's job to load+concat; don't double-load.
            self._console.print(
                "  [dim]Data is a directory -- the load step will read it "
                "(df persists in the kernel afterwards).[/dim]")
            return
        try:
            if parquet_dir:
                # A Parquet dataset folder (parts / hive partitions, e.g. an
                # object-store prefix): pandas, cuDF and dask-cuDF all read
                # the whole folder with one read_parquet call.
                size_mb = sum(f.stat().st_size for f in Path(sp).rglob("*")
                              if f.is_file()) / 1e6
            else:
                size_mb = Path(sp).stat().st_size / 1e6
        except Exception:
            size_mb = 0
        # K-05: mirror the shell's own reader (conversation._read_any) so the
        # kernel's `df` is the same data the shell described. Before this,
        # .tsv was read as comma-separated (one column), .json/.feather were
        # never preloaded -- `df` simply did not exist in the kernel and
        # every cell hit NameError -- and upper-case extensions (.PARQUET)
        # were not recognised at all.
        suffix = Path(sp).suffix.lower()
        if suffix in (".json", ".feather"):
            pd_reader = "read_json" if suffix == ".json" else "read_feather"
            src = (f"import pandas as _pd_load\n"
                   f"df = _pd_load.{pd_reader}(r'''{sp}''')\n"
                   "if hasattr(_xdf, 'from_pandas'):\n"
                   "    try:\n"
                   "        df = _xdf.from_pandas(df) if _xdf.__name__ == 'cudf' "
                   "else df\n"
                   "    except Exception:\n"
                   "        pass\n"
                   "print('rows', len(df), 'cols', len(df.columns))")
            r = self._exec_raw(src, timeout=max(self._timeout, 300))
            if not r.success:
                self._console.print(f"  [yellow]kernel preload failed: "
                                    f"{(r.error or '')[:300]}[/yellow]")
            return
        if suffix in (".parquet", ".pq") or parquet_dir:
            reader, read_extra = "read_parquet", ""
        else:
            # .csv, .tsv, and -- like the shell -- anything unrecognised.
            reader = "read_csv"
            read_extra = ", sep='\\t'" if suffix == ".tsv" else ""
        # dask-cudf reads in parallel across every GPU, but hands back a LAZY
        # frame -- `df.groupby(...).mean()` becomes an unevaluated graph and
        # `print()` shows a repr instead of numbers. That silently breaks
        # every pandas-shaped helper (evaluate_feature, cached_agg, qcut) and
        # is the reason multi-GPU was fenced off in the first place. So: take
        # the parallel multi-GPU read, then materialise to an EAGER cuDF frame
        # so the generated-code contract is exactly what it was. With RMM
        # managed memory on, that eager frame may exceed one card and page to
        # host rather than OOM. If materialising fails anyway, keep the lazy
        # frame but say so loudly -- a warned-about lazy frame beats a crash.
        load_src = f"df = _xdf.{reader}(r'''{sp}'''{read_extra})\n"
        if self._engine == "dask_cudf":
            load_src += (
                "try:\n"
                "    df = df.compute()   # dask-cudf -> eager cudf\n"
                "    print('materialised to eager cuDF after multi-GPU read')\n"
                "except Exception as _e:\n"
                "    print('=' * 68)\n"
                "    print('LAZY-FRAME WARNING: could not materialise "
                "(%s: %s).' % (type(_e).__name__, _e))\n"
                "    print('df is a LAZY dask-cudf frame. groupby(...).mean() "
                "returns an')\n"
                "    print('unevaluated graph, print() shows a repr not "
                "numbers, and the')\n"
                "    print('pandas-shaped helpers will reject it. Call "
                ".compute() before')\n"
                "    print('printing or passing df to any helper.')\n"
                "    print('')\n"
                "    print('To avoid this entirely: set "
                "SENTINEL_ENABLE_DASK_CUDF=0 and')\n"
                "    print('restart -- that falls back to single-GPU cuDF, "
                "the tested path.')\n"
                "    print('=' * 68)\n")
        load_src += "print('rows', len(df), 'cols', len(df.columns))"

        with self._console.status(
                f"[magenta]Loading data into kernel ({size_mb:.0f} MB)...[/magenta]"):
            t = time.perf_counter()
            r = self._exec_raw(load_src, timeout=max(self._timeout, 300))
            dt = time.perf_counter() - t
        if r.success:
            self._console.print(f"  [dim]Data loaded once into kernel ({dt:.0f}s) -- "
                                f"persists across all steps.[/dim]")
        elif self._engine in ("cudf", "dask_cudf"):
            # GPU read failed (e.g. OOM) -> fall back to pandas for the load.
            # dask_cudf included: it was previously excluded, so a failed
            # multi-GPU load left `df` undefined instead of retrying on CPU.
            was_multi_gpu = self._engine == "dask_cudf"
            # RCA FIRST. The pandas retry below re-reads the whole file and
            # at crore scale that is minutes of apparent silence -- the
            # operator should know why they are waiting while they wait.
            note = self._record_gpu_fallback(
                "data_load",
                getattr(r, "error", "") or getattr(r, "output", "")
                or "GPU read failed with no captured output",
                "CPU pandas")
            if was_multi_gpu:
                from sentinel.backends.compute_router import DASK_CUDF_OFF_HINT
                note["remedy"] += " " + DASK_CUDF_OFF_HINT
            self._console.print(
                "  [yellow]Re-reading on CPU -- this will be slow at "
                "crore scale.[/yellow]")
            self._engine = "pandas"
            with self._console.status("[magenta]Retrying load with pandas...[/magenta]"):
                self._exec_raw(f"import pandas as _xdf\n"
                               f"df = _xdf.{reader}(r'''{sp}'''{read_extra})",
                               timeout=max(self._timeout, 300))

    # ------------------------------------------------------------------ #

    _MET_START = "__SENTINEL_METRICS__"
    _MET_END = "__END_METRICS__"
    _FP_START = "__SENTINEL_DFFP__"
    _FP_END = "__END_DFFP__"
    _OLF_START = "__SENTINEL_OLF__"
    _OLF_END = "__END_OLF__"

    def execute(self, code: str) -> ExecResult:
        files_before = set(self._scratch.rglob("*"))
        # Append an epilogue that emits the `metrics` dict as delimited JSON so
        # we can capture computed facts from the kernel namespace (structured
        # grounding). Wrapped in try/except so it never breaks the user code.
        #
        # V3-017: also emit a CHEAP df fingerprint (row count + column list
        # only -- no null-rate scan here, that would mean a full-column scan
        # on every single cell, including non-mutating ones, which is a real
        # cost at crore-scale). The shell compares this fingerprint before
        # vs after a cell and only pays for a null-rate delta (a second,
        # explicit round-trip) when rows or columns actually changed.
        epilogue = (
            "\ntry:\n"
            "    import json as _json\n"
            "    _m = metrics if 'metrics' in dir() and isinstance(metrics, dict) else {}\n"
            f"    print('{self._MET_START}' + _json.dumps(_m, default=str) + '{self._MET_END}')\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    import json as _json\n"
            "    if 'df' in dir():\n"
            "        _fp = {'n_rows': int(len(df)), 'columns': [str(c) for c in df.columns],\n"
            "               'other_large_frames': {}}\n"
            "    else:\n"
            "        _fp = {}\n"
            f"    print('{self._FP_START}' + _json.dumps(_fp, default=str) + '{self._FP_END}')\n"
            "except Exception:\n"
            "    pass\n"
            # Separate try/except (V3-034): a per-variable failure scanning
            # for redundant large copies must never take the df_fingerprint
            # above down with it -- that fingerprint drives the pre-existing
            # V3-017 mutation-delta check, not just this new one.
            "try:\n"
            "    import json as _json\n"
            "    if 'df' in dir():\n"
            "        _olf = {}\n"
            "        for _k in list(dir()):\n"
            "            try:\n"
            "                if _k.startswith('_') or _k == 'df':\n"
            "                    continue\n"
            "                _v = eval(_k)\n"
            "                if hasattr(_v, 'columns') and hasattr(_v, '__len__'):\n"
            "                    _olf[_k] = int(len(_v))\n"
            "            except Exception:\n"
            "                pass\n"
            f"        print('{self._OLF_START}' + _json.dumps(_olf, default=str) + '{self._OLF_END}')\n"
            "except Exception:\n"
            "    pass\n"
        )
        t0 = time.perf_counter()
        # V3-047: pull any large CPU frame the cell left behind back onto the
        # GPU, before the next cell inherits it. Runs last, after the metrics
        # and fingerprint epilogue, so a conversion failure cannot cost us
        # the cell's own results.
        if (self._engine in ("cudf", "dask_cudf")
                and _force_gpu_frames_enabled()):
            epilogue += _FORCE_GPU_FRAMES.replace(
                "__FORCE_MIN_ROWS__", str(_force_gpu_min_rows()))

        res = self._exec_raw(code + epilogue, timeout=self._timeout)
        elapsed = (time.perf_counter() - t0) * 1000

        # Extract and strip the metrics and fingerprint payloads from stdout.
        metrics, clean_output = self._extract_metrics(res.output)
        df_fingerprint, clean_output = self._extract_delimited(
            clean_output, self._FP_START, self._FP_END)
        other_large_frames, clean_output = self._extract_delimited(
            clean_output, self._OLF_START, self._OLF_END)
        if df_fingerprint:
            df_fingerprint["other_large_frames"] = other_large_frames

        files_after = set(self._scratch.rglob("*"))
        new_files = [str(f.relative_to(self._scratch))
                     for f in (files_after - files_before) if f.is_file()]

        return ExecResult(
            success=res.success,
            output=clean_output,
            error=res.error,
            execution_time_ms=round(elapsed, 1),
            artefacts=new_files,
            metrics=metrics,
            df_fingerprint=df_fingerprint,
        )

    def null_rates(self) -> dict:
        """On-demand, EXPENSIVE (full column scan) null-rate query -- only
        called when a cheap df_fingerprint comparison already showed rows or
        columns changed (V3-017), never on every cell."""
        res = self._exec_raw(
            "try:\n"
            "    import json as _json\n"
            "    _nr = {str(c): float(df[c].isna().mean()) for c in df.columns} \\\n"
            "          if 'df' in dir() else {}\n"
            f"    print('{self._FP_START}' + _json.dumps(_nr, default=str) + '{self._FP_END}')\n"
            "except Exception:\n"
            "    pass\n",
            timeout=self._timeout,
        )
        rates, _ = self._extract_delimited(res.output, self._FP_START, self._FP_END)
        return rates

    def _extract_delimited(self, output: str, start: str, end: str):
        import json as _json
        s = output.find(start)
        if s == -1:
            return {}, output
        e = output.find(end, s)
        if e == -1:
            return {}, output
        payload = output[s + len(start):e]
        clean = (output[:s] + output[e + len(end):]).strip()
        try:
            d = _json.loads(payload)
            return (d if isinstance(d, dict) else {}), clean
        except Exception:
            return {}, clean

    def _extract_metrics(self, output: str):
        import json as _json
        start = output.find(self._MET_START)
        if start == -1:
            return {}, output
        end = output.find(self._MET_END, start)
        if end == -1:
            return {}, output
        payload = output[start + len(self._MET_START):end]
        clean = (output[:start] + output[end + len(self._MET_END):]).strip()
        try:
            m = _json.loads(payload)
            return (m if isinstance(m, dict) else {}), clean
        except Exception:
            return {}, clean

    def _exec_raw(self, code: str, timeout: int) -> ExecResult:
        """Send code to the kernel and collect outputs."""
        from queue import Empty
        msg_id = self._kc.execute(code)
        out_chunks: list[str] = []
        error_text = ""

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except Empty:
                # Hung -- interrupt and report as failure
                try:
                    self._km.interrupt_kernel()
                except Exception:
                    pass
                return ExecResult(
                    success=False, output="".join(out_chunks).strip(),
                    error=f"TimeoutError: step exceeded {timeout}s and was interrupted.",
                )

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mt = msg["msg_type"]
            content = msg["content"]

            if mt == "stream":
                out_chunks.append(content.get("text", ""))
            elif mt == "execute_result":
                out_chunks.append(content.get("data", {}).get("text/plain", ""))
            elif mt == "display_data":
                txt = content.get("data", {}).get("text/plain", "")
                if txt:
                    out_chunks.append(txt)
            elif mt == "error":
                error_text = _ANSI.sub("", "\n".join(content.get("traceback", [])))
            elif mt == "status" and content.get("execution_state") == "idle":
                break

        output = "".join(out_chunks).strip()
        if error_text:
            # Trim to the most relevant tail (mirrors ScratchExecutor behaviour)
            lines = [l for l in error_text.splitlines() if l.strip()]
            short = "\n".join(lines[-30:]) if lines else error_text
            return ExecResult(success=False, output=output, error=short)
        return ExecResult(
            success=True,
            output=(output or "[Executed -- no output]"),
        )

    # ------------------------------------------------------------------ #

    def release_gpu(self) -> str:
        """Drop dataframes and hand GPU memory back to the driver.

        Killing the kernel process does free the GPU, but only once the
        process actually exits -- and a session that restarts the kernel for
        a new dataset (``_start_kernel_for``) previously overlapped the old
        and new CUDA contexts, so the second load contended with memory the
        first run had not released yet. On a shared 3-GPU box that is the
        difference between the next run getting the card and the next run
        quietly falling back to CPU because ``memGetInfo`` says there is no
        room.

        Call at shutdown or a dataset switch ONLY. This is destructive by
        design: it deletes every dataframe-shaped global, so `df` and
        everything derived from it are gone afterwards. It is not a
        mid-analysis "free some memory" button. Never raises.
        """
        if self._engine not in ("cudf", "dask_cudf"):
            return ""
        try:
            r = self._exec_raw(_GPU_RELEASE, timeout=120)
            for line in (getattr(r, "output", "") or "").splitlines():
                if line.startswith("GPU released:"):
                    return line
        except Exception:  # noqa: BLE001 -- teardown is best-effort
            pass
        return ""

    def shutdown(self) -> None:
        # K-06: idempotent. close() shuts down explicitly and __del__ calls
        # this again at garbage collection -- the second pass used to run the
        # GPU-release snippet against already-stopped channels.
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        # Release the GPU BEFORE the channels close -- afterwards there is no
        # way left to run the teardown snippet in the kernel.
        try:
            note = self.release_gpu()
            if note:
                self._console.print(f"  [dim]{note}[/dim]")
        except Exception:
            pass
        try:
            self._kc.stop_channels()
        except Exception:
            pass
        try:
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 -- never raise from a finaliser
            pass
