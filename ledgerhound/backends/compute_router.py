"""
ledgerhound.backends.compute_router
=================================

Detects available compute (GPU count, memory, data size) and routes
to the optimal backend: CPU pandas, single-GPU cuDF, or multi-GPU.

Decision tree::

    data_size = estimate_gb(data_path)
    gpus = detect_gpus()

    if gpus == 0 or data_size > 0.7 * smallest_gpu_mem:
        -> CPU pandas (safe fallback)
    elif gpus == 1:
        -> CPU pandas for data + GPU 0 for LLM
    elif gpus >= 2:
        -> GPU 0 for LLM, GPU 1+ for cuDF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ComputeProfile:
    """Detected compute environment + the role->device allocation.

    Allocation rule (the thing that was previously computed but never
    enforced):

    * API LLM (gateway Kimi / DeepSeek / any remote endpoint) -- no GPU is
      reserved for the model, so GPU0 is free: cuDF/compute uses it and
      every GPU is available for data work.
    * Local LLM (a HuggingFace model Ledgerhound loads, or a local
      vLLM/localhost server) -- reserve ``llm_gpu`` (default GPU0) for the
      model; cuDF/compute uses the remaining GPUs.

    ``llm_device`` is ``""`` when the LLM is a remote API (nothing to pin).
    """
    gpu_count: int = 0
    gpu_names: list[str] = None
    gpu_memory_gb: list[float] = None
    # Currently-free memory per GPU, at detection time -- separate from
    # gpu_memory_gb (total card capacity). On a shared/multi-tenant box
    # these diverge a lot: confirmed on a live staging GPU with another
    # process already resident, total=49GB but free=5GB. Empty when a probe
    # couldn't report it (older nvidia-smi, etc.) -- _route() falls back to
    # the total-only check in that case rather than raising.
    gpu_memory_free_gb: list[float] = None
    data_size_gb: float = 0.0
    backend_choice: str = "pandas"       # "pandas" | "cudf"
    llm_device: str = ""                 # "cuda:0" (local) | "" (API/CPU)
    data_device: str = "cpu"             # "cuda:0"/"cuda:1" | "cpu"
    llm_is_local: bool = False
    llm_gpu: int = 0
    compute_gpus: list[int] = None       # GPUs available to cuDF/model
    reason: str = ""
    # Which probe actually found the GPUs ("torch"/"cupy"/"nvidia-smi"), or
    # "" when none did. Recorded so a "why am I on CPU?" question is
    # answerable from the profile alone.
    gpu_detected_via: str = ""

    def __post_init__(self):
        self.gpu_names = self.gpu_names or []
        self.gpu_memory_gb = self.gpu_memory_gb or []
        self.gpu_memory_free_gb = self.gpu_memory_free_gb or []
        self.compute_gpus = self.compute_gpus or []

    def data_device_index(self) -> int | None:
        """Numeric GPU index for the data backend, or None if on CPU."""
        if self.data_device.startswith("cuda:"):
            return int(self.data_device.split(":", 1)[1])
        return None

    def summary(self) -> str:
        via = f" [via {self.gpu_detected_via}]" if self.gpu_detected_via else ""
        lines = [
            f"GPUs: {self.gpu_count}{via} "
            f"({', '.join(self.gpu_names) if self.gpu_names else 'none'})",
            f"Data: {self.data_size_gb:.1f} GB",
            f"Backend: {self.backend_choice}",
            f"LLM device: {self.llm_device}",
            f"Data device: {self.data_device}",
            f"Reason: {self.reason}",
        ]
        return "\n".join(lines)


class ComputeRouter:
    """Detect hardware and choose the optimal backend."""

    # Substrings in an LLM config that mean the model runs on a LOCAL GPU
    # (either loaded by Ledgerhound, or a local inference server we own).
    _LOCAL_LLM_TOKENS = (
        "huggingface_local", "vllm", "ollama", "tgi", "llama.cpp",
        "localhost", "127.0.0.1", "0.0.0.0",
    )
    # Substrings that unambiguously mean a REMOTE API (no local GPU).
    _API_LLM_TOKENS = (
        "anthropic", "claude", "openai", "azure", "bedrock", "vertex",
        "gemini", "cohere", "together", "groq", "deepseek.com",
        "gateway",
    )

    @classmethod
    def infer_llm_is_local(cls, llm_hint: str | None) -> bool:
        """True if the LLM occupies a local GPU (so we must reserve one).

        ``llm_hint`` is a base_url or raw text (NOT a config path -- see
        ``placement_from_config`` for that). Remote API endpoints (incl.
        gateway) return False so GPU0 stays free for cuDF.
        """
        s = (llm_hint or "").lower()
        if not s:
            return False  # default: assume API -- safer on a shared box
        if any(t in s for t in cls._API_LLM_TOKENS):
            return False
        return any(t in s for t in cls._LOCAL_LLM_TOKENS)

    @classmethod
    def placement_from_config(cls, cfg: dict) -> tuple[bool, int]:
        """Return ``(llm_is_local, llm_gpu)`` from a PARSED LLM config dict.

        This inspects the config CONTENT (provider / base_url), not a file
        name -- the earlier version matched the config path string, so a
        normally-named ``configs/llm_local.yaml`` with
        ``provider: huggingface_local`` inside was misread as a remote API
        and never got a GPU reserved. An explicit ``llm_is_local`` /
        ``llm_gpu`` in the config always wins (needed for same-box vLLM
        served on a cluster IP, which no URL heuristic can detect).
        """
        llm = cfg.get("llm", cfg) if isinstance(cfg, dict) else {}
        gpu = int(llm.get("llm_gpu", 0) or 0)
        if "llm_is_local" in llm:                       # explicit override
            return bool(llm["llm_is_local"]), gpu
        provider = str(llm.get("provider", "api")).lower()
        if provider == "huggingface_local":
            return True, gpu
        # API provider: decide by endpoint (localhost/vllm => local server).
        return cls.infer_llm_is_local(str(llm.get("base_url", ""))), gpu

    def detect(self, data_path: str | Path | None = None,
               data_size_gb: float | None = None,
               *, llm_is_local: bool = False,
               llm_gpu: int = 0) -> ComputeProfile:
        """Probe the environment and return a ``ComputeProfile``."""
        profile = ComputeProfile(llm_is_local=llm_is_local, llm_gpu=llm_gpu)

        # 1. Detect GPUs
        self._detect_gpus(profile)

        # 2. Estimate data size
        if data_size_gb is not None:
            profile.data_size_gb = data_size_gb
        elif data_path:
            profile.data_size_gb = self._estimate_size(Path(data_path))

        # 3. Route
        self._route(profile)
        return profile

    # -- GPU detection ------------------------------------------------------
    #
    # Detection used to be `import torch` and nothing else: if torch was
    # missing, gpu_count was set to 0 and the router reported "No GPU
    # detected -- CPU pandas" on a box with working GPUs and cuDF. torch is
    # NOT a dependency of this project (it appears only in the [gpu]/[local]
    # extras), so a RAPIDS install -- cuDF present, torch absent -- silently
    # never got the GPU. A CPU-only torch build (device_count()==0) had the
    # same effect.
    #
    # Now: try each source in turn and take the first that reports a GPU.
    # cupy is the meaningful middle probe because cuDF depends on it -- if
    # cuDF can run at all, cupy is importable. nvidia-smi is the last resort
    # and needs no Python GPU package at all, only the driver.

    @staticmethod
    def _gpus_via_torch() -> tuple[int, list[str], list[float], list[float]]:
        import torch
        n = torch.cuda.device_count()
        names: list[str] = []
        mems: list[float] = []
        free_mems: list[float] = []
        for i in range(n):
            names.append(torch.cuda.get_device_name(i))
            props = torch.cuda.get_device_properties(i)
            mems.append(round(props.total_memory / (1024**3), 1))
            free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
            free_mems.append(round(free_bytes / (1024**3), 1))
        return n, names, mems, free_mems

    @staticmethod
    def _gpus_via_cupy() -> tuple[int, list[str], list[float], list[float]]:
        import cupy
        n = int(cupy.cuda.runtime.getDeviceCount())
        names: list[str] = []
        mems: list[float] = []
        free_mems: list[float] = []
        for i in range(n):
            props = cupy.cuda.runtime.getDeviceProperties(i)
            raw = props.get("name", b"")
            names.append(raw.decode() if isinstance(raw, bytes) else str(raw))
            mems.append(round(int(props.get("totalGlobalMem", 0)) / (1024**3), 1))
            with cupy.cuda.Device(i):
                free_bytes, _total_bytes = cupy.cuda.runtime.memGetInfo()
            free_mems.append(round(free_bytes / (1024**3), 1))
        return n, names, mems, free_mems

    @staticmethod
    def _gpus_via_nvidia_smi() -> tuple[int, list[str], list[float], list[float]]:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0, [], [], []
        names: list[str] = []
        mems: list[float] = []
        free_mems: list[float] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, total_mib, free_mib = parts[0], parts[1], parts[2]
            names.append(name)
            try:
                mems.append(round(float(total_mib) / 1024, 1))
            except ValueError:
                mems.append(80.0)
            try:
                free_mems.append(round(float(free_mib) / 1024, 1))
            except ValueError:
                free_mems.append(mems[-1])  # unknown -- assume worst case (all free)
        return len(names), names, mems, free_mems

    def _detect_gpus(self, p: ComputeProfile) -> None:
        """Populate gpu_count/names/memory from the first source that sees a GPU."""
        for source, probe in (("torch", self._gpus_via_torch),
                              ("cupy", self._gpus_via_cupy),
                              ("nvidia-smi", self._gpus_via_nvidia_smi)):
            try:
                n, names, mems, free_mems = probe()
            except ImportError:
                log.debug("GPU probe %s: not installed", source)
                continue
            except Exception as e:  # noqa: BLE001 -- a broken probe must not
                log.debug("GPU probe %s failed: %s", source, e)  # mask the next
                continue
            if n > 0:
                p.gpu_count = n
                p.gpu_names = names
                p.gpu_memory_gb = mems or [self._nvidia_smi_mem(i) for i in range(n)]
                p.gpu_memory_free_gb = free_mems
                p.gpu_detected_via = source
                log.info("Detected %d GPU(s) via %s: %s", n, source, ", ".join(names))
                return
            log.debug("GPU probe %s: reported 0 GPUs", source)
        p.gpu_count = 0
        p.gpu_detected_via = ""
        log.info("No GPU found by any probe (torch/cupy/nvidia-smi) -- CPU only")

    def _route(self, p: ComputeProfile):
        """Apply the decision tree.

        Key split: a local LLM reserves ``llm_gpu`` (nothing else may use
        it); a remote API LLM reserves nothing, so GPU0 is free for cuDF.
        """
        if p.gpu_count == 0:
            p.backend_choice = "pandas"
            p.llm_device = ""
            p.data_device = "cpu"
            p.compute_gpus = []
            p.reason = ("No GPU detected by torch, cupy or nvidia-smi -- CPU "
                        "pandas. (If this box HAS a GPU: check `nvidia-smi` "
                        "runs, and that cudf/cupy import cleanly.)")
            return

        all_gpus = list(range(p.gpu_count))

        # Which GPUs may the data backend use? A local LLM removes its GPU
        # from the pool; an API LLM leaves every GPU available.
        if p.llm_is_local and p.llm_gpu in all_gpus:
            p.llm_device = f"cuda:{p.llm_gpu}"
            p.compute_gpus = [g for g in all_gpus if g != p.llm_gpu]
        else:
            p.llm_device = ""  # remote API -- no GPU reserved
            p.compute_gpus = all_gpus

        smallest_gpu = min(p.gpu_memory_gb) if p.gpu_memory_gb else 0

        # No GPU left for data (e.g. 1 GPU fully taken by a local LLM)?
        if not p.compute_gpus:
            p.backend_choice = "pandas"
            p.data_device = "cpu"
            p.reason = (f"{p.gpu_count} GPU(s), local LLM on cuda:{p.llm_gpu} "
                        "-- CPU pandas for data")
            return

        # Capacity ceiling: 70% of TOTAL card memory, same as before -- but
        # also bounded by what's actually FREE right now. Confirmed on a
        # live shared GPU (an unrelated vLLM process already resident):
        # total was 49GB (34GB "allowed" by the 70% rule) while free was
        # only 5GB. Total-only sizing would have told cuDF to go ahead on a
        # dataset the card cannot actually hold, OOMing against -- or
        # contending with -- whatever else is running there. Free-memory
        # data is only available for GPUs the compute_gpus pool can use;
        # missing entries (older nvidia-smi, a probe that didn't report it)
        # fall back to the total-only ceiling rather than blocking cuDF.
        free_for_compute = [
            p.gpu_memory_free_gb[g] for g in p.compute_gpus
            if g < len(p.gpu_memory_free_gb)
        ]
        smallest_free = min(free_for_compute) if free_for_compute else smallest_gpu
        capacity_ceiling = min(0.7 * smallest_gpu, 0.7 * smallest_free)

        # Data too big for ONE compute GPU, but the box has several? Use
        # dask-cudf across the pool rather than dropping to CPU.
        #
        # Real production run (2026-08-06): a 3x H100 box (240GB aggregate)
        # ran a 74.5M-row job entirely on CPU pandas, because this ceiling
        # only ever considered the SMALLEST SINGLE card. 144 of 369 cells
        # were pandas, 16 were cuDF. That is the whole "3 H100s and it was
        # still slow" complaint. Only taken when dask_cudf actually imports
        # -- otherwise we keep the honest CPU fallback rather than picking a
        # backend that would fail at runtime.
        if p.data_size_gb > capacity_ceiling and len(p.compute_gpus) > 1:
            pool_free = sum(
                p.gpu_memory_free_gb[g] for g in p.compute_gpus
                if g < len(p.gpu_memory_free_gb)
            ) or sum(p.gpu_memory_gb[g] for g in p.compute_gpus
                     if g < len(p.gpu_memory_gb))
            if p.data_size_gb <= 0.7 * pool_free and _dask_cudf_available():
                p.backend_choice = "dask_cudf"
                p.data_device = ",".join(f"cuda:{g}" for g in p.compute_gpus)
                p.reason = (
                    f"Data ({p.data_size_gb:.1f}GB) exceeds one GPU's 70% "
                    f"ceiling ({capacity_ceiling:.0f}GB) but fits the "
                    f"{len(p.compute_gpus)}-GPU pool "
                    f"({pool_free:.0f}GB free) -- dask-cudf across "
                    f"{p.data_device}"
                    + (f", LLM on {p.llm_device}" if p.llm_device else ""))
                return

        # Data too big for a single compute GPU? Single-GPU cuDF would OOM.
        if p.data_size_gb > capacity_ceiling:
            p.backend_choice = "pandas"
            p.data_device = "cpu"
            p.reason = (f"Data ({p.data_size_gb:.1f}GB) > 70% of "
                        f"{'free' if smallest_free < smallest_gpu else 'total'} GPU mem "
                        f"(total {smallest_gpu:.0f}GB, free {smallest_free:.0f}GB) "
                        "-- CPU pandas"
                        + (f", LLM on {p.llm_device}" if p.llm_device else ""))
            return

        # Data fits: single-GPU cuDF on the first compute GPU.
        data_gpu = p.compute_gpus[0]
        p.backend_choice = "cudf"
        p.data_device = f"cuda:{data_gpu}"
        if p.llm_device:
            p.reason = (f"{p.gpu_count} GPUs -- local LLM on {p.llm_device}, "
                        f"cuDF on {p.data_device}"
                        + (f", free: {p.compute_gpus[1:]}"
                           if len(p.compute_gpus) > 1 else ""))
        else:
            p.reason = (f"{p.gpu_count} GPUs -- API LLM (no GPU reserved), "
                        f"cuDF on {p.data_device}"
                        + (f", free: {p.compute_gpus[1:]}"
                           if len(p.compute_gpus) > 1 else ""))

    @staticmethod
    def _estimate_size(path: Path) -> float:
        """Estimate dataset size in GB.

        Files: on-disk size, with a 4x multiplier for parquet (compressed
        ~3-5x smaller on disk than in memory).

        Directories (partitioned parquet/csv): sum of the recognised data
        files inside, same per-file multiplier. A directory's own
        ``stat().st_size`` is meaningless (it's not the sum of its
        contents) -- using it here previously made this method report a
        few KB for an 8-crore-row partitioned dataset, which made
        ``_route()`` fail OPEN: it picked cuDF for a directory of any real
        size because the size looked tiny. An unreadable/empty/unrecognised
        directory now returns a large ledgerhound instead of 0 -- this number
        gates whether cuDF is ever chosen, and failing open on an unknown
        size means picking cuDF for a dataset of unknown (possibly huge)
        size.
        """
        if not path.exists():
            return 0.0
        if path.is_dir():
            total_bytes = 0.0
            found = False
            for f in path.rglob("*"):
                if not f.is_file():
                    continue
                suffix = f.suffix.lower()
                if suffix not in (".parquet", ".pq", ".csv", ".tsv"):
                    continue
                found = True
                total_bytes += ComputeRouter._file_bytes_in_memory(f, suffix)
            if not found:
                return 1e6  # unrecognised contents -- force the pandas fallback
            return round(total_bytes / (1024**3), 2)
        suffix = path.suffix.lower()
        return round(
            ComputeRouter._file_bytes_in_memory(path, suffix) / (1024**3), 2)

    @staticmethod
    def _file_bytes_in_memory(f: Path, suffix: str) -> float:
        """In-memory byte estimate for one data file.

        For parquet, read the footer's own uncompressed column-chunk sizes
        instead of applying a flat 4x to the on-disk size. The 4x guess erred
        in the direction that hurts: it INFLATES the estimate, pushing
        ``_route()`` past its 70% ceiling and dropping a dataset that would
        have fitted the card onto CPU pandas.

        Measured on the three parquet files in this repo, as
        estimate/actual-pandas-``memory_usage(deep=True)`` ratios::

            file                on-disk   old 4x    this    actual
            raw_txn_sample        3.8MB    0.77x   0.43x    19.7MB
            upi_summary           5.2MB    2.23x   0.93x     9.4MB
            raw_txn_sample(p3)    2.1MB    0.83x   0.48x     9.9MB

        Read that honestly: neither estimator is reliable. The old 4x
        overshot 2.2x on one file and undershot on the other two; this one
        undershoots pandas residency by up to 2.3x on string-heavy data. No
        single multiplier fits both string-heavy and numeric layouts, and
        these tiny synthetic files are nothing like an 8-crore UPI extract,
        so the calibration is genuinely unresolved -- see V3-044 "Not done".

        The factor is nonetheless biased LOW deliberately, for two reasons.
        The observed production failure is over-estimation (a 3xH100 box that
        never left CPU), so a cliff that fires early is the costly error. And
        the failure mode on the other side changed: RMM managed memory
        (V3-044) turns over-subscription into host paging rather than an OOM,
        and ``_preload_data`` already retries a failed GPU load on pandas.

        One caveat on the yardstick: the target is cuDF, not pandas. cuDF
        stores strings as offsets into a single char buffer and should sit
        much closer to the footer number than pandas object columns do -- but
        that is unmeasured, because this box has no GPU. Run
        ``scripts/diagnose_gpu.py`` on the real box to calibrate.

        The footer is metadata only -- no column data is read, so this stays
        cheap on an 8-crore-row file. Falls back to the old multiplier if the
        footer cannot be read (corrupt file, no pyarrow).
        """
        try:
            on_disk = float(f.stat().st_size)
        except OSError:
            return 0.0
        if suffix not in (".parquet", ".pq"):
            return on_disk  # CSV/TSV: text in, roughly text-sized frame out
        try:
            import pyarrow.parquet as pq
            md = pq.ParquetFile(f).metadata
            uncompressed = 0
            for rg in range(md.num_row_groups):
                group = md.row_group(rg)
                for col in range(group.num_columns):
                    uncompressed += group.column(col).total_uncompressed_size
            if uncompressed > 0:
                # 1.5x: parquet's uncompressed encoding is still denser than a
                # live frame (offsets/dictionaries become real objects). See
                # the measurements in the docstring.
                return float(uncompressed) * 1.5
        except Exception:  # noqa: BLE001 -- metadata is best-effort
            log.debug("parquet footer unreadable for %s -- using 4x on-disk", f)
        return on_disk * 4.0

    @staticmethod
    def _nvidia_smi_mem(gpu_idx: int = 0) -> float:
        """Fallback: read GPU memory from nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits", f"--id={gpu_idx}"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                mb = float(result.stdout.strip())
                return round(mb / 1024, 1)
        except Exception:
            pass
        return 80.0  # safe default for H100


def _dask_cudf_available() -> bool:
    """True if dask_cudf imports and the operator has not opted out.

    **Default ON** (operator decision, 2026-08-13). It was opt-in until
    V3-046 because `dask_cudf.read_parquet()` returns a LAZY frame, which
    silently changes the semantics of every cell the analyst writes:
    `df.groupby(...).mean()` yields an unevaluated graph, `print()` shows a
    repr instead of numbers, and pandas-only code paths (feature_eval's
    qcut/histogram) receive something they cannot consume.

    Two things make default-on defensible now:

    * V3-044 materialises the frame with `.compute()` straight after the
      parallel multi-GPU read, so what the analyst gets back is an EAGER
      cuDF frame and the semantics are unchanged. The lazy frame only
      survives if that materialisation itself fails, and that path prints a
      loud warning naming this switch.
    * The gate degrades safely where the packages are missing: no
      `dask_cudf` wheel means this returns False and nothing changes. On a
      box that never installed RAPIDS, flipping the default is a no-op.

    It remains UNVALIDATED ON HARDWARE. If a run produces frames that print
    as reprs, helpers that reject the dataframe, or a cluster that will not
    start, turn it off with ``gpu.enable_dask_cudf: false`` in
    main_config.yaml (or ``export LEDGERHOUND_ENABLE_DASK_CUDF=0``) and the
    router falls back to single-GPU cuDF, which is the tested path.
    """
    from ledgerhound.config import get_bool
    if not get_bool("gpu", "enable_dask_cudf", env="LEDGERHOUND_ENABLE_DASK_CUDF",
                    default=True):
        return False
    try:
        import dask_cudf  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# Appended to every dask-cudf failure message. One place, so the escape
# hatch is never described three slightly different ways.
DASK_CUDF_OFF_HINT = (
    "To disable multi-GPU and fall back to single-GPU cuDF (the tested "
    "path), set LEDGERHOUND_ENABLE_DASK_CUDF=0 and restart the session.")


def classify_gpu_error(text: str, rmm_status: str = "") -> tuple[str, str]:
    """Map a GPU failure's text to ``(error_class, remedy)`` for the RCA note.

    The point is to answer "why did this drop to CPU?" at the moment it
    happens, rather than leaving one yellow line for someone to find in
    scrollback three hours into a run.

    ``rmm_status`` is the Phase-0a RMM line, and it matters most for OOM.
    Since V3-044 initialises RMM with ``managed_memory=True``, a genuine
    out-of-memory should NOT happen -- the allocation ought to page to host
    instead. So an OOM is usually evidence that RMM never initialised, and
    reporting "data too big for the card" would send the operator to shrink
    a dataset that would have fitted fine. Check the allocator first.
    """
    t = (text or "").lower()

    oom_markers = ("out of memory", "outofmemory", "memoryerror",
                   "bad_alloc", "cudaerrormemoryallocation",
                   "rmm::bad_alloc", "cuda error: out of memory")
    if any(m in t for m in oom_markers):
        if rmm_status and not rmm_status.lower().startswith("pool"):
            return ("gpu_oom_without_managed_memory",
                    "Out of GPU memory AND RMM managed memory is not active "
                    f"({rmm_status}). With managed memory this allocation "
                    "would have paged to host instead of failing, so the "
                    "allocator is the root cause, not the dataset size. "
                    "Install/repair `rmm` (it ships with the RAPIDS wheels) "
                    "and re-run before reducing the data.")
        return ("gpu_oom",
                "Out of GPU memory even with managed memory active. The "
                "working set genuinely exceeds device+host capacity. Reduce "
                "the frame (select fewer columns, filter earlier), or spread "
                "it across cards -- multi-GPU is on by default and needs the "
                "dask-cudf wheels.")

    if "no module named" in t or "modulenotfounderror" in t:
        return ("gpu_stack_not_installed",
                "The GPU dataframe stack is not importable. Install the "
                "wheels matching this box's CUDA major version, e.g. "
                "`pip install --extra-index-url=https://pypi.nvidia.com "
                "\".[gpu-cu13,multigpu-cu13]\"` on CUDA 13.")

    if "undefined symbol" in t or "libcu" in t or "version `libcu" in t:
        return ("cuda_wheel_mismatch",
                "A CUDA library failed to load -- almost always a RAPIDS "
                "wheel built for a different CUDA major than the driver "
                "provides. Check `nvidia-smi`'s CUDA version against the "
                "-cu<N> suffix of the installed cudf, and reinstall to match.")

    if "no cuda-capable device" in t or "driver version is insufficient" in t:
        return ("cuda_driver_problem",
                "The CUDA driver rejected the device. Confirm `nvidia-smi` "
                "runs and that this process can see the GPUs "
                "(CUDA_VISIBLE_DEVICES).")

    if "timeout" in t or "timed out" in t:
        return ("gpu_init_timeout",
                "GPU initialisation exceeded its timeout. A first CUDA "
                "context on a large box can be slow; retrying often "
                "succeeds. If it persists, check for another process "
                "holding the cards.")

    return ("gpu_failed_unclassified",
            "GPU path failed for a reason this classifier does not "
            "recognise. The raw error is recorded verbatim below -- please "
            "attach it when reporting, so a case can be added.")


def _pin_cudf_device(index: int | None) -> None:
    """Best-effort: point cuDF/cupy at ``index`` before it allocates.

    Defensive -- any failure just leaves the default device in place, so
    this can never make things worse than before.
    """
    if index is None:
        return
    try:
        import cupy
        cupy.cuda.Device(index).use()
        log.info("Pinned cuDF/cupy to GPU %d", index)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not pin cuDF to GPU %d: %s", index, e)


def get_backend(data_path: str | Path, compute: ComputeProfile | None = None):
    """Return the right backend instance based on compute profile.

    ``data_path`` may also be an already-materialised DataFrame.
    """
    # An in-memory frame has nothing left to read, so the cuDF branch below
    # cannot help it (and `str(df)` would send the whole rendered table to
    # cudf.read_csv, which then fails into the generic fallback with a scary
    # warning). Short-circuit explicitly instead of relying on that crash.
    if hasattr(data_path, "columns") and hasattr(data_path, "dtypes"):
        from ledgerhound.backends.pandas_backend import PandasBackend
        return PandasBackend(data_path)

    if compute is None:
        compute = ComputeRouter().detect(data_path)

    if compute.backend_choice == "cudf":
        try:
            # Pin cuDF to the router-chosen GPU BEFORE any allocation, so it
            # never lands on the GPU a local LLM has reserved (previously it
            # always defaulted to device 0 and collided with the model).
            _pin_cudf_device(compute.data_device_index())
            import cudf
            # Honest about what this does: the GPU accelerates the READ, then
            # the frame is copied to host because PandasBackend's profiling
            # surface (scipy.stats calls, .isna().sum(), numpy interop) is
            # pandas-shaped and would break on a cuDF frame. So every
            # operation AFTER this line is CPU.
            #
            # This is the profiler/`core.py` path, not the shell -- the shell
            # keeps its frame on the GPU inside the kernel. A genuinely
            # GPU-resident backend means reworking PandasBackend's internals
            # and is unbuilt scope (see V3-044 "Not done").
            log.info("cuDF-accelerated read on %s, then host copy for "
                     "PandasBackend (CPU from here on)", compute.data_device)
            df = cudf.read_parquet(str(data_path)) if str(data_path).endswith(".parquet") \
                else cudf.read_csv(str(data_path))
            from ledgerhound.backends.pandas_backend import PandasBackend
            return PandasBackend(df.to_pandas())
        except ImportError:
            log.warning("cuDF not available, falling back to pandas")
        except Exception as e:
            log.warning("cuDF failed: %s, falling back to pandas", e)

    from ledgerhound.backends.pandas_backend import PandasBackend
    return PandasBackend(data_path)
