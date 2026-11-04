"""
ledgerhound.runtime.isolation
=============================

How the analysis kernel is sandboxed (ADR 0001, docs/adr/0001-kernel-isolation.md).

The guarantee is narrow on purpose: **the analysis kernel cannot open network
connections.** It is not "data stays home" -- the host agent reads every tool
result and can send it anywhere. Nothing in this module contains the agent.

An `IsolationPolicy` turns into a `LaunchPlan`: a command prefix that wraps the
kernel, the environment it starts with, a memory limit for the RSS poller when
no cgroup is available, and an `IsolationReport` that names only what is
actually enforced. The report is what the tools, the event log and the
certificate show.

Deliberately absent: RLIMIT_AS (counts reserved address space; cuDF/CUDA and
Arrow reserve far more than they touch and die at startup) and RLIMIT_CPU
(multi-hour runs are legitimate). Runaway cells are caught by a wall-clock
per-cell timeout in KernelExecutor instead.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODES = ("off", "best", "require")

# Environment the kernel may inherit. Anything not listed is dropped, and any
# listed name that looks like a credential is dropped anyway (_SECRETISH).
_ENV_EXACT = {
    "PATH", "LANG", "LANGUAGE", "TZ", "TERM", "USER", "LOGNAME", "SHELL",
    "PYTHONPATH", "PYTHONHASHSEED", "VIRTUAL_ENV", "CONDA_PREFIX",
    "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    # Windows: Python and its DLL loader need these to start at all.
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "USERPROFILE",
}
_ENV_PREFIXES = ("LC_", "LEDGERHOUND_", "CUDF_", "RAPIDS_", "NVIDIA_", "RMM_",
                 "DASK_", "ARROW_")
_BUS_VARS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
_SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|SESSION|COOKIE",
                        re.IGNORECASE)


class IsolationError(RuntimeError):
    """The policy requires a level this machine cannot provide."""


@dataclass(frozen=True)
class IsolationReport:
    """What is actually enforced on the running kernel. Never aspirational."""

    level: str                 # "sandboxed" | "partial" | "off"
    network: str               # "blocked" | "open"
    writes: str                # "confined" | "unrestricted"
    env: str                   # "allow-listed" | "inherited"
    memory: str                # "cgroup 12.0 GiB" | "rss-poll 12.0 GiB" | "none"
    cell_timeout_s: int
    missing: tuple[str, ...] = ()   # why the level is not "sandboxed"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["missing"] = list(self.missing)
        return d

    def summary(self) -> str:
        s = (f"isolation: {self.level} (network {self.network}, writes "
             f"{self.writes}, env {self.env}, memory cap {self.memory}, "
             f"cell timeout {self.cell_timeout_s}s)")
        if self.missing:
            s += " -- not enforced: " + "; ".join(self.missing)
        return s


@dataclass
class LaunchPlan:
    prefix: list[str]          # argv placed before the kernel command
    env: dict[str, str] | None  # None = inherit os.environ (mode "off")
    transport: str             # "ipc" inside a network namespace, else "tcp"
    rss_limit_bytes: int | None  # poll-and-kill threshold; None = cgroup or none
    report: IsolationReport


@dataclass
class IsolationPolicy:
    mode: str = "best"
    memory_max_bytes: int | None = None     # None = 80% of physical memory
    cell_timeout_s: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"isolation mode must be one of {MODES}, got {self.mode!r}")

    @classmethod
    def from_config(cls, default_mode: str = "off", require: bool = False) -> "IsolationPolicy":
        """main_config `kernel:` block, env fallbacks, then the caller's default.
        `require=True` (the --require-isolation flag) always wins."""
        from ledgerhound.config import get_int, get_str
        mode = get_str("kernel", "isolation", env="LEDGERHOUND_ISOLATION",
                       default=default_mode).strip().lower() or default_mode
        if require:
            mode = "require"
        gb = get_int("kernel", "memory_max_gb", env="LEDGERHOUND_KERNEL_MEMORY_GB", default=0)
        timeout = get_int("kernel", "cell_timeout_seconds",
                          env="LEDGERHOUND_CELL_TIMEOUT", default=1800)
        return cls(mode=mode, memory_max_bytes=gb * 2**30 if gb > 0 else None,
                   cell_timeout_s=max(30, timeout))

    # ------------------------------------------------------------------ #

    def plan(self, writable: list[Path], home: Path,
             readable: list[Path] | None = None) -> LaunchPlan:
        """Build the launch for a kernel that may write only to `writable`
        (plus a private /tmp) and whose HOME is `home` (must be in writable).
        `readable` (the dataset) is re-exposed read-only even where the
        sandbox hides the host path, e.g. a dataset under /tmp."""
        if self.mode == "off":
            return LaunchPlan(prefix=[], env=None, transport="tcp", rss_limit_bytes=None,
                              report=IsolationReport(
                                  level="off", network="open", writes="unrestricted",
                                  env="inherited", memory="none",
                                  cell_timeout_s=self.cell_timeout_s,
                                  missing=("isolation turned off",)))

        missing: list[str] = []
        env = kernel_env(home, self.extra_env)
        limit = self.memory_max_bytes or default_memory_limit()

        prefix: list[str] = []
        transport = "tcp"
        network, writes = "open", "unrestricted"
        if sys.platform.startswith("linux") and bwrap_works():
            prefix += bwrap_args(writable, readable or [])
            transport, network, writes = "ipc", "blocked", "confined"
        elif sys.platform.startswith("linux"):
            missing.append("bwrap unavailable or blocked (install bubblewrap; "
                           "check unprivileged user namespaces)")
        else:
            missing.append(f"no network or write sandbox on {sys.platform}")

        rss_limit: int | None = None
        if limit and cgroup_scope_works():
            # systemd-run execs the rest of the argv inside a transient scope,
            # so the kernel keeps the PID jupyter_client launched. It needs
            # the session-bus variables to reach the user manager; `env -u`
            # takes them away again before anything else starts.
            for k in _BUS_VARS:
                if k in os.environ:
                    env[k] = os.environ[k]
            unset = [a for k in _BUS_VARS for a in ("-u", k)]
            prefix = (["systemd-run", "--user", "--scope", "--quiet",
                       "-p", f"MemoryMax={limit}", "-p", "MemorySwapMax=0", "--",
                       "env", *unset] + prefix)
            memory = f"cgroup {_gib(limit)}"
        elif limit and _psutil_available():
            rss_limit = limit
            memory = f"rss-poll {_gib(limit)}"
        else:
            memory = "none"
            missing.append("no memory cap (no cgroup scope, psutil not installed)")

        level = "sandboxed" if not missing else "partial"
        report = IsolationReport(level=level, network=network, writes=writes,
                                 env="allow-listed", memory=memory,
                                 cell_timeout_s=self.cell_timeout_s,
                                 missing=tuple(missing))
        if self.mode == "require" and level != "sandboxed":
            raise IsolationError(
                "kernel isolation required but not available here -- "
                + "; ".join(missing)
                + ". Run on Linux/WSL with bubblewrap, or drop --require-isolation.")
        return LaunchPlan(prefix=prefix, env=env, transport=transport,
                          rss_limit_bytes=rss_limit, report=report)


# ---------------------------------------------------------------------- #

def kernel_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The allow-listed environment: no API keys, cloud credentials or tokens."""
    env = {}
    for k, v in os.environ.items():
        allowed = k.upper() in _ENV_EXACT or k.upper().startswith(_ENV_PREFIXES)
        if allowed and not _SECRETISH.search(k):
            env[k] = v
    env["HOME"] = str(home)
    env["MPLCONFIGDIR"] = str(Path(home) / ".matplotlib")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update(extra)
    return env


def bwrap_args(writable: list[Path], readable: list[Path] = ()) -> list[str]:
    """Read-only root, private /tmp, /run and /dev/shm, read-write only
    `writable`, `readable` re-exposed read-only, and a fresh network
    namespace (loopback only)."""
    args = ["bwrap", "--ro-bind", "/", "/",
            "--dev-bind", "/dev", "/dev",        # keeps GPU device nodes
            "--tmpfs", "/dev/shm",               # but not the host's shared memory
            "--proc", "/proc",
            # Unix sockets ignore the network namespace and a read-only bind
            # does not stop connect(). /run holds the user's D-Bus (where
            # `systemd-run --user` would start a process OUTSIDE the
            # sandbox), the WSL interop socket (which runs Windows .exe files
            # with full network), docker.sock and the system bus. Hide it.
            "--tmpfs", "/run",
            "--tmpfs", "/tmp"]                   # before the binds, or it hides them
    seen = set()
    for p in writable:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            args += ["--bind", rp, rp]
    for p in readable:
        rp = str(Path(p).resolve())
        if rp not in seen and Path(rp).exists():
            seen.add(rp)
            args += ["--ro-bind", rp, rp]
    args += ["--unshare-net", "--die-with-parent", "--"]
    return args


@functools.lru_cache(maxsize=1)
def bwrap_works() -> bool:
    if not shutil.which("bwrap"):
        return False
    try:
        r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
                            "--proc", "/proc", "--unshare-net", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def cgroup_scope_works() -> bool:
    if not shutil.which("systemd-run"):
        return False
    try:
        r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet",
                            "-p", "MemoryMax=64M", "--", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def default_memory_limit() -> int | None:
    """80% of physical memory: room for the server and the OS, while a
    runaway cell is still stopped before the machine starts swapping."""
    try:
        import psutil
        return int(psutil.virtual_memory().total * 0.8)
    except Exception:  # noqa: BLE001
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.8)
        except Exception:  # noqa: BLE001
            return None


def tree_rss(pid: int) -> int:
    """Resident memory of a process and all its descendants, in bytes."""
    import psutil
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"
