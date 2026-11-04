"""
ledgerhound.auto.sandbox
=====================

ScratchExecutor -- like CodeExecutor, but code runs inside a dated scratch
directory so it can persist artefacts (charts, CSVs, JSON) to disk.

Safety model
------------
* ``scratch_dir`` is the ONLY writable path exposed to generated code.
* The repo root is read-only: any write attempt to a path outside
  ``scratch_dir`` raises ``PermissionError`` at validation time.
* Dangerous shell/network modules are still blocked.
* ``df`` (caller's DataFrame) is passed as a copy -- originals are immutable.
"""

from __future__ import annotations

import io, os, textwrap, time, traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(v):
    """Coerce a metric value into a JSON-safe scalar/list."""
    try:
        import numpy as _np
        if isinstance(v, (_np.integer,)):
            return int(v)
        if isinstance(v, (_np.floating,)):
            return float(v)
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v][:50]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in list(v.items())[:50]}
    return str(v)[:200]


@dataclass
class ExecResult:
    success: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0.0
    artefacts: list[str] = field(default_factory=list)  # relative paths written
    metrics: dict = field(default_factory=dict)         # computed facts (name->value)
    df_fingerprint: dict = field(default_factory=dict)  # {n_rows, columns} -- V3-017


class ScratchExecutor:
    """Execute agent-generated code with full file-write access inside *scratch_dir*.

    The sandbox exposes:
      * ``df``          -- copy of caller's DataFrame (if provided)
      * ``scratch``     -- ``pathlib.Path`` pointing at the run's scratch dir
      * ``pd``, ``np``, ``matplotlib``, ``seaborn``, ``json``, ``re``, ``math``
    """

    _BLOCKED = [
        ("import subprocess", "subprocess not allowed"),
        ("import shutil",     "shutil not allowed"),
        ("__import__",        "dynamic imports not allowed"),
        ("exec(",             "nested exec not allowed"),
        ("eval(",             "eval not allowed"),
        ("requests.",         "network access not allowed"),
        ("urllib",            "network access not allowed"),
        ("socket",            "network access not allowed"),
    ]
    _MAX_OUTPUT = 8_000

    def __init__(self, scratch_dir: Path, df: pd.DataFrame | None = None, timeout: int = 120):
        self._scratch = scratch_dir
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._df = df
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def validate(self, code: str) -> tuple[bool, str]:
        for pattern, reason in self._BLOCKED:
            if pattern in code:
                return False, reason
        return True, ""

    def execute(self, code: str) -> ExecResult:
        ok, reason = self.validate(code)
        if not ok:
            return ExecResult(success=False, output="", error=f"Blocked: {reason}")

        # Build namespace
        import builtins, json, math, re as re_mod
        _SAFE = (
            "abs","all","any","bin","bool","bytes","chr","dict","divmod",
            "enumerate","filter","float","format","frozenset","hasattr","hash",
            "hex","int","isinstance","issubclass","iter","len","list","map",
            "max","min","next","oct","ord","pow","print","range","repr",
            "reversed","round","set","slice","sorted","str","sum","tuple",
            "type","zip","open",   # open allowed but write-path guarded below
        )
        safe_builtins = {n: getattr(builtins, n) for n in _SAFE if hasattr(builtins, n)}
        # Allow normal `import pandas / numpy / ...` statements inside exec'd code.
        # The string-blocklist above still catches explicit __import__("subprocess")
        # calls in generated source before we ever reach exec().
        safe_builtins["__import__"] = builtins.__import__

        # Wrap open() to enforce scratch-only writes
        real_open = open
        scratch = self._scratch

        def _guarded_open(file, mode="r", **kwargs):
            p = Path(file)
            if not p.is_absolute():
                p = scratch / p
            if any(m in mode for m in ("w", "a", "x")):
                try:
                    p.resolve().relative_to(scratch.resolve())
                except ValueError:
                    raise PermissionError(
                        f"Write access denied outside scratch dir: {file}\n"
                        f"Use Path(scratch / 'filename') to save artefacts."
                    )
            return real_open(p if not Path(file).is_absolute() else file, mode, **kwargs)

        safe_builtins["open"] = _guarded_open

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            plt = None
            sns = None

        # IMPORTANT: use a SINGLE namespace dict for exec (passed as globals).
        # If globals and locals are different dicts, any helper FUNCTION the
        # generated code defines captures the globals dict as its __globals__,
        # so names like `re`, `pd`, `np` (which would live only in locals)
        # become NameError inside that function. One shared dict fixes it.
        local_ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "scratch": self._scratch,
            "pd": pd,
            "np": np,
            "json": json,
            "math": math,
            "re": re_mod,
        }
        if plt:
            local_ns["plt"] = plt
        if sns:
            local_ns["sns"] = sns
        if self._df is not None:
            # Share by reference (no copy) -- avoids duplicating a 20M-row frame
            # on every step. The code prompt forbids inplace= mutations, and the
            # files-on-disk contract means steps don't rely on mutating df anyway.
            local_ns["df"] = self._df

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        files_before = set(self._scratch.rglob("*"))
        t0 = time.perf_counter()
        try:
            import contextlib
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                # Single namespace: functions defined in `code` capture this same
                # dict as __globals__, so pre-injected libs resolve everywhere.
                exec(code, local_ns)

            elapsed = (time.perf_counter() - t0) * 1000
            output = stdout_buf.getvalue()

            res = local_ns.get("result")
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    rows = len(res)
                    output += f"\n[DataFrame {rows} rows x {res.shape[1]} cols]\n"
                    output += res.head(50).to_string(index=False)
                elif isinstance(res, pd.Series):
                    output += res.head(50).to_string()
                else:
                    output += str(res)

            files_after = set(self._scratch.rglob("*"))
            new_files = [
                str(f.relative_to(self._scratch))
                for f in (files_after - files_before)
                if f.is_file()
            ]

            # Structured grounding: capture a `metrics` dict of computed facts
            # so findings can be derived from real values, not re-read prose.
            metrics = {}
            m = local_ns.get("metrics")
            if isinstance(m, dict):
                metrics = {str(k): _jsonable(v) for k, v in m.items()}

            return ExecResult(
                success=True,
                output=(output.strip() or "[Executed -- no output]")[: self._MAX_OUTPUT],
                execution_time_ms=round(elapsed, 1),
                artefacts=new_files,
                metrics=metrics,
            )

        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc().strip()
            # Keep more context for the LLM fix loop. Strip only the
            # internal sandbox frame (line referring to exec(code, ...))
            # so the LLM sees the *user* code frames + the real exception.
            lines = [l for l in tb.split("\n")
                     if "sandbox.py" not in l and "in execute" not in l]
            short = "\n".join(lines[-30:]) if lines else tb
            return ExecResult(
                success=False, output="", error=short,
                execution_time_ms=round(elapsed, 1),
            )
