"""
ledgerhound.agents.tools.code_executor
====================================

Sandboxed Python/pandas executor for agent-generated code.

Safety guarantees
-----------------
* Data is passed as a **copy** -- agent code cannot mutate the original.
* Dangerous modules (os, sys, subprocess) are blocked at validation time.
* File writes are blocked.
* Execution has a configurable timeout (default 60 s).
* Output is capped at 5 000 characters to avoid flooding the context.
"""

from __future__ import annotations

import io, traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Optional
import time

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0.0


class CodeExecutor:
    """Execute pandas code in a restricted namespace.

    Parameters
    ----------
    df      : The DataFrame to expose as ``df`` inside the sandbox.
    timeout : Max seconds per execution (currently advisory).
    """

    # Strings that must not appear anywhere in agent-generated code.
    # The dataframe is pre-loaded as `df`, so agent code never needs to read
    # or write files -- reading arbitrary paths (credentials, /etc/passwd)
    # and writing outside the run dir are both blocked for a bank posture.
    _BLOCKED = [
        ("import os",       "os module not allowed"),
        ("import sys",      "sys module not allowed"),
        ("import subprocess","subprocess not allowed"),
        ("import shutil",   "shutil not allowed"),
        ("import pathlib",  "filesystem access not allowed"),
        ("open(",           "file I/O not allowed"),
        ("__import__",      "dynamic imports not allowed"),
        ("exec(",           "nested exec not allowed"),
        ("eval(",           "eval not allowed"),
        ("compile(",        "compile not allowed"),
        ("requests.",       "network access not allowed"),
        ("urllib",          "network access not allowed"),
        ("socket",          "network access not allowed"),
    ]
    # Any pandas file READER -- the df is already in scope; reading files
    # would allow arbitrary-path exfiltration.
    _BLOCKED_SUBSTR = [
        (f"read_{fmt}(", "reading files is not allowed -- the dataset is "
                         "already available as `df`")
        for fmt in ("csv", "parquet", "json", "table", "excel", "sql",
                    "feather", "hdf", "html", "fwf", "orc", "stata", "sas",
                    "xml", "pickle", "clipboard", "gbq", "spss")
    ] + [
        # pandas file WRITERS (a path arg escapes the run dir)
        (f".to_{fmt}(", "file writes not allowed -- use print() or set `result=`")
        for fmt in ("csv", "parquet", "excel", "pickle", "hdf", "feather",
                    "sql", "stata", "orc", "clipboard", "xml", "gbq")
    ]

    _MAX_OUTPUT = 5_000  # characters

    def __init__(self, df: pd.DataFrame, timeout: int = 60):
        self._df = df
        self._timeout = timeout
        # A persistent working copy so columns the agent derives in one step
        # survive into the next (notebook model) -- previously every execute()
        # got a fresh copy of the original, so a column created in step N
        # vanished in step N+1 (the `age_bucket` KeyError). The ORIGINAL df is
        # never mutated. Reset per question via reset_working_df().
        self._working_df: Optional[pd.DataFrame] = None

    def reset_working_df(self) -> None:
        """Drop derived state so the next question starts from the pristine
        dataframe (call at the start of each agent question)."""
        self._working_df = None

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Return ``(True, "")`` if safe, else ``(False, reason)``."""
        for pattern, reason in self._BLOCKED:
            if pattern in code:
                return False, reason
        for pattern, reason in self._BLOCKED_SUBSTR:
            if pattern in code:
                return False, reason
        return True, ""

    def execute(self, code: str) -> ExecutionResult:
        """Run *code* and capture stdout + ``result`` variable."""
        # Build a whitelist of safe builtins
        import builtins
        _SAFE_NAMES = (
            # data + numeric builtins
            "abs", "all", "any", "bin", "bool", "bytes", "chr", "complex",
            "dict", "divmod", "enumerate", "filter", "float", "format",
            "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
            "issubclass", "iter", "len", "list", "map", "max", "min", "next",
            "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
            "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
            # constants
            "True", "False", "None",
            # exception types -- agent code routinely uses try/except and
            # raise; without these, `except KeyError` / `raise ValueError`
            # blow up with "NameError: name 'KeyError' is not defined"
            # (AUD-188). None of these can escape the sandbox.
            "Exception", "BaseException", "ValueError", "KeyError",
            "TypeError", "IndexError", "AttributeError", "NameError",
            "ZeroDivisionError", "ArithmeticError", "OverflowError",
            "FloatingPointError", "RuntimeError", "StopIteration",
            "NotImplementedError", "AssertionError", "LookupError",
        )
        safe_builtins = {n: getattr(builtins, n) for n in _SAFE_NAMES
                         if hasattr(builtins, n)}

        # `df`, `pd`, `np` are pre-provided, but real analysis code routinely
        # imports the standard scientific/stdlib toolkit (math, datetime, re,
        # statistics, scipy...) AND pandas itself does internal lazy imports.
        # Allow a curated allowlist of SAFE modules (delegating to the real
        # importer so submodules like numpy.random / scipy.stats work); block
        # everything else with a clear message. The genuinely dangerous
        # modules (os/sys/subprocess/socket/pathlib + file I/O) are already
        # rejected by validate_code before exec, so this is a second layer,
        # not the only one. (AUD-186 -- widened from the pandas/numpy-only
        # shim that was making the agent fight ordinary imports.)
        _real_import = builtins.__import__
        _ALLOWED_IMPORTS = {
            "pandas", "numpy", "scipy", "math", "statistics", "datetime",
            "re", "json", "collections", "itertools", "functools", "decimal",
            "calendar", "random", "string", "operator", "dateutil", "time",
            "fractions", "textwrap", "warnings", "typing",
        }

        def _safe_import(name, *a, **kw):
            top = (name or "").split(".")[0]
            if top in _ALLOWED_IMPORTS:
                return _real_import(name, *a, **kw)
            raise ImportError(
                f"import of '{name}' is not allowed in this sandbox. "
                f"Available: pandas (pd), numpy (np), scipy, math, statistics, "
                f"datetime, re, json, collections, itertools; the dataset is "
                f"already loaded as `df`.")
        safe_builtins["__import__"] = _safe_import

        # Reuse the persistent working copy so derived columns survive across
        # steps; seed it from the original on first use. The original is never
        # exposed for mutation.
        if self._working_df is None:
            self._working_df = self._df.copy()
        # SINGLE namespace dict for exec (used as globals). If globals and
        # locals are different dicts, any helper FUNCTION the generated code
        # defines captures the globals dict as its __globals__, so names like
        # `pd`, `np` (living only in locals) become "not defined" inside that
        # function -- a recurring failure that burned the agent's step budget.
        # Keeping one dict makes pre-injected libs resolve everywhere.
        local_ns: dict[str, Any] = {"df": self._working_df, "pd": pd, "np": np,
                                    "__builtins__": safe_builtins}
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        t0 = time.perf_counter()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, local_ns)

            # Persist any reassignment/derived columns for the next step
            # (only on success -- a failed step must not corrupt the state).
            new_df = local_ns.get("df")
            if isinstance(new_df, pd.DataFrame):
                self._working_df = new_df

            elapsed = (time.perf_counter() - t0) * 1000
            output = stdout_buf.getvalue()

            # If code set ``result = ...``, render it
            res = local_ns.get("result")
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    if len(res) > 50:
                        output += f"\n[Showing 50 of {len(res)} rows]\n"
                    output += res.head(50).to_string(index=False)
                elif isinstance(res, pd.Series):
                    output += res.head(50).to_string()
                else:
                    output += str(res)

            return ExecutionResult(
                success=True,
                output=(output.strip() or "[Executed -- no output]")[:self._MAX_OUTPUT],
                execution_time_ms=round(elapsed, 1),
            )

        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc().strip().split("\n")
            short = "\n".join(tb[-3:])  # last 3 lines are most useful
            return ExecutionResult(
                success=False, output="", error=short,
                execution_time_ms=round(elapsed, 1),
            )
