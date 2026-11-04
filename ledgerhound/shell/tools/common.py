"""
ledgerhound.shell.tools.common
==============================

Module-level helpers shared by the tool layer and the loop: tool-argument
parsing, failure signatures, data-file readers and limits.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd



# V6.3 postmortem fix: tools that only manage plan/goal bookkeeping, never
# analysis -- excluded from the "evidence" counter update_plan checks before
# letting a step close "done" (ledgerhound/tools/plan_tools.py).
_PLAN_META_TOOLS = {"propose_plan", "update_plan", "add_node", "invalidate_node",
                    "set_goal", "set_methodology", "complete_step", "add_step",
                    "advance_phase", "phase_status",
                    "request_modeling_authorisation"}
_DATA_EXTS = (".parquet", ".csv", ".tsv", ".pq", ".json", ".feather")

# In KERNEL mode only: above this row count the shell process stops loading
# the whole dataset (the kernel already holds it) and keeps a bounded schema
# sample instead. Exec mode is unaffected -- there, self._df IS the data the
# analysis runs on, so it must stay complete.
_SHELL_LOAD_MAX_ROWS = 2_000_000
_SHELL_SCHEMA_SAMPLE_ROWS = 50_000

# Keywords that mark a question as touching a data-mutating or business
# decision (drop rows, choose an imputation, set a threshold, overwrite a
# save) -- these are the ONLY questions the firewall must never auto-answer,
# per the risk-tier policy in the system prompt's "Decide, don't ask" section.
_MUTATION_KEYWORDS = ("drop", "delete", "impute", "overwrit", "remove",
                     "discard", "truncate", "threshold", "save over",
                     "save-over")


# -- kernel-mode traceback parsing (V3-009) -----------------------------------
# A real IPython kernel runs in a separate process, so a failure comes back
# as a traceback STRING, not a live exception object -- best-effort recover
# (exception_class_name, message) from its last "Kind: message" line so the
# same teaching-hint logic (_teach_error_text) can be reused unchanged.
_TB_LAST_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)):\s*(.*)$")


def _parse_traceback_signature(tb: str) -> tuple[str, str]:
    for line in reversed((tb or "").strip().splitlines()):
        m = _TB_LAST_LINE_RE.match(line.strip())
        if m:
            return m.group(1), m.group(2)
    lines = (tb or "").strip().splitlines()
    return "Error", (lines[-1] if lines else "unknown error")


# -- identical-failure detection (V3-007) -------------------------------------

_REPEAT_FAILURE_NOTE = ("\n\nNOTE: you already tried exactly this and it failed "
                        "the same way. Do NOT retry it unchanged -- diagnose the "
                        "root cause and change your approach.")


def _parse_tool_args(raw) -> tuple[dict | None, str]:
    """(args, error) for a tool call's `arguments`, which arrive as a JSON
    string (OpenAI shape), a dict, or empty. On failure args is None and
    error says why -- NEVER a silent {}: substituting {} made a call whose
    JSON merely failed to parse look like "run_python() missing the 'code'
    argument", a message that points at the tool instead of at the real
    cause (the gateway/model produced arguments that are not valid JSON,
    typically code with unescaped quotes or newlines)."""
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, dict):
        return raw, ""
    if not isinstance(raw, str):
        return None, f"expected a JSON object, got {type(raw).__name__}"
    try:
        val = json.loads(raw, strict=False)
        if isinstance(val, str):                 # double-encoded JSON string
            val = json.loads(val, strict=False)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(val, dict):
        return None, f"expected a JSON object, got {type(val).__name__}"
    return val, ""


def _bad_args_result(name: str, raw, err: str) -> str:
    shown = raw if isinstance(raw, str) else repr(raw)
    return (f"error: the arguments for {name} could not be read as a JSON object "
            f"({err}). Nothing was run. Raw text received: {shown[:300]!r}. "
            "Resend the call with valid JSON arguments -- escape every double quote "
            "inside a string as \\\" and every line break as \\n, or write the "
            "Python with single-quoted strings so there is nothing to escape.")


def _failure_signature(result: str) -> str | None:
    """Return a short signature if a tool result reads as a failure, else
    None. Used to key the identical-failure detector -- deliberately
    conservative (prefix/substring checks on known failure phrasings) so a
    normal informational reply is never mistaken for a failure."""
    if not isinstance(result, str):
        return None
    if result.startswith("Error:"):
        return result.splitlines()[0][len("Error:"):].strip()
    if result.startswith("blocked:"):
        return result.splitlines()[0].strip()
    # Guardrail refusals (goal gates, restatement guard, chapter dedupe,
    # phase gate). Without this a refused cell counted as a success: it was
    # tallied as evidence, offered an "undo" for a change that never ran,
    # and never tripped the identical-failure note.
    if result.startswith("REFUSED"):
        return result.splitlines()[0].strip()[:160]
    # The tool registry's own failures ("error: bad arguments for X: ...",
    # "error running X: ...", "error: unknown tool X") and unreadable
    # arguments. They used to slip past this check (it only knew "Error:"),
    # so the identical-failure note never fired and a model retried the
    # same dead call ~14 times in one turn (pod log, 2026-10-05).
    if result[:6].lower() == "error:" or result[:13].lower() == "error running":
        return result.splitlines()[0].strip()[:160]
    if "could not find data at" in result:
        return "load_data: not found"
    if "failed to load it:" in result:
        return "load_data: load failure"
    if result.startswith("save FAILED entirely"):
        return "save_report: failed entirely"
    return None


def _norm_args_for_failure(args: dict) -> str:
    if not isinstance(args, dict):
        return str(args)
    return str(sorted(args.items()))



def _ts() -> str:
    """Run-folder timestamp. V3-057: was local time, the only such stamp in
    the codebase -- every other one (dataset_card.py, project.py,
    workspace/layout.py) is UTC ISO. Verified before changing: nothing
    parses this specific format back into a datetime (ledgerhound/eval/
    trend.py's regex matches ledgerhound.auto.runner's SEPARATE run_<ts> id,
    a different engine with its own strftime call, untouched here); every
    other reference to a shell run_dir's name is an opaque string/equality
    check, never re-parsed for its calendar meaning. Kept "-" as the
    field separator (not ":") so the string stays filesystem-safe on
    Windows without extra escaping."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _dataset_parts(path: Path) -> list[Path]:
    """Parquet part files under a dataset folder (hive partitions included)."""
    return sorted(f for f in Path(path).rglob("*")
                  if f.is_file() and f.suffix.lower() in (".parquet", ".pq"))


def _is_parquet_dataset_dir(path: Path) -> bool:
    """A folder that is ONE dataset split into Parquet parts -- what Spark,
    Trino exports and `to_parquet(partition_cols=...)` write -- rather than
    a folder holding several separate datasets (which still asks "which
    one?"). Recognised by hive `col=value` subfolders, a `_SUCCESS` marker,
    or every data file being a `part-*`/numbered Parquet part."""
    p = Path(path)
    if not p.is_dir():
        return False
    from ledgerhound.io.data_source import parts_form_one_dataset
    # Cheap on purpose: a user can point at a big folder (a home or project
    # dir), so never walk the whole tree -- top level, plus the inside of
    # `col=value` partition folders only.
    names: list[str] = []
    try:
        for e in p.iterdir():
            if e.is_file():
                names.append(e.name)
            elif e.is_dir() and "=" in e.name:
                for f in e.rglob("*"):
                    if f.is_file():
                        names.append(f.relative_to(p).as_posix())
                    if len(names) > 100_000:
                        break
    except OSError:
        return False
    return parts_form_one_dataset(names)


def _sha256_dataset_dir(path: Path) -> str:
    """Content hash of a Parquet dataset folder: sha256 over each part's
    relative path and its own sha256."""
    import hashlib
    from ledgerhound.runtime.events import sha256_file
    h = hashlib.sha256()
    for f in _dataset_parts(path):
        h.update(f"{f.relative_to(path).as_posix()}\0{sha256_file(f)}\n".encode())
    return h.hexdigest()


def _read_any(path: Path) -> pd.DataFrame:
    if Path(path).is_dir():
        return pd.read_parquet(path)          # a Parquet dataset folder
    s = path.suffix.lower()
    if s in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if s in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if s == ".tsv" else ",")
    if s == ".json":
        return pd.read_json(path)
    if s == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def _estimate_rows(path: Path) -> tuple[int | None, bool]:
    """(row_count, is_exact). Exact and free from parquet metadata; for CSV
    only a byte-scaled estimate is available without a full scan.

    The `is_exact` flag exists so an ESTIMATE is never reported to the user
    or logged as if it were the true row count -- callers must render an
    inexact count as "~N".
    """
    s = path.suffix.lower()
    try:
        if Path(path).is_dir():
            import pyarrow.parquet as pq
            return sum(int(pq.ParquetFile(f).metadata.num_rows)
                       for f in _dataset_parts(path)), True
        if s in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows), True
        if s in (".csv", ".tsv"):
            size = path.stat().st_size
            sep = "\t" if s == ".tsv" else ","
            head = pd.read_csv(path, sep=sep, nrows=1000)
            if len(head) < 1000:          # whole file fit in the probe: exact
                return int(len(head)), True
            with open(path, "rb") as f:
                probe = f.read(1_000_000)
            nl = probe.count(b"\n")
            if nl <= 1:
                return None, False
            return int(size / (len(probe) / nl)), False
    except Exception:  # noqa: BLE001 -- estimation is advisory
        return None, False
    return None, False


def _read_sample(path: Path, n_rows: int) -> pd.DataFrame | None:
    """Read at most `n_rows` for SCHEMA purposes only (columns + dtypes).
    Never used for statistics without being labelled as a sample."""
    s = path.suffix.lower()
    try:
        if Path(path).is_dir():
            import pyarrow.dataset as pads
            ds = pads.dataset(str(path), format="parquet", partitioning="hive")
            return ds.head(n_rows).to_pandas()
        if s in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=n_rows):
                return batch.to_pandas()
            return pf.read().to_pandas()
        if s in (".csv", ".tsv"):
            return pd.read_csv(path, sep="\t" if s == ".tsv" else ",", nrows=n_rows)
    except Exception:  # noqa: BLE001
        return None
    return None


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        sv = str(v)
        parts.append(f"{k}={sv[:40]}..." if len(sv) > 40 else f"{k}={sv}")
    return ", ".join(parts)
