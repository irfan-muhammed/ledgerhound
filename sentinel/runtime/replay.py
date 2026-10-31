"""
sentinel.runtime.replay
=========================

V3-013: ``sentinel replay <run_dir>`` -- re-executes a recorded session's
code against the same data and reports whether it reproduces.

Scope (a deliberate v1 boundary, chosen after an architecture review, not
an oversight):

  - EXEC-MODE cells only. A kernel-mode cell may depend on things a bare
    ``exec()`` sandbox doesn't have (a different write-guard, a different
    namespace setup). Replaying it outside the kernel would produce false
    DIFFs for reasons that have nothing to do with real non-reproducibility.
    Kernel-mode ``code_exec`` events are reported SKIPPED, not DIFF.
  - Comparison is on the FULL captured output's hash first (exact,
    authoritative), falling back to NORMALIZED text comparison of the
    (possibly-truncated) preview. Raw preview text can differ for reasons
    that are not real divergence -- object-repr memory addresses, absolute
    paths that embed the (necessarily different) replay run folder,
    timestamps -- so ``_normalize()`` strips those before falling back to a
    text comparison. This trades a little strictness for not crying wolf on
    every replay of a real session.
  - Seeds (numpy, random) are pinned to 0 at the start of replay, so any
    code relying on them is deterministic across runs.

A DIFF on one cell does not imply DIFFs elsewhere: each ``code_exec`` event
is independently re-executed and independently compared, so a change to one
cell's recorded content is localized to that cell's report line.
"""

from __future__ import annotations

import hashlib
import io
import random
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.runtime.events import EventStream, sha256_file

_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"]+|(?<![\w/])/[^\s'\"]+/[^\s'\"]+")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?")


def _normalize(text: str) -> str:
    t = _HEX_ADDR_RE.sub("0xADDR", text or "")
    t = _TS_RE.sub("<TS>", t)
    t = _PATH_RE.sub("<PATH>", t)
    return t.strip()


def _read_any(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t")
    if s == ".json":
        return pd.read_json(path)
    if s == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


@dataclass
class CellResult:
    seq: int
    status: str            # "PASS" | "DIFF" | "SKIPPED" | "ERROR"
    detail: str = ""


@dataclass
class ReplayReport:
    run_dir: str
    chain_ok: bool
    data_ok: bool
    cells: list = field(default_factory=list)
    message: str = ""

    @property
    def passed(self) -> bool:
        return (self.chain_ok and self.data_ok and
                all(c.status in ("PASS", "SKIPPED") for c in self.cells))

    def render(self) -> str:
        lines = [f"Replay: {self.run_dir}"]
        lines.append(f"  chain integrity: {'OK' if self.chain_ok else 'BROKEN'}")
        lines.append(f"  data unchanged:  {'OK' if self.data_ok else 'CHANGED'}")
        if self.message:
            lines.append(f"  {self.message}")
        for c in self.cells:
            detail = f" -- {c.detail}" if c.detail else ""
            lines.append(f"  [{c.status}] seq={c.seq}{detail}")
        lines.append(f"RESULT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def replay(run_dir: str | Path) -> ReplayReport:
    run_dir = Path(run_dir)
    stream = EventStream(run_dir / "events.jsonl")
    report = ReplayReport(run_dir=str(run_dir), chain_ok=True, data_ok=True)

    vr = stream.verify()
    report.chain_ok = vr.ok
    if not vr.ok:
        report.message = f"event chain broken at seq {vr.first_bad_seq}: {vr.reason}"
        return report

    events = stream.load()
    load_events = [e for e in events if e.type == "load_data"]
    if not load_events:
        report.data_ok = False
        report.message = "no load_data event found -- nothing to replay against"
        return report

    last_load = load_events[-1]
    data_path = Path(last_load.payload.get("path", ""))
    recorded_hash = last_load.payload.get("data_sha256", "")
    if not data_path.exists():
        report.data_ok = False
        report.message = f"data file no longer exists: {data_path}"
        return report

    current_hash = sha256_file(data_path)
    if recorded_hash and current_hash != recorded_hash:
        report.data_ok = False
        report.message = ("input data has changed since this session -- "
                          f"recorded {recorded_hash[:12]}..., now {current_hash[:12]}...")
        return report

    df = _read_any(data_path)
    np.random.seed(0)
    random.seed(0)
    ns = {"pd": pd, "np": np, "df": df}

    for ev in events:
        if ev.type != "code_exec" or not ev.payload.get("success", True):
            continue
        if ev.payload.get("executor") != "exec":
            report.cells.append(CellResult(ev.seq, "SKIPPED",
                                           "kernel-mode cell, replay not supported"))
            continue

        code = ev.payload.get("code", "")
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, ns)
        except Exception as e:  # noqa: BLE001
            report.cells.append(CellResult(ev.seq, "ERROR", f"{type(e).__name__}: {e}"))
            continue

        out = buf.getvalue().strip()
        res = ns.get("result")
        if not out and res is not None:
            out = str(res)

        recorded_output = ev.payload.get("output_preview", "")
        recorded_hash_out = ev.payload.get("output_hash", "")
        replay_hash_out = hashlib.sha256(out.encode("utf-8", "replace")).hexdigest()

        if recorded_hash_out and replay_hash_out == recorded_hash_out:
            report.cells.append(CellResult(ev.seq, "PASS"))
        elif _normalize(out) == _normalize(recorded_output):
            report.cells.append(CellResult(ev.seq, "PASS", "matched after normalization"))
        else:
            report.cells.append(CellResult(
                ev.seq, "DIFF",
                f"recorded={recorded_output[:80]!r} replayed={out[:80]!r}"))

    return report
