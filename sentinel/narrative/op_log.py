"""
sentinel.narrative.op_log
=========================

OperationLog -- append-only record of every meaningful thing that happens
to data inside a Sentinel session.

This is the source of truth for two downstream consumers:

  1. The narrative renderer (sentinel.narrative.render) which turns the log
     into a Markdown "data scientist's blog" and an ipynb notebook.
  2. Reproducibility -- every op records its inputs, outputs, params, device,
     and timing, so a session can be replayed or audited.

Design principles
-----------------
* **Append-only.** No edits, no deletes. Once written, an entry is final.
* **Crash-safe.** Each entry is flushed to disk immediately. A crash mid-session
  still leaves a valid log up to the last successful op.
* **Step counter is monotonic.** Within a single log, step numbers never reuse.
* **Schema is forward-compatible.** Unknown fields are preserved on read; old
  logs stay readable as new fields are added.
* **No frame data inside the log.** Only handle ids, shapes, and small params.
  The log is meant to be cheap to load entirely into memory.

Kind taxonomy
-------------
* `load`       -- a frame entered the system (file read, generated, etc.)
* `transfer`   -- device migration (CPU<->GPU)
* `transform`  -- any dataframe mutation producing a child handle
* `tool`       -- a Sentinel Phase 2/3 tool invocation (psi_certify, motifs, ...)
* `finding`    -- a Finding was recorded
* `save`       -- an artifact was written to disk
* `reasoning`  -- free-text agent thought (no data change)

Usage
-----
    log = OperationLog(path="reports/op_log.jsonl")
    with log.step(kind="transform", op="groupby_amount",
                   inputs=[parent_handle], output=child_handle,
                   reasoning="checking velocity_hub motif counts by hour") as step:
        step.add_param("by", "account")
        step.add_param("agg", "sum")
        # ... do the work ...

On context exit the entry is written. Duration is the wall-clock time inside
the `with` block.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from sentinel.compute.handle import DataFrameHandle


# -- Kind constants ---------------------------------------------------------


class OperationKind(str):
    LOAD = "load"
    TRANSFER = "transfer"
    TRANSFORM = "transform"
    TOOL = "tool"
    FINDING = "finding"
    SAVE = "save"
    REASONING = "reasoning"

    @classmethod
    def all(cls) -> list[str]:
        return [cls.LOAD, cls.TRANSFER, cls.TRANSFORM, cls.TOOL,
                cls.FINDING, cls.SAVE, cls.REASONING]


# -- Operation entry --------------------------------------------------------


HandleLike = Union[DataFrameHandle, str, None]


def _to_id(h: HandleLike) -> Optional[str]:
    if h is None:
        return None
    if isinstance(h, DataFrameHandle):
        return h.id
    return str(h)


@dataclass
class Operation:
    step: int
    ts: float
    kind: str
    op: str
    inputs: list[str] = field(default_factory=list)
    output: Optional[str] = None
    device: str = "cpu"
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    duration_ms: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Operation":
        # Forward-compat: ignore unknown keys, supply defaults for missing.
        known = {
            "step", "ts", "kind", "op", "inputs", "output", "device",
            "rows_in", "rows_out", "duration_ms", "params", "reasoning",
            "tags", "notes",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# -- In-progress step (builder used inside the context manager) -------------


class _StepBuilder:
    """
    Mutable builder for the entry currently being written. Returned by
    OperationLog.step(); the entry is finalised and written when the
    context exits.
    """

    def __init__(self, op: Operation) -> None:
        self._op = op

    @property
    def step(self) -> int:
        return self._op.step

    def add_param(self, key: str, value: Any) -> None:
        self._op.params[key] = value

    def add_tag(self, tag: str) -> None:
        if tag not in self._op.tags:
            self._op.tags.append(tag)

    def set_output(self, handle: HandleLike) -> None:
        self._op.output = _to_id(handle)
        if isinstance(handle, DataFrameHandle):
            self._op.rows_out = handle.shape[0]
            self._op.device = handle.device

    def set_reasoning(self, text: str) -> None:
        self._op.reasoning = text

    def set_notes(self, text: str) -> None:
        self._op.notes = text


# -- The log ----------------------------------------------------------------


class OperationLog:
    """
    Append-only JSONL log. Thread-safe. Crash-safe (flush after every write).

    Parameters
    ----------
    path : str | Path | None
        Where to persist. If None, the log is in-memory only (useful for tests).
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path: Optional[Path] = Path(path) if path is not None else None
        self._step: int = 0
        self._lock = threading.RLock()
        self._buffer: list[Operation] = []  # used in-memory; also a recent cache
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Recover step counter from existing file (resume-friendly).
            if self.path.exists():
                self._step = self._max_step_in_file()

    def _max_step_in_file(self) -> int:
        try:
            mx = 0
            with self.path.open("r", encoding="utf-8") as f:  # type: ignore[union-attr]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = int(json.loads(line).get("step", 0))
                        if s > mx:
                            mx = s
                    except Exception:
                        continue
            return mx
        except Exception:
            return 0

    # -- Direct append (no timing context) --------------------------------

    def append(self, op: Operation) -> Operation:
        """Write a fully-formed Operation. Used by ComputeContext."""
        with self._lock:
            if op.step <= 0:
                self._step += 1
                op.step = self._step
            else:
                # External step number provided -- keep counter ahead of it.
                if op.step > self._step:
                    self._step = op.step
            self._buffer.append(op)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(op.to_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            return op

    # -- Timed context-manager step (recommended) -------------------------

    @contextmanager
    def step(
        self,
        *,
        kind: str,
        op: str,
        inputs: Optional[list[HandleLike]] = None,
        output: HandleLike = None,
        device: str = "cpu",
        reasoning: Optional[str] = None,
        tags: Optional[list[str]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Iterator[_StepBuilder]:
        """
        Time the body of the `with` block and persist one Operation on exit.

        The yielded _StepBuilder lets the body mutate the entry -- set output,
        add params, add tags -- before it gets written.
        """
        with self._lock:
            self._step += 1
            step_no = self._step

        input_ids = [i for i in (_to_id(x) for x in (inputs or [])) if i is not None]
        rows_in: Optional[int] = None
        for h in (inputs or []):
            if isinstance(h, DataFrameHandle):
                rows_in = h.shape[0]
                break

        entry = Operation(
            step=step_no,
            ts=time.time(),
            kind=kind,
            op=op,
            inputs=input_ids,
            output=_to_id(output),
            device=output.device if isinstance(output, DataFrameHandle) else device,
            rows_in=rows_in,
            rows_out=output.shape[0] if isinstance(output, DataFrameHandle) else None,
            params=dict(params or {}),
            reasoning=reasoning,
            tags=list(tags or []),
        )
        builder = _StepBuilder(entry)

        t0 = time.perf_counter()
        try:
            yield builder
        finally:
            entry.duration_ms = (time.perf_counter() - t0) * 1000.0
            self._buffer.append(entry)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(entry.to_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())

    # -- Reading / iteration ----------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[Operation]:
        with self._lock:
            return iter(list(self._buffer))

    def read_all(self) -> list[Operation]:
        """
        Return every operation in this log. Prefers the on-disk file (canonical)
        if `path` is set; otherwise returns the in-memory buffer.
        """
        if self.path is not None and self.path.exists():
            ops: list[Operation] = []
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ops.append(Operation.from_dict(json.loads(line)))
                    except Exception:
                        continue
            return ops
        return list(self._buffer)

    # -- Convenience emitters used by tools and agents --------------------

    def reasoning(self, text: str, *, tags: Optional[list[str]] = None) -> Operation:
        """Drop a free-text thought into the log."""
        with self.step(kind=OperationKind.REASONING, op="thought", tags=tags) as s:
            s.set_reasoning(text)
        return self._buffer[-1]

    def finding(
        self,
        message: str,
        *,
        confidence: Optional[float] = None,
        sample_frame: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Operation:
        """Record that a Finding was emitted."""
        with self.step(kind=OperationKind.FINDING, op="finding", tags=tags) as s:
            s.set_reasoning(message)
            if confidence is not None:
                s.add_param("confidence", confidence)
            if sample_frame is not None:
                s.add_param("sample_frame", sample_frame)
        return self._buffer[-1]

    def save(self, artifact_path: Union[str, Path], *, kind: str = "report") -> Operation:
        """Record that we saved something."""
        with self.step(kind=OperationKind.SAVE, op="save_artifact") as s:
            s.add_param("path", str(artifact_path))
            s.add_param("artifact_kind", kind)
        return self._buffer[-1]
