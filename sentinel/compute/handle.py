"""
sentinel.compute.handle
=======================

DataFrameHandle -- an opaque, LLM-safe reference to a dataframe.

Problem this solves
-------------------
When an agent reasons over dataframes, two bad things happen:

  1. The raw frame (millions of rows) cannot fit in an LLM context window.
     Even a head(5) dump quickly drowns the context with cell values.
  2. When the frame migrates between devices (pandas <-> cuDF) the agent
     loses track of identity, schema, and freshness. A groupby on GPU that
     gets copied back to CPU silently creates a NEW frame; the LLM still
     thinks it's holding the old one.

Both problems disappear if the LLM never sees a dataframe directly.
Instead it sees a handle:

    <DF df_007 device=gpu shape=(1.2M, 18) schema=a9f3 last_op=groupby_sum parent=df_006>

Roughly 60 tokens. The handle:
  * has a stable string id that survives device transfers
  * tracks the current device explicitly
  * carries a schema_hash that changes if columns/dtypes drift
  * remembers the operation that produced it (last_op) and the parent handle
  * exposes shape without materialising data

The actual dataframe is stored in a process-local registry, looked up by id.
Tools/operations resolve the handle to a frame, mutate, register a new handle,
and return the new handle. The LLM only ever sees identifiers.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# -- Device enum ------------------------------------------------------------


class Device(str):
    """
    Device tag carried on a handle. Plain string subclass so it serialises
    cleanly to JSON and prints nicely.
    """

    CPU = "cpu"
    GPU = "gpu"
    UNKNOWN = "unknown"

    @classmethod
    def normalise(cls, value: str) -> str:
        v = (value or "").lower().strip()
        if v in {"cpu", "pandas"}:
            return cls.CPU
        if v in {"gpu", "cuda", "cudf"}:
            return cls.GPU
        return cls.UNKNOWN


# -- Schema fingerprint -----------------------------------------------------


def schema_fingerprint(frame: Any) -> str:
    """
    Stable short hash of (column_name, dtype) tuples.

    Same logical schema -> same hash regardless of device backend.
    Drift in column order, dtype, or column set produces a different hash.
    """
    try:
        columns = list(frame.columns)
        # cuDF dtypes stringify the same as pandas for our purposes
        dtypes = [str(frame[c].dtype) for c in columns]
    except Exception:
        return "unknown"

    payload = "|".join(f"{c}:{d}" for c, d in zip(columns, dtypes))
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def frame_shape(frame: Any) -> tuple[int, int]:
    """
    Return (n_rows, n_cols). Works for both pandas and cuDF.
    """
    try:
        n_rows, n_cols = frame.shape
        return int(n_rows), int(n_cols)
    except Exception:
        return (-1, -1)


# -- Handle -----------------------------------------------------------------


@dataclass(frozen=True)
class DataFrameHandle:
    """
    Immutable handle to a dataframe stored in the active registry.

    Frozen: once issued, a handle does not change. Device transfers, ops,
    and re-indexing all produce NEW handles (children of the old one).
    This makes the operation log a clean DAG.
    """

    id: str
    device: str                       # Device.CPU / Device.GPU / Device.UNKNOWN
    shape: tuple[int, int]            # (n_rows, n_cols), (-1,-1) if unknown
    schema_hash: str                  # 8-char sha1 of (col,dtype) tuples
    last_op: str = "create"           # short name of producing op
    parent_id: Optional[str] = None   # handle this was derived from
    name: Optional[str] = None        # optional friendly label
    created_at: float = field(default_factory=time.time)

    # -- LLM-facing display ------------------------------------------------

    def __str__(self) -> str:
        """One-line, ~60-token representation for LLM context."""
        rows, cols = self.shape
        rows_s = _humanise_rows(rows)
        name_part = f" name={self.name}" if self.name else ""
        parent_part = f" parent={self.parent_id}" if self.parent_id else ""
        return (
            f"<DF {self.id}{name_part} device={self.device} "
            f"shape=({rows_s},{cols}) schema={self.schema_hash} "
            f"last_op={self.last_op}{parent_part}>"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "device": self.device,
            "shape": list(self.shape),
            "schema_hash": self.schema_hash,
            "last_op": self.last_op,
            "parent_id": self.parent_id,
            "name": self.name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataFrameHandle":
        return cls(
            id=d["id"],
            device=d["device"],
            shape=tuple(d.get("shape", (-1, -1))),  # type: ignore[arg-type]
            schema_hash=d.get("schema_hash", "unknown"),
            last_op=d.get("last_op", "create"),
            parent_id=d.get("parent_id"),
            name=d.get("name"),
            created_at=d.get("created_at", time.time()),
        )


def _humanise_rows(n: int) -> str:
    if n < 0:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# -- Registry ---------------------------------------------------------------


class HandleRegistry:
    """
    Process-local store mapping handle.id -> dataframe.

    Single source of truth for "which frame does this handle point to".
    Thread-safe. Strong refs by default -- the registry IS the owner.

    Lifecycle:
      * register(frame, ...)  -> issues a new handle, stores the frame
      * resolve(handle)       -> returns the live frame for a handle
      * release(handle)       -> drops the frame (for memory management)
      * replace(handle, new_frame, op) -> in-place schema-compatible update,
                                          issues a new handle pointing at new_frame
                                          (old handle still resolvable until released)

    The registry is intentionally small. It is not a cache -- there is no
    eviction policy. Memory hygiene is the caller's responsibility (release
    handles you no longer need). For long-running sessions, wire a
    LifecycleManager on top (deferred to later N-series work).
    """

    def __init__(self) -> None:
        self._frames: dict[str, Any] = {}
        self._handles: dict[str, DataFrameHandle] = {}
        self._lock = threading.RLock()
        self._counter = 0  # for friendlier sequential ids: df_001, df_002...

    # -- Issue new handles ------------------------------------------------

    def register(
        self,
        frame: Any,
        *,
        device: str = Device.CPU,
        last_op: str = "create",
        parent_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> DataFrameHandle:
        """
        Take ownership of `frame` and return a new handle.
        """
        with self._lock:
            self._counter += 1
            handle_id = f"df_{self._counter:03d}"
            # Guarantee uniqueness even across registry resets within a process
            if handle_id in self._frames:
                handle_id = f"df_{uuid.uuid4().hex[:8]}"

            handle = DataFrameHandle(
                id=handle_id,
                device=Device.normalise(device),
                shape=frame_shape(frame),
                schema_hash=schema_fingerprint(frame),
                last_op=last_op,
                parent_id=parent_id,
                name=name,
            )
            self._frames[handle_id] = frame
            self._handles[handle_id] = handle
            return handle

    # -- Resolve / inspect ------------------------------------------------

    def resolve(self, handle_or_id: "DataFrameHandle | str") -> Any:
        """
        Return the live frame for a handle. Raises KeyError if released or unknown.
        """
        hid = handle_or_id.id if isinstance(handle_or_id, DataFrameHandle) else handle_or_id
        with self._lock:
            if hid not in self._frames:
                raise KeyError(f"No frame registered for handle '{hid}' (released or unknown)")
            return self._frames[hid]

    def get_handle(self, hid: str) -> DataFrameHandle:
        with self._lock:
            if hid not in self._handles:
                raise KeyError(f"No handle '{hid}'")
            return self._handles[hid]

    def __contains__(self, handle_or_id: "DataFrameHandle | str") -> bool:
        hid = handle_or_id.id if isinstance(handle_or_id, DataFrameHandle) else handle_or_id
        with self._lock:
            return hid in self._frames

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __iter__(self) -> Iterator[DataFrameHandle]:
        with self._lock:
            return iter(list(self._handles.values()))

    # -- Mutate -----------------------------------------------------------

    def replace(
        self,
        parent: "DataFrameHandle | str",
        new_frame: Any,
        *,
        device: Optional[str] = None,
        last_op: str = "transform",
        name: Optional[str] = None,
    ) -> DataFrameHandle:
        """
        Register `new_frame` as a child of `parent`. Returns the child handle.
        The parent handle remains valid (still resolvable) until explicitly released.
        """
        parent_id = parent.id if isinstance(parent, DataFrameHandle) else parent
        parent_handle = self.get_handle(parent_id)
        return self.register(
            new_frame,
            device=device or parent_handle.device,
            last_op=last_op,
            parent_id=parent_id,
            name=name,
        )

    def release(self, handle_or_id: "DataFrameHandle | str") -> bool:
        """
        Drop the frame for a handle. Returns True if something was released.
        The handle metadata stays accessible via get_handle for lineage queries.
        """
        hid = handle_or_id.id if isinstance(handle_or_id, DataFrameHandle) else handle_or_id
        with self._lock:
            if hid in self._frames:
                del self._frames[hid]
                return True
            return False

    def release_all(self) -> int:
        """Drop every frame. Returns count of frames released."""
        with self._lock:
            n = len(self._frames)
            self._frames.clear()
            return n

    # -- Lineage ----------------------------------------------------------

    def lineage(self, handle_or_id: "DataFrameHandle | str") -> list[DataFrameHandle]:
        """
        Walk parent pointers back to the root. Returns oldest-first.
        Lineage works even for released handles (metadata is kept).
        """
        hid = handle_or_id.id if isinstance(handle_or_id, DataFrameHandle) else handle_or_id
        chain: list[DataFrameHandle] = []
        with self._lock:
            cur: Optional[str] = hid
            seen: set[str] = set()
            while cur is not None and cur in self._handles and cur not in seen:
                seen.add(cur)
                h = self._handles[cur]
                chain.append(h)
                cur = h.parent_id
        chain.reverse()
        return chain


# -- Module-level default registry ------------------------------------------
#
# Most code uses the default registry. Tests and isolated tools may construct
# their own HandleRegistry instances.

_DEFAULT_REGISTRY: HandleRegistry = HandleRegistry()


def default_registry() -> HandleRegistry:
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Drop all handles in the module-level registry. Test helper."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = HandleRegistry()
