"""
sentinel.compute.context
========================

ComputeContext -- the only legal way to move dataframes between CPU and GPU.

Background (the bug this prevents)
----------------------------------
A real session went: pandas df on CPU -> cuDF groupby on GPU -> later op
needed pandas API -> implicit .to_pandas() -> further mutation -> result
never propagated back into the original handle. The agent kept reasoning
over a stale frame; the context window filled up with confused
re-explanations. No single line was wrong, but nobody owned device state.

ComputeContext owns it. Rules:

  1. Frames enter the system via ctx.load(frame, name=...). You get a handle.
  2. Frames change device via ctx.to_device(handle, "gpu"). You get a NEW handle.
     The transfer is logged. Schema is verified before/after; mismatch raises.
  3. Tools and ops never call .to_pandas() / .to_cudf() directly. They ask
     the context: "give me this handle on CPU for the next call". The context
     issues a fresh handle pointing at the migrated frame.
  4. Every transfer is recorded in an in-memory event log (graduates to a
     persistent OperationLog in N2, the next session).

GPU detection
-------------
cuDF is imported lazily and treated as optional. If unavailable, all GPU
requests transparently fall back to CPU with a warning recorded in the
transfer log. This means code can be written assuming GPU exists; on
machines without it, Sentinel still runs (slower) without raising.
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from sentinel.compute.handle import (
    DataFrameHandle,
    Device,
    HandleRegistry,
    default_registry,
    schema_fingerprint,
)

# -- Lazy backend detection -------------------------------------------------


def _try_import_pandas() -> Any:
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def _try_import_cudf() -> Any:
    try:
        import cudf  # type: ignore[import-not-found]
        return cudf
    except Exception:
        # cuDF can fail to import even when installed (missing CUDA libs).
        # Treat any import-time error as "no GPU".
        return None


# Module-level cache -- these never change within a process.
_PANDAS = _try_import_pandas()
_CUDF = _try_import_cudf()


def gpu_available() -> bool:
    """True if cuDF imports successfully on this machine."""
    return _CUDF is not None


def cpu_available() -> bool:
    """True if pandas is importable (effectively always true)."""
    return _PANDAS is not None


# -- Transfer event log (in-memory; persisted by N2) ------------------------


@dataclass
class TransferEvent:
    ts: float
    from_handle: str
    to_handle: str
    from_device: str
    to_device: str
    op: str
    rows: int
    schema_hash_before: str
    schema_hash_after: str
    duration_ms: float
    notes: Optional[str] = None  # e.g. "GPU unavailable, fell back to CPU"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# -- Errors -----------------------------------------------------------------


class SchemaDriftError(RuntimeError):
    """Raised when a device transfer alters the schema fingerprint."""


class DeviceUnavailableError(RuntimeError):
    """Raised when a hard device requirement cannot be met (and fallback disallowed)."""


# -- The context ------------------------------------------------------------


class ComputeContext:
    """
    Single ownership point for device-aware dataframe operations.

    Parameters
    ----------
    default_device : "auto" | "cpu" | "gpu"
        "auto" picks GPU if cuDF is importable, else CPU.
    allow_gpu_fallback : bool
        If True (default), a request for "gpu" on a CPU-only machine falls
        back to "cpu" with a warning logged in the transfer history. If
        False, raises DeviceUnavailableError.
    registry : HandleRegistry | None
        Defaults to the module-level shared registry. Tests pass a fresh one.
    strict_schema : bool
        If True (default), device transfers that change schema_hash raise
        SchemaDriftError. This catches the silent-corruption bug.
    """

    def __init__(
        self,
        *,
        default_device: str = "auto",
        allow_gpu_fallback: bool = True,
        registry: Optional[HandleRegistry] = None,
        strict_schema: bool = True,
        op_log: Optional[Any] = None,
    ) -> None:
        self.registry = registry if registry is not None else default_registry()
        self.allow_gpu_fallback = allow_gpu_fallback
        self.strict_schema = strict_schema
        self.op_log = op_log  # sentinel.narrative.OperationLog or None
        self._transfer_log: list[TransferEvent] = []
        self._device_stack: list[str] = []  # for the `with ctx.on(...)` manager

        if default_device == "auto":
            self.default_device = Device.GPU if gpu_available() else Device.CPU
        else:
            self.default_device = Device.normalise(default_device)
        if self.default_device == Device.UNKNOWN:
            self.default_device = Device.CPU

    # -- Backend introspection --------------------------------------------

    @staticmethod
    def detect_device(frame: Any) -> str:
        """Inspect a frame and return its actual device tag."""
        if _CUDF is not None:
            try:
                if isinstance(frame, _CUDF.DataFrame):
                    return Device.GPU
            except Exception:
                pass
        if _PANDAS is not None:
            try:
                if isinstance(frame, _PANDAS.DataFrame):
                    return Device.CPU
            except Exception:
                pass
        return Device.UNKNOWN

    @property
    def transfers(self) -> list[TransferEvent]:
        """Read-only view of the transfer log (in-memory for now)."""
        return list(self._transfer_log)

    # -- Loading / preferred device ---------------------------------------

    @contextmanager
    def on(self, device: str) -> Iterator[str]:
        """
        Scope-bound device preference. Within the `with` block, calls to
        get(handle) without an explicit device use this one.

            with ctx.on("gpu"):
                gpu_frame = ctx.get(handle)  # migrates if needed
        """
        resolved = self._resolve_device(device)
        self._device_stack.append(resolved)
        try:
            yield resolved
        finally:
            self._device_stack.pop()

    @property
    def current_device(self) -> str:
        """Device preference for the active scope, or the context default."""
        return self._device_stack[-1] if self._device_stack else self.default_device

    def load(
        self,
        frame: Any,
        *,
        name: Optional[str] = None,
        op: str = "load",
    ) -> DataFrameHandle:
        """
        Register an existing frame with the context. Device is auto-detected.
        Returns the handle the rest of Sentinel should reason about.

        If an OperationLog is attached, a `kind=load` entry is emitted so the
        narrative renderer can open a new chapter for this dataset.
        """
        device = self.detect_device(frame)
        handle = self.registry.register(frame, device=device, last_op=op, name=name)
        self._emit_op_log_load(handle)
        return handle

    def _emit_op_log_load(self, handle: DataFrameHandle) -> None:
        """Mirror a load event into the attached OperationLog (if any)."""
        if self.op_log is None:
            return
        try:
            from sentinel.narrative.op_log import Operation, OperationKind
            entry = Operation(
                step=0,
                ts=time.time(),
                kind=OperationKind.LOAD,
                op="load",
                inputs=[],
                output=handle.id,
                device=handle.device,
                rows_in=None,
                rows_out=handle.shape[0],
                duration_ms=0.0,
                params={
                    "name": handle.name or handle.id,
                    "schema_hash": handle.schema_hash,
                    "n_cols": handle.shape[1],
                },
                tags=["load"],
            )
            self.op_log.append(entry)
        except Exception:
            pass

    # -- Device transfers -------------------------------------------------

    def to_device(
        self,
        handle: DataFrameHandle,
        device: str,
        *,
        op: str = "transfer",
    ) -> DataFrameHandle:
        """
        Return a handle to the same logical data on the requested device.

        If the handle is already on the target device, returned unchanged.
        Otherwise the frame is migrated, schema is verified, a new handle
        is issued, and a TransferEvent is recorded.
        """
        requested = Device.normalise(device)
        target = self._resolve_device(device)
        fallback_note: Optional[str] = None
        if requested == Device.GPU and target == Device.CPU and not gpu_available():
            fallback_note = "GPU requested, cuDF unavailable -- fell back to CPU"

        if handle.device == target:
            # No data movement needed. But if the *request* was for GPU and we
            # ended up on CPU due to fallback, log it loudly -- silent fallbacks
            # are the bug class this whole module exists to prevent.
            if fallback_note is not None:
                self._record_transfer(
                    TransferEvent(
                        ts=time.time(),
                        from_handle=handle.id,
                        to_handle=handle.id,
                        from_device=handle.device,
                        to_device=target,
                        op=op,
                        rows=handle.shape[0],
                        schema_hash_before=handle.schema_hash,
                        schema_hash_after=handle.schema_hash,
                        duration_ms=0.0,
                        notes=fallback_note,
                    )
                )
            return handle

        before = self.registry.resolve(handle)
        schema_before = handle.schema_hash
        t0 = time.perf_counter()
        notes: Optional[str] = None

        if target == Device.GPU:
            after, notes_inner = self._to_gpu(before)
            if notes_inner is not None:
                # GPU unavailable -> we stayed on CPU. Update target accordingly.
                target = Device.CPU
                notes = notes_inner
        else:
            after = self._to_cpu(before)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        schema_after = schema_fingerprint(after)

        new_handle = self.registry.replace(
            handle,
            after,
            device=target,
            last_op=op,
            name=handle.name,
        )

        if self.strict_schema and schema_before != schema_after:
            self._transfer_log.append(
                TransferEvent(
                    ts=time.time(),
                    from_handle=handle.id,
                    to_handle=new_handle.id,
                    from_device=handle.device,
                    to_device=target,
                    op=op,
                    rows=handle.shape[0],
                    schema_hash_before=schema_before,
                    schema_hash_after=schema_after,
                    duration_ms=duration_ms,
                    notes="SCHEMA DRIFT",
                )
            )
            raise SchemaDriftError(
                f"Device transfer {handle.device}->{target} altered schema: "
                f"{schema_before} -> {schema_after} (handle {handle.id} -> {new_handle.id})"
            )

        self._record_transfer(
            TransferEvent(
                ts=time.time(),
                from_handle=handle.id,
                to_handle=new_handle.id,
                from_device=handle.device,
                to_device=target,
                op=op,
                rows=handle.shape[0],
                schema_hash_before=schema_before,
                schema_hash_after=schema_after,
                duration_ms=duration_ms,
                notes=notes,
            )
        )
        return new_handle

    def get(
        self,
        handle: DataFrameHandle,
        *,
        device: Optional[str] = None,
    ) -> tuple[DataFrameHandle, Any]:
        """
        Materialise a handle on the requested (or current scope's) device.
        Returns (possibly-new-handle, frame).

        Use this in tools instead of registry.resolve() directly -- it
        guarantees the frame is on the device the caller expects.
        """
        target = self._resolve_device(device or self.current_device)
        new_handle = self.to_device(handle, target)
        return new_handle, self.registry.resolve(new_handle)

    def update(
        self,
        parent: DataFrameHandle,
        new_frame: Any,
        *,
        op: str,
        name: Optional[str] = None,
    ) -> DataFrameHandle:
        """
        Register the result of a transform as a child of `parent`.
        Device is auto-detected from new_frame. Use this after every
        non-trivial dataframe mutation.
        """
        device = self.detect_device(new_frame)
        return self.registry.replace(
            parent,
            new_frame,
            device=device,
            last_op=op,
            name=name,
        )

    # -- Internals --------------------------------------------------------

    def _record_transfer(self, ev: TransferEvent) -> None:
        """Append to in-memory log AND mirror to OperationLog if attached."""
        self._transfer_log.append(ev)
        self._emit_op_log_transfer(ev)

    def _emit_op_log_transfer(self, ev: TransferEvent) -> None:
        """If an OperationLog is attached, mirror this transfer event to it."""
        if self.op_log is None:
            return
        try:
            from sentinel.narrative.op_log import Operation, OperationKind
            entry = Operation(
                step=0,  # OperationLog assigns
                ts=ev.ts,
                kind=OperationKind.TRANSFER,
                op=ev.op,
                inputs=[ev.from_handle],
                output=ev.to_handle,
                device=ev.to_device,
                rows_in=ev.rows,
                rows_out=ev.rows,
                duration_ms=ev.duration_ms,
                params={
                    "from_device": ev.from_device,
                    "to_device": ev.to_device,
                    "schema_hash_before": ev.schema_hash_before,
                    "schema_hash_after": ev.schema_hash_after,
                },
                notes=ev.notes,
                tags=["transfer"],
            )
            self.op_log.append(entry)
        except Exception:
            # Logging must never break the data path.
            pass

    def _resolve_device(self, device: str) -> str:
        d = Device.normalise(device)
        if d == Device.GPU and not gpu_available():
            if not self.allow_gpu_fallback:
                raise DeviceUnavailableError(
                    "GPU (cuDF) requested but not available and fallback is disabled"
                )
            return Device.CPU
        if d == Device.UNKNOWN:
            return self.default_device
        return d

    def _to_gpu(self, frame: Any) -> tuple[Any, Optional[str]]:
        """Return (gpu_frame, notes). notes is non-None on fallback."""
        if _CUDF is None:
            if not self.allow_gpu_fallback:
                raise DeviceUnavailableError("cuDF not available")
            warnings.warn(
                "GPU requested but cuDF unavailable -- staying on CPU",
                RuntimeWarning,
                stacklevel=3,
            )
            return frame, "cuDF unavailable, fell back to CPU"
        # Already cuDF?
        try:
            if isinstance(frame, _CUDF.DataFrame):
                return frame, None
        except Exception:
            pass
        # pandas -> cuDF
        return _CUDF.DataFrame.from_pandas(frame), None

    def _to_cpu(self, frame: Any) -> Any:
        # Already pandas?
        if _PANDAS is not None:
            try:
                if isinstance(frame, _PANDAS.DataFrame):
                    return frame
            except Exception:
                pass
        # cuDF -> pandas
        if _CUDF is not None:
            try:
                if isinstance(frame, _CUDF.DataFrame):
                    return frame.to_pandas()
            except Exception:
                pass
        # Unknown -> return as-is; caller's problem
        return frame


# -- Module-level default context -------------------------------------------


_DEFAULT_CONTEXT: Optional[ComputeContext] = None


def default_context() -> ComputeContext:
    """Lazily construct and return the process-wide default context."""
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = ComputeContext()
    return _DEFAULT_CONTEXT


def reset_default_context() -> None:
    """Test helper -- drop the global context so the next call rebuilds it."""
    global _DEFAULT_CONTEXT
    _DEFAULT_CONTEXT = None
