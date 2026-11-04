"""
Keep the MCP wire clean (ADR 0001, Phase 2 plan item 6).

Over stdio, fd 1 IS the protocol: one stray byte from a library print, a
warning, a rich spinner or a child process corrupts the stream. The SDK
diverts fd 1 only while it serves, which leaves startup and imports exposed.
`protect_stdout()` runs first, before anything else is imported: it keeps a
private duplicate of the real stdout for the wire and points fd 1 and
`sys.stdout` at stderr for the rest of the process's life. Children (the
analysis kernel) inherit the diverted fd 1 too.
"""

from __future__ import annotations

import io
import os
import sys

_wire_fd: int | None = None


def protect_stdout() -> int:
    """Divert fd 1 and sys.stdout to stderr; return the private wire fd.
    Idempotent."""
    global _wire_fd
    if _wire_fd is not None:
        return _wire_fd
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    _wire_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return _wire_fd


def wire_stdout():
    """The protocol's stdout as an anyio text file over the private fd."""
    import anyio
    fd = protect_stdout()
    raw = os.fdopen(fd, "wb", buffering=0, closefd=False)
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="\n", write_through=True)
    return anyio.wrap_file(text)
