"""
ledgerhound.shell.intra_compaction
===============================

V5-005. The second compaction strategy: shrink ONE oversized result in place,
before it ever enters the history.

``ShellContextCompressor`` trims *history* once the window crosses a fraction
of its budget. That is inter-turn compaction, and it was the only strategy we
had. It cannot help with the case that actually breaks a session at scale: a
single ``run_python`` cell whose output is enormous on its own -- a 400-column
``describe()``, a full ``value_counts()`` over a high-cardinality VPA column,
a traceback with a thousand frames. One cell like that can consume most of the
window in one turn, and the history compactor only reacts afterwards, once the
damage is already in the transcript.

Gateway's compaction crate carries three strategies (inter, intra, code) where
we had one; this closes the gap that mattered. The host CLIs solve the same
problem by spilling the full payload to disk and handing the model a
truncated view -- so does this, using the run folder we already have.

Design rules:

  * **The head and the tail both survive.** Truncating only the tail loses the
    summary line that usually matters most (``[1000 rows x 40 columns]``,
    the final assertion, the exception type).
  * **Tabular output keeps its shape.** A DataFrame repr cut mid-row is
    unreadable; cut whole lines and say how many were dropped.
  * **Nothing is destroyed.** The full text lands in the run folder and the
    replacement names the file, so a number the model needs is one read away
    rather than gone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Above this many characters a single tool result is compacted in place.
# ~3.6 chars/token (the estimate shell/compressor.py already uses) puts this
# near 3.3k tokens -- large enough that ordinary output passes untouched,
# small enough that one cell cannot eat a 256k window.
MAX_RESULT_CHARS = 12_000

# How much of an oversized result to keep from each end.
HEAD_CHARS = 7_000
TAIL_CHARS = 3_000


@dataclass
class CompactionOutcome:
    text: str
    compacted: bool
    original_chars: int = 0
    spill_path: str = ""

    @property
    def saved_chars(self) -> int:
        return max(0, self.original_chars - len(self.text))


def _spill(text: str, spill_dir: Path | None) -> str:
    """Write the full text somewhere retrievable. Returns a path or ""."""
    if spill_dir is None:
        return ""
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
        path = spill_dir / f"result_{digest}.txt"
        if not path.exists():
            path.write_text(text, encoding="utf-8", errors="replace")
        return str(path)
    except Exception:  # noqa: BLE001 -- spilling is a convenience, not a gate
        return ""


def _cut_on_line_boundary(text: str, head: int, tail: int) -> tuple[str, str, int]:
    """(head_part, tail_part, n_lines_dropped), cutting whole lines only."""
    lines = text.splitlines(keepends=True)

    head_lines, used = [], 0
    for ln in lines:
        if used + len(ln) > head:
            break
        head_lines.append(ln)
        used += len(ln)

    tail_lines, used = [], 0
    for ln in reversed(lines):
        if used + len(ln) > tail:
            break
        tail_lines.append(ln)
        used += len(ln)
    tail_lines.reverse()

    dropped = len(lines) - len(head_lines) - len(tail_lines)
    if dropped <= 0:
        # The two halves overlap: the text fits after all.
        return text, "", 0
    return "".join(head_lines), "".join(tail_lines), dropped


def compact_result(text: str, *,
                   spill_dir: Path | None = None,
                   max_chars: int = MAX_RESULT_CHARS,
                   head_chars: int = HEAD_CHARS,
                   tail_chars: int = TAIL_CHARS) -> CompactionOutcome:
    """Shrink one oversized tool result, keeping both ends."""
    if not text or len(text) <= max_chars:
        return CompactionOutcome(text=text, compacted=False,
                                 original_chars=len(text or ""))

    original = len(text)
    head, tail, dropped = _cut_on_line_boundary(text, head_chars, tail_chars)
    if dropped <= 0:
        return CompactionOutcome(text=text, compacted=False,
                                 original_chars=original)

    path = _spill(text, spill_dir)
    marker = [
        "",
        f"[... {dropped:,} lines / ~{original - len(head) - len(tail):,} "
        f"characters elided from the middle of this result ...]",
    ]
    if path:
        marker.append(f"[full output: {path}  -- read it if you need a value "
                      f"from the elided middle; do NOT re-run the cell]")
    else:
        marker.append("[the elided middle was not saved; re-run with a "
                      "narrower selection if you need it]")
    marker.append("")

    out = head.rstrip("\n") + "\n" + "\n".join(marker) + tail.lstrip("\n")
    return CompactionOutcome(text=out, compacted=True,
                             original_chars=original, spill_path=path)
