"""
sentinel.runtime.notebook_export
==================================

V3-014: emit a runnable ``analysis.ipynb`` into the run folder, built
directly from a session's event stream (``events.jsonl``).

Uses the real ``nbformat`` API (``nbformat.v4.new_notebook`` /
``new_markdown_cell`` / ``new_code_cell`` / ``new_output``) so the output
validates as a genuine notebook and ``jupyter nbconvert --execute`` can run
it end to end -- unlike ``sentinel.auto.report``'s hand-rolled dict writer,
which serializes JSON that merely *looks* like a notebook without going
through nbformat's own schema validation.

This is a third notebook writer in the codebase (alongside
``auto/report.py`` and ``notebook/journal.py``) because neither existing
one's input trace shape matches ``events.jsonl`` -- see
docs/v3.0/V3_GAP_ANALYSIS_AND_ROADMAP.md V3-014 for the reasoning; unifying
all three is future work, not in scope here.

Cell mapping from events, in stream order:
  load_data        -> code cell, `df = pd.read_csv(...)` (import cell +
                       the load itself; the last load_data wins if a
                       session loaded more than one file, matching how
                       `self._df` works during the live session) --
                       WITHOUT this, every code_exec cell that references
                       `df` would fail on `nbconvert --execute`.
  user             -> markdown, "**User:** ..."
  assistant_final  -> markdown, "**Assistant:** ..."
  finding          -> markdown, "**Finding** [sample_frame]: claim -- evidence"
  code_exec        -> code cell (only if success=True -- a failed cell would
                       make `nbconvert --execute` fail on replay, and
                       session_code.py already excludes failures for the
                       same reason), with its captured output as a stream
                       output.
Everything else (tool_call/tool_result for non-code/finding tools,
budget/compaction/firewall bookkeeping) is intentionally not rendered --
this is a narrative artifact for a human, not a full audit dump (that's
what events.jsonl itself is for).
"""

from __future__ import annotations

from pathlib import Path

from sentinel.runtime.events import Event, EventStream

_NBFORMAT_AVAILABLE = True
try:
    import nbformat
    from nbformat.v4 import (
        new_code_cell, new_markdown_cell, new_notebook, new_output,
    )
except Exception:  # noqa: BLE001
    _NBFORMAT_AVAILABLE = False


def _read_call(path: str) -> str:
    """The pandas call that reproduces how the live session loaded this
    file -- mirrors sentinel.shell.conversation._read_any's suffix dispatch."""
    suffix = Path(path).suffix.lower()
    if suffix in (".parquet", ".pq"):
        return f"pd.read_parquet({path!r})"
    if suffix == ".tsv":
        return f"pd.read_csv({path!r}, sep='\\t')"
    if suffix == ".json":
        return f"pd.read_json({path!r})"
    if suffix == ".feather":
        return f"pd.read_feather({path!r})"
    return f"pd.read_csv({path!r})"


def build_notebook(events: list[Event]):
    """Build an in-memory nbformat NotebookNode from a session's events.
    Raises RuntimeError if nbformat isn't installed."""
    if not _NBFORMAT_AVAILABLE:
        raise RuntimeError("nbformat is not installed -- cannot build a notebook")

    cells = [new_markdown_cell("# Analysis session\n\nGenerated from events.jsonl (V3-014).")]

    load_events = [e for e in events if e.type == "load_data"]
    if load_events:
        last_load = load_events[-1]
        cells.append(new_code_cell(
            "import pandas as pd\nimport numpy as np\n\n"
            f"df = {_read_call(last_load.payload.get('path', ''))}\ndf.head()"
        ))

    for ev in events:
        if ev.type == "user":
            text = ev.payload.get("text", "")
            if text.strip():
                cells.append(new_markdown_cell(f"**User:** {text}"))
        elif ev.type == "assistant_final":
            text = ev.payload.get("text", "")
            if text.strip():
                cells.append(new_markdown_cell(f"**Assistant:** {text}"))
        elif ev.type == "finding":
            claim = ev.payload.get("claim", "")
            evidence = ev.payload.get("evidence", "")
            sample_frame = ev.payload.get("sample_frame", "unknown")
            md = f"**Finding** [{sample_frame}]: {claim}"
            if evidence:
                md += f"\n\n_Evidence: {evidence}_"
            cells.append(new_markdown_cell(md))
        elif ev.type == "code_exec" and ev.payload.get("success", True):
            code = ev.payload.get("code", "")
            output_text = ev.payload.get("output_preview", "")
            outputs = []
            if output_text:
                outputs.append(new_output(output_type="stream", name="stdout",
                                          text=output_text))
            cells.append(new_code_cell(code, outputs=outputs))

    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


def export_notebook(run_dir: str | Path, filename: str = "analysis.ipynb") -> Path:
    """Read <run_dir>/events.jsonl and write <run_dir>/<filename>. Returns
    the written path. Raises if nbformat isn't installed or there's no
    event stream to build from -- callers that want this to be a
    best-effort side-effect (e.g. save_report) should catch and ignore."""
    run_dir = Path(run_dir)
    events = EventStream(run_dir / "events.jsonl").load()
    nb = build_notebook(events)
    out_path = run_dir / filename
    nbformat.write(nb, str(out_path))
    return out_path
