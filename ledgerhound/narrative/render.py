"""
ledgerhound.narrative.render
=========================

Turn an OperationLog into a data scientist's blog (Markdown) or a Jupyter
notebook (.ipynb JSON). Same source, two output formats -- they never drift
because they read the same log.

Voice: first-person agent. Reads like a colleague's working notebook.

Chapter structure
-----------------
A new chapter begins whenever a `load` op appears (i.e. fresh data entered
the system). Within a chapter, ops are rendered as paragraphs grouped by kind:

  load        ->  "I started with <name>, a <rows>-row dataset (schema=<hash>)."
  transfer    ->  "I moved df_007 from CPU to GPU (took 12.4ms)."
                  (fallbacks become loud notes)
  transform   ->  "I ran <op> on <device>, producing df_008 (<rows_out> rows)."
  tool        ->  "I invoked <op> with params {...}."
  finding     ->  blockquote callout with confidence + sample_frame badge
  save        ->  "I saved <kind> to <path>."
  reasoning   ->  italic paragraph, free-form thought

For the notebook, the same paragraphs become markdown cells, and each
non-reasoning op also gets a small code cell with the params dict for
reproducibility. Code cells are illustrative -- they document what was
done, not always directly runnable, because the actual transform code
lives in Ledgerhound's tools.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from ledgerhound.narrative.op_log import Operation, OperationKind


# -- Public API -------------------------------------------------------------


def render_markdown(
    ops: Iterable[Operation],
    *,
    title: Optional[str] = None,
    author: str = "Ledgerhound agent",
) -> str:
    """Render an iterable of Operation entries as a Markdown narrative."""
    ops_list = sorted(ops, key=lambda o: o.step)
    lines: list[str] = []

    lines.append(f"# {title or 'EDA Session'}")
    lines.append("")
    lines.append(f"*Author: {author}* . *Generated: {_now()}* . *{len(ops_list)} operations*")
    lines.append("")

    if not ops_list:
        lines.append("_The log is empty -- nothing happened in this session._")
        return "\n".join(lines)

    # Group into chapters by `load` boundaries.
    chapters = _split_chapters(ops_list)
    for idx, chapter in enumerate(chapters, start=1):
        header_name = _chapter_title(chapter, idx)
        lines.append(f"## {idx}. {header_name}")
        lines.append("")
        for op in chapter:
            lines.append(_render_paragraph(op))
            lines.append("")

    # Closing summary
    lines.append("---")
    lines.append("")
    lines.append(_render_summary(ops_list))
    return "\n".join(lines).rstrip() + "\n"


def render_notebook(
    ops: Iterable[Operation],
    *,
    title: Optional[str] = None,
    author: str = "Ledgerhound agent",
) -> dict[str, Any]:
    """Render the same content as an ipynb-compatible JSON dict."""
    ops_list = sorted(ops, key=lambda o: o.step)
    cells: list[dict[str, Any]] = []

    # Title cell
    head = [
        f"# {title or 'EDA Session'}",
        "",
        f"*Author: {author}* . *Generated: {_now()}* . *{len(ops_list)} operations*",
    ]
    cells.append(_md_cell(head))

    chapters = _split_chapters(ops_list)
    for idx, chapter in enumerate(chapters, start=1):
        header_name = _chapter_title(chapter, idx)
        cells.append(_md_cell([f"## {idx}. {header_name}"]))
        for op in chapter:
            cells.append(_md_cell([_render_paragraph(op)]))
            code = _render_code_cell(op)
            if code is not None:
                cells.append(_code_cell(code))

    cells.append(_md_cell(["---", "", _render_summary(ops_list)]))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "ledgerhound": {"generated_at": _now(), "n_ops": len(ops_list)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_markdown(
    ops: Iterable[Operation],
    path: Union[str, Path],
    *,
    title: Optional[str] = None,
    author: str = "Ledgerhound agent",
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(ops, title=title, author=author), encoding="utf-8")
    return p


def save_notebook(
    ops: Iterable[Operation],
    path: Union[str, Path],
    *,
    title: Optional[str] = None,
    author: str = "Ledgerhound agent",
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nb = render_notebook(ops, title=title, author=author)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


# -- Internals --------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _split_chapters(ops: list[Operation]) -> list[list[Operation]]:
    chapters: list[list[Operation]] = []
    current: list[Operation] = []
    for op in ops:
        if op.kind == OperationKind.LOAD and current:
            chapters.append(current)
            current = [op]
        else:
            current.append(op)
    if current:
        chapters.append(current)
    if not chapters:
        chapters = [[]]
    return chapters


def _chapter_title(chapter: list[Operation], idx: int) -> str:
    for op in chapter:
        if op.kind == OperationKind.LOAD:
            name = op.params.get("name") or op.op
            return f"Loading {name}"
    # No explicit load -- title from dominant kind
    kinds = [o.kind for o in chapter]
    if OperationKind.TOOL in kinds:
        return "Analysis"
    if OperationKind.FINDING in kinds:
        return "Findings"
    return f"Stage {idx}"


def _render_paragraph(op: Operation) -> str:
    k = op.kind
    if k == OperationKind.LOAD:
        name = op.params.get("name") or op.op
        rows = _fmt_rows(op.rows_out)
        return (
            f"I started with **{name}** -- a {rows}-row dataset on `{op.device}` "
            f"(handle `{op.output}`, schema `{op.params.get('schema_hash', '?')}`)."
        )

    if k == OperationKind.TRANSFER:
        from_d = op.params.get("from_device", "?")
        to_d = op.params.get("to_device", "?")
        if op.notes and "fell back" in (op.notes or "").lower():
            return (
                f"! I asked for GPU but cuDF wasn't available, so I stayed on "
                f"`{to_d}` for handle `{op.inputs[0] if op.inputs else '?'}`. "
                f"_(Note: {op.notes})_"
            )
        if op.notes == "SCHEMA DRIFT":
            return (
                f"X **Schema drift detected** during transfer `{from_d}` -> `{to_d}` "
                f"on `{op.inputs[0] if op.inputs else '?'}` -- operation aborted."
            )
        return (
            f"I moved `{op.inputs[0] if op.inputs else '?'}` from "
            f"`{from_d}` to `{to_d}` (took {op.duration_ms:.1f}ms)."
        )

    if k == OperationKind.TRANSFORM:
        rin = _fmt_rows(op.rows_in)
        rout = _fmt_rows(op.rows_out)
        parent = op.inputs[0] if op.inputs else "?"
        return (
            f"I ran **{op.op}** on `{op.device}` against `{parent}` "
            f"({rin} rows in -> {rout} rows out, {op.duration_ms:.1f}ms), "
            f"producing `{op.output}`."
        )

    if k == OperationKind.TOOL:
        params_blob = _short_params(op.params)
        return (
            f"I invoked the **{op.op}** tool"
            + (f" with `{params_blob}`" if params_blob else "")
            + f" (took {op.duration_ms:.1f}ms)."
        )

    if k == OperationKind.FINDING:
        conf = op.params.get("confidence")
        frame = op.params.get("sample_frame")
        badge_bits = []
        if conf is not None:
            try:
                badge_bits.append(f"confidence {float(conf):.2f}")
            except Exception:
                pass
        if frame:
            badge_bits.append(f"frame: {frame}")
        badge = f" _({' . '.join(badge_bits)})_" if badge_bits else ""
        return f">  **Finding.** {op.reasoning or '(no message)'}{badge}"

    if k == OperationKind.SAVE:
        kind = op.params.get("artifact_kind", "artifact")
        path = op.params.get("path", "?")
        return f"I saved the {kind} to `{path}`."

    if k == OperationKind.REASONING:
        return f"_{op.reasoning or '(thinking...)'}_"

    # Unknown kind
    return f"`{op.kind}` . {op.op} -- {op.reasoning or ''}"


def _fmt_rows(n: Optional[int]) -> str:
    if n is None or n < 0:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _short_params(params: dict[str, Any], max_chars: int = 120) -> str:
    if not params:
        return ""
    try:
        s = json.dumps(params, default=str, sort_keys=True)
    except Exception:
        s = str(params)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _render_summary(ops: list[Operation]) -> str:
    counts: dict[str, int] = {}
    total_ms = 0.0
    findings_count = 0
    for op in ops:
        counts[op.kind] = counts.get(op.kind, 0) + 1
        total_ms += op.duration_ms
        if op.kind == OperationKind.FINDING:
            findings_count += 1
    bits = [f"{n} {k}" for k, n in sorted(counts.items())]
    return (
        f"**Session summary.** {len(ops)} operations . "
        f"{findings_count} findings . "
        f"{total_ms / 1000:.2f}s of recorded work . "
        f"({', '.join(bits)})."
    )


# -- Notebook cell helpers --------------------------------------------------


def _md_cell(lines: list[str]) -> dict[str, Any]:
    src = "\n".join(lines)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_for_ipynb(src),
    }


def _code_cell(code: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_for_ipynb(code),
    }


def _split_for_ipynb(s: str) -> list[str]:
    """ipynb stores source as a list of lines, each WITH the trailing \\n."""
    if not s:
        return []
    parts = s.splitlines(keepends=True)
    # Ensure last line ends with newline-less if it doesn't have one
    return parts


def _render_code_cell(op: Operation) -> Optional[str]:
    """
    Produce an illustrative code cell for an op. Returns None for kinds where
    a code cell would just add noise (reasoning, save).
    """
    if op.kind in (OperationKind.REASONING, OperationKind.SAVE):
        return None
    header = f"# step {op.step} . {op.kind} . {op.op}"
    body_lines: list[str] = [header]
    if op.inputs:
        body_lines.append(f"# inputs: {op.inputs}")
    if op.output:
        body_lines.append(f"# output: {op.output} ({_fmt_rows(op.rows_out)} rows on {op.device})")
    if op.params:
        try:
            params_repr = json.dumps(op.params, default=str, indent=2)
        except Exception:
            params_repr = repr(op.params)
        body_lines.append(f"params = {params_repr}")
    if op.duration_ms:
        body_lines.append(f"# duration: {op.duration_ms:.1f} ms")
    return "\n".join(body_lines)
