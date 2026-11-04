"""
ledgerhound.notebook.journal
=========================

Build a read-only ``.ipynb`` journal from an EDA agent trace.

The trace produced by ``EDAAgent.ask()`` is a list of dicts:

* ``{"type": "thinking", "text": ...}``                        -- model thought
* ``{"type": "dict_tool", "tool": ..., "args": ..., "result": ...}``  -- KB call
* ``{"type": "findings_tool", "tool": "record_finding", "args": ...,
       "result": ..., "finding_id": ...}``                     -- finding emitted
* ``{"type": "action", "code": ..., "result": ...}``           -- Python exec
* ``{"type": "chart", "code": ..., "result": ...}``            -- chart code
* ``{"type": "hitl", "question": ..., "answer": ...}``         -- human prompt

Each entry is rendered to one or more notebook cells.  The journal
closes with the Final Answer and a Findings summary (IDs + warnings).

Usage
-----

>>> from ledgerhound.notebook import JournalWriter
>>> w = JournalWriter()  # default dir: .ledgerhound/journals
>>> path = w.write(
...     question="What drives fraud?",
...     trace=result.thinking_trace,
...     answer=result.answer,
...     dataset_name="upi_q1_2026",
...     finding_ids=result.finding_ids,
...     finding_warnings=result.finding_warnings,
...     elapsed_ms=result.elapsed_ms,
... )
>>> print(path)
.ledgerhound/journals/upi_q1_2026_20260427_143052.ipynb
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    import nbformat
    from nbformat.v4 import (
        new_code_cell,
        new_markdown_cell,
        new_notebook,
        new_output,
    )
    _NBFORMAT_AVAILABLE = True
except ImportError:  # pragma: no cover -- pyproject lists nbformat>=5.9
    _NBFORMAT_AVAILABLE = False


_DEFAULT_JOURNAL_DIR = ".ledgerhound/journals"


# -- Writer ----------------------------------------------------------------


class JournalWriter:
    """Build .ipynb journals from agent traces."""

    def __init__(self, output_dir: str | Path = _DEFAULT_JOURNAL_DIR):
        if not _NBFORMAT_AVAILABLE:
            raise ImportError(
                "nbformat is required for journal writing. "
                "Install with: pip install 'nbformat>=5.9'"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Public API ---------------------------------------------------------

    def write(
        self,
        *,
        question: str,
        trace: list[dict],
        answer: str,
        dataset_name: Optional[str] = None,
        finding_ids: Optional[list[str]] = None,
        finding_warnings: Optional[list[str]] = None,
        elapsed_ms: float = 0.0,
        path: str | Path | None = None,
    ) -> Path:
        """Write a single-question journal and return its path.

        Parameters
        ----------
        question:
            The user-facing question that drove this session.
        trace:
            ``EDAAgent.AgentResult.thinking_trace`` -- list of typed dicts.
        answer:
            The agent's Final Answer string.
        dataset_name:
            Tagged in the title cell and used in the default filename.
        finding_ids, finding_warnings:
            From ``AgentResult`` -- rendered as a summary section.
        elapsed_ms:
            Wall-clock time of the session, shown in the header.
        path:
            Override the auto-generated filename.  When None, writes to
            ``<output_dir>/<dataset>_<UTC-timestamp>.ipynb``.
        """
        nb = new_notebook()
        cells: list = []

        # Title + meta
        cells.append(new_markdown_cell(
            self._build_header(question, dataset_name, elapsed_ms)
        ))
        cells.append(new_markdown_cell(
            f"## Question\n\n> {self._escape_md_quote(question)}"
        ))

        # Trace cells (in order)
        cells.extend(self._render_trace(trace))

        # Final Answer
        cells.append(new_markdown_cell(
            f"## Final Answer\n\n{answer.strip() or '_(empty)_'}"
        ))

        # Findings summary
        if finding_ids or finding_warnings:
            cells.append(new_markdown_cell(
                self._build_findings_summary(finding_ids, finding_warnings)
            ))

        nb.cells = cells

        # Resolve output path
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ds = self._slug(dataset_name) if dataset_name else "session"
            out_path = self.output_dir / f"{ds}_{ts}.ipynb"
        else:
            out_path = Path(path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        log.debug("Journal written to %s", out_path)
        return out_path

    # -- Header / footer builders -------------------------------------------

    @staticmethod
    def _build_header(
        question: str,
        dataset_name: Optional[str],
        elapsed_ms: float,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ds_line = f"**Dataset:** `{dataset_name}`  \n" if dataset_name else ""
        elapsed_line = (
            f"**Elapsed:** {elapsed_ms / 1000:.1f}s  \n"
            if elapsed_ms else ""
        )
        return (
            "# Ledgerhound EDA Journal\n\n"
            f"{ds_line}"
            f"**Generated:** {ts}  \n"
            f"{elapsed_line}"
            "\n---\n"
        )

    @staticmethod
    def _build_findings_summary(
        finding_ids: Optional[list[str]],
        finding_warnings: Optional[list[str]],
    ) -> str:
        lines = ["## Findings Recorded\n"]
        if finding_ids:
            lines.append(f"**{len(finding_ids)} finding(s) recorded:**\n")
            for fid in finding_ids:
                lines.append(f"- `{fid}`")
        else:
            lines.append("_No findings were recorded for this session._")
        if finding_warnings:
            lines.append("\n### ! Sample-Frame & Evidence Warnings\n")
            for w in finding_warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    # -- Trace rendering ----------------------------------------------------

    def _render_trace(self, trace: list[dict]) -> list:
        """Convert each trace entry into one or more notebook cells."""
        cells: list = []
        for i, entry in enumerate(trace):
            t = entry.get("type")
            if t == "thinking":
                cells.append(self._cell_thinking(entry, i))
            elif t == "dict_tool":
                cells.append(self._cell_dict_tool(entry, i))
            elif t == "findings_tool":
                cells.append(self._cell_findings_tool(entry, i))
            elif t == "action":
                cells.extend(self._cells_action(entry, i))
            elif t == "chart":
                cells.extend(self._cells_chart(entry, i))
            elif t == "hitl":
                cells.append(self._cell_hitl(entry, i))
            else:
                # Unknown type -- render as raw json fallback
                cells.append(new_markdown_cell(
                    f"### Step {i + 1}: `{t or 'unknown'}`\n\n"
                    f"```json\n{json.dumps(entry, default=str, indent=2)}\n```"
                ))
        return cells

    @staticmethod
    def _cell_thinking(entry: dict, idx: int):
        text = (entry.get("text") or "").strip()
        # Use blockquote for thinking -- visually distinct from prose
        body = "\n".join(f"> {ln}" for ln in text.splitlines() or [""])
        return new_markdown_cell(
            f"### Step {idx + 1}:  Thinking\n\n{body}"
        )

    @staticmethod
    def _cell_dict_tool(entry: dict, idx: int):
        tool = entry.get("tool", "?")
        args = entry.get("args", "") or ""
        result = entry.get("result", "") or ""
        # Pretty-print the JSON result if possible
        result_md = JournalWriter._pretty_json(result)
        args_inline = " ".join(str(args).split())[:120]
        return new_markdown_cell(
            f"### Step {idx + 1}:  KB call -- `{tool}`\n\n"
            f"**Args:** `{args_inline}`\n\n"
            f"<details><summary>KB response</summary>\n\n"
            f"```json\n{result_md}\n```\n\n</details>"
        )

    @staticmethod
    def _cell_findings_tool(entry: dict, idx: int):
        tool = entry.get("tool", "?")
        fid = entry.get("finding_id")
        result = entry.get("result", "") or ""
        args_str = entry.get("args", "") or ""

        # Try to extract claim / sample_frame / confidence for a clean header
        claim = sample_frame = confidence = None
        try:
            if args_str:
                args_dict = json.loads(args_str)
                claim = args_dict.get("claim")
                sample_frame = args_dict.get("sample_frame")
                confidence = args_dict.get("confidence")
        except Exception:
            pass

        if tool == "record_finding" and fid:
            sf_line = f"**Sample frame:** `{sample_frame}`  \n" if sample_frame else ""
            conf_line = f"**Confidence:** {confidence}  \n" if confidence is not None else ""
            claim_line = f"**Claim:** {claim}\n\n" if claim else ""
            return new_markdown_cell(
                f"### Step {idx + 1}:  Finding `{fid}`\n\n"
                f"{claim_line}"
                f"{sf_line}"
                f"{conf_line}"
                f"\n<details><summary>Full record + warnings</summary>\n\n"
                f"```json\n{JournalWriter._pretty_json(result)}\n```\n\n</details>"
            )
        # list_findings or record_finding without id (rejected)
        return new_markdown_cell(
            f"### Step {idx + 1}:  `{tool}`\n\n"
            f"```json\n{JournalWriter._pretty_json(result)}\n```"
        )

    @staticmethod
    def _cells_action(entry: dict, idx: int):
        code = (entry.get("code") or "").strip()
        result = (entry.get("result") or "").strip()
        cells = [
            new_markdown_cell(f"### Step {idx + 1}:  Code"),
        ]
        cc = new_code_cell(code)
        if result:
            cc.outputs = [
                new_output(
                    output_type="stream",
                    name="stdout",
                    text=result,
                )
            ]
        cells.append(cc)
        return cells

    @staticmethod
    def _cells_chart(entry: dict, idx: int):
        code = (entry.get("code") or "").strip()
        result = (entry.get("result") or "").strip()
        cells = [
            new_markdown_cell(f"### Step {idx + 1}:  Chart"),
        ]
        cc = new_code_cell(code)
        if result:
            cc.outputs = [
                new_output(
                    output_type="stream",
                    name="stdout",
                    text=result,
                )
            ]
        cells.append(cc)

        # If the chart was saved to disk, embed an image markdown link
        chart_path = JournalWriter._extract_chart_path(result)
        if chart_path:
            cells.append(new_markdown_cell(
                f"![Chart]({chart_path})"
            ))
        return cells

    @staticmethod
    def _cell_hitl(entry: dict, idx: int):
        q = entry.get("question", "")
        a = entry.get("answer", "")
        return new_markdown_cell(
            f"### Step {idx + 1}:  Human-in-the-loop\n\n"
            f"**Q:** {q}\n\n"
            f"**A:** {a}"
        )

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _pretty_json(text: Any) -> str:
        """Re-format a JSON-shaped string with indent=2; pass-through on failure."""
        if not isinstance(text, str):
            try:
                return json.dumps(text, indent=2, default=str)
            except Exception:
                return str(text)
        try:
            return json.dumps(json.loads(text), indent=2, default=str)
        except Exception:
            return text

    @staticmethod
    def _extract_chart_path(result: str) -> Optional[str]:
        """Pull a path out of 'Chart saved: /tmp/xxx.png'-style strings."""
        if not result:
            return None
        marker = "Chart saved:"
        if marker in result:
            tail = result.split(marker, 1)[1].strip()
            # Trim at first newline to handle multi-line stdout
            return tail.split("\n", 1)[0].strip()
        return None

    @staticmethod
    def _escape_md_quote(text: str) -> str:
        """Multi-line safe blockquote rendering."""
        return text.replace("\n", "\n> ")

    @staticmethod
    def _slug(s: str) -> str:
        """Filesystem-safe stub of a string."""
        return "".join(
            c if (c.isalnum() or c in "-_.") else "_"
            for c in s
        )[:80] or "session"


# -- Convenience ----------------------------------------------------------


def write_journal(
    *,
    question: str,
    trace: list[dict],
    answer: str,
    dataset_name: Optional[str] = None,
    finding_ids: Optional[list[str]] = None,
    finding_warnings: Optional[list[str]] = None,
    elapsed_ms: float = 0.0,
    output_dir: str | Path = _DEFAULT_JOURNAL_DIR,
    path: str | Path | None = None,
) -> Path:
    """One-shot journal writer.  See :class:`JournalWriter.write` for params."""
    writer = JournalWriter(output_dir=output_dir)
    return writer.write(
        question=question,
        trace=trace,
        answer=answer,
        dataset_name=dataset_name,
        finding_ids=finding_ids,
        finding_warnings=finding_warnings,
        elapsed_ms=elapsed_ms,
        path=path,
    )
