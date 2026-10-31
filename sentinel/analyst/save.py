"""
sentinel.analyst.save
=====================

``save_output`` -- the server-aware save capability (Phase 2 primitive).

Sentinel runs inside a server/kernel, so the honest capability envelope is
limited to formats it can actually produce there. This module states that
envelope and renders a report payload into the chosen format. The Analyst is
expected to *ask the user* which format to use (Phase 0 decision), defaulting to
``md`` when the user does not choose.

Rendering is intentionally self-contained (md/html/ipynb rendered natively; pdf
delegated to the existing fpdf report with a graceful fallback) so the primitive
has no hard dependency on the full report pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The formats this server can actually produce, with a one-line envelope note.
SAVE_ENVELOPE: dict[str, str] = {
    "md":    "Markdown report (always available).",
    "ipynb": "Jupyter notebook with inline figures (re-runnable).",
    "html":  "HTML report with inline figures.",
    "pdf":   "PDF report (embeds figures; needs fpdf).",
}
DEFAULT_FORMAT = "md"


def envelope_text() -> str:
    """Human-readable list of what can be produced here -- for asking the user."""
    lines = ["I can save the report in these formats on this server:"]
    for fmt, note in SAVE_ENVELOPE.items():
        star = " (default)" if fmt == DEFAULT_FORMAT else ""
        lines.append(f"  - {fmt}{star}: {note}")
    return "\n".join(lines)


@dataclass
class ReportPayload:
    """The content to save, format-agnostic."""
    title: str
    body_md: str = ""                       # main narrative, markdown
    findings: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)  # image file paths


@dataclass
class SaveResult:
    format: str
    path: Path
    ok: bool = True
    note: str = ""


def _findings_md(findings: list[str]) -> str:
    if not findings:
        return ""
    return "\n## Findings\n\n" + "\n".join(f"- {f}" for f in findings) + "\n"


def _full_md(p: ReportPayload) -> str:
    parts = [f"# {p.title}\n"]
    if p.body_md.strip():
        parts.append(p.body_md.rstrip() + "\n")
    parts.append(_findings_md(p.findings))
    for fig in p.figures:
        parts.append(f"\n![figure]({fig})\n")
    return "\n".join(x for x in parts if x)


def _render_md(p: ReportPayload, out: Path) -> SaveResult:
    out.write_text(_full_md(p), encoding="utf-8")
    return SaveResult("md", out)


def _render_html(p: ReportPayload, out: Path) -> SaveResult:
    from html import escape as _esc
    body = _esc(_full_md(p), quote=False)
    imgs = "".join(f'<img src="{_esc(str(f))}" style="max-width:100%">'
                   for f in p.figures)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(p.title)}</title></head><body>"
            f"<pre>{body}</pre>{imgs}</body></html>")
    out.write_text(html, encoding="utf-8")
    return SaveResult("html", out)


def _render_ipynb(p: ReportPayload, out: Path) -> SaveResult:
    # Minimal, dependency-free notebook: one markdown cell.
    cell = {"cell_type": "markdown", "metadata": {},
            "source": _full_md(p).splitlines(keepends=True)}
    nb = {"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return SaveResult("ipynb", out)


def _render_pdf(p: ReportPayload, out: Path) -> SaveResult:
    try:
        from fpdf import FPDF
    except Exception:
        # Graceful fallback: write md alongside and say so (honest envelope).
        md = out.with_suffix(".md")
        _render_md(p, md)
        return SaveResult("pdf", md, ok=False,
                          note="fpdf not available; saved markdown instead")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    safe = _full_md(p).encode("latin-1", "replace").decode("latin-1")
    for line in safe.splitlines() or [""]:
        pdf.multi_cell(0, 6, line)
    pdf.output(str(out))
    return SaveResult("pdf", out)


_RENDERERS = {"md": _render_md, "html": _render_html,
              "ipynb": _render_ipynb, "pdf": _render_pdf}


def save_output(payload: ReportPayload, fmt: str = DEFAULT_FORMAT,
                out_dir: str | Path = ".", stem: str | None = None) -> SaveResult:
    """Render ``payload`` into ``fmt`` under ``out_dir``. Unknown fmt -> md."""
    fmt = (fmt or DEFAULT_FORMAT).lower().lstrip(".")
    if fmt not in _RENDERERS:
        fmt = DEFAULT_FORMAT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or _slug(payload.title) or "report"
    out = out_dir / f"{stem}.{fmt}"
    return _RENDERERS[fmt](payload, out)


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s.lower()).strip("_")
