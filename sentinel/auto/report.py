"""
sentinel.auto.report
====================

Builds the final deliverables from a completed run's trace:
  * HTML exec report  (run_dir/report.html)   -- self-contained, plots inline
  * Jupyter notebook  (run_dir/notebook.ipynb) -- narrative + inline plots

Both embed PNG artefacts as base64 so the deliverable is self-contained and
viewable without re-running anything. When an ``llm`` is supplied, a
progressive narrative is generated linking each step to the next
("step 1 found X, which step 2 needs because ...").
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# -- shared helpers --------------------------------------------------------- #

def _artefact_path(run_dir: Path, artefact: str) -> Path:
    """Resolve an artefact (relative to the run's scratch/outputs dir)."""
    p = Path(artefact)
    if p.is_absolute():
        return p
    return run_dir / "outputs" / artefact


def _is_image(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))


def _b64_image(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -- progressive narrative (LLM-powered, optional) -------------------------- #

_NARRATIVE_PROMPT = """\
You are documenting a data-analysis pipeline so a reader understands the STORY,
not just the code. Below are the executed steps with their outputs.

For EACH step write three short parts (1-2 sentences each):
  "did"        : what this step actually did, in plain language
  "learned"    : the key result or insight -- CITE concrete numbers/columns from
                 the output. If the step failed, say what blocked it.
  "feeds_next" : how this result is used by, or sets up, the following step.
                 For the last step, describe how it completes the goal.

Goal: {goal}

Steps:
{steps_block}

Respond ONLY with a JSON object keyed by step number (as string), e.g.:
{{"1": {{"did": "...", "learned": "...", "feeds_next": "..."}}, "2": {{...}}}}
No markdown fences, JSON only.
"""

_SUMMARY_PROMPT = """\
Write a 4-6 sentence executive summary of this analysis for a business reader.
State what was investigated, the most important findings (with numbers), and
any caveats (failed/degraded steps). Plain prose, no bullet lists.

Goal: {goal}

Step outcomes:
{steps_block}
"""


def _build_narrative(goal: str, trace: list[dict], llm) -> dict[str, dict]:
    """Return {step_num_str: {did, learned, feeds_next}}. Falls back to a
    templated narrative when no LLM or on any failure."""
    fallback = {}
    for i, e in enumerate(trace, 1):
        out = (e.get("output") or e.get("error") or "").strip()
        fallback[str(i)] = {
            "did": e.get("title", f"Step {i}"),
            "learned": (out[:200] + "...") if len(out) > 200 else (out or "(no output)"),
            "feeds_next": "Provides context for subsequent steps.",
        }
    if llm is None:
        return fallback

    steps_block = "\n".join(
        f"[{i}] {e.get('title','')}\n"
        f"    description: {e.get('description','')}\n"
        f"    success: {e.get('success')}  degraded: {e.get('degraded', False)}\n"
        f"    output: {((e.get('output') or e.get('error') or '')[:600]).strip()}"
        for i, e in enumerate(trace, 1)
    )
    try:
        resp = llm.generate([{"role": "user", "content":
                              _NARRATIVE_PROMPT.format(goal=goal, steps_block=steps_block)}])
        raw = resp.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            # Merge over fallback so any missing step still has content
            for k, v in data.items():
                if isinstance(v, dict):
                    fallback[str(k)] = {
                        "did": v.get("did", fallback.get(str(k), {}).get("did", "")),
                        "learned": v.get("learned", ""),
                        "feeds_next": v.get("feeds_next", ""),
                    }
    except Exception:
        pass
    return fallback


def _build_summary(goal: str, trace: list[dict], llm) -> str:
    succeeded = sum(1 for e in trace if e.get("success"))
    degraded = sum(1 for e in trace if e.get("degraded"))
    base = (f"Executed {len(trace)} steps toward the goal; "
            f"{succeeded} succeeded"
            + (f", {degraded} produced only degraded fallback output" if degraded else "")
            + ".")
    if llm is None:
        return base
    steps_block = "\n".join(
        f"[{i}] {e.get('title','')} -> "
        f"{'OK' if e.get('success') else 'FAILED'}"
        f"{' (degraded)' if e.get('degraded') else ''}: "
        f"{((e.get('output') or e.get('error') or '')[:300]).strip()}"
        for i, e in enumerate(trace, 1)
    )
    try:
        resp = llm.generate([{"role": "user", "content":
                              _SUMMARY_PROMPT.format(goal=goal, steps_block=steps_block)}])
        text = resp.content.strip()
        return text or base
    except Exception:
        return base


# -- environment-aware audit ------------------------------------------------ #

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _collect_metric_values(trace: list[dict]) -> list[float]:
    vals: list[float] = []
    for e in trace:
        for v in (e.get("metrics") or {}).values():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    return vals


def _nums_in(text: str) -> list[tuple[str, float]]:
    out = []
    for tok in _NUM_RE.findall(text or ""):
        raw = tok.replace(",", "")
        try:
            out.append((tok, float(raw)))
        except ValueError:
            continue
    return out


def audit_claims(summary: str, findings: list[str], trace: list[dict]) -> dict:
    """Cross-check every number in the LLM narrative against the COMPUTED
    metrics (which came from executing code in the kernel) and the raw step
    outputs. A number that matches no computed value is flagged as unverified.

    This is the environment-aware integrity check: claims are validated against
    what the data actually produced, not against the model's own prose.
    """
    metric_vals = _collect_metric_values(trace)
    output_blob = " ".join((e.get("output") or "") for e in trace)
    claim_text = summary + " " + " ".join(findings)

    verified, flagged = 0, []
    seen = set()
    for tok, num in _nums_in(claim_text):
        if tok in seen:
            continue
        seen.add(tok)
        # ignore trivial small integers (years handled separately, 0/1/2 noise)
        if num != num:  # NaN
            continue
        ok = False
        # match against computed metrics (relative tolerance), trying the
        # number as-is AND its percent<->ratio forms (98.6 vs 0.986).
        candidates = (num, num / 100.0, num * 100.0)
        for mv in metric_vals:
            denom = max(abs(mv), 1e-9)
            if any(abs(mv - c) / denom <= 0.02 or abs(mv - c) <= 0.5
                   for c in candidates):
                ok = True
                break
        # or appears verbatim in a real step output (kernel-produced)
        if not ok and (tok in output_blob or str(int(num)) in output_blob
                       if num == int(num) else False):
            ok = True
        if ok:
            verified += 1
        else:
            flagged.append(tok)
    total = verified + len(flagged)
    return {
        "verified": verified,
        "total": total,
        "flagged": flagged[:20],
        "rate": (verified / total) if total else 1.0,
    }


def _audit_html(audit: dict) -> str:
    if not audit or audit.get("total", 0) == 0:
        return ""
    rate = audit["rate"]
    color = "#27ae60" if rate >= 0.9 else ("#e67e22" if rate >= 0.7 else "#e74c3c")
    flagged = ""
    if audit["flagged"]:
        flagged = ("<p class='meta'>Unverified figures (not found in computed "
                   "metrics or step outputs): "
                   + ", ".join(f"<code>{_esc(x)}</code>" for x in audit["flagged"])
                   + "</p>")
    return (
        f"<h2 id='audit'>Verification</h2>"
        f"<div class='audit' style='border-left-color:{color}'>"
        f"<p><strong>{audit['verified']}/{audit['total']}</strong> numeric claims "
        f"in the summary &amp; findings reproduce from values the analysis actually "
        f"computed ({rate:.0%} grounded).</p>{flagged}</div>"
    )


# -- session-derived content ------------------------------------------------ #

_NON_FINDING = (
    "no output", "saved:", "saved to", "report saved", "loading", "loaded ",
    "analyzing", "processing", "/home/", "/sentinel", ".html", ".parquet",
    "writing", "exporting", "executed",
)


def _is_real_finding(text: str) -> bool:
    t = text.lower()
    if any(tok in t for tok in _NON_FINDING):
        return False
    return any(ch.isdigit() for ch in t) or len(t) > 40


def _collect_findings(session) -> list[str]:
    """Durable findings recorded during the run, with log-noise filtered out."""
    if session is None:
        return []
    try:
        msgs = session.load_messages()
    except Exception:
        return []
    raw = [m.get("content", "") for m in msgs if m.get("tag") == "finding"]
    return [r for r in raw if _is_real_finding(r)]


_KEYFIND_PROMPT = """\
You are writing the "Key Findings" section of a data-analysis report. Below are
the executed steps and their outputs. Extract the 5-10 MOST IMPORTANT, concrete
findings a decision-maker needs. Each must:
  - state a real discovery (not a progress message or file path)
  - cite specific numbers / columns / percentages from the outputs
  - be one sentence

Goal: {goal}

Step outputs:
{steps_block}

Respond as a plain bullet list (one finding per line, starting with "- ").
No preamble, no headings.
"""


def _build_key_findings(goal: str, trace: list[dict], llm, session) -> list[str]:
    """High-quality findings via ONE report-time LLM call (no per-step cost).

    Falls back to the filtered per-step findings recorded during the run.
    """
    fallback = _collect_findings(session)
    if llm is None:
        return fallback
    steps_block = "\n".join(
        f"[{i}] {e.get('title','')}: "
        f"{((e.get('output') or e.get('error') or '')[:400]).strip()}"
        for i, e in enumerate(trace, 1)
        if (e.get("output") or e.get("error"))
    )
    if not steps_block.strip():
        return fallback
    try:
        resp = llm.generate([{"role": "user", "content":
                              _KEYFIND_PROMPT.format(goal=goal, steps_block=steps_block)}])
        lines = [l.strip().lstrip("-* ").strip()
                 for l in resp.content.splitlines() if l.strip()]
        found = [l for l in lines if len(l) > 15 and not l.lower().startswith(("here", "key find"))]
        return found or fallback
    except Exception:
        return fallback


def _collect_images(run_dir: Path, trace: list[dict]) -> list[tuple[int, str, str]]:
    """Return (step_num, artefact_name, base64) for every plot produced."""
    out = []
    for i, e in enumerate(trace, 1):
        for art in e.get("artefacts", []):
            if _is_image(art):
                b64 = _b64_image(_artefact_path(run_dir, art))
                if b64:
                    out.append((i, art, b64))
    return out


def _data_overview_html(session) -> str:
    if session is None or not getattr(session, "datasets", None):
        return ""
    rows = ""
    for d in session.datasets:
        rows += (f"<tr><td>{_esc(d.name)}</td>"
                 f"<td>{d.rows if d.rows is not None else '?':,}</td>"
                 f"<td>{d.cols if d.cols is not None else '?'}</td>"
                 f"<td>{_esc(d.format or '')}</td>"
                 f"<td>{_esc((d.description or '')[:160])}</td></tr>"
                 if isinstance(d.rows, int) else
                 f"<tr><td>{_esc(d.name)}</td><td>{d.rows or '?'}</td>"
                 f"<td>{d.cols or '?'}</td><td>{_esc(d.format or '')}</td>"
                 f"<td>{_esc((d.description or '')[:160])}</td></tr>")
    return (
        "<h2 id='overview'>Data overview</h2>"
        "<table class='dtable'><thead><tr>"
        "<th>Dataset</th><th>Rows</th><th>Cols</th><th>Format</th><th>Description</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _findings_html(findings: list[str]) -> str:
    if not findings:
        return ""
    items = "".join(f"<li>{_esc(f)}</li>" for f in findings)
    return ("<h2 id='findings'>Key findings</h2>"
            "<div class='findings'><ul>" + items + "</ul></div>")


def _methodology_html(trace: list[dict]) -> str:
    if not trace:
        return ""
    items = ""
    for i, e in enumerate(trace, 1):
        desc = _esc((e.get("description") or "")[:200])
        items += (f"<li><strong>{_esc(e.get('title',''))}</strong>"
                  + (f" -- {desc}" if desc else "") + "</li>")
    return "<h2 id='methodology'>Methodology</h2><ol class='method'>" + items + "</ol>"


def _gallery_html(images: list[tuple[int, str, str]]) -> str:
    if not images:
        return ""
    figs = "".join(
        f'<figure><img src="data:image/png;base64,{b64}" alt="{_esc(name)}"/>'
        f'<figcaption>Step {step}: {_esc(name)}</figcaption></figure>'
        for step, name, b64 in images
    )
    return ("<h2 id='viz'>Visualizations</h2>"
            f"<p class='meta'>{len(images)} chart(s) produced across the run.</p>"
            "<div class='gallery'>" + figs + "</div>")


# -- HTML ------------------------------------------------------------------- #

def build_html(run_dir: Path, goal: str, trace: list[dict], llm=None,
               session=None) -> Path:
    narrative = _build_narrative(goal, trace, llm)
    summary = _build_summary(goal, trace, llm)
    findings = _build_key_findings(goal, trace, llm, session)
    images = _collect_images(run_dir, trace)
    audit = audit_claims(summary, findings, trace)

    # Table of contents -- top-level sections + per-step
    toc = ""
    if session is not None and getattr(session, "datasets", None):
        toc += '<li><a href="#overview">Data overview</a></li>'
    if findings:
        toc += '<li><a href="#findings">Key findings</a></li>'
    if audit.get("total", 0):
        toc += '<li><a href="#audit">Verification</a></li>'
    toc += '<li><a href="#methodology">Methodology</a></li>'
    if images:
        toc += '<li><a href="#viz">Visualizations</a></li>'
    toc += "".join(
        f'<li><a href="#step{i}">Step {i}: {_esc(e.get("title",""))}</a> '
        f'{"OK" if e.get("success") else "X"}'
        f'{" (degraded)" if e.get("degraded") else ""}</li>'
        for i, e in enumerate(trace, 1)
    )

    steps_html = ""
    for i, entry in enumerate(trace, 1):
        status_color = "#27ae60" if entry.get("success") else "#e74c3c"
        if entry.get("degraded"):
            status_color = "#e67e22"
        status_label = "OK" if entry.get("success") else "FAILED"
        if entry.get("degraded"):
            status_label = "DEGRADED"

        n = narrative.get(str(i), {})
        narrative_html = ""
        if n:
            narrative_html = f"""
            <div class="narrative">
              <p><span class="lbl">What it did:</span> {_esc(n.get('did',''))}</p>
              <p><span class="lbl">What we learned:</span> {_esc(n.get('learned',''))}</p>
              <p><span class="lbl">Why it matters next:</span> {_esc(n.get('feeds_next',''))}</p>
            </div>"""

        # Inline images
        imgs_html = ""
        for art in entry.get("artefacts", []):
            if _is_image(art):
                b64 = _b64_image(_artefact_path(run_dir, art))
                if b64:
                    imgs_html += (f'<figure><img src="data:image/png;base64,{b64}" '
                                  f'alt="{_esc(art)}"/>'
                                  f'<figcaption>{_esc(art)}</figcaption></figure>')

        non_img = [a for a in entry.get("artefacts", []) if not _is_image(a)]
        art_html = ""
        if non_img:
            art_html = "<p><strong>Files:</strong> " + ", ".join(
                f"<code>{_esc(a)}</code>" for a in non_img) + "</p>"

        output = _esc(entry.get("output", ""))
        error = _esc(entry.get("error", ""))
        retries = entry.get("retries", 0)
        retry_note = (f" <em>(fixed after {retries} retr"
                      f"{'y' if retries == 1 else 'ies'})</em>") if retries else ""

        steps_html += f"""
        <div class="step" id="step{i}">
          <div class="step-header">
            <span class="step-num">Step {i}</span>
            <span class="step-title">{_esc(entry.get('title',''))}</span>
            <span class="step-status" style="background:{status_color}">{status_label}{retry_note}</span>
          </div>
          {narrative_html}
          {imgs_html}
          <details>
            <summary>Code</summary>
            <pre><code>{_esc(entry.get('code',''))}</code></pre>
          </details>
          {f'<pre class="output">{output}</pre>' if output else ''}
          {f'<pre class="error">{error}</pre>' if error else ''}
          {art_html}
        </div>"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    succeeded = sum(1 for e in trace if e.get("success"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Sentinel Auto -- {now}</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;color:#222;line-height:1.55}}
  h1{{color:#1a252f}} h2{{color:#1a252f;border-bottom:2px solid #eee;padding-bottom:4px}}
  .meta{{color:#555;font-size:.9em}}
  .summary-box{{background:#eaf4fb;border-radius:8px;padding:20px;margin:24px 0}}
  .toc{{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:12px 24px;margin:16px 0}}
  .toc ul{{margin:6px 0;padding-left:18px}} .toc li{{margin:3px 0}}
  .step{{border:1px solid #ddd;border-radius:8px;margin:20px 0;padding:18px}}
  .step-header{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
  .step-num{{background:#1a252f;color:#fff;border-radius:4px;padding:2px 8px;font-size:.8em}}
  .step-title{{font-weight:600;flex:1;font-size:1.05em}}
  .step-status{{border-radius:4px;padding:2px 8px;color:#fff;font-size:.8em}}
  .narrative{{background:#f7fbff;border-left:4px solid #3498db;padding:10px 16px;margin:12px 0;border-radius:0 6px 6px 0}}
  .narrative p{{margin:6px 0}} .lbl{{font-weight:600;color:#2c3e50}}
  figure{{margin:14px 0;text-align:center}} figure img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
  figcaption{{font-size:.8em;color:#777;margin-top:4px}}
  pre{{background:#f5f5f5;padding:12px;border-radius:4px;overflow-x:auto;font-size:.85em}}
  pre.error{{background:#fff0f0;border-left:4px solid #e74c3c}}
  pre.output{{background:#f0fff4;border-left:4px solid #27ae60}}
  details summary{{cursor:pointer;color:#2980b9;margin:8px 0}}
  table.dtable{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}}
  table.dtable th,table.dtable td{{border:1px solid #e3e3e3;padding:6px 10px;text-align:left}}
  table.dtable th{{background:#f3f6f9}}
  .findings{{background:#fffaf0;border-left:4px solid #e67e22;padding:8px 20px;border-radius:0 6px 6px 0}}
  .findings li{{margin:6px 0}}
  .audit{{background:#f5fbf6;border-left:4px solid #27ae60;padding:8px 20px;border-radius:0 6px 6px 0}}
  ol.method li{{margin:5px 0}}
  .gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
  .gallery figure{{margin:0}}
</style>
</head>
<body>
<h1>Sentinel Auto -- Analysis Report</h1>
<p class="meta">Generated: {now} | Steps: {len(trace)} | Succeeded: {succeeded}/{len(trace)} | Charts: {len(images)}</p>
<div class="summary-box">
  <strong>Goal:</strong> {_esc(goal)}
  <hr style="border:none;border-top:1px solid #cfe2f3;margin:12px 0">
  <strong>Executive summary:</strong><br>{_esc(summary)}
</div>
<div class="toc"><strong>Contents</strong><ul>{toc}</ul></div>
{_data_overview_html(session)}
{_findings_html(findings)}
{_audit_html(audit)}
{_methodology_html(trace)}
{_gallery_html(images)}
<h2>Step-by-step analysis</h2>
{steps_html}
</body></html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


# -- Notebook --------------------------------------------------------------- #

def build_notebook(run_dir: Path, goal: str, trace: list[dict], llm=None,
                   session=None) -> Path:
    narrative = _build_narrative(goal, trace, llm)
    summary = _build_summary(goal, trace, llm)
    findings = _build_key_findings(goal, trace, llm, session)
    cells: list[dict] = []
    _cid = {"n": 0}

    def _next_id() -> str:
        _cid["n"] += 1
        return f"cell-{_cid['n']:03d}"

    def md_cell(source: str) -> dict:
        return {"cell_type": "markdown", "id": _next_id(),
                "metadata": {}, "source": source}

    def code_cell(source: str, outputs: list | None = None) -> dict:
        return {"cell_type": "code", "id": _next_id(), "execution_count": None,
                "metadata": {}, "outputs": outputs or [], "source": source}

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Title + summary + TOC
    toc_lines = "\n".join(
        f"{i}. [{e.get('title','')}](#Step-{i}) "
        f"{'OK' if e.get('success') else 'X'}"
        f"{' (degraded)' if e.get('degraded') else ''}"
        for i, e in enumerate(trace, 1)
    )
    cells.append(md_cell(
        f"# Sentinel Auto -- Analysis Notebook\n\n"
        f"**Goal:** {goal}\n\n"
        f"_Generated {now}_\n\n"
        f"## Executive summary\n\n{summary}\n\n"
        f"## Contents\n\n{toc_lines}\n\n"
        f"> This notebook is a narrated, reproducible record. Each step shows "
        f"**what it did**, **what we learned**, and **why the next step needs it**. "
        f"Plots are embedded inline; re-run any cell to reproduce.\n"
    ))

    # Data overview (from session)
    if session is not None and getattr(session, "datasets", None):
        ov = ["## Data overview\n", "| Dataset | Rows | Cols | Format | Description |",
              "|---|---|---|---|---|"]
        for d in session.datasets:
            ov.append(f"| {d.name} | {d.rows if d.rows is not None else '?'} | "
                      f"{d.cols if d.cols is not None else '?'} | {d.format or ''} | "
                      f"{(d.description or '')[:120]} |")
        cells.append(md_cell("\n".join(ov) + "\n"))

    # Key findings (consolidated sacred memory)
    if findings:
        fl = "\n".join(f"- {f}" for f in findings)
        cells.append(md_cell(f"## Key findings\n\n{fl}\n"))

    # Methodology
    meth = ["## Methodology\n"]
    for i, e in enumerate(trace, 1):
        d = (e.get("description") or "").strip()
        meth.append(f"{i}. **{e.get('title','')}**" + (f" -- {d}" if d else ""))
    cells.append(md_cell("\n".join(meth) + "\n"))

    cells.append(code_cell(
        "# Setup -- shared imports. Re-run cells top-to-bottom to reproduce.\n"
        "from pathlib import Path\n"
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "try:\n"
        "    import seaborn as sns\n"
        "except ImportError:\n"
        "    sns = None\n"
        "run_dir = Path('.')  # adjust if running outside the run directory\n"
    ))

    for i, entry in enumerate(trace, 1):
        status = "OK" if entry.get("success") else "X"
        if entry.get("degraded"):
            status = "! degraded"
        n = narrative.get(str(i), {})

        # Narrative markdown -- the progressive knowledge thread
        md = [f"## Step {i}: {entry.get('title','')} {status}\n"]
        if entry.get("description"):
            md.append(f"_{entry['description']}_\n")
        if n.get("did"):
            md.append(f"**What this step does.** {n['did']}\n")
        if n.get("learned"):
            md.append(f"**What we learned.** {n['learned']}\n")
        if n.get("feeds_next"):
            md.append(f"**Why this matters for what's next.** {n['feeds_next']}\n")
        retries = entry.get("retries", 0)
        if retries:
            md.append(f"\n<sub>Self-healed after {retries} fix "
                      f"attempt{'s' if retries != 1 else ''}.</sub>\n")
        cells.append(md_cell("\n".join(md)))

        # Code cell with embedded outputs (stream text + inline images)
        nb_outputs: list[dict] = []
        output_text = entry.get("output", "") or entry.get("error", "")
        if output_text:
            nb_outputs.append({
                "output_type": "stream", "name": "stdout",
                "text": output_text,
            })
        for art in entry.get("artefacts", []):
            if _is_image(art):
                b64 = _b64_image(_artefact_path(run_dir, art))
                if b64:
                    nb_outputs.append({
                        "output_type": "display_data",
                        "data": {"image/png": b64},
                        "metadata": {},
                    })
        cells.append(code_cell(entry.get("code", ""), nb_outputs))

    nb: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    out = run_dir / "notebook.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out
