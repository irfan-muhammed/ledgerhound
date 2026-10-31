"""
sentinel.findings.render
========================

Markdown rendering for findings -- used by the notebook-as-journal
output path (no ipywidgets required) and by the CLI when printing a
session summary.

This is intentionally text-only. Charts are referenced by path; the
notebook journal embeds them as image cells separately.
"""

from __future__ import annotations

from .models import Finding, PriorResult, SampleFrame


# Symbols for at-a-glance scanning. Plain ASCII (Windows-safe).
_FRAME_BADGE = {
    SampleFrame.WITHIN_STRATUM:      "[within-stratum]",
    SampleFrame.CROSS_STRATUM:       "[cross-stratum, weighted]",
    SampleFrame.POPULATION_ESTIMATE: "[population estimate]",
    SampleFrame.DERIVED_SAMPLE:      "[derived-sample, bias-flag]",
    SampleFrame.UNSPECIFIED:         "[frame unspecified]",
}

_PRIOR_BADGE = {
    PriorResult.SURVIVES:  "[prior: survives]",
    PriorResult.COLLAPSES: "[prior: collapses]",
    PriorResult.SHIFTS:    "[prior: shifts]",
    PriorResult.UNTESTED:  "",
}


def render_finding_md(f: Finding) -> str:
    """Render a single Finding as a markdown block."""
    parts: list[str] = []
    parts.append(f"### Finding `{f.id}`")
    parts.append("")
    parts.append(f"**Claim.** {f.claim}")
    parts.append("")

    meta_bits = [
        _FRAME_BADGE.get(f.sample_frame, ""),
        _PRIOR_BADGE.get(f.priors_result, ""),
        f"confidence: {f.confidence:.2f}",
    ]
    if f.dataset:
        meta_bits.append(f"dataset: `{f.dataset}`")
    parts.append("  ".join(b for b in meta_bits if b))
    parts.append("")

    if f.sample_frame_note:
        parts.append(f"_Frame note: {f.sample_frame_note}_")
        parts.append("")

    if f.priors_tested:
        parts.append(f"**Priors tested:** {', '.join(f.priors_tested)} -> "
                     f"`{f.priors_result.value}`")
        parts.append("")

    if not f.evidence.is_empty():
        parts.append("**Evidence**")
        if f.evidence.summary:
            parts.append(f"- {f.evidence.summary}")
        if f.evidence.n_rows_examined is not None:
            parts.append(f"- rows examined: {f.evidence.n_rows_examined:,}")
        if f.evidence.query:
            parts.append("```sql")
            parts.append(f.evidence.query.strip())
            parts.append("```")
        if f.evidence.code:
            parts.append("```python")
            parts.append(f.evidence.code.strip())
            parts.append("```")
        if f.evidence.chart_path:
            parts.append(f"![chart]({f.evidence.chart_path})")
        parts.append("")
    else:
        parts.append("> ! No reproducible evidence attached. Treat as opinion.")
        parts.append("")

    if f.derived_from:
        parts.append(f"_Derived from: {', '.join(f.derived_from)}_")
    if f.enables_sampling_for:
        parts.append(f"_Enables sampling for: {', '.join(f.enables_sampling_for)}_  "
                     f"(Phase-2 hook)")
    if f.tags:
        parts.append(f"_Tags: {', '.join(f.tags)}_")

    return "\n".join(parts).rstrip() + "\n"


def render_findings_log_md(findings: list[Finding], title: str = "Findings") -> str:
    """Render a list of findings as a markdown report."""
    if not findings:
        return f"# {title}\n\n_No findings yet._\n"

    out = [f"# {title}", "", f"_{len(findings)} findings_", ""]

    # Summary table -- quick scan of claims.
    out.append("| ID | Claim | Frame | Confidence |")
    out.append("|---|---|---|---|")
    for f in findings:
        claim_short = (f.claim[:80] + "...") if len(f.claim) > 80 else f.claim
        out.append(
            f"| `{f.id}` | {claim_short} | "
            f"{f.sample_frame.value} | {f.confidence:.2f} |"
        )
    out.append("")
    out.append("---")
    out.append("")

    for f in findings:
        out.append(render_finding_md(f))
        out.append("")

    return "\n".join(out)
