"""
ledgerhound.eval.fluidity_dashboard
====================================

V3-028 -- the "markdown dashboard per run + trend file" half of the item,
extending V3-004's fluidity metrics (ledgerhound.eval.fluidity) with the
part that never existed: a way to see MOVEMENT -- this run vs. the V3-004
baseline, and this run vs. every prior run.

Two artifacts, deliberately separate:
  - a per-run DASHBOARD (render_dashboard): this run's numbers plus a
    delta against one fixed reference point (the V3-004 baseline,
    persisted at eval/results/fluidity_baseline_v3004.json since the
    original artifact, fluidity_post-V3-001-002.md, is markdown-only and
    not machine-diffable).
  - a TREND file (append_trend / render_trend_summary): one JSON line per
    run, appended forever, so "per-epic metric movement" can be read
    across the whole history, not just against one fixed baseline.
    Mirrors ledgerhound.eval.trend's build_trend()/render_console() shape,
    but for fluidity metrics specifically -- confirmed before writing
    code that eval/trend.py's _TREND_KEYS covers AutoRunner/scorecard.json
    metrics (step_success_rate, grounded_step_rate, ...), none of which
    are fluidity metrics; nothing existing trends this metric family.

"Every subsequent PR links its dashboard delta" (the roadmap's own
close-out line) is a PROCESS convention, not something code can enforce
by itself -- documented in eval/results/README.md, not built as a git
hook/CI gate in this closure (see the v3 roadmap's V3-028 closure for the
explicit disclosure).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ledgerhound.eval.fluidity import FluidityResult, render_report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FluidityBaseline:
    """Aggregate fluidity numbers for one run -- the unit both the
    dashboard's delta and the trend file's rows are built from."""
    label: str
    n_tasks: int
    avg_questions_per_task: float
    avg_redundant_recomputes: float
    avg_unrecovered_errors: float
    error_recovery_rate: Optional[float]
    completed_fraction: float
    avg_turns: float
    generated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FluidityBaseline":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def aggregate(results: list[FluidityResult], label: str = "") -> FluidityBaseline:
    n = len(results) or 1
    totals = FluidityResult()
    for r in results:
        totals.questions_asked += r.questions_asked
        totals.redundant_recomputes += r.redundant_recomputes
        totals.unrecovered_errors += r.unrecovered_errors
        totals.total_distinct_errors += r.total_distinct_errors
        totals.turns += r.turns
    completed_n = sum(1 for r in results if r.completed)
    return FluidityBaseline(
        label=label, n_tasks=len(results),
        avg_questions_per_task=totals.questions_asked / n,
        avg_redundant_recomputes=totals.redundant_recomputes / n,
        avg_unrecovered_errors=totals.unrecovered_errors / n,
        error_recovery_rate=totals.error_recovery_rate,
        completed_fraction=completed_n / n,
        avg_turns=totals.turns / n,
    )


def save_baseline(baseline: FluidityBaseline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")


def load_baseline(path: str | Path) -> Optional[FluidityBaseline]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return FluidityBaseline.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _delta_line(name: str, current: float, baseline: Optional[float],
                lower_is_better: bool = True) -> str:
    if baseline is None:
        return f"- {name}: {current:.2f} (no baseline to compare)"
    diff = current - baseline
    if abs(diff) < 1e-9:
        arrow = "="
    elif (diff < 0) == lower_is_better:
        arrow = "better"
    else:
        arrow = "worse"
    sign = "+" if diff >= 0 else ""
    return f"- {name}: {current:.2f} (baseline {baseline:.2f}, {sign}{diff:.2f}, {arrow})"


def render_dashboard(
    results: list[FluidityResult],
    baseline: Optional[FluidityBaseline] = None,
    label: str = "",
) -> str:
    """Per-run markdown dashboard: the existing render_report() table, plus
    a delta-vs-baseline section (the roadmap's "dashboard shows per-epic
    metric movement from the V3-004 baseline")."""
    current = aggregate(results, label=label)
    lines = [render_report(results, label=label), "", "## vs. baseline"]
    if baseline is None:
        lines.append("(no baseline available -- nothing to compare against)")
    else:
        lines.append(f"Baseline: `{baseline.label}` ({baseline.generated_at})")
        lines.append("")
        lines.append(_delta_line("avg questions/task", current.avg_questions_per_task,
                                 baseline.avg_questions_per_task, lower_is_better=True))
        lines.append(_delta_line("avg redundant recomputes/task",
                                 current.avg_redundant_recomputes,
                                 baseline.avg_redundant_recomputes, lower_is_better=True))
        lines.append(_delta_line("avg unrecovered errors/task",
                                 current.avg_unrecovered_errors,
                                 baseline.avg_unrecovered_errors, lower_is_better=True))
        if current.error_recovery_rate is not None and baseline.error_recovery_rate is not None:
            lines.append(_delta_line("error recovery rate", current.error_recovery_rate,
                                     baseline.error_recovery_rate, lower_is_better=False))
        else:
            lines.append("- error recovery rate: n/a for this comparison "
                         "(no errors occurred in the current run, the baseline, or both)")
        lines.append(_delta_line("completed fraction", current.completed_fraction,
                                 baseline.completed_fraction, lower_is_better=False))
        lines.append(_delta_line("avg turns/task", current.avg_turns,
                                 baseline.avg_turns, lower_is_better=True))
    return "\n".join(lines)


def append_trend(baseline_row: FluidityBaseline, trend_path: str | Path) -> None:
    """Append one run's aggregate numbers to the trend file (JSONL,
    append-only -- same "immutable audit substrate" reasoning as
    FindingsLog: a trend file that could be silently rewritten isn't a
    trend you can trust)."""
    trend_path = Path(trend_path)
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trend_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(baseline_row.to_dict(), default=str) + "\n")


def load_trend(trend_path: str | Path) -> list[FluidityBaseline]:
    trend_path = Path(trend_path)
    if not trend_path.exists():
        return []
    out = []
    for line in trend_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(FluidityBaseline.from_dict(json.loads(line)))
        except Exception:
            continue
    return out


def render_trend_summary(trend_path: str | Path) -> str:
    """Markdown summary of movement across every recorded run, oldest to
    newest -- first vs. last, mirroring ledgerhound.eval.trend.build_trend's
    first/last/delta shape but for the fluidity metric family."""
    rows = load_trend(trend_path)
    if not rows:
        return "# Fluidity trend\n\n(no runs recorded yet)"
    lines = ["# Fluidity trend", "",
            f"{len(rows)} run(s) recorded, oldest to newest.", "",
            "| run | tasks | avg questions | avg redundant | "
            "avg unrecovered | recovery rate | completed | avg turns |",
            "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        rate_str = f"{row.error_recovery_rate:.2f}" if row.error_recovery_rate is not None else "n/a"
        lines.append(f"| {row.label or row.generated_at} | {row.n_tasks} | "
                     f"{row.avg_questions_per_task:.2f} | "
                     f"{row.avg_redundant_recomputes:.2f} | "
                     f"{row.avg_unrecovered_errors:.2f} | {rate_str} | "
                     f"{row.completed_fraction:.0%} | {row.avg_turns:.2f} |")
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        lines += ["", "## First vs. latest"]
        lines.append(_delta_line("avg questions/task", last.avg_questions_per_task,
                                 first.avg_questions_per_task, lower_is_better=True))
        lines.append(_delta_line("avg redundant recomputes/task",
                                 last.avg_redundant_recomputes,
                                 first.avg_redundant_recomputes, lower_is_better=True))
        lines.append(_delta_line("completed fraction", last.completed_fraction,
                                 first.completed_fraction, lower_is_better=False))
    return "\n".join(lines)
