"""
ledgerhound.eval.trend
===================

Aggregate every run's ``scorecard.json`` into a trend view so you can see
whether Ledgerhound is improving over time -- the enterprise "is it going in the
right direction" signal, derived purely from per-run scorecards (no golden
data required).

Scans a workspace for ``**/scorecard.json``, orders by run time, and reports
each tracked metric as a time series plus the latest-vs-first delta.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# Metrics we trend (higher is better except where noted).
_TREND_KEYS = [
    ("step_success_rate", "higher"),
    ("grounded_step_rate", "higher"),
    ("claim_groundedness", "higher"),
    ("avg_retries_per_step", "lower"),
    ("degraded_steps", "lower"),
    ("claims_flagged", "lower"),
    ("charts_produced", "higher"),
]

_RUN_TS = re.compile(r"run_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def _run_time(path: Path) -> float:
    """Best-effort run timestamp: parse from run_<ts> dir, else file mtime."""
    m = _RUN_TS.search(str(path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def collect_scorecards(workspace: str | Path) -> list[dict]:
    """Return scorecards ordered oldest->newest:
    [{run_id, when, ...scorecard fields}]."""
    root = Path(workspace)
    rows = []
    for sc_file in root.rglob("scorecard.json"):
        try:
            data = json.loads(sc_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_dir = sc_file.parent
        rows.append({
            "run_id": run_dir.name,
            "path": str(run_dir),
            "_t": _run_time(run_dir),
            **data,
        })
    rows.sort(key=lambda r: r["_t"])
    return rows


def build_trend(scorecards: list[dict]) -> dict:
    """Per-metric series + first/last/delta + direction verdict."""
    out = {"runs": len(scorecards), "metrics": {}}
    for key, better in _TREND_KEYS:
        series = [(r["run_id"], r.get(key)) for r in scorecards if key in r]
        vals = [v for _, v in series if isinstance(v, (int, float))]
        if not vals:
            continue
        first, last = vals[0], vals[-1]
        delta = last - first
        improving = (delta >= 0) if better == "higher" else (delta <= 0)
        out["metrics"][key] = {
            "better": better,
            "first": first,
            "last": last,
            "delta": round(delta, 3),
            "mean": round(sum(vals) / len(vals), 3),
            "n": len(vals),
            "improving": improving if len(vals) > 1 else None,
            "series": series,
        }
    return out


def render_console(trend: dict) -> str:
    lines = [f"Scorecard trend across {trend['runs']} run(s):", ""]
    if trend["runs"] < 1:
        return "No scorecards found. Run `ledgerhound auto` to generate some."
    header = f"  {'metric':<22}{'first':>8}{'last':>8}{'delta':>8}  dir"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    arrows = {True: "improving", False: "regressing", None: "(single run)"}
    for key, m in trend["metrics"].items():
        d = m["delta"]
        sign = "+" if d >= 0 else ""
        verdict = arrows[m["improving"]]
        lines.append(
            f"  {key:<22}{m['first']:>8.2f}{m['last']:>8.2f}"
            f"{sign+format(d,'.2f'):>8}  {verdict} ({m['better']} better)"
        )
    return "\n".join(lines)
