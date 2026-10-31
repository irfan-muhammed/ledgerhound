"""
sentinel.eval.scorecard
=======================

Per-run scorecard -- computed from a run's trace alone (no golden data needed),
so every run produces trackable trajectory metrics. Written to
``<run_dir>/scorecard.json`` at finalize and surfaced in the report.

This is the "is it going in the right direction" signal you can trend across
runs even before the golden dataset exists.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_run_scorecard(trace: list[dict], audit: dict | None = None) -> dict:
    """Trajectory + integrity metrics derived purely from the run trace."""
    n = len(trace)
    succeeded = sum(1 for e in trace if e.get("success"))
    degraded = sum(1 for e in trace if e.get("degraded"))
    retries = sum(e.get("retries", 0) for e in trace)
    grounded_steps = sum(1 for e in trace if e.get("metrics"))
    charts = sum(1 for e in trace for a in e.get("artefacts", [])
                 if str(a).lower().endswith((".png", ".jpg", ".jpeg")))
    total_ms = round(sum(e.get("execution_time_ms", 0) for e in trace), 1)

    sc = {
        "steps": n,
        "step_success_rate": round(succeeded / n, 3) if n else 0.0,
        "degraded_steps": degraded,
        "total_retries": retries,
        "avg_retries_per_step": round(retries / n, 2) if n else 0.0,
        "grounded_step_rate": round(grounded_steps / n, 3) if n else 0.0,
        "charts_produced": charts,
        "total_exec_ms": total_ms,
    }
    if audit:
        sc["claim_groundedness"] = round(audit.get("rate", 1.0), 3)
        sc["claims_verified"] = audit.get("verified", 0)
        sc["claims_total"] = audit.get("total", 0)
        sc["claims_flagged"] = len(audit.get("flagged", []))
    return sc


def write_run_scorecard(run_dir: Path, scorecard: dict) -> Path:
    out = Path(run_dir) / "scorecard.json"
    out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    return out


def scorecard_summary(sc: dict) -> str:
    parts = [
        f"steps={sc.get('steps')}",
        f"success={sc.get('step_success_rate', 0):.0%}",
        f"grounded_steps={sc.get('grounded_step_rate', 0):.0%}",
        f"retries={sc.get('total_retries')}",
        f"charts={sc.get('charts_produced')}",
    ]
    if "claim_groundedness" in sc:
        parts.append(f"claims_grounded={sc['claim_groundedness']:.0%}")
    return " | ".join(parts)
