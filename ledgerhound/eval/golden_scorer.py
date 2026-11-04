"""
ledgerhound.eval.golden_scorer
===========================

Score a Ledgerhound run against a golden task. The cornerstone is
RECOMPUTATION: each golden fact's `computation` is executed against the
data to establish ground truth, then we check whether the run actually
produced that value (in its per-step `metrics`). Groundedness is measured
by recomputation, never by reading prose.

Metrics produced (per task):
  groundedness     fraction of facts the run's metrics matched (within tol)
  fact_pass_rate   fraction of facts whose computation reproduced the
                   expected_value (data/ground-truth integrity)
  finding_recall   fraction of expected_findings matched in the run
  hallucination    count of must_not_claim patterns found in the report
  trajectory       fraction of required capabilities present in the plan
  completeness      deliverables satisfied (charts, sections)
  operational      cost/latency/retries summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


@dataclass
class Scorecard:
    task_id: str
    groundedness: float = 0.0
    fact_pass_rate: float = 0.0
    finding_recall: float = 0.0
    hallucinations: int = 0
    trajectory: float = 0.0
    completeness: float = 0.0
    fact_detail: list[dict] = field(default_factory=list)
    finding_detail: list[dict] = field(default_factory=list)
    hallucination_detail: list[dict] = field(default_factory=list)
    operational: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (f"[{self.task_id}] grounded={self.groundedness:.0%} "
                f"facts={self.fact_pass_rate:.0%} recall={self.finding_recall:.0%} "
                f"halluc={self.hallucinations} traj={self.trajectory:.0%} "
                f"complete={self.completeness:.0%}")


def _within_tol(actual, expected, tol: float) -> bool:
    try:
        a = float(actual); e = float(expected)
    except (TypeError, ValueError):
        return str(actual).strip() == str(expected).strip()
    return abs(a - e) <= (tol if tol else 1e-9)


def _recompute_fact(fact, df) -> object:
    """Run the fact's computation against df; return computed `result`."""
    ns = {"df": df}
    try:
        import pandas as pd, numpy as np  # noqa
        ns.update({"pd": pd, "np": np})
    except Exception:
        pass
    exec(fact.computation, ns)        # noqa: S102 -- trusted golden computations
    return ns.get("result")


def _all_run_metrics(trace: list[dict]) -> dict:
    """Flatten every step's metrics into one name->value dict (last wins)."""
    merged = {}
    for e in trace:
        m = e.get("metrics") or {}
        if isinstance(m, dict):
            merged.update(m)
    return merged


def score_run(task, trace: list[dict], df=None,
              report_text: str = "", plan: list[dict] | None = None) -> Scorecard:
    """Score one golden task against a completed run.

    task        GoldenTask
    trace       list of step entries (each may carry a `metrics` dict)
    df          loaded dataframe for fact recomputation (optional but needed
                for groundedness/fact_pass_rate)
    report_text concatenated report/findings text (for hallucination scan)
    plan        the executed plan (for trajectory scoring)
    """
    sc = Scorecard(task_id=task.id)
    run_metrics = _all_run_metrics(trace)
    run_values = [v for v in run_metrics.values()]

    # -- Facts: recompute truth, check run reproduced it --
    fact_pass = grounded = 0
    for fact in task.facts:
        truth = None
        recomputed_ok = None
        if df is not None:
            try:
                truth = _recompute_fact(fact, df)
                recomputed_ok = _within_tol(truth, fact.expected_value, fact.tolerance)
            except Exception as ex:
                sc.notes.append(f"fact {fact.id} computation error: {ex}")
        # ground truth integrity
        if recomputed_ok:
            fact_pass += 1
        # did the run produce this value anywhere in its metrics?
        target = truth if truth is not None else fact.expected_value
        matched = any(_within_tol(v, target, fact.tolerance or 1e-6)
                      for v in run_values)
        if matched:
            grounded += 1
        sc.fact_detail.append({
            "id": fact.id, "expected": fact.expected_value,
            "recomputed": truth, "recompute_pass": recomputed_ok,
            "run_reproduced": matched,
        })
    n_facts = max(len(task.facts), 1)
    sc.fact_pass_rate = fact_pass / n_facts
    sc.groundedness = grounded / n_facts

    # -- Findings recall (keyword + linked-fact match) --
    findings_blob = report_text.lower()
    matched_f = 0
    for gf in task.expected_findings:
        terms = [t.lower() for t in gf.must_include_terms]
        ok = all(t in findings_blob for t in terms) if terms else False
        sc.finding_detail.append({"text": gf.text, "matched": ok,
                                  "importance": gf.importance})
        if ok:
            matched_f += 1
    if task.expected_findings:
        sc.finding_recall = matched_f / len(task.expected_findings)

    # -- Hallucination: must_not_claim patterns present in report --
    for trap in task.must_not_claim:
        pat = trap.get("pattern", "")
        if pat and re.search(pat, report_text, re.IGNORECASE):
            sc.hallucinations += 1
            sc.hallucination_detail.append(
                {"pattern": pat, "reason": trap.get("reason", "")})

    # -- Trajectory: required capabilities present in the plan --
    plan = plan or []
    plan_text = " ".join((s.get("title", "") + " " + s.get("description", ""))
                         for s in plan).lower()
    req = [c for c in task.expected_capabilities if c.get("required")]
    if req:
        present = sum(1 for c in req
                      if c.get("id", "").replace("_", " ") in plan_text
                      or c.get("id", "") in plan_text)
        sc.trajectory = present / len(req)
    else:
        sc.trajectory = 1.0

    # -- Completeness: deliverables --
    deliv = task.expected_deliverables or {}
    checks = []
    if "min_charts" in deliv:
        n_charts = sum(1 for e in trace for a in e.get("artefacts", [])
                       if str(a).lower().endswith((".png", ".jpg", ".jpeg")))
        checks.append(n_charts >= deliv["min_charts"])
    for sec in deliv.get("required_sections", []):
        checks.append(sec.replace("_", " ") in report_text.lower())
    sc.completeness = (sum(checks) / len(checks)) if checks else 1.0

    # -- Operational --
    sc.operational = {
        "steps": len(trace),
        "total_retries": sum(e.get("retries", 0) for e in trace),
        "degraded_steps": sum(1 for e in trace if e.get("degraded")),
        "total_ms": round(sum(e.get("execution_time_ms", 0) for e in trace), 1),
        "failed_steps": sum(1 for e in trace if not e.get("success")),
    }
    return sc


def aggregate(scorecards: list[Scorecard]) -> dict:
    """Aggregate scorecards into a single dashboard row."""
    n = max(len(scorecards), 1)
    return {
        "tasks": len(scorecards),
        "groundedness": round(sum(s.groundedness for s in scorecards) / n, 3),
        "fact_pass_rate": round(sum(s.fact_pass_rate for s in scorecards) / n, 3),
        "finding_recall": round(sum(s.finding_recall for s in scorecards) / n, 3),
        "total_hallucinations": sum(s.hallucinations for s in scorecards),
        "trajectory": round(sum(s.trajectory for s in scorecards) / n, 3),
        "completeness": round(sum(s.completeness for s in scorecards) / n, 3),
    }
