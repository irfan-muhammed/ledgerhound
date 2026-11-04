"""
ledgerhound.eval.runner
===================

Drive the golden evaluation: for each task, run Ledgerhound (or a provided
run function), recompute every fact against the data, score, and aggregate
into a dashboard you can trend across releases.

Usage
-----
    from ledgerhound.eval.runner import evaluate_golden
    dash = evaluate_golden("eval/golden", data_root="eval/data",
                           llm_config="claude-sonnet")

For testing or scoring pre-recorded runs, pass a custom ``run_fn(task)`` that
returns ``(trace, report_text, plan)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ledgerhound.eval.golden import load_golden, GoldenTask
from ledgerhound.eval.golden_scorer import score_run, aggregate, Scorecard


def _load_df(task: GoldenTask, data_root: str | Path):
    """Load the task's primary dataset for fact recomputation."""
    if not task.datasets:
        return None
    p = Path(data_root) / task.datasets[0].path
    if not p.exists():
        p = Path(task.datasets[0].path)
    if not p.exists():
        return None
    import pandas as pd
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".tsv"):
        return pd.read_csv(p)
    return None


def _default_run_fn(llm_config: str | None, data_root: str | Path):
    """Build a run function that executes Ledgerhound on a task via AutoRunner."""
    def run(task: GoldenTask):
        from ledgerhound.auto.runner import AutoRunner, RunConfig
        data_path = None
        if task.datasets:
            cand = Path(data_root) / task.datasets[0].path
            data_path = str(cand) if cand.exists() else task.datasets[0].path
        cfg = RunConfig(goal=task.goal, llm_config=llm_config, data=data_path)
        runner = AutoRunner(cfg)
        runner.run()
        trace = runner._trace
        plan = getattr(runner._session, "plan", []) if runner._session else []
        # report text = summary + findings, recomputed from the trace
        from ledgerhound.auto.report import _build_summary, _build_key_findings
        report_text = (_build_summary(task.goal, trace, runner._llm) + " " +
                       " ".join(_build_key_findings(task.goal, trace,
                                                    runner._llm, runner._session)))
        return trace, report_text, plan
    return run


def evaluate_golden(golden_path: str | Path,
                    data_root: str | Path = ".",
                    run_fn=None,
                    llm_config: str | None = None,
                    out_path: str | Path | None = None) -> dict:
    """Run + score every golden task. Returns the aggregate dashboard."""
    tasks = load_golden(golden_path)
    if run_fn is None:
        run_fn = _default_run_fn(llm_config, data_root)

    cards: list[Scorecard] = []
    rows = []
    for task in tasks:
        df = _load_df(task, data_root)
        try:
            trace, report_text, plan = run_fn(task)
        except Exception as e:
            rows.append({"task": task.id, "error": str(e)})
            continue
        sc = score_run(task, trace, df=df, report_text=report_text, plan=plan)
        cards.append(sc)
        rows.append({"task": task.id, **sc.to_dict()})
        print("  " + sc.summary_line())

    dash = {"aggregate": aggregate(cards), "tasks": rows}
    if out_path:
        Path(out_path).write_text(json.dumps(dash, indent=2), encoding="utf-8")

    agg = dash["aggregate"]
    print("\n=== GOLDEN EVAL DASHBOARD ===")
    print(f"  tasks            : {agg.get('tasks', 0)}")
    print(f"  groundedness     : {agg.get('groundedness', 0):.0%}")
    print(f"  fact_pass_rate   : {agg.get('fact_pass_rate', 0):.0%}")
    print(f"  finding_recall   : {agg.get('finding_recall', 0):.0%}")
    print(f"  hallucinations   : {agg.get('total_hallucinations', 0)}")
    print(f"  trajectory       : {agg.get('trajectory', 0):.0%}")
    print(f"  completeness     : {agg.get('completeness', 0):.0%}")
    return dash
