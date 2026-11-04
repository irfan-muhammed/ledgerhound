"""
ledgerhound.monitor.investigate
==============================

Turns a drift alert into an audit-grade, evidence-backed finding
automatically -- the piece that makes this a self-investigating drift
ledgerhound rather than "a scheduler that pings Slack."

When ``DriftMonitor.run_cycle()`` produces a WARNING/FAIL alert and
``auto_investigate`` is enabled in the config, this module spins up a
real ``ConversationalAnalyst`` (autonomous mode -- the same engine the
chat shell uses) against the SAME data slice the cycle just measured
drift on, asks it to explain the alert with grounded evidence, and lets
it record a finding via the normal ``record_finding`` tool. That finding
lands in the investigation's own ``findings.jsonl`` / hash-chained
``events.jsonl`` (V3-012), exactly like any human-driven session -- so
it is genuinely recomputable and replayable, not a one-off text summary
bolted onto a webhook payload.

This makes a REAL LLM call, off by default (``auto_investigate=False``
in ``DriftMonitorConfig``). Turning it on for an unattended cron job is
a deliberate cost + safety decision the operator makes explicitly: the
investigating agent runs with ``autonomous=True`` (auto-approve), so any
high-tier action it takes proceeds without a human present -- see
``docs/v3.0/INFOSEC.md``'s note on autonomous-mode HITL being advisory,
not a real pause, in that mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def investigate_alert(
    alert,
    fraud_df,
    nonfr_df,
    llm,
    workspace,
    dataset_name: str,
) -> Optional[dict[str, Any]]:
    """Run one autonomous investigation session for a single AlertEvent.

    Returns a dict describing the finding recorded (project root, run
    folder, finding count), or None if the investigation itself failed.
    Never raises -- a failed investigation must not break the monitor
    cycle that triggered it; the caller logs/records the None instead.
    """
    try:
        import pandas as pd

        from ledgerhound.project import Project
        from ledgerhound.shell.conversation import ConversationalAnalyst

        proj_root = Path(workspace.root) / "monitor" / "investigations" / alert.cycle_id
        proj_root.mkdir(parents=True, exist_ok=True)
        proj = Project.init(proj_root, name=f"drift_investigation_{alert.cycle_id}")

        # Persist the EXACT slice the drift detector measured, so the
        # investigation reasons about the same data the alert fired on,
        # not a fresh (possibly different) pull taken moments later.
        data_path = proj_root / "drift_slice.parquet"
        combined = pd.concat(
            [fraud_df.assign(_dm_label="fraud"), nonfr_df.assign(_dm_label="nonfraud")],
            ignore_index=True,
        )
        combined.to_parquet(data_path, index=False)

        target = alert.feature or alert.cluster_key or "the affected signal"
        goal = (
            f"A drift monitor just raised a {alert.severity} alert on dataset "
            f"'{dataset_name}', period {alert.period}, concerning {target}: "
            f"{alert.claim}. The fraud/non-fraud slice it was measured on is "
            f"at {data_path} (column _dm_label distinguishes the two groups). "
            f"Investigate why this drift is happening -- profile the relevant "
            f"feature(s), compare the fraud vs non-fraud distributions, and "
            f"record a grounded finding explaining the most likely cause. "
            f"Work autonomously; do not ask clarifying questions."
        )

        # This is a narrow, single-feature, unattended cron investigation --
        # not the open-ended "explore everything for hours" interactive EDA
        # session the shell defaults were widened for (see conversation.py's
        # MAX_STEPS comment). Keep it on the old, bounded budgets explicitly
        # so an automated drift check can't silently run for hours/cost
        # against the new much larger shell defaults.
        analyst = ConversationalAnalyst(
            llm, project=proj, out_root=proj_root / "run_outputs",
            renderer=lambda s: None, autonomous=True,
            turn_budget_seconds=1200, max_steps=60, step_call_budget=15,
        )
        analyst.send(goal)

        findings_path = analyst.run_dir / "findings.jsonl"
        n_findings = 0
        if findings_path.exists():
            n_findings = sum(
                1 for ln in findings_path.read_text(encoding="utf-8").splitlines() if ln.strip()
            )

        return {
            "cycle_id": alert.cycle_id,
            "alert_severity": alert.severity,
            "project_root": str(proj_root),
            "run_dir": str(analyst.run_dir),
            "n_findings_recorded": n_findings,
        }
    except Exception as exc:
        log.warning("Auto-investigation failed for alert cycle=%s: %s", alert.cycle_id, exc)
        return None


def investigate_worst_alert(
    alerts: list,
    fraud_df,
    nonfr_df,
    llm,
    workspace,
    dataset_name: str,
) -> Optional[dict[str, Any]]:
    """Investigate only the single highest-severity alert from a cycle --
    one real LLM session per cycle, not one per alert, so a cycle with
    many simultaneous alerts doesn't fan out into many unattended LLM
    calls. Returns None if `alerts` is empty."""
    if not alerts:
        return None
    _rank = {"FAIL": 2, "WARNING": 1, "INFO": 0}
    worst = max(alerts, key=lambda a: _rank.get(a.severity, 0))
    return investigate_alert(worst, fraud_df, nonfr_df, llm, workspace, dataset_name)
