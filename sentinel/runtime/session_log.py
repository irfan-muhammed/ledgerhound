"""
sentinel.runtime.session_log
==============================

ONE central file for the whole deployment, tracking SESSION QUALITY.

This is not a per-project or per-event log -- it lives at a single,
deployment-wide path (default ``~/.sentinel/sessions.jsonl``, override with
``observability.sessions_log`` in main_config.yaml or ``$SENTINEL_SESSIONS_LOG``
so a shared install points it at one mounted path everyone writes to), and
every session, from every user, on every project, appends to it.

The grain is a SESSION, not an event: one compact "scorecard" line is
appended after each turn (``ConversationalAnalyst.send()``), built from
running counts of that run's events (``SessionStats``, fed as each event is
written -- the file is not re-read per turn) -- tool-call/error/refusal/HITL
*counts*, not their bodies. A session with 10,000 tool calls still adds one
small line per turn, not 10,000; a whole deployment's history of sessions
stays a file you can `wc -l` and open in pandas, not something that needs
its own log-aggregation stack.

Because a line is appended on every turn (not only at clean exit), a killed
or crashed session still leaves its last-known scorecard behind -- take the
LAST line for a given ``run_id`` as that session's current state; earlier
lines for the same run_id are just its history turn over turn.

events.jsonl (sentinel.runtime.events) is untouched and remains the
full-fidelity, per-run forensic record -- this module never writes or
replaces it (build_record can still summarise it from file for tools). Rotation is still a safety net here (a very long-lived
deployment could in principle accumulate a huge number of sessions), not
the primary size control -- the one-line-per-turn grain is.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_MAX_BYTES = 20_000_000
DEFAULT_BACKUP_COUNT = 10

_logger_cache: dict[str, logging.Logger] = {}


def default_path() -> Path:
    try:
        from sentinel.config import get_str
        override = get_str("observability", "sessions_log",
                           env="SENTINEL_SESSIONS_LOG")
        if override:
            return Path(override)
    except Exception:  # noqa: BLE001
        pass
    # 2026-09-21: with everything else, under Agentic_eda_results/<user>/
    # (sentinel.paths), not the home folder.
    try:
        from sentinel.paths import run_outputs_dir
        return run_outputs_dir() / "sessions.jsonl"
    except Exception:  # noqa: BLE001
        return Path.home() / ".sentinel" / "sessions.jsonl"


def _logger(path: Path, max_bytes: int, backup_count: int) -> logging.Logger:
    key = str(path)
    cached = _logger_cache.get(key)
    if cached is not None:
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"sentinel.session_log::{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    _logger_cache[key] = logger
    return logger


def append_session_record(record: dict, *, path: Optional[Path] = None,
                          max_bytes: int = DEFAULT_MAX_BYTES,
                          backup_count: int = DEFAULT_BACKUP_COUNT) -> None:
    """Append one scorecard line. Never raises -- observability must not be
    able to break a session; a failure here is silently swallowed."""
    try:
        p = path or default_path()
        _logger(p, max_bytes, backup_count).info(json.dumps(record, default=str))
    except Exception:  # noqa: BLE001
        pass


# Flags surfaced when a session shows a pattern we've specifically been
# burned by before (see the v6.3 postmortem fixes) -- greppable across the
# whole deployment's history instead of only visible from inside one run's
# events.jsonl.
def _flags(types: Counter, refusals: int) -> list:
    flags = []
    if types.get("plan_lossy", 0):
        flags.append("plan_lossy_detected")
    if types.get("forced_continuation", 0) >= 2:
        flags.append("stopped_with_open_steps")
    if types.get("plan_done_forced", 0):
        flags.append("plan_step_force_closed")
    if refusals:
        flags.append("governance_refusal_fired")
    if types.get("identical_failure", 0):
        flags.append("repeated_identical_failure")
    return flags


class SessionStats:
    """Running counts for one session's scorecard, updated as each event is
    written (EventStream listener) instead of re-reading events.jsonl.

    The scorecard used to re-read and re-parse the WHOLE events.jsonl after
    every turn just to count it again -- a cost that grew with the session,
    and /adversarial runs ~150 turns per case. Only the COUNTING moved:
    every event is still written to events.jsonl in full, and replay,
    run-views and the notebook export still read that file as before.

    build_record() is defined in terms of this class, so the incremental
    and the from-file scorecard are the same computation by construction.
    """

    def __init__(self) -> None:
        self.types: Counter = Counter()
        self.hitl_high = 0
        self.hitl_declined = 0
        self.refusals = 0
        self.errors = 0
        self.n_events = 0

    def observe(self, event) -> None:
        etype = getattr(event, "type", "")
        payload = getattr(event, "payload", None) or {}
        self.n_events += 1
        self.types[etype] += 1
        if etype == "hitl":
            if payload.get("tier") == "high":
                self.hitl_high += 1
            if payload.get("approved") is False:
                self.hitl_declined += 1
        elif etype == "tool_result":
            result = payload.get("result")
            if isinstance(result, str) and result.strip().upper().startswith("REFUSED"):
                self.refusals += 1
            if payload.get("success") is False:
                self.errors += 1

    def record(self, *, run_id: str, user: str, project: str,
               started_at: str, objective: str, unit_of_analysis: str,
               phase: str, modeling_authorised: bool, methodology: str,
               mode: str, dataset_rows: Optional[int], reports: list,
               findings_count: int) -> dict:
        types = self.types
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "run_id": run_id,
            "user": user,
            "project": project,
            "objective": (objective or "")[:200],
            "unit_of_analysis": unit_of_analysis,
            "phase": phase,
            "modeling_authorised": bool(modeling_authorised),
            "methodology": methodology,
            "mode": mode,
            "dataset_rows": dataset_rows,
            "tool_calls": types.get("tool_call", 0),
            "errors": self.errors,
            "hitl_high_tier": self.hitl_high,
            "hitl_declined": self.hitl_declined,
            "refusals": self.refusals,
            "context_compactions": types.get("context_compressed", 0),
            "forced_continuation_nudges": types.get("forced_continuation", 0),
            "plan_lossy_events": types.get("plan_lossy", 0),
            "reports_saved": list(reports),
            "findings_count": findings_count,
            "flags": _flags(types, self.refusals),
            "n_events_total": self.n_events,
        }


def build_record(events: list, **meta) -> dict:
    """Summarise a list of events into one compact quality record -- the
    from-file form (tools, tests). The live shell keeps a SessionStats
    instead and never re-reads its events.jsonl."""
    stats = SessionStats()
    for e in events:
        stats.observe(e)
    return stats.record(**meta)

