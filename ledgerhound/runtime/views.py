"""
ledgerhound.runtime.views
=======================

V3-012: projections of an ``EventStream`` into the two file formats a shell
session already produced -- ``session_code.py`` (a replayable script) and
``findings.jsonl`` (evidence-backed claims). The same functions here are
called live (as each event is appended, so the file exists immediately --
unchanged behaviour from before this module existed) and in bulk by
``ledgerhound run-views <run_dir>`` (regenerating both files from
``events.jsonl`` alone). One projection, two call sites -- so the two files
cannot drift from the stream that recorded them.
"""

from __future__ import annotations

import json
from pathlib import Path

from ledgerhound.runtime.events import Event, EventStream


def finding_line(ev: Event) -> str:
    p = ev.payload
    entry = {
        "claim": p.get("claim", ""),
        "evidence": p.get("evidence", ""),
        "sample_frame": p.get("sample_frame", "unknown"),
        "ts": ev.ts,
    }
    # V3-052: optional predicate -- the exact condition/window a numeric
    # claim was computed under (e.g. "hour>=22 OR hour<6"). Additive and
    # defaulted to "" so old events (no predicate field at all) regenerate
    # identically via `ledgerhound run-views`; only present when the caller
    # supplied one. This is what lets two findings about "the same"
    # quantity be checked for actually meaning the same thing before they
    # are compared -- the reported bug (fraud vs non-fraud night-share
    # computed under two different hour windows, then compared as if
    # equivalent) had no such field to catch it on.
    if p.get("kb_contradicts"):
        entry["kb_contradicts"] = p["kb_contradicts"]
    predicate = p.get("predicate", "")
    if predicate:
        entry["predicate"] = predicate
    return json.dumps(entry) + "\n"


def code_block(ev: Event) -> str:
    p = ev.payload
    code = (p.get("code", "") or "").rstrip()
    return f"# --- {ev.ts} ---\n{code}\n\n"


def regenerate_findings(events: list[Event]) -> str:
    return "".join(finding_line(e) for e in events if e.type == "finding")


def regenerate_session_code(events: list[Event]) -> str:
    return "".join(
        code_block(e) for e in events
        if e.type == "code_exec" and e.payload.get("success", True)
    )


def regenerate_all(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    events = EventStream(run_dir / "events.jsonl").load()
    out: dict[str, Path] = {}

    findings_path = run_dir / "findings.jsonl"
    findings_path.write_text(regenerate_findings(events), encoding="utf-8")
    out["findings"] = findings_path

    code_path = run_dir / "session_code.py"
    code_path.write_text(regenerate_session_code(events), encoding="utf-8")
    out["session_code"] = code_path

    return out
