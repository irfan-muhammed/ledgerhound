"""
sentinel.shell.ambiguity_checks
=================================

Mechanical check for columns that LOOK like operational/workflow artifacts
rather than genuine fraud-behaviour signal -- fires on whatever frames
actually exist in the sandbox, not a tool the model may choose to call.
Fifth sibling to ``data_checks``/``leakage_checks``/``grain_checks``/
``numeric_checks``, same shape, but a DIFFERENT kind of recommendation:
those are all either-a-real-problem-or-not; this one is a genuine judgment
call the model should not make silently either way.

Why (V3-032 gap)
------------------
Real pilot feedback, two separate reviewers: "Variables such as case status
and manual creation status do not seem to provide strong fraud signals and
are more reflective of post-transaction investigation workflows" and "some
findings may be technically correct but expected from a business
perspective." Unlike a Risk Score field (which IS a downstream model
output -- a certain, name-lexicon-catchable case handled by
``leakage_checks``), whether a "case_status"-style column is pure noise or
carries real signal genuinely depends on context the model does not have
(what does this specific status field track, in this specific pipeline?).

The right move is neither "silently include it" (risks exactly what the
pilot flagged) nor "silently exclude it" (might throw away real signal --
some workflow fields DO correlate with fraud, e.g. escalation flags) -- it
is to ask. This check's job is only to make sure the ambiguity is actually
surfaced, with the same "ask_user" tool this module recommends, so the
decision is the analyst's rather than a guess in either direction.
"""

from __future__ import annotations

from sentinel.shell.data_checks import _is_frame

# Column-name hints for operational/case-management/workflow fields, checked
# as substrings on the lowered name. Not a leak-lexicon (these columns are
# not "the answer" or "a model output") -- a genuinely different failure
# mode: plausible-looking but likely-low-value signal from process, not
# behaviour.
_OPERATIONAL_HINTS = (
    "case_status", "casestatus", "manual_creation", "manual_flag",
    "investigation_status", "investigation_result", "workflow_status",
    "review_status", "disposition", "assigned_to", "closed_by",
    "escalation_status", "ticket_status", "queue_status",
)


def _operational_hits(columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        lname = str(col).lower()
        if any(h in lname for h in _OPERATIONAL_HINTS):
            out.append(str(col))
    return out


def _guidance(col: str, fname: str) -> str:
    return (
        f"'{col}' in '{fname}' looks like a case-management/workflow field, "
        "not raw behavioural data -- real pilot feedback named exactly this "
        "class of column as more reflective of post-transaction investigation "
        "process than genuine fraud signal. This is a genuine judgment call, "
        "not something to decide silently either way: call ask_user with "
        "options like [\"include as a feature\", \"exclude -- operational "
        "artifact\", \"not sure, ask the analyst what this field tracks\"] "
        "before using it."
    )


def check_ambiguous_columns(ns: dict, seen: set) -> str:
    """Inspect DataFrames in a sandbox namespace for operational/workflow-
    artifact-looking column names; return an [AMBIGUOUS COLUMN] block for
    ones not already reported, else "".

    `seen` accumulates issue keys so a warning fires once per column, not
    on every subsequent cell -- same contract as the sibling checks.
    """
    frames = {str(k): v for k, v in ns.items()
              if not str(k).startswith("_") and _is_frame(v)}
    if not frames:
        return ""

    msgs: list[str] = []
    for fname, df in frames.items():
        try:
            if len(df) == 0:
                continue
            for col in _operational_hits([str(c) for c in df.columns]):
                key = f"ambiguous:{fname}:{col}"
                if key in seen:
                    continue
                seen.add(key)
                msgs.append(_guidance(col, fname))
        except Exception:
            continue

    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[AMBIGUOUS COLUMN] Judgment call, do not decide silently:\n{body}"
