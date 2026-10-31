"""
sentinel.tools.plan_tools
=========================

Lightweight plan-then-execute for the shell: the model calls
``propose_plan`` to lay out its steps as a visible checklist, then
``update_plan`` as it works, so the user always sees where a long analysis
stands -- the Claude-Code todo-list pattern, sized for a data session.

V3-015: the checklist is now backed by a ``sentinel.plan.graph.PlanGraph``
-- typed nodes with dependencies, not just a flat list -- so that a finding
discovered mid-session can extend the plan (``add_node``) or retire a step
that turned out not to apply (``invalidate_node``) without losing the audit
trail. ``PlanState.steps`` stays a computed, backward-compatible view (list
of ``{text, status}`` dicts in node order) so every existing reader
(``ConversationalAnalyst._open_plan_steps``, the budget check, and the
pre-V3-015 tests) keeps working unchanged -- the graph is a strict
superset, not a breaking replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from sentinel.plan.graph import PlanGraph, VALID_STATUSES
from sentinel.tools.registry import ToolDef

_ICONS = {"pending": "[ ]", "active": "[>]", "done": "[x]",
          "skipped": "[-]", "blocked": "[!]"}
_STATUSES = VALID_STATUSES
# V6.3 postmortem "Filter 2": raised from 12/120 (the original silent caps)
# -- a real multi-phase ask can need more than 12 checklist lines.
_MAX_PLAN_STEPS = 24
_MAX_STEP_CHARS = 200
# strip a leading enumeration the model tends to add ("1.", "2)", "3 -"),
# otherwise the renderer's own numbering yields "1. 1. <step>"
_LEAD_NUM_RE = re.compile(r"^\s*\d+\s*[.)\-:]\s*")


@dataclass
class PlanState:
    graph: PlanGraph = field(default_factory=PlanGraph)

    @property
    def steps(self) -> list:
        """Legacy compatibility view: every node, in creation order, as a
        plain {text, status} dict -- unchanged shape from pre-V3-015."""
        return [{"text": n.text, "status": n.status} for n in self.graph.nodes]

    def render(self) -> str:
        return self.graph.render()


def build_plan_tools(state: PlanState,
                     renderer: Callable[[str], None] = lambda s: None,
                     plan_path: Optional[Path] = None,
                     evidence_counter: Callable[[], int] = lambda: 0,
                     event_logger: Optional[Callable] = None,
                     ) -> list[ToolDef]:
    # V6.3 postmortem fix: `update_plan` used to let a step go straight to
    # "done" with zero evidence that any analysis actually happened for it
    # (seq 238-249 in the 2026-09-09 run: 6 steps closed in 3 seconds, no
    # run_python/record_finding calls in between). `touch_evidence` snapshots
    # the session's substantive-tool-call counter (see
    # ConversationalAnalyst._evidence_count) at the moment each node is
    # created or set "active"; closing it "done" is refused unless that
    # counter has moved since, or the caller explicitly forces it -- and a
    # forced close is always logged, never silent.
    touch_evidence: dict[str, int] = {}

    def _elog(event: str, **fields) -> None:
        if event_logger is not None:
            try:
                event_logger(event, **fields)
            except Exception:
                pass

    def _persist() -> None:
        if plan_path is not None:
            try:
                state.graph.save(plan_path)
            except Exception:
                pass

    def propose_plan(steps: list) -> str:
        # A fresh checklist replaces the graph wholesale (matches
        # pre-V3-015 behaviour exactly) -- ids restart at 1, so any node
        # added dynamically via add_node in a PRIOR plan is gone too.
        state.graph = PlanGraph()
        # V6.3 postmortem "Filter 2": this used to slice steps[:12] and each
        # string to [:120] chars SILENTLY -- part of the original ask could
        # be dropped with no trace anywhere in the log. _MAX_STEPS/_MAX_CHARS
        # are raised (a step checklist covering a real multi-phase ask can
        # legitimately need more than 12 lines), and anything still cut is
        # now reported back to the model and logged, instead of vanishing.
        raw = [str(s).strip() for s in steps if str(s).strip()]
        dropped_steps = raw[_MAX_PLAN_STEPS:]
        kept = raw[:_MAX_PLAN_STEPS]
        texts = []
        truncated_texts = []
        for s in kept:
            t = _LEAD_NUM_RE.sub("", s).strip()
            if len(t) > _MAX_STEP_CHARS:
                truncated_texts.append(t)
            texts.append(t[:_MAX_STEP_CHARS])
        for t in texts:
            state.graph.add_node(t)
        if not state.graph.nodes:
            return "error: give at least one step"
        touch_evidence.clear()
        now = evidence_counter()
        for n in state.graph.nodes:
            touch_evidence[n.id] = now
        renderer("plan:\n" + state.render())
        _persist()
        warn = ""
        if dropped_steps or truncated_texts:
            _elog("plan_lossy", dropped=len(dropped_steps),
                 truncated=len(truncated_texts),
                 dropped_text=[s[:200] for s in dropped_steps])
            parts = []
            if dropped_steps:
                parts.append(f"{len(dropped_steps)} step(s) beyond the "
                             f"{_MAX_PLAN_STEPS}-step limit were DROPPED: "
                             + "; ".join(s[:80] for s in dropped_steps))
            if truncated_texts:
                parts.append(f"{len(truncated_texts)} step(s) were cut to "
                             f"{_MAX_STEP_CHARS} chars.")
            warn = ("\n\nWARNING (not silent -- act on this): " + " ".join(parts)
                   + " If any of this matters, call add_node for the dropped "
                   "steps or add_step to restore detail -- do not assume the "
                   "checklist above is the whole ask.")
        return "plan recorded:\n" + state.render() + warn

    def update_plan(step: int, status: str, force: bool = False) -> str:
        if not state.graph.nodes:
            return "no plan yet -- call propose_plan first."
        if status not in _STATUSES:
            return f"error: status must be one of {_STATUSES}"
        node_id = str(step)
        node = state.graph.get(node_id)
        if node is None:
            return f"error: step must be 1..{len(state.graph.nodes)}"
        if status == "done" and not force:
            if evidence_counter() <= touch_evidence.get(node_id, -1):
                _elog("plan_done_refused", step=node_id, text=node.text)
                return (f"refused: step {node_id} ('{node.text}') has had no "
                        "analysis activity (run_python / record_finding / "
                        "load_data etc.) since it was created or last set "
                        "active -- do the work first, then mark it done. If "
                        "it genuinely needs no separate work (e.g. it was "
                        "already covered by another step), call update_plan "
                        "again with force=true, and say why in your next "
                        "message so it's on the record.")
        result = state.graph.update_status(node_id, status)
        if result.startswith("error"):
            return result
        if status == "active":
            touch_evidence[node_id] = evidence_counter()
        if status == "done" and force:
            _elog("plan_done_forced", step=node_id, text=node.text)
        renderer("plan:\n" + state.render())
        _persist()
        return state.render()

    def add_node(text: str, type: str = "analysis", depends_on: Optional[list] = None,
                budget: Optional[int] = None, produces: str = "") -> str:
        node = state.graph.add_node(text=text, type=type, depends_on=depends_on,
                                    budget=budget, produces=produces)
        if isinstance(node, str):
            return node
        touch_evidence[node.id] = evidence_counter()
        renderer("plan:\n" + state.render())
        _persist()
        return f"added node {node.id}: {node.text}\n" + state.render()

    def invalidate_node(step: int, reason: str) -> str:
        result = state.graph.invalidate_node(str(step), reason)
        if result.startswith("error"):
            return result
        renderer("plan:\n" + state.render())
        _persist()
        return state.render()

    return [
        ToolDef(name="propose_plan",
                description=("Lay out your analysis plan as a short visible "
                             f"checklist BEFORE a multi-step piece of work. "
                             f"Keep steps concrete (<={_MAX_PLAN_STEPS}, "
                             f"<={_MAX_STEP_CHARS} chars each -- anything "
                             "beyond that is dropped/cut and reported back, "
                             "not silently lost)."),
                parameters={"type": "object", "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}}},
                    "required": ["steps"]},
                fn=propose_plan),
        ToolDef(name="update_plan",
                description=("Mark a plan step active/done/skipped/blocked as "
                             "you work. Closing a step 'done' is refused if no "
                             "analysis tool ran since it was created or set "
                             "active -- do the work first, or pass force=true "
                             "with a reason in your next message."),
                parameters={"type": "object", "properties": {
                    "step": {"type": "integer", "description": "1-based index"},
                    "status": {"type": "string",
                               "enum": list(_STATUSES)},
                    "force": {"type": "boolean",
                              "description": ("bypass the done-with-no-evidence "
                                              "refusal; explain why afterward")}},
                    "required": ["step", "status"]},
                fn=update_plan),
        ToolDef(name="add_node",
                description=("Add a new plan step mid-session -- use this when a "
                             "finding surfaces work the original plan didn't "
                             "anticipate, instead of silently going off-plan."),
                parameters={"type": "object", "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["analysis", "verification", "ask_user", "model"]},
                    "depends_on": {"type": "array", "items": {"type": "string"},
                                   "description": "ids of steps this one needs done first"},
                    "budget": {"type": "integer"},
                    "produces": {"type": "string"}},
                    "required": ["text"]},
                fn=add_node),
        ToolDef(name="invalidate_node",
                description=("Retire a plan step that turned out not to apply, with "
                             "a one-line reason -- marks it blocked rather than "
                             "silently dropping it, so the plan stays auditable."),
                parameters={"type": "object", "properties": {
                    "step": {"type": "integer", "description": "1-based index"},
                    "reason": {"type": "string"}},
                    "required": ["step", "reason"]},
                fn=invalidate_node),
    ]
