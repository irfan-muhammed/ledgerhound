"""
sentinel.analyst.hitl
====================

The human-in-the-loop gate (Phase 2 primitive) -- ``ask_user`` as a tool.

Every action the Analyst wants to take is classified into a gate level:

    none    -> proceed silently (cheap, safe, reversible)
    confirm -> yes/no before proceeding (e.g. "create digest.yaml?")
    choose  -> pick among options (e.g. "which save format?")

The prompt function is injected so the gate is fully testable headless and can be
rendered as buttons by an interactive shell. Trino/privileged access always maps
to at least ``confirm`` (production-data safety invariant).

V3-016 -- risk-tier HITL v2. This original none/confirm/choose gate is left
UNCHANGED (git/jira/config integrations still use it exactly as before) and
a second, additive classification sits alongside it for the interactive
shell specifically: every tool action gets a risk TIER --

    low    -> proceed silently, nothing shown (reads/computes)
    medium -> proceed automatically, but the observation carries a note the
              model is expected to relay to the user (reversible, kernel-
              local mutations -- run_python)
    high   -> route through the SAME injected `prompt` as `confirm`/`choose`,
              but with a decision-ready package (findings so far, options,
              a recommendation, and the consequence of proceeding) instead
              of a bare yes/no -- (save-over/deploy/external-system writes)

Gating happens IN-LINE, before the tool dispatches -- never deferred across
turns (a deferred tool call would break the model's "call -> result"
contract). `queue_high`/`flush_batch` let several high-tier asks that arise
within ONE assistant response's simultaneous tool_calls collapse into a
single combined prompt instead of pausing once per action -- batching is
scoped to that one response, never across turns (see
`ConversationalAnalyst.send`'s native tool_calls handling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

GateLevel = str  # "none" | "confirm" | "choose"
VALID_GATES = ("none", "confirm", "choose")

RiskTier = str  # "low" | "medium" | "high"
VALID_TIERS = ("low", "medium", "high")

# prompt(question, options) -> chosen string.
#   confirm: options == ["yes","no"], returns "yes"/"no"
#   choose : options == [...], returns one of them
#   high   : options[0] is always the "proceed" option by convention (so the
#            non-interactive default -- `auto_yes`, which always returns
#            options[0] -- proceeds with the model's own recommended action,
#            matching how confirm()/choose() already default-safe today)
PromptFn = Callable[[str, list[str]], str]


def auto_yes(question: str, options: list[str]) -> str:
    """Non-interactive prompt: always take the first (safe default) option."""
    return options[0] if options else "yes"


_APPROVE_WORDS = frozenset({"y", "yes", "ok", "okay", "approve", "approved",
                            "proceed", "go", "true"})


def resolve_high_answer(raw: Optional[str], options) -> str:
    """Map a raw high-tier answer onto one of `options`, failing CLOSED.

    `options[0]` approves and `options[-1]` declines. An exact
    (case-insensitive) option match wins; otherwise only an explicit
    approval word ("y", "yes", "proceed", ...) approves. Everything else --
    "no", "n", empty input, typos, unknown words -- declines. Before this,
    any answer that wasn't literally an option snapped to options[0], so
    typing "no" or "n" APPROVED a high-risk action.
    """
    options = list(options)
    ans = (raw or "").strip().lower()
    for opt in options:
        if ans == opt.lower():
            return opt
    if ans in _APPROVE_WORDS:
        return options[0]
    return options[-1]


@dataclass
class HITLDecision:
    question: str
    level: GateLevel
    answer: str
    approved: bool
    package: Optional[dict] = None   # the full decision package, for high tier
    note: str = ""                   # the undo/report note, for medium tier
    picked: list = field(default_factory=list)  # multi-select choices, for choose_multi


@dataclass
class HITLGate:
    """Routes actions through the injected prompt at the right gate level."""
    prompt: PromptFn = auto_yes
    log: list[HITLDecision] = field(default_factory=list)
    _pending_high: list[dict] = field(default_factory=list)

    def _record(self, d: HITLDecision) -> HITLDecision:
        self.log.append(d)
        return d

    def confirm(self, question: str) -> HITLDecision:
        ans = (self.prompt(question, ["yes", "no"]) or "no").strip().lower()
        approved = ans in ("y", "yes", "true", "ok")
        return self._record(HITLDecision(question, "confirm", ans, approved))

    def choose(self, question: str, options: list[str]) -> HITLDecision:
        if not options:
            raise ValueError("choose requires at least one option")
        ans = (self.prompt(question, list(options)) or options[0]).strip()
        if ans not in options:
            ans = options[0]      # snap to a valid option (default)
        return self._record(HITLDecision(question, "choose", ans, True))

    def choose_multi(self, question: str, options: list[str]) -> HITLDecision:
        """Multi-select: the user may pick more than one option (an MSQ,
        vs. choose()'s MCQ). Reuses the SAME `prompt(question, options) ->
        str` contract as choose()/confirm() -- no change needed to
        confirm_prompt/auto_yes -- by asking for a comma-separated answer
        and parsing the raw string returned, rather than requiring the
        prompt function itself to understand multi-select.

        `answer` on the returned HITLDecision is comma-joined for display;
        `picked` is the actual list a caller should use.
        """
        if not options:
            raise ValueError("choose_multi requires at least one option")
        example = options[0] if len(options) == 1 else f"{options[0]}, {options[1]}"
        prompt_text = f"{question} (comma-separated -- may pick more than one, e.g. \"{example}\")"
        raw = (self.prompt(prompt_text, list(options)) or "").strip()
        picked = [p.strip() for p in raw.split(",") if p.strip() in options]
        if not picked:
            picked = [options[0]]     # safe default: the first/recommended option
        return self._record(HITLDecision(
            question, "choose_multi", ", ".join(picked), True, picked=picked))

    def gate(self, question: str, level: GateLevel,
             options: Optional[list[str]] = None) -> HITLDecision:
        if level not in VALID_GATES:
            raise ValueError(f"unknown gate level {level!r}")
        if level == "none":
            return self._record(HITLDecision(question, "none", "", True))
        if level == "confirm":
            return self.confirm(question)
        return self.choose(question, options or [])

    # -- V3-016: risk tiers ---------------------------------------------------

    def low(self, action: str) -> HITLDecision:
        """Proceeds silently -- recorded for the audit log, never prompted."""
        return self._record(HITLDecision(action, "low", "", True))

    def medium(self, action: str, note: str = "") -> HITLDecision:
        """Proceeds automatically. `note` is what the caller wants relayed to
        the user (e.g. an undo affordance) -- carried on the decision, not
        synthesized here, since only the caller knows what actually happened."""
        return self._record(HITLDecision(action, "medium", "", True, note=note))

    @staticmethod
    def _render_package(pkg: dict) -> str:
        lines = [f"HIGH-RISK ACTION: {pkg['question']}"]
        if pkg.get("findings_so_far"):
            lines.append("Findings so far:")
            for f in pkg["findings_so_far"][:5]:
                lines.append(f"  - {f}")
        lines.append(f"Options: {', '.join(pkg['options'])}")
        lines.append(f"Recommendation: {pkg['recommendation']}")
        if pkg.get("consequence"):
            lines.append(f"Consequence: {pkg['consequence']}")
        return "\n".join(lines)

    def high(self, question: str, findings_so_far: Optional[list] = None,
             options: tuple = ("proceed", "cancel"), recommendation: str = "proceed",
             consequence: str = "") -> HITLDecision:
        """A single high-tier action, gated immediately (not deferred) --
        this is `confirm`/`choose`'s binary yes/no upgraded to a
        decision-ready package."""
        pkg = {"question": question, "findings_so_far": list(findings_so_far or []),
              "options": list(options), "recommendation": recommendation,
              "consequence": consequence}
        text = self._render_package(pkg)
        ans = resolve_high_answer(self.prompt(text, list(options)), options)
        approved = ans == options[0]
        return self._record(HITLDecision(question, "high", ans, approved, package=pkg))

    def queue_high(self, question: str, findings_so_far: Optional[list] = None,
                   options: tuple = ("proceed", "cancel"), recommendation: str = "proceed",
                   consequence: str = "") -> None:
        """Queue a high-tier ask instead of gating it immediately -- for
        combining several asks that arise within ONE assistant response's
        simultaneous tool_calls into a single prompt via `flush_batch()`.
        Never used to defer a tool call across turns."""
        self._pending_high.append({
            "question": question, "findings_so_far": list(findings_so_far or []),
            "options": list(options), "recommendation": recommendation,
            "consequence": consequence,
        })

    def flush_batch(self) -> list:
        """Combine every queued high-tier ask into ONE prompt call and apply
        the same approve/decline to all of them. Returns [] if nothing was
        queued (a single queued item still gets its own combined-format
        prompt -- callers with exactly one high-tier action in a response
        should just call `high()` directly instead)."""
        if not self._pending_high:
            return []
        pending = self._pending_high
        self._pending_high = []
        lines = ["BATCHED HIGH-RISK ACTIONS (multiple in one response):"]
        for i, pkg in enumerate(pending, 1):
            lines.append(f"\n{i}) {pkg['question']}")
            if pkg.get("consequence"):
                lines.append(f"   consequence: {pkg['consequence']}")
            lines.append(f"   recommendation: {pkg['recommendation']}")
        lines.append("\nApprove all, or cancel all?")
        text = "\n".join(lines)
        options = ("proceed", "cancel")
        ans = resolve_high_answer(self.prompt(text, list(options)), options)
        approved = ans == options[0]
        return [self._record(HITLDecision(pkg["question"], "high", ans, approved,
                                          package=pkg))
                for pkg in pending]


# -- action -> risk-tier classification --------------------------------------

_LOW_ACTIONS = frozenset({
    "list_directory", "load_data", "profile_data", "record_finding",
    "consult_skill", "record_decision", "record_data_answer", "undo_last_transform",
    # save_report / save_understanding are MANDATORY deliverables written
    # into this run's own output folder -- not external side effects. Gating
    # them high-tier meant a 2026-08-06 production run stopped to ask
    # "proceed/cancel?" four times for saving the report it was asked to
    # produce, and twice more for save_understanding: pure friction on a
    # non-optional step, and dead GPU time in an unattended job.
    "save_report", "save_understanding",
    "propose_plan", "update_plan", "add_node", "invalidate_node",
    "git_status", "git_log", "git_diff", "git_today",
    "jira_pending_tasks", "show_config",
})
_MEDIUM_ACTIONS = frozenset({"run_python"})
_HIGH_ACTIONS = frozenset({
    "git_pull", "git_commit_push",
    "jira_post_update", "jira_create_issue", "jira_transition",
    "configure_integration",
    # V3-023: L0 framing capture (cost matrix, unit of analysis, latency
    # budget, regulatory bounds) is high tier "by definition" per the v3
    # roadmap -- classified here so a future interactive capture flow
    # inherits high-tier treatment automatically. No such flow is built
    # yet (sentinel.framing.l0.L0Framing.capture() is called directly by
    # trusted code today); this entry is registered ahead of that build.
    "capture_l0_framing",
    # V3-025: promoting a challenger to champion (sentinel.modeling.
    # registry.ModelRegistry.promote) flips which model is served --
    # "promotion without HITL approval impossible" is the roadmap's own
    # close-out wording. Registered here even though ModelRegistry.promote
    # already calls HITLGate.high() directly (a trusted, non-LLM caller,
    # same status as capture_l0_framing) -- ahead of a future tool-calling
    # wrapper that would need this classification to gate itself.
    "promote_challenger",
})


def classify_action(tool_name: str) -> RiskTier:
    """Reads/computes are low by default -- an unrecognized/future tool
    name proceeds silently (today's ungated behavior) rather than being
    surprise-gated. `run_python` is deliberately medium, not tier-split by
    content: parsing model-authored code to guess "is this a drop/impute"
    is fragile, and V3-017's post-mutation delta is the real safety net --
    see docs/v3.0/HITL_TIERS.md for the reasoning."""
    if tool_name in _HIGH_ACTIONS:
        return "high"
    if tool_name in _MEDIUM_ACTIONS:
        return "medium"
    return "low"
