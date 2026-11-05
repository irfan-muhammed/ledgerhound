"""
ledgerhound.shell.approvals
===========================

Person decisions when Ledgerhound is NOT the one talking to the person.

In the CLI shell, HITLGate's prompt is a terminal prompt: a person answers.
Behind MCP the host agent sits between Ledgerhound and the person, so there
are two honest routes (ADR 0001, "Approvals"):

1. **Elicitation.** The MCP client supports asking the user directly; the
   person's answer comes back through `elicit`. Recorded `approved_by=person`.
2. **Relay through the agent.** No elicitation: the tool stops with
   `NeedsPersonDecision`, the agent is told to ask the user and retry with
   `confirm=<their answer>`. Ledgerhound cannot tell whether the agent asked
   anyone, so the decision is recorded `approved_by=agent` and every report
   flags it as approved by the agent, not verified by a person.

Nothing here ever picks an option on its own: a missing or unmatched answer
stops the call instead of defaulting to options[0].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ledgerhound.analyst.hitl import _ASKED_LEVELS, HITLDecision, HITLGate

# Returns the person's answer; "" if they declined or cancelled; None if this
# client cannot ask at all.
ElicitFn = Callable[[str, list], Optional[str]]

# approved_by for a form the client returned as "decline": safe, but not
# provably a person (see RelayGate._relay).
DECLINED_VIA_CLIENT = "declined via client"


class NeedsPersonDecision(Exception):
    """A tool needs a person's choice that no one has given yet."""

    def __init__(self, question: str, options: list, reason: str = ""):
        self.question = question
        self.options = list(options)
        self.reason = reason
        super().__init__(question)

    def tool_result(self, tool: str) -> str:
        opts = " | ".join(f'"{o}"' for o in self.options)
        why = f"{self.reason} " if self.reason else ""
        return (f"NEEDS USER DECISION: {why}{tool} needs the user's own answer, "
                f"which Ledgerhound could not ask for directly.\n"
                f"Ask the user, word for word:\n{self.question}\n"
                f"Options: {opts}\n"
                f"Then call {tool} again with the same arguments plus "
                f"confirm=<the option they chose>. Do not choose for them: the "
                f"answer is recorded as relayed by you, and the report flags it "
                f"as approved by the agent, not verified by a person.")


def match_option(answer: str, options: list) -> Optional[str]:
    """Exact match, else a unique case-insensitive prefix match; else None."""
    a = (answer or "").strip().lower()
    if not a:
        return None
    for o in options:
        if a == str(o).lower():
            return o
    hits = [o for o in options if str(o).lower().startswith(a)]
    return hits[0] if len(hits) == 1 else None


@dataclass
class RelayGate(HITLGate):
    """HITLGate for hosts where an agent sits between Ledgerhound and the
    person (MCP). See the module docstring for the two routes."""

    elicit: Optional[ElicitFn] = None
    # Set by the MCP layer for ONE tool call: the answer the agent relayed.
    agent_answer: Optional[str] = None
    last_approved_by: str = ""
    # The question the last call stopped on, so an MCP layer that can ask the
    # person out-of-band (2026-07-28 input_required) knows what to ask.
    pending: Optional[tuple] = None

    def __post_init__(self):
        self.prompt = self._relay

    @property
    def can_ask_person(self) -> bool:
        return self.elicit is not None

    def _relay(self, question: str, options: list) -> str:
        if self.elicit is not None:
            ans = self.elicit(question, list(options))
            if ans is not None:
                # Only an accepted choice is provably a person's. A decline
                # may be the person -- or the client's own policy (Codex with
                # approval "never" declines every form unseen). It is safe
                # (nothing is authorised) but never claimed as a person's.
                self.last_approved_by = "person" if ans else DECLINED_VIA_CLIENT
                if not ans:
                    # Declined/cancelled. By convention the LAST option is the
                    # safe "no" (cancel, keep analysing, free-form); never
                    # fall through to options[0], which is usually "yes".
                    return options[-1]
                return match_option(ans, options) or options[-1]
        if self.agent_answer is not None:
            raw, self.agent_answer = self.agent_answer, None
            hit = match_option(raw, options)
            if hit is None:
                self.pending = (question, list(options))
                raise NeedsPersonDecision(
                    question, options,
                    reason=f'confirm="{raw}" matches none of the options.')
            self.last_approved_by = "agent"
            return hit
        self.pending = (question, list(options))
        raise NeedsPersonDecision(question, options)

    def _record(self, d: HITLDecision) -> HITLDecision:
        if d.level in _ASKED_LEVELS:
            d.approved_by = self.last_approved_by or "person"
        else:
            d.approved_by = "not asked"
        self.last_approved_by = ""
        self.log.append(d)
        return d
