"""Human-in-the-loop gate: high-risk actions must fail closed.

Regression for the Phase 1 bug where any answer that was not literally
"proceed"/"cancel" snapped to "proceed" -- so typing "no" approved.
"""

from __future__ import annotations

import pytest

from ledgerhound.analyst.hitl import HITLGate, auto_yes, classify_action, resolve_high_answer
from ledgerhound.cli.prompts import confirm_prompt
from ledgerhound.tools import ToolDef
from tests.conftest import calls, reply, tool_call

DECLINE = ["no", "n", "No", "N", "NO", "", "   ", None, "nah", "stop", "cancel",
           "CANCEL", "procede", "yes please", "1"]
APPROVE = ["proceed", "PROCEED", " proceed ", "yes", "y", "Y", "ok", "approve"]


def _gate(answer):
    return HITLGate(prompt=lambda q, o: answer)


@pytest.mark.parametrize("answer", DECLINE)
def test_resolve_high_answer_declines_anything_not_an_explicit_yes(answer):
    assert resolve_high_answer(answer, ("proceed", "cancel")) == "cancel"


@pytest.mark.parametrize("answer", APPROVE)
def test_resolve_high_answer_approves_explicit_yes(answer):
    assert resolve_high_answer(answer, ("proceed", "cancel")) == "proceed"


@pytest.mark.parametrize("answer", DECLINE)
def test_high_declines(answer):
    d = _gate(answer).high("Push to origin?")
    assert d.approved is False
    assert d.answer == "cancel"


@pytest.mark.parametrize("answer", APPROVE)
def test_high_approves(answer):
    d = _gate(answer).high("Push to origin?")
    assert d.approved is True
    assert d.answer == "proceed"


@pytest.mark.parametrize("answer,expected", [("no", False), ("n", False), ("", False),
                                             ("proceed", True), ("y", True)])
def test_flush_batch_applies_one_answer_to_every_queued_action(answer, expected):
    gate = _gate(answer)
    gate.queue_high("push?")
    gate.queue_high("create jira issue?")
    decisions = gate.flush_batch()
    assert [d.approved for d in decisions] == [expected, expected]
    assert gate.flush_batch() == []          # queue drained


def test_high_decisions_are_logged():
    gate = _gate("n")
    gate.high("deploy?")
    assert len(gate.log) == 1
    assert gate.log[0].level == "high" and gate.log[0].approved is False


def test_auto_yes_still_approves_non_interactive_runs():
    # Unattended runs (Kubeflow, evals) rely on auto_yes taking options[0].
    assert HITLGate(prompt=auto_yes).high("deploy?").approved is True


@pytest.mark.parametrize("answer,expected", [("yes", True), ("y", True), ("no", False),
                                             ("n", False), ("", False), ("maybe", False)])
def test_confirm_gate(answer, expected):
    assert _gate(answer).confirm("create digest.yaml?").approved is expected


class _Console:
    def __init__(self, answer=None, exc=None):
        self.answer, self.exc = answer, exc

    def print(self, *a, **k):
        pass

    def input(self, *a, **k):
        if self.exc:
            raise self.exc
        return self.answer


@pytest.mark.parametrize("console", [_Console(answer=""), _Console(exc=EOFError()),
                                     _Console(exc=KeyboardInterrupt())])
def test_terminal_prompt_empty_or_interrupt_declines(console):
    ans = confirm_prompt(console, str, "deploy?", ["proceed", "cancel"])
    gate = HITLGate(prompt=lambda q, o: ans)
    assert gate.high("deploy?").approved is False


def test_classify_action_tiers():
    assert classify_action("git_commit_push") == "high"
    assert classify_action("run_python") == "medium"
    assert classify_action("load_data") == "low"


# -- through the real tool loop --------------------------------------------

def _register_push(analyst, sink):
    analyst.registry.register(ToolDef(
        name="git_commit_push", description="push",
        parameters={"type": "object", "properties": {"message": {"type": "string"}}},
        fn=lambda message="": sink.append(message) or "pushed"))


@pytest.mark.parametrize("answer", ["no", "n", ""])
def test_declined_high_tier_tool_never_runs(make_analyst, answer):
    pushed: list = []
    a = make_analyst([calls(tool_call("git_commit_push", message="wip")), reply("ok")],
                     gate=_gate(answer))
    _register_push(a, pushed)
    a.send("push it")
    assert pushed == []
    tool_msgs = [m for m in a.messages if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("declined:")


def test_approved_high_tier_tool_runs(make_analyst):
    pushed: list = []
    a = make_analyst([calls(tool_call("git_commit_push", message="wip")), reply("ok")],
                     gate=_gate("proceed"))
    _register_push(a, pushed)
    a.send("push it")
    assert pushed == ["wip"]


def test_batched_high_tier_tools_declined_together(make_analyst):
    pushed: list = []
    prompts: list = []
    gate = HITLGate(prompt=lambda q, o: prompts.append(q) or "no")
    a = make_analyst([calls(tool_call("git_commit_push", message="a"),
                            tool_call("git_commit_push", message="b")),
                      reply("ok")], gate=gate)
    _register_push(a, pushed)
    a.send("push both")
    assert pushed == []
    assert len(prompts) == 1 and "BATCHED" in prompts[0]
