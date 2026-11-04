"""A full analyst session driven by a scripted LLM -- no network, no model.

The script plays the part of the model: it loads data, computes, hits a
guardrail, records a finding, saves a report and answers. The test checks
what a reviewer would check afterwards: the reply, the artefacts on disk,
the refusal, and that the audit trail verifies.
"""

from __future__ import annotations

import json

import pytest

from sentinel.analyst.hitl import HITLGate
from sentinel.runtime.events import EventStream
from tests.conftest import ScriptedLLM, calls, reply, tool_call

FIT = ("from sklearn.linear_model import LogisticRegression\n"
       "LogisticRegression().fit(df[['amount']], df.is_fraud)")
FINAL = "Fraud transactions average 550 vs 23 for non-fraud; see the saved report."


def _script(csv):
    return [
        calls(tool_call("load_data", path=str(csv))),
        calls(tool_call("run_python",
                        code="print(df.groupby('is_fraud').amount.mean().round(1).to_dict())")),
        calls(tool_call("run_python", code=FIT)),                     # refused
        calls(tool_call("record_finding",
                        claim="fraud transactions have far higher amounts than non-fraud",
                        sample_frame="full", evidence="mean 550.0 vs 23.3")),
        calls(tool_call("save_report", title="amount skew",
                        markdown="# Amount skew\nFraud mean 550 vs 23.")),
        reply(FINAL),
    ]


@pytest.fixture
def session(make_analyst, fraud_csv):
    llm = ScriptedLLM(_script(fraud_csv))
    a = make_analyst(llm=llm, gate=HITLGate(prompt=lambda q, o: o[-1]))
    answer = a.send("Profile this fraud data and tell me what drives fraud.")
    return a, llm, answer


def _tool_results(a):
    return [m["content"] for m in a.messages if m["role"] == "tool"]


def test_turn_ends_with_the_models_reply(session):
    a, llm, answer = session
    assert answer == FINAL
    assert llm.script == []                      # every scripted step consumed
    assert a.messages[-1] == {"role": "assistant", "content": FINAL}


def test_tool_results_flow_back_to_the_model(session):
    a, llm, _ = session
    results = _tool_results(a)
    assert len(results) == 5
    assert results[0].startswith("loaded")
    assert "550.0" in results[1] and "23.3" in results[1]
    assert results[2].startswith("REFUSED")
    assert results[3].startswith("recorded finding")
    assert results[4].startswith("saved")
    # each later request carries the earlier results
    assert any(m.get("role") == "tool" and "550.0" in m["content"]
               for m in llm.requests[-1]["messages"])


def test_native_tools_are_offered(session):
    _, llm, _ = session
    names = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert {"load_data", "run_python", "record_finding", "save_report"} <= names


def test_findings_and_report_land_on_disk(session):
    a, _, _ = session
    findings = [json.loads(line) for line in
                a._findings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [f["claim"] for f in findings] == [
        "fraud transactions have far higher amounts than non-fraud"]
    reports = list(a.run_dir.glob("*.md"))
    assert len(reports) == 1 and "Amount skew" in reports[0].read_text(encoding="utf-8")


def test_audit_trail_verifies_and_records_the_refusal(session):
    a, _, _ = session
    stream = EventStream(a.run_dir / "events.jsonl")
    assert stream.verify().ok
    types = [e.type for e in stream.load()]
    assert types[0] == "user"
    assert "goal_gate" in types and "finding" in types
    assert types[-1] == "assistant_final"


def test_text_protocol_session(make_analyst, fraud_csv):
    """Same flow for a gateway without native function-calling: calls come
    back as text, observations go back as user turns, no tools= is sent."""
    llm = ScriptedLLM([
        reply(f'<tool>{{"name": "load_data", "arguments": {{"path": {json.dumps(str(fraud_csv))}}}}}</tool>'),
        reply('{"name": "run_python", "arguments": {"code": "print(int(df.is_fraud.sum()))"}}'),
        reply("Two fraud transactions."),
    ])
    a = make_analyst(llm=llm, tool_mode="prompt", gate=HITLGate(prompt=lambda q, o: o[-1]))
    assert a.send("how many frauds?") == "Two fraud transactions."
    assert all("tools" not in r for r in llm.requests)
    observations = [m["content"] for m in a.messages
                    if m["role"] == "user" and m["content"].startswith("Observation from")]
    assert len(observations) == 2
    assert "2" in observations[1]
    assert EventStream(a.run_dir / "events.jsonl").verify().ok
