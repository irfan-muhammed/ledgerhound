"""Person decisions behind MCP (ADR 0001, "Approvals"): elicited answers are
a person's; relayed answers are the agent's and every report says so; nothing
ever defaults to options[0]. Plus MCP mode's no-exec-fallback rule."""

from __future__ import annotations

import re
from pathlib import Path

from ledgerhound.analyst.hitl import HITLGate
from ledgerhound.shell.approvals import RelayGate, match_option
from ledgerhound.shell.tools import AnalystTools

AUTH = "request_modeling_authorisation"


def _session(tmp_path, gate, **kw):
    return AnalystTools(out_root=tmp_path / "runs", renderer=lambda _: None, gate=gate, **kw)


def _report(s, title="r"):
    res = s._run_tool("save_report", {"title": title, "markdown": "body"})
    path = re.search(r"-> (\S+)", res).group(1)
    return Path(path).read_text(encoding="utf-8")


def test_without_elicitation_the_agent_must_ask(tmp_path):
    gate = RelayGate()
    s = _session(tmp_path, gate)
    res = s._run_tool(AUTH, {"summary": "features look stable"})
    assert res.startswith("NEEDS USER DECISION")
    assert "Authorise modelling?" in res and "confirm=" in res
    assert s.goal.modeling_authorised is False


def test_relayed_answer_is_recorded_and_flagged_as_the_agents(tmp_path):
    gate = RelayGate()
    s = _session(tmp_path, gate)
    gate.agent_answer = "authorise modelling"
    res = s._run_tool(AUTH, {"summary": "features look stable"})
    assert s.goal.modeling_authorised is True
    assert "relayed by the agent" in res
    assert [d.approved_by for d in s.agent_approvals()] == ["agent"]
    report = _report(s)
    assert "1 decision(s) approved by the agent, not verified by a person" in report
    assert "authorise modelling" in report


def test_unmatched_relayed_answer_stops_instead_of_defaulting(tmp_path):
    gate = RelayGate()
    s = _session(tmp_path, gate)
    gate.agent_answer = "sure, go ahead"
    res = s._run_tool(AUTH, {"summary": "x"})
    assert res.startswith("NEEDS USER DECISION") and "matches none" in res
    assert s.goal.modeling_authorised is False
    assert gate.agent_answer is None          # one answer, one call


def test_elicited_answer_is_a_persons(tmp_path):
    gate = RelayGate(elicit=lambda q, o: "authorise modelling")
    s = _session(tmp_path, gate)
    s._run_tool(AUTH, {"summary": "x"})
    assert s.goal.modeling_authorised is True and not s.agent_approvals()
    report = _report(s)
    assert "Decisions made by a person: 1" in report
    assert "approved by the agent" not in report


def test_declined_elicitation_means_no_not_the_first_option(tmp_path):
    gate = RelayGate(elicit=lambda q, o: "")
    s = _session(tmp_path, gate)
    s._run_tool(AUTH, {"summary": "x"})
    assert s.goal.modeling_authorised is False


def test_client_without_elicitation_falls_back_to_relay(tmp_path):
    gate = RelayGate(elicit=lambda q, o: None)    # None = cannot ask
    s = _session(tmp_path, gate)
    assert s._run_tool(AUTH, {"summary": "x"}).startswith("NEEDS USER DECISION")


def test_cli_gate_records_who_answered():
    assert HITLGate(prompt=lambda q, o: o[0]).choose("q", ["a", "b"]).approved_by == "person"
    assert HITLGate().choose("q", ["a", "b"]).approved_by == "default"
    assert HITLGate().medium("run_python").approved_by == "not asked"


def test_match_option():
    opts = ["authorise modelling", "no -- keep doing feature/analysis work"]
    assert match_option("Authorise Modelling", opts) == opts[0]
    assert match_option("no", opts) == opts[1]
    assert match_option("yes", opts) is None and match_option("", opts) is None


def test_methodology_is_not_guessed_without_a_person(tmp_path, fraud_csv):
    s = _session(tmp_path, RelayGate())
    res = s._run_tool("load_data", {"path": str(fraud_csv)})
    assert "none adopted" in res and "ask the user" in res
    assert s.methodology is None


def test_no_exec_fallback_refuses_and_runs_nothing(tmp_path, fraud_csv):
    s = _session(tmp_path, RelayGate(), use_kernel=False, exec_fallback=False)
    s._run_tool("load_data", {"path": str(fraud_csv)})
    marker = tmp_path / "ran.txt"
    res = s._run_tool("run_python", {"code": f"open(r'{marker}', 'w').write('x')"})
    assert res.startswith("REFUSED") and "kernel mode is off" in res
    assert not marker.exists()
    assert "No analysis code was run." in _report(s)
