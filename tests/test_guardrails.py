"""Guardrails refuse; they do not merely advise.

Covers the gates on run_python (modelling before authorisation, account-
grain exports with duplicate keys, the exec-mode blocklist), the
SessionGoal checks they are built on, and how a refusal is treated by the
tool loop (it is a failure, not evidence).
"""

from __future__ import annotations

import pytest

from ledgerhound.analyst.hitl import HITLGate
from ledgerhound.shell.session_goal import SessionGoal

FIT = ("from sklearn.linear_model import LogisticRegression\n"
       "m = LogisticRegression().fit(df[['amount']], df.is_fraud)")


def _no_methodology_gate():
    # The first load asks which methodology to follow; the last option is
    # "no methodology -- free-form analysis", which leaves the session-goal
    # gates as the only authority.
    return HITLGate(prompt=lambda q, o: o[-1])


@pytest.fixture
def analyst(make_analyst, fraud_csv):
    a = make_analyst(gate=_no_methodology_gate())
    out = a._run_tool("load_data", {"path": str(fraud_csv)})
    assert "loaded" in out
    assert a.methodology is None
    return a


def _events(a, type_):
    return [e for e in a._events.load() if e.type == type_]


# -- SessionGoal (pure) ------------------------------------------------------

def test_goal_refuses_model_training_until_authorised():
    g = SessionGoal()
    assert g.check_modeling(FIT).startswith("REFUSED")
    g.modeling_authorised = True
    assert g.check_modeling(FIT) == ""


@pytest.mark.parametrize("code", ["df.describe()", "df.groupby('payer_id').amount.sum()"])
def test_goal_allows_analysis_code(code):
    assert SessionGoal().check_modeling(code) == ""


def test_goal_refuses_account_export_with_duplicate_keys():
    g = SessionGoal(unit_of_analysis="account", entity_key="payer_id")
    frames = {"out": {"n_rows": 8, "dup_keys": {"payer_id": 6}}}
    msg = g.check_export("out.to_csv('features.csv')", frames)
    assert msg.startswith("REFUSED") and "payer_id" in msg


def test_goal_allows_account_export_at_one_row_per_key():
    g = SessionGoal(unit_of_analysis="account", entity_key="payer_id")
    frames = {"out": {"n_rows": 2, "dup_keys": {"payer_id": 0}}}
    assert g.check_export("out.to_csv('features.csv')", frames) == ""


def test_goal_ignores_frames_the_cell_does_not_export():
    g = SessionGoal(unit_of_analysis="account", entity_key="payer_id")
    frames = {"df": {"n_rows": 8, "dup_keys": {"payer_id": 6}},
              "agg": {"n_rows": 2, "dup_keys": {"payer_id": 0}}}
    assert g.check_export("agg.to_csv('features.csv')", frames) == ""


# -- through the analyst's run_python ----------------------------------------

def test_model_training_is_refused_and_not_executed(analyst):
    out = analyst._run_tool("run_python", {"code": FIT + "\nprint('TRAINED')"})
    assert out.startswith("REFUSED")
    assert "TRAINED" not in out
    assert [e.payload["kind"] for e in _events(analyst, "goal_gate")] == ["modeling_unauthorised"]


def test_model_training_allowed_once_authorised(analyst):
    analyst.goal.modeling_authorised = True
    out = analyst._run_tool("run_python", {"code": FIT + "\nprint('TRAINED')"})
    assert "TRAINED" in out


def _account_goal(a):
    a._run_tool("set_goal", {"objective": "account features",
                             "unit_of_analysis": "account", "entity_key": "payer_id"})


def test_wrong_grain_export_is_refused_before_the_file_is_written(analyst):
    _account_goal(analyst)
    analyst._run_tool("run_python", {"code": "out = df[['payer_id', 'amount']]"})
    out = analyst._run_tool("run_python", {
        "code": "out.to_csv(features_dir / 'features.csv', index=False)"})
    assert out.startswith("REFUSED")
    assert not (analyst._features_dir() / "features.csv").exists()
    assert [e.payload["kind"] for e in _events(analyst, "goal_gate")] == ["grain_violation"]


def test_aggregated_export_is_allowed(analyst):
    _account_goal(analyst)
    analyst._run_tool("run_python", {
        "code": "agg = df.groupby('payer_id', as_index=False).amount.sum()"})
    out = analyst._run_tool("run_python", {
        "code": "agg.to_csv(features_dir / 'features.csv', index=False)"})
    assert not out.startswith("REFUSED")
    written = analyst._features_dir() / "features.csv"
    assert written.exists() and len(written.read_text().splitlines()) == 3   # header + 2 payers


def test_write_outside_workspace_is_redirected(analyst, tmp_path):
    target = tmp_path / "elsewhere.csv"
    out = analyst._run_tool("run_python", {"code": f"df.to_csv(r'{target}', index=False)"})
    assert "WRITE REDIRECTED" in out
    assert not target.exists()


@pytest.mark.parametrize("code", ["import os\nos.listdir('.')", "open('/etc/passwd').read()",
                                  "import subprocess"])
def test_exec_mode_blocklist(analyst, code):
    assert analyst._run_tool("run_python", {"code": code}).startswith("blocked:")


def test_run_python_before_load_is_refused(make_analyst):
    a = make_analyst()
    assert "no data loaded" in a._run_tool("run_python", {"code": "df.shape"})


# -- a refusal is a failure, not evidence ------------------------------------

def test_refusal_is_not_counted_as_evidence(analyst):
    before = analyst._evidence_count
    analyst._run_tool("run_python", {"code": FIT})
    assert analyst._evidence_count == before


def test_refusal_does_not_offer_an_undo(analyst):
    out = analyst._run_tool("run_python", {"code": FIT})
    assert "undo" not in out.lower()


def test_successful_cell_counts_as_evidence(analyst):
    before = analyst._evidence_count
    out = analyst._run_tool("run_python", {"code": "print(df.amount.sum())"})
    assert "1240" in out
    assert analyst._evidence_count == before + 1


def test_repeated_refusal_is_flagged(analyst):
    analyst._run_tool("run_python", {"code": FIT})
    second = analyst._run_tool("run_python", {"code": FIT})
    assert [e.payload["count"] for e in _events(analyst, "identical_failure")] == [2]
    assert second.startswith("REFUSED")


def test_tool_result_event_marks_refusal_unsuccessful(analyst):
    analyst._run_tool("run_python", {"code": FIT})
    last = _events(analyst, "tool_result")[-1]
    assert last.payload["success"] is False
