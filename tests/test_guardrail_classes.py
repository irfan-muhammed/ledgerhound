"""The guardrail classes on their own, without an analyst session -- the
form the MCP server will call them in."""

from __future__ import annotations

import json

from ledgerhound.shell.guardrails import (
    ChapterDedupeGate,
    ExecBlocklist,
    ExportGrainGate,
    ModelingGate,
    PhaseReportGate,
    PredicateConsistencyCheck,
    RestatementGuard,
    StatisticalAssumptionsGate,
    claim_tokens,
    claims_match,
)
from ledgerhound.shell.session_goal import SessionGoal

FIT = "from sklearn.linear_model import LogisticRegression\nLogisticRegression().fit(X, y)"


def test_modeling_gate_follows_the_live_goal():
    goal = SessionGoal()
    gate = ModelingGate(lambda: goal, lambda: None)
    assert gate.check(FIT).startswith("REFUSED")
    goal.modeling_authorised = True
    assert gate.check(FIT) == ""


def test_export_gate_only_applies_to_account_goals():
    frames = {"out": {"n_rows": 8, "dup_keys": {"payer_id": 6}}}
    goal = SessionGoal()
    gate = ExportGrainGate(lambda: goal, lambda: None, lambda: frames)
    assert gate.check("out.to_csv('f.csv')") == ""
    goal.unit_of_analysis, goal.entity_key = "account", "payer_id"
    goal.objective = "account features"
    assert gate.check("out.to_csv('f.csv')").startswith("REFUSED")
    assert gate.check("print(out.head())") == ""            # not an export


def test_export_gate_allows_when_frames_cannot_be_read():
    goal = SessionGoal(objective="x", unit_of_analysis="account", entity_key="payer_id")

    def broken():
        raise RuntimeError("kernel gone")
    assert ExportGrainGate(lambda: goal, lambda: None, broken).check("out.to_csv('f.csv')") == ""


def test_stat_gate_advisory_resets_each_cell():
    seen = []
    gate = StatisticalAssumptionsGate(lambda: 1000, on_issue=lambda **kw: seen.append(kw))
    code = "from scipy import stats\nstats.ttest_ind(a, b)"
    assert gate.check(code) == ""
    assert "equal_var" in gate.advisory and seen
    assert gate.check("df.describe()") == "" and gate.advisory == ""


def test_exec_blocklist():
    b = ExecBlocklist()
    assert b.check("df.describe()") == ""
    assert b.check("IMPORT OS").startswith("blocked:")       # case-insensitive


def test_claim_matching():
    known = claim_tokens("fraud concentrates in night hours between midnight and 4am")
    assert claims_match(claim_tokens("night hours midnight to 4am carry the fraud"), known)
    assert not claims_match(claim_tokens("night shift reviews alerts"), known)
    assert not claims_match(set(), known)


def test_restatement_guard_counts_and_resets():
    g = RestatementGuard()
    book = [("[full] mule accounts receive many small inbound credits",) * 2]
    restated = "mule accounts receive many small inbound credits daily"
    for _ in range(3):
        refusal, detail = g.check(restated, "", domain_profile=None,
                                  book_claims=book, team_claims=[])
        assert refusal.startswith("REFUSED")
    assert detail["other_matches"] == ["this book's history"]
    assert "NOTE: the last 3" in refusal
    assert g.check(restated, "adds the hour-of-day pattern", domain_profile=None,
                   book_claims=book, team_claims=[])[0] == ""
    g.note_recorded()
    assert g.consecutive_refusals == 0


def test_predicate_check(tmp_path):
    path = tmp_path / "findings.jsonl"
    path.write_text(json.dumps({"claim": "weekend share of fraud is elevated",
                                "predicate": "dow >= 5"}) + "\n", encoding="utf-8")
    assert PredicateConsistencyCheck.check("weekend share of fraud is elevated vs baseline",
                                           "dow == 6", path).startswith("NOTE")
    assert PredicateConsistencyCheck.check("weekend share of fraud is elevated vs baseline",
                                           "dow >= 5", path) == ""


def test_phase_report_gate():
    assert PhaseReportGate.check("p1", {}, force=False).startswith("REFUSED")
    assert PhaseReportGate.check("p1", {}, force=True) == ""
    assert PhaseReportGate.check("p1", {"p1": ["r.md"]}, force=False) == ""


def test_chapter_dedupe_refusal_shows_prior_pages():
    msg = ChapterDedupeGate.refusal("interaction-sweep",
                                    {"page_count": 2, "pages": ["amount x hour", "mcc x device"]})
    assert msg.startswith("REFUSED") and "mcc x device" in msg and "redo_chapter" in msg
