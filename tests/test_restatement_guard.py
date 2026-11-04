"""record_finding refuses to re-record something already known unless the
model states what the new claim adds (a delta)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

KNOWN = "[full] fraud transactions concentrate in night hours between midnight and 4am"
RESTATED = "fraud transactions concentrate in night hours, mostly midnight to 4am"
UNRELATED = "merchant category 5967 has triple the chargeback rate of grocery"


@pytest.fixture
def analyst(make_analyst):
    a = make_analyst()
    # The dataset card carries findings from earlier sessions on this book.
    a._card = SimpleNamespace(finding_refs=[KNOWN], add_finding=lambda ref: None)
    return a


def _findings(a):
    if not a._findings_path.exists():
        return []
    return [json.loads(line) for line in a._findings_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_restatement_without_delta_is_refused(analyst):
    out = analyst._tool_record_finding(RESTATED, sample_frame="full")
    assert out.startswith("REFUSED (not recorded)")
    assert "this book's history" in out
    assert _findings(analyst) == []
    blocked = [e for e in analyst._events.load() if e.type == "finding_blocked_as_restatement"]
    assert len(blocked) == 1
    assert not [e for e in analyst._events.load() if e.type == "finding"]


def test_restatement_with_delta_is_recorded(analyst):
    out = analyst._tool_record_finding(
        RESTATED, sample_frame="full",
        delta="holds at account level: 80% of fraud payers transact only at night")
    assert out.startswith("recorded finding [full]")
    assert "with stated delta" in out
    recs = _findings(analyst)
    assert len(recs) == 1 and recs[0]["claim"] == RESTATED
    assert "[delta:" in recs[0]["evidence"]


def test_blank_delta_counts_as_no_delta(analyst):
    out = analyst._tool_record_finding(RESTATED, sample_frame="full", delta="   ")
    assert out.startswith("REFUSED")


def test_new_claim_is_recorded(analyst):
    out = analyst._tool_record_finding(UNRELATED, sample_frame="full", evidence="3.1% vs 1.0%")
    assert out.startswith("recorded finding [full]")
    assert len(_findings(analyst)) == 1


def test_passing_mention_is_not_a_restatement(analyst):
    # Shares a couple of words with the known finding but not its substance.
    out = analyst._tool_record_finding("night shift analysts review fraud alerts",
                                       sample_frame="full")
    assert out.startswith("recorded finding")


def test_team_knowledge_base_match_is_refused(make_analyst):
    a = make_analyst()
    a.project = SimpleNamespace(root="proj")      # team KB claims need a project
    a._team_findings = [{"statement": KNOWN.removeprefix("[full] "), "name": "night skew",
                         "_published_by": "teammate"}]
    out = a._tool_record_finding(RESTATED, sample_frame="full")
    assert out.startswith("REFUSED") and "team knowledge base" in out


def test_stall_note_after_three_refusals(analyst):
    for _ in range(2):
        assert "NOTE: the last" not in analyst._tool_record_finding(RESTATED, sample_frame="full")
    third = analyst._tool_record_finding(RESTATED, sample_frame="full")
    assert "NOTE: the last 3 finding attempts" in third


def test_predicate_mismatch_is_flagged(make_analyst):
    a = make_analyst()
    a._tool_record_finding("weekend share of fraud transactions is elevated", "full",
                           predicate="dayofweek >= 5")
    out = a._tool_record_finding("weekend share of fraud transactions is elevated vs nonfraud",
                                 "full", predicate="dayofweek == 6", delta="vs non-fraud")
    assert "DIFFERENT predicate" in out
