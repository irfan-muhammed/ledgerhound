"""Phase 3 gate: the paths the MCP tools reach carry no UPI-specific wording
(the narrowed de-UPI scope: not graph/ or agents/)."""

from __future__ import annotations

import re

import pandas as pd

# "paise" may appear as one example of a minor unit beside "cents"; a
# question that ONLY offers rupees/paise is what this guards against.
UPI = re.compile(r"\b(upi|vpa|ifsc|npci|rupees?|crore)\b", re.IGNORECASE)


def test_prompts_and_instructions_are_payment_generic():
    from ledgerhound.mcp.server import INSTRUCTIONS
    from ledgerhound.shell.loop import _SYSTEM_PROMPT_FALLBACK, PROMPT_TOOL_PROTOCOL, SYSTEM_PROMPT
    for name, text in (("INSTRUCTIONS", INSTRUCTIONS), ("SYSTEM_PROMPT", SYSTEM_PROMPT),
                       ("fallback", _SYSTEM_PROMPT_FALLBACK),
                       ("tool protocol", PROMPT_TOOL_PROTOCOL)):
        assert not UPI.search(text), (name, UPI.search(text))


def test_falsification_and_data_questions_are_payment_generic():
    from ledgerhound.findings.falsification import plan_for
    from ledgerhound.shell.data_questions import open_questions
    plan = plan_for("fraud rows have larger amounts")
    assert not UPI.search(str(plan))
    df = pd.DataFrame({"amount": [1.0, 2.0], "is_fraud": [0, 1], "payer": ["a", "b"]})
    qs = open_questions(df)
    units = [q.text for q in qs if q.key.startswith("units:")]
    assert units and "minor units" in units[0]
    assert not any(UPI.search(q.text) for q in qs)


def test_confirmed_roles_answer_the_schema_questions(tmp_path):
    from ledgerhound.shell.tools import AnalystTools
    df = pd.DataFrame({
        "payer_id": [f"P{i % 7}" for i in range(40)], "payee_id": [f"Q{i % 5}" for i in range(40)],
        "txn_ts": pd.date_range("2026-01-01", periods=40, freq="h").astype(str),
        "settled_ts": pd.date_range("2026-01-02", periods=40, freq="h").astype(str),
        "amount": [float(i) for i in range(40)], "is_fraud": [i % 9 == 0 for i in range(40)]})
    p = tmp_path / "two_ids.csv"
    df.to_csv(p, index=False)
    t = AnalystTools(out_root=tmp_path / "runs", renderer=lambda s: None)
    try:
        t._run_tool("load_data", {"path": str(p)})
        t._run_tool("profile_data", {})
        before = {q.key for q in t._open_data_questions}
        assert {"entity", "event_time"} <= before, before

        class Yes:
            answer, approved_by = "yes, these roles are right", "person"
        t.registry.gate.choose = lambda q, opts: Yes()
        out = t._run_tool("confirm_roles", {"roles": {"txn_ts": "event_time",
                                                      "settled_ts": "feature",
                                                      "payer_id": "payer", "payee_id": "payee"}})
        assert "CONFIRMED" in out and "answer the open data question(s)" in out
        after = {q.key for q in t._open_data_questions}
        assert not after & {"entity", "event_time", "label"}
        t._run_tool("profile_data", {})                  # a later profile keeps them closed
        assert not {q.key for q in t._open_data_questions} & {"entity", "event_time"}
    finally:
        t.close()
