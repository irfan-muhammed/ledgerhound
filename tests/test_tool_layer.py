"""The tool layer stands on its own: a full session with no LLM, the way an
MCP host agent will drive it."""

from __future__ import annotations

from ledgerhound.analyst.hitl import HITLGate
from ledgerhound.runtime.events import EventStream
from ledgerhound.shell.conversation import ConversationalAnalyst
from ledgerhound.shell.loop import ConversationLoop
from ledgerhound.shell.tools import TOOL_SPECS, AnalystTools


def test_tools_run_without_an_llm(tmp_path, fraud_csv):
    s = AnalystTools(out_root=tmp_path / "runs", renderer=lambda _: None,
                     gate=HITLGate(prompt=lambda q, o: o[-1]))
    assert not hasattr(s, "messages")
    assert s._run_tool("load_data", {"path": str(fraud_csv)}).startswith("loaded")
    assert "550.0" in s._run_tool("run_python", {
        "code": "print(df.groupby('is_fraud').amount.mean().round(1).to_dict())"})
    assert s._run_tool("run_python", {
        "code": "from sklearn.linear_model import LogisticRegression\n"
                "LogisticRegression().fit(df[['amount']], df.is_fraud)"}).startswith("REFUSED")
    assert s._run_tool("record_finding", {
        "claim": "fraud amounts dwarf non-fraud amounts", "sample_frame": "full"}
    ).startswith("recorded finding")
    assert EventStream(s.run_dir / "events.jsonl").verify().ok


def test_every_spec_has_a_handler(tmp_path):
    s = AnalystTools(out_root=tmp_path / "runs", renderer=lambda _: None)
    for spec in TOOL_SPECS:
        name = spec["function"]["name"]
        assert name in s.registry, name
        assert callable(getattr(s, f"_tool_{name}")), name


def test_analyst_is_loop_over_tools():
    assert issubclass(ConversationalAnalyst, ConversationLoop)
    assert issubclass(ConversationalAnalyst, AnalystTools)
    assert not issubclass(ConversationLoop, AnalystTools)
