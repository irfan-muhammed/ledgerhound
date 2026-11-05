"""The Knowledge Pack feeding the guardrails (design doc, "Consumed by")."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from ledgerhound.kb import KnowledgePack
from ledgerhound.kb.policy import known_claims, maturity_for, role_hints, rule_outputs

mcp = pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")

from mcp import Client  # noqa: E402

from ledgerhound.mcp.server import LedgerhoundSession, build_server  # noqa: E402

pytestmark = pytest.mark.anyio
YES = "yes, these roles are right"
CLAIM = "fraud transactions carry far larger amounts than legitimate ones"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session(tmp_path):
    s = LedgerhoundSession(tmp_path / "runs")
    yield s
    s.tools.close()


@pytest.fixture
def rules_csv(tmp_path) -> Path:
    df = pd.DataFrame({
        "txn_id": range(1, 9), "payer_id": list("AAAABBBB"),
        "amount": [10.0, 20.0, 30.0, 40.0, 500.0, 600.0, 15.0, 25.0],
        "rule_hit_17": [0, 0, 0, 0, 1, 1, 0, 0], "segment": list("xyxyxyxy"),
        "is_fraud": [0, 0, 0, 0, 1, 1, 0, 0]})
    p = tmp_path / "cards_2026.csv"
    df.to_csv(p, index=False)
    return p


def text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def call(c, name, **args) -> str:
    return text(await c.call_tool(name, args))


# -- policy functions -------------------------------------------------------- #

def test_maturity_prefers_the_named_source_then_falls_back():
    pack = KnowledgePack()
    m = maturity_for(pack, "complaint")
    assert (m.days, m.fallback, m.scope) == (30, False, "builtin")
    m = maturity_for(pack, "investigation")
    assert m.fallback and m.days == 90 and "fallback" in m.note()
    pack.add({"type": "label_policy", "label_source": "chargeback", "maturity_days": 120,
              "dataset": "cards_2026.csv"})
    assert maturity_for(pack, "chargeback", ["cards_2026.csv"]).days == 120
    assert maturity_for(pack, "chargeback", ["other.csv"]).days == 90


def test_rules_dictionary_and_known_findings_are_scoped_by_dataset():
    pack = KnowledgePack()
    pack.add({"type": "rule", "name": "r17", "logic": "x", "output_column": "rule_hit_17"})
    pack.add({"type": "rule", "name": "old", "logic": "x", "output_column": "rule_hit_3",
              "status": "retired"})
    pack.add({"type": "dictionary", "column": "segment", "meaning": "card product", "role": "channel",
              "dataset": "cards_2026"})
    pack.add({"type": "known_finding", "statement": "night-time share is already a core feature"})
    assert rule_outputs(pack) == {"rule_hit_17": "r17"}
    assert role_hints(pack, ["segment", "rule_hit_17", "amount"], ["cards_2026"]) == {
        "segment": "channel", "rule_hit_17": "rule_output"}
    assert role_hints(pack, ["segment"], ["elsewhere"]) == {}
    assert known_claims(pack)[0][1] == "night-time share is already a core feature"


# -- through the MCP server ------------------------------------------------- #

async def test_rule_output_is_refused_as_a_feature_even_before_roles(session, rules_csv):
    KnowledgePack().add({"type": "rule", "name": "r17", "logic": "amount > 400",
                         "output_column": "rule_hit_17"})
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(rules_csv))
        r = await call(c, "check_feature", features=["rule_hit_17"], label="is_fraud")
        assert r.startswith("REFUSED") and "output of rule r17" in r
        assert session.tools.roles.columns["rule_hit_17"].role == "rule_output"
        assert session.tools.roles.columns["rule_hit_17"].reason == "from your Knowledge Pack"


async def test_dictionary_roles_are_put_to_the_person(session, rules_csv):
    KnowledgePack().add({"type": "dictionary", "column": "segment", "meaning": "card product",
                         "role": "channel"})
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(rules_csv))
        q = await call(c, "confirm_roles")
    assert "segment -> channel (medium; from your Knowledge Pack)" in q


async def test_known_finding_in_the_pack_blocks_a_restatement(session, rules_csv):
    KnowledgePack().add({"type": "known_finding", "statement": CLAIM})
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(rules_csv))
        await call(c, "confirm_roles", confirm=YES)
        await call(c, "falsify", claim=CLAIM)
        r = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        assert r.startswith("REFUSED") and "knowledge pack [project]" in r
        r = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate",
                       delta="holds at payer level too, not only per transaction")
        assert r.startswith("recorded finding")


async def test_contradicting_the_pack_is_flagged_in_the_report(session, rules_csv):
    KnowledgePack().add({"type": "known_finding", "id": "small_tickets",
                         "statement": "fraud is concentrated in small-ticket card testing"})
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(rules_csv))
        await call(c, "confirm_roles", confirm=YES)
        await call(c, "falsify", claim=CLAIM)
        bad = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate",
                         kb_contradicts="known_finding/nope")
        assert bad.startswith("REFUSED") and "names no Knowledge Pack entry" in bad
        ok = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate",
                        kb_contradicts="small_tickets")
        assert "contradicting your knowledge base (known_finding/small_tickets)" in ok
        saved = await call(c, "report", title="t", markdown="m")
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "verified on data" in report
    assert "contradicts your knowledge base (known_finding/small_tickets)" in report
    line = json.loads((session.tools.run_dir / "findings.jsonl").read_text().splitlines()[0])
    assert line["kb_contradicts"] == "known_finding/small_tickets"
