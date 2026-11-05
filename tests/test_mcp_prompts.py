"""MCP prompts (/fraud-eda, /feature-review, /sample-audit) and the
dataset-card resource."""

from __future__ import annotations

import json
import re

import pytest

mcp = pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")

from mcp import Client  # noqa: E402

from ledgerhound.mcp.server import LedgerhoundSession, build_server  # noqa: E402

pytestmark = pytest.mark.anyio
UPI = re.compile(r"\b(upi|vpa|ifsc|npci|rupees?|crore)\b", re.IGNORECASE)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session(tmp_path):
    s = LedgerhoundSession(tmp_path / "runs")
    yield s
    s.tools.close()


def body(result) -> str:
    return "\n".join(m.content.text for m in result.messages)


async def test_the_three_prompts_are_listed_with_their_arguments(session):
    async with Client(build_server(session)) as c:
        prompts = {p.name: p for p in (await c.list_prompts()).prompts}
    assert set(prompts) == {"fraud-eda", "feature-review", "sample-audit"}
    args = {n: {a.name: a.required for a in (p.arguments or [])} for n, p in prompts.items()}
    assert args["fraud-eda"] == {"path": True, "goal": False}
    assert args["feature-review"] == {"features": True, "label": False}
    assert args["sample-audit"] == {"design_prevalence": False, "label_lag": False}


async def test_prompts_name_real_tools_in_the_enforced_order(session):
    async with Client(build_server(session)) as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        eda = body(await c.get_prompt("fraud-eda", {"path": "/data/q1.parquet"}))
        rev = body(await c.get_prompt("feature-review", {"features": "velocity_1h, amt_z",
                                                          "label": "is_fraud"}))
        aud = body(await c.get_prompt("sample-audit", {"design_prevalence": "0.05"}))
    order = ["load_dataset", "profile", "confirm_roles", "set_goal", "kb_search",
             "falsify", "record_finding", "report"]
    pos = [eda.index(n) for n in order]
    assert pos == sorted(pos), dict(zip(order, pos))
    assert 'load_dataset(path="/data/q1.parquet")' in eda
    assert 'check_feature(features=[...] with label="is_fraud")' in rev
    assert 'build_sample(design_prevalence="0.05")' in aud and "ledgerhound://certificate" in aud
    for text in (eda, rev, aud):
        named = set(re.findall(r"\b([a-z]+(?:_[a-z]+)+)\(", text)) | {
            w for w in re.findall(r"\b[a-z_]+\b", text) if w in tools}
        assert named <= tools | {"record_data_answer"}, named - tools
        assert "NEEDS USER DECISION" in text and not UPI.search(text)


async def test_dataset_card_resource(session, fraud_csv):
    async with Client(build_server(session)) as c:
        empty = json.loads((await c.read_resource("ledgerhound://dataset-card")).contents[0].text)
        await c.call_tool("load_dataset", {"path": str(fraud_csv)})
        await c.call_tool("profile", {})
        card = json.loads((await c.read_resource("ledgerhound://dataset-card")).contents[0].text)
        uris = {str(r.uri) for r in (await c.list_resources()).resources}
    assert empty == {"dataset": "no data loaded"}
    assert card["rows"] == 8 and "is_fraud" in card["columns"]
    assert card["roles_confirmed_by"] == "not confirmed" and "label" in card["roles"]
    assert {"ledgerhound://dataset-card", "ledgerhound://certificate",
            "ledgerhound://findings"} <= uris
