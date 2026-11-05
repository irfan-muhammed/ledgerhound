"""build_sample: roles-driven, refuses until label lag and prevalence are
settled, and certifies how the sample was made."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ledgerhound.pipeline.build import BuildRefusal, parse_assumption, parse_prevalence

mcp = pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")

import mcp.types as mt  # noqa: E402
from mcp import Client  # noqa: E402

from ledgerhound.kb import KnowledgePack  # noqa: E402
from ledgerhound.mcp.server import LedgerhoundSession, build_server  # noqa: E402

pytestmark = pytest.mark.anyio
ROLES_YES = "yes, these roles are right"
LAG_YES = "yes, assume this label lag"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session(tmp_path):
    s = LedgerhoundSession(tmp_path / "runs")
    yield s
    s.tools.close()


@pytest.fixture
def paysim_like(tmp_path) -> Path:
    """Hour steps, a payer entity, no label time (the PaySim shape)."""
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({
        "step": rng.integers(0, 24 * 60, n),
        "nameOrig": [f"C{i}" for i in rng.integers(0, 900, n)],
        "amount": rng.gamma(2.0, 300.0, n).round(2),
        "isFraud": (rng.random(n) < 0.03).astype(int)})
    p = tmp_path / "paysim_like.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def labelled_later(tmp_path) -> Path:
    """Real timestamps and a report timestamp for each fraud."""
    rng = np.random.default_rng(11)
    n = 3000
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24, n), unit="h")
    fraud = rng.random(n) < 0.04
    lag = pd.to_timedelta(rng.integers(5, 40, n), unit="D")
    df = pd.DataFrame({
        "txn_ts": ts, "card_id": [f"K{i}" for i in rng.integers(0, 700, n)],
        "amount": rng.gamma(2.0, 100.0, n).round(2), "is_fraud": fraud.astype(int),
        "fraud_reported_at": pd.Series(ts + lag).where(fraud),
        "fraud_source": np.where(fraud, "chargeback", None)})
    df = df.sort_values("txn_ts")
    p = tmp_path / "cards_reported.csv"
    df.to_csv(p, index=False)
    return p


def text(result) -> str:
    if isinstance(result, mt.CallToolResult):
        return "\n".join(getattr(c, "text", "") for c in result.content)
    return str(result)


async def call(c, name, **args) -> str:
    return text(await c.call_tool(name, args))


def choose(pick):
    async def cb(context, params):
        opts = params.requested_schema["properties"]["choice"]["enum"]
        return mt.ElicitResult(action="accept", content={"choice": pick(opts)})
    return cb


# -- pure parts -------------------------------------------------------------- #

def test_assumption_and_prevalence_parsing():
    assert parse_assumption("final") == 0
    assert parse_assumption("45 days") == 45
    with pytest.raises(BuildRefusal):
        parse_assumption("policy")
    with pytest.raises(BuildRefusal):
        parse_assumption("soon")
    assert parse_prevalence("0.1", 0.02) == (0.1, "declared")
    assert parse_prevalence("99", 0.02)[0] == pytest.approx(0.01)
    assert parse_prevalence("natural", 0.02)[0] == 0.02
    with pytest.raises(BuildRefusal, match="natural fraud rate"):
        parse_prevalence("", 0.02)


# -- over MCP ---------------------------------------------------------------- #

async def test_refuses_until_roles_lag_unit_and_prevalence_are_settled(session, paysim_like):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(paysim_like))
        r = await call(c, "build_sample")
        assert r.startswith("REFUSED") and "not confirmed" in r
        await call(c, "confirm_roles", roles={"step": "event_time", "nameOrig": "payer"},
                   confirm=ROLES_YES)
        r = await call(c, "build_sample", label_lag="7", confirm=LAG_YES)
        assert r.startswith("REFUSED") and "time_unit" in r     # asked before the person is
        r = await call(c, "build_sample", time_unit="hours")
        assert r.startswith("REFUSED") and "no label-time column" in r
        r = await call(c, "build_sample", label_lag="7", time_unit="hours", confirm=LAG_YES)
        assert r.startswith("REFUSED") and "declare the design prevalence" in r
        assert session.tools.certificate is None


async def test_assumed_lag_route_certifies_and_says_it_is_an_assumption(session, paysim_like):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(paysim_like))
        await call(c, "confirm_roles", roles={"step": "event_time", "nameOrig": "payer"},
                   confirm=ROLES_YES)
        r = await call(c, "build_sample", label_lag="7", time_unit="hours",
                       design_prevalence="0.1", confirm=LAG_YES)
        assert "sample built -- CERTIFIED" in r, r
        assert "split=grouped_temporal, entity=nameOrig, time=step" in r
        assert "label maturity: 7 days, assumed" in r
        assert "ASSUMED, stated by the agent, not verified by a person" in r
        assert "in the analysis kernel as `train` and `test`" in r
        kern = await call(c, "run_analysis", code=(
            "print(round(train.isFraud.mean(), 2), "
            "len(set(train.nameOrig) & set(test.nameOrig)), int(test.step.max()))"))
        rate, shared, last = kern.split()[:3]
        assert rate == "0.1" and shared == "0"
        assert int(last) < 24 * 60 - 7 * 24          # the last 7 days are kept out of test
        cert = json.loads((await c.read_resource("ledgerhound://certificate")).contents[0].text)
        assert cert["can_certify"] is True
        mat = cert["sections"]["label_maturity"]
        assert mat["status"] == "WARNING" and mat["detail"]["basis"] == "assumed"
        assert "label maturity: assumed, not measured" in " ".join(cert["not_checked"])
        assert "relative" in cert["notes"]
        await call(c, "falsify", claim="fraud is concentrated in a few payers")
        await call(c, "record_finding", claim="fraud is concentrated in a few payers",
                   sample_frame="population_estimate")
        saved = await call(c, "report", title="t", markdown="m")
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "label maturity 7 days, **assumed** (no label-time column)" in report
    assert "certified, certificate" in report


async def test_person_confirms_the_assumption_by_elicitation(session, paysim_like):
    asked = []

    def person(opts):
        asked.append(opts)
        return opts[0]
    async with Client(build_server(session), elicitation_callback=choose(person)) as c:
        await call(c, "load_dataset", path=str(paysim_like))
        await call(c, "confirm_roles", roles={"step": "event_time", "nameOrig": "payer"})
        r = await call(c, "build_sample", label_lag="final", time_unit="hours",
                       design_prevalence="natural")
    assert any(LAG_YES in o for o in asked)
    assert "stated by a person" in r and "label maturity: 0 days, assumed" in r


async def test_label_time_is_measured_and_checked_against_the_pack(session, labelled_later):
    KnowledgePack().add({"type": "label_policy", "label_source": "chargeback",
                         "maturity_days": 20})
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(labelled_later))
        await call(c, "confirm_roles", roles={"fraud_reported_at": "label_time",
                                              "fraud_source": "label_source",
                                              "card_id": "card", "txn_ts": "event_time"},
                   confirm=ROLES_YES)
        r = await call(c, "build_sample", design_prevalence="natural", split="temporal")
        assert "sample built -- CERTIFIED" in r, r
        assert re.search(r"label maturity: (3[6-9]|40) days, measured", r)
        assert "policy says 20 days" in r          # the pack's policy is shorter than the data
        cert = session.tools.certificate.to_dict()
        assert cert["sections"]["label_maturity"]["detail"]["basis"] == "measured"
        assert "not assumed" not in r and "ASSUMED" not in r


async def test_an_agent_cannot_skip_the_split_rules(session, paysim_like):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(paysim_like))
        await call(c, "confirm_roles", roles={"step": "feature", "nameOrig": "feature"},
                   confirm=ROLES_YES)
        r = await call(c, "build_sample", label_lag="0", design_prevalence="natural",
                       confirm=LAG_YES)
    assert r.startswith("REFUSED") and "no honest split" in r
