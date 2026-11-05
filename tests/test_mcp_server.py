"""The MCP server, driven the way a host agent drives it.

In-process (mcp.Client against the MCPServer): the tool surface, the order
rules, refusals, sandboxing seen through MCP, all three approval routes, and
concurrent calls. Out-of-process (a real stdio subprocess): the protocol is
the only thing on stdout, through both entry points.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp import types as mt  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402

from ledgerhound.mcp.server import LedgerhoundSession, build_server  # noqa: E402
from ledgerhound.runtime import isolation as iso  # noqa: E402

pytestmark = pytest.mark.anyio
needs_bwrap = pytest.mark.skipif(not iso.bwrap_works(), reason="bubblewrap unavailable")

PHASE2 = {"load_dataset", "profile", "run_analysis", "check_feature", "falsify",
          "record_finding", "kb_search", "report", "set_goal", "set_methodology",
          "record_data_answer", "request_modeling_authorisation"}
TOOLS = PHASE2 | {"confirm_roles", "kb_add"}
NOT_YET = {"sql", "build_sample"}
CLAIM = "fraud transactions carry far larger amounts than legitimate ones"
AUTH_Q = "request_modeling_authorisation"
ROLES_YES = "yes, these roles are right"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session(tmp_path):
    s = LedgerhoundSession(tmp_path / "runs")
    yield s
    s.tools.close()


def _events(session, kind):
    path = session.tools.run_dir / "events.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [r["payload"] for r in rows if r["type"] == kind]


def text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def call(client, name, **args) -> str:
    return text(await client.call_tool(name, args))


async def confirm_roles(client, **args) -> str:
    """The agent relays the user's yes (no elicitation on this client)."""
    r = await call(client, "confirm_roles", confirm=ROLES_YES, **args)
    assert "CONFIRMED" in r, r
    return r


def choose(answer_for):
    """An elicitation callback: a person picking `answer_for(options)`."""
    async def cb(context, params):
        opts = params.requested_schema["properties"]["choice"]["enum"]
        pick = answer_for(opts)
        if pick is None:
            return mt.ElicitResult(action="decline")
        return mt.ElicitResult(action="accept", content={"choice": pick})
    return cb


# -- the surface ------------------------------------------------------------ #

async def test_exactly_the_current_tools(session):
    async with Client(build_server(session)) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert names == TOOLS and not names & NOT_YET


async def test_tool_schemas_fit_the_token_budget(session):
    # Design target: under 4,000 tokens of fixed overhead. Approximated as
    # characters / 4 (no tokenizer dependency), with headroom.
    async with Client(build_server(session)) as c:
        tools = (await c.list_tools()).tools
    blob = json.dumps([t.model_dump(mode="json", exclude_none=True) for t in tools])
    assert len(blob) / 4 < 3000, len(blob)


async def test_resources_are_listed(session):
    async with Client(build_server(session)) as c:
        uris = {str(r.uri) for r in (await c.list_resources()).resources}
        iso_doc = json.loads((await c.read_resource("ledgerhound://isolation")).contents[0].text)
    assert {"ledgerhound://findings", "ledgerhound://isolation", "ledgerhound://roles"} <= uris
    assert iso_doc == {"level": "no kernel running"}


# -- order rules and refusals ----------------------------------------------- #

async def test_nothing_works_before_load(session):
    async with Client(build_server(session)) as c:
        assert "no data loaded" in await call(c, "profile")
        assert "no data loaded" in await call(c, "run_analysis", code="print(1)")
        assert (await call(c, "report", title="t", markdown="m")).startswith("REFUSED")


async def test_record_finding_requires_falsify_first(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        await confirm_roles(c)
        r = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        assert r.startswith("REFUSED") and "falsify" in r
        assert not session.recorded_findings()
        plan = await call(c, "falsify", claim=CLAIM)
        assert "subgroup" in plan.lower() or "route" in plan.lower()
        # Same claim, different spacing and case, still counts as falsified.
        r = await call(c, "record_finding", claim="  Fraud transactions carry FAR larger "
                       "amounts than legitimate ones", sample_frame="population_estimate")
        assert r.startswith("recorded finding"), r
        assert (await call(c, "record_finding", claim="night-time share predicts fraud",
                           sample_frame="unknown")).startswith("REFUSED")


async def test_empty_check_feature_is_refused(session):
    async with Client(build_server(session)) as c:
        assert "at least one feature" in await call(c, "check_feature", features=[], label="is_fraud")


async def test_kb_search_finds_recorded_findings(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        before = await call(c, "kb_search", query="larger amounts legitimate")
        assert "this session" not in before
        await confirm_roles(c)
        await call(c, "falsify", claim=CLAIM)
        await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        hit = await call(c, "kb_search", query="larger amounts legitimate")
        assert "[finding · this session] (from data)" in hit and "larger amounts" in hit
        # the finding outranks knowledge-pack text that merely says "amounts"
        assert hit.index("this session") < hit.index("unverified)") if "unverified)" in hit else True
        assert "a query" in await call(c, "kb_search", query="the and of")


# -- the full flow, sandboxed ------------------------------------------------ #

@needs_bwrap
async def test_full_flow_sandboxed(session, fraud_csv):
    async with Client(build_server(session)) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
        assert "isolation: sandboxed" in loaded
        assert "rows:" in (await call(c, "profile")).lower()
        await confirm_roles(c)
        assert "goal" in (await call(c, "set_goal", objective="find fraud drivers",
                                     unit_of_analysis="transaction",
                                     label_column="is_fraud")).lower()
        assert "550.0" in await call(c, "run_analysis",
                                     code="print(df.groupby('is_fraud').amount.mean().round(1).to_dict())")
        fit = await call(c, "run_analysis",
                         code="from sklearn.linear_model import LogisticRegression\n"
                              "LogisticRegression().fit(df[['amount']], df.is_fraud)")
        assert fit.startswith("REFUSED")
        net = await call(c, "run_analysis",
                         code="import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)"
                              "; print('CONNECTED')\nexcept OSError:\n    print('BLOCKED')")
        assert "BLOCKED" in net and "CONNECTED" not in net
        feat = await call(c, "check_feature", features=["amount"], label="is_fraud")
        assert '"strength"' in feat and '"leakage"' in feat
        await call(c, "falsify", claim=CLAIM)
        await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate",
                   evidence="mean amount 550.0 fraud vs far lower legit")
        saved = await call(c, "report", title="drivers", markdown="Fraud is in the big tickets.")
        report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
        found = json.loads((await c.read_resource("ledgerhound://findings")).contents[0].text)
        level = json.loads((await c.read_resource("ledgerhound://isolation")).contents[0].text)["level"]
    assert CLAIM in report and "isolation: sandboxed" in report
    # the only agent-relayed decision is the roles confirmation
    assert "Column roles confirmed by **the agent, not verified by a person**" in report
    assert report.count(" -> ") == 1
    assert len(found) == 1 and level == "sandboxed"


async def test_partial_isolation_is_reported_and_warned(monkeypatch, session, fraud_csv):
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    async with Client(build_server(session)) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
        assert "isolation: partial" in loaded and "network open" in loaded
        await confirm_roles(c)
        await call(c, "falsify", claim=CLAIM)
        await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        saved = await call(c, "report", title="t", markdown="m")
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "**Warning:** the kernel was not fully sandboxed" in report


async def test_require_isolation_refuses_when_unavailable(monkeypatch, tmp_path, fraud_csv):
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    s = LedgerhoundSession(tmp_path / "runs", require_isolation=True)
    try:
        async with Client(build_server(s)) as c:
            assert "REFUSED" in await call(c, "load_dataset", path=str(fraud_csv))
            assert (await call(c, "run_analysis", code="print(1)")).startswith("REFUSED")
    finally:
        s.tools.close()


async def test_calls_are_serialised(session, fraud_csv):
    import anyio
    out = {}
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))

        async def run(i):
            out[i] = await call(c, "run_analysis",
                                code=f"import time\nx{i} = {i}\ntime.sleep(0.5)\nprint('done', x{i})")
        async with anyio.create_task_group() as tg:
            for i in range(3):
                tg.start_soon(run, i)
    assert all(f"done {i}" in out[i] for i in range(3)), out


# -- approvals over MCP ------------------------------------------------------ #

async def test_no_elicitation_relays_and_flags_the_agent(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        first = await call(c, AUTH_Q, summary="features ready")
        assert first.startswith("NEEDS USER DECISION")
        assert session.tools.goal.modeling_authorised is False
        second = await call(c, AUTH_Q, summary="features ready", confirm="authorise modelling")
        assert "relayed by the agent" in second
        await confirm_roles(c)
        await call(c, "falsify", claim=CLAIM)
        await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        saved = await call(c, "report", title="t", markdown="m")
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "approved by the agent, not verified by a person" in report


# "auto" negotiates 2026-07-28 (no back-channel: input_required round trip);
# "legacy" is the handshake protocol (server-initiated elicitation mid-call).
MODES = pytest.mark.parametrize("mode", ["auto", "legacy"])


def _authorise_or_skip_methodology(opts):
    return (next((o for o in opts if o.startswith("authorise")), None)
            or next((o for o in opts if o.startswith("no methodology")), opts[-1]))


@MODES
async def test_elicited_approval_is_the_persons(mode, session, fraud_csv):
    asked = []

    def person(opts):
        asked.append(opts)
        return _authorise_or_skip_methodology(opts)
    async with Client(build_server(session), mode=mode,
                      elicitation_callback=choose(person)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, AUTH_Q, summary="features ready")
    assert any(o[0].startswith("authorise") for o in asked), asked
    assert "AUTHORISED by the analyst" in r
    assert session.tools.goal.modeling_authorised is True
    assert not session.tools.agent_approvals()
    gate_log = session.gate.log
    assert [d.approved_by for d in gate_log if d.level == "choose"][-1] == "person"


@MODES
async def test_declined_elicitation_does_not_authorise(mode, session, fraud_csv):
    async with Client(build_server(session), mode=mode,
                      elicitation_callback=choose(lambda opts: None)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, AUTH_Q, summary="features ready")
    assert "NOT authorised" in r and session.tools.goal.modeling_authorised is False


@MODES
async def test_elicitation_error_never_authorises(mode, session, fraud_csv):
    async def broken(context, params):
        return mt.ErrorData(code=-32603, message="client cannot show forms right now")
    async with Client(build_server(session), mode=mode, elicitation_callback=broken) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        if mode == "legacy":
            # The server sees the failed elicit and relays to the agent.
            r = await call(c, AUTH_Q, summary="features ready")
            assert r.startswith("NEEDS USER DECISION"), r
        else:
            # 2026-07-28: the CLIENT's form failed, so the client fails the
            # call and never retries -- no answer ever reaches the server.
            with pytest.raises(MCPError, match="cannot show forms"):
                await call(c, AUTH_Q, summary="features ready")
    assert session.tools.goal.modeling_authorised is False
    assert not [d for d in session.gate.log
                if d.level == "choose" and d.approved_by == "person"]


async def test_confirm_wins_over_input_required(session, fraud_csv):
    # An agent that already relayed an answer is recorded as the agent's,
    # even on a client that could have asked the person.
    async with Client(build_server(session), elicitation_callback=choose(lambda o: o[0])) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, AUTH_Q, summary="x", confirm="authorise modelling")
    assert "relayed by the agent" in r
    assert [d.approved_by for d in session.tools.agent_approvals()] == ["agent"]


async def test_methodology_is_elicited_on_load_legacy(session, fraud_csv):
    seen = []

    def pick(opts):
        seen.append(opts)
        return next(o for o in opts if o.startswith("topdown_7phase"))
    async with Client(build_server(session), mode="legacy",
                      elicitation_callback=choose(pick)) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
    assert seen and "[METHODOLOGY]" in loaded and "none adopted" not in loaded
    assert session.tools.methodology is not None
    assert any(e.get("chosen_by") == "analyst" for e in _events(session, "methodology_set"))


async def test_methodology_on_modern_connection_asks_the_agent_to_ask(session, fraud_csv):
    # No back-channel mid-load: never guessed, and the agent is told to ask.
    async with Client(build_server(session), elicitation_callback=choose(lambda o: o[0])) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
    assert "none adopted" in loaded and "set_methodology" in loaded
    assert session.tools.methodology is None


# -- a real stdio process ---------------------------------------------------- #

def _entry_points():
    bindir = Path(sys.executable).parent
    eps = [("python -m", [sys.executable, "-m", "ledgerhound.mcp"])]
    for exe in (bindir / "ledgerhound", bindir / "ledgerhound.exe"):
        if exe.exists():
            eps.append(("ledgerhound mcp", [str(exe), "mcp"]))
            break
    return eps


@pytest.mark.parametrize("label,cmd", _entry_points(), ids=[e[0] for e in _entry_points()])
def test_stdout_carries_only_protocol(label, cmd, tmp_path, fraud_csv):
    """Every byte on stdout must be JSON-RPC, even while the server prints
    (rich 'Kernel ready' during load) and a kernel cell writes straight to
    file descriptor 1."""
    def msg(i, method, params=None):
        m = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if i is not None:
            m["id"] = i
        return json.dumps(m) + "\n"
    noise = ("import os\nos.write(1, b'RAW-FD1-NOISE\\n')\nprint('cell print')\n"
             "import sys; sys.__stdout__.write('DUNDER-STDOUT-NOISE\\n'); sys.__stdout__.flush()")
    stdin = "".join([
        msg(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "t", "version": "0"}}),
        msg(None, "notifications/initialized"),
        msg(2, "tools/list"),
        msg(3, "tools/call", {"name": "load_dataset", "arguments": {"path": str(fraud_csv)}}),
        msg(4, "tools/call", {"name": "run_analysis", "arguments": {"code": noise}}),
        msg(5, "tools/call", {"name": "profile", "arguments": {}}),
    ])
    p = subprocess.Popen(cmd + ["--out-root", str(tmp_path / "runs")], cwd=tmp_path,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(stdin.encode())
    p.stdin.flush()
    got, lines, deadline = {}, [], time.monotonic() + 120
    while len(got) < 5 and time.monotonic() < deadline:
        line = p.stdout.readline()
        if not line:
            break
        lines.append(line)
        m = json.loads(line)              # any non-JSON byte fails the test here
        assert m.get("jsonrpc") == "2.0", line
        if "id" in m:
            got[m["id"]] = m
    p.stdin.close()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    rest = p.stdout.read()
    err = p.stderr.read().decode(errors="replace")
    assert set(got) == {1, 2, 3, 4, 5}, (label, got.keys(), err[-2000:])
    assert not rest.strip() or all(json.loads(x) for x in rest.splitlines() if x.strip())
    out_text = b"".join(lines).decode()
    # The kernel cell's fd-1 write never appears as a raw line. On Linux
    # ipykernel captures it into the cell's output (inside a JSON result); on
    # Windows it is not captured and lands on the kernel's inherited fd 1,
    # which protect_stdout() pointed at stderr. Either way it happened, and
    # either way it stayed off the wire.
    assert not any(ln.strip() in (b"RAW-FD1-NOISE", b"DUNDER-STDOUT-NOISE") for ln in lines)
    assert "RAW-FD1-NOISE" in json.dumps(got[4]["result"]) or "RAW-FD1-NOISE" in err
    # Server-side prints really happened -- on stderr, not nowhere.
    assert "Kernel ready" not in out_text and "Kernel ready" in err
    names = {t["name"] for t in got[2]["result"]["tools"]}
    assert names == TOOLS
    assert "cell print" in json.dumps(got[4]["result"])
    assert "rows:" in json.dumps(got[5]["result"]).lower()
    assert "Compute Strategy" not in out_text and "Compute Strategy" in err


def test_protect_stdout_diverts_every_writer():
    # print, sys.__stdout__, a raw fd-1 write and a child process all land on
    # stderr; only bytes written to the wire fd reach stdout.
    code = (
        "import os, sys, subprocess\n"
        "from ledgerhound.mcp._stdio import protect_stdout\n"
        "wire = protect_stdout()\n"
        "print('PRINT'); sys.stdout.flush()\n"
        "sys.__stdout__.write('DUNDER\\n'); sys.__stdout__.flush()\n"
        "os.write(1, b'RAWFD\\n')\n"
        "subprocess.run([sys.executable, '-c', 'print(\"CHILD\")'])\n"
        "os.write(wire, b'WIRE\\n')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    assert r.stdout == b"WIRE\n", r.stdout
    for noise in (b"PRINT", b"DUNDER", b"RAWFD", b"CHILD"):
        assert noise in r.stderr


def test_sdk_stdio_client_end_to_end(tmp_path, fraud_csv):
    import anyio

    async def go():
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "ledgerhound.mcp", "--out-root",
                                             str(tmp_path / "runs")],
                                       cwd=str(tmp_path), env=dict(os.environ))
        async with Client(params) as c:
            assert {t.name for t in (await c.list_tools()).tools} == TOOLS
            assert (await call(c, "load_dataset", path=str(fraud_csv))).startswith("loaded")
            return await call(c, "run_analysis",
                              code="print(df.groupby('is_fraud').amount.mean().round(1).to_dict())")
    assert "550.0" in anyio.run(go)


async def test_broader_claim_than_the_falsified_one_is_refused(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        await confirm_roles(c)
        await call(c, "falsify", claim=CLAIM)
        broader =(CLAIM + ", because night-time device changes and new payees "
                   "drive mule cash-out velocity")
        r = await call(c, "record_finding", claim=broader, sample_frame="unknown")
    assert r.startswith("REFUSED"), r


def test_unsolicited_input_responses_are_ignored():
    from types import SimpleNamespace

    from ledgerhound.mcp.server import _DECISION_KEY, _person_answer
    yes = mt.ElicitResult(action="accept", content={"choice": "authorise modelling"})
    asked = SimpleNamespace(request_state=_DECISION_KEY, input_responses={_DECISION_KEY: yes})
    forged = SimpleNamespace(request_state=None, input_responses={_DECISION_KEY: yes})
    assert _person_answer(asked) == "authorise modelling"
    assert _person_answer(forged) is None
    no = mt.ElicitResult(action="decline")
    assert _person_answer(SimpleNamespace(request_state=_DECISION_KEY,
                                          input_responses={_DECISION_KEY: no})) == ""


@MODES
async def test_cancelled_elicitation_is_not_a_persons_decision(mode, session, fraud_csv):
    # `claude -p` and other clients that cannot show a form reply "cancel".
    # Nobody chose: not authorised, not recorded as a person, and the agent
    # is handed the question to relay.
    async def cancel(context, params):
        return mt.ElicitResult(action="cancel")
    async with Client(build_server(session), mode=mode, elicitation_callback=cancel) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, AUTH_Q, summary="features ready")
        assert r.startswith("NEEDS USER DECISION"), r
        assert session.tools.goal.modeling_authorised is False
        assert not [d for d in session.gate.log
                    if d.level == "choose" and d.approved_by == "person"]
        r2 = await call(c, AUTH_Q, summary="features ready", confirm="authorise modelling")
    assert "relayed by the agent" in r2
    assert [d.approved_by for d in session.tools.agent_approvals()] == ["agent"]


@MODES
async def test_declined_elicitation_is_not_claimed_as_a_person(mode, session, fraud_csv):
    # Codex with approval "never" declines every form unseen. A decline is
    # safe (nothing authorised) but is never counted as a person's decision.
    async with Client(build_server(session), mode=mode,
                      elicitation_callback=choose(lambda opts: None)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, AUTH_Q, summary="features ready")
        assert "NOT authorised" in r
        # A declining client cannot confirm roles either: settle them the
        # way an earlier session would have, outside this client.
        roles_r = await call(c, "confirm_roles")
        assert "NOT confirmed" in roles_r
        session.tools.roles.settle("person")
        await call(c, "falsify", claim=CLAIM)
        await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        saved = await call(c, "report", title="t", markdown="m")
    choices = [d for d in session.gate.log if d.level == "choose"]
    assert choices and all(d.approved_by == "declined via client" for d in choices)
    assert not [d for d in session.gate.log if d.approved_by == "person"]
    replies = [e for e in _events(session, "hitl") if e.get("tier") == "client_reply"]
    assert replies and all(e["action"] == "decline" for e in replies)
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "Decisions made by a person" not in report
    assert "declined through the client's form" in report


def test_answer_mapping():
    from ledgerhound.mcp.server import _answer_for
    assert _answer_for("accept", "authorise modelling") == "authorise modelling"
    assert _answer_for("decline", None) == ""
    assert _answer_for("cancel", None) is None
    assert _answer_for("", None) is None


async def test_kernel_helpers_load_and_are_reported_present(session, fraud_csv):
    async with Client(build_server(session)) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, "run_analysis", code=(
            "print('HAVE', all(n in globals() for n in "
            "('check_leakage', 'evaluate_feature', 'evaluate_features')))"))
    assert "HAVE True" in r
    assert "helpers (evaluate_feature/check_leakage) failed to load" not in loaded
    assert "unavailable" not in loaded


# -- column roles (Phase 3 step 1) ------------------------------------------ #

async def test_role_dependent_steps_refuse_until_roles_are_confirmed(session, fraud_csv):
    async with Client(build_server(session)) as c:
        loaded = await call(c, "load_dataset", path=str(fraud_csv))
        assert "column roles (proposed, NOT confirmed)" in loaded
        assert "is_fraud=label" in loaded
        # exploring is allowed
        assert "550.0" in await call(c, "run_analysis",
                                     code="print(df.groupby('is_fraud').amount.mean().round(1).to_dict())")
        feat = await call(c, "check_feature", features=["amount"], label="is_fraud")
        assert '"strength"' in feat and "column roles are not confirmed" in feat
        await call(c, "falsify", claim=CLAIM)
        r = await call(c, "record_finding", claim=CLAIM, sample_frame="population_estimate")
        assert r.startswith("REFUSED") and "confirm_roles" in r
        assert not session.recorded_findings()
        first = await call(c, "confirm_roles")
        assert first.startswith("NEEDS USER DECISION")
        assert "is_fraud -> label" in first and "label_time -> not in the data" in first
        assert not session.tools.roles_confirmed
        await confirm_roles(c)
        assert (await call(c, "record_finding", claim=CLAIM,
                           sample_frame="population_estimate")).startswith("recorded finding")
        saved = await call(c, "report", title="t", markdown="m")
    report = Path(re.search(r"-> (\S+)", saved).group(1)).read_text(encoding="utf-8")
    assert "Column roles confirmed by **the agent, not verified by a person**" in report
    # run_analysis and check_feature each ran a kernel cell before confirmation
    assert "2 analysis cell(s) ran before the column roles were confirmed" in report
    assert "label maturity is an assumption, not a check" in report


@MODES
async def test_roles_confirmed_by_a_person_through_elicitation(mode, session, fraud_csv):
    def person(opts):
        if "yes, these roles are right" in opts:
            return "yes, these roles are right"
        return next((o for o in opts if o.startswith("no methodology")), opts[-1])
    async with Client(build_server(session), mode=mode,
                      elicitation_callback=choose(person)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        r = await call(c, "confirm_roles")
        doc = json.loads((await c.read_resource("ledgerhound://roles")).contents[0].text)
    assert "CONFIRMED by the user" in r
    assert doc["confirmed_by"] == "person"
    assert doc["columns"]["is_fraud"] == {"role": "label", "confidence": "high",
                                          "reason": doc["columns"]["is_fraud"]["reason"],
                                          "status": "person"}
    assert doc["absent"]["label_time"] == "person"


async def test_label_roles_never_settle_without_asking(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        rm = session.tools.roles
    assert rm.columns["is_fraud"].status == "proposed"
    assert rm.columns["amount"].status == "auto"
    assert not rm.confirmed


async def test_check_feature_refuses_label_and_wrong_label_once_confirmed(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        await confirm_roles(c)
        r = await call(c, "check_feature", features=["is_fraud"], label="is_fraud")
        assert r.startswith("REFUSED") and "role label" in r
        r = await call(c, "check_feature", features=["amount"], label="hour")
        assert r.startswith("REFUSED") and "not the confirmed label" in r
        ok = await call(c, "check_feature", features=["amount"], label="is_fraud")
        assert '"leakage"' in ok and "not confirmed" not in ok


async def test_role_corrections_are_asked_again(session, fraud_csv):
    async with Client(build_server(session)) as c:
        await call(c, "load_dataset", path=str(fraud_csv))
        await confirm_roles(c)
        bad = await call(c, "confirm_roles", roles={"hour": "nonsense"})
        assert "is not a role" in bad and session.tools.roles_confirmed
        again = await call(c, "confirm_roles", roles={"hour": "event_time"})
        assert again.startswith("NEEDS USER DECISION") and "hour -> event_time" in again
        # until the person confirms the correction, the confirmed map stands
        assert session.tools.roles_confirmed and session.tools.roles.role_of("hour") == "feature"
        await confirm_roles(c, roles={"hour": "event_time"})
    assert session.tools.roles.role_of("hour") == "event_time"


async def test_confirmed_roles_carry_to_the_next_session(tmp_path, fraud_csv):
    s1 = LedgerhoundSession(tmp_path / "runs1")
    try:
        async with Client(build_server(s1)) as c:
            await call(c, "load_dataset", path=str(fraud_csv))
            await confirm_roles(c)
    finally:
        s1.tools.close()
    s2 = LedgerhoundSession(tmp_path / "runs2")
    try:
        async with Client(build_server(s2)) as c:
            loaded = await call(c, "load_dataset", path=str(fraud_csv))
            assert "confirmed in an earlier session by the agent (unverified)" in loaded
            assert s2.tools.roles_confirmed
    finally:
        s2.tools.close()


# -- the Knowledge Pack (Phase 3 step 2) ------------------------------------ #

async def test_kb_add_validates_and_scopes(session, tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGERHOUND_TEAM_KB_DIR", str(tmp_path / "team"))
    rule = {"type": "rule", "name": "new payee burst", "logic": "3+ payments to a new payee in 1h",
            "output_column": "rule_hit_9"}
    async with Client(build_server(session)) as c:
        bad = await call(c, "kb_add", entry={"type": "rule", "name": "x"})
        assert bad.startswith("REFUSED") and "logic" in bad
        assert "exactly one of" in await call(c, "kb_add")
        ok = await call(c, "kb_add", entry=rule)
        assert ok.startswith("added rule new_payee_burst in the project pack")
        team = await call(c, "kb_add", entry=rule, scope="team")
        assert team.startswith("NEEDS USER DECISION") and "TEAM knowledge pack" in team
        assert not (tmp_path / "team" / "entries.jsonl").exists()
        team = await call(c, "kb_add", entry=rule, scope="team", confirm="add to the team pack")
        assert "in the team pack" in team
        doc = tmp_path / "typology_note.md"
        doc.write_text("Mule accounts receive from many payers and pay out within hours.")
        added = await call(c, "kb_add", document=str(doc), source="internal note")
        assert added.startswith("added document")
    from ledgerhound.kb import KnowledgePack
    line = json.loads((tmp_path / "team" / "entries.jsonl").read_text().splitlines()[0])
    assert line["added_by"] == "agent, team add approved by agent"
    assert KnowledgePack().get("rule", "new_payee_burst").scope == "project"
