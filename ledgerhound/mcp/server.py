"""
ledgerhound.mcp.server
======================

The MCP front door: the analyst tool layer (`AnalystTools`) behind the Python
MCP SDK, so Claude Code, Codex or any MCP client drives it. The host agent is
the brain; this server supplies the data, the rules and the record.

Exposes 10 of the design's 12 tools, plus the 4 workflow tools that the order
rules need:

    load_dataset  profile  confirm_roles  run_analysis  check_feature
    falsify  record_finding  kb_search  kb_add  report
    set_goal  set_methodology  record_data_answer  request_modeling_authorisation

`sql` and `build_sample` arrive later in Phase 3.

Rules enforced here, server-side, whichever agent calls:

- Column roles must be confirmed (confirm_roles) before the steps that
  depend on them: record_finding, report, and check_feature's leakage
  check. Exploring with run_analysis is allowed before; the report counts
  those cells as "roles unconfirmed". label, label_time and label_source are
  always the person's call; an agent-relayed confirmation is flagged.

- Code runs only in the analysis kernel, sandboxed per ADR 0001 (default
  `best`; `--require-isolation` refuses anything below `sandboxed`). No
  in-process exec fallback.
- `record_finding` refuses a claim that `falsify` has not been called for.
  (Ledgerhound checks that the plan was requested, not that every route
  was run; the falsification plan tells the agent to run them.)
- `report` refuses while no finding has been recorded.
- Person decisions go through MCP elicitation when the client supports it,
  else through the agent with `confirm=`, flagged in the report as approved
  by the agent (ledgerhound.shell.approvals).

One session per server process; tool calls are serialised (one kernel, one
state) and run off the event loop, so a long cell never blocks the protocol.
"""

# No `from __future__ import annotations` here: the SDK reads the tool
# functions' annotations (Context, Literal) as live objects.
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional, Union

INSTRUCTIONS = (
    "Ledgerhound is a fraud-analytics specialist. Raw data stays in its "
    "sandboxed kernel; you see summaries. Order: load_dataset -> profile -> "
    "confirm_roles -> set_goal -> run_analysis / check_feature -> falsify(claim) -> "
    "record_finding(claim) -> report. Tools refuse out-of-order or unsafe "
    "steps and say how to fix them; follow the fix rather than working "
    "around it with your own pandas, because only findings recorded here "
    "appear in the certified report. When a tool returns NEEDS USER "
    "DECISION, ask the user exactly that question; never answer it yourself."
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class LedgerhoundSession:
    """One analyst session behind MCP: the tool layer, the relay gate, the
    order rules, and a lock so calls run one at a time."""

    def __init__(self, out_root: str | Path = "run_outputs", *,
                 require_isolation: bool = False, isolation=None, tools=None):
        from ledgerhound.runtime.isolation import IsolationPolicy
        from ledgerhound.shell.approvals import RelayGate
        from ledgerhound.shell.tools import AnalystTools
        self.gate = RelayGate()
        policy = isolation or IsolationPolicy.from_config(
            default_mode="best", require=require_isolation)
        self.tools = tools or AnalystTools(
            out_root=out_root, renderer=_stderr, gate=self.gate,
            use_kernel=True, exec_fallback=False, isolation=policy)
        self.tools.require_confirmed_roles = True
        self.tools.kb_actor = "agent"
        self._falsified: list[tuple[str, set]] = []
        self._lock = None

    # -- dispatch ------------------------------------------------------------
    async def call(self, ctx, name: str, args: dict, confirm: Optional[str] = None,
                   *, may_ask: bool = False):
        """Run one tool-layer tool. Person decisions take one of three routes:

        - handshake-era connection with a back-channel: `ctx.elicit` mid-call;
        - 2026-07-28 connection (no back-channel) whose client declared
          elicitation: return an InputRequiredResult (only when `may_ask`),
          and the client's retry carries the person's answer;
        - otherwise: NEEDS USER DECISION, and the agent retries with confirm=.
        """
        import anyio
        if self._lock is None:
            self._lock = anyio.Lock()
        reply = _person_reply(ctx)       # (answer, action) on an input_required retry
        answered = reply[0] if reply else None
        async with self._lock:
            if reply is not None:
                # What the client reported, verbatim, so a "cancel" from a
                # client that could not show the form is never mistaken for
                # a person's decision.
                self.tools._log("hitl", tier="client_reply", name=name,
                                action=reply[1], answer=answered or "")
            self.gate.pending = None
            route: dict = {}
            self.gate.elicit = (_preset(answered) if answered is not None
                                else _elicitor(ctx, route))
            self.gate.agent_answer = confirm
            try:
                result = await anyio.to_thread.run_sync(self.tools._run_tool, name, args)
            finally:
                self.gate.elicit = None
                self.gate.agent_answer = None
            for action in route.get("replies", []):
                self.tools._log("hitl", tier="client_reply", name=name, action=action)
            # input_required only where it is the protocol's way to ask: a
            # 2026-07-28 connection (no back-channel). On a handshake-era
            # connection it is not a valid result.
            if (may_ask and reply is None and self.gate.pending is not None
                    and confirm is None and route.get("no_back_channel")):
                question, options = self.gate.pending
                self.tools._log("hitl", tier="pending", name=name, approved=False,
                                answer="", approved_by="asking person via client",
                                question=question[:500])
                return _input_required(question, options)
            return result

    async def run_local(self, fn, *args):
        """A server-side step that is not a tool-layer tool (kb_search)."""
        import anyio
        if self._lock is None:
            self._lock = anyio.Lock()
        async with self._lock:
            return await anyio.to_thread.run_sync(fn, *args)

    # -- order rules ---------------------------------------------------------
    def note_falsified(self, claim: str) -> None:
        from ledgerhound.shell.guardrails.findings import claim_tokens
        self._falsified.append((_norm(claim), claim_tokens(claim)))

    def was_falsified(self, claim: str) -> bool:
        """The claim being recorded must itself be (mostly) the one that was
        falsified: claims_match(falsified, recorded) requires at least half of
        the RECORDED claim's distinctive words to be covered, so a broader
        claim that bolts something new onto a falsified one is refused."""
        from ledgerhound.shell.guardrails.findings import claim_tokens, claims_match
        n, toks = _norm(claim), claim_tokens(claim)
        return any(n == fn or claims_match(ft, toks) for fn, ft in self._falsified)

    def recorded_findings(self) -> list[dict]:
        path = getattr(self.tools, "_findings_path", None)
        if not path or not Path(path).exists():
            return []
        out = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
        return out

    def kb_search(self, query: str, max_hits: int = 8) -> str:
        """Keyword search over recorded findings: this session's, this
        dataset's earlier sessions (dataset card), and the team findings
        file. Phase 3 adds Knowledge Pack documents and embeddings."""
        from ledgerhound.shell.guardrails.findings import claim_tokens
        q = claim_tokens(query)
        if not q:
            return "kb_search needs a query with at least one distinctive word."
        pool: list[tuple[str, str]] = []
        for f in self.recorded_findings():
            pool.append(("this session", str(f.get("claim", ""))))
        for ref, claim in self.tools._book_history_claims():
            pool.append(("earlier session on this dataset", claim))
        for name, stmt in self.tools._team_kb_claims():
            pool.append((f"team: {name}", stmt))
        scored = []
        for source, text in pool:
            hit = len(q & claim_tokens(text))
            if hit:
                scored.append((hit, source, text))
        self.tools._log("kb_search", query=query[:300], hits=len(scored))
        if not scored:
            return (f"no recorded finding matches {query!r} (searched {len(pool)} "
                    "finding(s): this session, earlier sessions on this dataset, "
                    "team findings). Knowledge Pack documents are not searchable "
                    "until Phase 3.")
        scored.sort(key=lambda s: -s[0])
        lines = [f"{len(scored)} match(es) for {query!r}; findings are from data, "
                 "team entries are unverified on this dataset:"]
        for hit, source, text in scored[:max_hits]:
            lines.append(f"- [{source}] {text[:400]}")
        return "\n".join(lines)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_DECISION_KEY = "ledgerhound_decision"


def _client_elicits(ctx) -> bool:
    caps = getattr(ctx, "client_capabilities", None) if ctx is not None else None
    return caps is not None and getattr(caps, "elicitation", None) is not None


def _decision_model(options: list):
    from pydantic import Field, create_model
    choice_t = Literal.__getitem__(tuple(str(o) for o in options))
    return create_model("LedgerhoundDecision",
                        choice=(choice_t, Field(description="Your decision")))


def _input_required(question: str, options: list):
    """2026-07-28: ask the person through the client, outside this call."""
    from mcp import types as mt
    from mcp.server.elicitation import render_elicitation_schema
    req = mt.ElicitRequest(params=mt.ElicitRequestFormParams(
        message=question,
        requested_schema=render_elicitation_schema(_decision_model(options))))
    return mt.InputRequiredResult(input_requests={_DECISION_KEY: req},
                                  request_state=_DECISION_KEY)


def _answer_for(action: str, choice) -> Optional[str]:
    """Map an elicitation result to RelayGate's elicit() contract.

    accept  -> the choice (a person chose it);
    decline -> ""   (a person explicitly said no: the safe last option);
    cancel  -> None (nobody chose -- dismissed, or a client that cannot show
               forms, e.g. `claude -p`). Not a person's decision: the
               question goes back to the agent relay instead.
    """
    if action == "accept":
        return str(choice or "")
    if action == "decline":
        return ""
    return None


def _person_reply(ctx) -> Optional[tuple]:
    """(answer, action) on an input_required retry -- answer as _answer_for
    -- or None when this is not such a retry."""
    if ctx is None or getattr(ctx, "request_state", None) != _DECISION_KEY:
        return None      # only an answer to a question THIS server asked
    responses = getattr(ctx, "input_responses", None)
    res = (responses or {}).get(_DECISION_KEY)
    if res is None:
        return None
    action = str(getattr(res, "action", "") or "")
    choice = (getattr(res, "content", None) or {}).get("choice")
    return _answer_for(action, choice), action


def _person_answer(ctx) -> Optional[str]:
    reply = _person_reply(ctx)
    return reply[0] if reply else None


def _preset(answer: str):
    """An elicit hook that returns the answer the person already gave."""
    def elicit(question: str, options: list) -> Optional[str]:
        return answer
    return elicit


def _elicitor(ctx, route: Optional[dict] = None):
    """A sync elicit(question, options) for RelayGate, bound to this request;
    None when the client did not declare elicitation."""
    if not _client_elicits(ctx):
        return None

    def elicit(question: str, options: list) -> Optional[str]:
        import anyio
        try:
            res = anyio.from_thread.run(ctx.elicit, question, _decision_model(options))
        except Exception as e:  # noqa: BLE001 -- no back-channel, client error
            if type(e).__name__ == "NoBackChannelError":
                if route is not None:
                    route["no_back_channel"] = True
            else:
                _stderr(f"[ledgerhound] elicitation failed ({type(e).__name__}: {e}); "
                        "relaying the question through the agent")
            return None
        action = getattr(res, "action", "")
        data = getattr(res, "data", None)
        if route is not None:
            route.setdefault("replies", []).append(str(action))
        return _answer_for(action, getattr(data, "choice", None))
    return elicit


# --------------------------------------------------------------------------- #

def build_server(session: LedgerhoundSession):
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import InputRequiredResult

    from ledgerhound import __version__
    server = MCPServer(name="ledgerhound", version=__version__, instructions=INSTRUCTIONS)
    s = session

    @server.tool()
    async def load_dataset(path: str, ctx: Context) -> str:
        """Load a CSV/Parquet file or folder into the sandboxed analysis kernel
        as `df`. Returns shape, columns, open data questions and the kernel's
        isolation level. Call this first."""
        return await s.call(ctx, "load_data", {"path": path})

    @server.tool()
    async def profile(ctx: Context) -> str:
        """Profile the loaded dataset: types, nulls, cardinality,
        distributions, quality. Refuses if nothing is loaded."""
        return await s.call(ctx, "profile_data", {})

    @server.tool()
    async def run_analysis(code: str, ctx: Context) -> str:
        """Run Python in the sandboxed kernel: pandas (cuDF on GPU), `df`
        preloaded, helpers evaluate_feature, evaluate_features, check_leakage.
        No network inside. Refuses model training before
        request_modeling_authorisation, output at the wrong grain, and tests
        whose assumptions fail. print() what you need to see."""
        return await s.call(ctx, "run_python", {"code": code})

    @server.tool()
    async def check_feature(features: list[str], label: str, ctx: Context) -> str:
        """IV / KS / AUC strength and leakage checks for one or more feature
        columns against the label column, computed in the kernel."""
        if not features:
            return "check_feature needs at least one feature column."
        refused = s.tools.role_feature_refusal(features, label)
        if refused:
            return refused
        no_leak = s.tools._roles_refusal("the leakage check")
        leak = ("'REFUSED: column roles are not confirmed; call confirm_roles'"
                if no_leak else f"check_leakage(_f, label_column={label!r})")
        code = ("import json as _j\n"
                f"_f = {features!r}\n"
                f"_r = {{'strength': evaluate_features(df, _f, {label!r}),\n"
                f"      'leakage': {leak}}}\n"
                "print(_j.dumps(_r, default=str, indent=1)[:8000])")
        return await s.call(ctx, "run_python", {"code": code})

    @server.tool()
    async def confirm_roles(ctx: Context, roles: Optional[dict[str, str]] = None,
                            absent: Optional[list[str]] = None,
                            confirm: Optional[str] = None) -> Union[str, InputRequiredResult]:
        """Confirm what each column IS (payer, payee, event_time, amount,
        label, label_time, rule_output, ...); guardrails use roles, not
        names. Fix a role with roles={column: role}; declare missing label
        roles with absent=["label_time", ...]. If the result says NEEDS USER
        DECISION, ask the user and call again with confirm=<their choice>."""
        args = {k: v for k, v in (("roles", roles), ("absent", absent)) if v}
        return await s.call(ctx, "confirm_roles", args, confirm=confirm, may_ask=True)

    @server.tool()
    async def falsify(claim: str, ctx: Context) -> str:
        """The falsification plan for a claim: the routes that could break it
        (subgroup reversal, period holdout, prevalence shift, grain
        restatement, leakage origin, effect size). Run them with
        run_analysis. Required before record_finding for that claim."""
        out = await s.call(ctx, "propose_falsification", {"claim": claim})
        if not out.startswith(("error", "REFUSED")):
            s.note_falsified(claim)
        return out

    @server.tool()
    async def record_finding(claim: str, ctx: Context,
                             sample_frame: Literal["within_stratum", "cross_stratum",
                                                   "population_estimate", "unknown"] = "unknown",
                             evidence: str = "", predicate: str = "",
                             pvalue: Optional[float] = None, delta: str = "") -> str:
        """Record an evidence-backed finding for the certified report. Refuses
        a claim not yet passed to falsify, and a restatement of a known
        finding unless `delta` says what it adds. Give `predicate` (the exact
        condition the numbers were computed under) and `pvalue` if a test ran."""
        if not s.was_falsified(claim):
            return ("REFUSED: call falsify with this claim first, run the routes "
                    "it returns with run_analysis, then record it. Nothing was "
                    "recorded.")
        args = {"claim": claim, "sample_frame": sample_frame}
        for k, v in (("evidence", evidence), ("predicate", predicate), ("delta", delta)):
            if v:
                args[k] = v
        if pvalue is not None:
            args["pvalue"] = pvalue
        return await s.call(ctx, "record_finding", args)

    @server.tool()
    async def kb_search(query: str, ctx: Context) -> str:
        """Keyword search over recorded findings: this session, earlier
        sessions on this dataset, and team findings. Knowledge Pack documents
        are not searchable until Phase 3."""
        return await s.run_local(s.kb_search, query)

    @server.tool()
    async def report(title: str, markdown: str, ctx: Context) -> str:
        """Write the report: your markdown plus every finding recorded since
        the last report, and a provenance section Ledgerhound writes itself
        (isolation level, who approved what). Refuses with no recorded
        findings."""
        if not s.recorded_findings():
            return ("REFUSED: no finding has been recorded, so there is nothing "
                    "to certify. Record findings with record_finding first.")
        return await s.call(ctx, "save_report", {"title": title, "markdown": markdown})

    @server.tool()
    async def set_goal(objective: str, ctx: Context, unit_of_analysis: str = "",
                       entity_key: str = "", label_column: str = "", notes: str = "") -> str:
        """Declare the session goal: objective, unit of analysis (e.g.
        account or transaction), entity key and label column. Grain checks
        in run_analysis enforce it."""
        args = {"objective": objective}
        for k, v in (("unit_of_analysis", unit_of_analysis), ("entity_key", entity_key),
                     ("label_column", label_column), ("notes", notes)):
            if v:
                args[k] = v
        return await s.call(ctx, "set_goal", args)

    @server.tool()
    async def set_methodology(ctx: Context, template: str = "", mode: str = "") -> str:
        """Adopt a phase methodology the USER chose (call with no template to
        list them). Ask the user; do not pick one yourself."""
        args = {k: v for k, v in (("template", template), ("mode", mode)) if v}
        return await s.call(ctx, "set_methodology", args)

    @server.tool()
    async def record_data_answer(key: str, answer: str, ctx: Context, question: str = "") -> str:
        """Record the USER's answer to an open data question that load_dataset
        listed. Only answers the user gave; never your own guess."""
        args = {"key": key, "answer": answer}
        if question:
            args["question"] = question
        return await s.call(ctx, "record_data_answer", args)

    @server.tool()
    async def request_modeling_authorisation(
            summary: str, ctx: Context,
            confirm: Optional[str] = None) -> Union[str, InputRequiredResult]:
        """Ask the user to authorise moving from analysis to model training.
        If the result says NEEDS USER DECISION, ask the user that exact
        question and call again with confirm=<their choice>."""
        return await s.call(ctx, "request_modeling_authorisation",
                            {"summary": summary}, confirm=confirm, may_ask=True)

    @server.resource("ledgerhound://findings", mime_type="application/json")
    def findings() -> str:
        """Every finding recorded in this session."""
        return json.dumps(s.recorded_findings(), default=str, indent=1)

    @server.tool()
    async def kb_add(ctx: Context, entry: Optional[dict] = None, document: str = "",
                     scope: Literal["project", "team"] = "project", title: str = "",
                     source: str = "", confirm: Optional[str] = None) -> Union[str, InputRequiredResult]:
        """Add one Knowledge Pack entry (type typology | rule | dictionary |
        known_finding | label_policy) or one document (.md/.txt/.pdf path).
        Refuses entries that fail validation. scope=team asks the user; if the
        result says NEEDS USER DECISION, ask and retry with confirm=."""
        args = {k: v for k, v in (("entry", entry), ("document", document), ("scope", scope),
                                  ("title", title), ("source", source)) if v}
        return await s.call(ctx, "kb_add", args, confirm=confirm, may_ask=True)

    @server.resource("ledgerhound://roles", mime_type="application/json")
    def column_roles() -> str:
        """Every column's role, confidence, and who settled it."""
        rm = s.tools.roles
        return json.dumps(rm.to_dict() if rm else {"roles": "no data loaded"}, indent=1)

    @server.resource("ledgerhound://isolation", mime_type="application/json")
    def isolation() -> str:
        """What the current analysis kernel actually enforces (ADR 0001)."""
        rep = s.tools.isolation_report
        return json.dumps(rep.as_dict() if rep else {"level": "no kernel running"})

    return server


async def serve_stdio(server) -> None:
    from mcp.server.stdio import stdio_server

    from ledgerhound.mcp._stdio import wire_stdout
    low = server._lowlevel_server  # the SDK's own run_stdio_async does the same
    async with stdio_server(stdout=wire_stdout()) as (read, write):
        await low.run(read, write, low.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    """`ledgerhound mcp` / `python -m ledgerhound.mcp`: serve over stdio."""
    from ledgerhound.mcp._stdio import protect_stdout
    protect_stdout()
    import argparse

    import anyio
    p = argparse.ArgumentParser(prog="ledgerhound mcp")
    p.add_argument("--out-root", default="run_outputs",
                   help="where run folders go (default: ./run_outputs)")
    p.add_argument("--require-isolation", action="store_true",
                   help="refuse to run code unless the kernel is fully sandboxed")
    a = p.parse_args(argv)
    session = LedgerhoundSession(a.out_root, require_isolation=a.require_isolation)
    try:
        anyio.run(serve_stdio, build_server(session))
    finally:
        session.tools.close()
    return 0
