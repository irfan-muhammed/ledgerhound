"""
sentinel.agents.eda_agent
=========================

ReAct-based EDA agent with:

* Schema KB tool dispatch (P1.A)  -- dict_lookup / dict_search / ... are
  first-class tools in the loop; the agent must consult the KB before
  writing analysis code for any column.
* Sampling-frame-aware system prompt (P1.I)  -- prompt adapts to known
  priors and strata from DatasetMemory.
* Streaming display -- Think / Act / Observe printed live.
* Visualization -- generate_chart action.
* Human-in-the-loop with auto-timeout (20 s default).
* Incremental report saving (after every question).
* Append-only audit log.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from sentinel.llm.client import LLMResponse
from sentinel.knowledge.loader import KnowledgeBase
from sentinel.profiler.digest import Digest
from sentinel.context.router import ContextRouter
from sentinel.agents.tools.code_executor import CodeExecutor
from sentinel.agents.tools.visualizer import Visualizer
from sentinel.agents.prompts.system import build_system_prompt, SYSTEM_PROMPT

if TYPE_CHECKING:  # avoid circular import at runtime
    from sentinel.dictionary.store import SchemaKB
    from sentinel.findings.log import FindingsLog
    from sentinel.memory.models import DatasetMemory
    from sentinel.agents.tools.phase2_tools import Phase2Tools
    from sentinel.workspace.layout import WorkspaceLayout
    from sentinel.memory.knowledge import DatasetKnowledge
    from sentinel.memory.session_log import SessionLog
    from sentinel.compute.tier import TierClassifier
    from sentinel.agents.spec import AgentSpec

log = logging.getLogger(__name__)
console = Console()

# -- Default Schema KB location (relative to CWD) ---------------------------
_DEFAULT_KB_PATH = ".sentinel/upi_kb.duckdb"

# -- Default findings log directory (per-dataset file) ----------------------
_DEFAULT_FINDINGS_DIR = ".sentinel/findings"

# -- Default notebook journal directory -------------------------------------
_DEFAULT_JOURNAL_DIR = ".sentinel/journals"


# -- Result types -----------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    thinking_trace: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    total_tokens: int = 0
    elapsed_ms: float = 0.0
    charts: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)  # P1.B
    finding_warnings: list[str] = field(default_factory=list)  # P1.E
    journal_path: str | None = None  # P1.C -- path to the .ipynb journal

    def display(self):
        console.print(Panel(Markdown(self.answer),
                            title="Sentinel EDA", border_style="blue"))
        if self.charts:
            console.print(f"[dim]Charts: {', '.join(self.charts)}[/dim]")
        if self.finding_ids:
            console.print(
                f"[dim]Findings recorded: {len(self.finding_ids)} -> "
                f"{', '.join(self.finding_ids)}[/dim]"
            )
        if self.journal_path:
            console.print(f"[dim]Journal: {self.journal_path}[/dim]")
        if self.finding_warnings:
            for w in self.finding_warnings:
                console.print(f"[yellow]! {w}[/yellow]")


@dataclass
class PlaybookResult:
    findings: list[dict] = field(default_factory=list)
    summary: str = ""
    phases_completed: int = 0
    phases_total: int = 0
    analyses_run: int = 0
    analyses_succeeded: int = 0
    analyses_failed: int = 0
    elapsed_ms: float = 0.0
    charts: list[str] = field(default_factory=list)

    def save(self, path: str, digest: dict | None = None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._to_markdown(), encoding="utf-8")
        console.print(f"[green]Report saved: {path}[/green]")

        pdf_path = p.with_suffix(".pdf")
        try:
            from sentinel.output.pdf_report import save_pdf_report
            save_pdf_report(str(pdf_path), findings=self.findings,
                            summary=self.summary, digest=digest,
                            charts=self.charts, elapsed_ms=self.elapsed_ms,
                            analyses_run=self.analyses_run,
                            analyses_succeeded=self.analyses_succeeded)
            console.print(f"[green]PDF report: {pdf_path}[/green]")
        except Exception as e:
            console.print(f"[dim]PDF skipped: {e}[/dim]")

    def _to_markdown(self) -> str:
        lines = ["# Sentinel EDA Report\n"]
        lines.append(
            f"*{self.analyses_succeeded}/{self.analyses_run} analyses "
            f"| {self.elapsed_ms/1000:.1f}s | {len(self.charts)} charts*\n"
        )
        cur_phase = ""
        for f in self.findings:
            if f.get("phase", "") != cur_phase:
                cur_phase = f["phase"]
                lines.append(f"\n## {cur_phase}\n")
            lines.append(f"### {f.get('question', '')}\n")
            if f.get("status") == "success":
                lines.append(f"{f.get('finding', '')}\n")
                for chart in f.get("charts", []):
                    lines.append(f"\n![Chart]({chart})\n")
            else:
                lines.append(f"FAILED: {f.get('error', '')}\n")
        if self.summary:
            lines.append(f"\n## Executive Summary\n\n{self.summary}\n")
        return "\n".join(lines)

    def display(self):
        console.print(Panel(Markdown(self._to_markdown()),
                            title="Sentinel EDA Report", border_style="blue"))


# -- Audit logger -----------------------------------------------------------

class AuditLogger:
    """Append-only JSON-lines log of every agent action."""

    def __init__(self, path: str = "outputs/sentinel_audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data):
        entry = {"ts": time.time(), "event": event, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# -- Timed input ------------------------------------------------------------

def _timed_input(prompt: str, timeout: int = 20, default: str = "A") -> str:
    """Read user input with a timeout.  Returns *default* if no reply arrives."""
    result = [default]

    def _read():
        try:
            result[0] = console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        console.print(f"\n  [dim]No response in {timeout}s -- using default: {default}[/dim]")
        return default
    return result[0] or default


# -- Agent ------------------------------------------------------------------

class EDAAgent:
    """ReAct-based EDA agent.

    Parameters
    ----------
    llm:
        An ``LLMClient`` instance.
    backend:
        Data backend (provides ``.df`` DataFrame).
    knowledge:
        Legacy ``KnowledgeBase`` for playbook lookup and tag retrieval.
    digest:
        ``Digest`` from the profiler; set later via ``set_digest``.
    schema_kb:
        **P1.A** -- ``SchemaKB`` instance.  When provided, dictionary tools
        (``dict_lookup``, ``dict_search``, etc.) are available in the ReAct
        loop and the KB-First Protocol section is injected into the system
        prompt.  If *None*, the agent tries to lazy-load from ``kb_path``
        at first use; if that file also doesn't exist, KB tools are silently
        disabled.
    dataset_memory:
        **P1.A** -- ``DatasetMemory`` for the current dataset.  Provides
        ``sampling_design.priors_encoded`` and strata for the sample-frame
        section of the system prompt.
    kb_path:
        Override default KB path (default: ``.sentinel/upi_kb.duckdb``).
    output_dir:
        Root directory for charts, reports, and the audit log.
    """

    MAX_ITER = 12
    HITL_TIMEOUT = 20  # seconds before auto-answering

    # -- Text-based ReAct action patterns ----------------------------------

    # Schema KB tools  (Action: dict_<name>\n{...json...})
    _DICT_TOOLS = frozenset({
        "dict_lookup", "dict_search", "dict_quality_flags",
        "dict_decode_code", "dict_aliases", "dict_related", "dict_stats",
    })
    _DICT_TOOL_RE = re.compile(
        r"Action:\s*(dict_\w+)\s*\n(\{.*?\})",
        re.S,
    )
    # dict_stats has no args -- match separately
    _DICT_STATS_RE = re.compile(r"Action:\s*dict_stats\s*(?:\n\{\s*\})?")

    # Findings tools  (Action: record_finding\n{...possibly nested JSON...})
    # The JSON for record_finding is multi-line and has nested arrays, so
    # we locate the start with a regex and extract the balanced brace range.
    _FINDINGS_TOOLS = frozenset({"record_finding", "list_findings"})
    _FINDINGS_TOOL_START_RE = re.compile(
        r"Action:\s*(record_finding|list_findings)\s*\n",
    )

    # Phase 2 tools  (Action: <p2_tool>\n{...json...})
    # Same balanced-JSON extraction as findings tools.
    _PHASE2_TOOLS = frozenset({
        "psi_certify",
        "mine_hard_negatives",
        "cluster_typologies",
        "validate_consistency",
        "check_leakage",
        "tier_check",
        "deposit_insight",
        "measure_signal_decay",
        "track_typology_evolution",
        "detect_mule_motifs",
        "compute_account_centrality",
        "find_mule_rings",
        "trace_money_flow",
        "fingerprint_accounts",
    })
    # V3-032 Tier 2: check_leakage (V3-018) was documented in the system
    # prompt (system.py's _PHASE2_TOOLS_SECTION shows the model an exact
    # "Action: check_leakage" example) and dispatchable via
    # Phase2Tools.as_tool_map(), but was never added to THIS regex -- so a
    # model doing exactly what the prompt told it to do produced an action
    # the engine's own parser could not recognise. Found while wiring
    # feature_engineering.yaml's required_tools=[check_leakage]: the first
    # live test showed the call falling through unrecognised instead of
    # dispatching.
    _PHASE2_TOOL_START_RE = re.compile(
        r"Action:\s*(psi_certify|mine_hard_negatives|cluster_typologies"
        r"|validate_consistency|check_leakage|tier_check|deposit_insight"
        r"|measure_signal_decay"
        r"|track_typology_evolution|detect_mule_motifs"
        r"|compute_account_centrality|find_mule_rings"
        r"|trace_money_flow|fingerprint_accounts)\s*\n",
    )

    # Existing tools
    _ACTION_RE = re.compile(
        r"Action:\s*execute_python\s*```(?:python)?\s*\n(.*?)```", re.S
    )
    _CHART_RE = re.compile(
        r"Action:\s*generate_chart\s*```(?:python)?\s*\n(.*?)```", re.S
    )
    _ASK_RE = re.compile(
        r"Action:\s*ask_user\s*\n(.*?)(?=\n\n|\nThought:|\nAction:|\nFinal Answer:|$)",
        re.S,
    )
    @staticmethod
    def _normalize_protocol(text: str) -> str:
        """Tolerate markdown-decorated protocol keywords.

        Models routinely emit ``**Action:** tool``, ``**Final Answer**:`` or
        ``### Action:`` -- a strict plain-text match silently drops the tool
        call and the loop misroutes (AUD-173). Rewrite any decorated form of
        the three protocol keywords back to the canonical ``Keyword:``.
        """
        text = re.sub(
            r"(?:^|\n)[ \t]*(?:#{1,4}[ \t]*)?\*{0,2}"
            r"(Action|Final Answer|Thought)\*{0,2}[ \t]*:\*{0,2}",
            lambda m: "\n" + m.group(1) + ":",
            text)
        # A stray closing "**" glued directly onto the tool name (no
        # whitespace) breaks the downstream "Action: <tool>\s*```" regexes.
        # Valid protocol never has a literal "*" immediately after the tool
        # name, so this is unambiguously decoration noise, not content.
        text = re.sub(r"(Action:\s*\w+)\*+", r"\1", text)
        return text

    _FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S)
    _THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", re.S)

    # -- Constructor --------------------------------------------------------

    def __init__(
        self,
        llm,
        backend=None,
        knowledge=None,
        digest=None,
        schema_kb: "SchemaKB | None" = None,
        dataset_memory: "DatasetMemory | None" = None,
        findings_log: "FindingsLog | None" = None,
        kb_path: str | None = None,
        findings_path: str | None = None,
        journal_dir: str | None = None,
        write_journal: bool = True,
        output_dir: str = "outputs",
        # -- Phase 2 additions ----------------------------------------------
        phase2_tools: "Phase2Tools | None" = None,
        workspace: "WorkspaceLayout | None" = None,
        dataset_knowledge: "DatasetKnowledge | None" = None,
        session_log: "SessionLog | None" = None,
        tier_classifier: "TierClassifier | None" = None,
        role_prompt: str | None = None,
        spec: "AgentSpec | None" = None,
    ):
        """
        Parameters
        ----------
        journal_dir:
            Directory for ``.ipynb`` journals.  Default: ``<output_dir>/journals``.
            Pass an explicit path to override (e.g. ``".sentinel/journals"``).
        write_journal:
            If True (default), every ``ask()`` call writes a journal at
            the end.  Set to False to suppress (e.g. for unit tests where
            you don't want filesystem side effects).
        """
        self.llm = llm
        self.backend = backend
        self.knowledge = knowledge or KnowledgeBase()
        self.digest = digest
        # agent_def/v1 spec consumption (rank 3): the spec's role_prompt
        # replaces the default identity; the engine still owns every protocol
        # section. Explicit role_prompt wins over spec.role_prompt.
        self.spec = spec
        self._role_prompt = role_prompt or (spec.role_prompt if spec else None)
        self._schema_kb = schema_kb
        self._dataset_memory = dataset_memory
        self._findings_log = findings_log
        self._kb_path = kb_path or _DEFAULT_KB_PATH
        self._findings_path = findings_path  # explicit override
        # Resolve the journal directory: explicit > <output_dir>/journals
        self._journal_dir = journal_dir or os.path.join(output_dir, "journals")
        self._write_journal = bool(write_journal)
        self._dict_tools_cache: dict | None = None         # lazily populated
        self._findings_tools_cache: dict | None = None     # lazily populated
        self._executor: CodeExecutor | None = None
        self._router: ContextRouter | None = None
        self._visualizer = Visualizer(os.path.join(output_dir, "charts"))
        self._audit = AuditLogger(os.path.join(output_dir, "sentinel_audit.jsonl"))
        self._output_dir = output_dir
        # -- Phase 2 --------------------------------------------------------
        self._phase2_tools = phase2_tools        # Phase2Tools instance
        self._workspace = workspace
        self._dataset_knowledge = dataset_knowledge
        self._session_log = session_log
        self._tier_classifier = tier_classifier
        self._phase2_map: dict | None = None     # lazily resolved tool map

    def set_digest(self, digest: Digest):
        self.digest = digest
        self._router = None

    def set_schema_kb(self, kb: "SchemaKB"):
        """Attach (or replace) the Schema KB after construction."""
        self._schema_kb = kb
        self._dict_tools_cache = None  # invalidate

    def set_dataset_memory(self, mem: "DatasetMemory"):
        """Attach (or replace) the Dataset Memory after construction."""
        self._dataset_memory = mem
        self._findings_tools_cache = None  # invalidate (memory affects warnings)

    def set_findings_log(self, log: "FindingsLog"):
        """Attach (or replace) the FindingsLog after construction."""
        self._findings_log = log
        self._findings_tools_cache = None  # invalidate

    def set_phase2_tools(self, p2: "Phase2Tools"):
        """Attach or replace the Phase2Tools instance."""
        self._phase2_tools = p2
        self._phase2_map = None  # invalidate cache

    def set_dataset_knowledge(self, dk: "DatasetKnowledge"):
        """Attach or replace the DatasetKnowledge for context injection."""
        self._dataset_knowledge = dk

    def set_session_log(self, sl: "SessionLog"):
        """Attach or replace the SessionLog."""
        self._session_log = sl

    # -- Single question ----------------------------------------------------

    def ask(self, question: str, interactive: bool = True) -> AgentResult:
        t0 = time.perf_counter()
        # Each question starts from the pristine dataframe; derived columns
        # persist only within the steps of answering THIS question.
        if self._executor is not None:
            self._executor.reset_working_df()
        ctx = self._router_().build_context(question)

        # Resolve KB and memory context for this session
        dict_tools = self._dict_tools_()
        has_kb = dict_tools is not None
        priors: list[str] = []
        strata: dict | None = None
        dataset_name: str | None = None

        if self._dataset_memory:
            dataset_name = getattr(self._dataset_memory, "dataset_name", None)
            sd = getattr(self._dataset_memory, "sampling_design", None)
            if sd:
                priors = list(sd.priors_encoded or [])
                if sd.strata:
                    strata = {
                        name: {
                            "method": s.method.value if hasattr(s.method, "value") else str(s.method),
                            "n_sample": s.n_sample,
                            "p_inclusion": s.p_inclusion,
                            "filters": s.filters,
                        }
                        for name, s in sd.strata.items()
                    }

        if not dataset_name and self.digest:
            dataset_name = getattr(self.digest, "dataset_name", None)

        # Resolve findings log + tools (P1.B)
        findings_tools = self._findings_tools_(dataset_name=dataset_name)
        has_findings = findings_tools is not None

        # Resolve Phase 2 tools (P2)
        phase2_map = self._phase2_tools_map_()
        has_phase2 = phase2_map is not None

        # Build institutional memory context from DatasetKnowledge (P2)
        knowledge_context: str | None = None
        if self._dataset_knowledge is not None:
            try:
                knowledge_context = self._dataset_knowledge.for_session_context(max_entries=30)
            except Exception:
                pass

        # Start session log entry (P2)
        if self._session_log is not None and dataset_name:
            try:
                self._session_log.touch(dataset_name=dataset_name)
            except Exception:
                pass

        # Collect loaded dataset keys for the Phase 2 prompt section
        phase2_datasets: list[str] = []
        if has_phase2 and self._phase2_tools is not None:
            try:
                phase2_datasets = self._phase2_tools.dataset_keys()
            except Exception:
                pass

        system_prompt = build_system_prompt(
            has_kb=has_kb,
            has_findings=has_findings,
            has_phase2=has_phase2,
            phase2_datasets=phase2_datasets,
            priors_encoded=priors or None,
            dataset_name=dataset_name,
            sampling_strata=strata,
            role_override=self._role_prompt,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": self._user_prompt(
                question, ctx, knowledge_context=knowledge_context,
            )},
        ]
        trace: list[dict] = []
        tool_calls = 0
        tools_used: set[str] = set()  # which TOOL NAMES actually ran (AUD-175)
        tokens = 0
        charts: list[str] = []
        finding_ids: list[str] = []
        finding_warnings: list[str] = []

        self._audit.log("ask_start", question=question, has_kb=has_kb,
                        has_findings=has_findings, has_phase2=has_phase2,
                        priors=priors, dataset=dataset_name)

        _spec_defaults = (self.spec.defaults or {}) if self.spec else {}
        _required_tools = set(_spec_defaults.get("required_tools", []) or [])
        _remind_after = int(_spec_defaults.get("grounding_reminder_after", 0))
        # AUD-179: spec.defaults.max_steps was declared in every builtin
        # agent's YAML but never actually consumed -- ask() always used the
        # hardcoded class constant MAX_ITER regardless. A spec asking for
        # more room (e.g. a thorough grounded profiler run) silently got
        # the same 12-step budget as everything else.
        max_iter = int(_spec_defaults.get("max_steps", self.MAX_ITER))

        for iteration in range(max_iter):
            # Proactive grounding reminder (AUD-178): the Final-Answer-time
            # grounding floor only fires when the model ATTEMPTS a Final
            # Answer -- a model that loops on other tools (e.g.
            # record_finding/list_findings ping-pong) without ever
            # attempting one burns its whole budget before that check gets
            # a chance to run. Nudge once, mid-loop, before that happens.
            missing_now = _required_tools - tools_used
            if (_remind_after and iteration == _remind_after and missing_now
                    and not any(t.get("type") in
                               ("grounding_nudge", "grounding_reminder")
                               for t in trace)):
                trace.append({"type": "grounding_reminder",
                              "at_iteration": iteration,
                              "missing_required": sorted(missing_now)})
                self._audit.log("grounding_reminder",
                                missing_required=sorted(missing_now))
                reminder = (
                    f"REMINDER: {iteration} turns have passed and you still "
                    f"have not called {sorted(missing_now)}. Stop recording "
                    "findings from memory -- call it NOW with real code and "
                    "use its actual output, before you run out of turns.")
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + reminder
                else:
                    messages.append({"role": "user", "content": reminder})

            resp = self.llm.generate(messages)
            tokens += resp.total_tokens
            body = self._normalize_protocol(resp.content)

            # -- Thinking ---------------------------------------------------
            if resp.has_thinking:
                trace.append({"type": "thinking", "text": resp.thinking})
                if interactive:
                    preview = resp.thinking[:200]
                    suffix = "..." if len(resp.thinking) > 200 else ""
                    console.print(
                        f"\n  [bold yellow] Thinking:[/bold yellow] "
                        f"[dim]{preview}{suffix}[/dim]"
                    )

            # -- Thought line -----------------------------------------------
            thought_m = self._THOUGHT_RE.search(body)
            if thought_m and interactive:
                console.print(
                    f"  [bold yellow] Reasoning:[/bold yellow] "
                    f"{thought_m.group(1).strip()[:200]}"
                )

            # -- Final answer? ----------------------------------------------
            # ReAct discipline: if the message contains an Action BEFORE the
            # Final Answer, the model has hallucinated ahead (actions + fake
            # observations + conclusion in one turn). Dispatch the first
            # action instead -- the real Observation replaces the invented
            # one and the loop continues.
            m = self._FINAL_RE.search(body)
            if m:
                act = re.search(r"Action:\s*\w+", body)
                if act and act.start() < m.start():
                    trace.append({"type": "premature_final_ignored",
                                  "at_iteration": iteration})
                    m = None
            if m:
                # Grounding floor (agent_def defaults.min_tool_calls /
                # required_tools): a count alone is gameable -- a model can
                # satisfy min_tool_calls by calling record_finding with
                # INVENTED numbers instead of ever computing anything
                # (AUD-175: caught by the judge, not this floor, in the
                # first version). required_tools names SPECIFIC tools (e.g.
                # execute_python) that must have actually run. One
                # corrective nudge per ask -- if the model still refuses, we
                # accept the answer and let downstream verification fail it
                # honestly.
                min_tools = int(_spec_defaults.get("min_tool_calls", 0))
                required = _required_tools
                missing_required = required - tools_used
                ungrounded = tool_calls < min_tools or bool(missing_required)
                if (ungrounded and iteration < max_iter - 1
                        and not any(t.get("type") == "grounding_nudge"
                                    for t in trace)):
                    trace.append({"type": "grounding_nudge",
                                  "at_iteration": iteration,
                                  "missing_required": sorted(missing_required)})
                    self._audit.log("grounding_nudge", tool_calls=tool_calls,
                                    min_tool_calls=min_tools,
                                    missing_required=sorted(missing_required))
                    if interactive:
                        console.print(
                            "  [bold red]Grounding floor:[/bold red] answer "
                            f"rejected -- {tool_calls}/{min_tools} tool calls"
                            + (f", missing required: {sorted(missing_required)}"
                               if missing_required else ""))
                    messages.append({"role": "assistant", "content": body})
                    nudge = (
                        "REJECTED: you answered without grounding. "
                        f"Your spec requires at least {min_tools} tool "
                        "call(s) before a Final Answer."
                    )
                    if missing_required:
                        nudge += (
                            f" Specifically you MUST call {sorted(missing_required)} "
                            "and use its REAL Observation output -- calling a "
                            "different tool (e.g. record_finding with numbers "
                            "you made up) does not satisfy this requirement."
                        )
                    nudge += (
                        " Execute the required computation now, record "
                        "findings from its actual output with "
                        "`Action: record_finding`, and only then answer. "
                        "NEVER invent a number or a finding ID -- cite only "
                        "values and IDs that appeared in a real Observation.")
                    messages.append({"role": "user", "content": nudge})
                    continue
                if interactive:
                    console.print("  [bold green]OK Answer ready[/bold green]")
                self._audit.log(
                    "ask_done", question=question,
                    tool_calls=tool_calls, tokens=tokens,
                    findings_recorded=len(finding_ids),
                )
                return self._finalize_result(
                    answer=m.group(1).strip(),
                    question=question,
                    trace=trace,
                    tool_calls=tool_calls,
                    tokens=tokens,
                    elapsed_ms=self._ms(t0),
                    charts=charts,
                    finding_ids=finding_ids,
                    finding_warnings=finding_warnings,
                    dataset_name=dataset_name,
                )

            # -- Schema KB tool call? (P1.A) -------------------------------
            # Check dict_stats first (no args)
            if has_kb and self._DICT_STATS_RE.search(body):
                tool_calls += 1
                tools_used.add("dict_stats")
                if interactive:
                    console.print("  [bold blue] dict_stats ...[/bold blue]")
                obs = self._dispatch_dict_tool("dict_stats", "{}", dict_tools)
                trace.append({"type": "dict_tool", "tool": "dict_stats", "result": obs})
                self._audit.log("tool_call", tool="dict_stats", result=obs[:200])
                if interactive:
                    console.print(f"  [bold magenta] KB:[/bold magenta] {obs[:120]}...")
                messages.append({"role": "assistant", "content": body})
                messages.append({"role": "user",      "content": f"Observation:\n{obs}"})
                continue

            # dict_lookup / dict_search / etc. with JSON args
            if has_kb:
                dm = self._DICT_TOOL_RE.search(body)
                if dm:
                    tool_name = dm.group(1)
                    args_json = dm.group(2).strip()
                    tool_calls += 1
                    tools_used.add(tool_name)
                    if interactive:
                        console.print(
                            f"  [bold blue] {tool_name}[/bold blue] "
                            f"[dim]{args_json[:80]}[/dim]"
                        )
                    obs = self._dispatch_dict_tool(tool_name, args_json, dict_tools)
                    trace.append({"type": "dict_tool", "tool": tool_name,
                                  "args": args_json, "result": obs})
                    self._audit.log("tool_call", tool=tool_name,
                                    args=args_json[:200], result=obs[:200])
                    if interactive:
                        obs_short = obs[:150].replace("\n", " ")
                        suffix = "..." if len(obs) > 150 else ""
                        console.print(
                            f"  [bold magenta] KB:[/bold magenta] {obs_short}{suffix}"
                        )
                    messages.append({"role": "assistant", "content": body})
                    messages.append({"role": "user",      "content": f"Observation:\n{obs}"})
                    continue

            # -- Findings tool call? (P1.B) --------------------------------
            if has_findings:
                fm = self._FINDINGS_TOOL_START_RE.search(body)
                if fm:
                    tool_name = fm.group(1)
                    extracted = self._extract_balanced_json(body, fm.end())
                    if extracted is None:
                        # Bad JSON -- return error to LLM as observation
                        err_obs = json.dumps(
                            {"ok": False,
                             "error": (
                                 f"Could not extract balanced JSON args for "
                                 f"{tool_name}. Make sure your JSON object "
                                 f"starts immediately on the line after "
                                 f"'Action: {tool_name}' and that braces match."
                             )},
                            indent=2,
                        )
                        trace.append({"type": "findings_tool", "tool": tool_name,
                                      "result": err_obs})
                        self._audit.log("tool_call", tool=tool_name,
                                        result=err_obs[:200], error="bad_json")
                        messages.append({"role": "assistant", "content": body})
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{err_obs}"})
                        continue

                    args_json, _ = extracted
                    tool_calls += 1
                    tools_used.add(tool_name)
                    if interactive:
                        preview = args_json.replace("\n", " ")[:80]
                        console.print(
                            f"  [bold green] {tool_name}[/bold green] "
                            f"[dim]{preview}...[/dim]"
                        )
                    obs, recorded_id, warns = self._dispatch_findings_tool(
                        tool_name, args_json, findings_tools,
                    )
                    if recorded_id:
                        finding_ids.append(recorded_id)
                    if warns:
                        finding_warnings.extend(warns)
                    trace.append({"type": "findings_tool", "tool": tool_name,
                                  "args": args_json, "result": obs,
                                  "finding_id": recorded_id})
                    self._audit.log("tool_call", tool=tool_name,
                                    finding_id=recorded_id,
                                    warnings=warns,
                                    result=obs[:200])
                    if interactive:
                        if recorded_id:
                            console.print(
                                f"  [bold magenta]OK Finding[/bold magenta] "
                                f"[cyan]{recorded_id}[/cyan]"
                            )
                        for w in warns:
                            console.print(f"  [yellow]! {w[:160]}...[/yellow]"
                                          if len(w) > 160 else f"  [yellow]! {w}[/yellow]")
                    messages.append({"role": "assistant", "content": body})
                    messages.append({"role": "user",
                                     "content": f"Observation:\n{obs}"})
                    continue

            # -- Phase 2 tool call? (P2) ----------------------------------
            if has_phase2:
                p2m = self._PHASE2_TOOL_START_RE.search(body)
                if p2m:
                    tool_name = p2m.group(1)
                    extracted = self._extract_balanced_json(body, p2m.end())
                    if extracted is None:
                        err_obs = json.dumps(
                            {"ok": False,
                             "error": (
                                 f"Could not extract balanced JSON args for "
                                 f"{tool_name}. Ensure JSON starts immediately "
                                 f"after 'Action: {tool_name}' with matching braces."
                             )},
                            indent=2,
                        )
                        trace.append({"type": "phase2_tool", "tool": tool_name,
                                      "result": err_obs})
                        self._audit.log("tool_call", tool=tool_name,
                                        result=err_obs[:200], error="bad_json")
                        messages.append({"role": "assistant", "content": body})
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{err_obs}"})
                        continue

                    args_json, _ = extracted
                    tool_calls += 1
                    tools_used.add(tool_name)
                    if interactive:
                        preview = args_json.replace("\n", " ")[:80]
                        console.print(
                            f"  [bold magenta] {tool_name}[/bold magenta] "
                            f"[dim]{preview}...[/dim]"
                        )
                    obs = self._dispatch_phase2_tool(tool_name, args_json, phase2_map)
                    trace.append({"type": "phase2_tool", "tool": tool_name,
                                  "args": args_json, "result": obs})
                    self._audit.log("tool_call", tool=tool_name,
                                    args=args_json[:200], result=obs[:200])
                    if interactive:
                        try:
                            result_dict = json.loads(obs)
                            status_icon = "OK" if result_dict.get("ok") else "X"
                            summary = result_dict.get("interpretation",
                                       result_dict.get("message",
                                       result_dict.get("error", "done")))[:150]
                            console.print(
                                f"  [bold {'green' if result_dict.get('ok') else 'red'}]"
                                f"{status_icon} {tool_name}:[/bold {'green' if result_dict.get('ok') else 'red'}]"
                                f" {summary}"
                            )
                        except Exception:
                            console.print(f"  [dim] {obs[:150]}[/dim]")
                    messages.append({"role": "assistant", "content": body})
                    messages.append({"role": "user",
                                     "content": f"Observation:\n{obs}"})
                    continue

            # -- Execute Python? --------------------------------------------
            m = self._ACTION_RE.search(body)
            if m:
                code = m.group(1).strip()
                tool_calls += 1
                tools_used.add("execute_python")
                if interactive:
                    preview = code.split("\n")[0][:80]
                    console.print(f"  [bold cyan] Executing:[/bold cyan] {preview}...")
                obs = self._run_code(code)
                trace.append({"type": "action", "code": code, "result": obs})
                self._audit.log("tool_call", tool="execute_python",
                                code=code[:200], result=obs[:200])
                if interactive:
                    obs_short = obs[:150].replace("\n", " ")
                    suffix = "..." if len(obs) > 150 else ""
                    console.print(
                        f"  [bold magenta] Observed:[/bold magenta] {obs_short}{suffix}"
                    )
                messages.append({"role": "assistant", "content": body})
                messages.append({"role": "user",      "content": f"Observation:\n{obs}"})
                continue

            # -- Generate chart? --------------------------------------------
            m = self._CHART_RE.search(body)
            if m:
                code = m.group(1).strip()
                tool_calls += 1
                tools_used.add("generate_chart")
                if interactive:
                    console.print("  [bold green] Generating chart...[/bold green]")
                obs = self._run_chart(code)
                trace.append({"type": "chart", "code": code, "result": obs})
                self._audit.log("tool_call", tool="generate_chart", result=obs[:200])
                if "Chart saved:" in obs:
                    chart_path = obs.split("Chart saved:")[-1].strip()
                    charts.append(chart_path)
                    if interactive:
                        try:
                            from sentinel.output.banner import print_chart_live
                            print_chart_live(chart_path, chart_type="chart")
                        except Exception:
                            console.print(f"  [dim] {obs[:100]}[/dim]")
                elif interactive:
                    console.print(f"  [dim] {obs[:100]}[/dim]")
                messages.append({"role": "assistant", "content": body})
                messages.append({"role": "user",      "content": f"Observation:\n{obs}"})
                continue

            # -- Ask user? (with timeout) -----------------------------------
            m = self._ASK_RE.search(body)
            if m:
                question_text = m.group(1).strip()
                if interactive:
                    console.print(
                        "\n  [bold red] Agent needs clarification:[/bold red]"
                    )
                    console.print(Panel(question_text, border_style="yellow"))
                    console.print(
                        f"  [dim](Auto-selects 'A' in {self.HITL_TIMEOUT}s"
                        " if no response)[/dim]"
                    )
                    answer = _timed_input(
                        "[bold blue]Your choice:[/bold blue] ",
                        timeout=self.HITL_TIMEOUT,
                        default="A",
                    )
                else:
                    answer = "A"
                trace.append({"type": "hitl", "question": question_text, "answer": answer})
                self._audit.log("hitl", question=question_text, answer=answer)
                messages.append({"role": "assistant", "content": body})
                messages.append({"role": "user",      "content": f"User chose: {answer}"})
                continue

            # No action pattern matched -- return as-is
            return self._finalize_result(
                answer=body.strip(),
                question=question,
                trace=trace,
                tool_calls=tool_calls,
                tokens=tokens,
                elapsed_ms=self._ms(t0),
                charts=charts,
                finding_ids=finding_ids,
                finding_warnings=finding_warnings,
                dataset_name=dataset_name,
            )

        return self._finalize_result(
            answer="Analysis incomplete -- max iterations reached.",
            question=question,
            trace=trace,
            tool_calls=tool_calls,
            tokens=tokens,
            elapsed_ms=self._ms(t0),
            charts=charts,
            finding_ids=finding_ids,
            finding_warnings=finding_warnings,
            dataset_name=dataset_name,
        )

    # -- Playbook runner with incremental saves -----------------------------

    def run_playbook(
        self,
        playbook_name: str | None = None,
        playbook: dict | None = None,
        report_path: str | None = None,
    ) -> PlaybookResult:
        if playbook is None:
            playbook = self.knowledge.get_playbook(
                playbook_name or "Behavioral Features EDA"
            )
            if not playbook:
                raise ValueError(
                    f"Playbook '{playbook_name}' not found. "
                    f"Available: {self.knowledge.list_playbooks()}"
                )

        result = PlaybookResult()
        phases = playbook.get("phases", {})
        result.phases_total = len(phases)
        t0 = time.perf_counter()

        if report_path is None:
            report_path = f"outputs/eda_report_{int(time.time())}.md"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        self._audit.log(
            "playbook_start", name=playbook_name or "?",
            phases=result.phases_total,
        )

        for key in sorted(phases):
            phase = phases[key]
            name = phase.get("name", key)
            try:
                from sentinel.output.banner import print_agent_phase
                print_agent_phase(result.phases_completed + 1, result.phases_total, name)
            except Exception:
                console.print(Rule(
                    f"[bold cyan]Phase {result.phases_completed+1}/"
                    f"{result.phases_total}: {name}[/bold cyan]"
                ))

            for q in phase.get("questions", []):
                result.analyses_run += 1
                console.print(f"\n  [bold]> {q}[/bold]")
                try:
                    ar = self.ask(q, interactive=True)
                    result.findings.append({
                        "phase": name, "question": q,
                        "finding": ar.answer, "status": "success",
                        "charts": ar.charts,
                    })
                    result.charts.extend(ar.charts)
                    result.analyses_succeeded += 1
                    console.print("  [green]OK Done[/green]\n")
                except Exception as e:
                    result.findings.append({
                        "phase": name, "question": q,
                        "error": str(e), "status": "failed",
                    })
                    result.analyses_failed += 1
                    console.print(f"  [red]X {str(e)[:60]}[/red]\n")

                # Incremental save after every question
                result.elapsed_ms = self._ms(t0)
                try:
                    Path(report_path).write_text(
                        result._to_markdown(), encoding="utf-8"
                    )
                    console.print(f"  [dim] Report updated: {report_path}[/dim]")
                except Exception:
                    pass

            result.phases_completed += 1

        console.print(Rule("[bold cyan]Synthesizing findings...[/bold cyan]"))
        result.summary = self._synthesise(result.findings)
        result.elapsed_ms = self._ms(t0)

        console.print(
            f"\n[bold green]EDA Complete.[/bold green] "
            f"{result.analyses_succeeded}/{result.analyses_run} ok, "
            f"{result.analyses_failed} failed  "
            f"({result.elapsed_ms/1000:.1f}s, {len(result.charts)} charts)"
        )
        result.save(report_path,
                    digest=self.digest.to_dict() if self.digest else None)

        try:
            from sentinel.output.banner import print_eda_complete
            print_eda_complete(
                result.analyses_succeeded, result.analyses_run,
                result.analyses_failed, result.elapsed_ms,
                len(result.charts), report_path,
            )
        except Exception:
            pass

        self._audit.log(
            "playbook_done",
            succeeded=result.analyses_succeeded,
            failed=result.analyses_failed,
            elapsed_ms=result.elapsed_ms,
        )
        return result

    # -- Schema KB helpers (P1.A) -------------------------------------------

    def _dict_tools_(self) -> dict | None:
        """Return the bound dictionary tool callables, loading the KB lazily.

        Returns *None* (and logs a debug message) when no KB is available.
        """
        if self._dict_tools_cache is not None:
            return self._dict_tools_cache

        # Use explicitly provided KB
        if self._schema_kb is not None:
            from sentinel.dictionary import make_dictionary_tools
            self._dict_tools_cache = make_dictionary_tools(self._schema_kb)
            return self._dict_tools_cache

        # Try lazy load from disk
        kb_path = Path(self._kb_path)
        if kb_path.exists():
            try:
                from sentinel.dictionary import SchemaKB, make_dictionary_tools
                kb = SchemaKB(str(kb_path))
                self._schema_kb = kb
                self._dict_tools_cache = make_dictionary_tools(kb)
                log.debug("Lazy-loaded Schema KB from %s", kb_path)
                return self._dict_tools_cache
            except Exception as exc:
                log.warning(
                    "Could not lazy-load Schema KB from %s: %s -- "
                    "KB tools disabled for this session.",
                    kb_path, exc,
                )
                return None

        log.debug(
            "Schema KB not found at %s -- KB tools disabled. "
            "Run: sentinel dictionary load",
            kb_path,
        )
        return None

    @staticmethod
    def _dispatch_dict_tool(
        tool_name: str,
        args_json: str,
        dict_tools: dict,
    ) -> str:
        """Call a Schema KB tool by name with JSON args string.

        Returns a JSON string (pretty-printed) of the result, or an error
        message string if the call fails.
        """
        fn = dict_tools.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown KB tool: {tool_name!r}"})
        try:
            raw = args_json.strip()
            kwargs: dict = json.loads(raw) if raw and raw != "{}" else {}
            result = fn(**kwargs)
            return json.dumps(result, indent=2, default=str)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Bad JSON args for {tool_name}: {e}"})
        except TypeError as e:
            return json.dumps({"error": f"Wrong args for {tool_name}: {e}"})
        except Exception as e:
            return json.dumps({"error": f"{tool_name} failed: {e}"})

    # -- Findings helpers (P1.B + P1.E) -------------------------------------

    def _findings_tools_(
        self, dataset_name: str | None = None,
    ) -> dict | None:
        """Return the bound findings tool callables, lazily creating them.

        Returns *None* if no FindingsLog can be created (which currently
        means an explicit ``findings_log=None`` plus a non-writeable
        default path -- extremely rare).
        """
        if self._findings_tools_cache is not None:
            return self._findings_tools_cache

        # Resolve / construct the log
        if self._findings_log is None:
            from sentinel.findings.log import FindingsLog
            if self._findings_path:
                path = Path(self._findings_path)
            else:
                ds = dataset_name or "default"
                path = Path(_DEFAULT_FINDINGS_DIR) / f"{ds}.jsonl"
            try:
                self._findings_log = FindingsLog(path)
            except Exception as exc:
                log.warning("Could not create FindingsLog at %s: %s -- "
                            "findings tools disabled.", path, exc)
                return None

        from sentinel.findings.tool import make_findings_tools
        self._findings_tools_cache = make_findings_tools(
            log=self._findings_log,
            dataset_name=dataset_name,
            dataset_memory=self._dataset_memory,
        )
        return self._findings_tools_cache

    @staticmethod
    def _dispatch_findings_tool(
        tool_name: str,
        args_json: str,
        findings_tools: dict,
    ) -> tuple[str, str | None, list[str]]:
        """Call record_finding / list_findings.

        Returns (observation_str, finding_id_or_None, warnings_list).
        ``finding_id`` is set only on a successful ``record_finding``.
        """
        fn = findings_tools.get(tool_name)
        if fn is None:
            return (
                json.dumps({"ok": False,
                            "error": f"Unknown findings tool: {tool_name!r}"}),
                None,
                [],
            )
        try:
            raw = args_json.strip()
            kwargs: dict = json.loads(raw) if raw and raw != "{}" else {}
            result = fn(**kwargs)
            obs = json.dumps(result, indent=2, default=str)
            fid = result.get("id") if isinstance(result, dict) and result.get("ok") else None
            warns = result.get("warnings", []) if isinstance(result, dict) else []
            return obs, fid, list(warns)
        except json.JSONDecodeError as e:
            return (
                json.dumps({"ok": False,
                            "error": f"Bad JSON args for {tool_name}: {e}"}),
                None,
                [],
            )
        except TypeError as e:
            return (
                json.dumps({"ok": False,
                            "error": f"Wrong args for {tool_name}: {e}"}),
                None,
                [],
            )
        except Exception as e:
            return (
                json.dumps({"ok": False,
                            "error": f"{tool_name} failed: {e}"}),
                None,
                [],
            )

    # -- Phase 2 helpers (P2) --------------------------------------------------

    def _phase2_tools_map_(self) -> dict | None:
        """Return the Phase 2 tool map, lazily building it if needed.

        Returns *None* if no Phase2Tools instance is available.
        """
        if self._phase2_map is not None:
            return self._phase2_map

        if self._phase2_tools is not None:
            self._phase2_map = self._phase2_tools.as_tool_map()
            return self._phase2_map

        log.debug("Phase 2 tools not configured -- P2 tool actions are disabled.")
        return None

    @staticmethod
    def _dispatch_phase2_tool(
        tool_name: str,
        args_json: str,
        phase2_map: dict,
    ) -> str:
        """Call a Phase 2 tool by name with JSON args string.

        The callable is expected to accept a single dict and return a JSON string.
        Returns a JSON error string on failure.
        """
        fn = phase2_map.get(tool_name)
        if fn is None:
            return json.dumps({"ok": False, "error": f"Unknown Phase 2 tool: {tool_name!r}"})
        try:
            raw = args_json.strip()
            args: dict = json.loads(raw) if raw and raw != "{}" else {}
            return fn(args)
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"Bad JSON args for {tool_name}: {e}"})
        except Exception as e:
            return json.dumps({"ok": False, "error": f"{tool_name} raised: {e}"})

    def _finalize_result(
        self,
        *,
        answer: str,
        question: str,
        trace: list[dict],
        tool_calls: int,
        tokens: int,
        elapsed_ms: float,
        charts: list[str],
        finding_ids: list[str],
        finding_warnings: list[str],
        dataset_name: str | None,
    ) -> AgentResult:
        """Build the AgentResult and (if enabled) write a notebook journal.

        Journal writing failures must NEVER break the result -- they're
        logged and the journal_path is left None.
        """
        journal_path: str | None = None
        if self._write_journal and self._journal_dir:
            try:
                from sentinel.notebook.journal import JournalWriter
                writer = JournalWriter(output_dir=self._journal_dir)
                p = writer.write(
                    question=question,
                    trace=trace,
                    answer=answer,
                    dataset_name=dataset_name,
                    finding_ids=finding_ids,
                    finding_warnings=finding_warnings,
                    elapsed_ms=elapsed_ms,
                )
                journal_path = str(p)
                self._audit.log("journal_written", path=journal_path,
                                question=question[:200],
                                n_findings=len(finding_ids))
            except Exception as exc:
                # Don't fail the agent if nbformat is missing or fs is RO
                log.warning("Journal writing failed: %s", exc)
                self._audit.log("journal_failed", error=str(exc))

        return AgentResult(
            answer=answer,
            thinking_trace=trace,
            tool_calls=tool_calls,
            total_tokens=tokens,
            elapsed_ms=elapsed_ms,
            charts=charts,
            finding_ids=finding_ids,
            finding_warnings=finding_warnings,
            journal_path=journal_path,
        )

    @staticmethod
    def _extract_balanced_json(text: str, start_idx: int) -> tuple[str, int] | None:
        """Extract a balanced JSON object starting at ``text[start_idx]``.

        Walks brace depth, respecting string literals and escapes.  Returns
        (json_substring, end_index) or None if no balanced object is found.

        We need this for findings tools whose args can be deeply nested
        (priors_tested arrays, evidence_result dicts).  A non-greedy
        ``\\{.*?\\}`` regex stops at the first ``}`` and breaks on every
        nested object.
        """
        # Skip whitespace before the opening brace
        i = start_idx
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            return None

        depth = 0
        in_string = False
        escape_next = False
        start = i
        while i < len(text):
            c = text[i]
            if escape_next:
                escape_next = False
            elif in_string:
                if c == "\\":
                    escape_next = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1], i + 1
            i += 1
        return None

    # -- Context / executor helpers -----------------------------------------

    def _router_(self) -> ContextRouter:
        if self._router is None:
            if self.digest is None:
                raise ValueError("No digest. Call engine.profile() first.")
            self._router = ContextRouter(self.digest, self.knowledge)
        return self._router

    def _executor_(self) -> CodeExecutor:
        if self._executor is None:
            if self.backend is None:
                raise ValueError("No data backend for code execution.")
            self._executor = CodeExecutor(self.backend.df)
        return self._executor

    @staticmethod
    def _user_prompt(
        question: str,
        ctx,
        knowledge_context: str | None = None,
    ) -> str:
        parts = [ctx.tier0]
        if ctx.tier1:
            parts.append(ctx.tier1)
        if ctx.knowledge:
            parts.append("=== DOMAIN KNOWLEDGE ===\n" + ctx.knowledge)
        if knowledge_context:
            parts.append(
                "=== INSTITUTIONAL MEMORY (from previous sessions) ===\n"
                + knowledge_context
            )
        parts.append(f"\n=== YOUR TASK ===\n{question}")
        return "\n\n".join(parts)

    def _run_code(self, code: str) -> str:
        ex = self._executor_()
        ok, reason = ex.validate_code(code)
        if not ok:
            return f"Error: blocked -- {reason}"
        result = ex.execute(code)
        return result.output if result.success else f"Error:\n{result.error}"

    def _run_chart(self, code: str) -> str:
        """Execute chart code with the viz object in scope."""
        ex = self._executor_()
        ok, reason = ex.validate_code(code)
        if not ok:
            return f"Error: blocked -- {reason}"

        import builtins
        _SAFE = (
            "abs", "all", "any", "bool", "dict", "enumerate", "filter",
            "float", "format", "hasattr", "int", "isinstance", "iter",
            "len", "list", "map", "max", "min", "next", "print", "range",
            "repr", "reversed", "round", "set", "sorted", "str", "sum",
            "tuple", "type", "zip",
        )
        safe_builtins = {n: getattr(builtins, n) for n in _SAFE}

        import io
        import numpy as np
        import pandas as pd
        from contextlib import redirect_stdout

        local_ns = {
            "df": ex._df.copy(), "pd": pd, "np": np, "viz": self._visualizer,
        }
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, {"__builtins__": safe_builtins}, local_ns)
            return buf.getvalue().strip() or "[Chart generated]"
        except Exception:
            import traceback
            return f"Error:\n{traceback.format_exc()[-300:]}"

    def _synthesise(self, findings: list[dict]) -> str:
        text = "\n\n".join(
            f"Q: {f['question']}\nA: {f.get('finding', f.get('error', 'Failed'))}"
            for f in findings
        )
        try:
            r = self.llm.generate(
                [
                    {"role": "system", "content": (
                        "Synthesize these EDA findings into an executive summary. "
                        "Top 5 findings with numbers, recommended features, "
                        "data-quality blockers, next steps.  Be specific."
                    )},
                    {"role": "user", "content": text},
                ],
                max_tokens=4096,
            )
            return r.content
        except Exception as e:
            return f"Synthesis failed: {e}"

    @staticmethod
    def _ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)
