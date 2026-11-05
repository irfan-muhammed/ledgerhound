"""
ledgerhound.shell.tools.base
============================

Session state, the tool registry, gated dispatch and the audit log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ledgerhound.shell.tools.common import _PLAN_META_TOOLS, _REPEAT_FAILURE_NOTE, _brief, _failure_signature, _norm_args_for_failure, _ts
from ledgerhound.shell.tools.specs import TOOL_SPECS



class AnalystToolsBase:
    """Session state, the tool registry, gated dispatch and the audit log."""

    def __init__(self, project=None, out_root: str | Path = "run_outputs",
                 renderer: Callable[[str], None] = print, registry=None,
                 gate=None, autonomous: bool = False, use_kernel: bool = False,
                 methodology: str = "", methodology_mode: str = "",
                 isolation=None, exec_fallback: bool = True):
        self.project = project
        self.render = renderer
        # The LLM driving this session, if any. The tool layer only reads
        # its endpoint (compute routing); ConversationLoop sets the real one.
        self.llm = getattr(self, "llm", None)
        # autonomous: dev/eval flag (V3-006). When on, any final reply that
        # is a bare, non-risk-tier question gets auto-answered with "proceed
        # with your best judgment" instead of surfacing to a human -- so a
        # fluidity run measures only the questions that genuinely needed a
        # person. Off by default; production shells leave real questions to
        # the user. Tools read it too (methodology choice, ask_user).
        self.autonomous = autonomous
        # use_kernel (V3-009): run code in a real, persistent IPython kernel
        # instead of a restricted local `exec`. Opt-in and OFF by default --
        # starting a kernel is slow (a real subprocess) and this keeps the
        # test suite and any embedding code fast/deterministic unless a
        # caller explicitly asks for it. Falls back to exec automatically if
        # jupyter_client/ipykernel aren't installed, or if the kernel fails
        # to start (see _tool_load_data).
        self.use_kernel = use_kernel
        self._kernel = None
        self._kernel_active = False
        # ADR 0001: how the kernel is sandboxed (an IsolationPolicy; None =
        # off). `isolation_report` is what was actually enforced on the
        # current kernel; `_kernel_refusal` is set when the policy required a
        # level this machine cannot give, and run_python then refuses rather
        # than falling back to the in-process exec sandbox.
        self.isolation = isolation
        self.isolation_report = None
        self._kernel_refusal = ""
        # exec_fallback=False (MCP mode): code runs ONLY in the kernel. When
        # the kernel is missing or failed to start, run_python refuses and
        # says why (_kernel_note) instead of running in-process.
        self.exec_fallback = exec_fallback
        self._kernel_note = ""
        self.run_dir = Path(out_root) / _ts()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame | None = None
        # The declared goal this session is grounded to (unit of analysis,
        # entity key, phase). Pinned every turn and ENFORCED in
        # _tool_run_python -- see ledgerhound/shell/session_goal.py.
        from ledgerhound.shell.session_goal import SessionGoal
        self.goal = SessionGoal()
        # The declared phase sequence, when one is adopted (V3-043). None
        # means "no methodology" -- the analyst is free-forming, and the
        # gates fall back to the session goal alone.
        #
        # Selection is the ANALYST's call, not the model's: the senior
        # feedback that produced this feature was precisely that each
        # analyst follows their own approach from experience. So it is set
        # from the CLI (--methodology, for unattended/Kubeflow runs) or
        # asked once on first data load -- never silently guessed.
        self.methodology = None
        self._methodology_asked = False
        if methodology:
            try:
                from ledgerhound.methodology import load_template
                m = load_template(methodology, mode=methodology_mode)
                if m is not None:
                    # load_template already set the correct active phase
                    # (Methodology.apply_mode_start() moves it for
                    # mode='fe_only') -- do not override it here.
                    self.methodology = m
                    self._methodology_asked = True
            except Exception:  # noqa: BLE001 -- never block startup
                pass
        # True when self._df is only a bounded schema sample because the real
        # dataset is large and lives in the kernel (see _tool_load_data).
        self._df_is_sample: bool = False
        self._true_n_rows: int = 0
        # False when _true_n_rows came from a byte-scaled CSV estimate rather
        # than exact parquet metadata -- rendered as "~N", never as fact.
        self._true_rows_exact: bool = True
        self._ns: dict[str, Any] | None = None
        # Issue keys already surfaced by the data-comparability check, so a
        # warning fires once per real problem instead of on every cell.
        self._data_issues_seen: set[str] = set()
        # V3-032: same contract, for the leakage check (target-derived /
        # leak-lexicon-named columns).
        self._leakage_issues_seen: set[str] = set()
        # V3-032 Tier 4: same contract, for the output-grain check
        # (duplicate account rows in an exported feature sheet).
        self._grain_issues_seen: set[str] = set()
        # V3-032 (post-audit gap closure): same contract, for the numeric-
        # stability check (inf/-inf from an unguarded division) and the
        # ambiguous-column check (operational/workflow artifacts).
        self._numeric_issues_seen: set[str] = set()
        self._memory_issues_seen: set[str] = set()
        self._ambiguity_issues_seen: set[str] = set()
        # V3-017: cheap (rows+columns) post-mutation fingerprint, reset on
        # every load_data; null-rate profile is cached lazily (only ever
        # computed once a fingerprint change already fired), so the SECOND
        # mutation onward gets a genuine null-rate delta, not just a snapshot.
        self._last_fp: dict | None = None
        self._last_null_rates: dict | None = None
        # V3-016: one-step undo snapshot for medium-tier run_python calls
        # (exec-mode, small/medium data only -- see _maybe_snapshot_for_undo).
        self._undo_snapshot: pd.DataFrame | None = None
        # dataset card (V3-011): loaded/created in _tool_load_data, requires
        # a project (persistence across sessions is the point -- see
        # ledgerhound.knowledge.dataset_card).
        self._card = None
        self._data_sha256: str = ""
        self._digest: dict | None = None
        self._digest_summary: str = ""
        self._digest_from_card: bool = False
        self._failure_log: dict[tuple, int] = {}
        self._code_log = self.run_dir / "session_code.py"
        self._findings_path = self.run_dir / "findings.jsonl"
        self._transcript_path = self.run_dir / "transcript.jsonl"
        # V5-002: per-chapter test families. A finding found on the third look
        # and one found on the two-hundredth are not equally believable, and
        # nothing recorded the difference before this.
        from ledgerhound.findings.search_effort import SearchEffortLedger
        self._effort = SearchEffortLedger(self.run_dir / "search_effort.json")
        # V3-045: which phases have produced a saved report, and how far
        # into findings.jsonl the last report consumed. Both drive the
        # "every phase leaves an artifact" gate in advance_phase.
        self._reports_by_phase: dict[str, list[str]] = {}
        self._findings_consumed = 0
        # Set when a domain profile matches at load time (V3-048).
        self._domain_profile: dict | None = None
        # V3-012: canonical, hash-chained event stream. transcript.jsonl
        # stays as a parallel legacy log (see ledgerhound.runtime.events
        # module docstring for why it isn't derived from this yet).
        from ledgerhound.runtime.events import EventStream
        self._events = EventStream(self.run_dir / "events.jsonl")
        # Running scorecard counts (ledgerhound.runtime.session_log.SessionStats),
        # fed by the stream as each event is written -- so the per-turn
        # scorecard never re-reads events.jsonl. Seeded from anything already
        # in the file (normally nothing: every session gets a new run folder).
        from ledgerhound.runtime.session_log import SessionStats
        self._session_stats = SessionStats()
        for _ev in self._events.load():
            self._session_stats.observe(_ev)
        self._events.listeners.append(self._session_stats.observe)
        # Session-quality observability: ONE central, size-capped file
        # across the whole deployment (every user, every project), one
        # compact scorecard line per turn -- see ledgerhound.runtime.session_log
        # module docstring. Distinct from events.jsonl (per-run, unbounded,
        # full fidelity) and computed FROM it, so the two can't disagree.
        self._session_started_at = datetime.now(timezone.utc).isoformat()
        from ledgerhound.tools.plan_tools import PlanState
        self.plan = PlanState()
        # V6.3 postmortem fix: counts substantive tool calls (anything that
        # isn't plan/goal bookkeeping) so update_plan can refuse to close a
        # step "done" when nothing actually ran for it. See plan_tools.py.
        self._evidence_count = 0
        # V6.3 postmortem "Filter 1": the verbatim first user turn, captured
        # in send() before anything distills it. See SessionGoal.origin_instruction.
        self._first_user_turn_text = ""
        # Guardrails (ledgerhound.shell.guardrails). Providers are lambdas
        # because goal/methodology are replaced during a session.
        from ledgerhound.shell.guardrails import (
            ExecBlocklist,
            ExportGrainGate,
            ModelingGate,
            RestatementGuard,
            StatisticalAssumptionsGate,
        )
        self._stat_gate = StatisticalAssumptionsGate(
            n_rows=self._true_frame_rows,
            on_issue=lambda **kw: self._events.append("stat_check", **kw))
        self._code_gates = [
            ModelingGate(lambda: self.goal, lambda: self.methodology),
            ExportGrainGate(lambda: self.goal, lambda: self.methodology,
                            self._export_frames),
            # V5-003: the first gate that looks at INFERENCE rather than
            # data -- valid code, invalid conclusion.
            self._stat_gate,
        ]
        self._exec_blocklist = ExecBlocklist()
        self._restatement_guard = RestatementGuard()
        self.registry = registry or self._default_registry(gate)

    def _refresh_schema_pin(self) -> None:
        """Hook: the session's data, goal or phase changed. A tool-only
        session has no conversation to re-pin; ConversationLoop overrides
        this to keep its pinned context message current."""

    # -- unified registry ---------------------------------------------------
    def _default_registry(self, gate=None):
        """Build the session ToolRegistry from TOOL_SPECS + bound _tool_* methods,
        plus the plan checklist tools (propose_plan/update_plan) -- on by
        default so forced continuation (send()) always has a plan to check
        against. Extra tool families (config, git, jira) are registered on
        top by the CLI according to what the project has configured."""
        from ledgerhound.tools import ToolDef, ToolRegistry
        from ledgerhound.tools.plan_tools import build_plan_tools
        reg = ToolRegistry(gate=gate)
        for spec in TOOL_SPECS:
            f = spec["function"]
            handler = getattr(self, f"_tool_{f['name']}", None)
            if handler is None:
                continue
            reg.register(ToolDef(name=f["name"], description=f["description"],
                                 parameters=f["parameters"], fn=handler))
        reg.register_many(build_plan_tools(
            self.plan, renderer=self.render,
            plan_path=self.run_dir / "plan.json",
            evidence_counter=lambda: self._evidence_count,
            event_logger=self._log))
        return reg

    # -- durable transcript (debugging / audit -- never raises) ------------
    def _log(self, event: str, **fields) -> None:
        try:
            entry = {"ts": _ts(), "event": event, **fields}
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str)[:8000] + "\n")
        except Exception:
            pass
        # V3-012: every _log() call also lands in the canonical, hash-chained
        # event stream -- one chokepoint, so every existing call site gets
        # tamper-evidence for free. Logging must never break the session.
        try:
            self._events.append(event, **fields)
        except Exception:
            pass

    def _run_tool(self, name: str, args: dict, *, high_tier_decision=None) -> str:
        self.render(f"  - {name}({_brief(args)})")
        from ledgerhound.analyst.hitl import classify_action
        tier = classify_action(name)

        # V3-016: high-tier actions are gated IN-LINE, before dispatch --
        # never deferred across turns. `high_tier_decision` lets `send()`
        # pre-gate a whole response's simultaneous tool_calls as one batch
        # (see the native tool_calls handling below); a single high-tier
        # call with no batch gates itself right here.
        from ledgerhound.shell.approvals import NeedsPersonDecision
        if tier == "high":
            try:
                decision = high_tier_decision if high_tier_decision is not None \
                    else self.registry.gate.high(**self._decision_package(name, args))
            except NeedsPersonDecision as e:
                return self._needs_person(name, e)
            self._log("hitl", tier="high", name=name, approved=decision.approved,
                      answer=decision.answer,
                      approved_by=getattr(decision, "approved_by", "person"))
            if not decision.approved:
                result = (f"declined: {name} was not approved this turn -- "
                          f"do not retry it; ask what they want instead.")
                self._log("tool_result", name=name, result=result, success=False)
                return result

        if name == "run_python":
            self._maybe_snapshot_for_undo()

        # V3-012: tool_call (pre-dispatch) and tool_result (post-dispatch)
        # are two separate events in the canonical stream -- see
        # ledgerhound.eval.fluidity for the one consumer updated to match.
        from ledgerhound.tools.config_tools import redact_args
        self._log("tool_call", name=name, args=redact_args(name, args))
        try:
            result = self._dispatch(name, args)
        except NeedsPersonDecision as e:
            return self._needs_person(name, e)
        sig = _failure_signature(result)
        success = sig is None
        if success and name not in _PLAN_META_TOOLS:
            self._evidence_count += 1
        if success and name == "run_python":
            self._note_cell_ran()       # counted "roles unconfirmed" in the report
        if sig is not None:
            key = (name, _norm_args_for_failure(args), sig)
            count = self._failure_log.get(key, 0) + 1
            self._failure_log[key] = count
            if count >= 2:
                result += _REPEAT_FAILURE_NOTE
                self._log("identical_failure", name=name, signature=sig, count=count)

        if tier == "medium" and success:
            note = self._medium_tier_note(name)
            self.registry.gate.medium(name, note=note)
            self._log("hitl", tier="medium", name=name)
            result = f"{result}\n\n{note}"

        # V5-005: intra-result compaction. One oversized result -- a 400-column
        # describe(), a value_counts() over a high-cardinality VPA column --
        # can eat most of the window in a single turn, and the history
        # compactor only reacts on the NEXT turn, once it is already in the
        # transcript. Shrink it here, before it enters, keeping both ends and
        # spilling the full text to the run folder.
        try:
            from ledgerhound.shell.intra_compaction import compact_result
            outcome = compact_result(result, spill_dir=self.run_dir / "results")
            if outcome.compacted:
                self._events.append("intra_compaction", tool=name,
                                    original_chars=outcome.original_chars,
                                    saved_chars=outcome.saved_chars,
                                    spill_path=outcome.spill_path)
                result = outcome.text
        except Exception:  # noqa: BLE001 -- never lose a result to compaction
            pass

        self._log("tool_result", name=name, result=result[:2000], success=success)
        return result

    def _needs_person(self, name: str, e) -> str:
        """A tool stopped for a person's decision nobody has given yet (MCP
        without elicitation). Not a failure and not evidence: the agent is
        told exactly what to ask and how to retry."""
        result = e.tool_result(name)
        self._log("hitl", tier="pending", name=name, approved=False,
                  answer="", approved_by="nobody yet", question=e.question[:500])
        self._log("tool_result", name=name, result=result[:2000], success=False)
        return result

    def agent_approvals(self) -> list:
        """Decisions an MCP host agent relayed (approved_by="agent"): every
        report flags these as approved by the agent, not verified by a person."""
        gate = getattr(self.registry, "gate", None)
        return [d for d in getattr(gate, "log", [])
                if getattr(d, "approved_by", "") == "agent"]

    def _decision_package(self, name: str, args: dict) -> dict:
        """V3-016: builds the {question, findings_so_far, options,
        recommendation, consequence} package for a high-tier action. The
        model already decided to call this tool -- "recommendation" is
        mechanically "proceed" (what the agent itself proposed); a human
        with a real prompt fn is the actual check, not this harness."""
        findings = []
        if self._findings_path.exists():
            try:
                for line in self._findings_path.read_text(encoding="utf-8").splitlines()[-5:]:
                    findings.append(json.loads(line).get("claim", "")[:150])
            except Exception:
                pass
        consequences = {
            "save_report": "creates or overwrites a report file in the run folder.",
            "git_pull": "changes tracked files in the working tree from the remote.",
            "git_commit_push": "pushes a commit to the remote repository -- visible to the team.",
            "jira_post_update": "posts a comment on a Jira issue -- visible to the team.",
            "jira_create_issue": "creates a new Jira issue -- visible to the team.",
            "jira_transition": "changes a Jira issue's status -- visible to the team.",
            "configure_integration": "changes stored integration credentials/config.",
        }
        return {
            "question": f"About to run high-risk action '{name}' with {_brief(args)}. Proceed?",
            "findings_so_far": findings,
            "options": ("proceed", "cancel"),
            "recommendation": "proceed",
            "consequence": consequences.get(name, "an external or persisted side effect."),
        }

    _UNDO_MAX_ROWS = 500_000

    def _maybe_snapshot_for_undo(self) -> None:
        """V3-016: a real, one-step undo for small/medium exec-mode data --
        cheap enough there; a full df copy at crore-scale would be a real
        perf/memory regression, so above the threshold (and in kernel mode,
        where the data never comes back to this process) there's no copy
        and the medium-tier note says so honestly instead of promising an
        undo that doesn't exist."""
        if self._kernel_active:
            return
        if self._df is not None and len(self._df) <= self._UNDO_MAX_ROWS:
            self._undo_snapshot = self._df.copy()

    def _medium_tier_note(self, name: str) -> str:
        if name != "run_python":
            return ("(medium-risk action: proceeded automatically -- mention "
                    "what you did if it's relevant to your next reply.)")
        if self._kernel_active:
            return ("(medium-risk action: this changes data that persists for "
                    "the rest of this session. Tell the user what you changed "
                    "and why -- kernel-mode sessions don't keep an automatic "
                    "undo copy, so reloading via load_data is how to start "
                    "over if needed.)")
        if getattr(self, "_undo_snapshot", None) is not None:
            return ("(medium-risk action: this is a reversible, in-session "
                    "change. Tell the user what you changed and why, and that "
                    "they can say 'undo' -- call undo_last_transform to restore "
                    "the previous state.)")
        return ("(medium-risk action: this changes data. Tell the user what "
                "you changed and why -- this session's data is too large to "
                "keep an automatic undo copy, so reloading via load_data is "
                "how to start over if needed.)")

    def _tool_undo_last_transform(self) -> str:
        if self._kernel_active:
            return ("kernel-mode sessions don't keep an automatic undo "
                    "snapshot (the data lives in a separate process) -- "
                    "reload the data with load_data if you need to start over.")
        snap = getattr(self, "_undo_snapshot", None)
        if snap is None:
            return ("no undo snapshot available -- either nothing has been "
                    "run yet, the last change was already undone, or the "
                    f"data is too large (> {self._UNDO_MAX_ROWS:,} rows) for "
                    "this session to keep an automatic copy. Reload the data "
                    "with load_data if you need to start over.")
        self._df = snap
        if self._ns is not None:
            self._ns["df"] = snap
        self._undo_snapshot = None
        # V3-017 baseline must move with the restore, or the NEXT cell (even
        # a pure read) sees a phantom fingerprint change against the
        # pre-undo state and injects a bogus [VERIFICATION] block.
        from ledgerhound.shell.verification import fingerprint as _fp
        self._last_fp = _fp(len(snap), list(snap.columns))
        self._last_null_rates = None
        self._events.append("verification", action="undo", n_rows=len(snap))
        return f"restored the previous state ({len(snap):,} rows x {snap.shape[1]} columns)."

    def _rows_display(self) -> str:
        """Row count as text -- prefixed "~" when it came from a byte-scaled
        CSV estimate rather than exact parquet metadata, so an estimate is
        never shown as a hard number."""
        n = self._true_n_rows
        return f"{n:,}" if self._true_rows_exact else f"~{n:,}"

    # -- schema pinning -----------------------------------------------------
    def _schema_pin_block(self) -> str:
        d = self._df
        lines = ["=== ACTIVE DATA (pinned -- always current, do not re-profile "
                 "to re-see this) ==="]
        lines.append(f"loaded: {self._loaded_path or '(unknown path)'}")
        # Only a SAMPLED frame needs the load-time count -- otherwise `d` is
        # the live frame and len(d) must win, or a mutation (df = df.dropna())
        # would keep reporting the original row count.
        n_disp = self._rows_display() if self._df_is_sample else f"{len(d):,}"
        lines.append(f"shape: {n_disp} rows x {d.shape[1]} columns")
        if self._df_is_sample:
            lines.append(f"note: the kernel holds all {n_disp} rows (your "
                         f"run_python code sees the full data); this process "
                         f"kept a {len(d):,}-row schema sample only.")
        dtypes = ", ".join(f"{c}:{t}" for c, t in
                           zip(d.columns[:40], d.dtypes.astype(str).tolist()[:40]))
        lines.append(f"columns (dtype): {dtypes}")
        try:
            miss = (d.isna().mean() * 100).round(1).sort_values(ascending=False)
            top_missing = miss[miss > 0].head(5)
            if len(top_missing):
                lines.append("top null rates: " +
                             ", ".join(f"{c}={v}%" for c, v in top_missing.items()))
        except Exception:
            pass
        # Dataset card (V3-011): prior sessions' digest/decisions/findings on
        # this EXACT file, folded into the same pinned block so there's only
        # one message to track/reindex through compaction (V3-010).
        if self._card is not None:
            lines.append("")
            lines.append(self._card.summary_block())
        return "\n".join(lines)

    # -- tool dispatch ----------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> str:
        return self.registry.dispatch(name, args)

    # -- team knowledge base (automatic; see ledgerhound.knowledge.team_kb) ------
    _kb = None
    _team_findings: list = []
    _dataset_id: str = ""
    _dataset_name_override: str = ""

    # -- questions about the data (asked, never assumed) --------------------
    _open_data_questions: list = []
    _questions_reminded: bool = False

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        """Release resources held by this session (a live IPython kernel, if
        one was started). Safe to call multiple times / when none exists."""
        if self._kb is not None:
            try:
                self._kb.close()      # last upload try; anything left stays queued
            except Exception:  # noqa: BLE001
                pass
        if self._kernel is not None:
            try:
                self._kernel.shutdown()
            except Exception:
                pass
            self._kernel = None
        self._kernel_active = False

    def _workspace_root(self) -> Path:
        """V3-064: where ledgerhound_workspace/ lives (the Librarian's books,
        dataset cards, institutional knowledge -- cross-session, keyed by
        dataset, unrelated to per-run outputs). `paths.workspace_dir` in
        main_config.yaml overrides the default (project.root/
        ledgerhound_workspace)."""
        # paths.workspace_dir if set, else Agentic_eda_results/<user>/library
        from ledgerhound.paths import library_dir
        return library_dir()

    def _features_dir(self) -> Path:
        """V3-066: simplified per explicit correction -- generated feature
        tables land in THIS run's own folder (run_outputs/<timestamp>/
        features/), next to its findings/reports/events.jsonl, not in a
        separate cross-session ledgerhound_workspace/ tree keyed by dataset.
        `paths.run_outputs_dir` in main_config.yaml controls where
        run_outputs/<timestamp>/ itself lives (see cli/main.py); this is
        just the `features/` subfolder of whatever that resolved to.

        2026-09-19: with object-store data on a RAM-disk cache, features go
        next to that cache instead (a pod's PVC can be far smaller than a
        crore-row feature table) and are uploaded to the store after every
        run_python step -- see _set_run_paths."""
        d = getattr(self, "_features_override", None) or (self.run_dir / "features")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- data source + run paths (local vs object store) -------------------
    _features_override: Path | None = None
    _uploader = None
    _run_paths = None

    _force_one_dataset: bool = False

    # -- path resolution (the thing that was dumb before) -----------------
    _loaded_path: str = ""
