"""
ledgerhound.auto.runner
====================

AutoRunner -- self-healing agentic execution loop.

Flow
----
1. LLM generates a step-by-step plan from the user's goal.
2. For each step the LLM writes Python code.
3. Code runs inside the run's scratch directory (ScratchExecutor).
4. On failure the LLM reads the full traceback + code and writes a fix.
   This retries up to ``max_retries`` times per step.
5. If the LLM believes a tracked source file is buggy it proposes a patch
   via PatchStore (never auto-applies).
6. All actions go to trace.jsonl.
7. On completion: HTML report + .ipynb are written.

Safety invariant
----------------
``ledgerhound/`` and ``scripts/`` directories are NEVER written to by this
runner.  ScratchExecutor's ``_guarded_open`` enforces this at the executor
level; the runner itself also never touches those paths.
"""

from __future__ import annotations

import json, logging, os, re, signal, textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

if TYPE_CHECKING:
    from ledgerhound.session.models import Session

log = logging.getLogger(__name__)
console = Console()


# -- Config ----------------------------------------------------------------- #

@dataclass
class RunConfig:
    goal: str
    llm_config: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    data: str | None = None
    workspace: str = "ledgerhound_workspace"
    max_steps: int = 20
    max_retries: int = 5          # was 3 -- broader self-healing window
    timeout: int = 600            # was 120 -- big-data load/concat steps need room
    fast: bool = True             # skip latency-heavy LLM polish (post-filter,
                                  # LLM finding extraction); keep core self-heal
    verbose: bool = True
    session: "Session | None" = None       # attach a live session for checkpointing


# -- Prompts ---------------------------------------------------------------- #

_PLAN_PROMPT = textwrap.dedent("""\
    You are Ledgerhound Auto, an autonomous data analysis agent.

    The user wants you to accomplish the following goal:
    <goal>
    {goal}
    </goal>

    Available context:
    {context}

    Produce a numbered step-by-step execution plan (3-{max_steps} steps).
    Each step should be a concrete, executable Python task.

    IMPORTANT: Do NOT include steps that build a final HTML report or export a
    Jupyter notebook -- Ledgerhound generates a narrated report and notebook
    automatically at the end (with all plots, findings, and methodology embedded).
    Your steps should focus on ANALYSIS and producing charts/tables saved to the
    scratch dir; the reporting is handled for you. Make the LAST analysis step a
    summary/aggregation of key results, not a report-export step.

    Respond ONLY with a JSON array of objects, each with:
      "title": short step title (< 60 chars)
      "description": what this step does and what result it produces

    Example:
    [
      {{"title": "Load and inspect data", "description": "Load the parquet file into df, print shape, dtypes, and first 5 rows."}},
      {{"title": "Compute missing value rates", "description": "For each column compute pct_missing, flag columns > 20%."}}
    ]

    Output ONLY valid JSON. No markdown fences.
""")

_CODE_PROMPT = textwrap.dedent("""\
    You are writing Python code for step {step_num} of an analysis pipeline.

    Step title: {title}
    Step description: {description}

    Available variables:
    {available_vars}

    DataFrame engine: {df_engine}
    - If "cudf": ``df`` is a cuDF DataFrame on GPU. Avoid df.apply(lambda),
      iterrows(), Python UDFs, and python-side loops over rows. Prefer
      vectorised ops (df[col].mean(), df.groupby().agg()).
    - If "pandas": full pandas API available.
    Robust helpers are PRE-INJECTED -- prefer them over raw cuDF calls that
    commonly crash:
      * safe_merge(left, right, on=..., how=...)   # aligns key dtypes first
      * safe_to_datetime(series, fmt=None)          # coerces bad/mixed dates
      * safe_to_parquet(df, path)                   # no GPU-OOM, batched write
      * to_pdf(df)                                   # pandas view for pandas-only ops

    Scratch directory for saving artefacts: ``scratch`` (pathlib.Path)
    Available imports already in namespace: pd, np, plt (matplotlib), sns, json, math, re

    {knowledge}

    {baselines}

    {lessons}

    {memory}

    {librarian_block}

    Rules:
    - Apply the DOMAIN KNOWLEDGE above when it is relevant to this step.
    - Write self-contained code. Do NOT import pandas/numpy -- they are pre-imported.
    - ``df`` is shared by reference across steps. NEVER mutate it in place.
      Forbidden: df.drop(..., inplace=True), df.rename(..., inplace=True),
      df.fillna(..., inplace=True), df[col] = ..., del df[col].
      Always assign to a NEW variable: df2 = df.drop(...); df2 = df.assign(col=...).
    - Save any chart as: plt.savefig(scratch / "step{step_num}_{safe_title}.png", dpi=100, bbox_inches="tight")
    - To read a digest or previous output, use the EXACT path shown in KNOWN FILES above.
    - Set ``result = <value>`` for the key output you want logged.
    - ALSO set ``metrics = {{...}}`` -- a dict mapping short snake_case names to the
      KEY COMPUTED SCALAR FACTS of this step (numbers/counts/ratios), e.g.
      metrics = {{"fraud_rate": float(df['flag'].mean()), "rows": int(len(df)),
      "cols_missing_gt20pct": int((df.isna().mean()>0.2).sum())}}.
      These are the verifiable facts; any claim you make must come from them.
    - Use print() for intermediate info.
    - Do NOT write to any path outside ``scratch``.
    - Keep code concise (under 60 lines).

    Previous step outputs (for context):
    {prev_outputs}

    Respond with ONLY the Python code. No explanation, no markdown fences.
""")

_FIX_PROMPT = textwrap.dedent("""\
    The following Python code failed. Fix it.

    === CURRENT BROKEN CODE ===
    {code}

    === CURRENT ERROR (full traceback) ===
    {error}

    === LIVE DATAFRAME / RUNTIME STATE ===
    {df_context}

    === PRIOR FAILED ATTEMPTS (do NOT repeat these mistakes) ===
    {prior_attempts}

    {memory}

    {librarian_block}

    Step title: {title}
    Available variables: {available_vars}
    Scratch dir: ``scratch`` (pathlib.Path)

    Rules:
    - Output ONLY the corrected Python code. No markdown, no explanations.
    - If KeyError / column name mismatch, use EXACT names from LIVE STATE.
    - If IndexError on subplots/axes, recount: len(axes.flat) == nrows*ncols.
      Loop with `for i, col in enumerate(cols[:len(axes.flat)])`.
    - If reading a file, use EXACT path from KNOWN FILES.
    - If you have already failed this approach twice (see PRIOR ATTEMPTS),
      try a STRUCTURALLY DIFFERENT solution -- do not tweak; rethink.
    - Fix only the actual error. Do not rewrite unrelated logic.
""")

_REVISE_PROMPT = textwrap.dedent("""\
    You are re-planning a data-analysis pipeline mid-run because a step failed
    its entire approach (not just a fixable bug). Revise ONLY the remaining
    steps so the overall goal is still achievable, learning from the failure.

    Goal: {goal}

    Already-executed steps (do NOT repeat or change these):
    {done}

    The step that just FAILED: {failed_step}
    Failure detail:
    {failed_error}

    Originally-planned remaining steps (you may keep, drop, reorder, or replace):
    {remaining}

    What we know so far:
    {context}

    Produce the NEW remaining steps as a JSON array (max {max_steps}), each:
      "title": short title (< 60 chars)
      "description": what it does and what result it produces
    Avoid the approach that just failed. Output ONLY JSON, no fences.
""")

_LESSON_PROMPT = textwrap.dedent("""\
    Reflecting on this completed analysis, state up to 3 PROCEDURAL lessons for
    analysing THIS dataset next time -- reusable "how to" guidance, not findings.
    Examples: "Use IQR not z-score on the long-tailed amount column";
    "payee_device_id is unreliable, avoid device features".

    Goal: {goal}
    Steps and outcomes:
    {steps}

    Respond as a plain bullet list (one lesson per line, "- ..."). If there are
    no durable procedural lessons, respond with nothing.
""")

_ADAPT_PROMPT = textwrap.dedent("""\
    A data-analysis step just produced a VERIFIED discovery (computed from the
    data, not an opinion). Decide whether the remaining plan should adapt.

    Goal: {goal}

    Verified discovery (a real computed fact):
    {signal}

    Remaining planned steps:
    {remaining}

    Context so far:
    {context}

    Choose EXACTLY ONE action and respond with ONLY JSON:
      - Proceed unchanged:
        {{"action": "PROCEED"}}
      - Insert ONE focused follow-up step before continuing (only if the
        discovery clearly warrants deeper investigation):
        {{"action": "INSERT", "step": {{"title": "...", "description": "..."}}}}
      - Skip a remaining step that this discovery makes pointless (e.g. analysing
        a column that is almost entirely missing):
        {{"action": "SKIP", "skip_index": <number from the remaining list>}}

    Rules:
    - Prefer PROCEED unless the discovery clearly changes what is worth doing.
    - Never propose building reports/notebooks (handled automatically).
    - Stay focused on the original goal. Output ONLY the JSON object.
""")

_POSTFILTER_PROMPT = textwrap.dedent("""\
    The Python code below WORKS but went through several debug iterations, so it
    may contain leftover debug prints, dead code, redundant retries, or messy
    structure. Produce a CLEAN canonical version that does the same thing.

    Step: {title}

    === WORKING CODE ===
    {code}

    === ITS OUTPUT (must be preserved) ===
    {output}

    Rules:
    - Keep identical behaviour and the same saved artefacts / result value.
    - Remove debug noise, dead branches, and redundant work.
    - Do NOT add new functionality or change column names.
    - Output ONLY the cleaned Python code. No markdown, no explanation.
""")

_FINDING_PROMPT = textwrap.dedent("""\
    From this analysis step's output, extract the single most important factual
    finding as ONE sentence. It must be concrete and cite numbers/columns when
    present (e.g. "riskscore has 12.4% missing, concentrated in 2026-02").
    If the output contains no real finding, respond with exactly: NONE

    Step: {title}
    Output:
    {output}
""")

_PATCH_PROMPT = textwrap.dedent("""\
    The code failed with an error that may indicate a bug in a tracked source file.

    === ERROR ===
    {error}

    === FAILING CODE ===
    {code}

    === TRACKED FILE ({tracked_file}) ===
    {file_content}

    If you see a genuine bug in the tracked file (not a bug in the generated code),
    provide the corrected version of the ENTIRE tracked file.

    Respond with EXACTLY:
    PATCH_NEEDED: yes
    REASON: <one sentence>
    FIXED_CODE:
    <complete corrected file content>

    If the bug is in the generated code (not the tracked file), respond:
    PATCH_NEEDED: no
""")


# -- Runner ----------------------------------------------------------------- #

class AutoRunner:
    """Execute the self-healing agentic loop for a given goal."""

    def __init__(self, config: RunConfig):
        self.cfg = config
        self._session = config.session
        self._run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

        # Workspace: session-scoped if session attached, else legacy path
        if self._session:
            ws = Path(self._session.workspace_dir)
        else:
            ws = Path(config.workspace)

        self._run_dir = ws / "auto_runs" / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._patch_dir = ws / "patch_candidates"
        self._trace: list[dict] = []
        self._paused = False          # set True by SIGINT handler
        self._llm = self._build_llm()
        self._executor = self._build_executor()

        # Context compressor
        if self._session:
            from ledgerhound.session.compressor import ContextCompressor
            self._compressor = ContextCompressor(self._session, self._llm)
        else:
            self._compressor = None

        # File index -- knows where every file lives (distinct from the memory
        # Librarian below; kept on its own attribute to avoid a name collision).
        if self._session:
            from ledgerhound.session.file_index import FileIndex
            self._file_index = FileIndex(self._session)
            # Register the data file(s) immediately
            if config.data:
                self._file_index.register(config.data, "data",
                                          f"input dataset: {Path(config.data).name}")
        else:
            self._file_index = None

        # Expert knowledge base -- the organization fraud/EDA/PSI heuristics injected at
        # code-gen so generated code follows known-good domain patterns
        # instead of rediscovering them (cf. AutoMind, but domain-specific).
        try:
            from ledgerhound.knowledge import KnowledgeBase
            kb_dir = Path(__file__).resolve().parents[1] / "knowledge" / "defaults"
            self._kb = KnowledgeBase(kb_dir)
        except Exception as e:
            log.warning("Knowledge base unavailable: %s", e)
            self._kb = None

        # Bounded plan revisions (backtracking). Prevents thrash.
        self._revisions_used = 0
        self._max_revisions = 2
        self._memory_cache: str | None = None   # cached cross-step memory block

        # Long-term Data Memory -- accumulates grounded facts across runs so the
        # agent develops "instinct" (historical baselines + drift detection).
        self._dataset_key = (Path(config.data).stem if config.data else "adhoc")
        mem_root = (ws / "data_memory" if self._session
                    else Path(config.workspace) / "data_memory")
        # Route all long-term memory through the Librarian gateway (the moat).
        # The private FactMemory/LessonMemory are exposed as the existing
        # attributes so every call site keeps working unchanged; promotion to
        # the shared tier and shared+private recall now happen through the
        # gateway. When the shared tier is empty (fresh workspace), behaviour is
        # identical to the previous per-store wiring.
        try:
            from ledgerhound.memory.librarian import Librarian
            self._librarian = Librarian(mem_root)
            self._fact_memory = self._librarian.private_facts
            self._lesson_memory = self._librarian.private_lessons
        except Exception as e:
            log.warning("Librarian unavailable: %s", e)
            self._librarian = None
            self._fact_memory = None
            self._lesson_memory = None
        self._baseline_cache: str | None = None
        self._lessons_cache: str | None = None

        # Vector/associative memory -- "have we seen something like this before?"
        # Lazy: only instantiated if needed (embedder import can be heavy).
        self._mem_root = mem_root
        self._vector_memory = None
        self._vector_inited = False

        self._install_exit_handler()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute the full goal and return the run directory."""
        self._print_header()

        # Resume: figure out which steps are already done
        completed = self._session.completed_step_indices() if self._session else set()
        if completed:
            console.print(
                f"  [dim]Resuming -- {len(completed)} step(s) already complete, skipping.[/dim]"
            )

        # Re-verify historical facts against the CURRENT data before planning:
        # facts that no longer reproduce are demoted, so the agent never trusts
        # a rotted memory. (Memory that can recompute itself cannot rot.)
        self._reverify_memory()

        # Freeze the plan on first run; reuse it verbatim on resume so the
        # checkpoint indices always line up with the same steps.
        if self._session and self._session.plan:
            plan = self._session.plan
            console.print(f"  [dim]Reusing frozen plan ({len(plan)} steps) from session.[/dim]")
        else:
            plan = self._make_plan()
            if self._session:
                self._session.plan = plan
                self._session.save()
        self._log_trace({"event": "plan", "steps": plan})

        prev_outputs: list[str] = []

        # Seed prev_outputs from existing checkpoints so skipped steps
        # still contribute context to later steps
        if self._session:
            for cp in sorted(self._session.checkpoints, key=lambda c: c.step_index):
                prev_outputs.append(
                    f"Step {cp.step_index} ({cp.step_title}): {cp.output_summary}"
                )

        # Index-based loop so we can revise the remaining plan on failure
        # (backtracking). `idx` is 0-based into `plan`; step number is idx+1.
        idx = 0
        while idx < len(plan):
            i = idx + 1
            step = plan[idx]

            if self._paused:
                console.print(
                    f"\n[yellow]Paused at step {i}. "
                    f"Resume with: ledgerhound start (pick this session)[/yellow]"
                )
                break

            if i in completed:
                console.print(f"  [dim]Step {i}: {step['title']} (already done)[/dim]")
                idx += 1
                continue

            console.print(Rule(f"[bold cyan]Step {i}/{len(plan)}: {step['title']}[/bold cyan]"))
            entry = self._execute_step(i, step, prev_outputs)
            self._trace.append(entry)
            self._flush_trace()

            if entry["success"]:
                summary = (entry.get("output") or "")[:300]
                prev_outputs.append(f"Step {i} ({step['title']}): {summary}")

                # Save checkpoint so we can skip this step on resume
                if self._session:
                    self._session.save_checkpoint(i, step["title"], summary)
                    self._memory_cache = None   # new completed step to surface
                    self._session.append_message(
                        "assistant", summary, tag=f"step_{i}"
                    )
                    # Extract a durable finding into sacred memory (never
                    # compressed; injected into all future code-gen prompts).
                    self._record_finding(i, step, entry.get("output", ""),
                                         entry.get("metrics", {}))
                    if self._compressor:
                        self._compressor.maybe_compress()

                if entry.get("artefacts"):
                    console.print(f"  [dim]Saved: {', '.join(entry['artefacts'])}[/dim]")

                # Adaptive planning: if the step's COMPUTED metrics fire a
                # deterministic signal (data-quality cliff, drift, anomaly),
                # let the planner adapt the remaining steps. The trigger is a
                # fact, never the LLM's reading of prose -- so it can't fire on
                # a hallucinated finding.
                signal = self._detect_signal(entry.get("metrics", {}))
                if (signal and self._revisions_used < self._max_revisions
                        and idx + 1 < len(plan)):
                    revised = self._adapt_plan(plan, idx, signal, prev_outputs)
                    if revised is not None:
                        plan = revised
                        self._revisions_used += 1
                        if self._session:
                            self._session.plan = plan
                            self._session.save()
            else:
                console.print(f"  [red]Step {i} failed after all retries.[/red]")
                if self._session:
                    self._session.append_message(
                        "assistant",
                        f"Step {i} failed: {entry.get('error','')[:200]}",
                        tag=f"step_{i}_error",
                    )
                # Backtracking: a whole-approach failure (not a transient bug)
                # is a signal the remaining plan may be wrong. Revise the
                # downstream steps given what we just learned.
                if self._revisions_used < self._max_revisions and idx + 1 < len(plan):
                    revised = self._revise_plan(plan, idx, entry, prev_outputs)
                    if revised is not None:
                        plan = revised
                        self._revisions_used += 1
                        if self._session:
                            self._session.plan = plan
                            self._session.save()

            idx += 1

        result_path = self._finalize()
        self._shutdown_executor()
        return result_path

    def _reverify_memory(self) -> None:
        """Re-run stored fact computations against current df; demote rotted."""
        if self._fact_memory is None:
            return
        df = getattr(self._executor, "_df", None)
        if df is None:
            return
        try:
            results = self._fact_memory.reverify(self._dataset_key, df)
        except Exception as e:
            log.warning("Memory re-verification failed: %s", e)
            return
        bad = [r for r in results if r.get("reproduced") is False]
        if bad:
            self._fact_memory.save()
            self._baseline_cache = None
            console.print(Panel(
                "\n".join(f"  - {r['name']}: was {r.get('baseline'):.4g}, "
                          f"now {r.get('recomputed'):.4g} -- demoted (contradicted)"
                          for r in bad[:10]),
                title="Data Memory -- stale facts demoted (re-verification)",
                border_style="red"))

    def _consolidate_memory(self) -> list[str]:
        """Push the run's grounded metrics into Data Memory; return drift notes.

        This is how instinct forms: each run's computed facts update the
        per-dataset baselines, and deviations are flagged as drift.
        """
        if self._fact_memory is None:
            return []
        run_id = self._run_id
        notes: list[str] = []
        try:
            for entry in self._trace:
                metrics = entry.get("metrics") or {}
                results = self._fact_memory.consolidate_run(
                    self._dataset_key, run_id, metrics)
                for c in results:
                    if c.classification == "DRIFT":
                        notes.append(c.message)
            self._fact_memory.promote()    # recurring consistent facts -> stable
            self._fact_memory.save()
            # Governed promotion: corroborated private facts -> shared tier
            # (deterministic gate; no-op until facts clear the thresholds).
            if self._librarian is not None:
                try:
                    self._librarian.promote(self._dataset_key)
                except Exception as e:
                    log.warning("Shared-tier promotion skipped: %s", e)
        except Exception as e:
            log.warning("Memory consolidation failed: %s", e)
            return []
        if notes:
            console.print(Panel(
                "\n".join(f"  - {n}" for n in notes[:10]),
                title="Data Memory -- drift vs historical baselines",
                border_style="yellow"))
        return notes

    def _derive_lessons(self, drift_notes: list[str]) -> None:
        """Record procedural lessons from this run.

        Deterministic lessons (grounded): drift events become "watch X" lessons.
        Optional LLM lessons (non-fast): distil 1-3 procedural takeaways. Lessons
        are advisory hints; confidence grows only when they recur across runs.
        """
        if self._lesson_memory is None:
            return
        run_id = self._run_id
        try:
            for note in drift_notes:
                metric = note.split("=")[0].replace("DRIFT:", "").strip().strip("'")
                self._lesson_memory.add(
                    f"Watch '{metric}': it has drifted from its historical "
                    f"baseline -- verify and investigate before relying on it.",
                    self._dataset_key, run_id=run_id, tags=["drift", metric],
                    source="derived")

            if not self.cfg.fast:
                lessons = self._distil_lessons()
                for text in lessons:
                    self._lesson_memory.add(text, self._dataset_key,
                                            run_id=run_id, source="llm")
            self._lesson_memory.save()
        except Exception as e:
            log.warning("Lesson derivation failed: %s", e)

    def _distil_lessons(self) -> list[str]:
        """LLM distils 1-3 procedural lessons from the run (non-fast only)."""
        try:
            steps = "\n".join(
                f"[{i}] {e.get('title','')}: "
                f"{((e.get('output') or e.get('error') or '')[:200]).strip()}"
                for i, e in enumerate(self._trace, 1))
            prompt = _LESSON_PROMPT.format(goal=self.cfg.goal, steps=steps)
            raw = self._llm.generate([{"role": "user", "content": prompt}]).content
            out = []
            for line in raw.splitlines():
                t = line.strip().lstrip("-*0123456789. ").strip()
                if len(t) > 15 and not t.lower().startswith(("here", "lesson")):
                    out.append(t)
            return out[:3]
        except Exception:
            return []

    def _shutdown_executor(self) -> None:
        """Stop the persistent kernel if one is in use (no-op for sandbox)."""
        shutdown = getattr(self._executor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _make_plan(self) -> list[dict]:
        console.print(Panel("[bold]Planning...[/bold]", border_style="cyan"))
        ctx_parts = []
        if self.cfg.data:
            ctx_parts.append(f"Data file: {self.cfg.data}")
        if self._executor._df is not None:
            df = self._executor._df
            ctx_parts.append(
                f"DataFrame loaded: {df.shape[0]:,} rows x {df.shape[1]} cols\n"
                f"Columns: {list(df.columns[:20])}"
            )
        context = "\n".join(ctx_parts) or "No data file provided (analysis from scratch)."

        # Long-term memory: historical baselines + analogous past episodes, so
        # the plan benefits from accumulated experience with this data.
        baselines = self._baseline_block()
        recall = self._recall_block()
        if baselines:
            context += "\n\n" + baselines
        if recall:
            context += "\n\n" + recall

        prompt = _PLAN_PROMPT.format(
            goal=self.cfg.goal,
            context=context,
            max_steps=self.cfg.max_steps,
        )
        resp = self._llm.generate([{"role": "user", "content": prompt}])
        raw = resp.content.strip()

        # Strip markdown fences if model ignores instructions
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE)

        try:
            plan = json.loads(raw)
            assert isinstance(plan, list) and plan
        except Exception:
            # Fallback: parse numbered lines
            plan = [
                {"title": line.strip().lstrip("0123456789. "), "description": ""}
                for line in raw.splitlines()
                if line.strip() and line.strip()[0].isdigit()
            ] or [{"title": "Execute goal", "description": self.cfg.goal}]

        plan = plan[: self.cfg.max_steps]
        for step in plan:
            console.print(f"  [dim]- {step.get('title','')}[/dim]")
        return plan

    def _revise_plan(self, plan: list[dict], failed_idx: int,
                     entry: dict, prev_outputs: list[str]) -> list[dict] | None:
        """Revise the REMAINING steps after a whole-approach failure.

        Keeps completed/failed steps (indices <= failed_idx) untouched so
        their checkpoints stay valid, and replaces plan[failed_idx+1:] with
        LLM-revised steps informed by the failure. Returns the new full plan,
        or None if revision was not useful.
        """
        done = plan[: failed_idx + 1]
        remaining = plan[failed_idx + 1:]
        console.print("  [magenta]Revising downstream plan in light of the failure...[/magenta]")
        try:
            prompt = _REVISE_PROMPT.format(
                goal=self.cfg.goal,
                failed_step=plan[failed_idx].get("title", ""),
                failed_error=(entry.get("error", "") or "")[:600],
                done="\n".join(f"  - {s.get('title','')}" for s in done) or "(none)",
                remaining="\n".join(f"  - {s.get('title','')}" for s in remaining) or "(none)",
                context="\n".join(prev_outputs[-5:]) or "(none)",
                max_steps=self.cfg.max_steps - len(done),
            )
            raw = _strip_fences(self._llm.generate(
                [{"role": "user", "content": prompt}]).content)
            new_steps = json.loads(raw)
            assert isinstance(new_steps, list) and new_steps
            assert all(isinstance(s, dict) and s.get("title") for s in new_steps)
        except Exception as e:
            console.print(f"  [dim]Plan revision skipped: {e}[/dim]")
            return None
        new_plan = done + new_steps[: self.cfg.max_steps - len(done)]
        console.print(f"  [magenta]Plan revised: {len(remaining)} -> "
                      f"{len(new_steps)} downstream steps.[/magenta]")
        return new_plan

    # Deterministic adaptation signals: (metric-name-substring, predicate, label).
    # The trigger is a COMPUTED fact, so adaptation can never fire on a
    # hallucinated prose finding.
    _SIGNAL_RULES = [
        ("missing", lambda v: v >= 0.90,
         "a column/field is >=90% missing (analysis depending on it is invalid)"),
        ("null",    lambda v: v >= 0.90, "a field is >=90% null"),
        ("drift",   lambda v: v >= 0.5,  "significant distribution drift detected"),
        ("dup",     lambda v: v >= 0.30, "high duplicate rate (>=30%)"),
        ("outlier_rate", lambda v: v >= 0.20, "very high outlier rate (>=20%)"),
        ("corr",    lambda v: abs(v) >= 0.95, "near-perfect collinearity (|r|>=0.95)"),
        ("imbalance", lambda v: v >= 0.98, "extreme class imbalance (>=98%)"),
    ]

    def _detect_signal(self, metrics: dict) -> str | None:
        """Return a human-readable signal string if a computed metric crosses a
        hard threshold OR deviates from its LEARNED historical baseline. Purely
        deterministic -- no LLM. The learned-baseline path is the agent's
        instinct firing: a value that's abnormal *for this data* triggers
        adaptation even if it doesn't cross a generic threshold."""
        if not metrics:
            return None
        for name, val in metrics.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            lname = str(name).lower()
            # 1) hardcoded domain thresholds
            for key, pred, label in self._SIGNAL_RULES:
                if key in lname:
                    try:
                        if pred(v):
                            return f"{name}={val} -> {label}"
                    except Exception:
                        continue
            # 2) learned baseline anomaly (instinct)
            if self._fact_memory is not None:
                try:
                    chk = self._fact_memory.check_value(self._dataset_key, name, v)
                except Exception:
                    chk = None
                if chk and chk.get("is_anomaly"):
                    return (f"{name}={val} is anomalous vs historical baseline "
                            f"{chk['baseline']:.4g} (z={chk['zscore']:.1f}, "
                            f"{chk['n']} prior runs) -> investigate the deviation")
        return None

    def _adapt_plan(self, plan: list[dict], current_idx: int,
                    signal: str, prev_outputs: list[str]) -> list[dict] | None:
        """Given a deterministic discovery signal, let the LLM choose a CLOSED
        action on the remaining plan: insert a follow-up, skip a now-irrelevant
        step, or proceed. Completed steps (<= current_idx) are never touched.
        """
        done = plan[: current_idx + 1]
        remaining = plan[current_idx + 1:]
        if not remaining:
            return None
        console.print(f"  [magenta]Discovery signal: {signal}[/magenta]")
        try:
            prompt = _ADAPT_PROMPT.format(
                goal=self.cfg.goal,
                signal=signal,
                remaining="\n".join(f"  {j+1}. {s.get('title','')}"
                                    for j, s in enumerate(remaining)),
                context="\n".join(prev_outputs[-4:]) or "(none)",
            )
            raw = _strip_fences(self._llm.generate(
                [{"role": "user", "content": prompt}]).content).strip()
            data = json.loads(raw)
            action = str(data.get("action", "PROCEED")).upper()
        except Exception as e:
            console.print(f"  [dim]Adaptation skipped: {e}[/dim]")
            return None

        if action == "INSERT" and isinstance(data.get("step"), dict) \
                and data["step"].get("title"):
            new_step = {"title": data["step"]["title"][:60],
                        "description": data["step"].get("description", "")}
            new_plan = done + [new_step] + remaining
            console.print(f"  [magenta]Adapted: inserted '{new_step['title']}'.[/magenta]")
            return new_plan[: self.cfg.max_steps]
        if action == "SKIP" and isinstance(data.get("skip_index"), int):
            si = data["skip_index"] - 1   # 1-based within `remaining`
            if 0 <= si < len(remaining):
                skipped = remaining.pop(si)
                console.print(f"  [magenta]Adapted: skipped now-irrelevant "
                              f"step '{skipped.get('title','')}'.[/magenta]")
                return done + remaining
        return None   # PROCEED or unparseable -> no change

    def _execute_step(
        self, step_num: int, step: dict, prev_outputs: list[str]
    ) -> dict:
        available_vars = "df (DataFrame)" if self._executor._df is not None else "(no df)"
        safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", step.get("title", "step"))[:30]

        code = self._gen_code(step_num, step, available_vars, prev_outputs, safe_title)
        result = self._executor.execute(code)
        retries = 0
        # Remember every failed (code, error) pair so the LLM stops repeating
        # the same wrong approach across retries.
        prior_attempts: list[tuple[str, str]] = []

        while not result.success and retries < self.cfg.max_retries:
            retries += 1
            console.print(f"  [yellow]Attempt {retries}/{self.cfg.max_retries} -- fixing...[/yellow]")
            if self.cfg.verbose:
                console.print(f"  [dim red]{result.error[:300]}[/dim red]")
            prior_attempts.append((code, result.error))
            # Probe live runtime state (columns / shape / axes) when the
            # error pattern suggests the LLM needs concrete facts to fix it.
            df_context = self._probe_df_state(result.error)
            code = self._gen_fix(
                code, result.error, step, available_vars,
                df_context, prior_attempts,
            )
            result = self._executor.execute(code)

        # Last-resort safe fallback: a minimal describe/head dump so the step
        # produces *something* for downstream context instead of empty output.
        used_fallback = False
        if not result.success:
            console.print("  [yellow]All retries exhausted -- running safe fallback.[/yellow]")
            fallback = self._safe_fallback_code(step)
            fb_result = self._executor.execute(fallback)
            if fb_result.success:
                code, result = fallback, fb_result
                used_fallback = True
                console.print("  [green]Safe fallback succeeded (degraded output).[/green]")

        # Post-filtering (DatawiseAgent's cascade-killer): once a step has
        # self-healed, replace the messy debugged code with a clean canonical
        # version. Costs an extra LLM call + re-execution, so it is skipped in
        # fast mode (the working code is already correct, just possibly messy).
        if result.success and retries > 0 and not self.cfg.fast:
            code, result = self._post_filter(code, result, step)

        # After all retries -- check if tracked file is the culprit
        if not result.success:
            self._maybe_propose_patch(code, result.error, step)

        entry: dict = {
            "step": step_num,
            "title": step.get("title", ""),
            "description": step.get("description", ""),
            "code": code,
            "success": result.success,
            "degraded": used_fallback,   # True => only the safe fallback ran
            "output": result.output,
            "error": result.error,
            "retries": retries,
            "execution_time_ms": result.execution_time_ms,
            "artefacts": result.artefacts,
            "metrics": result.metrics,   # computed facts (structured grounding)
        }

        # Save the final code to a named script in the run dir
        script_path = self._run_dir / f"step_{step_num:02d}_{safe_title}.py"
        script_path.write_text(code, encoding="utf-8")

        # Tell the file index about everything created in this step
        if self._file_index:
            self._file_index.register(
                script_path, "script",
                f"Step {step_num}: {step.get('title','')}"
            )
            for artefact in result.artefacts:
                full = self._executor._scratch / artefact
                self._file_index.register(
                    full, "output",
                    f"output from step {step_num}: {artefact}"
                )

        return entry

    def _librarian_block(self) -> str:
        if self._file_index:
            return self._file_index.prompt_block()
        return ""

    def _memory_block(self) -> str:
        """Cross-step memory (findings, decisions, completed steps) for prompts.

        Cached because it only changes once per step (when a finding/checkpoint
        is added); recomputing it re-reads messages.jsonl on every LLM call.
        """
        if getattr(self, "_memory_cache", None) is not None:
            return self._memory_cache
        block = self._compressor.memory_block() if self._compressor else ""
        self._memory_cache = block
        return block

    def _baseline_block(self) -> str:
        """Historical baselines for THIS dataset = long-term instinct.

        Injected into code-gen so the agent knows what's normal and can flag
        anomalies (e.g. "fraud_rate historically 0.024 +/- 0.003; this run's
        0.041 is unusual"). Cached for the run (memory only changes at finalize).
        """
        if self._baseline_cache is not None:
            return self._baseline_cache
        block = ""
        # Recall spans the shared (corroborated) + private tiers via the
        # Librarian; identical to private-only when the shared tier is empty.
        if self._librarian is not None:
            try:
                block = self._librarian.recall(self._dataset_key).instinct_block
            except Exception:
                block = ""
        elif self._fact_memory is not None:
            try:
                block = self._fact_memory.baseline_block(self._dataset_key)
            except Exception:
                block = ""
        self._baseline_cache = block
        return block

    def _lessons_block(self) -> str:
        """Procedural lessons learned about this data, injected at code-gen."""
        if self._lessons_cache is not None:
            return self._lessons_cache
        block = ""
        if self._librarian is not None:
            try:
                block = self._librarian.recall(self._dataset_key).lessons_block
            except Exception:
                block = ""
        elif self._lesson_memory is not None:
            try:
                block = self._lesson_memory.block(self._dataset_key)
            except Exception:
                block = ""
        self._lessons_cache = block
        return block

    def _vmem(self):
        """Lazily build the vector memory (embedder import can be heavy)."""
        if self._vector_inited:
            return self._vector_memory
        self._vector_inited = True
        try:
            if self._librarian is not None:
                # Owner-scoped private-tier vectors (unified with the Librarian).
                self._vector_memory = self._librarian.vectors()
            else:
                from ledgerhound.memory.longitudinal import LongitudinalMemory
                self._vector_memory = LongitudinalMemory(self._mem_root / "vectors")
        except Exception as e:
            log.warning("Vector memory unavailable: %s", e)
            self._vector_memory = None
        return self._vector_memory

    def _recall_block(self) -> str:
        """Recall analogous past findings for this goal/dataset (vector search).
        Injected at planning so the agent benefits from prior episodes."""
        vm = self._vmem()
        if vm is None:
            return ""
        try:
            from ledgerhound.memory.longitudinal import MemorySource
            hits = vm.recall(self.cfg.goal, namespace="text", k=6,
                             source=MemorySource.FINDING)
        except Exception:
            return ""
        hits = [h for h in hits if h.record.text]
        if not hits:
            return ""
        lines = ["=== PAST EPISODES (institutional memory) ==="]
        for h in hits:
            ds = f" [{h.record.dataset}]" if h.record.dataset else ""
            when = (h.record.as_of or "")[:10]
            lines.append(f"  ({when}{ds}) {h.record.text[:180]}")
        lines.append("Use relevant prior experience; note if this run "
                     "confirms or contradicts it.")
        lines.append("=== END PAST EPISODES ===")
        return "\n".join(lines)

    def _remember_findings(self) -> int:
        """Store this run's findings as text memories for future recall."""
        vm = self._vmem()
        if vm is None or not self._session:
            return 0
        try:
            from ledgerhound.memory.longitudinal import MemorySource
            msgs = self._session.load_messages()
            findings = [m for m in msgs if m.get("tag") == "finding"]
            n = 0
            for m in findings:
                txt = m.get("content", "").strip()
                if txt:
                    vm.remember(source=MemorySource.FINDING, text=txt,
                                dataset=self._dataset_key)
                    n += 1
            if n:
                vm.save()
            return n
        except Exception as e:
            log.warning("Could not persist findings to vector memory: %s", e)
            return 0

    # step-text keyword -> knowledge tags for retrieval
    _KB_TAG_MAP: dict[str, list[str]] = {
        r"fraud|risk|suspicious|mule|launder": ["fraud_vector", "mule", "account_takeover"],
        r"amount|spend|value|round|high.?value": ["amount", "high_value", "velocity"],
        r"time|hour|night|temporal|trend|drift|month|week": ["temporal", "seasonality"],
        r"device|sim|fingerprint": ["device", "fraud_ring", "sim_swap"],
        r"merchant|mcc|p2m|shop": ["mcc", "merchant"],
        r"collect|request|pull": ["collect", "fraud_vector"],
        r"velocity|burst|rate|speed": ["velocity", "mule"],
        r"decline|fail|error|reject|response.?code": ["decline", "data_quality"],
        r"payee|incoming|receive": ["payee", "fraud_signal"],
        r"payer|outgoing|send": ["payer", "fraud_signal"],
        r"graph|network|ring|chain": ["graph", "mule", "fraud_ring"],
        r"missing|null|empty|quality": ["data_quality", "empty_columns"],
        r"sample|psi|stratif|certif": ["sampling", "psi", "descriptive"],
        r"correlat|depend|feature": ["fraud_signal", "amount"],
    }

    def _knowledge_block(self, step: dict) -> str:
        """Retrieve domain heuristics relevant to this step and format them."""
        if not self._kb:
            return ""
        text = f"{step.get('title','')} {step.get('description','')} {self.cfg.goal}".lower()
        tags: list[str] = []
        for pat, tg in self._KB_TAG_MAP.items():
            if re.search(pat, text):
                tags.extend(tg)
        cols = []
        df = getattr(self._executor, "_df", None)
        if df is not None:
            try:
                cols = [c for c in list(df.columns)[:80]]
            except Exception:
                cols = []
        try:
            items = self._kb.retrieve(topic_tags=list(set(tags)), columns=cols,
                                      max_items=5, dataset_columns=set(cols))
            block = self._kb.format_for_prompt(items)
        except Exception:
            return ""
        if not block:
            return ""
        return "=== DOMAIN KNOWLEDGE (expert heuristics) ===\n" + block + \
               "\n=== END DOMAIN KNOWLEDGE ==="

    def _gen_code(
        self,
        step_num: int,
        step: dict,
        available_vars: str,
        prev_outputs: list[str],
        safe_title: str,
    ) -> str:
        prompt = _CODE_PROMPT.format(
            step_num=step_num,
            title=step.get("title", ""),
            description=step.get("description", ""),
            available_vars=available_vars,
            df_engine=getattr(self, "df_engine", "pandas"),
            safe_title=safe_title,
            prev_outputs="\n".join(prev_outputs[-3:]) or "(none)",
            knowledge=self._knowledge_block(step) or "(no domain heuristics matched)",
            baselines=self._baseline_block() or "(no historical baselines yet)",
            lessons=self._lessons_block() or "(no lessons learned yet)",
            memory=self._memory_block() or "(no prior findings yet)",
            librarian_block=self._librarian_block(),
        )
        resp = self._llm.generate([{"role": "user", "content": prompt}])
        return _strip_fences(resp.content)

    def _gen_fix(
        self, code: str, error: str, step: dict, available_vars: str,
        df_context: str = "", prior_attempts: list[tuple[str, str]] | None = None,
    ) -> str:
        prior_block = self._format_prior_attempts(prior_attempts or [])
        prompt = _FIX_PROMPT.format(
            code=code,
            error=error,
            title=step.get("title", ""),
            available_vars=available_vars,
            df_context=df_context or "(not available)",
            prior_attempts=prior_block,
            memory=self._memory_block() or "(no prior findings yet)",
            librarian_block=self._librarian_block(),
        )
        resp = self._llm.generate([{"role": "user", "content": prompt}])
        return _strip_fences(resp.content)

    @staticmethod
    def _format_prior_attempts(prior: list[tuple[str, str]]) -> str:
        """Render the past failed (code, error) pairs into a compact block.

        Only the last error line and the code's first/last few lines are
        kept so the prompt stays bounded even after 4-5 retries.
        """
        if not prior:
            return "(none -- this is the first fix attempt)"
        parts = []
        for i, (c, e) in enumerate(prior, 1):
            err_last = e.strip().splitlines()[-1] if e.strip() else "(no error text)"
            code_lines = c.strip().splitlines()
            head = "\n".join(code_lines[:8])
            tail = "\n".join(code_lines[-4:]) if len(code_lines) > 12 else ""
            snippet = head + ("\n  ...\n" + tail if tail else "")
            parts.append(f"-- Attempt {i} failed with: {err_last}\n{snippet}")
        return "\n\n".join(parts)

    # Error tokens that mean the LLM almost certainly needs to *see* runtime
    # state before guessing again. Grouped by what to probe.
    _NAME_ERR_TOKENS  = ("KeyError", "AttributeError", "ColumnNotFoundError",
                         "column", "not in", "no attribute", "is not defined")
    _SHAPE_ERR_TOKENS = ("IndexError", "out of bounds", "out of range",
                         "tuple index", "too many indices", "shape mismatch")
    _TYPE_ERR_TOKENS  = ("TypeError", "ValueError", "cannot convert",
                         "could not convert", "unsupported operand")

    def _probe_df_state(self, error: str) -> str:
        """Run a diagnostic that matches the error class.

        - Name errors  -> columns / dtypes / shape
        - Shape errors -> same + first row sample (helps subplot miscounts)
        - Type errors  -> dtypes only (cheap)

        Returns empty string if df is not loaded or error is unrelated.
        """
        if self._executor._df is None:
            return ""

        is_name  = any(t in error for t in self._NAME_ERR_TOKENS)
        is_shape = any(t in error for t in self._SHAPE_ERR_TOKENS)
        is_type  = any(t in error for t in self._TYPE_ERR_TOKENS)
        if not (is_name or is_shape or is_type):
            return ""

        # Build the smallest probe that covers the failure mode.
        # `_pf` is a pandas view of df.head() so .to_dict() works on cuDF too.
        probe_lines = [
            "cols = list(df.columns)",
            "print('columns:', cols)",
            "print('shape:', df.shape)",
            "_h = df.head(1)",
            "_pf = _h.to_pandas() if hasattr(_h, 'to_pandas') else _h",
        ]
        if is_name or is_shape:
            probe_lines.append(
                "print('dtypes:', ' | '.join(f'{c} {df[c].dtype}' for c in cols[:30]))"
            )
        if is_shape:
            probe_lines.append("print('numeric_cols:', list(df.select_dtypes(include='number').columns))")
            probe_lines.append("print('first_row:', _pf.to_dict('records'))")
        result = self._executor.execute("\n".join(probe_lines))
        if result.success:
            console.print(f"  [dim cyan]Probe: {result.output[:200]}[/dim cyan]")
            return result.output
        return ""

    def _safe_fallback_code(self, step: dict) -> str:
        """A guaranteed-safe minimal step body used after all retries fail.

        Produces *something* the next step can consume instead of an empty
        slot in the trace. Adapts based on whether a df is loaded.
        """
        title = step.get("title", "").replace('"', "'")
        if self._executor._df is None:
            return textwrap.dedent(f'''\
                # Safe fallback for: {title}
                print("Step skipped (no df). Recording placeholder.")
                result = {{"status": "skipped", "step": "{title}"}}
            ''')
        return textwrap.dedent(f'''\
            # Safe fallback for: {title}
            # All LLM attempts failed -- emit a minimal summary so downstream
            # steps still have context to work with. Works on pandas AND cuDF
            # by converting a small head() sample to pandas before .to_dict().
            _cols = list(df.columns)[:50]
            _sample = df.head(3)
            if hasattr(_sample, "to_pandas"):      # cuDF -> pandas
                _sample = _sample.to_pandas()
            try:
                _desc = df[_cols].describe()
                if hasattr(_desc, "to_pandas"):
                    _desc = _desc.to_pandas()
                _desc = _desc.head(8).to_dict()
            except Exception as _e:
                _desc = {{"describe_failed": str(_e)}}
            summary = {{
                "shape":   list(df.shape),
                "columns": _cols,
                "dtypes":  {{c: str(df[c].dtype) for c in _cols}},
                "head":    _sample.to_dict("records"),
                "describe": _desc,
            }}
            (scratch / "fallback_summary.json").write_text(__import__("json").dumps(summary, default=str, indent=2))
            print("Fallback summary written to scratch/fallback_summary.json")
            result = summary
        ''')

    def _post_filter(self, code: str, result, step: dict):
        """Regenerate clean canonical code from working-but-messy debugged code.

        Returns (code, result). If the cleaned version fails or changes the
        output materially, we KEEP the original working code (safe default) --
        post-filtering must never break a step that already passed.
        """
        try:
            prompt = _POSTFILTER_PROMPT.format(
                title=step.get("title", ""),
                code=code,
                output=(result.output or "")[:500],
            )
            cleaned = _strip_fences(self._llm.generate(
                [{"role": "user", "content": prompt}]).content)
        except Exception:
            return code, result
        if not cleaned or cleaned.strip() == code.strip():
            return code, result
        clean_result = self._executor.execute(cleaned)
        # Accept the clean version only if it still succeeds.
        if clean_result.success:
            console.print("  [dim green]Post-filtered to clean canonical code.[/dim green]")
            return cleaned, clean_result
        return code, result

    def _record_finding(self, step_num: int, step: dict, output: str,
                        metrics: dict | None = None) -> None:
        """Store a one-line factual finding in sacred memory (never compressed,
        injected into every future code-gen prompt via the memory block).

        Structured grounding: if the step emitted a `metrics` dict, the finding
        is built directly from those computed facts (grounded by construction,
        no LLM, no hallucination). Otherwise falls back to rule/LLM extraction.
        """
        if not self._session:
            return
        # Prefer grounded metrics -- these are real computed values from df.
        if metrics:
            facts = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:8])
            if facts:
                self._session.append_message(
                    "assistant", f"[Step {step_num}] {facts}", tag="finding")
                self._memory_cache = None
                return
        if not output.strip():
            return
        if self.cfg.fast:
            text = self._rule_finding(output)
        else:
            try:
                prompt = _FINDING_PROMPT.format(
                    title=step.get("title", ""), output=output[:800])
                text = self._llm.generate(
                    [{"role": "user", "content": prompt}]).content.strip()
            except Exception:
                text = self._rule_finding(output)
        if text and text.upper() != "NONE":
            self._session.append_message(
                "assistant", f"[Step {step_num}] {text}", tag="finding")
            self._memory_cache = None   # invalidate: new finding to surface

    _FINDING_NOISE = ("no output", "saved", "loading", "loaded", "analyzing",
                      "processing", "writing", "exporting", "executed",
                      "/home/", "ledgerhound_workspace", ".html", ".parquet")

    @classmethod
    def _rule_finding(cls, output: str) -> str:
        """Cheap finding: the most informative numeric line, skipping obvious
        log/progress noise (paths, 'Loading...', '[Executed - no output]')."""
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        clean = [l for l in lines
                 if not any(tok in l.lower() for tok in cls._FINDING_NOISE)]
        numeric = [l for l in clean if any(ch.isdigit() for ch in l)]
        pick = numeric[0] if numeric else (clean[0] if clean else "")
        return pick[:200]

    def _maybe_propose_patch(self, code: str, error: str, step: dict) -> None:
        """Check if a tracked file is buggy; if so, drop a patch candidate."""
        repo_root = Path(__file__).resolve().parents[2]
        # Look for file references in the traceback
        tracked_candidates = []
        for line in error.splitlines():
            m = re.search(r'File "([^"]+\.py)"', line)
            if m:
                p = Path(m.group(1)).resolve()
                try:
                    rel = p.relative_to(repo_root)
                    if rel.parts[0] in ("ledgerhound", "scripts"):
                        tracked_candidates.append(p)
                except ValueError:
                    pass

        if not tracked_candidates:
            return

        target = tracked_candidates[-1]
        if not target.exists():
            return

        file_content = target.read_text(encoding="utf-8", errors="replace")
        prompt = _PATCH_PROMPT.format(
            error=error,
            code=code,
            tracked_file=str(target.relative_to(repo_root)),
            file_content=file_content[:4000],
        )
        resp = self._llm.generate([{"role": "user", "content": prompt}])
        text = resp.content.strip()

        if not text.startswith("PATCH_NEEDED: yes"):
            return

        reason_m = re.search(r"REASON:\s*(.+)", text)
        reason = reason_m.group(1).strip() if reason_m else "LLM-diagnosed bug"
        fixed_m = re.search(r"FIXED_CODE:\n(.*)", text, re.DOTALL)
        if not fixed_m:
            return
        fixed_code = fixed_m.group(1).strip()

        from ledgerhound.auto.patch import PatchStore
        store = PatchStore(self._patch_dir)
        candidate = store.propose(
            target_file=str(target.relative_to(repo_root)),
            original_code=file_content,
            patched_code=fixed_code,
            reason=reason,
            repro_test=code,
        )
        console.print(
            f"  [yellow]Patch proposed -> {candidate.id}[/yellow]\n"
            f"  [dim]Review: ledgerhound auto review-patches --diff {candidate.id}[/dim]"
        )

    def _finalize(self) -> Path:
        from ledgerhound.auto.report import (build_html, build_notebook,
                                          audit_claims, _build_summary,
                                          _build_key_findings)
        from ledgerhound.eval.scorecard import (build_run_scorecard,
                                             write_run_scorecard, scorecard_summary)
        console.print(Rule("[bold green]Finalizing -- building narrated report[/bold green]"))
        # Pass the LLM so the report/notebook get a progressive narrative
        # (what each step found and why the next step needs it).
        html_path = build_html(self._run_dir, self.cfg.goal, self._trace,
                                llm=self._llm, session=self._session)
        nb_path   = build_notebook(self._run_dir, self.cfg.goal, self._trace,
                                   llm=self._llm, session=self._session)

        # Environment-aware audit + per-run scorecard. Recompute the summary/
        # findings here (cheap; cached LLM output is reused if fast) so the
        # scorecard reflects the same claims shown in the report.
        summary = _build_summary(self.cfg.goal, self._trace, self._llm)
        findings = _build_key_findings(self.cfg.goal, self._trace,
                                       self._llm, self._session)
        audit = audit_claims(summary, findings, self._trace)
        scorecard = build_run_scorecard(self._trace, audit)
        sc_path = write_run_scorecard(self._run_dir, scorecard)
        self._emit_eval_record(scorecard)

        # Consolidate this run's grounded facts into long-term Data Memory.
        # Only numeric, computed facts enter -- re-verifiable, never prose.
        drift_notes = self._consolidate_memory()
        # Persist this run's findings to vector memory for future recall.
        self._remember_findings()
        # Derive procedural lessons from this run (deterministic + optional LLM).
        self._derive_lessons(drift_notes)

        # Clear, copy-pasteable "where everything is" panel.
        outputs_dir = self._run_dir / "outputs"
        n_charts = len(list(outputs_dir.glob("*.png"))) if outputs_dir.exists() else 0
        console.print(Panel(
            f"[bold]Deliverables[/bold]\n"
            f"  Report (HTML, plots inline) : {html_path}\n"
            f"  Notebook (narrated)         : {nb_path}\n"
            f"  Scorecard (metrics)         : {sc_path}\n"
            f"  Charts ({n_charts})              : {outputs_dir}\n"
            f"  Step scripts                : {self._run_dir}\n"
            f"  Trace log                   : {self._run_dir / 'trace.jsonl'}\n\n"
            f"[bold]Run scorecard[/bold]\n  {scorecard_summary(scorecard)}\n\n"
            f"[dim]Open the HTML in a browser, or the notebook in Jupyter.[/dim]",
            title="Where to find your results", border_style="green",
        ))

        # Register final deliverables with the file index so they appear in
        # future KNOWN FILES blocks and can be found by path lookups.
        if self._file_index is not None:
            self._file_index.register(html_path, "report", "final HTML analysis report (plots inline)")
            self._file_index.register(nb_path,   "report", "final narrated Jupyter notebook")
            self._file_index.register(sc_path,   "trace",  "per-run trajectory scorecard")
            trace_path = self._run_dir / "trace.jsonl"
            if trace_path.exists():
                self._file_index.register(trace_path, "trace", "run execution trace log")
        return self._run_dir

    def _emit_eval_record(self, scorecard: dict):
        """Write an eval_record/v1 for this run (local always, MinIO if set).

        Eval-first flywheel: one record per run. Never raises -- a telemetry
        failure must not break a finished analysis.
        """
        try:
            from ledgerhound.eval.eval_record import EvalRecord

            log = self._llm.call_log if self._llm is not None else []
            tokens = sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
                         for c in log)
            milestones = [
                {"name": e.get("title", f"step_{i+1}"),
                 "reached": bool(e.get("success")),
                 "detail": "degraded" if e.get("degraded") else ""}
                for i, e in enumerate(self._trace)
            ]
            rec = EvalRecord.from_run(
                run_id=self._run_id,
                dataset=str(self.cfg.data or self.cfg.goal[:40]),
                scorecard=scorecard,
                tokens=tokens,
                llm_calls=len(log),
                milestones=milestones,
            )
            local = rec.write_local(self._run_dir / "eval_record.jsonl")

            # Optional MinIO publish (owner-scoped private tier).
            from ledgerhound.config import get_str
            cfg_path = get_str("integrations", "minio_config", env="LEDGERHOUND_MINIO_CONFIG")
            if cfg_path and Path(cfg_path).exists():
                try:
                    from ledgerhound.io import MinIOStore
                    store = MinIOStore.from_yaml(cfg_path)
                    if store.config.endpoint:
                        rec.write_minio(store)
                except Exception as e:  # never break a run on telemetry
                    log.warning("eval_record MinIO publish skipped: %s", e)
            if self._file_index is not None:
                self._file_index.register(local, "trace", "eval_record/v1 (flywheel)")
            return local
        except Exception as e:
            log.warning("eval_record emission skipped: %s", e)
            return None

    def _flush_trace(self) -> None:
        trace_path = self._run_dir / "trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as f:
            entry = self._trace[-1]
            f.write(json.dumps(entry) + "\n")

    def _log_trace(self, entry: dict) -> None:
        trace_path = self._run_dir / "trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _install_exit_handler(self) -> None:
        """Catch Ctrl-C / SIGTERM -- pause cleanly instead of crashing."""
        def _handler(sig, frame):
            self._paused = True
            console.print(
                "\n[yellow bold]Ledgerhound pausing... saving checkpoint.[/yellow bold]"
            )
            if self._session:
                self._session.status = "paused"
                self._session.save()
                console.print(
                    f"[dim]Session saved: {self._session.id}\n"
                    f"Resume with: ledgerhound start  (select this session)[/dim]"
                )
        try:
            signal.signal(signal.SIGINT,  _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            pass  # signal can only be set in main thread; skip in workers

    def _print_header(self) -> None:
        resume_note = ""
        if self._session and self._session.last_completed_step > 0:
            resume_note = (
                f"\nResuming from step {self._session.last_completed_step + 1}"
            )
        console.print(Panel(
            f"[bold]Ledgerhound Auto[/bold]\n"
            f"Goal:    {self.cfg.goal[:120]}\n"
            f"Run ID:  {self._run_id}\n"
            f"Scratch: {self._run_dir}{resume_note}",
            border_style="cyan",
        ))

    def _build_llm(self):
        from ledgerhound.llm.client import LLMClient, LLMConfig
        if self.cfg.llm_config:
            return LLMClient.from_config(self.cfg.llm_config)
        cfg = LLMConfig(
            base_url=self.cfg.base_url or "http://localhost:8000/v1",
            model=self.cfg.model or "default",
            api_key=self.cfg.api_key or "EMPTY",
        )
        return LLMClient(cfg)

    def _build_executor(self):
        from ledgerhound.auto.sandbox import ScratchExecutor
        from ledgerhound.runtime import GPUPlanner

        # Build (or restore) the GPU allocation plan.
        if self._session and getattr(self._session, "gpu_plan", None):
            self.gpu = GPUPlanner.from_dict(self._session.gpu_plan)
        else:
            self.gpu = GPUPlanner(llm_config=self.cfg.llm_config or "")
            if self._session:
                self._session.gpu_plan = self.gpu.to_dict()
                self._session.save()
        console.print(Panel(self.gpu.summary(), title="GPU Plan", border_style="magenta"))

        # Decide engine from the GPU plan (cuDF only if a GPU is assigned).
        self.df_engine = "pandas"
        if self.cfg.data and self.gpu.use_cudf():
            try:
                import cudf  # noqa: F401
                self.gpu.apply_env("cudf")
                self.df_engine = "cudf"
            except Exception as e:
                log.warning("cuDF unavailable (%s) -- using pandas", e)

        scratch = self._run_dir / "outputs"
        scratch.mkdir(parents=True, exist_ok=True)

        # Preferred backend: a real persistent IPython kernel (robust + fast --
        # df loaded once, full Python, no emulation). Falls back to the
        # in-process ScratchExecutor when jupyter_client/ipykernel are absent.
        from ledgerhound.auto.kernel import kernels_available, KernelExecutor
        if kernels_available():
            try:
                ex = KernelExecutor(
                    scratch_dir=scratch,
                    data_path=self.cfg.data if self.cfg.data and Path(self.cfg.data).exists() else None,
                    engine=self.df_engine,
                    timeout=self.cfg.timeout,
                )
                # Kernel may have downgraded cuDF->pandas internally on load failure
                self.df_engine = ex._engine
                console.print(f"  [dim]Engine: {self.df_engine} | Backend: persistent kernel[/dim]")
                return ex
            except Exception as e:
                log.warning("Kernel backend failed to start (%s) -- using in-process sandbox", e)

        # Fallback: in-process sandbox (loads df in this process).
        df = None
        if self.cfg.data:
            p = Path(self.cfg.data)
            if p.exists():
                df, self.df_engine = self._load_dataframe(p)
            else:
                log.warning("Data file not found: %s", p)
        console.print(f"  [dim]Engine: {self.df_engine} | Backend: in-process sandbox[/dim]")
        return ScratchExecutor(scratch_dir=scratch, df=df, timeout=self.cfg.timeout)

    def _load_dataframe(self, p: Path):
        """Load a parquet/csv with cuDF if planner says so, else pandas."""
        if self.gpu.use_cudf():
            self.gpu.apply_env("cudf")
            try:
                import cudf  # type: ignore
                if p.suffix in (".parquet", ".pq"):
                    return cudf.read_parquet(p), "cudf"
                if p.suffix in (".csv", ".tsv"):
                    return cudf.read_csv(p), "cudf"
            except Exception as e:
                log.warning("cuDF load failed (%s) -- falling back to pandas", e)
        import pandas as pd
        if p.suffix in (".parquet", ".pq"):
            return pd.read_parquet(p), "pandas"
        if p.suffix in (".csv", ".tsv"):
            return pd.read_csv(p), "pandas"
        log.warning("Unrecognised data format: %s -- no df loaded", p)
        return None, "pandas"


# -- Helpers ---------------------------------------------------------------- #

def _strip_fences(text: str) -> str:
    """Remove ```python ... ``` fences that some models add despite instructions."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
