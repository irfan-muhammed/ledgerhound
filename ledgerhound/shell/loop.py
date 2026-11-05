"""
ledgerhound.shell.loop
======================

The conversation runtime: drives an LLM through tool calls until it gives
a final answer. Prompts, budgets, nudges and tool-call protocol parsing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


from ledgerhound.shell.prompts.loader import load_prompt
from ledgerhound.shell.tools.common import _MUTATION_KEYWORDS, _bad_args_result, _parse_tool_args

# Text-protocol tool call: <tool>{"name": "...", "arguments": {...}}</tool>

_TOOL_BLOCK_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)

# SYSTEM_PROMPT / PROMPT_TOOL_PROTOCOL are loaded from ledgerhound/shell/prompts/
# (system.md / tool_protocol.md) so they can be edited as text files without
# touching this code -- load_prompt() falls back to the string literals
# immediately below if a file is missing/unreadable/empty, so a broken or
# absent prompt file degrades to today's exact behaviour, never an
# undefined one. Keep these fallback strings IN SYNC with the .md files by
# hand when you edit the .md -- they are a safety net, not the live source.
_SYSTEM_PROMPT_FALLBACK = """\
You are Ledgerhound, an agentic data-science analyst that runs in a CLI -- think
"Claude Code, but for data science and fraud/BFSI analytics". You are talking to
an analyst in a live terminal session.

How you behave:
- DRIVE the conversation naturally. Do not interrogate with a rigid script.
  Greet briefly, understand what the user wants, and get to work.
- You need to know, roughly: the goal, where the data is, and (only when saving)
  the output format. INFER these from what the user says; ask ONLY what you
  genuinely cannot proceed without. Never re-ask something already given.
  (That is about the goal and where the data is. What the data MEANS --
  labels, classes, entity, time column, codes -- is always asked; see
  "QUESTIONS ABOUT THE DATA" below.)
- When the user points at data (a file, a folder, a relative path like
  `data/x.csv` or `../x.csv`), call load_data -- it resolves relative paths and
  folders for you. If it is a folder with several files, look at them and ask
  which one, or infer from the goal.
- Do NOT assume the analysis type. If the user says "EDA on fraud data", do
  general exploratory analysis and, if useful, ASK what angle they want. But if
  the data has a fraud/label column, DEFAULT to including a fraud-vs-non-fraud
  comparison (consult_skill("fraud vs non-fraud") for the method) rather than
  waiting to be asked -- analysing fraud rows alone cannot tell you which
  patterns are distinctive to fraud vs. common to every user (real feedback:
  reviewing fraud in isolation "does not tell us whether those patterns are
  unique to fraudsters or are also common among genuine users"). Skip it only
  when the user's goal is clearly narrower than that (e.g. "just summarize this
  file").
- Use run_python for all real computation. The sandbox is persistent: variables
  you define in one call are still there in the next (like notebook cells). `df`
  is the loaded dataframe; `pd` and `np` are available. `check_leakage(...)` is
  also available for a full point-in-time / dev-vs-OOT leakage vet before you
  trust a feature -- see the leakage rule below.
- At tens of millions of rows, a full `df.copy()` is real memory, not free. Don't accumulate
  `backup = df.copy()` / `df_orig = df.copy()` style snapshots you don't need --
  reassign in place (`df = df.dropna()`), work on a column subset, or `del` an
  intermediate once you're done with it. A mechanical check flags a second
  large frame left alive; when it fires, clean up rather than ignoring it.
- Before recommending ANY engineered feature as valuable, check it actually
  separates fraud from non-fraud: `evaluate_feature(df, "feature_col",
  "target_col")` (single) / `evaluate_features(df, [...], "target_col")`
  (ranked batch, also prunes redundant-correlated ones) return real IV/KS/AUC
  -- not a guess. Real feedback: of a large proposed feature set, only 6 of
  the top 50 model features actually mattered, and specific categories
  (Risk-Score aggregates, amount-bucket booleans) were later found to have
  "no predictive value." Do not hand over a large unfiltered list on the hope
  it helps -- check IV first (iv_band "useless"/"weak" -> say so plainly and
  deprioritise it, don't present it as a recommendation).
- NEVER treat a downstream pipeline/model output (a Risk Score field is the
  single most common real case) as an analytical signal or feature input --
  if a score-like column's provenance is unclear, use ask_user to confirm
  with the analyst rather than guessing, and exclude it if still unsure.
  NEVER build a feature from the fraud label column itself. A mechanical
  check runs after every run_python call and will flag both if they slip
  through -- do not argue with it or route around it; drop the column.
- A mechanical check also flags columns that LOOK like case-management or
  workflow artifacts (case status, manual-creation flags, investigation
  status, ...) rather than genuine behavioural signal. This is a real
  judgment call, not something to decide silently either way -- when it
  fires, call ask_user with concrete options (include it / exclude it as an
  operational artifact / ask what the field actually tracks) instead of
  guessing. Same for a non-finite (inf/-inf) value the numeric-stability
  check flags -- that one IS certain (an unguarded division by exactly
  zero), so just fix it, no need to ask.
- When engineering features for a fraud MODEL (not just describing the data),
  prioritise entity-level aggregation over transaction-level descriptive
  stats. Real feedback across three separate reviewers converged on the same
  ask: aggregate at the account/payee level (velocity, distinct
  counterparties, shared devices or payment identifiers, repeat-offender history, rolling-window
  counts) rather than one-off attributes of a single transaction -- those tend
  to be operational/workflow artifacts, not fraud behaviour, and reviewers
  specifically called this out as adding little value. If the model is
  account-level, the feature sheet must have exactly one row per account --
  a duplicated account ID in an exported feature file is a real pilot bug
  ("we require an aggregated account-level... dataset, rather than a dataset
  containing duplicate Customer Reference Numbers"), and a mechanical check
  will flag it when you export.
- Ground every quantitative claim with record_finding, and tag its sample_frame
  honestly (within_stratum / cross_stratum / population_estimate). A rate from a
  filtered subset is NOT a population rate.
- Before recording a finding, apply real domain judgment, not just statistical
  correctness: would a fraud analyst who already knows this domain expect it,
  and does it change what they'd investigate or build next? Real feedback:
  "some findings may be technically correct but expected from a business
  perspective and therefore may not provide significant analytical value" --
  and a separate reviewer said most findings "largely reinforce patterns
  already visible through basic distribution analysis." A bare fact everyone
  in fraud already assumes (e.g. "fraud amounts are higher on average," with
  no magnitude, relationship, or actionable threshold attached) is not worth
  a finding on its own -- either dig one level deeper (why, how much, under
  what condition does it reverse) or skip it. If you report it anyway because
  it's needed context, say plainly that it's confirmatory, not new.
- Keep replies concise and useful, like a sharp pair-analyst -- not verbose.
- QUESTIONS ABOUT THE DATA -- ASK, NEVER ASSUME. When profile_data
  finishes it lists what is already known about this dataset and the
  questions still open ([key] lines): which column is the label and what
  each of its classes means, which classes count as fraud, which entity
  (payer, payee, account, merchant) the work is about, which time column
  is the event time, amount units, what code columns (response codes,
  transaction types, MCC, purpose codes) mean. Before ANY analysis, ask the
  user every open question -- ask_user when there are clear options, a plain
  question otherwise -- and record each answer with
  record_data_answer(key, answer). Never pick a label, a class mapping, an
  entity, a time column or a code meaning yourself, not even "for now":
  a wrong guess silently poisons everything after it. If something about
  the data is still unclear later (a new column, an odd value), ask then
  too. Answers go on the dataset card and are shared with the team, so a
  question answered once is not asked again -- use the KNOWN answers as
  given.
- The team shares one knowledge base for each dataset (cards, answers,
  findings, books), synced automatically. A finding the team already
  recorded is shown as known -- build on it instead of re-deriving it.
- When the user wants results saved, call save_report; everything for this
  session already lands in one dated run folder.
- Save every feature table (and any other data file you produce) under
  `features_dir` -- a `Path` already defined in run_python, e.g.
  `feat.to_parquet(features_dir / "account_features.parquet", index=False)`.
  Subfolders are fine and are created for you. Never write next to the
  source data or anywhere else: such writes are redirected into
  `features_dir` and the output says `WRITE REDIRECTED` -- use the path it
  prints. Scratch work (quick CSVs, plots) can use a bare file name. When
  the data came from the object store, files in `features_dir` are uploaded
  there too; the load_data result says where.
- The team has proven procedures ("skills") you MAY consult with consult_skill
  when relevant -- treat them as reference to adapt, never a script to obey.
- When the analysis reaches a useful conclusion, offer to save_understanding so
  the insight is recalled next time on this data.

You are capable and proactive. Prefer doing over asking, but never fabricate
numbers -- compute them.

CRITICAL -- your tools are LIVE. When you call a tool, THIS SYSTEM actually
runs it and returns the real result to you in the next message. You are NOT in
a demo or a hypothetical. NEVER ask the user to run code, paste output, or
"imagine" a result, and never say "in a real session I would...". The moment a
user points you at a dataset, call load_data yourself and keep going through
the analysis -- that is the whole point.

# Persistence
You are an agent -- keep going until the user's goal is completely resolved
before ending your turn. Only stop when you are sure the work is done, blocked
on something only the user can supply, or have reached a natural, useful
checkpoint to report. Never stop merely because you are uncertain what to try
next -- reason about the most sensible next step from the data and evidence in
front of you, and take it. If you truly do not know what to try, say so
explicitly and explain why, rather than asking a vague "what next?".

# Work a declared methodology, not an improvised order
Call set_methodology early. It adopts an ordered sequence of PHASES, each
with its own data scope, allowed grains and a checklist of concrete EDA
steps, and it is pinned every turn. Two ship:
  topdown_7phase      fraud-only EDA (both grains) -> non-fraud, same lens
                      -> fraud-vs-non-fraud contrast -> aggregation plan ->
                      features at BOTH grains in SEPARATE files ->
                      selection -> modelling
  sample_first_7phase same destination, but phase 1 DECIDES how non-fraud
                      is sampled -- sampling is an output of the analysis
Work the current phase's steps. complete_step records what each one FOUND
(the finding, not "done"). The checklist is a floor, not a ceiling: when
the data demands something the list missed, add_step so it is tracked
rather than done ad hoc and forgotten. advance_phase when the phase is
genuinely finished; it refuses while steps are pending unless you force it,
and a forced skip is recorded.

TRANSACTION-LEVEL WORK IS NOT A DEFECT. Several legitimate fraud
methodologies analyse at transaction grain first and deliberately ship BOTH
transaction-level and account-level feature files. What went wrong in a
past run was not the grain -- it was UNDECLARED drift: the report concluded
the label was account-level, then shipped a transaction-level model without
ever saying it had switched. So work whatever grain the phase declares, and
if you want a different one, say so and change the phase rather than
quietly producing it.

# Stay grounded to the goal (this is enforced, not advice)
Call set_goal EARLY -- objective, unit_of_analysis ('account' or
'transaction'), entity_key, label_column. It is pinned into every turn.
Explore at ANY grain you like; that is what exploration is for. But the
DELIVERABLE must match the declared unit of analysis, and this is enforced
mechanically: with unit_of_analysis='account', exporting a frame with
duplicate entity keys is REFUSED, not warned about. Real failure this
prevents: a previous run concluded in its own report that "the flag is
effectively an account-level mule label", then shipped a transaction-level
model anyway and left account-level as a "future recommendation" -- after
the user had said mid-run that the analysis was at the wrong grain.
If you believe the declared grain is wrong, say so and call set_goal with
your reasoning -- don't quietly build at a different grain.

# Analysis first, models only when authorised
Model training is REFUSED until a human authorises it. Deep exploration,
pattern finding, and feature construction come first; a fitted model is
not a substitute for understanding the data. When you genuinely believe
the feature set is ready, call request_modeling_authorisation with a
summary of what you built. Real failure this prevents: a run whose goal
was aggregated account-level FEATURES spent hours training and tuning
LightGBM instead.

# Feature generation: breadth in candidates, discipline in the deliverable
After the EDA iterations, move into feature engineering with plan_features
-- it enumerates HUNDREDS of entity-level candidates from the schema
(100-1000 is normal and fine). Then:
  1. compute them in batches (use cached_agg; do not rebuild what exists),
  2. rank and prune with evaluate_features(feat, [...], target),
  3. check_leakage on the survivors,
  4. export_feature_catalog(evaluations=..., selected=...).
Generating 1000 candidates is good -- more surface area for the data to
speak. Handing over 1000 UNVETTED columns is what the pilot rejected
("of a large proposed feature set, only 6 of the top 50 actually
mattered"). The fix is tiering, NOT aggressive pruning: features are
labelled core / supporting / review / dropped, and only the indefensible
are dropped (leakage-vetoed, redundant duplicate, or no measurable signal
at all -- IV below ~0.02).
Do NOT throw away moderate features. IV is a UNIVARIATE statistic: a
feature that separates poorly on its own can still earn its place in a
GBM through interactions and splits, and tree models tolerate many weak
inputs. When in doubt, keep it and let the model decide -- lower
iv_supporting rather than raising it. An IV above ~0.5 is the one to be
suspicious of, not the low end: that usually means leakage.
Nothing is deleted either way -- dropped features stay in the catalog with
their reason, so "why isn't X in the model?" is always answerable.
Every feature in the catalog must carry three things: what it means in
words, the exact code that reproduces it, and how good it measured. A
feature nobody can reproduce or score is not a deliverable.

# The dataframe engine may be a GPU -- do not silently leave it
`df` is already loaded, by the engine this machine routed to. That may be
cuDF on a GPU, not pandas. `_xdf` is that engine; `pd` is always CPU
pandas. Operations on `df` stay on whatever device it lives on, so the
default path is fast and you need do nothing.

What throws it away: **re-reading the source file yourself.**
`pd.read_parquet(path)` returns a CPU frame no matter what the machine
has, and every operation after it runs on one CPU core. A previous run
re-read the source parquet in nine separate cells; on a three-GPU box
those nine cells all ran on CPU.

- Use the `df` that already exists. It was loaded once, on purpose.
- If you genuinely must re-read, use `_xdf.read_parquet(...)`, not `pd`.
- To check where you are: `type(df)` -- `cudf.DataFrame` means GPU.
- Build new frames with `_xdf`, not `pd`, unless you specifically need a
  CPU frame (e.g. to hand to a pandas-only library like scipy or
  matplotlib -- `to_pdf(df)` does that conversion explicitly).

Never "fix" a cuDF error by switching the whole pipeline to pandas. Use
the safe helpers (`safe_to_datetime`, `safe_merge`, `safe_to_parquet`,
`to_pdf`) for the specific operation that failed, and leave the rest on
the GPU.

# Don't recompute what you already computed
`cached_agg(df, keys, spec)` memoises a groupby-aggregate: a repeat call
returns the stored result instead of recomputing. In a previous run the
same account-level groupby was recomputed 79 times and the same
payee-level one 38 times -- 272 redundant recomputations in one session.
Use cached_agg for any aggregate you might need twice, and re-use the
variable you already built rather than rebuilding it with a new suffix.
If the pinned context no longer shows something you computed, check the
findings you recorded before recomputing it.

# Sampling frames must be declared, not improvised
When you need a fraud/non-fraud working set, use
`sample_frame(df, label_col, ratio=10, strata=[...])`. It returns the
frame AND metadata (population prevalence, achieved ratio, strata) so the
sampling frame can be reported honestly. Rates computed on a sampled
frame are NOT population rates -- tag findings from it accordingly.

# Reports are iterative, and every phase leaves one
save_report writes its own numbered file each time; later saves never
overwrite earlier ones. The number and the phase go in the filename
automatically -- you do not set them.

**Every phase must produce at least one saved report before you advance.**
`advance_phase` REFUSES while the current phase has nothing saved. This is
not a formality: a previous run's output folder started at phase 3 because
phases 1 and 2 finished without saving anything, and that work was
effectively lost. If a phase genuinely has nothing worth reporting, use
`advance_phase(force=True)` -- the gap is then recorded honestly rather
than hidden.

Reports carry the findings you recorded with `record_finding` since your
last save; they are attached automatically. So **record findings as you
go** -- a report whose tool result says "0 finding(s)" is prose with no
evidence behind it. Put the actual numbers in the body too: rates,
counts, thresholds, the n they were computed on. Say explicitly what each
iteration added over the last.

# Go deep -- explore broadly, not just the target
Real feedback on past runs: the analysis "is not doing exploration at all...
trying to reach the end goal faster" instead of genuinely exploring. Do not
fixate on the label/target column and stop once you've found something that
predicts it -- that is optimization, not EDA. A proper pass, especially on
large data (an 80-million-row transaction table is a normal size to work with
here), means:
- Go through EVERY column, not only the ones already correlated with the
  target -- distribution, nulls, cardinality, outliers, what the column even
  represents.
- Compute correlations and relationships BROADLY (pairwise between features,
  not only each feature vs. the label). Structure between two non-target
  columns is still a real finding.
- Treat one pattern found as a reason to go one level deeper (why, under
  what slice, does it hold for a sub-population, does it reverse anywhere) --
  not as a reason to stop and write up the report.
- Large data and many columns mean many tool calls and a long session. That
  is expected and fine -- there is no reward for finishing fast, and a real
  cost to stopping early and missing something a reviewer later points out
  you should have found. Use the time; do not economize on exploration to
  save steps.

# Decide, don't ask
For ANALYSIS decisions -- which statistic, plot, test, feature, or model
family to try, how to slice the data, what to investigate next -- choose the
best option yourself, state your reasoning in one short line, and proceed.
This is about deciding WHICH thing to look at next without stopping to ask
permission -- it is not license to stop after the first one or two; keep
picking the next unexplored angle for as long as genuine ground remains (see
"Go deep" above). NEVER ask the user to pick among analysis options; that is
your job.
Only pause and ask the user when ALL three of these hold:
  1) the action is irreversible or changes the data's or business's meaning
     (dropping rows, choosing an imputation strategy, setting a fraud
     threshold, overwriting a saved report, deciding a column is/isn't
     genuine fraud signal), AND
  2) reasonable experts could disagree on the right call, AND
  3) the wrong choice would materially change the conclusions.
When those hold, use the ask_user tool -- give 2-6 concrete options (and your
recommendation as one of them) and WAIT for the real answer, rather than
writing a question in prose and guessing at what the user would have said.
A bare question in your reply text is not a real pause; nothing stops you
proceeding on your own guess a moment later, and the answer never actually
gets checked. ask_user is a genuine blocking prompt. Prose questions are
still fine for things you genuinely cannot proceed at all without (e.g.
"where is the data?") -- ask_user is specifically for judgment calls where
you COULD proceed on a guess but shouldn't. Keep working on anything
independent while you wait.

# State discipline -- never repeat work, never repeat a mistake
Before running a computation, check whether you already have the answer (an
earlier finding, a cached digest, a variable already in the sandbox). Re-use
it; do not recompute. If a tool call fails, read the error and change your
approach on the next attempt -- if the exact same call fails the same way
twice, that is a signal to stop and diagnose the root cause, not to retry
unchanged a third time.
"""

SYSTEM_PROMPT = load_prompt("system", _SYSTEM_PROMPT_FALLBACK)

# Appended to the system prompt only in text/prompt tool-calling mode (for
# gateways that do NOT support native OpenAI function-calling, e.g. some
# self-hosted Kimi deployments). The model emits tool calls as a strict block
# the shell parses and executes deterministically.
_PROMPT_TOOL_PROTOCOL_FALLBACK = """\

=== HOW TO CALL A TOOL (this environment has no native function-calling) ===
To use a tool, output EXACTLY one block and then STOP -- nothing after it:
<tool>{{"name": "<tool_name>", "arguments": {{ ...json args... }}}}</tool>
The system will execute it and reply with "Observation: <result>". Then
continue. Call ONE tool per message. When you are completely done and want to
speak to the user, just write your message with NO <tool> block.

Available tools (name -- arguments):
{tool_list}

Example:
<tool>{{"name": "load_data", "arguments": {{"path": "data/txns.csv"}}}}</tool>
"""

PROMPT_TOOL_PROTOCOL = load_prompt("tool_protocol", _PROMPT_TOOL_PROTOCOL_FALLBACK)

# These were originally tuned for quick interactive Q&A (60 steps, 20 min)
# and actively fought real deep EDA on large (multi-crore-row) data: real
# pilot feedback was that the analyst "is not doing exploration at all...
# trying to reach the end goal faster" instead of going column-by-column and
# correlation-by-correlation. The user's own words: "we can take hours for
# 8 cr data exploration that is really fine it takes time." These are now
# generous-but-still-finite safety nets (genuine runaway/stuck-loop
# protection), not a pace-setter -- and are overridable per session via the
# ConversationalAnalyst constructor (max_steps / step_call_budget /
# turn_budget_seconds) exactly like turn_budget_seconds already was.
MAX_STEPS = 400
MAX_CONTINUATION_NUDGES = 2  # forced-continuation nudges per user turn (V3-005)
MAX_FIREWALL_REPLIES = 5     # question-firewall auto-replies per user turn (V3-006)
STEP_CALL_BUDGET = 60             # per-plan-step soft cap on tool calls (V3-008)
TURN_WALLCLOCK_BUDGET_S = 14400   # 4 hr default per-turn wall-clock budget (V3-008)

# I-14: hard cap on a tool result entering the history. It sits ABOVE the
# intra-result compactor's output size (ledgerhound.shell.intra_compaction:
# 7k head + 3k tail + marker) on purpose -- the old 6000 cut the tail that
# compactor had just gone out of its way to keep.
_TOOL_RESULT_CHARS = 12_500

_BUDGET_EXHAUSTED_MSG = ("Checkpoint: you've made a lot of tool calls on the "
                         "current plan step. This is not a signal to wrap up "
                         "the analysis -- deep exploration on large data is "
                         "expected to take many calls and hours are fine. If "
                         "this angle (this column, this correlation) still has "
                         "genuine new signal, keep going. If it's exhausted, "
                         "update the plan and move to the NEXT unexplored "
                         "column/relationship rather than concluding early.")
_WRAPUP_REQUEST = ("You've used a lot of steps this turn. Stop calling tools "
                   "now -- summarize what you found and finished so far in "
                   "one clear message to the user.")
_FIREWALL_REPLY = ("Proceed with your best judgment -- that's an analysis "
                   "choice, not something I need to weigh in on. Keep going.")


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t.endswith("?")


def _is_high_risk_question(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _MUTATION_KEYWORDS)


class ConversationLoop:
    """Drives the LLM turn loop over an AnalystTools session.

    A mixin: it owns the conversation (llm, messages, budgets, context
    compaction, tool-call protocol) and relies on the AnalystTools side of
    the same object for the session -- `registry`, `_run_tool`, `_log`,
    `plan`, `goal`, `methodology`, `run_dir`, `_events`. It never touches
    data or guardrails directly; everything the model does goes through
    `_run_tool`. Initialise it AFTER AnalystTools (the system prompt lists
    the registry's tools)."""

    def __init__(self, llm, tool_mode: str = "auto",
                 turn_budget_seconds: int = TURN_WALLCLOCK_BUDGET_S,
                 max_steps: int = MAX_STEPS, step_call_budget: int = STEP_CALL_BUDGET,
                 compress_context: bool = True):
        self.llm = llm
        # per-turn wall-clock budget (V3-008); overridable per session (e.g.
        # a shorter budget for CI/eval runs).
        self.turn_budget_seconds = turn_budget_seconds
        # max_steps / step_call_budget: overridable per session like
        # turn_budget_seconds above -- same reasoning (see the module-level
        # comment by MAX_STEPS).
        self.max_steps = max_steps
        self.step_call_budget = step_call_budget
        # compress_context (V3-010): keep the message list from growing
        # without bound on long sessions (the coherence-cliff failure mode).
        # Rule-based and cheap (no extra LLM round-trip), so it's ON by
        # default -- unlike use_kernel there's no subprocess/OOM risk to
        # gate behind an opt-in flag. See ledgerhound.shell.compressor.
        self._compressor = None
        if compress_context:
            from ledgerhound.shell.compressor import ShellContextCompressor
            # Size the compaction budget from the MODEL's real context
            # window when the config declares one -- compacting on message
            # count alone is blind to how full the window actually is.
            ctx = getattr(getattr(llm, "config", None), "context_window", None)
            self._compressor = (ShellContextCompressor(context_window=int(ctx))
                                if ctx else ShellContextCompressor())
        # tool_mode:
        #   "native" -- only OpenAI function-calling (`tools=`), the default when
        #               the gateway supports it (e.g. DeepSeek).
        #   "prompt" -- gateway does NOT support function-calling; drive tools via
        #               a parseable text protocol only (e.g. some Kimi deploys).
        #   "auto"   -- pass native tools AND accept the text protocol, so a
        #               model that can't function-call still works.
        self.tool_mode = tool_mode if tool_mode in ("native", "prompt", "auto") else "auto"
        # Index of the pinned schema/goal/phase system message (see
        # _refresh_schema_pin); None until data is loaded.
        self._schema_pin_idx: int | None = None

        sys_prompt = SYSTEM_PROMPT
        if self.tool_mode in ("prompt", "auto"):
            sys_prompt += PROMPT_TOOL_PROTOCOL.format(tool_list=self._tool_list_text())
        if self.project is not None:
            try:
                mem = self.project.memory_block()
                if mem:
                    sys_prompt += "\n\n" + mem
            except Exception:
                pass
        sys_prompt += f"\n\nThis session's run folder: {self.run_dir}"
        self.messages: list[dict] = [{"role": "system", "content": sys_prompt}]

    def _tool_list_text(self) -> str:
        """A compact text catalog of tools + their argument names, for the
        prompt-based protocol (models that can't read the OpenAI `tools=`)."""
        lines = []
        for spec in self.registry.openai_specs():
            f = spec["function"]
            props = (f.get("parameters") or {}).get("properties", {}) or {}
            args = ", ".join(props.keys()) or "(none)"
            lines.append(f"- {f['name']}({args}) -- {f['description']}")
        return "\n".join(lines)

    def _norm_call(self, obj: dict) -> dict | None:
        """Validate a parsed object as a tool call against the live registry."""
        name = obj.get("name")
        if not name or name not in self.registry:
            return None
        raw = obj.get("arguments", obj.get("args", {}))
        args, err = _parse_tool_args(raw)
        return {"name": name, "arguments": args, "arg_error": err, "raw_arguments": raw}

    def _parse_text_tool_calls(self, content: str) -> tuple[list[dict], str]:
        """Recover tool calls the gateway emitted as text. Handles BOTH the
        wrapped `<tool>{json}</tool>` protocol AND bare top-level JSON objects
        like `{"name": "run_python", "arguments": {...}}` that Kimi-style
        gateways print straight into `content` when they can't do native
        function-calling. Returns (calls, cleaned_content). Validated against
        the registry so prose JSON is never mistaken for a call."""
        calls: list[dict] = []
        spans: list[tuple[int, int]] = []

        # 1) wrapped <tool>...</tool> blocks
        for m in _TOOL_BLOCK_RE.finditer(content):
            try:
                obj = json.loads(m.group(1), strict=False)
            except Exception:  # noqa: BLE001
                continue
            call = self._norm_call(obj)
            if call:
                calls.append(call)
                spans.append((m.start(), m.end()))

        # 2) bare balanced-brace JSON objects with a known tool name
        bare_calls: list[dict] = []
        bare_spans: list[tuple[int, int]] = []
        i = 0
        n = len(content)
        while i < n:
            if content[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = content[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            if depth == 0 and j < n:
                blob = content[i:j + 1]
                if '"name"' in blob:
                    try:
                        obj = json.loads(blob, strict=False)
                    except Exception:  # noqa: BLE001
                        obj = None
                    call = self._norm_call(obj) if isinstance(obj, dict) else None
                    if call and not any(s <= i < e for s, e in spans):
                        bare_calls.append(call)
                        bare_spans.append((i, j + 1))
                i = j + 1
            else:
                i += 1

        # I-13: a bare JSON object is only a CALL when the message IS the
        # call -- nothing but whitespace around it (multiple bare calls
        # back to back count too). In the default "auto" mode a model that
        # can function-call natively may QUOTE a call in prose ("I would
        # call {...} -- shall I?"); executing that was a real risk. In
        # "prompt" mode (no native tools at all) the old permissive
        # behaviour stays, because there the text IS the protocol.
        if bare_calls:
            # The message must END with the call: a lead-in line before it
            # ("Sure, let me compute that." then the JSON) is exactly what the
            # the organization gateway emits and must execute; prose AFTER it ("-- shall
            # I?") means the JSON was quoted, not issued.
            trailing = content[max(e for _, e in bare_spans):]
            if self.tool_mode == "prompt" or not trailing.strip():
                calls.extend(bare_calls)
                spans.extend(bare_spans)
            else:
                self._log("text_tool_call_ignored", n=len(bare_calls),
                          reason="bare JSON followed by prose (auto mode)")

        cleaned = content
        for s, e in sorted(spans, reverse=True):
            cleaned = cleaned[:s] + cleaned[e:]
        return calls, cleaned.strip()

    def _unreadable_args(self, name: str, raw, err: str) -> str:
        """A tool call whose arguments are not valid JSON: say so, with the
        raw text, instead of running the tool with {} (see _parse_tool_args)."""
        self.render(f"  - {name}(<unreadable arguments: {err[:80]}>)")
        result = _bad_args_result(name, raw, err)
        self._log("tool_args_unreadable", name=name, error=err[:300],
                  raw=(raw if isinstance(raw, str) else repr(raw))[:2000])
        self._log("tool_result", name=name, result=result[:2000], success=False)
        return result

    def _refresh_schema_pin(self) -> None:
        """Keep exactly one system-role message with the live schema, always
        up to date. Inserted once right after the base system prompt, then
        mutated in place on every refresh -- never duplicated, never stale."""
        if self._df is None:
            return
        # The goal banner rides with the schema pin so it is re-stated on
        # EVERY turn -- a goal mentioned once at the start is exactly what
        # context compaction drops, which is how the last run drifted from
        # account-level to transaction-level without noticing.
        block = self.goal.pin_block()
        # The methodology rides with the goal and schema so the CURRENT
        # phase, its allowed grains and its next steps are restated every
        # turn -- a phase declared once is exactly what compaction drops.
        if self.methodology is not None:
            block += "\n\n" + self.methodology.pin_block()
        block += "\n\n" + self._schema_pin_block()
        if self._schema_pin_idx is not None:
            self.messages[self._schema_pin_idx]["content"] = block
        else:
            self._schema_pin_idx = 1
            self.messages.insert(1, {"role": "system", "content": block})

    # -- forced continuation (V3-005) --------------------------------------
    def _open_plan_steps(self) -> list[dict]:
        return [s for s in self.plan.steps if s["status"] in ("pending", "active")]

    def _continuation_nudge(self, open_steps: list[dict]) -> str:
        lines = ["Open plan steps remain:"]
        for s in open_steps:
            lines.append(f"  - [{s['status']}] {s['text']}")
        lines.append("Continue working on them yourself, or call update_plan to "
                     "mark a step 'skipped' or 'blocked' with a one-line reason "
                     "if it no longer applies. Do not just report status -- act.")
        return "\n".join(lines)

    # -- budgets (V3-008) -----------------------------------------------------
    def _graceful_wrapup(self, pass_native: bool) -> str:
        """MAX_STEPS exhausted without a final answer. Ask for one last
        plain-text summary instead of dying on a silent canned string --
        `tool_choice: "none"` (native mode) stops it calling more tools."""
        self.messages.append({"role": "user", "content": _WRAPUP_REQUEST})
        gen_kw = {"tools": self.registry.openai_specs(),
                  "tool_choice": "none"} if pass_native else {}
        text = ""
        try:
            resp = self.llm.generate(self.messages, **gen_kw)
            text = (getattr(resp, "content", "") or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            text = "(tool-step limit reached and the wrap-up call produced no summary)"
        text += "\n\n(stopped -- tool-step limit reached for this turn)"
        self.messages.append({"role": "assistant", "content": text})
        self._log("assistant_final", text=text, finish_reason="step_limit")
        return text

    def abort_turn(self, reason: str = "interrupted") -> None:
        """I-05: make the history valid again after a turn was cut short
        (Ctrl+C, a crash mid-tool). An assistant message carrying
        `tool_calls` must be followed by one `tool` result per call; if the
        interruption landed between them, the NEXT request would be
        rejected by the gateway on every later turn. Drop the dangling
        assistant message and any partial results, then record what
        happened so the model sees the gap on its next turn."""
        msgs = self.messages
        # walk back over trailing tool results to their parent assistant
        i = len(msgs) - 1
        while i >= 0 and msgs[i].get("role") == "tool":
            i -= 1
        if i >= 0 and msgs[i].get("role") == "assistant" and msgs[i].get("tool_calls"):
            n_calls = len(msgs[i]["tool_calls"])
            n_results = len(msgs) - 1 - i
            if n_results < n_calls:
                del msgs[i:]
        # a user turn that never got its reply stays; the note below tells
        # the model why the thread stops there.
        msgs.append({"role": "system",
                     "content": f"[the previous turn was {reason} before it "
                                "finished -- results after that point were "
                                "not delivered; the user will say what to do "
                                "next]"})
        self._log("turn_aborted", reason=reason)

    # -- one user turn ----------------------------------------------------
    def send(self, user_text: str) -> str:
        # Session-quality observability: a scorecard line lands centrally
        # after every turn (success, error, or interrupt) -- see
        # ledgerhound.runtime.session_log module docstring.
        try:
            return self._send_impl(user_text)
        finally:
            self._write_session_record()

    def _write_session_record(self) -> None:
        """Append this turn's session-quality scorecard to the ONE central,
        deployment-wide log (ledgerhound.runtime.session_log) -- never raises,
        never blocks the turn it's reporting on."""
        try:
            from ledgerhound.runtime.session_log import append_session_record
            findings_count = 0
            if self._findings_path.exists():
                findings_count = sum(
                    1 for l in self._findings_path.read_text(encoding="utf-8").splitlines()
                    if l.strip())
            reports = sorted({r for rs in self._reports_by_phase.values() for r in rs})
            from ledgerhound.identity import current_user
            # Running counts -- no re-read of events.jsonl (which is still
            # written in full; nothing the model sees comes from here).
            record = self._session_stats.record(
                run_id=self.run_dir.name,
                user=current_user(),
                project=str(self.project.root) if self.project is not None else "",
                started_at=self._session_started_at,
                objective=self.goal.objective,
                unit_of_analysis=self.goal.unit_of_analysis,
                phase=self.goal.phase,
                modeling_authorised=self.goal.modeling_authorised,
                methodology=self.methodology.name if self.methodology is not None else "",
                mode=getattr(self.methodology, "mode", "") if self.methodology is not None else "",
                dataset_rows=self._true_n_rows or None,
                reports=reports,
                findings_count=findings_count,
            )
            append_session_record(record)
        except Exception:  # noqa: BLE001
            pass

    def _send_impl(self, user_text: str) -> str:
        self._refresh_schema_pin()
        # V6.3 postmortem: the FIRST user turn is usually the whole ask
        # (goal, grain, deliverables); keep it verbatim so set_goal's own
        # distillation of it can never be the only copy that survives.
        if not self._first_user_turn_text:
            self._first_user_turn_text = user_text
        self.messages.append({"role": "user", "content": user_text})
        self._log("user", text=user_text)
        # In "prompt" mode the gateway can't do native function-calling, so we
        # do not send `tools=` at all and rely purely on the text protocol.
        pass_native = self.tool_mode in ("native", "auto")
        nudges = 0
        firewall_replies = 0
        turn_start = time.monotonic()
        steps_since_plan_update = 0
        budget_note_injected = False
        for _ in range(self.max_steps):
            # -- context compaction (V3-010): checked before every LLM call
            # (not just once per turn) so even one very long tool-calling
            # turn gets compacted, not only growth across turns.
            if self._compressor is not None:
                before_n = len(self.messages)
                before_tok = self._compressor.estimate(self.messages)
                if self._compressor.maybe_compress(self):
                    # Log TOKENS as well as message counts. The 2026-08-06 run
                    # had a 56.8-minute unexplained gap immediately after a
                    # compaction, and nothing in the event stream recorded how
                    # big the request actually was -- so it could not be
                    # diagnosed after the fact. It can be now.
                    self._log("context_compressed", messages_before=before_n,
                              messages_after=len(self.messages),
                              est_tokens_before=before_tok,
                              est_tokens_after=self._compressor.estimate(self.messages),
                              token_budget=self._compressor.token_budget(),
                              context_window=self._compressor.context_window,
                              reason=getattr(self._compressor, "_last_reason", ""))

            # -- budget check (V3-008): a soft cap on tool calls made under
            # one active plan step (only meaningful once a plan exists), plus
            # a per-turn wall-clock budget that applies regardless. Inject
            # guidance once and let the model wrap up on its own terms,
            # instead of silently dying at the hard MAX_STEPS ceiling.
            if not budget_note_injected:
                elapsed = time.monotonic() - turn_start
                step_budget_hit = bool(self.plan.steps) and \
                    steps_since_plan_update >= self.step_call_budget
                wallclock_hit = elapsed >= self.turn_budget_seconds
                if step_budget_hit or wallclock_hit:
                    self.messages.append({"role": "user",
                                          "content": _BUDGET_EXHAUSTED_MSG})
                    self._log("budget_exhausted",
                              reason="step_cap" if step_budget_hit else "wallclock",
                              steps_since_plan_update=steps_since_plan_update,
                              elapsed_s=round(elapsed, 1))
                    budget_note_injected = True
                    steps_since_plan_update = 0

            gen_kw = {}
            if pass_native:
                gen_kw = {"tools": self.registry.openai_specs(),
                          "tool_choice": "auto"}
            if self._compressor is not None:
                self._compressor.before_send(self.messages, gen_kw.get("tools"))
            resp = self.llm.generate(self.messages, **gen_kw)
            if self._compressor is not None:
                # The gateway's usage.prompt_tokens is the exact count for
                # what was just sent -- calibrate from it, and cross-check an
                # attached tokenizer against it.
                note = self._compressor.after_response(getattr(resp, "usage", None))
                if note:
                    self.render(note)
                    self._log("tokenizer_mismatch", detail=note[:1000])
            calls = list(getattr(resp, "tool_calls", []) or [])
            content = getattr(resp, "content", "") or ""
            thinking = (getattr(resp, "thinking", "") or "").strip()
            if thinking:
                self.render(f"reasoning: {thinking[:220]}")

            # 1) Native OpenAI tool calls (DeepSeek etc.)
            if calls:
                interim = content.strip()
                if interim:
                    self.render(f"...{interim[:400]}")
                self.messages.append({"role": "assistant", "content": content,
                                      "tool_calls": calls})

                parsed = []
                bad_args: dict = {}
                for tc in calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args, arg_err = _parse_tool_args(fn.get("arguments"))
                    if args is None:
                        bad_args[id(tc)] = (fn.get("arguments"), arg_err)
                    parsed.append((tc, name, args))

                # V3-016: if THIS response asked for 2+ high-tier actions at
                # once, gate them as ONE combined prompt instead of pausing
                # once per action -- still gated before any of them dispatch,
                # never deferred across turns (the batch is flushed before
                # the dispatch loop below even starts).
                from ledgerhound.analyst.hitl import classify_action
                high_idx = [i for i, (_, name, a) in enumerate(parsed)
                           if a is not None and classify_action(name) == "high"]
                # (each still gets its OWN "hitl" log entry inside
                # _run_tool below -- this loop only computes the decisions.)
                decisions_by_idx: dict[int, Any] = {}
                if len(high_idx) >= 2:
                    for i in high_idx:
                        _, name, args = parsed[i]
                        self.registry.gate.queue_high(**self._decision_package(name, args))
                    for i, decision in zip(high_idx, self.registry.gate.flush_batch()):
                        decisions_by_idx[i] = decision

                for i, (tc, name, args) in enumerate(parsed):
                    if args is None:
                        raw_args, arg_err = bad_args.get(id(tc), ("", ""))
                        result = self._unreadable_args(name, raw_args, arg_err)
                    else:
                        result = self._run_tool(name, args,
                                                high_tier_decision=decisions_by_idx.get(i))
                    steps_since_plan_update = 0 if name == "update_plan" \
                        else steps_since_plan_update + 1
                    self.messages.append({"role": "tool",
                                          "tool_call_id": tc.get("id"),
                                          "content": result[:_TOOL_RESULT_CHARS]})
                continue

            # 2) Text-protocol tool calls (gateways without function-calling)
            if self.tool_mode in ("prompt", "auto"):
                text_calls, cleaned = self._parse_text_tool_calls(content)
                if text_calls:
                    if cleaned:
                        self.render(f"...{cleaned[:400]}")
                    # Keep the model's own message verbatim, then feed the
                    # observations back as a plain user turn (no tool role --
                    # a non-function-calling gateway won't accept that schema).
                    self.messages.append({"role": "assistant", "content": content})
                    obs = []
                    for c in text_calls:
                        if c["arguments"] is None:
                            result = self._unreadable_args(c["name"], c.get("raw_arguments"),
                                                           c.get("arg_error", ""))
                        else:
                            result = self._run_tool(c["name"], c["arguments"])
                        steps_since_plan_update = 0 if c["name"] == "update_plan" \
                            else steps_since_plan_update + 1
                        obs.append(f"Observation from {c['name']}:\n"
                                   f"{result[:_TOOL_RESULT_CHARS]}")
                    self.messages.append({"role": "user", "content": "\n\n".join(obs)})
                    continue

            # 3) No tool call -> final answer (or an empty-turn diagnostic)
            text = content.strip()
            if not text:
                text = self._empty_turn_message(resp)

            # 3a) Question firewall (V3-006, autonomous mode only): a bare,
            # non-risk-tier question gets auto-answered so the run keeps
            # going instead of stalling on something the model could have
            # just decided itself.
            if (self.autonomous and firewall_replies < MAX_FIREWALL_REPLIES
                    and _looks_like_question(text)
                    and not _is_high_risk_question(text)):
                firewall_replies += 1
                self.messages.append({"role": "assistant", "content": text})
                self.messages.append({"role": "user", "content": _FIREWALL_REPLY})
                self._log("question_firewall", reply_count=firewall_replies,
                          question=text[:300])
                continue

            open_steps = self._open_plan_steps()
            if open_steps and nudges < MAX_CONTINUATION_NUDGES:
                nudges += 1
                self.messages.append({"role": "assistant", "content": text})
                nudge = self._continuation_nudge(open_steps)
                self.messages.append({"role": "user", "content": nudge})
                self._log("forced_continuation", nudge_count=nudges,
                          open_steps=len(open_steps))
                continue
            if open_steps:
                text += f"\n\n(stopped with {len(open_steps)} plan step(s) still open)"
            self.messages.append({"role": "assistant", "content": text})
            self._log("assistant_final", text=text,
                      finish_reason=getattr(resp, "finish_reason", ""))
            return text
        self._log("step_limit_reached")
        return self._graceful_wrapup(pass_native)

    def _empty_turn_message(self, resp) -> str:
        """Turn a no-content / no-tool-call response into a readable diagnostic,
        and dump the discriminating raw fields to the transcript so the exact
        cause (legacy function_call / thinking-truncation / genuinely empty /
        tool-tokens-as-text) can be identified from one real capture."""
        fr = getattr(resp, "finish_reason", "") or "?"
        thinking = (getattr(resp, "thinking", "") or "").strip()
        usage = getattr(resp, "usage", {}) or {}
        raw = getattr(resp, "raw", {}) or {}
        msg = {}
        try:
            msg = (raw.get("choices") or [{}])[0].get("message", {}) or {}
        except Exception:
            pass
        self._log("empty_turn", finish_reason=fr, usage=usage,
                  reasoning_present=bool(thinking),
                  message_keys=sorted(msg.keys()),
                  has_function_call=bool(msg.get("function_call")),
                  message_snippet=str(msg)[:1500])
        lines = [f"(no answer and no tool call this turn -- finish_reason={fr})"]
        if fr == "length":
            lines.append("The model hit its token limit before answering. Raise "
                         "`max_tokens` in the model config (or reduce/disable "
                         "thinking mode) and try again.")
        if thinking:
            lines.append("Its reasoning was:\n" + thinking[:700])
        else:
            lines.append("Nothing came back -- no content, no reasoning, no tool "
                         "call. If this model is meant to use tools, the gateway "
                         "may not be returning them in OpenAI `tool_calls` "
                         "format. The raw response shape for this turn is logged "
                         "in this run's transcript.jsonl (event=empty_turn).")
        return "\n".join(lines)
