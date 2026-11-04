"""
ledgerhound.shell.conversation
===========================

The conversational analyst -- Claude-Code-style UX for data science. There is
NO wizard and NO hardcoded on-ramp: the LLM drives the whole conversation from
the first token via native function-calling, exactly like Claude Code drives a
coding session. The user talks naturally; the model decides when to ask, how to
resolve a data path, what analysis fits, and when it is done.

Tools the model can call:
  list_directory   -- see what files are in a folder
  load_data        -- resolve a path (tries ./ ../ cwd project-root, globs a
                      folder) and load the dataframe for the session
  profile_data     -- deterministic digest of the loaded data
  run_python       -- persistent notebook-style sandbox (df/pd/np in scope,
                      variables SURVIVE across calls)
  record_finding   -- log an evidence-backed, sample-frame-tagged finding
  save_report      -- write a report into run_outputs/<timestamp>/

Skills are offered as *reference the model may consult*, never force-injected --
so "EDA on fraud data" does EDA, not a railroaded fraud-vs-nonfraud procedure.
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ledgerhound.shell.prompts.loader import load_prompt

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
- At crore-scale, a full `df.copy()` is real memory, not free. Don't accumulate
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
  counterparties, device/VPA sharing, repeat-offender history, rolling-window
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
large data (an 8-crore-row transaction table is a normal size to work with
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
# V6.3 postmortem fix: tools that only manage plan/goal bookkeeping, never
# analysis -- excluded from the "evidence" counter update_plan checks before
# letting a step close "done" (ledgerhound/tools/plan_tools.py).
_PLAN_META_TOOLS = {"propose_plan", "update_plan", "add_node", "invalidate_node",
                    "set_goal", "set_methodology", "complete_step", "add_step",
                    "advance_phase", "phase_status",
                    "request_modeling_authorisation"}
STEP_CALL_BUDGET = 60             # per-plan-step soft cap on tool calls (V3-008)
TURN_WALLCLOCK_BUDGET_S = 14400   # 4 hr default per-turn wall-clock budget (V3-008)
_DATA_EXTS = (".parquet", ".csv", ".tsv", ".pq", ".json", ".feather")

# In KERNEL mode only: above this row count the shell process stops loading
# the whole dataset (the kernel already holds it) and keeps a bounded schema
# sample instead. Exec mode is unaffected -- there, self._df IS the data the
# analysis runs on, so it must stay complete.
_SHELL_LOAD_MAX_ROWS = 2_000_000
_SHELL_SCHEMA_SAMPLE_ROWS = 50_000

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

# Keywords that mark a question as touching a data-mutating or business
# decision (drop rows, choose an imputation, set a threshold, overwrite a
# save) -- these are the ONLY questions the firewall must never auto-answer,
# per the risk-tier policy in the system prompt's "Decide, don't ask" section.
_MUTATION_KEYWORDS = ("drop", "delete", "impute", "overwrit", "remove",
                     "discard", "truncate", "threshold", "save over",
                     "save-over")
_FIREWALL_REPLY = ("Proceed with your best judgment -- that's an analysis "
                   "choice, not something I need to weigh in on. Keep going.")


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t.endswith("?")


def _is_high_risk_question(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _MUTATION_KEYWORDS)


# -- kernel-mode traceback parsing (V3-009) -----------------------------------
# A real IPython kernel runs in a separate process, so a failure comes back
# as a traceback STRING, not a live exception object -- best-effort recover
# (exception_class_name, message) from its last "Kind: message" line so the
# same teaching-hint logic (_teach_error_text) can be reused unchanged.
_TB_LAST_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)):\s*(.*)$")


def _parse_traceback_signature(tb: str) -> tuple[str, str]:
    for line in reversed((tb or "").strip().splitlines()):
        m = _TB_LAST_LINE_RE.match(line.strip())
        if m:
            return m.group(1), m.group(2)
    lines = (tb or "").strip().splitlines()
    return "Error", (lines[-1] if lines else "unknown error")


# -- identical-failure detection (V3-007) -------------------------------------

_REPEAT_FAILURE_NOTE = ("\n\nNOTE: you already tried exactly this and it failed "
                        "the same way. Do NOT retry it unchanged -- diagnose the "
                        "root cause and change your approach.")


def _parse_tool_args(raw) -> tuple[dict | None, str]:
    """(args, error) for a tool call's `arguments`, which arrive as a JSON
    string (OpenAI shape), a dict, or empty. On failure args is None and
    error says why -- NEVER a silent {}: substituting {} made a call whose
    JSON merely failed to parse look like "run_python() missing the 'code'
    argument", a message that points at the tool instead of at the real
    cause (the gateway/model produced arguments that are not valid JSON,
    typically code with unescaped quotes or newlines)."""
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, dict):
        return raw, ""
    if not isinstance(raw, str):
        return None, f"expected a JSON object, got {type(raw).__name__}"
    try:
        val = json.loads(raw, strict=False)
        if isinstance(val, str):                 # double-encoded JSON string
            val = json.loads(val, strict=False)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(val, dict):
        return None, f"expected a JSON object, got {type(val).__name__}"
    return val, ""


def _bad_args_result(name: str, raw, err: str) -> str:
    shown = raw if isinstance(raw, str) else repr(raw)
    return (f"error: the arguments for {name} could not be read as a JSON object "
            f"({err}). Nothing was run. Raw text received: {shown[:300]!r}. "
            "Resend the call with valid JSON arguments -- escape every double quote "
            "inside a string as \\\" and every line break as \\n, or write the "
            "Python with single-quoted strings so there is nothing to escape.")


def _failure_signature(result: str) -> str | None:
    """Return a short signature if a tool result reads as a failure, else
    None. Used to key the identical-failure detector -- deliberately
    conservative (prefix/substring checks on known failure phrasings) so a
    normal informational reply is never mistaken for a failure."""
    if not isinstance(result, str):
        return None
    if result.startswith("Error:"):
        return result.splitlines()[0][len("Error:"):].strip()
    if result.startswith("blocked:"):
        return result.splitlines()[0].strip()
    # Guardrail refusals (goal gates, restatement guard, chapter dedupe,
    # phase gate). Without this a refused cell counted as a success: it was
    # tallied as evidence, offered an "undo" for a change that never ran,
    # and never tripped the identical-failure note.
    if result.startswith("REFUSED"):
        return result.splitlines()[0].strip()[:160]
    # The tool registry's own failures ("error: bad arguments for X: ...",
    # "error running X: ...", "error: unknown tool X") and unreadable
    # arguments. They used to slip past this check (it only knew "Error:"),
    # so the identical-failure note never fired and a model retried the
    # same dead call ~14 times in one turn (pod log, 2026-10-05).
    if result[:6].lower() == "error:" or result[:13].lower() == "error running":
        return result.splitlines()[0].strip()[:160]
    if "could not find data at" in result:
        return "load_data: not found"
    if "failed to load it:" in result:
        return "load_data: load failure"
    if result.startswith("save FAILED entirely"):
        return "save_report: failed entirely"
    return None


def _norm_args_for_failure(args: dict) -> str:
    if not isinstance(args, dict):
        return str(args)
    return str(sorted(args.items()))

_BLOCKED = ("import os", "import sys", "import subprocess", "import shutil",
            "open(", "__import__", "eval(", "exec(", "requests.", "socket.")


def _ts() -> str:
    """Run-folder timestamp. V3-057: was local time, the only such stamp in
    the codebase -- every other one (dataset_card.py, project.py,
    workspace/layout.py) is UTC ISO. Verified before changing: nothing
    parses this specific format back into a datetime (ledgerhound/eval/
    trend.py's regex matches ledgerhound.auto.runner's SEPARATE run_<ts> id,
    a different engine with its own strftime call, untouched here); every
    other reference to a shell run_dir's name is an opaque string/equality
    check, never re-parsed for its calendar meaning. Kept "-" as the
    field separator (not ":") so the string stays filesystem-safe on
    Windows without extra escaping."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


# ---- native tool specs (OpenAI function-calling) ----------------------------
TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and subfolders at a path so you can see what data is available.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder path (default current dir)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "load_data",
        "description": "Resolve a data path (tries as-given, ./, ../, cwd, project root, and globs a folder) and load it as the session dataframe `df`. Returns shape + columns, or a folder listing to choose from.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File or folder path, may be relative."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "profile_data",
        "description": "Deterministically profile the loaded dataframe into a compact digest (columns, dtypes, missingness, target).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python in a persistent sandbox. `df`, `pd`, `np` are in scope; variables persist across calls. Print results you want to see. Save feature tables and other data files under `features_dir` (a Path in scope), e.g. `feat.to_parquet(features_dir / 'account_features.parquet')`.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "propose_falsification",
        "description": "Given a claim you have recorded (or are about to), "
                       "return the specific ways a claim of that shape has "
                       "actually turned out wrong before, as an attack plan. "
                       "Use inside the 'falsification' chapter. Every other "
                       "chapter looks for support; this one tries to kill the "
                       "claim, which is the only test that tells you anything.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string",
                      "description": "The claim to attack, stated as recorded."}},
            "required": ["claim"]}}},
    {"type": "function", "function": {
        "name": "record_finding",
        "description": "Log an evidence-backed finding with an honest sample-frame tag. Checked BEFORE recording against the domain profile, this book's own history, and the team knowledge base -- a claim that matches one of these is REFUSED (not recorded) unless you also pass delta explaining what it adds.",
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "string"},
            "sample_frame": {"type": "string",
                             "enum": ["within_stratum", "cross_stratum",
                                      "population_estimate", "unknown"]},
            "predicate": {"type": "string",
                         "description": "The exact condition/window the "
                         "claim's numbers were computed under, e.g. "
                         "'hour>=22 OR hour<6'. Optional, but if you record "
                         "a claim that will later be COMPARED against "
                         "another finding about the same quantity (e.g. "
                         "fraud vs non-fraud night-share), give both the "
                         "SAME predicate wording or they cannot be checked "
                         "for actually meaning the same thing."},
            "pvalue": {"type": "number",
                      "description": "The raw p-value of the statistical test "
                      "behind this claim, if there was one. Supply it whenever "
                      "a test was actually run: it is corrected across every "
                      "test declared in this chapter, and only a claim that "
                      "survives that correction is recorded as ESTABLISHED "
                      "rather than EXPLORATORY. Omitting it does not make a "
                      "claim stronger -- it makes it unquantified."},
            "delta": {"type": "string",
                     "description": "Required ONLY if this claim is refused"
                     "as a restatement of something already known "
                     "(established finding, this book's history, or the "
                     "team KB). State specifically what this version ADDS "
                     "-- a mechanism, an interaction, account-level "
                     "behaviour, a feature it justifies -- then call "
                     "record_finding again with the same claim plus this."}},
            "required": ["claim", "sample_frame"]}}},
    {"type": "function", "function": {
        "name": "save_report",
        "description": "Save a markdown report into the session's dated run folder. Each save writes a NEW numbered file rather than overwriting, so a first broad pass and a later deeper pass both survive. The filename carries the current phase and the sequence number automatically -- you do not supply either. Findings you recorded since your last save are attached to the report automatically, so record_finding as you work. EVERY PHASE MUST HAVE AT LEAST ONE SAVED REPORT: advance_phase refuses while the current phase has none. Saving itself is not gated -- just call it.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "markdown": {"type": "string", "description": "Full report body in markdown. Include the actual numbers -- rates, counts, thresholds, and the n they were computed on."}},
            "required": ["title", "markdown"]}}},
    {"type": "function", "function": {
        "name": "consult_skill",
        "description": "Optionally look up the team's proven procedure for a task (e.g. 'fraud vs non-fraud', 'mule graph', 'signal decay'). Reference only -- adapt it, don't follow blindly.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What you're trying to do."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "Pause and ask the analyst a concrete multiple-choice (single answer) or multi-select (several answers) question, then WAIT for their real answer before continuing -- for genuine judgment calls a mechanical check flagged as ambiguous (e.g. 'is this column operational metadata or real behavioural signal?'), not for things you can decide yourself. Give concrete options, not an open-ended question -- the analyst picks from what you offer.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The concrete question, with enough context to answer it."},
            "options": {"type": "array", "items": {"type": "string"},
                       "description": "2-6 concrete options to choose from."},
            "multi": {"type": "boolean",
                     "description": "true if more than one option may apply (multi-select); false (default) for pick-exactly-one."}},
            "required": ["question", "options"]}}},
    {"type": "function", "function": {
        "name": "set_methodology",
        "description": "Adopt a declared analysis methodology: an ordered sequence of phases, each with its own data scope, allowed grains and a checklist of concrete EDA steps. Call this at the start. Templates: 'topdown_7phase' (fraud-only EDA -> non-fraud -> contrast -> aggregation plan -> features at BOTH grains -> selection -> modelling), 'sample_first_7phase' (phase 1 DECIDES how non-fraud is sampled), 'bottom_up_7phase' (label withheld until phase 4 -- profile the whole population's structure first, then test what survives contact with the fraud label), 'deep_eda_only' (bottom-up exploration only, refuses to reach feature construction), 'fe_only' (skips exploration, starts straight at feature construction off already-known signal), 'eda_fe_only' (exploration + features, refuses to reach modelling). Call with no name to list what is available. The methodology is pinned every turn and drives the grain/phase gates -- transaction-level work is legitimate wherever a phase declares it.",
        "parameters": {"type": "object", "properties": {
            "template": {"type": "string", "description": "Template name, e.g. 'topdown_7phase', 'deep_eda_only'. Omit to list the available templates."},
            "mode": {"type": "string", "enum": ["", "eda_only", "fe_only", "eda_fe_only"],
                    "description": "A SEPARATE axis from which template -- controls how far the pipeline goes, not the reasoning order. '' (default) = full sequence. 'eda_only' = stop before feature construction. 'fe_only' = skip exploration, jump straight to building features off already-known/stated signal. 'eda_fe_only' = go through feature construction but stop before modelling authorisation."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "complete_step",
        "description": "Mark a step of the current phase done (or skipped, with a reason) and record what it found. Keeps 'this phase is complete' checkable rather than asserted.",
        "parameters": {"type": "object", "properties": {
            "step_id": {"type": "string", "description": "Step id, e.g. 'p1.04'."},
            "note": {"type": "string", "description": "What this step established -- the finding, not just 'done'."},
            "skipped": {"type": "boolean", "description": "true if the step was deliberately skipped; put the reason in note."}},
            "required": ["step_id"]}}},
    {"type": "function", "function": {
        "name": "add_step",
        "description": "Append a step the data actually demands to the current phase. The shipped checklist is a floor, not a ceiling -- when the analysis surfaces something worth chasing, add it here so it is tracked rather than done ad hoc and forgotten.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to investigate, concretely."},
            "grain": {"type": "string", "description": "'transaction', 'account' or 'any'."}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "advance_phase",
        "description": "Finish the current phase and move to the next. Refuses while steps are pending unless force=true; a forced skip marks the remaining steps skipped and is recorded, so the audit trail stays honest.",
        "parameters": {"type": "object", "properties": {
            "force": {"type": "boolean", "description": "Move on with steps still pending (they are recorded as skipped)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "phase_status",
        "description": "Where the methodology stands: current phase, per-phase step progress, and what remains in this phase.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "plan_features",
        "description": "Enumerate MANY candidate entity-level features (hundreds; 1000 is fine) from the loaded schema, and get runnable code to compute them in batches. Use this after the EDA iterations, when you move into feature engineering. These are CANDIDATES, not recommendations -- you must then compute them and run evaluate_features(...) to rank/prune, and only hand over what survives. Breadth here is good; breadth in the deliverable is what the pilot rejected.",
        "parameters": {"type": "object", "properties": {
            "entity_key": {"type": "string", "description": "Column to aggregate to, e.g. payer_account_number. Defaults to the session goal's entity_key."},
            "time_column": {"type": "string", "description": "Timestamp column, to enable velocity/rolling-window and temporal-shape features."},
            "max_features": {"type": "integer", "description": "Cap on candidates generated (default 1000)."},
            "windows": {"type": "array", "items": {"type": "integer"}, "description": "Rolling windows in days (default [1,7,30])."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "export_feature_catalog",
        "description": "Export the documented feature catalog: every candidate with its description, the exact reproducing code, its measured quality (IV/KS/AUC), and its TIER. Features are tiered (core / supporting / review / dropped), not pruned to a shortlist -- a modest univariate IV can still earn its place in a multivariate model, so only leakage-vetoed, redundant, or genuinely signal-free features are dropped. Thresholds are adjustable. Call after computing the features and running evaluate_features. Writes feature_catalog.csv + .md. When the goal declares entity_key_secondary, every feature appears for BOTH sides (`<name>__remitter`, `<name>__beneficiary`): an evaluation keyed by the plain name counts for the primary (remitter) side only -- compute the beneficiary-side feature and pass its evaluation as `<name>__beneficiary`, or that side stays unmeasured.",
        "parameters": {"type": "object", "properties": {
            "evaluations": {"type": "array", "items": {"type": "object"}, "description": "Per-feature dicts from evaluate_features(...)['evaluations'] -- feature/iv/ks/auc/iv_band."},
            "vetoed": {"type": "array", "items": {"type": "string"}, "description": "Feature names the leakage check vetoed -- these are hard-dropped."},
            "redundant": {"type": "array", "items": {"type": "string"}, "description": "Feature names pruned as correlated duplicates of a stronger feature."},
            "iv_core": {"type": "number", "description": "IV at/above which a feature is 'core' (default 0.10)."},
            "iv_supporting": {"type": "number", "description": "IV at/above which a feature is KEPT as 'supporting' (default 0.02 -- deliberately permissive; lower it to keep even more)."},
            "title": {"type": "string", "description": "Catalog title."},
            "percentiles": {"type": "object", "description": "Optional {feature_name: percentile_table(fraud, nonfraud) result} map -- adds a percentile check to each feature's entry showing whether the fraud/non-fraud tails cross or where they diverge, which IV/KS/AUC alone cannot show. Build it by calling percentile_table(...) per feature before exporting."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "recall_history",
        "description": "Search the VERBATIM archive of earlier messages that were compacted out of the context window. Compaction summarises, it never deletes -- so if a number, code cell, error or decision is missing from the summary, look it up here instead of recomputing it. Use it whenever you are about to redo work that you suspect was already done.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words to search for (e.g. a variable name, a column, 'fraud rate', an error type)."},
            "max_hits": {"type": "integer", "description": "Max matching messages to return (default 5)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "set_goal",
        "description": "Declare (or refine) what this session is FOR: the objective, the unit of analysis the DELIVERABLE must be at (account or transaction), the entity key, and the label column. Call this early -- before building or exporting features. The goal is pinned into every turn and is ENFORCED: with unit_of_analysis='account', exporting a frame with duplicate entity keys is refused, not warned about. Exploration at any grain is always allowed; this constrains what you SHIP.",
        "parameters": {"type": "object", "properties": {
            "objective": {"type": "string", "description": "What the user actually wants out of this session, in one sentence."},
            "unit_of_analysis": {"type": "string", "description": "'account' (one row per entity in the deliverable) or 'transaction'."},
            "entity_key": {"type": "string", "description": "Column that identifies the entity, e.g. payer_account_number. Required when unit_of_analysis='account'."},
            "entity_key_secondary": {"type": "string", "description": "The OTHER side of a payment, e.g. payee_account_number, if this dataset has one. Aggregation defaults to entity_key alone (typically the payer/remitter side) -- set this so feature generation can ALSO be run against the beneficiary side, which is often where the account-level verdict actually lives."},
            "label_column": {"type": "string", "description": "The target/label column, if known."},
            "notes": {"type": "string", "description": "Any constraint worth pinning (e.g. 'features only, no modelling yet')."}},
            "required": ["objective"]}}},
    {"type": "function", "function": {
        "name": "request_modeling_authorisation",
        "description": "Ask the human to authorise moving from analysis/feature work into model training. Model training is REFUSED until this is granted. Call it only when you believe the feature set is genuinely ready, and summarise what you built so the human can judge.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What you have built/found so far, and why you think modelling is the right next step."}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "save_understanding",
        "description": "Archive this session's understanding as a durable 'book' (summary + goal + data + findings) so it is recalled in future sessions on this data. Offer this when the analysis reaches a useful conclusion.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string", "description": "One-paragraph index summary."},
            "goal": {"type": "string"}},
            "required": ["title", "summary"]}}},
    {"type": "function", "function": {
        "name": "record_decision",
        "description": "Log a data-mutating or business decision (e.g. imputation strategy, threshold chosen, rows dropped) to the dataset's durable card, so future sessions on this data know it was already decided and why.",
        "parameters": {"type": "object", "properties": {
            "decision": {"type": "string", "description": "What was decided."},
            "rationale": {"type": "string", "description": "Why, in one line."}},
            "required": ["decision"]}}},
    {"type": "function", "function": {
        "name": "record_data_answer",
        "description": "Save the USER's answer to a question about the data (the [key] questions listed after profile_data: what a label's classes mean, which entity/time column, units, what codes mean). Stored on the dataset card and shared with the team, so nobody is asked again on this dataset. Only record what the user actually said -- never your own guess.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "The question key in brackets, e.g. 'label_classes:fraud_flag'."},
            "answer": {"type": "string", "description": "The user's answer, in their words."},
            "question": {"type": "string", "description": "The question as asked (optional)."}},
            "required": ["key", "answer"]}}},
    {"type": "function", "function": {
        "name": "open_chapter",
        "description": "Declare the APPROACH for the next stretch of analysis on this dataset (a chapter, from a fixed taxonomy -- e.g. 'percentile-profile', 'interaction-sweep'). Deduped against this exact data version: the same chapter with pages already recorded on UNCHANGED data is refused, showing what was already found -- this is what stops a second session on the same data from redoing the first session's work. A new content hash (new month, re-extracted file) auto-allows. Call this before starting a genuinely new line of analysis; add_chapter_page records progress under it.",
        "parameters": {"type": "object", "properties": {
            "chapter": {"type": "string", "description": "One name from the chapter taxonomy (ask if unsure -- an unknown name returns the list)."}},
            "required": ["chapter"]}}},
    {"type": "function", "function": {
        "name": "redo_chapter",
        "description": "Override a chapter refusal when the data has NOT changed but you have a real reason to redo it (the first pass was buggy, a wrong assumption, etc.). Always recorded, never silent.",
        "parameters": {"type": "object", "properties": {
            "chapter": {"type": "string"},
            "reason": {"type": "string", "description": "Why this needs to be redone despite unchanged data."}},
            "required": ["chapter", "reason"]}}},
    {"type": "function", "function": {
        "name": "add_chapter_page",
        "description": "Record one page of progress/findings under the currently open chapter. Append-only -- call this as you go, not just at the end.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What this page covers -- the specific thing done or found."}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "record_percentile_snapshot",
        "description": "Record a feature's percentile table (from percentile_table(a, b) -- pass either side's flat {label: value} dict) so a LATER session on different data can check whether the pattern held. This is the numeric record cross-period confirmation compares against; add_chapter_page's prose notes are not enough for that.",
        "parameters": {"type": "object", "properties": {
            "feature": {"type": "string", "description": "The feature this snapshot is for."},
            "percentiles": {"type": "object", "description": "Flat {label: value} dict, e.g. percentile_table(fraud, nonfraud)['a']."},
            "chapter": {"type": "string", "description": "Defaults to the currently open chapter if omitted."}},
            "required": ["feature", "percentiles"]}}},
    {"type": "function", "function": {
        "name": "check_cross_period_stability",
        "description": "Compare a recorded percentile snapshot for `feature` between the CURRENT data version and an earlier story on this same dataset -- classifies STABLE or SHIFTED at each quantile. Requires record_percentile_snapshot to have been called in both stories first. Use this instead of eyeballing two percentile tables when confirming a finding holds across a later period.",
        "parameters": {"type": "object", "properties": {
            "feature": {"type": "string"},
            "chapter": {"type": "string", "description": "Defaults to the currently open chapter if omitted."},
            "against_story": {"type": "string", "description": "A story_id (see /story or /library) to compare against. Defaults to the story immediately before the current one."}},
            "required": ["feature"]}}},
    {"type": "function", "function": {
        "name": "undo_last_transform",
        "description": "Restore the dataframe to its state before the last run_python call that changed it -- use this when a mutation (a filter, drop, etc.) turns out to be wrong. Only available for small/medium in-session (exec-mode) data; kernel-mode or very large data has no automatic undo copy -- use load_data to start over instead.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


class ConversationalAnalyst:
    """An LLM-driven analyst session. `send(text)` runs one user turn to its
    natural stopping point (a text reply), executing tool calls along the way."""

    def __init__(self, llm, project=None, out_root: str | Path = "run_outputs",
                 renderer: Callable[[str], None] = print, registry=None,
                 gate=None, tool_mode: str = "auto", autonomous: bool = False,
                 turn_budget_seconds: int = TURN_WALLCLOCK_BUDGET_S,
                 max_steps: int = MAX_STEPS, step_call_budget: int = STEP_CALL_BUDGET,
                 use_kernel: bool = False, compress_context: bool = True,
                 methodology: str = "", methodology_mode: str = ""):
        self.llm = llm
        self.project = project
        self.render = renderer
        # autonomous: dev/eval flag (V3-006). When on, any final reply that
        # is a bare, non-risk-tier question gets auto-answered with "proceed
        # with your best judgment" instead of surfacing to a human -- so a
        # fluidity run measures only the questions that genuinely needed a
        # person. Off by default; production shells leave real questions to
        # the user.
        self.autonomous = autonomous
        # per-turn wall-clock budget (V3-008); overridable per session (e.g.
        # a shorter budget for CI/eval runs).
        self.turn_budget_seconds = turn_budget_seconds
        # max_steps / step_call_budget: overridable per session like
        # turn_budget_seconds above -- same reasoning (see the module-level
        # comment by MAX_STEPS).
        self.max_steps = max_steps
        self.step_call_budget = step_call_budget
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
        # V5-003: repeat advice on every cell is noise, and noise is
        # what gets a check switched off.
        self._stat_issues_seen: set = set()
        self._stat_advisory: str = ""
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
        self._schema_pin_idx: int | None = None
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
        self.registry = registry or self._default_registry(gate)

        sys_prompt = SYSTEM_PROMPT
        if self.tool_mode in ("prompt", "auto"):
            sys_prompt += PROMPT_TOOL_PROTOCOL.format(tool_list=self._tool_list_text())
        if project is not None:
            try:
                mem = project.memory_block()
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

    def _run_tool(self, name: str, args: dict, *, high_tier_decision=None) -> str:
        self.render(f"  - {name}({_brief(args)})")
        from ledgerhound.analyst.hitl import classify_action
        tier = classify_action(name)

        # V3-016: high-tier actions are gated IN-LINE, before dispatch --
        # never deferred across turns. `high_tier_decision` lets `send()`
        # pre-gate a whole response's simultaneous tool_calls as one batch
        # (see the native tool_calls handling below); a single high-tier
        # call with no batch gates itself right here.
        if tier == "high":
            decision = high_tier_decision if high_tier_decision is not None \
                else self.registry.gate.high(**self._decision_package(name, args))
            self._log("hitl", tier="high", name=name, approved=decision.approved,
                      answer=decision.answer)
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
        result = self._dispatch(name, args)
        sig = _failure_signature(result)
        success = sig is None
        if success and name not in _PLAN_META_TOOLS:
            self._evidence_count += 1
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

    # -- tool dispatch ----------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> str:
        return self.registry.dispatch(name, args)

    def _tool_list_directory(self, path: str = ".") -> str:
        p = self._resolve_dir(path)
        if p is None or not p.exists():
            return f"path not found: {path}"
        if p.is_file():
            return f"{p} is a file ({p.stat().st_size} bytes)"
        entries = sorted(p.iterdir())
        lines = [f"{p}:"]
        for e in entries[:60]:
            tag = "/" if e.is_dir() else f"  {e.stat().st_size}B"
            lines.append(f"  {e.name}{tag}")
        return "\n".join(lines)

    def _tool_load_data(self, path: str) -> str:
        from ledgerhound.io import data_source as _ds
        fetched = None
        if _ds.is_s3(path):
            # Object store: one parallel bulk copy into the cache (RAM disk
            # on a pod), then everything below reads it like a local file.
            try:
                fetched = self._fetch_from_store(_ds.parse(path))
            except _ds.DataSourceError as e:
                self._events.append("load_data_failed", path=path, error=e.message)
                return f"could not load {path}: {e}"
            resolved = fetched.local_path
        else:
            resolved = self._resolve_data(path)
        if isinstance(resolved, list):  # a folder with choices
            if len(resolved) == 1:
                resolved = resolved[0]
            else:
                listing = "\n".join(f"  - {r}" for r in resolved[:40])
                return (f"'{path}' is a folder with multiple data files; "
                        f"which one?\n{listing}")
        if resolved is None:
            return (f"could not find data at '{path}'. Tried it as-given, "
                    f"./, ../, cwd and project root. Use list_directory to look around.")
        # In kernel mode the kernel loads the real data itself (cuDF on GPU).
        # Reading the WHOLE file into this process too was a pure duplicate:
        # full CPU RAM cost, paid unconditionally, BEFORE the compute router
        # was even consulted -- so "use the GPU" never actually avoided it,
        # and at 8-crore scale it risks OOM-ing the shell before any analysis
        # starts (audit finding, 2026-08-06). Above the threshold we read a
        # bounded SCHEMA sample only; the kernel still owns the full data.
        df = None
        sampled_from = None
        rows_exact = True
        if self.use_kernel:
            est, est_exact = _estimate_rows(resolved)
            if est is not None and est > _SHELL_LOAD_MAX_ROWS:
                df = _read_sample(resolved, _SHELL_SCHEMA_SAMPLE_ROWS)
                if df is not None:
                    sampled_from = est
                    rows_exact = est_exact
        if df is None:
            try:
                df = _read_any(resolved)
            except Exception as e:  # noqa: BLE001
                return f"found {resolved} but failed to load it: {e}"
        self._df = df
        # True row count for the schema pin / event log, so a sampled load
        # never misreports the dataset's real size.
        self._df_is_sample = sampled_from is not None
        self._true_n_rows = int(sampled_from) if sampled_from else len(df)
        self._true_rows_exact = rows_exact
        # V3-017: fresh baseline for the post-mutation delta check -- a new
        # load_data always starts a new "before" state, never diffs across
        # datasets.
        from ledgerhound.shell.verification import fingerprint as _fp
        self._last_fp = _fp(len(df), list(df.columns))
        self._last_null_rates = None
        self._undo_snapshot = None   # V3-016: fresh data -- no undo history
        # V3-012/V3-013: fingerprint the data file so `ledgerhound replay` can
        # detect if it changed since this session ran.
        try:
            if fetched is not None:
                # From the store's ETags: identifies the content without
                # re-reading GBs that were just downloaded.
                data_hash = fetched.content_id
            elif Path(resolved).is_dir():
                data_hash = _sha256_dataset_dir(Path(resolved))
            else:
                from ledgerhound.runtime.events import sha256_file
                data_hash = sha256_file(resolved)
        except Exception:
            data_hash = ""
        # V3-053: the Librarian keys stories off this. Stored on self (not
        # just the event payload) so /library, open_chapter, etc. can read
        # "what data version is loaded right now" without re-hashing.
        self._data_sha256 = data_hash
        self._events.append("load_data", path=str(resolved),
                            source=(fetched.source.uri if fetched else "local"),
                            rows=self._true_n_rows,
                            rows_exact=self._true_rows_exact,
                            cols=int(df.shape[1]),
                            shell_sampled=self._df_is_sample,
                            data_sha256=data_hash)
        # Outputs follow the source: decide features_dir (and the upload
        # target for store data) BEFORE the kernel starts, since the kernel
        # is handed features_dir at start.
        paths_note = self._set_run_paths(path, resolved, fetched)
        self._ns = None  # reset sandbox namespace for the new data
        self._digest = None  # force a fresh profile for the new data
        self._digest_summary = ""
        self._digest_from_card = False
        from ledgerhound.knowledge.team_kb import dataset_id_for
        self._dataset_id = dataset_id_for(
            name=self._dataset_name_override,
            s3_uri=fetched.source.uri if fetched is not None else "",
            content_sha256=data_hash)
        self._load_card_for(resolved, self._dataset_id, data_hash)  # V3-011: may pre-seed self._digest
        self._refresh_schema_pin()  # visible to the rest of THIS turn, not just the next
        kernel_note = self._start_kernel_for(resolved) if self.use_kernel else ""
        cols = ", ".join(map(str, df.columns[:40]))
        n_disp = self._rows_display()
        msg = (f"loaded {resolved} -> df with {n_disp} rows x "
               f"{df.shape[1]} columns.\ncolumns: {cols}")
        if fetched is not None:
            msg = f"from the object store: {fetched.summary()}\n" + msg
            for n in fetched.notes:
                msg += f"\n({n})"
        if paths_note:
            msg += f"\n{paths_note}"
        if self._df_is_sample:
            msg += (f"\n(large dataset: the kernel holds all {n_disp} rows -- "
                    f"your run_python code sees the FULL data. This process "
                    f"kept only a {len(df):,}-row schema sample, so "
                    f"profile_data's summary stats are estimates from that "
                    f"sample; compute exact figures with run_python.)")
        if kernel_note:
            msg += f"\n{kernel_note}"
        # Recognise the dataset from its column names alone and PROPOSE the
        # goal immediately. The 2026-08-06 run had to rediscover "this is an
        # account-level mule problem" over hours, then shipped the wrong
        # grain anyway -- the schema said so from the first second.
        recognised = self._recognise_dataset(df.columns)
        if recognised:
            msg += f"\n\n{recognised}"
        chosen = self._choose_methodology_once()
        if chosen:
            msg += f"\n\n{chosen}"
        prior = self._recall_books(f"{Path(resolved).name} {cols}")
        if prior:
            msg += "\n\n" + prior
        return msg

    # -- team knowledge base (automatic; see ledgerhound.knowledge.team_kb) ------
    _kb = None
    _team_findings: list = []
    _dataset_id: str = ""
    _dataset_name_override: str = ""

    def team_kb(self):
        """The shared knowledge base, or None without a project (the
        adversarial debate agents and ad-hoc sessions have none)."""
        if self.project is None:
            return None
        if self._kb is None:
            try:
                from ledgerhound.knowledge.team_kb import TeamKB, enabled
                if not enabled():
                    return None
                self._kb = TeamKB()
                self._kb._kick()        # upload anything queued by an earlier session
            except Exception:  # noqa: BLE001 -- the KB is never allowed to break a session
                self._kb = None
        return self._kb

    def _kb_fetch_card(self, dataset_id: str):
        kb = self.team_kb()
        if kb is None or not dataset_id:
            return None
        try:
            return kb.fetch_card(dataset_id)
        except Exception:  # noqa: BLE001
            return None

    def _kb_publish_card(self) -> None:
        kb = self.team_kb()
        if kb is None or self._card is None or not getattr(self._card, "dataset_id", ""):
            return
        try:
            kb.publish_card(self._card)
        except Exception as e:  # noqa: BLE001
            self._events.append("team_kb_error", op="publish_card", error=str(e)[:300])

    def _kb_pull_extras(self, dataset_id: str) -> None:
        """Team findings (for the 'already known' check) and books (for
        recall) recorded on this dataset by anyone."""
        kb = self.team_kb()
        if kb is None or not dataset_id:
            return
        try:
            self._team_findings = kb.fetch_findings(dataset_id)
            from ledgerhound.analyst.books import BOOKS_DIR
            books_dir = Path(self.project.dotdir) / BOOKS_DIR
            for b in kb.fetch_books(dataset_id):
                bid = str(b.get("id") or "")
                if not bid or (books_dir / f"{bid}.json").exists():
                    continue
                from ledgerhound.analyst.books import Book
                b = {k: v for k, v in b.items() if not k.startswith("_") and k != "dataset_id"}
                book = Book.from_dict(b)
                books_dir.mkdir(parents=True, exist_ok=True)
                (books_dir / f"{bid}.json").write_text(json.dumps(book.to_dict(), indent=2),
                                                       encoding="utf-8")
                (books_dir / f"{bid}.md").write_text(book.to_markdown(), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self._events.append("team_kb_error", op="pull", error=str(e)[:300])

    def _save_card(self) -> None:
        """Save the dataset card locally, then share it (every card change
        -- a digest, a decision, a chapter page, an answer -- reaches the
        team automatically)."""
        from ledgerhound.knowledge.dataset_card import DatasetCardStore
        DatasetCardStore(self.project).save(self._card)
        self._kb_publish_card()

    def _load_card_for(self, resolved: Path, dataset_id: str = "",
                       data_sha256: str = "") -> None:
        """Load (or create) this dataset's durable card (V3-011). If a prior
        session already profiled this exact file, pre-seed the digest cache
        so profile_data does not recompute it. Requires a project --
        persistence across sessions is the entire point; with none, cards
        are skipped and behaviour is unchanged from before V3-011."""
        self._card = None
        self._team_findings = []
        if self.project is None:
            return
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore, merge_cards
            store = DatasetCardStore(self.project)
            card = store.load_or_create(resolved)
            had_digest = bool(card.digest)
            if dataset_id:
                card.dataset_id = dataset_id
            # Team knowledge base: fold in what everyone has recorded on this
            # dataset (stories, decisions, findings, answers to data
            # questions). Store unreachable -> local card only, never an error.
            remote = self._kb_fetch_card(dataset_id)
            if remote is not None:
                remote_digest_ok = (not data_sha256 or remote.data_sha256 == data_sha256)
                card = merge_cards(remote, card)
                if not had_digest and not remote_digest_ok:
                    card.digest = {}          # a teammate's profile of OTHER content
            card.add_run(self.run_dir.name)
            store.save(card)
            self._card = card
            self._kb_publish_card()
            self._kb_pull_extras(dataset_id)
            if card.digest:
                self._digest = card.digest
                self._digest_from_card = True
                try:
                    from ledgerhound.profiler.digest import Digest
                    self._digest_summary = Digest(**card.digest).summary()[:4000]
                except Exception:
                    self._digest_summary = ""
                    self._digest_from_card = False
        except Exception:  # noqa: BLE001
            self._card = None

    def _resolve_compute_engine(self, resolved: Path) -> str:
        """Which dataframe engine the kernel should use:
        "dask_cudf" | "cudf" | "pandas".

        Asks the SAME router the digest path already uses
        (ledgerhound.backends.compute_router), so the shell and the profiler
        agree on the machine instead of each deciding separately. The router
        accounts for GPU count, data size vs GPU memory, and whether a LOCAL
        LLM is occupying a GPU (a remote API like DeepSeek/gateway Kimi leaves
        every GPU free for cuDF).

        Advisory only: if the router raises, or cuDF turns out to be broken
        at kernel start, KernelExecutor falls back to pandas by itself. There
        is no path here that can make a GPU-less machine behave differently
        than it does today.
        """
        try:
            from ledgerhound.backends.compute_router import ComputeRouter
            router = ComputeRouter()
            # `llm` may be a test stub / any object -- never assume .config.
            base_url = getattr(getattr(self.llm, "config", None), "base_url", "")
            profile = router.detect(
                str(resolved), llm_is_local=router.infer_llm_is_local(base_url))
            # Pass the router's choice STRAIGHT through. The previous line
            # was `"cudf" if choice == "cudf" else "pandas"`, which silently
            # rewrote a `dask_cudf` decision -- the multi-GPU path V3-038
            # added -- into CPU pandas. KernelExecutor has handled
            # "dask_cudf" since V3-038 but nothing ever passed it, so the
            # whole branch was unreachable and a 3xH100 box kept landing on
            # CPU no matter what the router worked out. Anything unexpected
            # still degrades to pandas.
            if profile.backend_choice in ("cudf", "dask_cudf"):
                return profile.backend_choice
            return "pandas"
        except Exception:  # noqa: BLE001 -- routing is advisory, never fatal
            return "pandas"

    def _record_gpu_fallback(self, stage: str, error_text: str,
                             fell_back_to: str, remedy: str = "",
                             error_class: str = "",
                             rmm_status: str = "") -> None:
        """Log a GPU->CPU downgrade and append it to the run's errornote.

        V3-047. Two destinations on purpose: `gpu_fallback` in the
        hash-chained event stream so it is replayable and auditable, and
        `gpu_fallback_rca.md` in the run folder so a human opening the
        output directory finds out why the run was slow without grepping
        JSON. Never raises -- an RCA that fails must not take the session
        with it.
        """
        if not error_class or not remedy:
            from ledgerhound.backends.compute_router import classify_gpu_error
            cls, rem = classify_gpu_error(error_text, rmm_status)
            error_class = error_class or cls
            remedy = remedy or rem
        try:
            self._events.append("gpu_fallback", stage=stage,
                                error_class=error_class,
                                error_text=(error_text or "")[:2000],
                                remedy=remedy, fell_back_to=fell_back_to,
                                rmm_status=rmm_status)
        except Exception:  # noqa: BLE001
            pass
        try:
            path = self.run_dir / "gpu_fallback_rca.md"
            new = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    f.write("# Why this run did not use the GPU\n\n"
                            "Each entry is one point where the GPU path was "
                            "abandoned, with the cause and the fix. Written "
                            "as it happened.\n")
                f.write(f"\n## {stage}\n\n"
                        f"- **Fell back to:** {fell_back_to}\n"
                        f"- **Cause:** `{error_class}`\n"
                        f"- **Fix:** {remedy}\n")
                if rmm_status:
                    f.write(f"- **RMM at the time:** {rmm_status}\n")
                if error_text:
                    f.write(f"\n```\n{(error_text or '')[:2000].strip()}\n```\n")
        except Exception:  # noqa: BLE001
            pass

    def _start_kernel_for(self, resolved: Path) -> str:
        """(Re)start the persistent IPython kernel for newly-loaded data
        (V3-009). Best-effort: any failure (missing jupyter_client/ipykernel,
        no working kernelspec, sandboxed subprocess restrictions) falls back
        to the local exec sandbox instead of breaking the load."""
        if self._kernel is not None:
            try:
                self._kernel.shutdown()
            except Exception:
                pass
            self._kernel = None
        self._kernel_active = False
        try:
            from ledgerhound.auto.kernel import KernelExecutor, kernels_available
            if not kernels_available():
                # The exec sandbox is pandas-only -- it has NO cuDF branch --
                # so this is a total GPU loss, not a degraded mode (V3-047).
                self._record_gpu_fallback(
                    "kernel_unavailable",
                    "jupyter_client/ipykernel not installed",
                    "CPU exec sandbox",
                    "The exec sandbox has no GPU code path at all, so no "
                    "GPU can be used in this session regardless of hardware. "
                    "Install the kernel dependencies: pip install "
                    "jupyter_client ipykernel")
                return ("(kernel mode requested but jupyter_client/ipykernel "
                        "aren't installed -- using the restricted exec sandbox "
                        "instead; NO GPU is reachable this way, see "
                        "gpu_fallback_rca.md)")
            engine = self._resolve_compute_engine(resolved)
            self._loaded_path = str(resolved)
            # I-10: the per-cell timeout used to be a hardcoded 120s. On a
            # crore-scale frame a single groupby on the CPU fallback takes
            # longer than that, the cell was interrupted, and the identical-
            # failure nudge then pushed the model toward shortcuts -- the
            # opposite of "hours are fine". Configurable via main_config's
            # kernel.cell_timeout_seconds (env LEDGERHOUND_CELL_TIMEOUT);
            # default 30 min, a runaway-cell net rather than a pace-setter.
            from ledgerhound.config import get_int as _get_int
            cell_timeout = _get_int("kernel", "cell_timeout_seconds",
                                    env="LEDGERHOUND_CELL_TIMEOUT", default=1800)
            self._kernel = KernelExecutor(
                scratch_dir=self.run_dir / "scratch",
                data_path=str(resolved), engine=engine,
                timeout=max(30, cell_timeout),
                features_dir=self._features_dir())
            self._kernel_active = True
            # Drain whatever the kernel downgraded during startup into the
            # canonical log. KernelExecutor has no EventStream of its own.
            for note in getattr(self._kernel, "gpu_fallbacks", []):
                self._record_gpu_fallback(
                    note["stage"], note["error_text"], note["fell_back_to"],
                    note["remedy"], error_class=note["error_class"],
                    rmm_status=note.get("rmm_status", ""))
            # Define check_leakage/evaluate_feature/evaluate_features INSIDE
            # the kernel. Without this they exist only in the exec sandbox's
            # namespace, so in kernel mode the model follows the system
            # prompt's mandate to call them and gets a bare NameError (audit
            # finding, 2026-08-06 -- see ledgerhound/shell/kernel_helpers.py).
            # Inject, then VERIFY by asking the kernel whether the names
            # actually exist -- the 2026-08-06 production run showed the
            # "failed to load" warning on 2 of 5 kernel starts, i.e.
            # _exec_raw's success flag alone is not reliable here (a slow
            # first start races the 60s timeout). Retry once, longer, and
            # confirm by introspection rather than by exit status.
            helper_note = ""
            try:
                from ledgerhound.shell.kernel_helpers import ANALYST_HELPERS_SNIPPET

                def _helpers_present() -> bool:
                    chk = self._kernel._exec_raw(
                        "print('HELPERS_OK' if all(n in dir() for n in "
                        "('check_leakage','evaluate_feature','evaluate_features')) "
                        "else 'HELPERS_MISSING')", timeout=60)
                    return "HELPERS_OK" in (getattr(chk, "output", "") or "")

                self._kernel._exec_raw(ANALYST_HELPERS_SNIPPET, timeout=180)
                if not _helpers_present():
                    self._kernel._exec_raw(ANALYST_HELPERS_SNIPPET, timeout=300)
                    if not _helpers_present():
                        helper_note = (
                            " -- WARNING: analyst helpers "
                            "(evaluate_feature/check_leakage) failed to load; "
                            "evaluate features with an explicit IV/AUC "
                            "computation instead of assuming")
            except Exception:  # noqa: BLE001 -- never break a working kernel
                helper_note = (" -- WARNING: analyst helpers "
                               "(evaluate_feature/check_leakage) unavailable")
            # The kernel may still downgrade cudf->pandas internally if the
            # CUDA context fails to initialise; report what it ACTUALLY ended
            # up with, not what we asked for.
            actual = getattr(self._kernel, "_engine", engine)
            gpu_note = (" -- data engine: cuDF (GPU)" if actual == "cudf"
                        else " -- data engine: pandas (CPU)")
            return ("(real Python kernel started -- full language, "
                    f"no restrictions{gpu_note}{helper_note})")
        except Exception as e:  # noqa: BLE001
            self._kernel = None
            self._kernel_active = False
            return (f"(kernel start failed ({type(e).__name__}: {e}) -- "
                    f"using the restricted exec sandbox instead)")

    def _tool_profile_data(self) -> str:
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        # Already profiled -- either earlier this session, or (V3-011) in a
        # PRIOR session on this exact file via the dataset card -- return the
        # cached digest instead of re-running the full profile (and
        # re-detecting GPUs) every time. Deterministic for fixed data;
        # recomputing wastes time and clutters the terminal.
        if self._digest is not None and self._digest_summary:
            if self._digest_from_card:
                note = ("(already profiled in a prior session on this exact "
                        "file -- returning the saved digest from its dataset "
                        "card; call again with fresh data if it has changed)")
            else:
                note = ("(already profiled -- returning the cached digest; the "
                        "data has not changed this session)")
            return note + "\n\n" + self._digest_summary + self._data_questions_section()
        summary = ""
        try:
            from ledgerhound.core import Ledgerhound
            # Profile the frame we ALREADY hold, instead of handing Ledgerhound
            # the path and making it read the same file a second time (this
            # session loaded it once in _read_any). The re-read bought
            # nothing: on a GPU box Ledgerhound's cuDF branch reads the file
            # with cudf and then immediately calls .to_pandas() before
            # profiling, so the digest was computed on CPU either way -- the
            # duplicate read was pure cost. `source=` keeps the digest's
            # provenance pointing at the real file, not the frame.
            eng = Ledgerhound(data=self._df, source=str(self._loaded_path or ""))
            digest = eng.profile()
            self._digest = digest.to_dict()
            summary = digest.summary()[:4000]
        except Exception:
            d = self._df
            miss = (d.isna().mean() * 100).round(1)
            top = miss.sort_values(ascending=False).head(8)
            self._digest = {"rows": len(d), "cols": int(d.shape[1])}
            summary = (f"{len(d):,} rows x {d.shape[1]} cols. dtypes: "
                       f"{d.dtypes.value_counts().to_dict()}. "
                       f"most-missing %: {top.to_dict()}")
        # A sampled shell-side frame must NEVER have its stats presented as
        # whole-dataset facts -- that would be a correctness bug worse than
        # the memory cost this sampling avoids. Label it loudly, on both the
        # text the model reads and the persisted digest.
        if self._df_is_sample:
            n_s = len(self._df) if self._df is not None else 0
            n_disp = self._rows_display()
            summary = (f"[SAMPLE-BASED ESTIMATE -- computed on {n_s:,} of "
                       f"{n_disp} rows, because the full dataset lives in the "
                       f"kernel. Distributions/rates below are ESTIMATES, not "
                       f"population values. Recompute anything you intend to "
                       f"report with run_python, which sees all {n_disp} "
                       f"rows.]\n\n" + summary)
            if isinstance(self._digest, dict):
                self._digest["sampled"] = True
                self._digest["sample_rows"] = n_s
                self._digest["true_rows"] = self._true_n_rows
        self._digest_summary = summary
        self._digest_from_card = False
        # Persist the digest into the run folder so the session is auditable /
        # reproducible (one digest, one place, tied to this run).
        try:
            (self.run_dir / "digest.json").write_text(
                json.dumps(self._digest, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        # V3-011: also persist onto the dataset card so the NEXT session on
        # this exact file skips re-profiling entirely. NOT done for a
        # sample-based digest -- a future session would load it as a full
        # profile and silently inherit sample estimates as population facts.
        if self._card is not None and not self._df_is_sample:
            try:
                from ledgerhound.knowledge.dataset_card import DatasetCardStore
                self._card.set_digest(self._digest)
                self._save_card()
            except Exception:
                pass
        return summary + self._data_questions_section()

    # -- questions about the data (asked, never assumed) --------------------
    _open_data_questions: list = []
    _questions_reminded: bool = False

    def _answered_questions(self) -> dict:
        if self._card is None:
            return dict(getattr(self, "_session_answers", {}) or {})
        return {**(getattr(self, "_session_answers", {}) or {}),
                **self._card.current_clarifications()}

    def _data_questions_section(self) -> str:
        """Known answers + open questions about the loaded data, appended to
        the profile. The model is told to ask the open ones before any
        analysis (system prompt, 'Questions about the data')."""
        if self._df is None:
            return ""
        try:
            from ledgerhound.shell.data_questions import open_questions, questions_block
            known = self._answered_questions()
            qs = open_questions(self._df, set(known))
            self._open_data_questions = qs
            self._questions_reminded = False
            if qs:
                self._events.append("data_questions", keys=[q.key for q in qs])
            block = questions_block(qs, known)
            if block and self._df_is_sample:
                block += ("\n(class counts above come from the schema sample, "
                          "not the full data.)")
            return f"\n\n{block}" if block else ""
        except Exception:  # noqa: BLE001 -- advisory
            return ""

    def _tool_record_data_answer(self, key: str, answer: str, question: str = "") -> str:
        """Save the user's answer to a data question on the dataset card
        (shared with the team, so it is never asked again for this data)."""
        key, answer = (key or "").strip(), (answer or "").strip()
        if not key or not answer:
            return "record_data_answer needs the question's key and the user's answer."
        q = next((x for x in self._open_data_questions if x.key == key), None)
        question = question or (q.text if q else "")
        from ledgerhound.identity import current_user
        who = current_user()
        self._events.append("data_answer", key=key, answer=answer[:2000])
        self._open_data_questions = [x for x in self._open_data_questions if x.key != key]
        if self._card is None or self.autonomous:
            by = who if not self.autonomous else "autonomous run -- UNCONFIRMED"
            self._session_answers = {**(getattr(self, "_session_answers", {}) or {}),
                                     key: {"key": key, "answer": answer, "by": by}}
            why = ("no human confirmed it in this unattended run, so it is NOT saved "
                   "to the shared dataset card -- state it as an assumption in the report"
                   if self.autonomous else "no project, so it is not saved to a dataset card")
            return f"noted [{key}]: {answer} (this session only -- {why})."

        try:
            self._card.add_clarification(key, question, answer, by=who)
            self._save_card()
        except Exception as e:  # noqa: BLE001
            return f"could not save the answer: {e}"
        left = len(self._open_data_questions)
        return (f"recorded [{key}] on the dataset card and shared with the team."
                + (f" {left} question(s) still open." if left else " All data questions answered."))

    def _tool_run_python(self, code: str) -> str:
        out = self._run_python_step(code)
        note = self._sync_outputs()
        out = f"{out}\n{note}" if note else out
        if self._open_data_questions and not self._questions_reminded:
            self._questions_reminded = True
            keys = ", ".join(q.key for q in self._open_data_questions[:8])
            out += ("\n[DATA QUESTIONS STILL OPEN: " + keys + " -- ask the user and "
                    "record_data_answer before drawing any conclusion that depends on them. "
                    "Do not assume.]")
        return out

    def _run_python_step(self, code: str) -> str:
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        # GOAL GATES (blocking, not advisory). The 2026-08-06 production run
        # showed an advisory grain warning being acknowledged in a code
        # comment and then ignored ("This is intentional"), and hours spent
        # training models that were never authorised. These REFUSE.
        refusal = self.goal.check_modeling(code, methodology=self.methodology)
        if refusal:
            self._events.append("goal_gate", kind="modeling_unauthorised",
                                code=code[:500])
            return refusal
        refusal = self._precheck_export_grain(code)
        if refusal:
            self._events.append("goal_gate", kind="grain_violation",
                                code=code[:500])
            return refusal
        # V5-003: the first check in this chain that looks at INFERENCE rather
        # than data. Static, so it only catches misuse visible in the code --
        # but that is the class Fisher-R1 found agents making: valid code,
        # invalid conclusion.
        refusal = self._check_test_assumptions(code)
        if refusal:
            self._events.append("goal_gate", kind="invalid_test",
                                code=code[:500])
            return refusal
        if self._kernel_active and self._kernel is not None:
            return self._run_python_kernel(code)
        low = code.lower()
        for bad in _BLOCKED:
            if bad in low:
                return f"blocked: '{bad}' is not allowed in the sandbox (df/pd/np are in scope)."
        if self._ns is None:
            import numpy as np

            from ledgerhound.shell.feature_strength import (
                make_evaluate_feature_helper,
                make_evaluate_features_helper,
                make_percentile_table_helper,
                make_separation_callout_helper,
            )
            from ledgerhound.shell.leakage_checks import make_check_leakage_helper
            # V3-049: exec mode had NO chdir and no scratch/features_dir of
            # its own, so any relative df.to_csv(...)/plt.savefig(...) in
            # generated code landed wherever the process happened to be
            # launched from -- confirmed on disk as a stray `features.csv`
            # at the repo root. scratch/features_dir give the model
            # somewhere explicit to write; the chdir bracket around exec()
            # below (not here) makes relative paths land there too.
            scratch_dir = self.run_dir / "scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            from ledgerhound.shell.write_guard import link_features_into
            link_features_into(scratch_dir, self._features_dir())
            self._ns = {"pd": pd, "np": np, "df": self._df,
                       "scratch": scratch_dir,
                       "features_dir": self._features_dir(),
                       # V3-050: the GPU-awareness section of system.md is
                       # unconditional and tells the model to use `_xdf`/
                       # `to_pdf`/`safe_*` -- names that only really exist
                       # in kernel mode. In exec mode (no kernel) those were
                       # a guaranteed NameError. `_xdf` is pandas here (this
                       # IS the CPU path) and the safe helpers are no-ops/
                       # identity, so the same generated code works whether
                       # or not this session has a GPU kernel behind it.
                       "_xdf": pd,
                       "to_pdf": lambda x: x,
                       "safe_to_datetime": pd.to_datetime,
                       "safe_merge": pd.merge,
                       "safe_to_parquet": lambda d, path, **kw: d.to_parquet(path, **kw),
                       "check_leakage": make_check_leakage_helper(),
                       "evaluate_feature": make_evaluate_feature_helper(),
                       "evaluate_features": make_evaluate_features_helper(),
                       "percentile_table": make_percentile_table_helper(),
                       "separation_callout": make_separation_callout_helper()}
            # cached_agg / sample_frame -- same toolkit the kernel gets, so
            # the two execution paths expose the same helper surface.
            try:
                from ledgerhound.shell.workbench import WORKBENCH_SNIPPET
                exec(WORKBENCH_SNIPPET, self._ns)
            except Exception:  # noqa: BLE001 -- helpers are a bonus, not a gate
                pass
        buf = io.StringIO()
        # V3-049: bracket exec() with a chdir into this run's scratch dir so
        # a bare relative `df.to_csv("out.csv")` lands under the run folder
        # instead of wherever the shell process happened to be launched
        # from. Single-threaded REPL, one cell at a time -- safe to mutate
        # process CWD for the duration of one exec() call.
        cwd_before = os.getcwd()
        scratch_dir = self._ns.get("scratch") or (self.run_dir / "scratch")
        features_dir_ns = self._ns.get("features_dir") or scratch_dir
        try:
            os.chdir(scratch_dir)
        except OSError:
            pass
        # V3-063: chdir only protects a BARE RELATIVE write -- an absolute
        # path (or one built from the loaded data file's own directory,
        # e.g. os.path.join(os.path.dirname(data_path), "out.csv")) sails
        # straight through it. Confirmed in production: feature tables
        # landing in the ORIGINAL DATA FOLDER. Bracket per-cell, not
        # installed for the process lifetime -- exec mode shares this
        # interpreter with the rest of Ledgerhound, so a global monkeypatch
        # left in place would affect the host app's own pandas writes too.
        from ledgerhound.shell.write_guard import install as _install_write_guard

        def _wg_notice(orig: str, redirected: str) -> None:
            print(f"WRITE REDIRECTED: '{orig}' is outside the workspace "
                 f"-> '{redirected}'")
        _uninstall_wg = _install_write_guard(
            lambda: [scratch_dir, features_dir_ns],
            lambda: features_dir_ns, _wg_notice)
        try:
            try:
                with redirect_stdout(buf):
                    exec(code, self._ns)  # single namespace -> vars persist, funcs see pd/np
            except Exception as e:  # noqa: BLE001
                self._events.append("code_exec", code=code, executor="exec", success=False,
                                    error=f"{type(e).__name__}: {e}")
                return self._teach_error(e, buf.getvalue())
        finally:
            _uninstall_wg()
            try:
                os.chdir(cwd_before)
            except OSError:
                pass
        out = buf.getvalue().strip()
        res = self._ns.get("result")
        if not out and res is not None:
            out = str(res)
        # V3-017: `df` may have been reassigned in the sandbox namespace
        # (e.g. `df = df.dropna()`) -- resync self._df so the schema pin and
        # error-hint columns don't go stale, and check for a mutation delta
        # to surface to the model. Cheap (rows+columns) always; a null-rate
        # scan only fires when that already showed a real change.
        new_df = self._ns.get("df")
        delta_block = ""
        if new_df is not None and hasattr(new_df, "columns"):
            self._df = new_df
            delta_block = self._check_mutation_delta(new_df, code)
        # Data-comparability check on whatever frames the cell actually
        # created. Runs here, not at load_data, because code that reads its
        # own files (pd.read_csv inside run_python) never touches load_data
        # -- which is exactly how a 1-month-vs-1-day fraud/non-fraud pair
        # reached feature engineering unflagged.
        check_block = self._check_data_comparability(code)
        leakage_block = self._check_leakage(code)
        grain_block = self._check_output_grain(code)
        numeric_block = self._check_numeric_stability(code)
        ambiguity_block = self._check_ambiguous_columns()
        memory_block = self._check_memory_usage()
        # V3-012: session_code.py is now a projection of the event stream --
        # append the canonical code_exec event, then write the SAME view
        # function `ledgerhound run-views` uses for bulk regeneration, so the
        # live file and a regenerated one can never drift apart.
        self._log_code_exec(code, out, executor="exec")
        # V5-005: same 3500-char budget as before, but cut from the MIDDLE
        # instead of the tail. A bare out[:3500] drops the end of the output,
        # which on a DataFrame repr is where the shape summary lives
        # ("[1000 rows x 40 columns]") -- the single most useful line, and
        # the one the model needs to know it is looking at a truncated view.
        # The full text spills to the run folder either way.
        text = self._compact_cell_output(out)
        for block in (delta_block, check_block, leakage_block, grain_block,
                     numeric_block, ambiguity_block, memory_block,
                     getattr(self, "_stat_advisory", "")):
            if block:
                text = f"{text}\n\n{block}" if text else block
        return text if text.strip() else "(ran; no output -- print() what you want to see)"

    # Cell-output budget. Unchanged from the bare out[:3500] this replaced --
    # what changed is WHERE the cut falls.
    _CELL_OUTPUT_CHARS = 3500

    def _compact_cell_output(self, out: str) -> str:
        """V5-005: hold a cell's output to the budget without losing its end."""
        if not out:
            return ""
        try:
            from ledgerhound.shell.intra_compaction import compact_result
            outcome = compact_result(
                out, spill_dir=self.run_dir / "results",
                max_chars=self._CELL_OUTPUT_CHARS,
                head_chars=int(self._CELL_OUTPUT_CHARS * 0.68),
                tail_chars=int(self._CELL_OUTPUT_CHARS * 0.32))
            if outcome.compacted:
                self._events.append("intra_compaction", tool="run_python",
                                    original_chars=outcome.original_chars,
                                    saved_chars=outcome.saved_chars,
                                    spill_path=outcome.spill_path)
            return outcome.text
        except Exception:  # noqa: BLE001
            return out[:self._CELL_OUTPUT_CHARS]

    def _tool_propose_falsification(self, claim: str) -> str:
        """V5-004. The attack plan for a claim, not more support for it."""
        from ledgerhound.findings.falsification import plan_for
        plan = plan_for(claim)
        self._events.append("falsification_planned", claim=claim[:500],
                            routes=[r[0] for r in plan.routes])
        chapter = getattr(self, "_current_chapter", "")
        out = plan.render()
        if chapter != "falsification":
            out += ("\n\nNOTE: you are not in the 'falsification' chapter. "
                    "Open it (open_chapter('falsification')) so the attack "
                    "and its outcome are recorded as their own approach "
                    "rather than folded into the chapter that produced the "
                    "claim.")
        return out

    def _check_test_assumptions(self, code: str) -> str:
        """V5-003. Returns a REFUSAL string for a test that cannot be valid as
        written, an advisory block for one that merely needs care, or "".

        The advisory is stashed on the instance and appended to the cell
        output by the caller, so a cell that runs still carries the caveat
        with the numbers it produced rather than in a separate turn.
        """
        self._stat_advisory = ""
        try:
            from ledgerhound.shell.stat_checks import (
                check_statistical_assumptions,
                is_blocking,
                render,
            )
            n_rows = None
            if self._df is not None and not self._df_is_sample:
                try:
                    n_rows = int(len(self._df))
                except Exception:  # noqa: BLE001
                    n_rows = None
            issues = check_statistical_assumptions(
                code, n_rows=n_rows, seen=self._stat_issues_seen)
            if not issues:
                return ""
            block = render(issues)
            for i in issues:
                self._events.append("stat_check", test=i.test, kind=i.kind,
                                    blocking=i.blocking, detail=i.message[:1000])
            if is_blocking(issues):
                return block
            self._stat_advisory = block
            return ""
        except Exception:  # noqa: BLE001 -- a check must never break a cell
            return ""

    def _check_data_comparability(self, code: str = "") -> str:
        """Mechanical time-window comparability check over the sandbox's
        frames (see ledgerhound.shell.data_checks). Best-effort: a failure here
        must never break a cell that otherwise ran fine. `code` lets
        unchanged frames skip their full-column scans (J-02)."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.data_checks import check_frames
            block = check_frames(self._ns, self._data_issues_seen, code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="time_window_mismatch",
                                detail=block[:2000])
        return block

    def _check_leakage(self, code: str = "") -> str:
        """Mechanical leakage check over the sandbox's frames (V3-032, see
        ledgerhound.shell.leakage_checks). Best-effort: a failure here must
        never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.leakage_checks import check_leakage
            block = check_leakage(self._ns, self._leakage_issues_seen, code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="target_leakage",
                                detail=block[:2000])
        return block

    def _check_output_grain(self, code: str) -> str:
        """Mechanical output-grain check, gated on an export call in `code`
        (V3-032 Tier 4, see ledgerhound.shell.grain_checks). Best-effort: a
        failure here must never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.grain_checks import check_output_grain
            block = check_output_grain(code, self._ns, self._grain_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="output_grain",
                                detail=block[:2000])
        return block

    def _check_numeric_stability(self, code: str = "") -> str:
        """Mechanical inf/-inf check over the sandbox's frames (V3-032, see
        ledgerhound.shell.numeric_checks). Best-effort: a failure here must
        never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.numeric_checks import check_numeric_stability
            block = check_numeric_stability(self._ns, self._numeric_issues_seen,
                                            code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="numeric_stability",
                                detail=block[:2000])
        return block

    def _check_ambiguous_columns(self) -> str:
        """Mechanical operational/workflow-artifact-column check over the
        sandbox's frames (V3-032, see ledgerhound.shell.ambiguity_checks).
        Best-effort: a failure here must never break a cell that otherwise
        ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.ambiguity_checks import check_ambiguous_columns
            block = check_ambiguous_columns(self._ns, self._ambiguity_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="ambiguous_column",
                                detail=block[:2000])
        return block

    def _check_memory_usage(self) -> str:
        """Mechanical redundant-large-copy check over the sandbox's frames
        (V3-034, see ledgerhound.shell.memory_checks). Best-effort: a failure
        here must never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.memory_checks import check_memory_usage
            block = check_memory_usage(self._ns, self._memory_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="redundant_memory_copy",
                                detail=block[:2000])
        return block

    def _check_mutation_delta(self, df, code: str) -> str:
        """V3-017: exec-mode delta check -- df is already in-process, so the
        null-rate scan (when needed) is a direct, synchronous computation."""
        from ledgerhound.shell.verification import fingerprint, fingerprints_differ, format_delta
        fp = fingerprint(len(df), list(df.columns))
        before = self._last_fp
        self._last_fp = fp
        if not fingerprints_differ(before, fp):
            return ""
        try:
            rates = {str(c): float(df[c].isna().mean()) for c in df.columns}
        except Exception:
            rates = {}
        prior_rates = self._last_null_rates
        self._last_null_rates = rates
        self._events.append("verification", code=code, before=before, after=fp,
                            null_rates=rates)
        return format_delta(before, fp, rates, prior_rates)

    def _log_code_exec(self, code: str, output: str, *, executor: str) -> None:
        """Record a successful code execution as a canonical event, and
        project it into session_code.py (see ledgerhound.runtime.views)."""
        # V5-002: the cheapest visible proxy for tests we cannot otherwise
        # see. Thirty cells against one recorded finding says most of what
        # was computed never reached the ledger -- which is exactly the
        # situation a multiplicity correction cannot account for on its own.
        try:
            self._effort.note_cell(getattr(self, "_current_chapter", ""))
        except Exception:  # noqa: BLE001
            pass
        import hashlib
        ev = self._events.append(
            "code_exec", code=code, executor=executor, success=True,
            output_preview=output[:2000],
            output_hash=hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
        )
        try:
            from ledgerhound.runtime.views import code_block
            with open(self._code_log, "a", encoding="utf-8") as f:
                f.write(code_block(ev))
        except Exception:
            pass

    def _run_python_kernel(self, code: str) -> str:
        """Execute in the real, persistent IPython kernel (V3-009) -- full
        Python, no `_BLOCKED` string-list (the kernel's own write-guard +
        process isolation cover that instead). Failures come back as a
        traceback string rather than a live exception, so we parse a
        (kind, msg) signature and reuse the same teaching-hint logic."""
        res = self._kernel.execute(code)
        if not res.success:
            kind, msg = _parse_traceback_signature(res.error or "")
            self._events.append("code_exec", code=code, executor="kernel", success=False,
                                error=f"{kind}: {msg}")
            return self._teach_error_text(kind, msg, res.output)
        out = (res.output or "").strip()
        # V3-017: the kernel's own execute() epilogue returns a cheap
        # (rows+columns) df_fingerprint on every call (see auto/kernel.py);
        # the null-rate scan is a SEPARATE, explicit kernel round-trip via
        # null_rates(), only paid for when that cheap check already showed
        # a change -- df itself never comes back to this process.
        delta_block = self._check_kernel_mutation_delta(res.df_fingerprint, code)
        memory_block = self._check_kernel_memory_usage(res.df_fingerprint)
        # The other four checks (leakage, numeric-stability, ambiguous-column,
        # output-grain) run against a compact JSON summary computed INSIDE the
        # kernel -- exec mode reads self._ns, which is empty here (audit
        # finding, 2026-08-06; see ledgerhound/shell/kernel_checks.py).
        kernel_blocks = self._run_kernel_checks(code)
        self._log_code_exec(code, out, executor="kernel")
        # V5-005: same 3500-char budget as before, but cut from the MIDDLE
        # instead of the tail. A bare out[:3500] drops the end of the output,
        # which on a DataFrame repr is where the shape summary lives
        # ("[1000 rows x 40 columns]") -- the single most useful line, and
        # the one the model needs to know it is looking at a truncated view.
        # The full text spills to the run folder either way.
        text = self._compact_cell_output(out)
        # V5-003's advisory is computed before dispatch, so it applies to both
        # execution paths -- a caveat that only appeared in exec mode would be
        # missing from exactly the GPU runs that matter most.
        for block in ([delta_block, memory_block] + kernel_blocks
                      + [getattr(self, "_stat_advisory", "")]):
            if block:
                text = f"{text}\n\n{block}" if text else block
        return text if text.strip() else "(ran; no output -- print() what you want to see)"

    def _recognise_dataset(self, columns) -> str:
        """Match the loaded schema against the domain profiles and, when one
        fits, pre-seed the session goal with its proposal.

        Deliberately does NOT overwrite a goal the user or model already
        declared -- a profile is a prior, not an authority. Best-effort: any
        failure returns "" so recognition can never break data loading."""
        try:
            from ledgerhound.context.domain_profiles import (
                describe,
                match_profile,
                resolve_entity_key,
                resolve_label_column,
            )
            prof = match_profile(list(columns))
            if not prof:
                return ""
            # Kept so record_finding can tell an already-established claim
            # from a new one (V3-048).
            self._domain_profile = prof
            block = describe(prof, list(columns))
            self._events.append("dataset_recognised", profile=prof.get("name"),
                                detail=block[:2000])
            if not self.goal.is_set():
                from ledgerhound.shell.session_goal import normalise_grain
                g = (prof.get("goal") or {})
                self.goal.objective = " ".join(str(g.get("objective", "")).split())
                self.goal.unit_of_analysis = normalise_grain(
                    g.get("unit_of_analysis", ""))
                self.goal.entity_key = resolve_entity_key(prof, list(columns))
                self.goal.label_column = resolve_label_column(prof, list(columns))
                self.goal.notes = f"proposed by domain profile '{prof.get('name')}'"
                self.goal.save(self.run_dir / "session_goal.json")
                block += ("\n(The session goal has been PRE-SET from this "
                          "profile and is now pinned. Change it with set_goal "
                          "if the user wants something different.)")
            return block
        except Exception:  # noqa: BLE001 -- recognition must never break a load
            return ""

    def _precheck_export_grain(self, code: str) -> str:
        """Check the grain of frames about to be exported BEFORE the write
        runs, so a wrong-grain deliverable is never produced at all.

        Only frames that already exist can be inspected -- one created and
        exported inside the same cell is caught by the post-execution check
        instead. Best-effort: any failure here returns "" (allow), because a
        broken gate must not block legitimate work."""
        if not self.goal.is_set() or self.goal.unit_of_analysis != "account":
            return ""
        try:
            from ledgerhound.shell.session_goal import _EXPORT_RE
            if not _EXPORT_RE.search(code):
                return ""
            frames: dict = {}
            if self._kernel_active and self._kernel is not None:
                from ledgerhound.shell.kernel_checks import collect_payload
                # force the (otherwise export-gated) duplicate scan
                frames = collect_payload(self._kernel, "to_csv(")
            elif self._ns:
                from ledgerhound.shell.data_checks import _is_frame
                from ledgerhound.shell.grain_checks import _entity_key_col
                for k, v in self._ns.items():
                    if str(k).startswith("_") or not _is_frame(v):
                        continue
                    col = self.goal.entity_key or _entity_key_col(v)
                    if not col or col not in getattr(v, "columns", []):
                        continue
                    n = len(v)
                    frames[str(k)] = {"n_rows": n,
                                      "dup_keys": {col: int(n - v[col].nunique())}}
            return self.goal.check_export(code, frames,
                                          methodology=self.methodology)
        except Exception:  # noqa: BLE001 -- a broken gate must not block work
            return ""

    def _run_kernel_checks(self, code: str) -> list[str]:
        """Kernel-mode counterpart to the four exec-mode checks that need
        real frame access. Best-effort: a failure here must never break a
        cell that otherwise ran fine."""
        if self._kernel is None:
            return []
        try:
            from ledgerhound.shell.kernel_checks import checks_from_payload, collect_payload
            payload = collect_payload(self._kernel, code)
            if not payload:
                return []
            seen = {"leakage": self._leakage_issues_seen,
                    "numeric": self._numeric_issues_seen,
                    "ambiguity": self._ambiguity_issues_seen,
                    "grain": self._grain_issues_seen}
            blocks = checks_from_payload(payload, code, seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return []
        for issue, block in blocks:
            self._events.append("data_check", issue=issue, detail=block[:2000])
        return [b for _, b in blocks]

    def _check_kernel_memory_usage(self, fp: dict) -> str:
        """Kernel-mode counterpart to _check_memory_usage (V3-034): `df`
        never comes back to this process in kernel mode, so this reads the
        `other_large_frames` {name: n_rows} map the kernel's own execute()
        epilogue already computes cheaply (auto/kernel.py) instead of
        needing a second round-trip. Best-effort: a failure here must never
        break a cell that otherwise ran fine."""
        if not fp:
            return ""
        try:
            from ledgerhound.shell.memory_checks import format_memory_block
            ref_len = int(fp.get("n_rows") or 0)
            pairs = fp.get("other_large_frames") or {}
            block = format_memory_block(pairs, ref_len, self._memory_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="redundant_memory_copy",
                                detail=block[:2000])
        return block

    def _check_kernel_mutation_delta(self, fp: dict, code: str) -> str:
        from ledgerhound.shell.verification import fingerprints_differ, format_delta
        if not fp:
            return ""
        before = self._last_fp
        self._last_fp = fp
        if not fingerprints_differ(before, fp):
            return ""
        try:
            rates = self._kernel.null_rates()
        except Exception:
            rates = {}
        prior_rates = self._last_null_rates
        self._last_null_rates = rates
        self._events.append("verification", code=code, before=before, after=fp,
                            null_rates=rates)
        return format_delta(before, fp, rates, prior_rates)

    def _teach_error(self, e: Exception, stdout: str) -> str:
        """Turn a raw exception into a message that tells the model exactly
        what to try next, instead of a bare traceback it has to guess at.
        Local-exec path only (a real exception object); the kernel path
        (V3-009) only gets a traceback string, so it goes through
        `_teach_error_text` directly via a parsed (kind, msg) pair."""
        names = [str(k) for k in (self._ns or {}) if not k.startswith("_")]
        return self._teach_error_text(type(e).__name__, str(e), stdout, names)

    def _teach_error_text(self, kind: str, msg: str, stdout: str,
                          names: list[str] | None = None) -> str:
        """Fuzzy-matches missing column/name references against the live
        schema so a typo ('amnt') resolves to the real column ('amount') in
        one retry instead of a blind-guess loop. `names` is the list of live
        sandbox variables when known (exec mode); the kernel path (V3-009)
        has no cheap way to introspect kernel-side variables, so it passes
        an empty list -- the column-based hints still work either way."""
        names = names or []
        cols = [str(c) for c in self._df.columns] if self._df is not None else []

        hint = ""
        if kind == "KeyError":
            missing = msg.strip("'\"")
            close = difflib.get_close_matches(missing, cols, n=3, cutoff=0.5)
            if close:
                hint = f"column {missing!r} not found; closest matches: {close}. "
            hint += f"full columns ({len(cols)}): {cols[:40]}"
        elif kind == "NameError":
            m = re.search(r"name '(\w+)' is not defined", msg)
            if m:
                missing = m.group(1)
                close = difflib.get_close_matches(missing, names + cols, n=3, cutoff=0.5)
                if close:
                    hint = f"{missing!r} is not defined; closest matches: {close}. "
                hint += f"variables available: {names[:30]}"
        elif kind == "AttributeError":
            m = re.search(r"has no attribute '(\w+)'", msg)
            if m:
                missing = m.group(1)
                close = difflib.get_close_matches(missing, cols, n=3, cutoff=0.5)
                if close:
                    hint = (f"{missing!r} not found as an attribute; if you meant a "
                            f"column, use df[{close[0]!r}] -- closest matches: {close}")
        elif kind == "MemoryError":
            hint = ("out of memory -- work on a sample instead, e.g. "
                    "`sample = df.sample(n=100_000, random_state=0)`, and retry.")
        elif kind in ("ImportError", "ModuleNotFoundError"):
            hint = ("that library isn't available in this sandbox. Guaranteed "
                    "available: pandas (pd), numpy (np). Do not retry the same "
                    "import -- use a pandas/numpy-only approach instead.")

        out = f"Error: {kind}: {msg}"
        if hint:
            out += f"\n{hint}"
        tail = stdout[-1500:]
        if tail.strip():
            out += f"\n{tail}"
        return out

    # Words too common in fraud-analytics prose to distinguish two claims.
    _CLAIM_STOPWORDS = frozenset("""
        a an and are as at be been by for from has have in is it its of on or
        that the this to with which while when where than then these those
        show shows showed shown indicate indicates higher lower high low more
        less most least very rate rates fraud fraudulent non nonfraud data
        dataset transaction transactions txn account accounts analysis
        observed observation appears appear seem seems significant
        significantly increase increases increased decrease decreases across
        between compared comparison overall total count counts value values
        """.split())

    def _claim_tokens(self, s: str) -> set:
        raw = re.findall(r"[a-z0-9]+", (s or "").lower())
        return {t for t in raw if len(t) > 1 and t not in self._CLAIM_STOPWORDS}

    def _claims_match(self, mine: set, theirs: set) -> bool:
        """Both bars: enough absolute overlap to be the same subject, and
        enough of the known claim covered that it is a restatement rather
        than a passing mention. Deliberately crude -- distinctive-token
        overlap, no semantics. False positives cost one sentence of
        pushback; the alternative cost four review iterations that
        produced "no substantially new fraud insights"."""
        if not mine or not theirs:
            return False
        shared = mine & theirs
        return len(shared) >= 3 and len(shared) / len(theirs) >= 0.5

    def _restates_established_finding(self, claim: str) -> tuple[str, str]:
        """``(finding_id, established_claim)`` if `claim` looks like a
        restatement of a domain-profile established finding specifically.
        Kept as its own method (rather than folded into
        `_check_all_known_sources`) because its message is worded around
        "the reviewer already accepted this" -- a different framing than a
        cross-book or team-KB match."""
        prof = self._domain_profile
        if not prof or not claim:
            return ("", "")
        try:
            from ledgerhound.context.domain_profiles import established_claims
        except Exception:  # noqa: BLE001
            return ("", "")
        mine = self._claim_tokens(claim)
        if not mine:
            return ("", "")
        for fid, known in established_claims(prof):
            if self._claims_match(mine, self._claim_tokens(known)):
                return (fid, known)
        return ("", "")

    def _book_history_claims(self) -> list[tuple[str, str]]:
        """(ref, claim) for every finding recorded on THIS book across
        every past session/story -- DatasetCard.finding_refs is already a
        cross-session history for this exact dataset path, no new plumbing
        needed. Ref is the stored "[frame] claim" string itself (there is
        no separate id at this layer)."""
        if self._card is None:
            return []
        return [(ref, ref) for ref in (self._card.finding_refs or [])]

    def _team_kb_claims(self) -> list[tuple[str, str]]:
        """(name, statement) from the team knowledge base, if one has been
        imported -- ledgerhound_workspace/memory/knowledge/team_findings.jsonl,
        one JSON object per line per the schema in
        docs/v3.0/KNOWLEDGE_BASE_INGESTION_PROMPT.md. No importer exists
        yet (see that doc); this reads the file directly so the check is
        real the moment one is dropped there, without waiting on the CLI
        tool. Missing file -> [], never an error."""
        if self.project is None:
            return []
        out: list[tuple[str, str]] = []
        # Findings teammates recorded on THIS dataset, pulled from the shared
        # knowledge base at load time (own findings from earlier runs too).
        for rec in self._team_findings or []:
            stmt = str(rec.get("statement") or rec.get("claim") or "").strip()
            if stmt:
                who = rec.get("_published_by") or ""
                name = str(rec.get("name") or stmt[:40]).strip()
                out.append((f"{name} (team: {who})" if who else name, stmt))
        try:
            from ledgerhound.workspace.layout import WorkspaceLayout
            ws = WorkspaceLayout.from_path(self._workspace_root())
            path = ws.memory_knowledge_dir / "team_findings.jsonl"
            if not path.exists():
                return out
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                stmt = (rec.get("statement") or "").strip()
                name = (rec.get("name") or "").strip()
                if stmt:
                    out.append((name or stmt[:40], stmt))
            return out
        except Exception:  # noqa: BLE001 -- advisory, never blocks a run
            return out

    def _check_all_known_sources(self, claim: str) -> list[tuple[str, str, str]]:
        """(source, ref, matched_text) for every known-claim source this
        claim looks like a restatement of: the domain profile (handled
        separately by `_restates_established_finding` for its own
        wording), this book's own cross-session history, and the team
        knowledge base. Used to BLOCK an unjustified restatement at
        record_finding time, not just warn after recording."""
        mine = self._claim_tokens(claim)
        if not mine:
            return []
        matches = []
        for ref, known in self._book_history_claims():
            if self._claims_match(mine, self._claim_tokens(known)):
                matches.append(("this book's history", ref, known))
        for name, known in self._team_kb_claims():
            if self._claims_match(mine, self._claim_tokens(known)):
                matches.append(("the team knowledge base", name, known))
        return matches

    def _tool_record_finding(self, claim: str, sample_frame: str,
                             evidence: str = "", predicate: str = "",
                             delta: str = "", pvalue: float = None) -> str:
        """V3-061 (adversarial loop, strengthened): checked BEFORE
        anything is written, against three sources -- the domain profile,
        this book's own cross-session history, and the team knowledge
        base. A match with no `delta` is REFUSED outright, not recorded
        and pushed back on after the fact -- the model must state
        explicitly what this claim adds before it lands. A `delta` (or no
        match at all) proceeds to record normally, with the delta folded
        into the evidence trail when given."""
        # V3-048's domain-profile check keeps its own wording (references
        # "the reviewer already accepted this"); cross-book/team-KB matches
        # get a differently-worded, source-labelled refusal below.
        # V5-002: every attempt joins the family, including the ones refused
        # below. Counting only successes would make a long fruitless search
        # look like a short decisive one.
        self._effort.note_attempt(getattr(self, "_current_chapter", ""))

        fid, known = self._restates_established_finding(claim)
        other_matches = self._check_all_known_sources(claim)

        if (fid or other_matches) and not delta.strip():
            self._events.append("finding_blocked_as_restatement",
                                claim=claim[:500],
                                domain_profile_match=fid or "",
                                other_matches=[m[0] for m in other_matches])
            self._note_no_new_finding()
            self._effort.note_refused(getattr(self, "_current_chapter", ""))
            stall = self._stall_suggestion()
            if fid:
                msg = (
                    f"REFUSED (not recorded) -- this looks like established "
                    f"finding [{fid}], which the reviewer has already "
                    f"accepted:\n  \"{known}\"\n"
                    "Re-deriving accepted results is what made four review "
                    "iterations produce no new insight. Call record_finding "
                    "again with the SAME claim plus delta=\"...\" stating "
                    "what this specifically ADDS (a mechanism, an "
                    "interaction, account-level behaviour, a feature it "
                    "justifies) -- or drop it and move on.")
                return f"{msg}\n{stall}" if stall else msg
            lines = ["REFUSED (not recorded) -- this looks like a restatement of:"]
            for source, ref, known_claim in other_matches:
                lines.append(f"  [{source}] {ref}: \"{known_claim}\"")
            lines.append(
                "Call record_finding again with the SAME claim plus "
                "delta=\"...\" stating what this specifically ADDS beyond "
                "what's already known -- or drop it and move on.")
            if stall:
                lines.append(stall)
            return "\n".join(lines)

        self._note_new_finding()
        full_evidence = f"{evidence}\n[delta: {delta}]" if delta.strip() else evidence
        # V3-012: findings.jsonl is now a projection of the event stream --
        # the same view function backs both this live write and
        # `ledgerhound run-views` bulk regeneration.
        ev = self._events.append("finding", claim=claim, evidence=full_evidence,
                                 sample_frame=sample_frame, predicate=predicate)
        # V3-052: catch two findings comparing the "same" quantity under
        # DIFFERENT predicates before the model treats them as equivalent.
        # Real bug this reproduces: fraud night-share computed as
        # hour>=22 OR hour<6, non-fraud night-share as hour in 0..5 --
        # compared directly as if they measured the same thing.
        predicate_warning = self._check_predicate_consistency(claim, predicate)
        try:
            from ledgerhound.runtime.views import finding_line
            with open(self._findings_path, "a", encoding="utf-8") as f:
                f.write(finding_line(ev))
        except Exception:
            pass
        # V3-011: also land on the dataset card so future sessions on this
        # exact file see it without re-deriving it.
        if self._card is not None:
            try:
                from ledgerhound.knowledge.dataset_card import DatasetCardStore
                self._card.add_finding(f"[{sample_frame}] {claim}")
                self._save_card()
            except Exception:
                pass
        # Team knowledge base: the full finding, not just its one-line ref,
        # so a teammate on the same dataset sees claim + evidence + frame.
        kb = self.team_kb()
        if kb is not None and self._dataset_id:
            try:
                kb.publish_finding(self._dataset_id, {
                    "name": claim[:80], "statement": claim, "claim": claim,
                    "evidence": full_evidence[:4000], "sample_frame": sample_frame,
                    "predicate": predicate, "run": self.run_dir.name,
                    "data_sha256": getattr(self, "_data_sha256", "")})
            except Exception as e:  # noqa: BLE001
                self._events.append("team_kb_error", op="publish_finding", error=str(e)[:300])

        # V5-002: how hard we looked is part of what the finding means.
        assessment = self._effort.assess(
            getattr(self, "_current_chapter", ""), claim, pvalue)
        self._events.append("finding_assessed", claim=claim[:500],
                            **assessment.to_dict())

        base = f"recorded finding [{sample_frame}]"
        if assessment.status != "unquantified":
            base += f" -- {assessment.status.upper()}"
        if delta.strip():
            base += " (with stated delta over a known match)"
        stall = self._stall_suggestion()
        parts = [p for p in (base, assessment.note, predicate_warning, stall) if p]
        return "\n".join(parts)

    def _note_new_finding(self) -> None:
        """V3-061 stop-heuristic: reset the consecutive-no-new-finding
        counter on anything that actually got recorded."""
        self._consecutive_no_new_finding = 0

    def _note_no_new_finding(self) -> None:
        """V3-061 stop-heuristic: count a blocked (restatement, no delta)
        record_finding attempt as "this step found nothing new". No
        automatic ceiling -- add_step/the phase checklist can grow
        indefinitely -- this only ever SUGGESTS wrapping up, never forces
        it (see `_stall_suggestion`)."""
        self._consecutive_no_new_finding = getattr(
            self, "_consecutive_no_new_finding", 0) + 1

    def _stall_suggestion(self, threshold: int = 3) -> str:
        n = getattr(self, "_consecutive_no_new_finding", 0)
        if n < threshold:
            return ""
        return (
            f"NOTE: the last {n} finding attempts in a row were restatements "
            "of something already known, with no new delta stated. Consider "
            "whether this chapter/phase has run its course -- wrap up and "
            "move on, or open_chapter on a genuinely different approach. "
            "Not forced; your call.")

    def _check_predicate_consistency(self, claim: str, predicate: str) -> str:
        """V3-052: if a prior finding shares this claim's distinctive
        subject (same token-overlap test as `_restates_established_
        finding`, applied to this run's OWN findings.jsonl instead of the
        domain profile) but was recorded under a DIFFERENT, non-empty
        predicate, flag it. Two findings that sound like they measure the
        same thing but used different windows/conditions are not
        comparable -- that mismatch is exactly what produced the
        7.8%-vs-31.9%-vs-31.7% weekend-share contradiction. Mechanical
        token overlap, not semantic -- a false positive costs one line of
        pushback, same tradeoff as the restatement check above."""
        if not predicate or not claim or not self._findings_path.exists():
            return ""

        def _toks(s: str) -> set:
            raw = re.findall(r"[a-z0-9]+", (s or "").lower())
            return {t for t in raw
                    if len(t) > 1 and t not in self._CLAIM_STOPWORDS}

        mine = _toks(claim)
        if not mine:
            return ""
        try:
            lines = self._findings_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return ""
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            other_claim = rec.get("claim", "")
            other_pred = rec.get("predicate", "")
            if not other_pred or other_pred == predicate:
                continue
            theirs = _toks(other_claim)
            if not theirs:
                continue
            shared = mine & theirs
            if len(shared) >= 3 and len(shared) / len(theirs) >= 0.5:
                return (
                    "NOTE: this looks like it measures the same thing as an "
                    f"earlier finding (\"{other_claim}\"), but under a "
                    f"DIFFERENT predicate -- this one: \"{predicate}\", the "
                    f"earlier one: \"{other_pred}\". Do not compare the two "
                    "numbers directly until you've confirmed they use the "
                    "same condition/window.")
        return ""

    def _tool_record_decision(self, decision: str, rationale: str = "") -> str:
        if self._card is None:
            return ("noted, but not persisted -- decisions are saved to the "
                    "dataset card, which requires a project (none is active "
                    "this session).")
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore
            self._card.add_decision(decision, rationale)
            self._save_card()
            return "recorded decision to the dataset card"
        except Exception as e:  # noqa: BLE001
            return f"could not save the decision: {e}"

    # -- Librarian: chapters (V3-053) -------------------------------------

    def _tool_open_chapter(self, chapter: str) -> str:
        """Declare the approach for the next stretch of work. Deduped
        against this exact dataset's card: the same chapter on UNCHANGED
        data (same content hash as when it last produced pages) is
        refused, showing what was already found; a new content hash
        (new month, re-extracted file) auto-allows, since that is exactly
        the cross-period comparison this discipline exists to support."""
        from ledgerhound.knowledge.chapter_taxonomy import is_known_chapter, taxonomy_block
        if not is_known_chapter(chapter):
            return (f"'{chapter}' is not in the chapter taxonomy.\n\n"
                    f"{taxonomy_block()}\n\n"
                    "Pick one of these, or say which existing chapter is "
                    "closest and why this is genuinely different -- new "
                    "entries are added deliberately, not invented per-call.")
        if self._card is None:
            self._current_chapter = chapter
            return (f"opened chapter '{chapter}' (not persisted -- no "
                    "project active this session, so dedup across sessions "
                    "isn't available; this session's own repetition is not "
                    "checked either).")
        allowed, info = self._card.open_chapter(
            self._data_sha256, chapter, row_count=self._true_n_rows or 0)
        from ledgerhound.knowledge.dataset_card import DatasetCardStore
        try:
            self._save_card()
        except Exception:
            pass
        if not allowed:
            pages = info.get("pages", [])
            found = "\n".join(f"  - {p}" for p in pages[-5:])
            return (
                f"REFUSED: chapter '{chapter}' already has {info.get('page_count', 0)} "
                f"page(s) recorded against this EXACT data version.\n"
                f"What was already found:\n{found}\n\n"
                "If you have a real reason to redo it anyway (the first "
                "pass was buggy, wrong assumption, etc.), call "
                "redo_chapter(chapter, reason) -- that is recorded, not "
                "silent. If the data has genuinely changed since, re-load "
                "it first; a new content hash opens a new story "
                "automatically and this chapter reopens on its own.")
        self._current_chapter = chapter
        return f"opened chapter '{chapter}'."

    def _tool_redo_chapter(self, chapter: str, reason: str) -> str:
        """Explicit override to redo a chapter whose data hasn't changed.
        Always recorded (card + event log), never silent."""
        if not reason.strip():
            return "a reason is required -- this is an override, not a default."
        self._events.append("chapter_redo", chapter=chapter, reason=reason[:1000])
        if self._card is None:
            self._current_chapter = chapter
            return f"redo of '{chapter}' noted (not persisted -- no project active)."
        self._card.redo_chapter(self._data_sha256, chapter, reason)
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore
            self._save_card()
        except Exception:
            pass
        self._current_chapter = chapter
        return f"redo of '{chapter}' recorded -- proceed."

    def _tool_add_chapter_page(self, summary: str) -> str:
        """Record one page of work under the currently open chapter.
        Append-only -- pages are never rewritten."""
        chapter = getattr(self, "_current_chapter", "")
        if not chapter:
            return "no chapter is open -- call open_chapter(chapter) first."
        if self._card is None:
            return f"page noted under '{chapter}' (not persisted -- no project active)."
        n = self._card.add_chapter_page(self._data_sha256, chapter, summary)
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore
            self._save_card()
        except Exception:
            pass
        return f"recorded page {n} of chapter '{chapter}'."

    def _tool_record_percentile_snapshot(self, feature: str, percentiles: dict,
                                         chapter: str = "") -> str:
        """V3-055: record a structured percentile snapshot for `feature`
        so a LATER session on different data can check it for stability.
        Separate from add_chapter_page's prose notes on purpose -- this is
        the numeric record check_cross_period_stability actually compares."""
        chapter = chapter or getattr(self, "_current_chapter", "")
        if not chapter:
            return ("no chapter given or open -- call open_chapter(chapter) "
                    "first, or pass chapter= explicitly.")
        if self._card is None:
            return (f"snapshot for '{feature}' noted (not persisted -- no "
                    "project active this session).")
        self._card.add_percentile_snapshot(self._data_sha256, chapter,
                                           feature, percentiles)
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore
            self._save_card()
        except Exception:
            pass
        return f"recorded percentile snapshot for '{feature}' under chapter '{chapter}'."

    def _tool_check_cross_period_stability(self, feature: str, chapter: str = "",
                                           against_story: str = "") -> str:
        """V3-055: compare a recorded percentile snapshot for `feature`
        between the CURRENT data version and an earlier story on this same
        book. Defaults to the story immediately before the current one;
        pass against_story (a story_id from /story or /library) to name a
        specific earlier one. This is the automated half of cross-period
        confirmation -- record_percentile_snapshot must have been called
        in both stories first, or there is nothing to compare."""
        chapter = chapter or getattr(self, "_current_chapter", "")
        if not chapter:
            return "no chapter given or open -- specify chapter, or open one first."
        if self._card is None:
            return "no project active -- cross-period comparison requires the dataset card."
        stories = self._card.stories
        current = self._card.find_story(self._data_sha256)
        if current is None:
            return ("the current data version has no story yet -- load data "
                    "and record a snapshot before comparing.")
        if against_story:
            other = next((s for s in stories if s.get("story_id") == against_story
                         or s.get("data_sha256", "").startswith(against_story)), None)
        else:
            idx = stories.index(current)
            other = stories[idx - 1] if idx > 0 else None
        if other is None:
            return ("no earlier story to compare against -- this looks like "
                    "the first story recorded for this dataset, or the story "
                    "named wasn't found. Use /story or /library to see what's "
                    "recorded.")
        if other.get("data_sha256") == current.get("data_sha256"):
            return "that resolves to the SAME story as the current data -- nothing to compare."
        from ledgerhound.knowledge.stability import check_cross_period_stability
        result = check_cross_period_stability(
            self._card, other["data_sha256"], current["data_sha256"], chapter, feature)
        if not result.get("ok"):
            return result.get("reason", "comparison failed")
        verdict = "STABLE" if result["stable"] else "SHIFTED"
        lines = [f"{verdict}: '{feature}' under chapter '{chapter}', "
                f"{other['story_id']} -> {current['story_id']} "
                f"({result['compared_labels']} quantile(s) compared)"]
        if result["shifted_labels"]:
            lines.append("shifted at: " + ", ".join(
                f"{l} ({result['deltas'][l]:+.1%})" for l in result["shifted_labels"]))
        return "\n".join(lines)

    def _findings_since_last_save(self) -> list[str]:
        """Claims recorded since the previous save_report, oldest first.

        Reads the run's findings.jsonl (the projection record_finding
        already maintains) and computes how far it would advance -- but
        does NOT commit that advance yet. V3-052: this used to mark
        findings consumed here, unconditionally, before save_output ran. If
        the save then failed, the findings were already marked read and
        NO later report could ever include them again -- a report-time
        failure silently and permanently dropped evidence. The consumed
        marker is now only committed by `_commit_findings_consumed()`,
        called from `_tool_save_report` after a save (or its fallback)
        actually persisted the findings text.
        """
        seen = getattr(self, "_findings_consumed", 0)
        out: list[str] = []
        self._findings_consumed_pending = seen
        try:
            if self._findings_path.exists():
                lines = self._findings_path.read_text(
                    encoding="utf-8").splitlines()
                for line in lines[seen:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    claim = (rec.get("claim") or "").strip()
                    if not claim:
                        continue
                    ev = (rec.get("evidence") or "").strip()
                    frame = (rec.get("sample_frame") or "").strip()
                    bits = [claim]
                    if ev:
                        bits.append(f"evidence: {ev}")
                    if frame:
                        bits.append(f"frame: {frame}")
                    out.append(" — ".join(bits))
                self._findings_consumed_pending = len(lines)
        except Exception:  # noqa: BLE001 -- never break a save over this
            pass
        return out

    def _commit_findings_consumed(self) -> None:
        """Advance the real consumed marker to what `_findings_since_last_
        save` staged -- call ONLY after the findings it returned have
        actually been written somewhere durable (V3-052)."""
        pending = getattr(self, "_findings_consumed_pending", None)
        if pending is not None:
            self._findings_consumed = pending

    def _tool_save_report(self, title: str, markdown: str,
                          iteration: int = 0) -> str:  # noqa: ARG002
        # `iteration` is accepted and IGNORED (V3-045). It is off the tool
        # schema now; the parameter stays so an older caller or a replayed
        # transcript that still sends it does not blow up.
        fmt = "md"
        if self.project is not None:
            try:
                fmt = self.project.config().save_format or "md"
            except Exception:
                pass
        # Sequence number is derived here, never taken from the model
        # (V3-045). The `iteration` argument used to set it, which gave two
        # competing numberings -- the model's idea and the folder's -- and
        # made a gap in the sequence unreadable: you could not tell a skipped
        # save from a mis-numbered one.
        self._report_iteration = getattr(self, "_report_iteration", 0) + 1
        it = self._report_iteration
        stem = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in title.lower())[:44] or "report"

        # Put the PHASE in the filename when a methodology is active. The
        # operator's complaint (2026-08-11) was that a run "started from
        # saving phase 3" -- with `iteration_01_...` as the only name, a
        # missing phase 1 and 2 is invisible in the folder listing. With the
        # phase in the stem, `ls` shows the gap.
        phase_key = ""
        if self.methodology is not None:
            ph = self.methodology.current_phase()
            if ph is not None:
                phase_key = ph.id
                stem = (f"phase_{self.methodology.current + 1}_"
                        f"{ph.id}__{it:02d}_{stem}")
        if not phase_key:
            stem = f"iteration_{it:02d}_{stem}"

        # Findings recorded SINCE THE LAST SAVE. The report used to carry
        # only the model's prose: `ReportPayload` has had `findings` and
        # `figures` fields all along and `_tool_save_report` passed neither,
        # so every number the analyst had actually recorded via
        # record_finding was absent from the artifact. Scoped to new findings
        # rather than all of them, or phase 5's report restates phases 1-4.
        new_findings = self._findings_since_last_save()
        # Carry a short pointer to the previous iteration into the document
        # so the continuity is in the artifact itself, not just the folder.
        prev = getattr(self, "_report_history", [])
        if prev:
            markdown = (
                f"*Iteration {it}. Builds on: "
                + ", ".join(f"`{p}`" for p in prev[-3:])
                + ".*\n\n" + markdown)
        self._report_history = prev + [f"{stem}.{fmt}"]
        try:
            from ledgerhound.analyst.save import ReportPayload, save_output
            res = save_output(
                ReportPayload(title=title, body_md=markdown,
                              findings=new_findings),
                fmt, out_dir=self.run_dir, stem=stem)
            self._commit_findings_consumed()  # V3-052: only after the write succeeded
            self._export_notebook()  # V3-014: best-effort, never blocks save_report
            if phase_key:
                self._reports_by_phase.setdefault(phase_key, []).append(
                    f"{stem}.{fmt}")
            # Say how much evidence went in. A "0 findings" report is the
            # signal that the analyst wrote prose without recording anything
            # -- visible to the model in the tool result, not buried.
            # J-07: a renderer that fell back (pdf without fpdf -> md) says
            # so in res.note; without it the model reported "saved pdf"
            # for a markdown file.
            fallback = (f" NOTE: {res.note}" if getattr(res, "note", "")
                        and not getattr(res, "ok", True) else "")
            return (f"saved {res.format} -> {res.path} "
                    f"({len(new_findings)} finding(s) recorded since the "
                    f"last save){fallback}")
        except Exception as e:  # noqa: BLE001
            # Do NOT silently claim success on a different format than asked
            # for -- surface the real error, then fall back to md so the
            # content is never lost, and say plainly that the fallback fired.
            # V3-052: the fallback used to write ONLY the model's prose,
            # with no findings section -- and the findings were about to be
            # marked consumed regardless, so they would vanish from every
            # future report too. The fallback now carries the same findings
            # text the primary save would have, and the consumed marker is
            # only advanced once this fallback write actually lands.
            try:
                findings_md = ""
                if new_findings:
                    findings_md = "\n\n## Findings\n\n" + "\n".join(
                        f"- {f}" for f in new_findings)
                path = self.run_dir / f"{stem}.md"
                path.write_text(f"# {title}\n\n{markdown}{findings_md}\n",
                                encoding="utf-8")
                self._commit_findings_consumed()
                return (f"'{fmt}' save failed ({type(e).__name__}: {e}); "
                        f"fell back to md -> {path} "
                        f"({len(new_findings)} finding(s) included)")
            except Exception as e2:  # noqa: BLE001
                # Total failure: do NOT commit -- these findings must remain
                # unconsumed so the NEXT save_report attempt still carries
                # them, instead of being silently lost forever.
                return (f"save FAILED entirely -- {fmt} error: "
                        f"{type(e).__name__}: {e}; md fallback also failed: "
                        f"{type(e2).__name__}: {e2}. Run folder: {self.run_dir}. "
                        f"{len(new_findings)} finding(s) NOT lost -- they "
                        f"remain unconsumed for the next save_report.")

    def _export_notebook(self) -> None:
        """V3-014: emit analysis.ipynb from the event stream on save_report.
        Best-effort -- a notebook-export failure must never break the
        report save it rides along with."""
        try:
            from ledgerhound.runtime.notebook_export import export_notebook
            export_notebook(self.run_dir)
        except Exception:
            pass

    def _tool_consult_skill(self, query: str) -> str:
        try:
            from ledgerhound.skills import SkillLibrary
            hits = SkillLibrary().load().recall(query, k=1)
            if not hits:
                return f"no team skill matches '{query}' -- use your own judgment."
            s = hits[0]
            return (f"Skill '{s.name}' (when: {s.when_to_use}). Reference -- adapt, "
                    f"don't obey blindly:\n\n{s.body.strip()[:2500]}")
        except Exception as e:  # noqa: BLE001
            return f"could not load skills: {e}"

    def _tool_ask_user(self, question: str, options: list, multi: bool = False) -> str:
        """V3-032: a real, blocking MCQ/MSQ -- routes through the same
        HITLGate the high-tier save_report confirmation already uses, so
        this is a genuine pause for a real answer, not just text the model
        writes and hopes gets read. Low tier (see hitl.classify_action):
        the blocking happens HERE, inside the tool body via gate.choose[_multi],
        not via the tier wrapper around dispatch -- calling ask_user is itself
        already "asking", so there is nothing to additionally gate before it runs.
        """
        opts = [str(o) for o in (options or [])]
        if not opts:
            return "ask_user needs at least one option -- give the analyst something concrete to pick from."
        # I-04: an UNATTENDED run (--autonomous) has nobody to answer, and
        # the terminal prompt below would sit there for hours (a production
        # run once idled ~59 minutes on exactly this). The question
        # firewall only covers prose questions; the ask_user TOOL still
        # blocked. Take the first option -- the system prompt asks the model
        # to put its own recommendation among the options, and by the same
        # convention the HITL gate already uses (options[0] = the safe/
        # recommended one) -- and say so in the observation so the model
        # knows it was a default, not a human decision. Governance asks
        # (modelling authorisation, high-tier writes) are NOT routed here
        # and stay blocking.
        if self.autonomous:
            self._log("hitl", tier="ask_user", name="ask_user", approved=True,
                      answer=opts[0], auto=True)
            return (f"analyst picked: {opts[0]} (AUTONOMOUS RUN -- no human "
                    "was available, so the first option was taken as your "
                    "recommendation; state this assumption in your report "
                    "and revisit it if later evidence contradicts it)")
        gate = self.registry.gate
        if multi:
            decision = gate.choose_multi(question, opts)
            self._log("hitl", tier="ask_user", name="ask_user",
                      approved=True, answer=decision.answer)
            return f"analyst picked: {', '.join(decision.picked)}"
        decision = gate.choose(question, opts)
        self._log("hitl", tier="ask_user", name="ask_user",
                  approved=True, answer=decision.answer)
        return f"analyst picked: {decision.answer}"

    # -- methodology (V3-043) --------------------------------------------
    _AUTONOMOUS_DEFAULT_METHODOLOGY = "topdown_7phase"

    def _choose_methodology_once(self) -> str:
        """Ask the analyst which methodology to work, once, on first load.

        WHY ASK RATHER THAN PICK: the feedback that produced this feature
        was that each analyst follows their own sequence based on their own
        experience -- one works top-down, another lets phase 1 decide the
        sampling. That is a genuine judgment call about how THIS person
        works, not something inferable from the data, so the model must not
        guess it.

        Three ways it gets decided, in order:
          1. --methodology on the CLI      -> already set, never asked
          2. interactive session           -> asked here, once
          3. autonomous/unattended run     -> a stated default, NOT a
             blocking prompt (an earlier production run sat idle for ~59
             minutes waiting on a human for an ask_user; an unattended job
             must never do that again)
        """
        if self.methodology is not None or self._methodology_asked:
            return ""
        self._methodology_asked = True
        try:
            from ledgerhound.methodology import available_templates, load_template
            names = available_templates()
            if not names:
                return ""
            if self.autonomous:
                m = load_template(self._AUTONOMOUS_DEFAULT_METHODOLOGY) \
                    or load_template(names[0])
                if m is None:
                    return ""
                self.methodology = m
                if m.phases:
                    m.phases[0].status = "active"
                self._events.append("methodology_set", template=m.name,
                                    chosen_by="autonomous_default")
                self._save_methodology()
                self._refresh_schema_pin()
                return (f"[METHODOLOGY] autonomous run -- defaulted to "
                        f"'{m.name}'. Pass --methodology to choose a "
                        f"different one.\n\n{m.pin_block()}")

            # Only ask when a HUMAN can actually answer. HITLGate's default
            # prompt is `auto_yes`, which returns options[0] -- asking
            # through that would "choose" the first template on the
            # analyst's behalf, which is precisely the silent guess this
            # feature exists to prevent. With no interactive prompt we
            # leave the methodology unset and say so; set_methodology and
            # --methodology both remain available.
            from ledgerhound.analyst.hitl import auto_yes
            if getattr(self.registry.gate, "prompt", None) is auto_yes:
                self._events.append("methodology_set", template=None,
                                    chosen_by="not_asked_non_interactive")
                return ("[METHODOLOGY] none adopted -- no interactive prompt "
                        "is available to choose one, and picking for you "
                        "would be a guess. Use --methodology, or call "
                        "set_methodology, to work a declared phase "
                        "sequence.")

            options, by_label = [], {}
            for n in names:
                t = load_template(n)
                if t is None:
                    continue
                label = f"{n} ({len(t.phases)} phases, " \
                        f"{sum(len(p.steps) for p in t.phases)} steps)"
                options.append(label)
                by_label[label] = n
            options.append("no methodology -- free-form analysis")
            question = (
                "Which analysis methodology should this session follow?\n"
                "Each is an ordered phase sequence with its own EDA step "
                "checklist; it drives what each phase is allowed to "
                "produce. Analysts differ here, so this is your call:\n"
                "  - topdown_7phase: fraud-only EDA (both grains) -> "
                "non-fraud -> contrast -> aggregation plan -> features -> "
                "selection -> modelling\n"
                "  - sample_first_7phase: same, but phase 1 decides how "
                "non-fraud is sampled")
            answer = self.registry.gate.choose(question, options).answer
            name = by_label.get(answer)
            if not name:
                self._events.append("methodology_set", chosen_by="analyst",
                                    template=None)
                return ("[METHODOLOGY] none -- free-form analysis. Adopt one "
                        "later with set_methodology if you change your mind.")
            m = load_template(name)
            if m is None:
                return ""
            self.methodology = m
            if m.phases:
                m.phases[0].status = "active"
            self._events.append("methodology_set", template=name,
                                chosen_by="analyst")
            self._save_methodology()
            self._refresh_schema_pin()
            return f"[METHODOLOGY] {m.name}\n\n{m.pin_block()}"
        except Exception:  # noqa: BLE001 -- never break a data load
            return ""

    def _save_methodology(self) -> None:
        if self.methodology is None:
            return
        try:
            (self.run_dir / "methodology.json").write_text(
                json.dumps(self.methodology.to_dict(), indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001 -- persistence is best-effort
            pass

    def _tool_set_methodology(self, template: str = "", mode: str = "") -> str:
        from ledgerhound.methodology import available_templates, load_template
        names = available_templates()
        if not template:
            lines = ["available methodologies:"]
            for n in names:
                m = load_template(n)
                if m:
                    steps = sum(len(p.steps) for p in m.phases)
                    lines.append(f"  - {n}: {m.name} "
                                 f"({len(m.phases)} phases, {steps} steps)")
                    lines.append(f"      {' '.join(m.description.split())[:220]}")
            lines.append("Call set_methodology(template='...') to adopt one. "
                        "Optionally add mode='eda_only'|'fe_only'|'eda_fe_only' "
                        "-- a separate axis controlling how far the pipeline "
                        "goes, not the reasoning order.")
            return "\n".join(lines)
        m = load_template(template, mode=mode)
        if m is None:
            return (f"no methodology named {template!r}. Available: "
                    f"{', '.join(names)}")
        # NOTE: load_template already set the correct active phase (index 0
        # normally; the first feature/model phase when mode='fe_only', via
        # Methodology.apply_mode_start()) -- do not override it here.
        self.methodology = m
        self._events.append("methodology_set", template=template, mode=m.mode,
                            phases=len(m.phases),
                            steps=sum(len(p.steps) for p in m.phases))
        self._save_methodology()
        self._refresh_schema_pin()
        mode_note = f" [mode: {m.mode}]" if m.mode else ""
        return (f"adopted '{m.name}'{mode_note} -- {len(m.phases)} phases, "
                f"{sum(len(p.steps) for p in m.phases)} steps.\n\n"
                + m.pin_block())

    def _tool_complete_step(self, step_id: str, note: str = "",
                            skipped: bool = False) -> str:
        if self.methodology is None:
            return ("no methodology set -- call set_methodology first, or "
                    "work without one.")
        ph = self.methodology.current_phase()
        if ph is None:
            return "methodology has no phases."
        for s in ph.steps:
            if s.id == step_id:
                s.status = "skipped" if skipped else "done"
                s.note = note or s.note
                self._events.append("step_complete", phase=ph.id,
                                    step=step_id, skipped=bool(skipped),
                                    note=note[:1000])
                self._save_methodology()
                self._refresh_schema_pin()
                done, total = ph.progress()
                left = len(ph.pending_steps())
                tail = (f" {left} step(s) left in this phase."
                        if left else
                        " All steps done -- call advance_phase when ready.")
                return (f"{'skipped' if skipped else 'completed'} {step_id} "
                        f"[{done}/{total}].{tail}")
        ids = ", ".join(s.id for s in ph.steps[:12])
        return (f"no step {step_id!r} in phase '{ph.id}'. Steps here: {ids}"
                f"{' ...' if len(ph.steps) > 12 else ''}")

    def _tool_add_step(self, text: str, grain: str = "any") -> str:
        from ledgerhound.methodology.phases import Step
        if self.methodology is None:
            return "no methodology set -- call set_methodology first."
        ph = self.methodology.current_phase()
        if ph is None:
            return "methodology has no phases."
        n = sum(1 for s in ph.steps if s.source == "analyst") + 1
        sid = f"{ph.id.split('_')[0]}.x{n:02d}"
        ph.steps.append(Step(id=sid, text=text, grain=grain or "any",
                             source="analyst"))
        self._events.append("step_added", phase=ph.id, step=sid, text=text[:500])
        self._save_methodology()
        self._refresh_schema_pin()
        return f"added {sid} to phase '{ph.id}': {text}"

    def _tool_advance_phase(self, force: bool = False) -> str:
        if self.methodology is None:
            return "no methodology set -- call set_methodology first."

        # V3-045: a phase does not close without an artifact.
        #
        # Observed 2026-08-11: a run's output folder began at phase 3 --
        # phases 1 and 2 had run and left nothing saved. Advisory wording
        # cannot fix that; an agent under completion pressure advances. So
        # this REFUSES, consistent with the standing rule that governance
        # refuses rather than warns where a deliverable is at stake.
        #
        # Deliberately does NOT auto-write a stub on force. A file for a
        # phase whose report nobody wrote is worse than no file: it makes
        # the folder listing claim coverage that does not exist, which is
        # the exact illusion being fixed here. Forcing is legitimate and
        # recorded; the missing report is then the honest artifact.
        # Order matters: the methodology's own pending-steps refusal comes
        # first. Doing the steps precedes reporting on them, so telling an
        # analyst "no report saved" while six steps are still open points at
        # the wrong problem.
        ph = self.methodology.current_phase()
        if ph is not None and not force and ph.pending_steps():
            ok, msg = self.methodology.advance(force=False)   # refuses, no-op
            self._events.append("phase_advance", ok=ok, forced=False,
                                detail=msg[:1000])
            return msg

        if ph is not None and not force and not self._reports_by_phase.get(ph.id):
            self._events.append("phase_advance_refused",
                                phase=ph.id, reason="no_report_saved")
            return (
                f"REFUSED: phase '{ph.id}' has no saved report.\n"
                "Every phase leaves an artifact -- call save_report with this "
                "phase's findings before advancing, so the run folder shows "
                "what each phase actually produced.\n"
                "If this phase genuinely has nothing to report, "
                "advance_phase(force=True) moves on and records the gap.")
        if ph is not None and force and not self._reports_by_phase.get(ph.id):
            self._events.append("phase_closed_without_report", phase=ph.id)

        ok, msg = self.methodology.advance(force=bool(force))
        self._events.append("phase_advance", ok=ok, forced=bool(force),
                            detail=msg[:1000])
        self._save_methodology()
        self._refresh_schema_pin()
        if not ok:
            return msg
        return msg + "\n\n" + self.methodology.pin_block()

    def _tool_phase_status(self) -> str:
        if self.methodology is None:
            return ("no methodology set. set_methodology() lists the "
                    "available ones.")
        s = self.methodology.summary()
        lines = [f"methodology: {s['methodology']}",
                 f"phase {s['phase_index']}/{s['phase_count']}: {s['phase']}"]
        for p in s["phases"]:
            mark = {"done": "x", "active": ">", "pending": " "}.get(p["status"], " ")
            lines.append(f"  [{mark}] {p['id']:22s} {p['done']}/{p['total']} steps")
        return "\n".join(lines) + "\n\n" + self.methodology.pin_block()

    def _tool_plan_features(self, entity_key: str = "", time_column: str = "",
                            max_features: int = 1000, windows: list = None) -> str:
        """Enumerate candidate entity-level features from the live schema.

        Deliberately returns CANDIDATES plus the code to compute them, not
        computed values: at crore scale the analyst must batch the work,
        and the vetting gate -- not this generator -- decides what is
        actually recommended."""
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        from ledgerhound.analysis.feature_factory import (
            emit_code,
            generate,
            infer_roles,
            summarise,
        )
        ek = entity_key or self.goal.entity_key
        if not ek:
            return ("no entity_key: pass one, or declare it with set_goal "
                    "(unit_of_analysis='account', entity_key='...'). Feature "
                    "generation needs to know what one row means.")
        cols = [str(c) for c in self._df.columns]
        if ek not in cols:
            return f"entity_key {ek!r} is not a column. Available: {cols[:40]}"
        dtypes = {str(c): str(t) for c, t in self._df.dtypes.items()}
        # Cardinality from whatever frame we hold. When that is only a schema
        # sample the ROLE split (identifier vs category) is an estimate --
        # say so rather than letting it read as measured.
        try:
            nun = {c: int(self._df[c].nunique()) for c in cols}
        except Exception:
            nun = {}
        tcol = time_column or next(
            (c for c, t in dtypes.items() if "date" in t.lower()
             or "time" in t.lower()), "")
        roles = infer_roles(dtypes, entity_key=ek,
                            label_column=self.goal.label_column,
                            nunique=nun, n_rows=len(self._df))
        specs = generate(roles, ek, time_column=tcol,
                         windows=tuple(windows) if windows else (1, 7, 30),
                         max_features=int(max_features or 1000))
        self._feature_specs = [s.to_dict() for s in specs]
        try:
            (self.run_dir / "feature_candidates.json").write_text(
                json.dumps(self._feature_specs, indent=2), encoding="utf-8")
        except Exception:
            pass
        code = emit_code(specs, ek, time_column=tcol)
        counts = summarise(specs)
        sample = ", ".join(s.name for s in specs[:12])
        note = ""
        if self._df_is_sample:
            note = ("\nNOTE: column roles were inferred from the schema "
                    "sample, so identifier-vs-category splits are estimates; "
                    "verify a couple before trusting the plan.\n")
        return (
            f"planned {counts['total']} candidate features at grain '{ek}'"
            f"{f' (time column: {tcol})' if tcol else ' (no time column -- no velocity features)'}.\n"
            f"by kind: {counts}\n"
            f"first few: {sample} ...\n"
            f"full list saved to feature_candidates.json\n{note}\n"
            "These are CANDIDATES. Compute them, then rank and prune with\n"
            "evaluate_features(feat, [...], target) and check_leakage(...)\n"
            "before recommending ANY of them -- real pilot feedback was that\n"
            "of a large proposed set only 6 of the top 50 mattered, so the\n"
            "deliverable must be the survivors, not this list.\n\n"
            "Starter code (batch it if the frame is large):\n"
            f"```python\n{code}\n```")

    def _tool_export_feature_catalog(self, evaluations: list = None,
                                     vetoed: list = None,
                                     redundant: list = None,
                                     iv_core: float = None,
                                     iv_supporting: float = None,
                                     title: str = "",
                                     percentiles: dict = None) -> str:
        """Write the documented, TIERED feature catalog to the run folder.

        Tiering rather than pruning: IV is univariate, so a feature that
        looks modest alone can still contribute through interactions in a
        GBM. Only leakage-vetoed, redundant, or genuinely signal-free
        features are dropped -- and even those stay in the catalog with
        their reason, so an excluded feature is explainable."""
        specs = getattr(self, "_feature_specs", None)
        if not specs:
            return ("no feature plan in this session -- call plan_features "
                    "first, then compute and evaluate them.")
        from ledgerhound.analysis.feature_factory import (
            IV_CORE,
            IV_SUPPORTING,
            build_catalog,
            build_dual_node_catalog,
            catalog_to_markdown,
            tier_features,
            tier_summary,
        )
        ek = self.goal.entity_key or "entity"
        ek2 = getattr(self.goal, "entity_key_secondary", "")
        if ek2:
            # V3-053: the goal declared BOTH sides of the payment -- build
            # every spec against both grains rather than defaulting to the
            # payer/remitter side alone (the testing-team-reported gap).
            rows = build_dual_node_catalog(specs, ek, ek2,
                                           evaluations=evaluations or [],
                                           percentiles=percentiles)
        else:
            rows = build_catalog(specs, ek, evaluations=evaluations or [],
                                 percentiles=percentiles)
        rows = tier_features(
            rows,
            iv_core=float(iv_core) if iv_core is not None else IV_CORE,
            iv_supporting=(float(iv_supporting) if iv_supporting is not None
                           else IV_SUPPORTING),
            vetoed=set(vetoed or ()), redundant=set(redundant or ()))
        tiers = tier_summary(rows)
        md = catalog_to_markdown(rows, ek, title=title or "Feature catalog")
        wrote = []
        try:
            import csv
            csv_path = self.run_dir / "feature_catalog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            wrote.append(csv_path.name)
        except Exception as e:  # noqa: BLE001
            return f"catalog CSV failed: {type(e).__name__}: {e}"
        try:
            md_path = self.run_dir / "feature_catalog.md"
            md_path.write_text(md, encoding="utf-8")
            wrote.append(md_path.name)
        except Exception:
            pass
        n_meas = sum(1 for r in rows if r.get("measured"))
        n_sel = sum(1 for r in rows if r.get("selected"))
        self._events.append("feature_catalog", candidates=len(rows),
                            measured=n_meas, kept=n_sel, tiers=tiers)
        warn = ""
        if n_meas == 0:
            warn = ("\nWARNING: none of these are measured yet. Compute the "
                    "features and run evaluate_features(...), then call this "
                    "again -- an unvetted list is exactly what the pilot "
                    "rejected.")
        elif ek2:
            # K-04 follow-up: say which side is unmeasured, and how to fix it.
            n_b = sum(1 for r in rows if r.get("node") == "beneficiary"
                      and not r.get("measured"))
            if n_b:
                warn = (f"\nNOTE: {n_b} beneficiary-side feature(s) are "
                        "unmeasured. Plain-named evaluations apply to the "
                        "remitter side only; compute the features grouped on "
                        f"'{ek2}' and pass their evaluations as "
                        "'<name>__beneficiary'.")
        return (f"wrote {', '.join(wrote)} -- {len(rows)} candidates, "
                f"{n_meas} measured, {n_sel} kept.\ntiers: {tiers}\n"
                f"Each row carries: description, source columns, computed_at/"
                f"fit_scope, the exact reproducing code, IV/KS/AUC, its tier "
                f"and WHY it landed there.\n"
                f"Nothing was deleted -- 'dropped' rows stay in the catalog "
                f"with their reason. Adjust iv_supporting (default 0.02) if "
                f"you want to keep even more.{warn}")

    def _tool_recall_history(self, query: str, max_hits: int = 5) -> str:
        """Search the verbatim archive of compacted-out messages.

        This is what makes compaction non-destructive: the summary is the
        cheap view, this is the exact one. Scores by number of distinct
        query terms present, so a multi-word query finds the message that
        matches the most of it rather than the one repeating one word."""
        from ledgerhound.shell.compressor import ShellContextCompressor
        path = ShellContextCompressor.archive_path(self)
        if path is None or not Path(path).exists():
            return ("no compacted history yet -- nothing has been compressed "
                    "out of the window, so everything is still in context.")
        terms = [t for t in str(query).lower().split() if t]
        if not terms:
            return "give recall_history a non-empty query."
        hits = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    body = str(rec.get("content", ""))
                    low = body.lower()
                    score = sum(1 for t in terms if t in low)
                    if score:
                        hits.append((score, len(body), rec, body))
        except Exception as e:  # noqa: BLE001
            return f"could not read the compacted history: {type(e).__name__}: {e}"
        if not hits:
            return (f"no compacted message matches {query!r}. It may still be "
                    "in the live context, or in findings.jsonl / "
                    "session_code.py -- check those before recomputing.")
        hits.sort(key=lambda h: (-h[0], -h[1]))
        out = [f"{len(hits)} match(es) for {query!r}; showing "
               f"{min(int(max_hits or 5), len(hits))}:"]
        for score, _, rec, body in hits[:int(max_hits or 5)]:
            out.append(f"\n--- batch {rec.get('batch')} · {rec.get('role')} "
                       f"· matched {score}/{len(terms)} terms ---\n"
                       f"{body[:2500]}")
        return "\n".join(out)

    def _tool_set_goal(self, objective: str, unit_of_analysis: str = "",
                       entity_key: str = "", entity_key_secondary: str = "",
                       label_column: str = "",
                       notes: str = "") -> str:
        """Declare what the session is for. Pinned every turn and enforced."""
        from ledgerhound.shell.session_goal import GRAIN_ACCOUNT, normalise_grain
        g = self.goal
        if not g.origin_instruction and self._first_user_turn_text:
            g.origin_instruction = self._first_user_turn_text
        g.objective = objective or g.objective
        if unit_of_analysis:
            g.unit_of_analysis = normalise_grain(unit_of_analysis)
        if entity_key:
            g.entity_key = entity_key
        if entity_key_secondary:
            g.entity_key_secondary = entity_key_secondary
        if label_column:
            g.label_column = label_column
        if notes:
            g.notes = notes
        self._events.append("goal_set", **g.to_dict())
        g.save(self.run_dir / "session_goal.json")
        self._refresh_schema_pin()
        warn = ""
        if g.unit_of_analysis == GRAIN_ACCOUNT and not g.entity_key:
            warn = ("\nWARNING: unit_of_analysis is 'account' but no "
                    "entity_key was given -- the grain gate can only enforce "
                    "one-row-per-entity once you set it. Call set_goal again "
                    "with entity_key.")
        return f"goal recorded and pinned.\n\n{g.pin_block()}{warn}"

    def _tool_request_modeling_authorisation(self, summary: str) -> str:
        """Ask a human to move the session from analysis into modelling.
        Nothing else can flip this -- the model cannot authorise itself."""
        q = ("The analyst wants to move from analysis/feature work into MODEL "
             f"TRAINING.\n\nIts summary: {summary}\n\nAuthorise modelling?")
        opts = ["authorise modelling",
                "no -- keep doing feature/analysis work"]
        decision = self.registry.gate.choose(q, opts)
        approved = str(decision.answer).startswith("authorise")
        self.goal.modeling_authorised = approved
        if approved:
            self.goal.phase = "model"
        self._events.append("goal_gate", kind="modeling_authorisation",
                            approved=approved, summary=summary[:1000])
        self.goal.save(self.run_dir / "session_goal.json")
        self._refresh_schema_pin()
        if approved:
            return ("modelling AUTHORISED by the analyst -- fit/train calls "
                    "are now allowed.")
        return ("modelling NOT authorised. Continue the analysis / feature "
                "work; do not attempt to train a model.")

    def _tool_save_understanding(self, title: str, summary: str,
                                 goal: str = "") -> str:
        try:
            from ledgerhound.analyst.books import Book, BookShelf
            findings = []
            if self._findings_path.exists():
                for line in self._findings_path.read_text(encoding="utf-8").splitlines():
                    try:
                        findings.append(json.loads(line).get("claim", "")[:120])
                    except Exception:
                        continue
            book = Book(title=title, summary=summary, goal=goal,
                        data_ref=self._loaded_path, finding_ids=findings,
                        report_path=str(self.run_dir))
            if self.project is not None:
                path = BookShelf(self.project).save(book)
                kb = self.team_kb()
                if kb is not None and self._dataset_id:
                    try:
                        kb.publish_book(self._dataset_id, book.to_dict())
                    except Exception as e:  # noqa: BLE001
                        self._events.append("team_kb_error", op="publish_book",
                                            error=str(e)[:300])
            else:  # no project -> keep it in the run folder
                path = self.run_dir / "book.md"
                path.write_text(book.to_markdown(), encoding="utf-8")
            return f"saved understanding to the books -> {path}"
        except Exception as e:  # noqa: BLE001
            return f"could not save the book: {e}"

    def _recall_books(self, query: str) -> str:
        if self.project is None:
            return ""
        try:
            from ledgerhound.analyst.books import BookShelf
            hits = BookShelf(self.project).recall(query, k=2)
            if not hits:
                return ""
            lines = ["(prior sessions on similar data:)"]
            for b in hits:
                lines.append(f"  - {b.title}: {b.summary.strip()[:140]}")
            return "\n".join(lines)
        except Exception:
            return ""

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

    def load_from_command(self, path: str, *, kind: str, as_one: bool = False,
                          name: str = "") -> str:
        """`/local --path` and `/s3 --path`: the same loader the model uses,
        with the source stated explicitly (so a path is never guessed as the
        wrong kind) and `--as-one` available for a folder of Parquet files
        that is one dataset. Raises DataSourceError for a bad argument."""
        from ledgerhound.io import data_source as _ds
        from ledgerhound.io.minio import shared_settings
        src = _ds.parse(path, kind=kind,
                        default_bucket=shared_settings().get("bucket", "") if kind == "s3" else "")
        self._force_one_dataset = as_one
        self._dataset_name_override = (name or "").strip()
        try:
            return self._tool_load_data(src.uri)
        finally:
            self._force_one_dataset = False
            self._dataset_name_override = ""

    def _fetch_from_store(self, src):
        """Download an s3:// source into the cache, with progress lines."""
        from ledgerhound.io import data_source as _ds
        from ledgerhound.io.minio import shared_settings
        s = shared_settings()
        self.render(f"fetching {src.uri} from the object store "
                    f"({s.get('endpoint') or 'endpoint not set'}) ...")
        last = {"pct": -10}

        def _progress(done_b, total_b, done_n, total_n):
            pct = int(100 * done_b / total_b) if total_b else 100
            if pct >= last["pct"] + 10 or done_n == total_n:
                last["pct"] = pct
                self.render(f"  {pct:3d}%  {done_n}/{total_n} files, "
                            f"{_ds._human(done_b)} of {_ds._human(total_b)}")
        fr = _ds.fetch(src, settings=s, progress=_progress,
                       as_one=self._force_one_dataset)
        self._events.append("data_fetched", source=src.uri, local=str(fr.local_path),
                            files=fr.files, bytes=fr.total_bytes,
                            downloaded=fr.downloaded_bytes,
                            seconds=round(fr.seconds, 1))
        return fr

    def _set_run_paths(self, asked: str, resolved, fetched) -> str:
        """Point outputs at the right place for this data source, write
        run_dir/paths.json, and return a line for the model/user."""
        from ledgerhound.io import data_source as _ds
        self._uploader = None
        self._features_override = None
        note = ""
        if fetched is not None:
            if _ds.is_ram_backed(fetched.cache_dir):
                self._features_override = (fetched.cache_dir / "runs" /
                                           self.run_dir.name / "features")
            try:
                bucket, prefix = _ds.outputs_prefix(self.run_dir.name)
                self._uploader = _ds.OutputUploader(self._features_dir(), bucket,
                                                    prefix + "features/")
                note = (f"outputs: save feature tables under features_dir "
                        f"({self._features_dir()}); each run_python step uploads "
                        f"new or changed files there to {self._uploader.uri}")
            except Exception as e:  # noqa: BLE001
                note = (f"outputs: features_dir is {self._features_dir()} -- NOT "
                        f"uploaded to the object store ({e}).")
        rp = _ds.RunPaths(
            source="s3" if fetched is not None else "local",
            data=fetched.source.uri if fetched is not None else str(resolved),
            local_copy=str(fetched.local_path) if fetched is not None else "",
            run_dir=str(self.run_dir), scratch_dir=str(self.run_dir / "scratch"),
            features_dir=str(self._features_dir()),
            features_upload=self._uploader.uri if self._uploader else "")
        try:
            rp.save(self.run_dir)
        except OSError:
            pass
        self._run_paths = rp
        return note

    def run_paths(self):
        """The current RunPaths (for /paths); a local placeholder before any
        data is loaded."""
        if self._run_paths is not None:
            return self._run_paths
        from ledgerhound.io.data_source import RunPaths
        return RunPaths(source="(no data loaded yet)", run_dir=str(self.run_dir),
                        scratch_dir=str(self.run_dir / "scratch"),
                        features_dir=str(self.run_dir / "features"))

    def _sync_outputs(self) -> str:
        """Upload new/changed feature files (object-store sessions only).
        Returns a short note for the tool result, '' when nothing moved."""
        if self._uploader is None:
            return ""
        try:
            done, errs = self._uploader.sync()
        except Exception as e:  # noqa: BLE001 -- never break a step on upload
            done, errs = [], [f"upload failed: {type(e).__name__}: {e}"]
        if done:
            self._events.append("outputs_uploaded", target=self._uploader.uri,
                                files=done[:200])
        if errs:
            self._events.append("outputs_upload_failed", target=self._uploader.uri,
                                errors=errs[:50])
        parts = []
        if done:
            shown = ", ".join(done[:8]) + (f", +{len(done) - 8} more" if len(done) > 8 else "")
            parts.append(f"[uploaded to {self._uploader.uri}: {shown}]")
        if errs:
            parts.append("[upload FAILED -- files are kept locally in features_dir "
                         "and retried after the next step: " + "; ".join(errs[:3]) + "]")
        return "\n".join(parts)

    # -- path resolution (the thing that was dumb before) -----------------
    _loaded_path: str = ""

    def _candidate_roots(self) -> list[Path]:
        roots = [Path.cwd(), Path.cwd().parent]
        if self.project is not None:
            try:
                pr = Path(self.project.root)
                roots += [pr, pr.parent]        # data at ../ from the project
            except Exception:
                pass
        # de-dup, keep order
        seen, out = set(), []
        for r in roots:
            rp = r.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(r)
        return out

    def _resolve_dir(self, path: str) -> Path | None:
        p = Path(path)
        if p.exists():
            return p
        for root in self._candidate_roots():
            if (root / path).exists():
                return root / path
        return None

    def _resolve_data(self, path: str):
        """Return a resolved file Path, a list of candidate files (folder), or None."""
        tried: list[Path] = []
        p = Path(path)
        cands = [p] + [root / path for root in self._candidate_roots()]
        for c in cands:
            tried.append(c)
            if c.exists():
                if c.is_dir() and (_is_parquet_dataset_dir(c) or (
                        self._force_one_dataset and _dataset_parts(c) and not any(
                            f.suffix.lower() in _DATA_EXTS and f.suffix.lower() not in (".parquet", ".pq")
                            for f in c.rglob("*") if f.is_file()))):
                    self._loaded_path = str(c)
                    return c
                if c.is_dir():
                    files = [str(f) for f in sorted(c.iterdir())
                             if f.suffix.lower() in _DATA_EXTS]
                    return files or None
                self._loaded_path = str(c)
                return c
        # last resort: glob the basename under cwd/project (one level)
        name = p.name
        for root in self._candidate_roots():
            hits = list(root.glob(name)) + list(root.glob(f"*/{name}"))
            if len(hits) > 1:
                # I-19: several files share this basename -- silently
                # loading the first would analyse the wrong month/extract.
                # Return the choices, same as a folder listing does.
                return [str(h) for h in sorted(hits)]
            if hits:
                self._loaded_path = str(hits[0])
                return hits[0]
        return None


def _dataset_parts(path: Path) -> list[Path]:
    """Parquet part files under a dataset folder (hive partitions included)."""
    return sorted(f for f in Path(path).rglob("*")
                  if f.is_file() and f.suffix.lower() in (".parquet", ".pq"))


def _is_parquet_dataset_dir(path: Path) -> bool:
    """A folder that is ONE dataset split into Parquet parts -- what Spark,
    Trino exports and `to_parquet(partition_cols=...)` write -- rather than
    a folder holding several separate datasets (which still asks "which
    one?"). Recognised by hive `col=value` subfolders, a `_SUCCESS` marker,
    or every data file being a `part-*`/numbered Parquet part."""
    p = Path(path)
    if not p.is_dir():
        return False
    from ledgerhound.io.data_source import parts_form_one_dataset
    # Cheap on purpose: a user can point at a big folder (a home or project
    # dir), so never walk the whole tree -- top level, plus the inside of
    # `col=value` partition folders only.
    names: list[str] = []
    try:
        for e in p.iterdir():
            if e.is_file():
                names.append(e.name)
            elif e.is_dir() and "=" in e.name:
                for f in e.rglob("*"):
                    if f.is_file():
                        names.append(f.relative_to(p).as_posix())
                    if len(names) > 100_000:
                        break
    except OSError:
        return False
    return parts_form_one_dataset(names)


def _sha256_dataset_dir(path: Path) -> str:
    """Content hash of a Parquet dataset folder: sha256 over each part's
    relative path and its own sha256."""
    import hashlib
    from ledgerhound.runtime.events import sha256_file
    h = hashlib.sha256()
    for f in _dataset_parts(path):
        h.update(f"{f.relative_to(path).as_posix()}\0{sha256_file(f)}\n".encode())
    return h.hexdigest()


def _read_any(path: Path) -> pd.DataFrame:
    if Path(path).is_dir():
        return pd.read_parquet(path)          # a Parquet dataset folder
    s = path.suffix.lower()
    if s in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if s in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if s == ".tsv" else ",")
    if s == ".json":
        return pd.read_json(path)
    if s == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def _estimate_rows(path: Path) -> tuple[int | None, bool]:
    """(row_count, is_exact). Exact and free from parquet metadata; for CSV
    only a byte-scaled estimate is available without a full scan.

    The `is_exact` flag exists so an ESTIMATE is never reported to the user
    or logged as if it were the true row count -- callers must render an
    inexact count as "~N".
    """
    s = path.suffix.lower()
    try:
        if Path(path).is_dir():
            import pyarrow.parquet as pq
            return sum(int(pq.ParquetFile(f).metadata.num_rows)
                       for f in _dataset_parts(path)), True
        if s in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows), True
        if s in (".csv", ".tsv"):
            size = path.stat().st_size
            sep = "\t" if s == ".tsv" else ","
            head = pd.read_csv(path, sep=sep, nrows=1000)
            if len(head) < 1000:          # whole file fit in the probe: exact
                return int(len(head)), True
            with open(path, "rb") as f:
                probe = f.read(1_000_000)
            nl = probe.count(b"\n")
            if nl <= 1:
                return None, False
            return int(size / (len(probe) / nl)), False
    except Exception:  # noqa: BLE001 -- estimation is advisory
        return None, False
    return None, False


def _read_sample(path: Path, n_rows: int) -> pd.DataFrame | None:
    """Read at most `n_rows` for SCHEMA purposes only (columns + dtypes).
    Never used for statistics without being labelled as a sample."""
    s = path.suffix.lower()
    try:
        if Path(path).is_dir():
            import pyarrow.dataset as pads
            ds = pads.dataset(str(path), format="parquet", partitioning="hive")
            return ds.head(n_rows).to_pandas()
        if s in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=n_rows):
                return batch.to_pandas()
            return pf.read().to_pandas()
        if s in (".csv", ".tsv"):
            return pd.read_csv(path, sep="\t" if s == ".tsv" else ",", nrows=n_rows)
    except Exception:  # noqa: BLE001
        return None
    return None


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        sv = str(v)
        parts.append(f"{k}={sv[:40]}..." if len(sv) > 40 else f"{k}={sv}")
    return ", ".join(parts)
