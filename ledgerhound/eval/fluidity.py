"""
ledgerhound.eval.fluidity
=======================

V3-004 -- fluidity metrics for the conversational shell.

The golden-eval harness (``eval/golden.py``) scores whether an agent's
ANSWER is correct. It says nothing about whether getting there felt like
"begging the AI" -- how many times it stopped to ask the user something it
could have decided itself, how much work it redid, and how well it recovered
from its own mistakes. This module computes those numbers from a session's
transcript, independent of whether the underlying answer was right.

Five metrics, all computed from ``transcript.jsonl``-shaped event dicts
(``{"event": ..., ...}``) plus the list of user-facing replies emitted
during the run:

  questions_asked         -- how many times the agent ended a turn with a
                              question to the user instead of proceeding
                              (each one is a real interruption to the flow)
  redundant_recomputes     -- tool calls that repeat work already done this
                              session (profile_data called again after being
                              cached; run_python re-running byte-identical
                              code)
  unrecovered_errors       -- distinct (tool, failure-signature) pairs that
                              failed 2+ times identically without the agent
                              changing its approach
  error_recovery_rate      -- (V3-028) of every distinct (tool, signature)
                              pair that failed at least once, the fraction
                              that did NOT go on to fail identically again --
                              i.e. the agent changed approach and moved on.
                              None (not 0.0 or 1.0) when no errors occurred
                              at all -- recovery isn't defined without a
                              failure to recover from.
  completed                -- whether the session produced concrete output
                              (a recorded finding or a saved report), not
                              just conversation

A sixth, cross-session metric -- pass^k -- is computed by
``pass_at_k()`` below over MULTIPLE independent reruns of the same task,
not from a single transcript, so it isn't a ``FluidityResult`` field.

Pure functions over data structures -- no LLM, no I/O -- so they are cheap
to unit test and safe to run against any transcript, live or replayed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_QUESTION_RE = re.compile(r"\?\s*$")


@dataclass
class FluidityResult:
    task_id: str = ""
    questions_asked: int = 0
    redundant_recomputes: int = 0
    unrecovered_errors: int = 0
    total_distinct_errors: int = 0    # V3-028: denominator for error_recovery_rate
    completed: bool = False
    turns: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def error_recovery_rate(self) -> Optional[float]:
        """None when no (tool, signature) pair ever failed -- recovery is
        undefined, not automatically a perfect 1.0, when there was nothing
        to recover from."""
        if self.total_distinct_errors == 0:
            return None
        recovered = self.total_distinct_errors - self.unrecovered_errors
        return recovered / self.total_distinct_errors

    def to_row(self) -> str:
        rate = self.error_recovery_rate
        rate_str = f"{rate:.2f}" if rate is not None else "n/a"
        return (f"| {self.task_id} | {self.questions_asked} | "
                f"{self.redundant_recomputes} | {self.unrecovered_errors} | "
                f"{rate_str} | {'yes' if self.completed else 'no'} | {self.turns} |")


def is_question(reply: str) -> bool:
    """A final turn that reads as a question to the user rather than a
    completed answer or a decision-ready pause (report #1 SS4.2 distinguishes
    the two -- a decision-ready pause states findings + options + a
    recommendation, which this heuristic treats as NOT a bare question)."""
    text = (reply or "").strip()
    if not text:
        return False
    if not _QUESTION_RE.search(text):
        return False
    # a decision-ready package names its own recommendation -- not a bare ask
    low = text.lower()
    if "recommend" in low and ("option" in low or "1)" in text or "1." in text):
        return False
    return True


def _norm_args(args: dict) -> str:
    """Stable, order-independent string form of tool args for equality
    checks (code strings compared verbatim -- whitespace differences are a
    real difference in what was executed)."""
    if not isinstance(args, dict):
        return str(args)
    return str(sorted(args.items()))


def _error_signature(result: str) -> str | None:
    if not isinstance(result, str) or not result.startswith("Error:"):
        return None
    # "Error: KeyError: 'amnt'\n..." -> "KeyError: 'amnt'" (first line only,
    # so a fuzzy-match hint appended after it doesn't defeat the signature)
    first_line = result.splitlines()[0]
    return first_line[len("Error:"):].strip()


def compute_fluidity(events: list[dict], replies: list[str],
                      task_id: str = "") -> FluidityResult:
    """Compute fluidity metrics from one session's transcript events
    (as written by ``ConversationalAnalyst._log``) and the sequence of
    final-turn replies returned by ``send()`` across the session.

    ``events`` is the parsed contents of transcript.jsonl for the session
    (list of dicts with an ``"event"`` key). ``replies`` is every string
    returned by ``.send()`` calls made during the session, in order.
    """
    r = FluidityResult(task_id=task_id, turns=len(replies))

    for reply in replies:
        if is_question(reply):
            r.questions_asked += 1

    # -- redundant recompute: repeated profile_data calls, or run_python
    # calls with byte-identical code, both signal recomputation of work
    # already done this session.
    profile_calls = 0
    seen_code: dict[str, int] = {}
    for ev in events:
        if ev.get("event") != "tool_call":
            continue
        name = ev.get("name")
        args = ev.get("args") or {}
        if name == "profile_data":
            profile_calls += 1
        elif name == "run_python":
            code = str(args.get("code", ""))
            seen_code[code] = seen_code.get(code, 0) + 1
    if profile_calls > 1:
        r.redundant_recomputes += profile_calls - 1
        r.notes.append(f"profile_data called {profile_calls} times")
    for code, n in seen_code.items():
        if n > 1 and code.strip():
            r.redundant_recomputes += n - 1
            r.notes.append(f"run_python repeated identical code {n} times")

    # -- unrecovered errors: same (tool, error signature) failing 2+ times
    # V3-012: tool_call (args) and tool_result (result) are now separate
    # events -- the result lives on tool_result.
    fail_counts: dict[tuple[str, str], int] = {}
    for ev in events:
        if ev.get("event") != "tool_result":
            continue
        sig = _error_signature(ev.get("result", ""))
        if sig is None:
            continue
        key = (ev.get("name", ""), sig)
        fail_counts[key] = fail_counts.get(key, 0) + 1
    r.total_distinct_errors = len(fail_counts)
    for (name, sig), n in fail_counts.items():
        if n >= 2:
            r.unrecovered_errors += 1
            r.notes.append(f"{name} failed identically ({sig}) {n} times")

    # -- completion: did the session produce concrete output?
    for ev in events:
        if ev.get("event") == "tool_call" and ev.get("name") in (
                "record_finding", "save_report"):
            r.completed = True
            break

    return r


def render_report(results: list[FluidityResult], *, label: str = "") -> str:
    lines = [f"# Fluidity baseline{f' -- {label}' if label else ''}", ""]
    n = len(results) or 1
    totals = FluidityResult(task_id="TOTAL")
    lines.append("| task | questions asked | redundant recomputes | "
                  "unrecovered errors | error recovery rate | completed | turns |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(r.to_row())
        totals.questions_asked += r.questions_asked
        totals.redundant_recomputes += r.redundant_recomputes
        totals.unrecovered_errors += r.unrecovered_errors
        totals.total_distinct_errors += r.total_distinct_errors
        totals.turns += r.turns
    completed_n = sum(1 for r in results if r.completed)
    overall_rate = totals.error_recovery_rate
    overall_rate_str = f"{overall_rate:.2f}" if overall_rate is not None else "n/a"
    lines.append(f"| **avg / {len(results)} tasks** | "
                  f"{totals.questions_asked / n:.2f} | "
                  f"{totals.redundant_recomputes / n:.2f} | "
                  f"{totals.unrecovered_errors / n:.2f} | "
                  f"{overall_rate_str} | "
                  f"{completed_n}/{len(results)} | "
                  f"{totals.turns / n:.2f} |")
    lines.append("")
    for r in results:
        if r.notes:
            lines.append(f"**{r.task_id}:**")
            for note in r.notes:
                lines.append(f"  - {note}")
    return "\n".join(lines)


# -- pass^k (V3-028) ---------------------------------------------------------


def task_passed(result: FluidityResult) -> bool:
    """Operational pass criterion for one rerun of one task: it produced
    concrete output AND didn't get visibly stuck on a repeated identical
    failure. Deliberately independent of golden-answer correctness --
    that's ledgerhound.eval.golden_scorer's job; this is "did the agent
    functionally get somewhere," the fluidity harness's whole concern."""
    return result.completed and result.unrecovered_errors == 0


def pass_at_k(reruns_by_task: dict[str, list[FluidityResult]]) -> dict[str, dict]:
    """pass^k per task: did AT LEAST ONE of k independent reruns pass
    (``task_passed``)? Standard pass@k semantics (Chen et al.), applied to
    the fluidity pass/fail criterion rather than a golden-answer check.

    ``reruns_by_task``: {task_id: [FluidityResult, ...]} -- one entry per
    independent rerun of that task (k reruns -> k entries).

    Returns {task_id: {"k": int, "n_passed": int, "passed_at_k": bool}}.

    Determinism caveat, read before trusting this number: with a
    SCRIPTED/deterministic LLM (the CI-runnable subset's own convention --
    see eval/fluidity_ci_subset.py), every rerun of the same task is
    byte-for-byte identical, so pass^k there degenerates to "did this one
    deterministic run pass, repeated k times" -- a regression guard (the
    scripted trajectory keeps passing across k independent executions of
    the harness, catching nondeterminism bugs in the harness ITSELF, e.g.
    a flaky tmp-path race), not a measurement of real model-to-model
    variance. pass^k only measures genuine non-determinism when computed
    from k independent REAL-LLM reruns (eval/fluidity_baseline.py's live
    mode) -- not built/run in this closure; see the module docstring.
    """
    out: dict[str, dict] = {}
    for task_id, results in reruns_by_task.items():
        k = len(results)
        n_passed = sum(1 for r in results if task_passed(r))
        out[task_id] = {"k": k, "n_passed": n_passed, "passed_at_k": n_passed > 0}
    return out


def pass_at_k_rate(verdicts: dict[str, dict]) -> Optional[float]:
    """Fraction of tasks that passed at least once across their k reruns.
    None if verdicts is empty (undefined, not 0.0)."""
    if not verdicts:
        return None
    return sum(1 for v in verdicts.values() if v["passed_at_k"]) / len(verdicts)
