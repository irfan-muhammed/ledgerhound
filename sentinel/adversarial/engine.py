"""
sentinel.adversarial.engine
==============================

The `/adversarial` deep-research loop.

Why this exists: fraud is a tiny fraction of the data, and a single
analyst pass tends to accept its own first plausible story. The team
found a real case of this going wrong the other direction too -- a run
that flipped fraud and non-fraud labels wholesale. The idea here is
adversarial pressure as a check: for each of a SAMPLE of fraud cases, two
full agents with full tool access argue it out against that one case's
own data slice (the fraud entity + every account it touched) --

  - the ANALYST (on the regular/base model) has to build a concrete,
    evidence-based case for why this is fraud, every round.
  - the ADVERSARY (on the JUDGE model) is never convinced -- demands
    evidence, proposes innocent explanations, checks the analyst's own
    numbers with its own tool calls.

Fixed iteration budget per case (not configurable per-call by design --
change the module constants if the split itself needs to change):
  - PAYER side (the fraud side): 100 rounds total, 50 analyst + 50
    adversary, alternating.
  - PAYEE side (the counterparty -- same scrutiny is not needed, but
    isn't skipped either): 50 rounds total, 25 + 25.
Then a THIRD pass, also on the judge model, reads the ENTIRE transcript
(both sides, every round) and delivers one verdict for that case --
explicitly instructed not to rubber-stamp the original label.

Each agent is a real ConversationalAnalyst (sentinel.shell.conversation)
-- full tool access (run_python, record_finding, load_data, GPU/kernel
mode, everything an ordinary shell session has), not a stripped-down
loop. "Tool-calling limit: infinity" is implemented as a very large
ceiling (UNBOUNDED_MAX_STEPS) rather than a literal unbounded loop --
the deliberate budget is the ROUND count above, not tool calls within a
round.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

DEFAULT_CASES = 25
PAYER_ROUNDS = 100    # 50 analyst + 50 adversary
PAYEE_ROUNDS = 50     # 25 + 25
# Not literally infinite (a real loop needs SOME ceiling) -- large enough
# that it is never what cuts a debate round short. The round counts above
# are the actual, deliberate budget.
UNBOUNDED_MAX_STEPS = 100_000
UNBOUNDED_STEP_BUDGET = 100_000
UNBOUNDED_TURN_BUDGET_S = 24 * 3600

_ANALYST_PERSONA = (
    "\n\n=== ADVERSARIAL DEBATE ROLE: ANALYST ===\n"
    "This entity is currently flagged as fraud. Your job across this "
    "debate is to build a concrete, evidence-based case for WHY that "
    "label is correct -- specific transactions, amounts, timing, "
    "counterparties, network structure, using your tools on the loaded "
    "case data. Every round, respond directly to the adversary's last "
    "challenge with NEW evidence or a sharper argument -- repeating an "
    "earlier point without adding anything is a losing move here."
)
_ADVERSARY_PERSONA = (
    "\n\n=== ADVERSARIAL DEBATE ROLE: ADVERSARY ===\n"
    "You are NEVER convinced by assertion alone. For every claim the "
    "analyst makes, demand specific evidence, propose a plausible "
    "innocent explanation, and use your OWN tool calls on the same case "
    "data to independently check whether the analyst's evidence actually "
    "holds up. Find the weakest point in the fraud case, every round, "
    "without exception -- but if the evidence is genuinely conclusive, "
    "say so; the point is rigor, not contrarianism for its own sake."
)
_JUDGE_SYSTEM = (
    "You are the JUDGE in an adversarial fraud-review debate. Read the "
    "ENTIRE transcript below carefully -- every round, both sides, both "
    "the payer-side and payee-side debates -- before deciding anything. "
    "Then deliver ONE careful, evidence-grounded verdict for this case: "
    "is the fraud label correct? Cite specific rounds/evidence. If the "
    "adversary raised a point the analyst never actually answered, say so "
    "explicitly. Do not default to agreeing with the original label -- "
    "your job is to catch a wrong one, not rubber-stamp it. End your "
    "reply with exactly one line: 'VERDICT: CONFIRMED FRAUD' or "
    "'VERDICT: LABEL QUESTIONABLE' or 'VERDICT: LIKELY MISLABELLED'."
)


@dataclass
class CaseResult:
    case_id: str
    entity_key: str
    n_payer_rows: int = 0
    n_payee_related: int = 0
    payer_transcript: list = field(default_factory=list)
    payee_transcript: list = field(default_factory=list)
    verdict_text: str = ""
    verdict_label: str = ""     # "CONFIRMED FRAUD" | "LABEL QUESTIONABLE" | "LIKELY MISLABELLED" | ""
    flipped: bool = False       # True when the judge disagrees with the original fraud label

    def to_dict(self) -> dict:
        return asdict(self)


def _round_prompt(round_no: int, total_rounds: int, last_reply: str) -> str:
    if round_no == 1:
        return ("Round 1 -- begin. State your opening position with "
                "concrete evidence from the loaded case data (call your "
                "tools first; don't answer from assumption).")
    return (f"Round {round_no}/{total_rounds}. The other side just said:\n\n"
           f"{last_reply}\n\nRespond directly to it -- new evidence or a "
           "direct rebuttal, not a repeat of your earlier points.")


def render_transcript(transcript: list) -> str:
    return "\n\n".join(
        f"--- round {t['round']} ({t['role']}) ---\n{t['text']}"
        for t in transcript)


def run_debate(analyst, adversary, *, total_rounds: int, side: str,
              say: Callable[[str], None] = lambda s: None) -> list:
    """Alternate `total_rounds` turns between two ALREADY-CONSTRUCTED
    agents (analyst goes first each side), each getting the other's last
    reply as its next prompt. Returns
    [{"role": "analyst"|"adversary", "round": int, "text": str}, ...]."""
    transcript: list = []
    last_reply = ""
    current, role = analyst, "analyst"
    other, other_role = adversary, "adversary"
    for round_no in range(1, total_rounds + 1):
        prompt = _round_prompt(round_no, total_rounds, last_reply)
        say(f"[{side} {round_no}/{total_rounds}] {role}...")
        reply = current.send(prompt)
        transcript.append({"role": role, "round": round_no, "text": reply})
        last_reply = reply
        current, role, other, other_role = other, other_role, current, role
    return transcript


# I-08: the judge's reading budget. 150 rounds of full-length replies can
# run to several hundred thousand tokens -- more than any judge window --
# and a 400 from the gateway after all that debate lost the whole case.
#
# Sized in TOKENS: the judge's window minus its reply reservation
# (max_tokens) minus room for its instructions, 80% of what is left. Counted
# with the judge model's own tokenizer when one is available; otherwise at
# the densest ratio measured on this data (Gemma 4, 1.48 chars/token) --
# the old 3.6 undercounted Qwen/Gemma output by more than 2x. Debate prose
# is usually sparser, so this over-segments a little: an extra digest call
# is cheap, an overflowing verdict call loses the case.
JUDGE_CHARS_PER_TOKEN_FLOOR = 1.45
JUDGE_HEADROOM_TOKENS = 4_096
JUDGE_MIN_SEGMENT_TOKENS = 1_000
JUDGE_MAX_TOKENS_DEFAULT = 60_000       # judge window unknown
_JUDGE_CHUNK_SYSTEM = (
    "You are the JUDGE's note-taker in an adversarial fraud-review debate. "
    "This is ONE segment of a longer transcript. Write a faithful, dense "
    "digest of it for the judge who will read every segment's digest and "
    "decide: every concrete piece of evidence each side produced (numbers, "
    "transactions, counterparties, timing), every challenge the adversary "
    "raised, and whether the analyst actually answered it. Cite round "
    "numbers. Do not give a verdict; do not omit anything a careful judge "
    "would want."
)


def _judge_budget_tokens(judge_llm) -> int:
    cfg = getattr(judge_llm, "config", None)
    try:
        cw = int(getattr(cfg, "context_window", 0) or 0)
        reserve = int(getattr(cfg, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        cw, reserve = 0, 0
    if not cw:
        return JUDGE_MAX_TOKENS_DEFAULT
    return max(JUDGE_MIN_SEGMENT_TOKENS,
               int((cw - reserve - JUDGE_HEADROOM_TOKENS) * 0.8))


def _token_counter(counter):
    """count(text) -> tokens: the judge's own tokenizer, or the density floor."""
    if counter is not None:
        return counter.count
    return lambda t: int(len(t or "") / JUDGE_CHARS_PER_TOKEN_FLOOR) + 1


def _chunk_by_tokens(sections: list, budget: int, count) -> list:
    """Group consecutive (title, text) sections so each group stays under
    `budget` tokens. A single section over budget gets its own group and is
    cut (head + tail, proportionally) -- a truncated round beats a rejected
    request."""
    groups, cur, cur_tok = [], [], 0
    for title, text in sections:
        n = count(text)
        if n > budget:
            keep = max(200, int(len(text) * budget / max(n, 1)))
            head, tail = int(keep * 0.66), int(keep * 0.33)
            text = (text[:head] + "\n[... middle of this round elided for "
                    "length ...]\n" + text[-tail:])
            n = count(text)
        if cur and cur_tok + n > budget:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append((title, text))
        cur_tok += n
    if cur:
        groups.append(cur)
    return groups


def judge_case(judge_llm, payer_transcript: list, payee_transcript: list,
              case_id: str, *, max_chars: int | None = None,
              counter=None) -> tuple:
    """One careful reading pass over the WHOLE transcript -- a plain
    LLMClient.generate() call, not a tool-calling agent; the judge's job
    here is to read and reason over what already happened, not run more
    analysis. Returns (full_verdict_text, extracted_label).

    I-08: when the transcript does not fit the judge's window it is read
    in consecutive segments first (each digested by the same judge model
    with _JUDGE_CHUNK_SYSTEM, nothing dropped silently), and the verdict
    is then delivered over the digests. Everything still gets read; it is
    just read in an order that fits."""
    count = _token_counter(counter)
    budget = (int(max_chars / JUDGE_CHARS_PER_TOKEN_FLOOR) if max_chars
              else _judge_budget_tokens(judge_llm))
    sections = (
        [(f"PAYER-SIDE round {t['round']} ({t['role']})", t["text"] or "")
         for t in payer_transcript]
        + [(f"PAYEE-SIDE round {t['round']} ({t['role']})", t["text"] or "")
           for t in payee_transcript])

    def _render(group):
        return "\n\n".join(f"--- {title} ---\n{text}" for title, text in group)

    groups = _chunk_by_tokens(sections, budget, count)
    if len(groups) <= 1:
        full = (f"=== DEBATE TRANSCRIPT (case {case_id}) ===\n"
                f"{_render(groups[0]) if groups else '(empty)'}")
    else:
        digests = []
        for i, group in enumerate(groups, start=1):
            resp = judge_llm.generate([
                {"role": "system", "content": _JUDGE_CHUNK_SYSTEM},
                {"role": "user",
                 "content": f"Segment {i}/{len(groups)} of case {case_id}:\n\n"
                            f"{_render(group)}"}])
            digests.append(f"=== SEGMENT {i}/{len(groups)} DIGEST ===\n"
                           f"{(resp.content or '').strip()}")
        full = (f"=== DEBATE TRANSCRIPT (case {case_id}), read in "
                f"{len(groups)} segments -- these are your own digests of "
                "each segment, in order ===\n\n" + "\n\n".join(digests))
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": full},
    ]
    resp = judge_llm.generate(messages)
    text = resp.content or ""
    label = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            label = line.split(":", 1)[1].strip().upper()
    return text, label


def _close_agents(*agents) -> None:
    """I-09: release each debate agent's resources (a live kernel holding
    a CUDA context, in kernel mode) as soon as its side is over. The
    registry holds bound methods of the agent, so refcounting never frees
    it on its own -- without this, kernels piled up across cases."""
    for a in agents:
        close = getattr(a, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- teardown is best-effort
                pass


def run_case(*, case_id: str, entity_key, entity_key_secondary,
            build_agent: Callable[[str, str], object],
            judge_llm, extract_payer_slice: Callable[[], tuple],
            extract_payee_slice: Callable[[object], tuple],
            say: Callable[[str], None] = lambda s: None,
            judge_counter=None) -> CaseResult:
    """Run one case end to end: payer debate, payee debate, judge verdict.

    `build_agent(role, data_path)` constructs a ready-to-debate
    ConversationalAnalyst for "analyst" or "adversary", already pointed at
    `data_path` (see sentinel.adversarial.shell_wiring for the real
    factory -- kept out of this module so the debate mechanics stay
    testable without a real LLM/kernel).
    `extract_payer_slice()` -> (data_path, n_rows) for the fraud entity's
    own transaction neighbourhood.
    `extract_payee_slice(payer_transcript)` -> (data_path, n_rows) for the
    payee-side accounts worth examining next (typically derived from
    what surfaced during the payer debate, or just the raw counterparties).
    """
    result = CaseResult(case_id=case_id, entity_key=str(entity_key))

    payer_path, n_payer = extract_payer_slice()
    result.n_payer_rows = n_payer
    if n_payer == 0:
        result.verdict_text = "skipped -- no transactions found for this entity"
        return result

    payer_analyst = build_agent("analyst", payer_path)
    payer_adversary = build_agent("adversary", payer_path)
    try:
        result.payer_transcript = run_debate(
            payer_analyst, payer_adversary, total_rounds=PAYER_ROUNDS,
            side="payer", say=say)
    finally:
        _close_agents(payer_analyst, payer_adversary)

    payee_path, n_payee = extract_payee_slice(result.payer_transcript)
    result.n_payee_related = n_payee
    if n_payee > 0:
        payee_analyst = build_agent("analyst", payee_path)
        payee_adversary = build_agent("adversary", payee_path)
        try:
            result.payee_transcript = run_debate(
                payee_analyst, payee_adversary, total_rounds=PAYEE_ROUNDS,
                side="payee", say=say)
        finally:
            _close_agents(payee_analyst, payee_adversary)

    say(f"[{case_id}] judging...")
    verdict_text, label = judge_case(
        judge_llm, result.payer_transcript, result.payee_transcript, case_id,
        counter=judge_counter)
    result.verdict_text = verdict_text
    result.verdict_label = label
    result.flipped = label in ("LABEL QUESTIONABLE", "LIKELY MISLABELLED")
    return result


def run_adversarial(*, cases: list, build_agent: Callable[[str, str], object],
                    judge_llm, extract_payer_slice: Callable[[dict], tuple],
                    extract_payee_slice: Callable[[dict, list], tuple],
                    say: Callable[[str], None] = lambda s: None,
                    on_case_done: Callable[[CaseResult], None] = lambda r: None,
                    judge_counter=None,
                    ) -> list:
    """Drive run_case over every sampled case
    (sentinel.adversarial.sampling.sample_fraud_cases output). Never
    raises on a single case's failure -- one bad case must not lose the
    results already gathered for the others.

    Exception: a REJECTED API KEY (LLMAuthError) is not a case failure --
    every remaining case would fail the same way, one slow debate at a
    time. The run stops there with one message; the cases not reached are
    recorded as not run, and everything already finished is kept."""
    from sentinel.llm.client import LLMAuthError
    results = []
    for i, case in enumerate(cases, start=1):
        case_id = f"case_{i:03d}_{case.get('entity_key')}"
        say(f"=== {case_id} ({i}/{len(cases)}) ===")
        try:
            result = run_case(
                case_id=case_id, entity_key=case.get("entity_key"),
                entity_key_secondary=case.get("entity_key_secondary"),
                build_agent=build_agent, judge_llm=judge_llm,
                extract_payer_slice=lambda c=case: extract_payer_slice(c),
                extract_payee_slice=lambda transcript, c=case:
                    extract_payee_slice(c, transcript),
                say=say, judge_counter=judge_counter)
        except LLMAuthError as e:
            say(f"STOPPED: an API key was rejected during {case_id} ({e}). "
                "Every remaining case would fail the same way, so the run "
                f"ends here: {len(results)} case(s) finished, "
                f"{len(cases) - len(results)} not run. Fix the key (/model-"
                "judge for the judge/adversary, or the actor's key), then "
                "run /adversarial again.")
            aborted = CaseResult(case_id=case_id,
                                 entity_key=str(case.get("entity_key")),
                                 verdict_text=f"not run: API key rejected ({e})")
            results.append(aborted)
            on_case_done(aborted)
            for j, rest in enumerate(cases[i:], start=i + 1):
                skipped = CaseResult(
                    case_id=f"case_{j:03d}_{rest.get('entity_key')}",
                    entity_key=str(rest.get("entity_key")),
                    verdict_text="not run: stopped after an API key was rejected")
                results.append(skipped)
                on_case_done(skipped)
            break
        except Exception as e:  # noqa: BLE001
            say(f"[{case_id}] FAILED: {e}")
            result = CaseResult(case_id=case_id,
                                entity_key=str(case.get("entity_key")),
                                verdict_text=f"error: {e}")
        else:
            pass
        results.append(result)
        on_case_done(result)
    return results


def write_report(results: list, out_path: str | Path) -> None:
    """One JSON summary across every case -- flags surfaced first, so a
    human scanning it sees exactly which fraud labels the adversary/judge
    thinks are shaky, without reading 150 rounds x 25 cases to find out."""
    flipped = [r for r in results if r.flipped]
    summary = {
        "n_cases": len(results),
        "n_flipped": len(flipped),
        "flipped_case_ids": [r.case_id for r in flipped],
        "cases": [r.to_dict() for r in results],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2, default=str),
                              encoding="utf-8")
