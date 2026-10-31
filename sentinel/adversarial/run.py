"""
sentinel.adversarial.run
===========================

Top-level orchestrator wiring sentinel.adversarial.{sampling, engine,
shell_wiring} together into one `/adversarial` run against the PARENT
shell session's already-loaded dataset and declared goal (entity_key,
label_column, entity_key_secondary). See sentinel.cli.main's
`/adversarial` slash command for the actual entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from sentinel.adversarial.engine import DEFAULT_CASES, run_adversarial, write_report
from sentinel.adversarial.sampling import (
    extract_case_slice, extract_related_slice, related_accounts,
    sample_fraud_cases,
)
from sentinel.adversarial.shell_wiring import build_agent_factory, build_judge_llm


def preflight_judge(judge_llm, judge_spec, *, say=lambda s: None) -> None:
    """One minimal request to the judge before any debate. Raises ValueError
    with a plain, specific message (the CLI prints it, no traceback) when
    the judge cannot answer -- rejected key vs unreachable vs anything else."""
    from sentinel.llm.client import LLMAuthError
    where = f"{judge_spec.model} @ {judge_spec.base_url}"
    try:
        judge_llm.generate([{"role": "user", "content": "Reply with OK."}],
                           max_tokens=8)
    except LLMAuthError as e:
        raise ValueError(
            f"the judge ({where}) rejected its API key ({e}). Nothing was "
            "run. Fix the judge key -- judge.<method>.api_key in "
            "main_config.yaml, or /model-judge -- then run /adversarial "
            "again.") from e
    except ConnectionError as e:
        raise ValueError(
            f"the judge ({where}) did not answer ({e}). Nothing was run. "
            "Check the judge endpoint and network, then retry.") from e
    say(f"judge check: {where} answered.")


def safe_file_stem(value) -> str:
    """I-15: an entity id is data, not a filename -- a VPA like
    'name@okbank' or anything with '/', ':', spaces, or a very long value
    raised on write and failed the case. Keep [A-Za-z0-9._-], collapse
    the rest to '_', cap the length, and add a short hash so two ids that
    sanitise the same never collide."""
    import hashlib
    import re
    s = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-") or "id"
    digest = hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{cleaned[:60]}_{digest}"


def parse_adversarial_args(text: str) -> dict:
    """Parse `/adversarial --mode on --cases 25 --seed 7` into
    {"off": bool, "cases": int, "seed": int|None}. Raises ValueError with
    a plain, actionable message on a bad flag value -- the CLI just
    prints str(e) and stops, no traceback."""
    import shlex
    tokens = shlex.split(text or "")
    opts: dict = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--mode", "--cases", "--seed") and i + 1 < len(tokens):
            opts[t[2:]] = tokens[i + 1]
            i += 2
        else:
            i += 1
    off = opts.get("mode", "on").lower() in ("off", "false", "0")
    try:
        cases = int(opts.get("cases", DEFAULT_CASES))
    except ValueError:
        raise ValueError(f"--cases must be a number, got {opts.get('cases')!r}")
    seed = None
    if "seed" in opts:
        try:
            seed = int(opts["seed"])
        except ValueError:
            raise ValueError(f"--seed must be a number, got {opts['seed']!r}")
    return {"off": off, "cases": cases, "seed": seed}


def run_from_session(analyst, *, cases: int = DEFAULT_CASES,
                     seed: Optional[int] = None,
                     say: Callable[[str], None] = lambda s: None) -> tuple:
    """Run the full adversarial loop against `analyst`'s (a
    ConversationalAnalyst) currently loaded dataset and declared goal.

    Returns (results, report_path). Raises ValueError with a plain,
    actionable message (never a stack trace) for the ordinary "not set up
    yet" cases -- no data loaded, no label/entity key declared, no fraud
    rows found -- since those are genuinely the analyst's own next step to
    fix, not a bug in this module.
    """
    goal = analyst.goal
    if not analyst._loaded_path:
        raise ValueError(
            "no data loaded yet -- load_data first, then set_goal, then "
            "/adversarial.")
    if not goal.label_column:
        raise ValueError(
            "no label_column declared -- call set_goal(label_column=...) "
            "first so /adversarial knows which column marks fraud.")
    if not goal.entity_key:
        raise ValueError(
            "no entity_key declared -- call set_goal(entity_key=...) first "
            "so /adversarial knows the payer account column.")

    say(f"sampling up to {cases} fraud case(s) from {analyst._loaded_path}...")
    sampled = sample_fraud_cases(
        analyst._loaded_path, label_column=goal.label_column,
        entity_key=goal.entity_key,
        entity_key_secondary=goal.entity_key_secondary, n=cases, seed=seed)
    if not sampled:
        raise ValueError(
            f"no fraud rows found in column {goal.label_column!r} -- "
            "nothing to debate.")
    say(f"{len(sampled)} case(s) sampled.")

    run_root = analyst.run_dir / "adversarial"
    run_root.mkdir(parents=True, exist_ok=True)

    judge_llm, judge_spec = build_judge_llm(
        analyst.project, actor_base_url=analyst.llm.config.base_url,
        actor_model=analyst.llm.config.model,
        actor_key=analyst.llm.config.resolve_api_key(),
        # duck-typed clients (tests, embeddings) may lack these -- default
        # to the LLMConfig defaults rather than AttributeError.
        actor_verify_ssl=getattr(analyst.llm.config, "verify_ssl", True),
        actor_models_url=getattr(analyst.llm.config, "models_url", ""))
    say(f"judge: {judge_spec.model} @ {judge_spec.base_url} "
       f"({'self-judge' if judge_spec.self_judge else judge_spec.source})")
    if judge_spec.self_judge:
        if getattr(judge_spec, "fallback_reason", ""):
            say("WARNING: a judge IS configured but could not be used -- "
                f"{judge_spec.fallback_reason}. Falling back to the ACTOR's "
                "own model for the adversary and the verdict; fix the judge "
                "key/config to get an independent judge.")
        else:
            say("WARNING: no separate judge configured -- the adversary and "
                "the final verdict are running on the ACTOR's own model "
                "(/model-judge to set a real one). An actor judging its own "
                "work is weaker evidence than an independent judge.")

    # Item 8: prove the judge answers BEFORE 150-round debates start. A bad
    # judge key used to surface only at the first adversary turn of case 1,
    # then again for every case. One tiny request settles it up front.
    preflight_judge(judge_llm, judge_spec, say=say)

    # Item 1/2: size the judge's window for ITS model (it was left at the
    # client default), and count its segments with its own tokenizer.
    from sentinel.llm.catalog import apply_context_window
    from sentinel.llm.tokenizer import resolve_tokenizer
    judge_counter, tstatus = resolve_tokenizer(judge_spec.model)
    say("judge " + tstatus.render(
        fallback="judge segments are sized at the densest measured ratio "
                 "(1.45 chars/token), so a long debate is read in a few more "
                 "segments -- nothing is skipped."))
    say("judge " + apply_context_window(
        judge_llm, hint=(tstatus.details or {}).get("context_window")))

    build_agent = build_agent_factory(
        analyst_llm=analyst.llm, adversary_llm=judge_llm, out_root=run_root,
        use_kernel=analyst.use_kernel, renderer=say,
        analyst_counter=getattr(getattr(analyst, "_compressor", None), "counter", None),
        adversary_counter=judge_counter)

    def extract_payer(case: dict):
        out = run_root / f"payer_{safe_file_stem(case['entity_key'])}.parquet"
        n = extract_case_slice(
            analyst._loaded_path, out, account_id=case["entity_key"],
            payer_col=goal.entity_key, payee_col=goal.entity_key_secondary or "")
        return str(out), n

    def extract_payee(case: dict, payer_transcript: list):
        import pandas as pd
        payer_slice = run_root / f"payer_{safe_file_stem(case['entity_key'])}.parquet"
        if not payer_slice.exists():
            return str(run_root / "empty.parquet"), 0
        df = pd.read_parquet(payer_slice)
        related = related_accounts(
            df, case["entity_key"], payer_col=goal.entity_key,
            payee_col=goal.entity_key_secondary or "")
        out = run_root / f"payee_{safe_file_stem(case['entity_key'])}.parquet"
        n = extract_related_slice(
            analyst._loaded_path, out, account_ids=related,
            payer_col=goal.entity_key, payee_col=goal.entity_key_secondary or "")
        return str(out), n

    report_path = run_root / "report.json"
    done: list = []

    def _checkpoint(result) -> None:
        # Written after EVERY case: Ctrl+C or a crash in case 20 keeps the
        # 19 verdicts already paid for.
        done.append(result)
        try:
            write_report(done, report_path)
        except Exception:  # noqa: BLE001 -- the final write below retries
            pass

    results = run_adversarial(
        cases=sampled, build_agent=build_agent, judge_llm=judge_llm,
        extract_payer_slice=extract_payer, extract_payee_slice=extract_payee,
        say=say, on_case_done=_checkpoint, judge_counter=judge_counter)

    write_report(results, report_path)
    n_flipped = sum(1 for r in results if r.flipped)
    say(f"done: {len(results)} case(s), {n_flipped} flagged as possibly "
       f"mislabelled. Report: {report_path}")
    return results, report_path
