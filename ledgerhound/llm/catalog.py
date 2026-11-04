"""
ledgerhound.llm.catalog
======================

Model discovery + "pick the best one" for whatever LLM endpoint is
configured. Most OpenAI-compatible gateways (vLLM, TensorRT-LLM, the org's
gateway gateway, DeepSeek, etc.) expose ``GET /models`` listing what is
actually served right now -- this queries that (``LLMClient.list_models``),
ranks the results with a small, inspectable preference order (never a
"the AI decided" black box), and returns the top one as the default so
`ledgerhound shell` never has to block on a prompt unless the analyst wants
to look at the list.

Ranking is a maintained list, not a live capability probe or a benchmark
lookup -- it encodes what we currently know performs best on this
workload (thinking-mode / larger variants over flash-tier ones), and
should be revised as the org's gateway catalogue changes. It is deliberately
simple enough to read in one screen: anyone can see exactly why a model
was picked.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Callable, Optional

# Preference order, highest first: regex matched against a model id
# (case-insensitive, `search` not `match`). First pattern that hits wins;
# a later pattern never overrides an earlier one even if both match.
_PREFERENCE_PATTERNS = [
    r"thinking",
    r"\bpro\b",
    r"kimi-k2",
    r"deepseek-v\d.*pro",
    r"deepseek",
    r"glm-4\.?\d*",
    r"qwen3.*235b",
    r"gpt-4o(?!-mini)",
]
# Anything matching these, and NOT already caught above, is pushed to the
# back rather than left in gateway order -- these patterns mark
# deliberately smaller/faster/distilled variants.
_DEPRIORITISE_PATTERNS = [r"flash", r"\bmini\b", r"\blite\b", r"distill"]


@dataclass
class ModelChoice:
    id: str
    rank: int   # 0 = best; ties keep the gateway's own listing order


def _score(model_id: str) -> tuple:
    low = model_id.lower()
    for i, pat in enumerate(_PREFERENCE_PATTERNS):
        if re.search(pat, low):
            return (0, i)
    for pat in _DEPRIORITISE_PATTERNS:
        if re.search(pat, low):
            return (2, 0)
    return (1, 0)


def rank_models(model_ids: list) -> list:
    """Stable-sort model ids best-first (a tie keeps the gateway's own
    order -- some servers list the primary deployment first, that's
    signal worth keeping rather than discarding)."""
    scored = sorted(enumerate(model_ids), key=lambda t: (_score(t[1]), t[0]))
    return [ModelChoice(id=mid, rank=i) for i, (_, mid) in enumerate(scored)]


def best_model(model_ids: list) -> Optional[str]:
    ranked = rank_models(model_ids)
    return ranked[0].id if ranked else None


# Known context windows, checked in order (first regex match wins). Used
# to auto-size ledgerhound.shell.compressor's compaction threshold to
# whatever model actually got selected, instead of a single fixed
# default (256k) that's wrong for most models and silently stale for
# whatever comes next (Kimi K3, ...). Deliberately a maintained allow-
# list, not a guess: an unrecognised model id (a new K3 release before
# this list is updated, a custom deployment) falls through to `None` --
# callers keep whatever context_window was already configured rather
# than silently mis-sizing compaction against a wrong number.
_CONTEXT_WINDOWS = [
    # Model MAXIMUMS from each model's own config.json (max_position_
    # embeddings), checked 2026-09-18 against the files in the team's
    # tokenizer store. First match wins, so more specific rows come first.
    # A deployment may serve LESS than this -- see apply_context_window.
    (r"kimi-k3", 1_048_576),
    (r"kimi-k2\.[5-9]", 262_144),             # K2.5, K2.6, K2.7(-Code)
    (r"kimi-k2-(?:instruct-0905|thinking)", 262_144),
    (r"kimi-linear", 1_048_576),
    (r"kimi-k2", 131_072),
    (r"deepseek-v4", 1_048_576),               # V4 Flash / Pro, V4.1
    (r"deepseek-(?:v3|r1)", 163_840),
    (r"deepseek", 128_000),                    # hosted deepseek-chat API
    (r"glm-5\.[2-9]", 1_048_576),
    (r"glm-5", 202_752),
    (r"glm-4\.[67]", 202_752),
    (r"glm-4\.5", 131_072),
    (r"glm-4", 128_000),
    (r"gemma-?4-?e[24]b", 131_072),
    (r"gemma-?4", 262_144),
    (r"qwen-?3\.[5-9]", 262_144),             # qwen-3.6-35B-A3B on gateway
    (r"qwen-?3.*(?:2507|next|coder)", 262_144),
    (r"qwen-?3", 40_960),
    (r"gpt-4o", 128_000),
]

# Keys a gateway's model entry may use for its real context window. vLLM
# and TensorRT-LLM front-ends report `max_model_len`; a few others use
# one of the rest. The gateway's own number always beats the regex table
# above -- it is the deployment's actual limit, not our recollection.
_LISTING_WINDOW_KEYS = ("max_model_len", "context_length", "context_window",
                        "max_context_length", "n_ctx")


def context_window_from_listing(model_id: str, listing: list) -> Optional[int]:
    """The context window the gateway itself reports for `model_id` in
    its models listing (LLMClient.list_models_detailed), or None when it
    reports nothing usable (I-11)."""
    for entry in listing or []:
        if not isinstance(entry, dict) or entry.get("id") != model_id:
            continue
        for key in _LISTING_WINDOW_KEYS:
            v = entry.get(key)
            if v is None and isinstance(entry.get("meta"), dict):
                v = entry["meta"].get(key)
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
    return None


# A model's config says what it CAN take; a deployment is often started with
# less (vLLM --max-model-len). Until the gateway reports its own limit, a
# known-model value is not trusted above this -- overstating the window is
# the dangerous direction (requests rejected mid-session), understating only
# compacts a little early.
ASSUMED_WINDOW_CAP = 262_144


def apply_context_window(client, *, hint: Optional[int] = None) -> str:
    """Set client.config.context_window for the model in use and return the
    one-line reason, in order of trust:

      1. an explicit `context_window` under llm.<method> in main_config.yaml
      2. the limit the gateway reports in its models listing
      3. the model's own config (`hint`, from the tokenizer manifest) or the
         table above -- capped at ASSUMED_WINDOW_CAP
      4. unchanged default, said to be an assumption
    Shared by the shell (actor) and /adversarial (judge)."""
    cfg = getattr(client, "config", None)
    if cfg is None or not hasattr(cfg, "context_window"):
        return "context window: unknown (this client exposes no config) -- left as is"
    model = getattr(cfg, "model", "") or "(unnamed model)"
    if getattr(cfg, "context_window_explicit", False):
        return (f"context window: {cfg.context_window:,} (set in "
                f"main_config.yaml, for {model})")
    try:
        listing = (client.list_models_detailed()
                   if hasattr(client, "list_models_detailed") else [])
    except Exception:  # noqa: BLE001 -- discovery never blocks
        listing = []
    reported = context_window_from_listing(model, listing)
    if reported:
        cfg.context_window = reported
        return f"context window: {reported:,} (reported by the gateway, for {model})"
    known = hint or context_window_for(model)
    if known:
        used = min(int(known), ASSUMED_WINDOW_CAP)
        cfg.context_window = used
        if used < known:
            return (f"context window: {used:,} for {model} (the model takes "
                    f"up to {known:,}, but the gateway did not report this "
                    "deployment's limit, so it is capped; set context_window "
                    "under llm.<method> in main_config.yaml to use more)")
        return f"context window: {used:,} (model config, for {model})"
    return (f"context window: {cfg.context_window:,} (ASSUMED -- {model} is "
            "not in the catalogue and the gateway did not report one; set "
            "context_window under llm.<method> in main_config.yaml)")


def context_window_for(model_id: str, listing: Optional[list] = None) -> Optional[int]:
    """Best-known context window for `model_id`, or None if unrecognised
    -- never a guess dressed up as certainty. The gateway's own listing
    (when given) wins over the regex table; keep _CONTEXT_WINDOWS in sync
    as new models (K3, ...) actually ship."""
    reported = context_window_from_listing(model_id, listing) if listing else None
    if reported:
        return reported
    low = (model_id or "").lower()
    for pat, size in _CONTEXT_WINDOWS:
        if re.search(pat, low):
            return size
    return None


def choose_model(llm, *, autonomous: bool = False, interactive: Optional[bool] = None,
                 ask: Callable[[str], str] = input,
                 say: Callable[[str], None] = print) -> Optional[str]:
    """Query `llm` (an LLMClient) for what it can actually serve and return
    the model id to use -- prompting only when there's a real choice AND
    someone is there to answer.

    Returns None when there's nothing to decide (the gateway doesn't
    support /models, is unreachable, or lists nothing) -- the caller
    should then just keep whatever model was already configured; this
    must never be the thing that blocks a session from starting.
    """
    try:
        models = llm.list_models()
    except Exception:  # noqa: BLE001 -- discovery is best-effort
        models = []
    if not models:
        return None
    ranked = rank_models(models)
    best = ranked[0].id

    if len(ranked) == 1:
        say(f"model: {best} (only one available)")
        return best

    if interactive is None:
        interactive = sys.stdin.isatty()
    if autonomous or not interactive:
        say(f"model: {best} (auto-selected, best of {len(ranked)} available)")
        return best

    say(f"{len(ranked)} models available:")
    for i, m in enumerate(ranked, start=1):
        marker = "  <- recommended" if i == 1 else ""
        say(f"  {i}. {m.id}{marker}")
    answer = (ask(f"Pick a model [1-{len(ranked)}], or Enter for the "
                  "recommended one: ") or "").strip()
    if not answer:
        return best
    try:
        idx = int(answer) - 1
        if 0 <= idx < len(ranked):
            return ranked[idx].id
    except ValueError:
        pass
    say(f"'{answer}' wasn't a valid choice -- using the recommended model.")
    return best
