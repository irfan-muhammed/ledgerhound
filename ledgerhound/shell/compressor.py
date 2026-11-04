"""
ledgerhound.shell.compressor
==========================

ShellContextCompressor -- context compaction for ConversationalAnalyst
(V3-010).

This is deliberately NOT a wrapper around ``ledgerhound.session.compressor.
ContextCompressor``: that class is built against ``ledgerhound.session.models.
Session`` (message ``tag`` fields, checkpoints, dataset entries, a session
store) which the shell has none of -- ``ConversationalAnalyst`` just holds a
plain ``messages: list[dict]`` OpenAI-style history. Forcing the shell
through a fake ``Session`` adapter would be more fragile than reusing the
one thing that actually transfers: the **tier concept**.

Three tiers, re-derived from what the shell actually tracks (no tagging
needed on individual messages):

  SACRED (kept verbatim, never compressed)
    - the system prompt (message 0)
    - the pinned live-schema block (``analyst._schema_pin_idx``), if any
    - the last ``recent_window`` messages -- immediate working memory

  COMPRESSED (the middle section)
    - everything else, collapsed into ONE dense summary message. Rule-based
      (cheap, deterministic, no extra LLM round-trip) -- one line per
      original message, deduplicated.

  FINDINGS (always fresh, never lost)
    - the shell doesn't tag "finding" messages in-context; instead findings
      are already durably logged to ``findings.jsonl`` by ``record_finding``.
      Compaction re-reads that file and appends a compact bullet list to the
      summary every time it runs, so recorded findings survive regardless of
      which raw messages get compressed away -- one source of truth, reused.
"""

from __future__ import annotations

import json
from pathlib import Path

RECENT_WINDOW = 20               # keep the last N messages verbatim
# Message-count BACKSTOP only -- the token budget below is the real trigger.
# At 40 it fired every ~9 tool calls on ordinary small results (39
# compactions, 39 summariser LLM calls, in one 369-call turn whose window
# never passed ~5k tokens): pure latency, and each summary loses detail
# (K-02). High enough to stay out of the way, low enough that a runaway
# message list is still bounded.
COMPRESS_TRIGGER_MESSAGES = 200
# Tokens kept free on top of the reply reservation (max_tokens): the
# estimate is approximate and tool schemas ride along in every request.
HEADROOM_TOKENS = 1_024
# The findings block rides in EVERY summary; bound it (K-08). The full list
# is always in findings.jsonl.
FINDINGS_BLOCK_MAX = 60

# Token-aware trigger. Message count alone is blind: 40 messages can be 8k
# tokens or 800k depending on how big the cell outputs are, and on a
# 74M-row dataset they are big. The 2026-08-06 production run compacted 58
# times without ever knowing how full the window actually was.
DEFAULT_CONTEXT_WINDOW = 256_000   # Kimi K2 class; override per model config
COMPRESS_AT_FRACTION = 0.70        # compact once the window is this full
CHARS_PER_TOKEN = 3.6              # module-level estimate_tokens() only (legacy)
# Used by a compressor BEFORE the gateway's first reply calibrates it.
# Measured on UPI-style rows: Gemma 4 1.48, Qwen 3.6 1.54, GLM 2.11,
# Kimi 2.29, DeepSeek 2.35 chars/token -- 3.6 undercounted over 2x.
UNCALIBRATED_CHARS_PER_TOKEN = 2.0
# The tokenizer's count may differ from the gateway's by this much
# (after learning the constant template/tool-schema overhead) before a
# request counts as a disagreement; two in a row drop the tokenizer.
MISMATCH_TOLERANCE = 0.15
PER_MESSAGE_TOKENS = 4             # role/framing tokens a chat template adds
# Below this much compacted text, summarising with the LLM is not worth an
# extra round-trip: the deterministic summary already captures a handful of
# short turns without loss, and the 2026-08-06 run compacted 58 times --
# paying a model call for every one of those is real latency and cost.
LLM_SUMMARY_MIN_CHARS = 4_000


def estimate_tokens(messages) -> int:
    """Cheap, dependency-free token estimate for a message list.

    Deliberately an ESTIMATE, and deliberately conservative (3.6 chars/token
    rather than the usual 4) so we compact slightly early rather than
    overrunning the window. Counting exactly would mean shipping a
    tokenizer per model; being roughly right and erring small is worth more
    here than being exactly right.
    """
    total = 0
    for m in messages or []:
        content = m.get("content") or ""
        total += len(str(content))
        for tc in (m.get("tool_calls") or []):
            total += len(str(tc))
        total += 8   # per-message role/framing overhead
    return int(total / CHARS_PER_TOKEN)


_COMPACTION_FALLBACK = """Summarize this analysis session segment under these headings, being
specific about names, numbers and column names:
1. Goal and constraints  2. Data  3. Computed artifacts still in the sandbox
4. Findings established  5. Decisions and their reasons
6. Errors hit and how they were resolved  7. Open threads
Section 3 matters most: name every variable that already exists and what it
holds, so the analyst does not recompute work it already has.
"""


class ShellContextCompressor:
    """Call ``maybe_compress(analyst)`` before each LLM call. Mutates
    ``analyst.messages`` in place when the trigger is crossed."""

    def __init__(self, recent_window: int = RECENT_WINDOW,
                 trigger: int = COMPRESS_TRIGGER_MESSAGES,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 compress_at_fraction: float = COMPRESS_AT_FRACTION,
                 reserve_tokens: int = 0):
        self.recent_window = recent_window
        self.trigger = trigger
        self.context_window = context_window
        self.compress_at_fraction = compress_at_fraction
        # The reply the server must still fit after the prompt (the model's
        # max_tokens). vLLM-style servers reject prompt + max_tokens >
        # window, so the prompt budget has to leave this free (K-01).
        self.reserve_tokens = reserve_tokens
        self.compressions_run = 0
        # Exact counting (ledgerhound.llm.tokenizer) when a tokenizer was
        # found for the model; None means estimate + calibrate.
        self.counter = None
        # chars/token measured from the gateway's usage.prompt_tokens
        self._ratio = None
        # tokens the gateway counts beyond the message text (chat
        # template + tool schemas), learned when a tokenizer is attached
        self._overhead = None
        self._extra_chars = 0        # size of the tools= schema last sent
        self._pending = None
        self._strikes = 0

    def token_budget(self) -> int:
        """Prompt tokens at which to compact: the fraction applies to the
        room LEFT after the reply reservation, not to the whole window.

        Measured on a 40,960-token window with a 16,384-token reply: the
        fraction-of-window budget let prompt + reply reach ~50k whenever
        tool output was denser than the 3.6 chars/token estimate (numeric
        tables run nearer 2.5). Applying the fraction to the room keeps the
        30% slack where it is actually needed. With no reservation this is
        exactly fraction x window, as before."""
        room = self.context_window - int(self.reserve_tokens or 0)
        return max(1_000, int(room * self.compress_at_fraction))

    def sync_from(self, llm) -> None:
        """Track the live model config -- a /model switch changes the
        window and the reply reservation mid-session.

        Only a CHANGE in the model config is adopted. A caller that built
        the compressor with its own window/reservation (a debate agent, a
        test, an embedder) keeps it until the model config actually moves;
        overwriting it on every check silently discarded that choice."""
        cfg = getattr(llm, "config", None)
        cw = getattr(cfg, "context_window", None)
        mt = getattr(cfg, "max_tokens", None)
        seen = (cw if isinstance(cw, int) and cw > 0 else None,
                mt if isinstance(mt, int) and mt >= 0 else None)
        last = getattr(self, "_last_cfg", None)
        self._last_cfg = seen
        if last is None:
            # First sighting: fill in only what the constructor left unset.
            if seen[1] is not None and not self.reserve_tokens:
                self.reserve_tokens = seen[1]
            return
        if seen[0] is not None and seen[0] != last[0]:
            self.context_window = seen[0]
        if seen[1] is not None and seen[1] != last[1]:
            self.reserve_tokens = seen[1]

    # -- counting -------------------------------------------------------------

    def set_counter(self, counter) -> None:
        """Attach (or clear) an exact TokenCounter for the current model."""
        self.counter = counter
        self._overhead = None
        self._strikes = 0

    def chars_per_token(self) -> float:
        return self._ratio or UNCALIBRATED_CHARS_PER_TOKEN

    @staticmethod
    def _chars(messages) -> int:
        return sum(len(str(m.get("content") or ""))
                   + sum(len(str(tc)) for tc in (m.get("tool_calls") or []))
                   + 8 for m in (messages or []))

    def _count_text(self, messages) -> int:
        n = 0
        for m in messages or []:
            n += self.counter.count(str(m.get("content") or ""))
            for tc in (m.get("tool_calls") or []):
                n += self.counter.count(str(tc))
            n += PER_MESSAGE_TOKENS
        return n

    def estimate(self, messages) -> int:
        """Prompt tokens the gateway will count for `messages` (+ the tool
        schemas sent with them). Exact message text when a tokenizer is
        attached; otherwise characters at the calibrated density."""
        if self.counter is not None:
            try:
                overhead = (self._overhead if self._overhead is not None
                            else int(self._extra_chars / UNCALIBRATED_CHARS_PER_TOKEN))
                return self._count_text(messages) + overhead
            except Exception:  # noqa: BLE001 -- a counting failure must not stop a turn
                self.counter = None
        return int((self._chars(messages) + self._extra_chars)
                   / self.chars_per_token())

    def before_send(self, messages, tools=None) -> None:
        """Snapshot what is about to be sent, so after_response can compare
        it with what the gateway actually counted."""
        try:
            self._extra_chars = len(json.dumps(tools)) if tools else 0
            chars = self._chars(messages) + self._extra_chars
            counted = self._count_text(messages) if self.counter is not None else None
            self._pending = (chars, counted)
        except Exception:  # noqa: BLE001
            self._pending = None

    def after_response(self, usage) -> str:
        """Learn from the gateway's own count. Returns a message when the
        attached tokenizer turns out to disagree with the gateway (it is then
        dropped and calibration takes over), else ""."""
        pending, self._pending = self._pending, None
        try:
            prompt = int((usage or {}).get("prompt_tokens") or 0)
        except (TypeError, ValueError, AttributeError):
            prompt = 0
        if not pending or prompt <= 0:
            return ""
        chars, counted = pending
        r = chars / prompt
        r = r if self._ratio is None else 0.5 * self._ratio + 0.5 * r
        self._ratio = max(1.0, min(8.0, r))
        if self.counter is None or counted is None:
            return ""
        if self._overhead is not None:
            off = abs(prompt - (counted + self._overhead)) / prompt
        else:
            off = 0.0
        # A tokenizer that counts MORE than the gateway saw is plainly the
        # wrong one; one that misses by a wide margin twice running is too.
        wrong = counted > prompt * (1 + MISMATCH_TOLERANCE) or off > MISMATCH_TOLERANCE
        self._strikes = self._strikes + 1 if wrong else 0
        self._overhead = max(0, prompt - counted)
        if self._strikes < 2:
            return ""
        name = getattr(self.counter, "name", "the tokenizer")
        self.counter = None
        self._overhead = None
        return (f"tokenizer [count_mismatch]: the {name} tokenizer counted "
                f"{counted:,} tokens where the gateway reported {prompt:,} for "
                "the same request, twice in a row -- it is not the tokenizer "
                "this model uses. Dropped it; counting is now calibrated from "
                f"the gateway's usage ({self.chars_per_token():.2f} chars/token). "
                "Fix: correct this model's entry or alias in the tokenizer "
                "manifest.")

    def should_compress(self, messages) -> tuple[bool, str]:
        """(compress?, why). Either bar can trigger: the message-count bar
        keeps behaviour familiar on small sessions, the token bar is what
        actually protects the window on a crore-scale one."""
        tokens = self.estimate(messages)
        if tokens >= self.token_budget():
            return True, (f"~{tokens:,} est. tokens >= "
                          f"{self.token_budget():,} budget "
                          f"({self.compress_at_fraction:.0%} of "
                          f"{self.context_window:,})")
        if len(messages) >= self.trigger:
            return True, f"{len(messages)} messages >= {self.trigger}"
        return False, ""

    def maybe_compress(self, analyst) -> bool:
        self.sync_from(getattr(analyst, "llm", None))
        do_it, why = self.should_compress(analyst.messages)
        if not do_it:
            return False
        self._last_reason = why
        return self.compress(analyst)

    def compress(self, analyst) -> bool:
        msgs = analyst.messages
        sacred_idx = {0}  # system prompt
        pin_idx = getattr(analyst, "_schema_pin_idx", None)
        if pin_idx is not None:
            sacred_idx.add(pin_idx)
        recent_start = max(len(msgs) - self.recent_window, 0)
        # A `tool` message must immediately follow the assistant message that
        # carries its `tool_calls` (OpenAI/DeepSeek reject an orphaned tool
        # message with HTTP 400). If the recent-window boundary lands in the
        # middle of a tool-call group, snap it back over the leading tool
        # messages so the parent assistant is kept in the recent tier with
        # them, never collapsed into the summary. (V3-010 follow-up: the
        # orphaned-tool-message crash flagged in the v3 roadmap.)
        while recent_start > 0 and msgs[recent_start].get("role") == "tool":
            recent_start -= 1
        recent_idx = set(range(recent_start, len(msgs)))

        sacred = [m for i, m in enumerate(msgs) if i in sacred_idx]
        compressible = [m for i, m in enumerate(msgs)
                        if i not in sacred_idx and i not in recent_idx]
        recent = [m for i, m in enumerate(msgs)
                 if i in recent_idx and i not in sacred_idx]

        if not compressible:
            return False   # nothing worth compressing yet

        # ARCHIVE FIRST, then compact. Compaction removes messages from the
        # live window -- it must never remove them from existence. Every
        # compacted message is appended verbatim to compacted_history.jsonl
        # and stays searchable via the recall_history tool, so "lost detail"
        # becomes "detail one lookup away".
        self._archive(analyst, compressible)
        summary_text = self._summarize_with_llm(analyst, compressible)
        findings_block = self._findings_block(analyst)
        content = (f"=== COMPACTED HISTORY ({len(compressible)} earlier "
                  f"messages, batch {self.compressions_run + 1}) ===\n"
                  f"{summary_text}\n\n"
                  "NOTE: nothing here was deleted. The full verbatim text of "
                  "these messages is archived in compacted_history.jsonl and "
                  "is searchable -- call recall_history(query=...) to pull "
                  "back exact detail (a number, a code cell, an error) that "
                  "this summary condensed. Do NOT recompute something just "
                  "because the detail left the window; look it up first.")
        if findings_block:
            content += f"\n\n{findings_block}"
        summary_msg = {"role": "system", "content": content}

        new_messages = sacred + [summary_msg] + recent
        msgs[:] = new_messages
        analyst._schema_pin_idx = self._reindex_schema_pin(new_messages)
        self.compressions_run += 1
        return True

    @staticmethod
    def _reindex_schema_pin(messages: list[dict]) -> int | None:
        for i, m in enumerate(messages):
            if m.get("role") == "system" and "ACTIVE DATA" in m.get("content", ""):
                return i
        return None

    @staticmethod
    def archive_path(analyst) -> Path | None:
        run_dir = getattr(analyst, "run_dir", None)
        return Path(run_dir) / "compacted_history.jsonl" if run_dir else None

    def _archive(self, analyst, messages: list[dict]) -> None:
        """Append the messages about to leave the live window, verbatim, to
        a durable JSONL. Best-effort but loud: if this fails we log it,
        because it is the guarantee that compaction is not destructive."""
        path = self.archive_path(analyst)
        if path is None:
            return
        try:
            batch = self.compressions_run + 1
            with open(path, "a", encoding="utf-8") as f:
                for i, m in enumerate(messages):
                    rec = {"batch": batch, "seq": i,
                           "role": m.get("role"),
                           "content": m.get("content") or ""}
                    if m.get("tool_calls"):
                        rec["tool_calls"] = str(m["tool_calls"])[:8000]
                    f.write(json.dumps(rec, default=str) + "\n")
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "compaction archive failed -- compacted detail will not be "
                "recallable for this batch", exc_info=True)

    def _summarize_with_llm(self, analyst, messages: list[dict]) -> str:
        """Real structured summarization (prompts/shell/compaction.md).

        The previous implementation truncated every message to its first 200
        characters and concatenated them -- that is not summarization, it is
        mid-sentence chopping, and it is why a 2026-08-06 run kept losing
        its own goal and recomputing aggregates it already had.

        Falls back to the old truncation only if the LLM call fails: a
        degraded summary still beats no compaction at all (which would hard
        fail the next request against the window).
        """
        llm = getattr(analyst, "llm", None)
        if llm is None:
            return self._summarize(messages)
        # Only pay for a model call when there is enough text that losing
        # detail would actually hurt.
        bulk = sum(len(str(m.get("content") or "")) for m in messages)
        if bulk < LLM_SUMMARY_MIN_CHARS:
            return self._summarize(messages)
        try:
            from ledgerhound.shell.prompts.loader import load_prompt
            instructions = load_prompt("compaction", _COMPACTION_FALLBACK)
            transcript = self._render_for_summary(
                messages, budget_chars=self._summary_budget_chars())
            resp = llm.generate(
                [{"role": "system", "content": instructions},
                 {"role": "user",
                  "content": "Compact this session segment:\n\n" + transcript}],
            )
            text = (getattr(resp, "content", "") or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "LLM compaction summary failed; falling back to truncation",
                exc_info=True)
        return self._summarize(messages)

    def _summary_budget_chars(self) -> int:
        """How much transcript the summariser request may carry: half of
        what is left after the reply reservation, capped at the old 120k
        chars. The fixed 120k (~33k tokens) was sized for a 256k window and
        could not fit a 40k one, so every summary fell back to truncation
        there (K-03)."""
        room = self.context_window - int(self.reserve_tokens or 0) - 2 * HEADROOM_TOKENS
        return max(8_000, min(120_000, int(room * 0.5 * self.chars_per_token())))

    @staticmethod
    def _render_for_summary(messages: list[dict], budget_chars: int = 120_000) -> str:
        """Render messages for the summarizer, newest-biased: if the segment
        is enormous, keep the most recent detail in full rather than
        truncating everything equally."""
        parts, total = [], 0
        for m in reversed(messages):
            role = m.get("role", "?")
            content = str(m.get("content") or "")
            if m.get("tool_calls"):
                content += f"\n[tool_calls] {str(m['tool_calls'])[:2000]}"
            chunk = f"[{role}]\n{content}\n"
            if total + len(chunk) > budget_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n".join(reversed(parts))

    @staticmethod
    def _summarize(messages: list[dict]) -> str:
        lines: list[str] = []
        seen: set[tuple] = set()
        for m in messages:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:200].replace("\n", " ").strip()
            if not content:
                continue
            key = (role, content[:80])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"[{role}] {content}")
        return "\n".join(lines) if lines else "(no textual content to summarize)"

    @staticmethod
    def _findings_block(analyst) -> str:
        path = getattr(analyst, "_findings_path", None)
        if not path or not Path(path).exists():
            return ""
        recs = []
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            return ""
        if not recs:
            return ""
        lines = ["Findings recorded this session (from findings.jsonl -- "
                 "durable regardless of compaction):"]
        if len(recs) > FINDINGS_BLOCK_MAX:
            lines.append(f"  ({len(recs) - FINDINGS_BLOCK_MAX} earlier "
                         "finding(s) not shown here -- all are in "
                         "findings.jsonl; recall_history finds them)")
        for rec in recs[-FINDINGS_BLOCK_MAX:]:
            lines.append(f"  - [{rec.get('sample_frame', '?')}] "
                         f"{rec.get('claim', '')[:200]}")
        return "\n".join(lines)
