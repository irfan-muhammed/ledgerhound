"""
ledgerhound.session.compressor
============================

ContextCompressor -- keeps the LLM's effective context window healthy.

Problem
-------
A long ledgerhound auto run accumulates hundreds of messages: wizard Q&A,
step outputs, error traces, fix attempts. At ~200K tokens the LLM starts
dropping early context silently, and at the hard limit it throws an error.

Strategy
--------
Three tiers are kept at all times:

  TIER 1 -- Sacred (never compressed)
    - All messages tagged "finding"   -- data facts, can't be re-derived
    - All messages tagged "decision"  -- user choices, must be reproducible
    - The last RECENT_WINDOW messages -- immediate working memory

  TIER 2 -- Summarised (middle section)
    - Step outputs, error/fix cycles, intermediate prints
    - Replaced with a dense paragraph per step produced by the LLM

  TIER 3 -- Archive (oldest, beyond ARCHIVE_AFTER messages)
    - Full text written to context/<ts>_archive.jsonl
    - Replaced in active context with a single "archived N messages" marker

Trigger
-------
  - Explicit: call compress() at any time
  - Automatic: AutoRunner calls maybe_compress() before each LLM call
               Wizard calls maybe_compress() after each step

The compressed summary is written to context/compressed_context.json
and its path is stored in session.compressed_context_path so any
future session load can reconstruct context without re-reading thousands
of lines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ledgerhound.session.models import Session

log = logging.getLogger(__name__)

# Tune these to your LLM's context window
RECENT_WINDOW    = 30     # always keep last N messages verbatim
COMPRESS_AFTER   = 80     # compress when total messages exceeds this
ARCHIVE_AFTER    = 300    # archive (off-context) messages older than this
TOKENS_PER_MSG   = 400    # rough estimate for trigger heuristic


class ContextCompressor:

    def __init__(self, session: "Session", llm=None):
        self._session = session
        self._llm = llm   # optional -- if None, rule-based compression only

    # -- Public API -----------------------------------------------------

    def maybe_compress(self) -> bool:
        """Compress if over the threshold. Returns True if compression ran."""
        n = self._session.message_count()
        if n - self._session.context_compressed_at < COMPRESS_AFTER:
            return False
        self.compress()
        return True

    def compress(self) -> dict:
        """Run a full compression pass. Returns summary metadata."""
        messages = self._session.load_messages()
        if not messages:
            return {}

        sacred, compressible, recent = self._split(messages)

        # Archive oldest messages beyond ARCHIVE_AFTER
        to_archive = []
        if len(messages) > ARCHIVE_AFTER:
            cutoff = len(messages) - ARCHIVE_AFTER
            to_archive = messages[:cutoff]
            compressible = [m for m in compressible if m not in to_archive]
            sacred       = [m for m in sacred       if m not in to_archive]
            self._archive(to_archive)

        # Summarise compressible tier
        step_summaries = self._summarise_steps(compressible)

        # Build compressed context document
        compressed = {
            "compressed_at": datetime.now(timezone.utc).isoformat(),
            "original_message_count": len(messages),
            "archived_count": len(to_archive),
            "session_id": self._session.id,
            "session_goal": self._session.goal_label(),
            "dataset_context": self._session.description,
            "llm_enrichment": self._session.llm_enrichment,
            "datasets": [
                {"name": d.name, "rows": d.rows, "cols": d.cols,
                 "description": d.description}
                for d in self._session.datasets
            ],
            # Sacred messages preserved verbatim
            "findings_and_decisions": [
                {"role": m["role"], "content": m["content"], "tag": m["tag"]}
                for m in sacred
            ],
            # Step summaries
            "step_summaries": step_summaries,
            # Checkpoint progress
            "completed_steps": [
                {"step": c.step_index, "title": c.step_title,
                 "summary": c.output_summary}
                for c in self._session.checkpoints
            ],
            # Last RECENT_WINDOW messages kept verbatim for immediate context
            "recent_messages": [
                {"role": m["role"], "content": m["content"], "tag": m["tag"]}
                for m in recent
            ],
        }

        # Write to disk
        out = self._session.context_dir() / "compressed_context.json"
        out.write_text(json.dumps(compressed, indent=2), encoding="utf-8")

        self._session.compressed_context_path = str(out)
        self._session.context_compressed_at = len(messages)
        self._session.save()

        log.info(
            "Context compressed: %d -> %d sacred + %d summaries + %d recent",
            len(messages), len(sacred), len(step_summaries), len(recent),
        )
        return compressed

    def memory_block(self, max_findings: int = 20, max_steps: int = 12) -> str:
        """Compact cross-step memory for injection into generation prompts.

        Unlike build_context_messages (used for full resume), this is a small,
        always-available block that works whether or not compression has run.
        It surfaces the two things the agent must never forget:

          - sacred messages (tag "finding" / "decision") -- verbatim
          - completed step summaries -- from checkpoints

        Returns "" when there is nothing worth injecting yet.
        """
        lines: list[str] = []

        msgs = self._session.load_messages()
        sacred = [m for m in msgs if m.get("tag") in ("finding", "decision")]
        if sacred:
            lines.append("Key findings & decisions so far:")
            for m in sacred[-max_findings:]:
                lines.append(f"  [{m.get('tag')}] {m.get('content','')[:220]}")

        cps = sorted(self._session.checkpoints, key=lambda c: c.step_index)
        if cps:
            lines.append("Completed steps (most recent last):")
            for c in cps[-max_steps:]:
                lines.append(
                    f"  [{c.step_index}] {c.step_title}: {c.output_summary[:150]}"
                )

        if not lines:
            return ""
        return "=== SESSION MEMORY ===\n" + "\n".join(lines) + "\n=== END MEMORY ==="

    def build_context_messages(self) -> list[dict]:
        """Return the reconstructed message list to send to the LLM on resume.

        Format:
          [system: compressed summary]  +  [recent messages verbatim]
        """
        cp_path = self._session.compressed_context_path
        if not cp_path or not Path(cp_path).exists():
            # No compression yet -- return raw messages
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self._session.load_messages()
            ]

        compressed = json.loads(Path(cp_path).read_text(encoding="utf-8"))
        system_content = self._render_compressed_system(compressed)

        result: list[dict] = [{"role": "system", "content": system_content}]
        for m in compressed.get("recent_messages", []):
            result.append({"role": m["role"], "content": m["content"]})
        return result

    # -- Internals ------------------------------------------------------

    def _split(self, messages: list[dict]):
        """Split messages into sacred / compressible / recent tiers."""
        sacred_tags = {"finding", "decision"}
        recent = messages[-RECENT_WINDOW:]
        recent_set = set(id(m) for m in recent)

        sacred      = [m for m in messages if m.get("tag") in sacred_tags
                       and id(m) not in recent_set]
        compressible = [m for m in messages
                        if m.get("tag") not in sacred_tags
                        and id(m) not in recent_set
                        and id(m) not in {id(s) for s in sacred}]
        return sacred, compressible, recent

    def _summarise_steps(self, messages: list[dict]) -> list[str]:
        """Group messages by step tag and produce one summary per step."""
        if not messages:
            return []

        # Group by step (tag format: "step_N")
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        untagged = []
        for m in messages:
            tag = m.get("tag", "")
            if tag.startswith("step_"):
                groups[tag].append(m)
            else:
                untagged.append(m)

        summaries = []

        # LLM-powered summaries if available
        if self._llm and groups:
            for tag in sorted(groups):
                block = "\n".join(
                    f"[{m['role']}] {m['content'][:500]}"
                    for m in groups[tag]
                )
                prompt = (
                    f"Summarise the following analysis step in 3-5 sentences. "
                    f"Preserve ALL numerical findings, column names, error causes "
                    f"and fix decisions. Drop conversational filler.\n\n{block}"
                )
                try:
                    resp = self._llm.generate([{"role": "user", "content": prompt}])
                    summaries.append(f"[{tag}] {resp.content.strip()}")
                except Exception as e:
                    summaries.append(self._rule_summary(tag, groups[tag]))
        else:
            # Rule-based fallback: keep first + last message of each group
            for tag in sorted(groups):
                summaries.append(self._rule_summary(tag, groups[tag]))

        # Untagged: keep first 200 chars each, deduplicated
        seen = set()
        for m in untagged:
            snip = m["content"][:200]
            if snip not in seen:
                summaries.append(f"[{m['role']}] {snip}")
                seen.add(snip)

        return summaries

    def _rule_summary(self, tag: str, msgs: list[dict]) -> str:
        parts = [f"[{tag}] {len(msgs)} messages."]
        if msgs:
            parts.append(f"First: {msgs[0]['content'][:150]}")
        if len(msgs) > 1:
            parts.append(f"Last: {msgs[-1]['content'][:150]}")
        return " ".join(parts)

    def _archive(self, messages: list[dict]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = self._session.context_dir() / f"{ts}_archive.jsonl"
        with archive.open("w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m) + "\n")
        log.info("Archived %d messages to %s", len(messages), archive)

    def _render_compressed_system(self, c: dict) -> str:
        lines = [
            "=== LEDGERHOUND SESSION CONTEXT (compressed) ===",
            f"Session: {self._session.name}",
            f"Goal: {c.get('session_goal','')}",
            f"Dataset context: {c.get('dataset_context','')}",
            "",
        ]
        if c.get("llm_enrichment"):
            lines += ["Dataset enrichment:", c["llm_enrichment"], ""]

        datasets = c.get("datasets", [])
        if datasets:
            lines.append("Datasets:")
            for d in datasets:
                lines.append(
                    f"  - {d['name']}  {d.get('rows') or '?'} rows x "
                    f"{d.get('cols') or '?'} cols  {d.get('description','')}"
                )
            lines.append("")

        completed = c.get("completed_steps", [])
        if completed:
            lines.append("Completed steps:")
            for s in completed:
                lines.append(f"  [{s['step']}] {s['title']}: {s['summary'][:120]}")
            lines.append("")

        findings = c.get("findings_and_decisions", [])
        if findings:
            lines.append("Key findings and decisions (preserved verbatim):")
            for f_ in findings:
                lines.append(f"  [{f_['tag']}] {f_['content'][:300]}")
            lines.append("")

        summaries = c.get("step_summaries", [])
        if summaries:
            lines.append("Step history (summarised):")
            for s in summaries:
                lines.append(f"  {s[:200]}")
            lines.append("")

        lines.append("=== END COMPRESSED CONTEXT -- recent messages follow ===")
        return "\n".join(lines)
