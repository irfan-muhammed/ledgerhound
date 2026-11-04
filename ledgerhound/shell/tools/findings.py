"""
ledgerhound.shell.tools.findings
================================

Recording findings, decisions, chapters and recall.
"""

from __future__ import annotations

import json
from pathlib import Path





class FindingsToolsMixin:
    """Recording findings, decisions, chapters and recall."""

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
        # V5-002: every attempt joins the family, including the ones refused
        # below. Counting only successes would make a long fruitless search
        # look like a short decisive one.
        self._effort.note_attempt(getattr(self, "_current_chapter", ""))

        refusal, detail = self._restatement_guard.check(
            claim, delta, domain_profile=self._domain_profile,
            book_claims=self._book_history_claims(),
            team_claims=self._team_kb_claims())
        if refusal:
            self._events.append("finding_blocked_as_restatement",
                                claim=claim[:500], **detail)
            self._effort.note_refused(getattr(self, "_current_chapter", ""))
            return refusal

        self._restatement_guard.note_recorded()
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
        # (Checked before this finding is appended to findings.jsonl.)
        from ledgerhound.shell.guardrails import PredicateConsistencyCheck
        predicate_warning = PredicateConsistencyCheck.check(
            claim, predicate, self._findings_path)
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
        parts = [p for p in (base, assessment.note, predicate_warning) if p]
        return "\n".join(parts)

    def _tool_record_decision(self, decision: str, rationale: str = "") -> str:
        if self._card is None:
            return ("noted, but not persisted -- decisions are saved to the "
                    "dataset card, which requires a project (none is active "
                    "this session).")
        try:
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
        try:
            self._save_card()
        except Exception:
            pass
        if not allowed:
            from ledgerhound.shell.guardrails import ChapterDedupeGate
            return ChapterDedupeGate.refusal(chapter, info)
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
