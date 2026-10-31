"""
sentinel.memory.librarian
==========================

The Librarian -- the single read/write gateway to long-term memory.

This is the keeper of the moat. Every agent reads from and writes to the
library *through* the Librarian, so the library stays clean, governed, and
trustworthy as it grows. It consolidates the three existing stores behind one
``memory_op/v1`` surface:

  * FactMemory          -- numeric, recomputable facts (the spine)
  * LessonMemory        -- procedural, advisory lessons
  * LongitudinalMemory  -- associative vector recall (optional, L2)

Scope (anti-junk design, AUD-142): writes land in the caller's PRIVATE tier;
nothing reaches the SHARED tier except via the deterministic promotion gate
(AUD-143). Recall reads SHARED (trust-boosted) + the caller's own PRIVATE.

Layout under ``root``::

    root/{owner}/facts.json   lessons.jsonl   vectors/
    root/_shared/facts.json   lessons.jsonl   vectors/

Naming note: this is distinct from ``sentinel.session.file_index.FileIndex``
(a per-run file index, formerly ``FileLibrarian``). This ``Librarian`` is the
memory curator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sentinel.identity import current_user
from sentinel.memory.fact_memory import FactMemory
from sentinel.memory.lesson_memory import LessonMemory
from sentinel.memory.promotion_gate import (
    PromotionGate,
    PromotionInput,
)

log = logging.getLogger(__name__)

SHARED = "_shared"
_TRUST_BOOST = 1.25  # shared (corroborated) facts rank above private ones


# -- recall result ------------------------------------------------------------


@dataclass
class RecallBundle:
    """What recall returns: tiered context, assembled under a token budget."""
    instinct_block: str = ""          # numeric baselines (facts) -- highest value
    lessons_block: str = ""           # procedural hints
    associative: list = field(default_factory=list)  # L2 vector hits (optional)

    def to_prompt_block(self, token_budget: int = 1500) -> str:
        """Assemble a single prompt block, trimming low-priority sections first.

        Priority: instinct (facts) > lessons > associative. Token estimate is
        the conventional ~4 chars/token; we never exceed the budget.
        """
        budget_chars = max(0, token_budget) * 4
        parts: list[str] = []
        used = 0
        for section in (self.instinct_block, self.lessons_block):
            if not section:
                continue
            if used + len(section) <= budget_chars:
                parts.append(section)
                used += len(section) + 2
        # associative hits, one line each, until budget runs out
        if self.associative and used < budget_chars:
            lines = ["=== RELATED PRIOR FINDINGS (recall) ==="]
            for hit in self.associative:
                text = getattr(getattr(hit, "record", None), "text", None) or str(hit)
                line = f"  - {text}"
                if used + len(line) > budget_chars:
                    break
                lines.append(line)
                used += len(line) + 1
            if len(lines) > 1:
                parts.append("\n".join(lines) + "\n=== END RECALL ===")
        return "\n\n".join(parts)


@dataclass
class IngestReport:
    facts: int = 0
    lessons: int = 0
    findings: int = 0
    anomalies: list[str] = field(default_factory=list)  # facts that drifted


@dataclass
class PromotionReport:
    promoted_facts: list[str] = field(default_factory=list)
    promoted_lessons: list[str] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # key -> reasons


# -- the gateway ------------------------------------------------------------


class Librarian:
    """Governed gateway over private + shared memory tiers."""

    def __init__(
        self,
        root: str | Path,
        *,
        owner: str | None = None,
        gate: PromotionGate | None = None,
        enable_vectors: bool = False,
        drift_sigma: float = 3.0,
        rel_tol: float = 0.10,
    ):
        self.root = Path(root)
        self.owner = owner or current_user()
        self.gate = gate or PromotionGate()
        self.enable_vectors = enable_vectors
        self._fm_kw = {"drift_sigma": drift_sigma, "rel_tol": rel_tol}

        priv = self.root / self.owner
        shared = self.root / SHARED
        self.private_facts = FactMemory(priv / "facts.json", **self._fm_kw)
        self.shared_facts = FactMemory(shared / "facts.json", **self._fm_kw)
        self.private_lessons = LessonMemory(priv / "lessons.jsonl")
        self.shared_lessons = LessonMemory(shared / "lessons.jsonl")

        self._priv_vectors = None  # lazy LongitudinalMemory
        self._priv_dir = priv
        self._shared_dir = shared

    # -- ingest (write path -> always private tier) ------------------------

    def ingest_run(
        self,
        dataset: str,
        run_id: str,
        *,
        metrics: dict | None = None,
        computations: dict | None = None,
        lessons: list[str] | None = None,
        findings: list[str] | None = None,
        lesson_tags: list[str] | None = None,
    ) -> IngestReport:
        """Consolidate a run's outputs into the PRIVATE tier. Never promotes."""
        report = IngestReport()
        if metrics:
            cons = self.private_facts.consolidate_run(
                dataset, run_id, metrics, computations or {})
            report.facts = len(cons)
            report.anomalies = [c.key for c in cons if c.is_anomaly]
        for text in (lessons or []):
            self.private_lessons.add(text, dataset, run_id, tags=lesson_tags)
            report.lessons += 1
        if findings and self.enable_vectors:
            vm = self._vectors()
            for text in findings:
                vm.remember(source="finding", text=text, dataset=dataset)
                report.findings += 1
        self.save()
        return report

    # -- recall (read path -> shared trust-boosted + own private) -----------

    def recall(
        self,
        dataset: str,
        query: str = "",
        *,
        token_budget: int = 1500,
        max_facts: int = 20,
        max_lessons: int = 8,
        k_vectors: int = 6,
    ) -> RecallBundle:
        """Build a budgeted recall bundle from shared + private tiers."""
        bundle = RecallBundle()

        # L1 instinct: shared (corroborated) first, then private.
        blocks = []
        shared_block = self.shared_facts.baseline_block(dataset, max_facts=max_facts)
        if shared_block:
            blocks.append(shared_block.replace(
                "HISTORICAL BASELINES (data memory)",
                "SHARED BASELINES (corroborated across runs)"))
        priv_block = self.private_facts.baseline_block(dataset, max_facts=max_facts)
        if priv_block:
            blocks.append(priv_block)
        bundle.instinct_block = "\n".join(blocks)

        # L1 lessons: shared + private (deduped by text via the merge key).
        lesson_lines: list[str] = []
        for store, tag in ((self.shared_lessons, "shared"),
                           (self.private_lessons, "")):
            block = store.block(dataset, query=query, max_items=max_lessons)
            if block:
                lesson_lines.append(block)
        bundle.lessons_block = "\n".join(lesson_lines)

        # L2 associative (optional): vector recall over private findings.
        if self.enable_vectors and query:
            try:
                vm = self._vectors()
                bundle.associative = vm.recall(query, namespace="text",
                                               k=k_vectors, dataset=dataset)
            except Exception as e:  # never fail recall on vector issues
                log.warning("vector recall skipped: %s", e)
        _ = token_budget  # honored by RecallBundle.to_prompt_block at render
        return bundle

    # -- plan-node-scoped briefs (V3-026 router, additive to recall) --------

    def brief_for_node(self, node, dataset: str, *, token_budget: int = 400,
                       max_lessons: int = 5, k_vectors: int = 5):
        """"You're at <node>, on dataset X; relevant: ..." -- a cited,
        token-budget-capped brief, instead of the caller phrasing its own
        recall() query. See sentinel.memory.router for the full contract;
        kept as a thin delegating method here since "Librarian answers
        plan-node-scoped briefs" is the roadmap's own phrasing."""
        from sentinel.memory.router import render_brief
        return render_brief(self, node, dataset, token_budget=token_budget,
                            max_lessons=max_lessons, k_vectors=k_vectors)

    # -- promotion (private -> shared via the deterministic gate) -----------

    def promote(self, dataset: str | None = None) -> PromotionReport:
        """Promote eligible private facts/lessons into the shared tier.

        Deterministic: the gate decides; this method only moves what passes.
        """
        report = PromotionReport()

        facts = (self.private_facts.facts_for(dataset) if dataset
                 else list(self.private_facts._facts.values()))
        for rec in facts:
            is_dup = self.shared_facts.get(rec.dataset, rec.name) is not None
            inp = PromotionInput(
                key=rec.key,
                distinct_runs=len({o.run_id for o in rec.observations}),
                confidence=rec.confidence,
                status=rec.status,
                # duplicates are merged (idempotent), so don't let the dup flag
                # block re-promotion of an already-shared fact:
                is_duplicate=False,
                reverified=(rec.last_verified != "" or None),
            )
            decision = self.gate.decide(inp)
            if decision.promote:
                self.shared_facts.merge_record(rec)
                report.promoted_facts.append(rec.key)
            else:
                report.rejected[rec.key] = decision.reasons
            _ = is_dup

        lessons = (self.private_lessons.lessons_for(dataset) if dataset
                   else list(self.private_lessons._by_key.values()))
        for lesson in lessons:
            inp = PromotionInput(
                key=lesson.id,
                distinct_runs=len(set(lesson.run_ids)),
                confidence=lesson.confidence,
                status="stable",
            )
            decision = self.gate.decide(inp)
            if decision.promote:
                self.shared_lessons.merge_lesson(lesson)
                report.promoted_lessons.append(lesson.id)
            else:
                report.rejected[lesson.id] = decision.reasons

        self.save()
        return report

    # -- re-verification + conflicts (delegate to FactMemory) ---------------

    def reverify(self, dataset: str, df, *, tier: str = "private") -> list[dict]:
        """Re-run stored computations vs current data (memory cannot rot)."""
        store = self.shared_facts if tier == "shared" else self.private_facts
        out = store.reverify(dataset, df)
        self.save()
        return out

    def conflicts(self, dataset: str) -> dict[str, list[str]]:
        """Facts currently drifting or contradicted, per tier."""
        def names(store):
            recs = store.facts_for(dataset)
            return [r.name for r in recs if r.status in ("drifting", "contradicted")]
        return {"private": names(self.private_facts),
                "shared": names(self.shared_facts)}

    # -- persistence + introspection ----------------------------------------

    def save(self) -> None:
        self.private_facts.save()
        self.shared_facts.save()
        self.private_lessons.save()
        self.shared_lessons.save()
        # Persist the associative (vector) store too, so it survives a restart
        # and is captured by MinIO durability (MemorySync mirrors vectors/).
        if self._priv_vectors is not None:
            try:
                self._priv_vectors.save()
            except Exception as e:  # never fail a run on vector persistence
                log.warning("vector store save skipped: %s", e)

    def stats(self) -> dict:
        return {
            "owner": self.owner,
            "private": {**self.private_facts.stats(),
                        "lessons": self.private_lessons.stats()["lessons"]},
            "shared": {**self.shared_facts.stats(),
                       "lessons": self.shared_lessons.stats()["lessons"]},
        }

    def vectors(self):
        """The PRIVATE-tier associative (vector) store, built lazily.

        Public accessor so callers (e.g. the runner) route findings through the
        owner-scoped private tier instead of a separate vector root.
        """
        return self._vectors()

    # -- durability (MinIO mirror of the memory tiers) -----------------------

    def push_to_minio(self, store, **kw) -> list[str]:
        """Mirror this owner's private tier + the shared tier to MinIO.

        ``store`` is a configured MinIOStore. Returns the object keys written.
        Flushes local state first so the mirror reflects the latest writes.
        """
        self.save()
        from sentinel.memory.sync import MemorySync
        sync = MemorySync(store, self.root, **kw)
        return sync.push(tiers=[self.owner, SHARED])

    def pull_from_minio(self, store, **kw) -> list[str]:
        """Restore this owner's private tier + the shared tier from MinIO.

        Overwrites local tier files with the object-storage copy, then reloads
        the in-memory stores so subsequent recall/promote sees the restored
        state. Returns the local paths written.
        """
        from sentinel.memory.sync import MemorySync
        sync = MemorySync(store, self.root, **kw)
        out = sync.pull(tiers=[self.owner, SHARED])
        self._reload_tiers()
        return out

    def _reload_tiers(self) -> None:
        """Rebuild the fact/lesson stores from disk after a pull."""
        priv = self.root / self.owner
        shared = self.root / SHARED
        self.private_facts = FactMemory(priv / "facts.json", **self._fm_kw)
        self.shared_facts = FactMemory(shared / "facts.json", **self._fm_kw)
        self.private_lessons = LessonMemory(priv / "lessons.jsonl")
        self.shared_lessons = LessonMemory(shared / "lessons.jsonl")
        self._priv_vectors = None  # lazily rebuilt from restored vectors/

    # -- internals ----------------------------------------------------------

    def _vectors(self):
        if self._priv_vectors is None:
            from sentinel.memory.longitudinal import LongitudinalMemory
            self._priv_vectors = LongitudinalMemory(self._priv_dir / "vectors")
        return self._priv_vectors
