"""
sentinel.memory.promotion_gate
==============================

The deterministic gate that decides whether a private-tier memory entry earns
promotion into the shared (collective) tier.

This is the anti-junk mechanism (AUD-142) and it is **pure code, never an LLM
judgment** (AUD-143). The gate takes already-computed, recomputable signals and
applies fixed thresholds, so every promotion is auditable and replayable. An
LLM may *propose* a fact upstream, but only this gate promotes it.

Criteria (all must hold):
  * corroborated  -- confirmed across >= min_distinct_runs distinct runs
                     (and >= min_distinct_owners owners, when owner info exists)
  * not contradicted -- status must not be 'contradicted'
  * re-verified   -- if require_reverified, the fact's computation must have
                     reproduced on its last check (reverified is True)
  * non-duplicate -- not a near-duplicate of an existing shared entry
                     (duplicates are *merged*, not blocked, by the Librarian;
                     this flag guards genuine semantic dupes)
  * confident     -- confidence >= min_confidence
  * has evidence  -- (V3-026, opt-in via require_evidence) the candidate is
                     backed by something reproducible. Default OFF
                     (require_evidence=False) because Facts and Lessons --
                     the two entry types Librarian.promote() has always
                     gated -- have no "evidence" concept of their own (a
                     fact is a recomputed number, not a claim needing a
                     query/code trail); this criterion only applies when a
                     caller opts in, which sentinel.findings.promotion does
                     for Findings specifically (see that module).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromotionCriteria:
    min_distinct_runs: int = 3
    min_distinct_owners: int = 1      # Tier-1 reality is single-owner; raise later
    min_confidence: float = 0.75
    require_reverified: bool = False   # True once reverify runs routinely
    block_if_contradicted: bool = True
    block_if_duplicate: bool = True
    require_evidence: bool = False    # V3-026, opt-in -- see module docstring


@dataclass
class PromotionInput:
    """Recomputable signals about one candidate entry."""
    key: str
    distinct_runs: int
    confidence: float
    distinct_owners: int = 1
    status: str = "new"               # new | stable | drifting | contradicted
    reverified: bool | None = None    # None = not checked; True/False = checked
    is_duplicate: bool = False
    has_evidence: bool = True         # V3-026 -- default True: irrelevant unless
                                       # the caller's criteria set require_evidence


@dataclass
class PromotionDecision:
    key: str
    promote: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


class PromotionGate:
    """Applies fixed thresholds to decide promotion. Deterministic + auditable."""

    def __init__(self, criteria: PromotionCriteria | None = None):
        self.criteria = criteria or PromotionCriteria()

    def decide(self, inp: PromotionInput) -> PromotionDecision:
        c = self.criteria
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["corroborated_runs"] = inp.distinct_runs >= c.min_distinct_runs
        if not checks["corroborated_runs"]:
            reasons.append(
                f"only {inp.distinct_runs} distinct run(s) "
                f"(need >= {c.min_distinct_runs})")

        checks["corroborated_owners"] = inp.distinct_owners >= c.min_distinct_owners
        if not checks["corroborated_owners"]:
            reasons.append(
                f"only {inp.distinct_owners} owner(s) "
                f"(need >= {c.min_distinct_owners})")

        checks["confident"] = inp.confidence >= c.min_confidence
        if not checks["confident"]:
            reasons.append(
                f"confidence {inp.confidence:.2f} < {c.min_confidence:.2f}")

        checks["not_contradicted"] = not (
            c.block_if_contradicted and inp.status == "contradicted")
        if not checks["not_contradicted"]:
            reasons.append("status is 'contradicted'")

        if c.require_reverified:
            checks["reverified"] = inp.reverified is True
            if not checks["reverified"]:
                reasons.append("not re-verified against current data")
        # else: re-verification not required this pass

        checks["non_duplicate"] = not (c.block_if_duplicate and inp.is_duplicate)
        if not checks["non_duplicate"]:
            reasons.append("semantic duplicate of an existing shared entry")

        if c.require_evidence:
            checks["has_evidence"] = inp.has_evidence
            if not checks["has_evidence"]:
                reasons.append("no evidence link (query/code/chart_path, or evidence_links)")
        # else: evidence not required for this candidate type (facts/lessons)

        promote = all(checks.values())
        if promote:
            reasons.append("all criteria met")
        return PromotionDecision(key=inp.key, promote=promote,
                                 reasons=reasons, checks=checks)
