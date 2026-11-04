"""
ledgerhound.memory.consolidation
================================

V3-026 -- the "post-run job" half of the item: distill run outputs into
durable memory. Two independently-testable primitives, plus one
orchestrator that runs both in sequence (the roadmap's literal "post-run
job" framing):

  - ``check_card_staleness`` -- compares a DatasetCard's captured
    ``data_sha256`` against a fresh hash of the file on disk
    (``ledgerhound.runtime.events.sha256_file``, already used elsewhere for
    exactly this fingerprinting purpose -- e.g. ``ledgerhound.runtime.replay``
    detects "input data changed since this session" the same way). No
    content-hash field existed on ``DatasetCard`` before this.
  - ``run_consolidation_pass`` -- calls ``FindingsLog.consolidate()``
    (near-duplicate merge, V3-026's other half, built directly on
    ``FindingsLog``) and ``check_card_staleness``, and optionally
    ``ledgerhound.findings.promotion.promote_findings`` if a knowledge store
    is supplied, so one call covers "merge duplicate findings, staleness
    flags on cards, reject promotions without evidence links" together.

Findings dedup/merge itself lives on ``FindingsLog.consolidate()``
(``ledgerhound/findings/log.py``), not here -- it's tightly coupled to
``Finding``/``FindingsLog`` internals (the atomic-rewrite pattern
``revalidate()`` already established), so it belongs there, the same way
``agreement_check`` does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ledgerhound.runtime.events import sha256_file

if TYPE_CHECKING:
    from ledgerhound.findings.log import ConsolidationReport, FindingsLog
    from ledgerhound.findings.promotion import PromotionResult
    from ledgerhound.knowledge.dataset_card import DatasetCard
    from ledgerhound.memory.knowledge import DatasetKnowledge


def check_card_staleness(card: "DatasetCard", data_path: str | Path) -> bool:
    """Compare `card.data_sha256` against a fresh hash of `data_path`.

    Bootstrap case: if the card has no hash captured yet (`data_sha256`
    is empty -- this is the first time anything checked), this call
    CAPTURES the current hash as the baseline and returns False (not
    stale) -- mirrors V3-024's PSI-baseline bootstrap ("no baseline yet"
    is not the same as "confirmed stale"). Only a genuine mismatch against
    a previously-captured hash flags staleness.
    """
    current_hash = sha256_file(data_path)
    if not card.data_sha256:
        card.set_data_hash(current_hash)
        return False
    if current_hash != card.data_sha256:
        card.flag_stale(
            f"data_sha256 changed: card has {card.data_sha256[:12]}..., "
            f"file on disk now hashes to {current_hash[:12]}... "
            f"-- reference_distributions/digest/last_eval on this card "
            f"may no longer describe the current data"
        )
        return True
    # Hash matches -- confirm not-stale (clears a previously-flagged card
    # if the underlying data was restored/reverted to the captured hash).
    card.set_data_hash(current_hash)
    return False


@dataclass
class ConsolidationPassReport:
    findings_report: Optional[dict] = None
    card_stale: Optional[bool] = None
    card_stale_reason: str = ""
    promotion_report: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_consolidation_pass(
    findings_log: Optional["FindingsLog"] = None,
    dataset: Optional[str] = None,
    similarity_threshold: float = 0.6,
    card: Optional["DatasetCard"] = None,
    data_path: Optional[str | Path] = None,
    knowledge: Optional["DatasetKnowledge"] = None,
    min_confidence: float = 0.80,
) -> ConsolidationPassReport:
    """One post-run job covering all three named checks, each optional and
    independently skippable (a caller running only the staleness check on
    a session with no findings yet passes findings_log=None, etc.)."""
    report = ConsolidationPassReport()

    if findings_log is not None:
        cons = findings_log.consolidate(similarity_threshold=similarity_threshold,
                                        dataset=dataset)
        report.findings_report = asdict(cons)

    if card is not None and data_path is not None:
        stale = check_card_staleness(card, data_path)
        report.card_stale = stale
        report.card_stale_reason = card.stale_reason

    if findings_log is not None and knowledge is not None:
        from ledgerhound.findings.promotion import promote_findings
        promo = promote_findings(findings_log, knowledge, dataset=dataset,
                                 min_confidence=min_confidence)
        report.promotion_report = promo.to_dict()

    return report
