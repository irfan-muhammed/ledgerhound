"""
ledgerhound.findings.promotion
===========================

Bridge that promotes high-confidence, sample-frame-validated findings into
DatasetKnowledge entries automatically.

Why this bridge exists
----------------------
FindingsLog captures every analytical claim the agent makes during a
session.  DatasetKnowledge persists what's worth carrying *into the next
session*.  Without an explicit bridge, the analyst either:
  (a) re-records important findings as deposit_insight calls (duplication),
  (b) lets them sit in the JSONL forever, invisible to future sessions.

The promotion bridge applies a deterministic gate -- confidence + sample-
frame validity -- and copies eligible findings into the knowledge store.

Gate criteria (all must be true)
--------------------------------
1. ``confidence >= 0.80``
2. ``sample_frame != UNSPECIFIED``
3. ``validation.is_valid is True`` (or validation is absent -- backwards-
   compat for findings recorded before B1 wiring)
4. ``finding.has_evidence()`` is True (own query/code/chart_path, or --
   for a merged finding -- inherited ``evidence_links``); routed through
   ``ledgerhound.memory.promotion_gate.PromotionGate`` (V3-026)
5. Not already promoted in a previous run (tracked via a tag)

Knowledge entry mapping
-----------------------
- entry_type defaults to ``ANALYST_NOTE``; overridden by hints in tags:
    ``typology`` -> TYPOLOGY
    ``quality``  -> QUALITY_FLAG
    ``boundary`` -> BOUNDARY
    ``drift``    -> DRIFT_EVENT
    ``decay``    -> SIGNAL_DECAY
    ``emerging`` -> SIGNAL_EMERGING
- evidence_ref is set to the finding ID so the knowledge entry traces
  back to the finding that produced it.
- tags from the finding carry over, plus a ``promoted_from_finding`` tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from ledgerhound.findings.models import Finding, SampleFrame
from ledgerhound.memory.promotion_gate import PromotionCriteria, PromotionGate, PromotionInput

if TYPE_CHECKING:
    from ledgerhound.findings.log import FindingsLog
    from ledgerhound.memory.knowledge import DatasetKnowledge

log = logging.getLogger(__name__)

# V3-026: the evidence-link check now genuinely routes through the shared
# ledgerhound.memory.promotion_gate.PromotionGate (require_evidence=True),
# not a bespoke re-implementation -- this is what "extends
# memory/promotion_gate.py" means concretely. Every other criterion on
# that gate (corroboration/reverification/duplicate) is neutralized here
# (min_distinct_runs=0 etc.) since it doesn't apply to a single Finding --
# only the evidence check is exercised through it. Confidence,
# sample_frame validity, and the already-promoted tag stay as
# Finding-specific checks below (no generic analog on the gate).
_EVIDENCE_GATE = PromotionGate(PromotionCriteria(
    min_distinct_runs=0, min_distinct_owners=0, min_confidence=0.0,
    block_if_contradicted=False, block_if_duplicate=False, require_evidence=True,
))


def _has_evidence(finding: Finding) -> bool:
    decision = _EVIDENCE_GATE.decide(PromotionInput(
        key=finding.id, distinct_runs=1, confidence=1.0,
        has_evidence=finding.has_evidence(),
    ))
    return decision.promote


# Tag -> KnowledgeType mapping (kept here, not in knowledge.py, to avoid
# circular import).
_TAG_TO_KTYPE = {
    "typology":         "typology",
    "quality":          "quality_flag",
    "quality_flag":     "quality_flag",
    "boundary":         "boundary",
    "hard_negatives":   "boundary",
    "drift":            "drift_event",
    "drift_event":      "drift_event",
    "decay":            "signal_decay",
    "signal_decay":     "signal_decay",
    "emerging":         "signal_emerging",
    "signal_emerging":  "signal_emerging",
    "anomaly":          "anomaly",
    "pipeline":         "pipeline_issue",
}

_PROMOTED_TAG = "promoted_from_finding"


@dataclass
class PromotionResult:
    """Output of promote_findings()."""

    n_findings_scanned: int = 0
    n_eligible: int = 0
    n_promoted: int = 0
    n_already_promoted: int = 0
    n_rejected_low_confidence: int = 0
    n_rejected_unspecified_frame: int = 0
    n_rejected_invalid: int = 0
    n_rejected_no_evidence: int = 0
    promoted_entry_ids: list[str] = field(default_factory=list)
    rejected_finding_ids: list[tuple[str, str]] = field(default_factory=list)
    # (finding_id, reason)

    def to_dict(self) -> dict:
        return {
            "n_findings_scanned":          self.n_findings_scanned,
            "n_eligible":                  self.n_eligible,
            "n_promoted":                  self.n_promoted,
            "n_already_promoted":          self.n_already_promoted,
            "n_rejected_low_confidence":   self.n_rejected_low_confidence,
            "n_rejected_unspecified_frame": self.n_rejected_unspecified_frame,
            "n_rejected_invalid":          self.n_rejected_invalid,
            "n_rejected_no_evidence":      self.n_rejected_no_evidence,
            "promoted_entry_ids":          self.promoted_entry_ids,
            "rejected_finding_ids":        self.rejected_finding_ids,
        }


def _classify_entry_type(finding: Finding) -> str:
    """Pick a KnowledgeType value from the finding's tags."""
    for tag in finding.tags or []:
        if tag in _TAG_TO_KTYPE:
            return _TAG_TO_KTYPE[tag]
        # Allow tags with prefix matching: "decay:KS_drop" -> signal_decay
        prefix = tag.split(":")[0]
        if prefix in _TAG_TO_KTYPE:
            return _TAG_TO_KTYPE[prefix]
    return "analyst_note"


def _eligible(
    finding: Finding,
    min_confidence: float,
) -> Optional[str]:
    """Return None if eligible, else a string explaining why not."""
    if finding.confidence < min_confidence:
        return f"low_confidence ({finding.confidence:.2f} < {min_confidence})"
    if finding.sample_frame == SampleFrame.UNSPECIFIED:
        return "unspecified_sample_frame"
    val = finding.validation
    if val is not None and not val.get("is_valid"):
        return "validation_failed"
    if not _has_evidence(finding):
        return "no_evidence"
    if _PROMOTED_TAG in (finding.tags or []):
        return "already_promoted"
    return None


def promote_findings(
    findings_log: "FindingsLog",
    knowledge: "DatasetKnowledge",
    *,
    dataset: Optional[str] = None,
    min_confidence: float = 0.80,
    session_id: str = "promotion",
    update_log: bool = True,
    dry_run: bool = False,
) -> PromotionResult:
    """
    Scan a FindingsLog and promote eligible findings into a DatasetKnowledge
    store.

    Parameters
    ----------
    findings_log : the source of findings
    knowledge    : the target knowledge store
    dataset      : if set, only consider findings whose ``dataset`` matches
    min_confidence : confidence gate; default 0.80
    session_id   : session id stamped onto the new knowledge entries
    update_log   : if True, rewrite the findings JSONL with the promoted
                   tag added.  Set False for read-only audits.
    dry_run      : if True, return what *would* be promoted without writing.

    Returns
    -------
    PromotionResult
    """
    from ledgerhound.memory.knowledge import KnowledgeEntry

    result = PromotionResult()
    findings = findings_log.all(dataset=dataset)
    result.n_findings_scanned = len(findings)

    to_persist_back: list[Finding] = []
    promoted_ids: list[str] = []
    for f in findings:
        reason = _eligible(f, min_confidence=min_confidence)
        if reason is not None:
            result.rejected_finding_ids.append((f.id, reason))
            if reason.startswith("low_confidence"):
                result.n_rejected_low_confidence += 1
            elif reason == "unspecified_sample_frame":
                result.n_rejected_unspecified_frame += 1
            elif reason == "validation_failed":
                result.n_rejected_invalid += 1
            elif reason == "no_evidence":
                result.n_rejected_no_evidence += 1
            elif reason == "already_promoted":
                result.n_already_promoted += 1
            to_persist_back.append(f)
            continue

        result.n_eligible += 1
        if dry_run:
            to_persist_back.append(f)
            continue

        entry = KnowledgeEntry(
            session_id=session_id,
            entry_type=_classify_entry_type(f),
            feature="",                          # findings don't carry feature in this version
            claim=f.claim,
            detail=(
                (f.evidence.summary or "") +
                (f"\n[query] {f.evidence.query}" if f.evidence.query else "") +
                (f"\n[code] {f.evidence.code}" if f.evidence.code else "")
            ).strip(),
            evidence_ref=f.id,
            confidence=f.confidence,
            period="",
            tags=list(f.tags or []) + ["promoted_from_finding", f"sf:{f.sample_frame.value}"],
        )
        knowledge.add(entry)
        promoted_ids.append(entry.entry_id)
        result.n_promoted += 1
        # Tag the finding so we don't re-promote on the next run
        new_tags = list(f.tags or [])
        if _PROMOTED_TAG not in new_tags:
            new_tags.append(_PROMOTED_TAG)
            new_tags.append(f"promoted_to:{entry.entry_id}")
        f.tags = new_tags
        to_persist_back.append(f)

    result.promoted_entry_ids = promoted_ids

    if not dry_run and update_log and to_persist_back:
        # Rewrite the JSONL atomically with the updated tags
        import json, tempfile, os
        all_other = (
            [] if dataset is None
            else [g for g in findings_log.all() if g.dataset != dataset]
        )
        payload = all_other + to_persist_back
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=findings_log.path.parent,
            prefix=".findings_promote_", suffix=".jsonl",
            encoding="utf-8",
        ) as tf:
            for g in payload:
                tf.write(json.dumps(g.to_dict(), default=str, ensure_ascii=False) + "\n")
            tmp_path = tf.name
        os.replace(tmp_path, findings_log.path)
        log.info(
            "Promoted %d findings (%d already, %d rejected)",
            result.n_promoted, result.n_already_promoted,
            len(result.rejected_finding_ids) - result.n_already_promoted,
        )

    return result
