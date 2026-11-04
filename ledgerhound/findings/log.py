"""
ledgerhound.findings.log
=====================

A simple JSONL-backed append-only log of findings.

Why JSONL?
----------
* Human-readable and grep-able.
* Append-only -- no rewrite races, safe for concurrent agents.
* Each line is a complete, parseable Finding -- corruption can't cascade.
* Plays nicely with ``cat findings.jsonl | jq`` for ad-hoc review.

Why not a database?
-------------------
Findings are the **audit substrate**. Treating them as immutable text
makes them auditable forever, even after the tool is uninstalled. A
database adds dependency and lock complexity without buying anything
Phase 1 needs. We can index them in DuckDB on read if we want querying.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

from .models import Evidence, Finding, SampleFrame as _FindingsSampleFrame

if TYPE_CHECKING:
    from ledgerhound.discipline.sample_frame import SampleFrameValidator
    from ledgerhound.memory.longitudinal import LongitudinalMemory

log = logging.getLogger(__name__)


# -- near-duplicate detection + merge (V3-026) -------------------------------


def _norm_claim(text: str) -> set[str]:
    """Same normalization style as ledgerhound.memory.lesson_memory._norm --
    lowercase, strip to alnum+space, tokenize."""
    return set(re.sub(r"[^a-z0-9 ]", "", text.lower()).split())


def _claim_similarity(a: str, b: str) -> float:
    """Token-Jaccard overlap of two normalized claims. 0.0 if either is
    empty (rather than dividing by zero / matching everything)."""
    ta, tb = _norm_claim(a), _norm_claim(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _group_near_duplicates(findings: list[Finding], threshold: float) -> list[list[Finding]]:
    """Transitive grouping: union-find over pairwise similarity >= threshold
    AND same dataset. Preserves input order within each group."""
    n = len(findings)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if findings[i].dataset != findings[j].dataset:
                continue
            if _claim_similarity(findings[i].claim, findings[j].claim) >= threshold:
                union(i, j)

    groups_by_root: dict[int, list[Finding]] = {}
    for i, f in enumerate(findings):
        groups_by_root.setdefault(find(i), []).append(f)
    return list(groups_by_root.values())


def _merge_finding_group(group: list[Finding]) -> Finding:
    """Merge 2+ near-duplicate findings into one. The representative claim
    is the highest-confidence source's; evidence_links carries one entry
    per source that had real evidence, so the merged finding can still
    pass has_evidence() even if no single source had complete evidence."""
    representative = max(group, key=lambda f: f.confidence)
    evidence_links: list[dict] = []
    for f in group:
        if not f.evidence.is_empty():
            evidence_links.append({
                "finding_id": f.id,
                "summary": f.evidence.summary,
                "query": f.evidence.query,
                "code": f.evidence.code,
                "chart_path": f.evidence.chart_path,
            })
        evidence_links.extend(f.evidence_links)   # carry forward prior merges

    tags: list[str] = []
    for f in group:
        for t in f.tags:
            if t not in tags:
                tags.append(t)
    if "merged" not in tags:
        tags.append("merged")

    sample_frame = representative.sample_frame
    for f in group:
        if f.sample_frame != _FindingsSampleFrame.UNSPECIFIED:
            sample_frame = f.sample_frame
            break

    return Finding(
        claim=representative.claim,
        evidence=Evidence(summary=f"Merged from {len(group)} near-duplicate findings."),
        sample_frame=sample_frame,
        confidence=max(f.confidence for f in group),
        dataset=representative.dataset,
        derived_from=[f.id for f in group],
        evidence_links=evidence_links,
        tags=tags,
    )


@dataclass
class ConsolidationReport:
    n_input: int = 0
    n_groups_merged: int = 0
    n_output: int = 0
    merged_finding_ids: list[list[str]] = field(default_factory=list)   # [[source ids], ...]
    merged_ids: list[str] = field(default_factory=list)                  # ids of the new merged findings


class FindingsLog:
    """Append-only JSONL store of Finding objects.

    Use as a context manager or call methods directly:

    >>> log = FindingsLog("artifacts/findings.jsonl")
    >>> log.append(my_finding)
    >>> for f in log.iter():
    ...     print(f.claim)

    Sample-frame discipline (Phase 2)
    ---------------------------------
    Pass ``validator=SampleFrameValidator(...)`` at construction time.
    Every appended Finding is then auto-validated and the result is stored
    in ``Finding.validation`` before serialisation.  This is the binding
    between the *formal* sample-frame algebra and the *operational*
    findings store -- the key piece that makes Ledgerhound's discipline claim
    falsifiable rather than rhetorical.
    """

    def __init__(
        self,
        path: str | Path,
        validator: "SampleFrameValidator | None" = None,
        memory: "LongitudinalMemory | None" = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.validator = validator
        self.memory = memory
        # Touch the file so it always exists.
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def set_validator(self, validator: "SampleFrameValidator") -> None:
        """Attach a validator after construction (e.g. once memory is loaded)."""
        self.validator = validator

    def set_memory(self, memory: "LongitudinalMemory") -> None:
        """Attach a LongitudinalMemory after construction so future findings
        are auto-embedded into the text namespace for recall."""
        self.memory = memory

    def append(self, finding: Finding) -> None:
        """Append a finding. Atomic on POSIX for single lines < PIPE_BUF.

        If a SampleFrameValidator is attached and the finding has no
        existing validation result, the validator is run and the result
        attached to the finding before serialisation.
        """
        # Auto-validate via SampleFrameValidator
        if self.validator is not None and finding.validation is None:
            try:
                finding.validation = self._validate(finding)
            except Exception as exc:
                log.warning("validator failed for finding %s: %s", finding.id, exc)
                finding.validation = {
                    "is_valid": True,
                    "errors": [f"validator_exception: {exc}"],
                    "warnings": [],
                }

        line = json.dumps(finding.to_dict(), default=str, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        log.debug("Appended finding %s to %s", finding.id, self.path)

        # Mirror into LongitudinalMemory (text namespace) so future sessions
        # can recall this finding semantically. Best-effort -- never raises.
        if self.memory is not None:
            try:
                from ledgerhound.memory.longitudinal import MemorySource
                self.memory.remember(
                    source=MemorySource.FINDING,
                    text=finding.claim,
                    dataset=finding.dataset,
                    period=getattr(finding, "period", None),
                    confidence=finding.confidence,
                    tags=list(getattr(finding, "tags", []) or []),
                    extra={
                        "finding_id": finding.id,
                        "sample_frame": finding.sample_frame.value,
                    },
                    record_id=finding.id,
                )
            except Exception as exc:
                log.warning("Could not mirror finding to memory: %s", exc)

    # -- Internal: bridge findings.SampleFrame <-> discipline.SampleFrame --
    def _validate(self, finding: Finding) -> dict:
        from ledgerhound.discipline.sample_frame import (
            SampleFrame as DSampleFrame,
        )
        # Map findings.SampleFrame -> discipline.SampleFrame
        # Both have identical values except: UNSPECIFIED <-> UNKNOWN.
        f_value = finding.sample_frame.value
        if f_value == _FindingsSampleFrame.UNSPECIFIED.value:
            d_frame = DSampleFrame.UNKNOWN
        else:
            d_frame = DSampleFrame(f_value)

        result = self.validator.validate_claim(
            declared_frame=d_frame,
            stratum=finding.stratum,
            parent_sample=finding.parent_sample,
            weights_documented=finding.weights_documented,
        )
        return result.to_dict()

    def iter(self, dataset: Optional[str] = None) -> Iterator[Finding]:
        """Stream findings from disk; optionally filter by dataset."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed line in %s: %s", self.path, e)
                    continue
                if dataset and d.get("dataset") != dataset:
                    continue
                yield Finding.from_dict(d)

    def all(self, dataset: Optional[str] = None) -> list[Finding]:
        """Materialise the full log into memory. Cheap up to ~10k findings."""
        return list(self.iter(dataset=dataset))

    def by_id(self, finding_id: str) -> Optional[Finding]:
        """Locate a finding by ID. O(n) -- fine for our scale."""
        for f in self.iter():
            if f.id == finding_id:
                return f
        return None

    def count(self, dataset: Optional[str] = None) -> int:
        return sum(1 for _ in self.iter(dataset=dataset))

    # -- Research-integrity primitives ----------------------------------

    def revalidate(
        self,
        validator: "SampleFrameValidator | None" = None,
        dataset: Optional[str] = None,
    ) -> dict:
        """Re-run the sample-frame validator over every finding in the log.

        Findings that were validated at append-time under one sampling design
        become stale if the design changes (e.g. a stratum is renamed, the
        i.i.d. reservoir is added/removed).  ``revalidate()`` reads every
        finding, re-runs the (current or provided) validator, and rewrites
        the JSONL file with refreshed ``validation`` fields.

        Parameters
        ----------
        validator : SampleFrameValidator, optional
            If None, uses the validator attached at construction time.
        dataset : str, optional
            If set, only revalidate findings whose ``dataset`` matches.

        Returns
        -------
        dict with keys: ``n_total``, ``n_revalidated``, ``n_now_invalid``,
        ``n_now_warning``, ``n_status_changed``, ``audit_log`` (list of
        (id, old_status, new_status) triples).
        """
        val = validator or self.validator
        if val is None:
            raise ValueError(
                "revalidate() requires a SampleFrameValidator. Pass one "
                "explicitly or attach via set_validator()."
            )

        all_findings = self.all(dataset=dataset)
        all_other = (
            [] if dataset is None
            else [f for f in self.all() if f.dataset != dataset]
        )

        old_validator = self.validator
        self.validator = val
        audit: list[tuple[str, str, str]] = []
        n_now_invalid = 0
        n_now_warning = 0
        n_status_changed = 0
        try:
            for f in all_findings:
                old = f.validation or {}
                old_valid = bool(old.get("is_valid"))
                new_v = self._validate(f)
                new_valid = bool(new_v.get("is_valid"))
                if old_valid != new_valid or set(
                    old.get("errors", [])
                ) != set(new_v.get("errors", [])):
                    n_status_changed += 1
                    audit.append((
                        f.id,
                        "valid" if old_valid else "invalid",
                        "valid" if new_valid else "invalid",
                    ))
                f.validation = new_v
                if not new_valid:
                    n_now_invalid += 1
                if new_v.get("warnings"):
                    n_now_warning += 1
        finally:
            self.validator = old_validator

        # Rewrite the log atomically (write to temp, replace)
        import tempfile, os
        all_payload = all_other + all_findings
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=self.path.parent,
            prefix=".findings_revalidate_", suffix=".jsonl",
            encoding="utf-8",
        ) as tf:
            for f in all_payload:
                tf.write(json.dumps(f.to_dict(), default=str, ensure_ascii=False) + "\n")
            tmp_path = tf.name
        os.replace(tmp_path, self.path)
        log.info(
            "Revalidated %d findings (%d status-changed, %d invalid, %d warning)",
            len(all_findings), n_status_changed, n_now_invalid, n_now_warning,
        )
        return {
            "n_total":           len(all_findings),
            "n_revalidated":     len(all_findings),
            "n_now_invalid":     n_now_invalid,
            "n_now_warning":     n_now_warning,
            "n_status_changed":  n_status_changed,
            "audit_log":         audit,
        }

    def consolidate(
        self,
        similarity_threshold: float = 0.6,
        dataset: Optional[str] = None,
    ) -> "ConsolidationReport":
        """Merge near-duplicate findings into one, atomically rewriting the
        log (same pattern as ``revalidate()``: read all, transform, replace).

        Two findings are near-duplicates if their normalized claim text has
        a token-Jaccard overlap >= ``similarity_threshold`` AND they share
        the same ``dataset``. Merging is transitive within one pass (if A~B
        and B~C, all three merge into one group), matching how a human
        skimming a findings log would group them.

        The merged Finding keeps the highest-confidence claim as its own,
        unions tags, sets ``derived_from`` to the source finding ids, and
        populates ``evidence_links`` with one entry per source finding that
        carried real evidence (query/code/chart_path) -- this is what lets
        a merged finding still pass ``Finding.has_evidence()`` even when no
        single source had a chart_path AND a query, only pieces of each.
        """
        all_findings = self.all(dataset=dataset)
        all_other = (
            [] if dataset is None
            else [f for f in self.all() if f.dataset != dataset]
        )

        groups = _group_near_duplicates(all_findings, similarity_threshold)
        merged: list[Finding] = []
        unmerged: list[Finding] = []
        merged_finding_ids: list[list[str]] = []
        for group in groups:
            if len(group) == 1:
                unmerged.append(group[0])
            else:
                merged_finding = _merge_finding_group(group)
                merged.append(merged_finding)
                merged_finding_ids.append([f.id for f in group])

        payload = all_other + unmerged + merged
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=self.path.parent,
            prefix=".findings_consolidate_", suffix=".jsonl",
            encoding="utf-8",
        ) as tf:
            for f in payload:
                tf.write(json.dumps(f.to_dict(), default=str, ensure_ascii=False) + "\n")
            tmp_path = tf.name
        os.replace(tmp_path, self.path)

        return ConsolidationReport(
            n_input=len(all_findings),
            n_groups_merged=len(merged),
            n_output=len(unmerged) + len(merged),
            merged_finding_ids=merged_finding_ids,
            merged_ids=[f.id for f in merged],
        )

    def agreement_check(
        self,
        dataset: Optional[str] = None,
    ) -> dict:
        """Detect contradictory findings on the same (feature, period).

        Two findings *agree* if their claims align (positive lift in both,
        or negative-or-neutral in both).  Two findings *contradict* if
        they reference the same feature + period but assert opposing
        directions of effect.

        This is heuristic -- we look at sign-words in the claim (``higher``,
        ``lower``, ``increases``, ``decreases``, ``no signal``, etc.) and
        report pairs that disagree.  An analyst should review.
        """
        all_findings = self.all(dataset=dataset)
        # Group by (feature, period) tags
        groups: dict[tuple, list] = {}
        for f in all_findings:
            feat = None
            period = None
            # Try evidence + tags for feature/period hints
            for t in f.tags:
                if t.startswith("feature:"):
                    feat = t.split(":", 1)[1]
                elif t.startswith("period:"):
                    period = t.split(":", 1)[1]
            # Fall back to dataset name
            if not feat and not period:
                continue
            groups.setdefault((feat, period), []).append(f)

        contradictions: list[dict] = []
        POSITIVE = {"higher", "increase", "lift", "more", "above", "spike"}
        NEGATIVE = {"lower", "decrease", "drop", "less", "below", "decay"}
        for key, fs in groups.items():
            if len(fs) < 2:
                continue
            sentiments: list[tuple[str, str]] = []
            for f in fs:
                claim_l = f.claim.lower()
                pos = any(w in claim_l for w in POSITIVE)
                neg = any(w in claim_l for w in NEGATIVE)
                if pos and not neg:
                    sentiments.append((f.id, "positive"))
                elif neg and not pos:
                    sentiments.append((f.id, "negative"))
            if len({s for _, s in sentiments}) > 1:
                contradictions.append({
                    "feature":   key[0],
                    "period":    key[1],
                    "finding_sentiments": sentiments,
                })
        return {
            "n_groups":         len(groups),
            "n_contradictions": len(contradictions),
            "contradictions":   contradictions,
        }
