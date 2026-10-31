"""
sentinel.pipeline.certificate
==============================

**The construction certificate** (V4-023) — the keystone, and the artifact
the product claim actually names.

What it is, and what it is not
-------------------------------
`sentinel/sampling/certificate.py` certifies one thing: whether a non-fraud
*draw* is representative on its features (per-feature PSI). That is still
valid and is not replaced.

This certifies something larger and different: **how the whole sample was
constructed** — grain, label rule and maturity, typology coverage,
prevalence against its declared design, the split policy, cluster-aware PSI,
and label provenance — assembled onto one face, and then used to **gate**
the downstream feature search.

Why the gate is the product
---------------------------
Every comparable tool emits advisory data-quality warnings that a user
clicks through. Here :meth:`ConstructionCertificate.require_certified`
**raises**. Feature search over an uncertified sample optimises against a
sampling artifact, so the refusal is not a courtesy — it is the contract,
and it is the one clause in the product claim that competitors do not have.

Three rules this artifact enforces on itself
---------------------------------------------
1. **Never assert what was not measured.** A section that was not run is
   ``NOT_ASSESSED``, never ``PASS``. This generalises V4-001's fix, where a
   certificate printed a baseline ratio nobody had measured.
2. **Absence of a required section blocks certification.** A certificate
   silent on prevalence is not a certificate that passed prevalence.
3. **Every limit travels with the verdict.** ``not_checked`` is aggregated
   from all sections and printed, so a ``PASS`` can never be read as
   "everything was verified".

The content hash (:meth:`content_hash`) is a sha256 over the canonical
serialisation, so a certificate can be anchored in the existing
hash-chained event log and a later replay can prove it was not edited.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sentinel.pipeline.appropriateness import (
    SEV_FAIL,
    SEV_PASS,
    SEV_WARNING,
    Finding,
)

log = logging.getLogger(__name__)

SCHEMA = "construction_certificate/v1"

STATUS_PASS = SEV_PASS
STATUS_WARNING = SEV_WARNING
STATUS_FAIL = SEV_FAIL
STATUS_NOT_ASSESSED = "NOT_ASSESSED"

# Worst-wins ordering. NOT_ASSESSED sits above WARNING but below FAIL: it
# does not prove a problem, but it must never be mistaken for a pass.
_ORDER = {
    STATUS_PASS: 0,
    STATUS_WARNING: 1,
    STATUS_NOT_ASSESSED: 2,
    STATUS_FAIL: 3,
}

KNOWN_STATUSES = frozenset(_ORDER)

SECTION_STORY = "story"
SECTION_TYPOLOGY = "typology"
SECTION_LABELLING = "labelling"
SECTION_PREVALENCE = "prevalence"
SECTION_SPLIT = "split"
SECTION_PSI = "psi"
SECTION_APPROPRIATENESS = "appropriateness"
SECTION_CONTAMINATION = "contamination"

# Sections without which the word "certificate" is not earned.
DEFAULT_REQUIRED = (SECTION_STORY, SECTION_LABELLING, SECTION_PREVALENCE)


class UncertifiedSampleError(RuntimeError):
    """Raised when work proceeds on a sample that is not certified.

    Deliberately not a warning. The gate is the product claim.
    """


@dataclass
class CertificateSection:
    """One assessed aspect of how the sample was constructed."""

    name: str
    status: str = STATUS_NOT_ASSESSED
    detail: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)

    def __post_init__(self):
        # An unrecognised status would sort above FAIL in the worst-wins
        # ordering and, worse, slip past the gate -- so it is a hard error at
        # the point it is introduced rather than a silent certification.
        if self.status not in KNOWN_STATUSES:
            raise ValueError(
                f"CertificateSection[{self.name}].status {self.status!r} is not "
                f"one of {sorted(KNOWN_STATUSES)}. An unrecognised status "
                f"cannot be ranked, and must never be allowed to certify."
            )

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEV_FAIL]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "findings": [f.to_dict() for f in self.findings],
            "not_checked": list(self.not_checked),
        }


@dataclass
class ConstructionCertificate:
    """How this sample was built, whether that is defensible, and the gate."""

    story_name: str
    grain: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sections: dict = field(default_factory=dict)
    required_sections: tuple = DEFAULT_REQUIRED
    notes: str = ""

    # -- verdict ------------------------------------------------------------

    @property
    def missing_required(self) -> list[str]:
        """Required sections that were never assessed."""
        return [
            name for name in self.required_sections
            if name not in self.sections
            or self.sections[name].status == STATUS_NOT_ASSESSED
        ]

    @property
    def unrecognised_statuses(self) -> list[str]:
        """Section statuses this certificate cannot rank.

        `CertificateSection` rejects these at construction, so reaching here
        means one was assigned afterwards. Either way the certificate must
        fail closed rather than rank an unknown value above FAIL.
        """
        return sorted({
            f"{name}={sec.status!r}"
            for name, sec in self.sections.items()
            if sec.status not in KNOWN_STATUSES
        })

    @property
    def status(self) -> str:
        # An unrankable status fails closed: it cannot be shown to pass.
        if self._has_fail() or self.unrecognised_statuses:
            return STATUS_FAIL
        if self.missing_required or not self.sections:
            return STATUS_NOT_ASSESSED
        return max((s.status for s in self.sections.values()),
                   key=lambda s: _ORDER[s])

    def _has_fail(self) -> bool:
        return any(s.status == STATUS_FAIL for s in self.sections.values())

    @property
    def can_certify(self) -> bool:
        """The gate. A FAIL, an unrankable status, or a missing required
        section blocks it."""
        return (not self._has_fail()
                and not self.missing_required
                and not self.unrecognised_statuses)

    @property
    def blocking_reasons(self) -> list[str]:
        reasons: list[str] = []
        for bad in self.unrecognised_statuses:
            reasons.append(
                f"section status {bad} cannot be ranked -- an unrecognised "
                f"status is treated as a failure, never as a pass")
        for name in self.missing_required:
            reasons.append(
                f"required section {name!r} was not assessed -- a certificate "
                f"silent on it is not a certificate that passed it")
        for name, sec in sorted(self.sections.items()):
            for f in sec.blocking_findings:
                reasons.append(f"[{name}] {f.code}: {f.message}")
            if sec.status == STATUS_FAIL and not sec.blocking_findings:
                reasons.append(f"[{name}] section failed")
        return reasons

    @property
    def all_not_checked(self) -> list[str]:
        """Every stated limit, from every section."""
        out: list[str] = []
        for name in sorted(self.sections):
            out.extend(f"[{name}] {n}" for n in self.sections[name].not_checked)
        return out

    def require_certified(self, action: str = "feature search") -> None:
        """Raise unless this sample is certified. **This is the gate.**

        Call it before any work whose validity depends on the sample having
        been constructed as declared.
        """
        if self.can_certify:
            return
        reasons = "\n  - ".join(self.blocking_reasons) or "unknown"
        raise UncertifiedSampleError(
            f"refusing to run {action}: this sample is not certified.\n"
            f"  - {reasons}\n"
            f"Running {action} on it would optimise against a sampling "
            f"artifact rather than fraud behaviour. Resolve the blocking "
            f"items, or proceed explicitly and tag every result "
            f"'uncertified_sample'."
        )

    # -- integrity ----------------------------------------------------------

    def canonical(self) -> str:
        """Stable serialisation used for hashing (excludes the hash itself)."""
        d = self.to_dict(include_hash=False)
        return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)

    def content_hash(self) -> str:
        """sha256 over the canonical form, for the hash-chained event log."""
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    # -- reporting ----------------------------------------------------------

    def summary(self) -> str:
        gate = "CERTIFIED" if self.can_certify else "NOT CERTIFIED"
        lines = [
            f"Construction certificate ({SCHEMA})",
            f"  story={self.story_name!r}  grain={self.grain}  "
            f"generated={self.generated_at}",
            f"  verdict: {self.status}   gate: {gate}",
        ]
        for name in sorted(self.sections):
            sec = self.sections[name]
            lines.append(f"  [{sec.status:<12}] {name}")
            for f in sec.findings:
                if f.severity != SEV_PASS:
                    lines.append(f"      {f.severity}: {f.message}")
        if not self.can_certify:
            lines.append("  BLOCKED BY:")
            lines.extend(f"    - {r}" for r in self.blocking_reasons)
        limits = self.all_not_checked
        if limits:
            lines.append("  NOT CHECKED (a PASS does not cover these):")
            lines.extend(f"    - {n}" for n in limits)
        lines.append(f"  content_hash: {self.content_hash()[:16]}...")
        return "\n".join(lines)

    def to_dict(self, include_hash: bool = True) -> dict:
        d = {
            "schema": SCHEMA,
            "story_name": self.story_name,
            "grain": self.grain,
            "generated_at": self.generated_at,
            "status": self.status,
            "can_certify": self.can_certify,
            "required_sections": list(self.required_sections),
            "missing_required": self.missing_required,
            "blocking_reasons": self.blocking_reasons,
            "not_checked": self.all_not_checked,
            "sections": {k: v.to_dict() for k, v in sorted(self.sections.items())},
            "notes": self.notes,
        }
        if include_hash:
            d["content_hash"] = self.content_hash()
        return d

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ConstructionCertificate":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = {}
        for name, s in (raw.get("sections") or {}).items():
            status = s.get("status", STATUS_NOT_ASSESSED)
            if status not in KNOWN_STATUSES:
                # A certificate written by a different version may carry a
                # status this one cannot rank. Coerce to FAIL rather than
                # raise -- refusing to load would lose the audit record, and
                # anything unrankable must not certify.
                log.warning(
                    "certificate section %r has unrecognised status %r; "
                    "coercing to %s so it cannot certify", name, status,
                    STATUS_FAIL)
                status = STATUS_FAIL
            sections[name] = CertificateSection(
                name=s.get("name", name),
                status=status,
                detail=s.get("detail") or {},
                findings=[Finding(**f) for f in (s.get("findings") or [])],
                not_checked=list(s.get("not_checked") or []),
            )
        return cls(
            story_name=raw.get("story_name", ""),
            grain=raw.get("grain", ""),
            generated_at=raw.get("generated_at", ""),
            sections=sections,
            required_sections=tuple(raw.get("required_sections") or DEFAULT_REQUIRED),
            notes=raw.get("notes", ""),
        )

    def verify_hash(self, expected: str) -> bool:
        """True when the content still hashes to ``expected`` (tamper-evident)."""
        return self.content_hash() == expected


class CertificateBuilder:
    """Assembles a certificate from what the other modules already produce.

    Nothing is recomputed here: each ``with_*`` takes the artifact its module
    returns. That keeps one owner per calculation and makes the certificate a
    *record* rather than a second, divergent implementation.
    """

    def __init__(self, story, *, grain: str = "",
                 required_sections: tuple = DEFAULT_REQUIRED):
        self._story = story
        self._grain = grain or getattr(story, "label_grain", "")
        self._sections: dict = {}
        self._required = tuple(required_sections)
        self._notes = ""
        self._add_story_section(story)

    # -- sections -----------------------------------------------------------

    def _add_story_section(self, story) -> None:
        problems = story.validate() if hasattr(story, "validate") else []
        findings = [
            Finding("STORY_INVALID", SEV_FAIL, p, {}) for p in problems
        ]
        self._sections[SECTION_STORY] = CertificateSection(
            name=SECTION_STORY,
            status=STATUS_FAIL if problems else STATUS_PASS,
            detail={
                "name": getattr(story, "name", ""),
                "label_grain": getattr(story, "label_grain", ""),
                "grains_emitted": list(getattr(story, "grains_emitted", []) or []),
                "fetch_keys": list(getattr(story, "fetch_keys", []) or []),
                "lookback_months": getattr(story, "lookback_months", None),
                "label_maturity_days": getattr(story, "label_maturity_days", None),
                "label_rules": [r.to_dict() for r in
                                getattr(story, "label_rules", []) or []],
                "default_label_rule": (
                    story.default_label_rule.to_dict()
                    if getattr(story, "default_label_rule", None) else None),
                "design_prevalence": getattr(story, "design_prevalence", None),
                "design_prevalence_grain": getattr(
                    story, "design_prevalence_grain", ""),
            },
            findings=findings,
        )

    def with_typology(self, reconciliation) -> "CertificateBuilder":
        unmapped = reconciliation.unmapped
        findings = []
        if unmapped:
            findings.append(Finding(
                "UNMAPPED_TYPOLOGIES", SEV_FAIL,
                f"{len(unmapped)} typology/ies have no label rule, covering "
                f"{reconciliation.n_unmapped_reports:,} reports "
                f"({reconciliation.unmapped_report_fraction:.1%}): "
                f"{[c.typology for c in unmapped]}. Labelling cannot proceed.",
                {"unmapped": [c.typology for c in unmapped]}))
        if reconciliation.defaulted:
            findings.append(Finding(
                "DEFAULT_RULE_USED", SEV_WARNING,
                f"{len(reconciliation.defaulted)} typology/ies were labelled by "
                f"the declared default rather than a specific rule: "
                f"{[c.typology for c in reconciliation.defaulted]}",
                {"defaulted": [c.typology for c in reconciliation.defaulted]}))
        self._sections[SECTION_TYPOLOGY] = CertificateSection(
            name=SECTION_TYPOLOGY,
            status=_worst(findings),
            detail=reconciliation.to_dict(),
            findings=findings,
        )
        return self

    def with_labelling(self, result) -> "CertificateBuilder":
        findings = []
        if result.n_fraud == 0:
            findings.append(Finding(
                "NO_FRAUD_LABELLED", SEV_FAIL,
                "no rows were labelled fraud -- nothing downstream can be "
                "modelled or certified", {}))
        if result.entities_without_report:
            findings.append(Finding(
                "ENTITIES_WITHOUT_REPORT", SEV_WARNING,
                f"{result.entities_without_report} entities in history had no "
                f"matching report and were left unlabelled", {}))
        if result.unresolvable_window_entities:
            findings.append(Finding(
                "UNRESOLVABLE_WINDOW", SEV_WARNING,
                f"{len(result.unresolvable_window_entities)} 'window' entities "
                f"lacked a usable compromise/report timestamp and were left "
                f"unlabelled rather than guessed", {}))
        if getattr(result, "n_via_default_rule", 0):
            findings.append(Finding(
                "LABELLED_VIA_DEFAULT", SEV_WARNING,
                f"{result.n_via_default_rule:,} rows were labelled by the "
                f"declared default rule, covering "
                f"{getattr(result, 'typologies_via_default', [])}", {}))
        self._sections[SECTION_LABELLING] = CertificateSection(
            name=SECTION_LABELLING,
            status=_worst(findings),
            detail=result.to_dict(),
            findings=findings,
            not_checked=[
                "whether unlabelled rows conceal unreported fraud "
                "(unknowable by construction -- see the contamination section)"
            ],
        )
        return self

    def with_prevalence(self, check, correction=None) -> "CertificateBuilder":
        from sentinel.pipeline.prevalence import (
            STATUS_FAIL as P_FAIL,
        )
        from sentinel.pipeline.prevalence import (
            STATUS_INCOMPARABLE as P_INCOMP,
        )
        from sentinel.pipeline.prevalence import (
            STATUS_PASS as P_PASS,
        )
        from sentinel.pipeline.prevalence import (
            STATUS_UNDECLARED as P_UNDECL,
        )
        from sentinel.pipeline.prevalence import (
            STATUS_WARNING as P_WARN,
        )
        sev = {P_PASS: SEV_PASS, P_WARN: SEV_WARNING, P_FAIL: SEV_FAIL,
               P_UNDECL: SEV_WARNING, P_INCOMP: SEV_WARNING}[check.status]
        findings = ([] if sev == SEV_PASS
                    else [Finding("PREVALENCE", sev, check.message,
                                  {"status": check.status})])
        detail = {"check": check.to_dict()}
        not_checked = []
        if correction is not None:
            detail["population_correction"] = correction.to_dict()
        else:
            not_checked.append(
                "no population correction recorded -- precision and alert "
                "volume here describe the SAMPLE's prevalence, not deployment")
        self._sections[SECTION_PREVALENCE] = CertificateSection(
            name=SECTION_PREVALENCE, status=_worst(findings),
            detail=detail, findings=findings, not_checked=not_checked,
        )
        return self

    def with_split(self, split_result) -> "CertificateBuilder":
        findings = list(split_result.findings)
        self._sections[SECTION_SPLIT] = CertificateSection(
            name=SECTION_SPLIT,
            status=_worst(findings),
            detail=split_result.to_dict(),
            findings=findings,
            not_checked=list(split_result.not_guaranteed),
        )
        return self

    def with_cluster_psi(self, assessments) -> "CertificateBuilder":
        assessments = list(assessments)
        findings = []
        disagreeing = [a for a in assessments if a.verdicts_disagree]
        for a in disagreeing:
            findings.append(Finding(
                "PSI_VERDICTS_DISAGREE", SEV_WARNING, a.message,
                {"feature": a.feature}))
        self._sections[SECTION_PSI] = CertificateSection(
            name=SECTION_PSI,
            status=_worst(findings),
            detail={
                "computation_grain": (assessments[0].computation_grain
                                      if assessments else None),
                "features": [a.to_dict() for a in assessments],
            },
            findings=findings,
            not_checked=[
                "PSI thresholds are asymptotic and assume a known reference "
                "distribution; one feature-level ICC is applied to every bin"
            ] if assessments else [],
        )
        return self

    def with_appropriateness(self, report) -> "CertificateBuilder":
        self._sections[SECTION_APPROPRIATENESS] = CertificateSection(
            name=SECTION_APPROPRIATENESS,
            status=report.status,
            detail=report.to_dict(),
            findings=list(report.findings),
            not_checked=list(report.not_checked),
        )
        return self

    def with_contamination(self, spike_in, *, threshold=None) -> "CertificateBuilder":
        robust = spike_in.is_robust(threshold=threshold) if threshold is not None \
            else spike_in.is_robust()
        findings = []
        if spike_in.sign_flips():
            findings.append(Finding(
                "CONTAMINATION_SIGN_FLIP", SEV_FAIL,
                f"{spike_in.statistic_name} changes sign under plausible "
                f"contamination -- the conclusion it supports is not safe to "
                f"state", {}))
        elif not robust:
            findings.append(Finding(
                "CONTAMINATION_SENSITIVE", SEV_WARNING,
                f"{spike_in.statistic_name} is sensitive to contamination of "
                f"the not-reported class", {}))
        self._sections[SECTION_CONTAMINATION] = CertificateSection(
            name=SECTION_CONTAMINATION,
            status=_worst(findings),
            detail=spike_in.to_dict(),
            findings=findings,
        )
        return self

    def with_notes(self, notes: str) -> "CertificateBuilder":
        self._notes = notes
        return self

    def build(self) -> ConstructionCertificate:
        return ConstructionCertificate(
            story_name=getattr(self._story, "name", ""),
            grain=self._grain,
            sections=dict(self._sections),
            required_sections=self._required,
            notes=self._notes,
        )


def _worst(findings: list[Finding]) -> str:
    """Worst severity among findings; PASS when there are none."""
    if not findings:
        return STATUS_PASS
    return max((f.severity for f in findings), key=lambda s: _ORDER.get(s, 99))
