"""
ledgerhound.pipeline.typology
===========================

**Typology reconciliation and a portable scope mapping.**

The problem this solves
-----------------------
Fraud typologies are unbounded and keep arriving: SIM swap, smishing,
investment scam, loan fraud, card-not-present, whatever next quarter brings.
The `labeller` refuses to run when a report carries a typology it has no
rule for — correct, but brittle as a workflow: one new scam name stops the
pipeline with an exception, and the only way to discover the gap is to hit it.

The structural fact that makes this tractable
----------------------------------------------
**Typologies are unbounded; the ways a report maps onto history are not.**
However many names exist, there are only a few answers to *"which of this
entity's history rows does this report make fraud?"* —
``all_history`` (never legitimate), ``window`` (compromised at a known time),
``reported_only`` (just that transaction). So a new typology needs a
**one-line mapping, never new code**.

That is not a claim the three scopes are provably exhaustive. A genuinely
novel typology could need a fourth (say, only transactions above a
threshold). The design contains that — a new scope is a change in
`model_story` and `labeller`, not a rewrite — but it is not ruled out.

What this module adds
---------------------
1. :func:`reconcile_typologies` — what is in this dataset, what is mapped,
   what is not, **with volumes**, so the gap is a report you read before
   running rather than an exception you hit during.
2. :class:`TypologyMap` — the mapping as a portable artifact that outlives
   any one dataset. The ~80% common core is reused; only the tail changes.
3. A **declared** default rule on the story, for the long tail. It is opt-in,
   it must name the scope, and every row it labels is stamped with it — an
   explicit fallback is defensible, a silent one never is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ledgerhound.pipeline.model_story import LabelRule, ModelStory

log = logging.getLogger(__name__)

SCHEMA = "typology_map/v1"

# Safest scope for an undeclared typology: claim only the reported rows.
SAFEST_SCOPE = "reported_only"


class TypologyError(ValueError):
    """A typology mapping cannot be resolved as specified."""


def _norm(t: Any) -> str:
    return str(t).strip().lower()


@dataclass
class TypologyCoverage:
    """One typology as it appears in a dataset, and whether it is mapped."""

    typology: str
    n_reports: int
    n_entities: int
    rule: Optional[LabelRule] = None
    via_default: bool = False

    @property
    def is_mapped(self) -> bool:
        return self.rule is not None

    @property
    def scope(self) -> Optional[str]:
        return self.rule.scope if self.rule else None

    def to_dict(self) -> dict:
        return {
            "typology": self.typology,
            "n_reports": self.n_reports,
            "n_entities": self.n_entities,
            "is_mapped": self.is_mapped,
            "scope": self.scope,
            "via_default": self.via_default,
        }


@dataclass
class TypologyReconciliation:
    """What this dataset contains versus what the story knows how to label."""

    coverage: list[TypologyCoverage] = field(default_factory=list)
    unused_rules: list[str] = field(default_factory=list)

    @property
    def mapped(self) -> list[TypologyCoverage]:
        return [c for c in self.coverage if c.is_mapped and not c.via_default]

    @property
    def defaulted(self) -> list[TypologyCoverage]:
        return [c for c in self.coverage if c.via_default]

    @property
    def unmapped(self) -> list[TypologyCoverage]:
        return [c for c in self.coverage if not c.is_mapped]

    @property
    def is_complete(self) -> bool:
        """True when every typology present can be labelled somehow."""
        return not self.unmapped

    @property
    def n_unmapped_reports(self) -> int:
        return sum(c.n_reports for c in self.unmapped)

    @property
    def unmapped_report_fraction(self) -> float:
        total = sum(c.n_reports for c in self.coverage)
        return (self.n_unmapped_reports / total) if total else 0.0

    def summary(self) -> str:
        lines = [f"Typology reconciliation ({SCHEMA})"]
        for c in sorted(self.coverage, key=lambda c: -c.n_reports):
            if c.is_mapped:
                how = f"{c.scope}{' (DEFAULT)' if c.via_default else ''}"
            else:
                how = "UNMAPPED"
            lines.append(
                f"  {c.typology:<24} {c.n_reports:>8,} reports  "
                f"{c.n_entities:>8,} entities  -> {how}")
        if self.unmapped:
            lines.append(
                f"  {len(self.unmapped)} unmapped typology/ies covering "
                f"{self.n_unmapped_reports:,} reports "
                f"({self.unmapped_report_fraction:.1%}). Labelling will refuse "
                f"until each is assigned a scope, or a default rule is declared.")
        if self.unused_rules:
            lines.append(
                f"  rules declared but not present in this dataset: "
                f"{sorted(self.unused_rules)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "is_complete": self.is_complete,
            "n_unmapped_reports": self.n_unmapped_reports,
            "unmapped_report_fraction": self.unmapped_report_fraction,
            "coverage": [c.to_dict() for c in self.coverage],
            "unused_rules": sorted(self.unused_rules),
        }


def reconcile_typologies(
    reports,
    story: ModelStory,
    *,
    typology_column: str,
    entity_keys: Optional[list[str]] = None,
) -> TypologyReconciliation:
    """Compare the typologies present in ``reports`` with the story's rules.

    Read this **before** labelling: it turns "a new scam type appeared" from
    an exception into a report with volumes attached, so the size of the gap
    is visible when deciding what to do about it.
    """
    if typology_column not in reports.columns:
        raise TypologyError(
            f"typology column {typology_column!r} not in reports; "
            f"available: {list(reports.columns)[:20]}")

    keys = list(entity_keys if entity_keys is not None else story.fetch_keys)
    keys = [k for k in keys if k in reports.columns]

    counts = reports[typology_column].astype(str).value_counts()
    coverage: list[TypologyCoverage] = []
    seen_rules: set[str] = set()

    for typ, n in counts.items():
        rule = story.rule_for(typ)
        via_default = False
        if rule is None:
            default = getattr(story, "default_label_rule", None)
            if default is not None:
                rule, via_default = default, True
        else:
            seen_rules.add(_norm(rule.typology))

        n_entities = 0
        if keys:
            sub = reports[reports[typology_column].astype(str) == typ]
            n_entities = len(sub[keys].drop_duplicates())

        coverage.append(TypologyCoverage(
            typology=str(typ), n_reports=int(n), n_entities=n_entities,
            rule=rule, via_default=via_default))

    unused = [r.typology for r in story.label_rules
              if _norm(r.typology) not in seen_rules]
    return TypologyReconciliation(coverage=coverage, unused_rules=unused)


@dataclass
class TypologyMap:
    """A portable typology -> scope mapping.

    Outlives any single dataset: the common core is carried between model
    stories and extended at the tail, which is what makes "80% is the same
    strategy" operational rather than aspirational.
    """

    name: str = ""
    rules: list[LabelRule] = field(default_factory=list)
    notes: str = ""

    def rule_for(self, typology: str) -> Optional[LabelRule]:
        key = _norm(typology)
        for r in self.rules:
            if _norm(r.typology) == key:
                return r
        return None

    def add(self, rule: LabelRule, *, overwrite: bool = False) -> "TypologyMap":
        existing = self.rule_for(rule.typology)
        if existing is not None:
            if not overwrite:
                raise TypologyError(
                    f"typology {rule.typology!r} is already mapped to "
                    f"{existing.scope!r}; pass overwrite=True to change it. "
                    f"Silently redefining a scope would retroactively change "
                    f"what past labels meant.")
            self.rules = [r for r in self.rules
                          if _norm(r.typology) != _norm(rule.typology)]
        problems = rule.validate()
        if problems:
            raise TypologyError("invalid LabelRule: " + "; ".join(problems))
        self.rules.append(rule)
        return self

    def merge(self, other: "TypologyMap", *, prefer: str = "self") -> "TypologyMap":
        """Combine two maps. ``prefer`` decides conflicts: ``self`` or ``other``."""
        if prefer not in {"self", "other"}:
            raise TypologyError("prefer must be 'self' or 'other'")
        out = TypologyMap(name=self.name, rules=list(self.rules), notes=self.notes)
        for r in other.rules:
            if out.rule_for(r.typology) is None:
                out.add(r)
            elif prefer == "other":
                out.add(r, overwrite=True)
        return out

    def missing_for(self, typologies) -> list[str]:
        """Which of ``typologies`` this map cannot resolve."""
        return sorted({str(t) for t in typologies if self.rule_for(t) is None})

    def apply_to(self, story: ModelStory, *, only_missing: bool = True) -> ModelStory:
        """Copy rules onto a story, by default without disturbing its own."""
        for r in self.rules:
            if only_missing and story.rule_for(r.typology) is not None:
                continue
            story.label_rules = [x for x in story.label_rules
                                 if _norm(x.typology) != _norm(r.typology)]
            story.label_rules.append(r)
        return story

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "name": self.name, "notes": self.notes,
                "rules": [r.to_dict() for r in self.rules]}

    @classmethod
    def from_dict(cls, d: dict) -> "TypologyMap":
        return cls(name=str(d.get("name", "")), notes=str(d.get("notes", "")),
                   rules=[LabelRule.from_dict(r) for r in (d.get("rules") or [])])

    def save(self, path: Path) -> Path:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "TypologyMap":
        import yaml
        return cls.from_dict(
            yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def default_rule(scope: str = SAFEST_SCOPE, *, reason: str = "") -> LabelRule:
    """Build an explicitly-declared fallback rule for unmapped typologies.

    The typology is fixed to ``"__default__"`` so it can never collide with a
    real one, and every row labelled through it is stamped with that name —
    an explicit fallback is defensible precisely because it is visible in the
    output. ``reason`` is required: a default nobody justified is the kind of
    thing that quietly becomes permanent.
    """
    if not reason.strip():
        raise TypologyError(
            "a default rule requires a reason -- it silently governs every "
            "typology nobody mapped, so why it is safe must be recorded")
    rule = LabelRule(typology="__default__", scope=scope, notes=reason)
    problems = rule.validate()
    if problems:
        raise TypologyError("invalid default rule: " + "; ".join(problems))
    if scope != SAFEST_SCOPE:
        log.warning(
            "default typology rule uses scope %r rather than the safest %r -- "
            "this will claim more history as fraud for every typology nobody "
            "mapped", scope, SAFEST_SCOPE)
    return rule
