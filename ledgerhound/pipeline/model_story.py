"""
ledgerhound.pipeline.model_story
==============================

The **model story template** (V4-010) -- a parameterized spec for the
fraud-model data pipeline, so the ~80% common case is configuration and
the remaining 20% is an override rather than a fork.

Why a template and not a hardcoded pipeline
-------------------------------------------
The canonical story (see ``docs/v3.0/next_steps.txt``) is: take raw fraud
transactions, fetch each fraud entity's history from Trino, aggregate,
design a non-fraud sample from what the fraud EDA found, fetch it, then
compare. That shape is stable across model stories; what varies is the
grain, the keys, the lookback, and the target prevalence.

**Label grain is a first-class parameter, not a puzzle to solve once.**
Institutions run both problems in parallel:

===================  ==================================  ==============================
                     ACCOUNT grain (mule / beneficiary)  TRANSACTION grain (ATO, phishing)
===================  ==================================  ==============================
label attaches to    the payee / account                 the individual transaction
class mixing         no -- whole history is fraud-side   yes -- 99% legit + 1 fraud txn
sampling unit        entity (payee + full history)       transaction (as-of features)
split rule           entity-disjoint                     temporal + point-in-time
===================  ==================================  ==============================

Nothing here computes anything. This module only *declares* the design so
downstream stages (fetch, aggregate, sample, certify) read one authoritative
spec instead of each re-deriving assumptions -- and so the certificate can
state what was intended versus what was measured.

Deliberately NOT included: physical table names and column mappings. Those
belong to the fetch layer (``ledgerhound.trino.sampling_sql``), matching how
``NonFraudSamplingDesign`` already separates *which strata* from *where the
rows live*.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

SCHEMA = "model_story/v1"

# Label grain -- see module docstring table.
GRAIN_ACCOUNT = "account"
GRAIN_TRANSACTION = "transaction"
_GRAINS = {GRAIN_ACCOUNT, GRAIN_TRANSACTION}


@dataclass
class LabelRule:
    """How a confirmed fraud report becomes row labels on fetched history.

    This is the gap that makes "550k fraud rows" undefined until answered:
    raw fraud reports arrive at some grain, but the *fetched history* rows
    are unlabelled. Is a mule payee's 500th inbound credit fraud? For a mule,
    effectively yes; for account-takeover, only the post-compromise window;
    for a phishing-induced payment the payer is a victim and the fraud entity
    is the payee. Each typology needs its own rule, so this is a list, not a
    single global setting.

    Parameters
    ----------
    typology
        Scam type this rule applies to (e.g. "mule", "ato", "phishing").
    scope
        Which of the entity's history rows inherit the fraud label:
        ``"all_history"``   -- entity is fraudulent, whole history is fraud-side
        ``"window"``        -- only rows inside [window_start_field, report date]
        ``"reported_only"`` -- only the specific reported transactions
    window_start_field
        Column naming the compromise/onset timestamp. Required for ``"window"``.
    notes
        Free text -- why this rule, who decided it.
    """

    typology: str
    scope: str = "reported_only"
    window_start_field: str = ""
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.typology.strip():
            problems.append("LabelRule.typology must not be empty")
        if self.scope not in {"all_history", "window", "reported_only"}:
            problems.append(
                f"LabelRule[{self.typology}].scope must be one of "
                f"all_history|window|reported_only; got {self.scope!r}"
            )
        if self.scope == "window" and not self.window_start_field.strip():
            problems.append(
                f"LabelRule[{self.typology}].scope='window' requires "
                f"window_start_field (the compromise/onset timestamp column)"
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRule":
        return cls(
            typology=str(d.get("typology", "")),
            scope=str(d.get("scope", "reported_only")),
            window_start_field=str(d.get("window_start_field", "")),
            notes=str(d.get("notes", "")),
        )


@dataclass
class ModelStory:
    """A complete, declarative model-story spec.

    Every downstream stage reads this rather than re-deriving assumptions.
    Unset numeric fields are ``None`` (unknown) rather than 0 -- an unknown
    prevalence must never silently read as a measured one, the same
    discipline V4-001 applied to the sampling certificate.
    """

    name: str
    label_grain: str = GRAIN_ACCOUNT

    # Fraud-side fetch
    fetch_keys: list[str] = field(default_factory=list)   # e.g. ["payee_vpa", "ifsc"]
    lookback_months: int = 6

    # Labelling -- per typology; see LabelRule.
    label_rules: list[LabelRule] = field(default_factory=list)
    # Optional, explicitly-declared fallback for typologies nobody mapped.
    # Build it with `typology.default_rule()`, which forces a stated reason.
    # None means "refuse to label an unmapped typology", which is the safe
    # default: an implicit fallback would silently decide what counts as
    # fraud for every scam type that arrives after this story was written.
    default_label_rule: Optional[LabelRule] = None
    # Reporting lag: labels arrive 30-90d late, so the most recent slice of
    # any window systematically UNDER-counts fraud. Prevalence must be
    # computed only over cohorts at least this old (V4-020/V4-022).
    label_maturity_days: int = 90

    # Aggregation
    grains_emitted: list[str] = field(default_factory=list)  # account and/or transaction

    # Prevalence. design_* is what we intend to draw; population_* is the
    # real-world base rate used to correct precision / alert volume back to
    # deployment conditions. They are different numbers and both are needed.
    #
    # *_grain is not optional decoration: account-grain and transaction-grain
    # prevalence are DIFFERENT numbers for the same data (aggregation
    # contracts rows), so a bare ratio like "1:145" is ambiguous and
    # comparing a measurement to it across grains is a category error. The
    # prevalence check refuses to compare unlike grains rather than silently
    # reporting a divergence that is really a unit mismatch.
    design_prevalence: Optional[float] = None       # e.g. 1/145
    design_prevalence_grain: str = ""               # account | transaction
    population_prevalence: Optional[float] = None   # e.g. 1/10000
    population_prevalence_grain: str = ""
    population_prevalence_source: str = ""

    notes: str = ""

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the spec is coherent.

        Warnings-as-problems by design: a story that cannot be validated
        should not silently drive a fetch of 8 crore rows.
        """
        problems: list[str] = []

        if not self.name.strip():
            problems.append("ModelStory.name must not be empty")

        if self.label_grain not in _GRAINS:
            problems.append(
                f"label_grain must be one of {sorted(_GRAINS)}; "
                f"got {self.label_grain!r}"
            )

        if not self.fetch_keys:
            problems.append(
                "fetch_keys is empty -- the fraud-side history fetch has no "
                "join key (e.g. ['payee_vpa', 'ifsc'])"
            )

        if self.lookback_months <= 0:
            problems.append(f"lookback_months must be > 0; got {self.lookback_months}")

        if self.label_maturity_days < 0:
            problems.append(
                f"label_maturity_days must be >= 0; got {self.label_maturity_days}"
            )

        if not self.label_rules:
            problems.append(
                "no label_rules -- fetched history rows would be unlabelled, "
                "so the fraud row count is undefined (see LabelRule docstring)"
            )
        if self.default_label_rule is not None:
            problems.extend(
                f"default_label_rule: {p}"
                for p in self.default_label_rule.validate())

        seen: set[str] = set()
        for rule in self.label_rules:
            problems.extend(rule.validate())
            key = rule.typology.strip().lower()
            if key in seen:
                problems.append(f"duplicate label_rule for typology {rule.typology!r}")
            seen.add(key)

        for g in self.grains_emitted:
            if g not in _GRAINS:
                problems.append(
                    f"grains_emitted contains {g!r}; must be one of {sorted(_GRAINS)}"
                )
        if self.grains_emitted and self.label_grain not in self.grains_emitted:
            problems.append(
                f"label_grain {self.label_grain!r} is not in grains_emitted "
                f"{self.grains_emitted} -- the labelled grain must be emitted"
            )

        for label, value in (("design_prevalence", self.design_prevalence),
                             ("population_prevalence", self.population_prevalence)):
            if value is not None and not (0.0 < value < 1.0):
                problems.append(
                    f"{label} must be a fraction strictly between 0 and 1 "
                    f"(e.g. 1/145 = 0.0069); got {value}"
                )

        if self.population_prevalence is not None \
                and not self.population_prevalence_source.strip():
            problems.append(
                "population_prevalence is set but population_prevalence_source "
                "is empty -- an unsourced base rate must not drive precision "
                "correction"
            )

        # A prevalence without its grain is an ambiguous number: the same data
        # has a different fraud rate at account and transaction grain.
        for label, value, grain in (
            ("design_prevalence", self.design_prevalence,
             self.design_prevalence_grain),
            ("population_prevalence", self.population_prevalence,
             self.population_prevalence_grain),
        ):
            if value is None:
                continue
            if not grain.strip():
                problems.append(
                    f"{label} is set but {label}_grain is empty -- account- and "
                    f"transaction-grain prevalence are different numbers, so a "
                    f"bare ratio cannot be checked against a measurement"
                )
            elif grain not in _GRAINS:
                problems.append(
                    f"{label}_grain must be one of {sorted(_GRAINS)}; got {grain!r}"
                )

        return problems

    def is_valid(self) -> bool:
        return not self.validate()

    # -- convenience --------------------------------------------------------

    @property
    def sampling_unit(self) -> str:
        """The unit a sample is drawn in.

        Account-grain draws ENTITIES and takes their full histories -- that is
        cluster sampling, so row-level PSI with i.i.d. thresholds runs at the
        wrong effective sample size (V4-024).
        """
        return "entity" if self.label_grain == GRAIN_ACCOUNT else "transaction"

    @property
    def is_cluster_sampled(self) -> bool:
        return self.sampling_unit == "entity"

    def rule_for(self, typology: str) -> Optional[LabelRule]:
        key = typology.strip().lower()
        for rule in self.label_rules:
            if rule.typology.strip().lower() == key:
                return rule
        return None

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["label_rules"] = [r.to_dict() for r in self.label_rules]
        d["default_label_rule"] = (
            self.default_label_rule.to_dict() if self.default_label_rule else None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelStory":
        return cls(
            name=str(d.get("name", "")),
            label_grain=str(d.get("label_grain", GRAIN_ACCOUNT)),
            fetch_keys=list(d.get("fetch_keys") or []),
            lookback_months=int(d.get("lookback_months", 6)),
            label_rules=[LabelRule.from_dict(r) for r in (d.get("label_rules") or [])],
            default_label_rule=(
                LabelRule.from_dict(d["default_label_rule"])
                if d.get("default_label_rule") else None),
            label_maturity_days=int(d.get("label_maturity_days", 90)),
            grains_emitted=list(d.get("grains_emitted") or []),
            design_prevalence=(
                float(d["design_prevalence"])
                if d.get("design_prevalence") is not None else None
            ),
            design_prevalence_grain=str(d.get("design_prevalence_grain", "")),
            population_prevalence=(
                float(d["population_prevalence"])
                if d.get("population_prevalence") is not None else None
            ),
            population_prevalence_grain=str(
                d.get("population_prevalence_grain", "")),
            population_prevalence_source=str(d.get("population_prevalence_source", "")),
            notes=str(d.get("notes", "")),
        )

    def save(self, path: Path) -> Path:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelStory":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)
