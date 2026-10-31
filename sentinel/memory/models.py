"""
sentinel.memory.models
======================

Dataclasses for Dataset Memory.

Sampling-design first-class
---------------------------
The most important dataclass here is ``SamplingDesign``. It records the
*recipe* by which a dataset was sampled -- not just "stratified" but
*which strata, by what filters, with what inclusion probabilities,
encoding which analyst priors*.

Without this, every statistic the agent reports is silently mis-
interpretable. With this, the agent can pick the right sample-frame
mode (within / cross / population) per question, and reason about what
the priors-encoded-in-sampling actually let it conclude.

Phase-2 hook
------------
``SamplingMethod.DERIVED`` plus ``SamplingStratum.derived_from_findings``
together support importance sampling driven by earlier findings.
Phase 1 doesn't *generate* derived samples, but it doesn't have to --
the model can already record one if a user supplies it manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# -- Enums --------------------------------------------------


class DatasetRole(str, Enum):
    """How Sentinel should treat a dataset."""

    FACT       = "fact"        # transactions, events
    DIMENSION  = "dimension"   # masters: merchant, bank, etc.
    ROLLUP     = "rollup"      # pre-aggregated summaries
    UNKNOWN    = "unknown"


class SamplingMethod(str, Enum):
    """How a stratum was drawn from its population."""

    SIMPLE_RANDOM = "simple_random"
    SYSTEMATIC    = "systematic"     # e.g. first N days, every 100th row
    STRATIFIED    = "stratified"     # within this stratum (recursive)
    PURPOSIVE     = "purposive"      # filter by analyst-chosen criteria
    CENSUS        = "census"         # the whole population, not a sample
    DERIVED       = "derived"        # Phase-2: derived from prior findings
    UNKNOWN       = "unknown"


# -- Sub-dataclasses ----------------------------------------


@dataclass
class SamplingStratum:
    """One stratum of a sampling design.

    Parameters
    ----------
    name           : Stratum label (e.g. ``"good"``, ``"fraud"``).
    method         : How rows were drawn from this stratum's population.
    n_sample       : Rows actually in the sample.
    n_population   : Population size (None if unknown).
    p_inclusion    : Per-row inclusion probability (None if unknown).
    filters        : Free-text predicates describing the population
                     (e.g. ``["last_6mo", "account_age_days >= 180"]``).
    derived_from_findings : Phase-2 hook -- list of finding IDs whose
                            output produced this stratum's sampling spec.
    notes          : Free text.
    """

    name: str
    method: SamplingMethod = SamplingMethod.UNKNOWN
    n_sample: Optional[int] = None
    n_population: Optional[int] = None
    p_inclusion: Optional[float] = None
    filters: list[str] = field(default_factory=list)
    derived_from_findings: list[str] = field(default_factory=list)  # Phase-2 hook
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method.value,
            "n_sample": self.n_sample,
            "n_population": self.n_population,
            "p_inclusion": self.p_inclusion,
            "filters": list(self.filters),
            "derived_from_findings": list(self.derived_from_findings),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingStratum":
        return cls(
            name=d["name"],
            method=SamplingMethod(d.get("method", "unknown")),
            n_sample=d.get("n_sample"),
            n_population=d.get("n_population"),
            p_inclusion=d.get("p_inclusion"),
            filters=list(d.get("filters") or []),
            derived_from_findings=list(d.get("derived_from_findings") or []),
            notes=d.get("notes", ""),
        )


@dataclass
class SamplingDesign:
    """The complete sampling recipe for a dataset.

    ``priors_encoded`` is the *most analytically important* field: it
    names the assumptions baked into the sample. The agent reads these
    when interpreting findings, and the what-if mode (Phase 2) tests them.
    """

    strata: dict[str, SamplingStratum] = field(default_factory=dict)
    priors_encoded: list[str] = field(default_factory=list)
    stale_if: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strata": {k: v.to_dict() for k, v in self.strata.items()},
            "priors_encoded": list(self.priors_encoded),
            "stale_if": list(self.stale_if),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingDesign":
        strata_d = d.get("strata") or {}
        return cls(
            strata={k: SamplingStratum.from_dict(v) for k, v in strata_d.items()},
            priors_encoded=list(d.get("priors_encoded") or []),
            stale_if=list(d.get("stale_if") or []),
        )


@dataclass
class DatasetRelationship:
    """A link between two datasets (typically fact -> dimension)."""

    from_dataset: str
    from_column: str
    to_dataset: str
    to_column: str
    kind: str = "join_key"      # join_key | enriches | derives_from
    notes: str = ""


@dataclass
class Rollup:
    """A pre-aggregated summary table the agent can hit instead of facts."""

    name: str                                 # "upi_hourly_summary"
    grain: str                                # "hour x city"
    source_dataset: str = ""                  # which fact dataset it summarises
    columns: list[str] = field(default_factory=list)
    refresh_cadence: str = ""                 # "daily" / "hourly" / "manual"
    location: str = ""                        # parquet path / table name
    notes: str = ""


# -- Top-level ----------------------------------------------


@dataclass
class DatasetMemory:
    """Per-dataset memory. Persisted to YAML next to the data."""

    dataset_name: str
    role: DatasetRole = DatasetRole.UNKNOWN
    description: str = ""
    schema_columns: list[str] = field(default_factory=list)
    relationships: list[DatasetRelationship] = field(default_factory=list)
    rollups: list[Rollup] = field(default_factory=list)
    sampling_design: Optional[SamplingDesign] = None
    findings_log_path: str = ""        # relative path to findings JSONL
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        """Bump updated_at -- call after any meaningful change."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # -- serialisation --------------------------------------

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "role": self.role.value,
            "description": self.description,
            "schema_columns": list(self.schema_columns),
            "relationships": [
                {
                    "from_dataset": r.from_dataset,
                    "from_column": r.from_column,
                    "to_dataset": r.to_dataset,
                    "to_column": r.to_column,
                    "kind": r.kind,
                    "notes": r.notes,
                }
                for r in self.relationships
            ],
            "rollups": [
                {
                    "name": r.name,
                    "grain": r.grain,
                    "source_dataset": r.source_dataset,
                    "columns": list(r.columns),
                    "refresh_cadence": r.refresh_cadence,
                    "location": r.location,
                    "notes": r.notes,
                }
                for r in self.rollups
            ],
            "sampling_design": self.sampling_design.to_dict() if self.sampling_design else None,
            "findings_log_path": self.findings_log_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetMemory":
        sd = d.get("sampling_design")
        return cls(
            dataset_name=d["dataset_name"],
            role=DatasetRole(d.get("role", "unknown")),
            description=d.get("description", ""),
            schema_columns=list(d.get("schema_columns") or []),
            relationships=[
                DatasetRelationship(
                    from_dataset=r["from_dataset"],
                    from_column=r["from_column"],
                    to_dataset=r["to_dataset"],
                    to_column=r["to_column"],
                    kind=r.get("kind", "join_key"),
                    notes=r.get("notes", ""),
                )
                for r in (d.get("relationships") or [])
            ],
            rollups=[
                Rollup(
                    name=r["name"],
                    grain=r.get("grain", ""),
                    source_dataset=r.get("source_dataset", ""),
                    columns=list(r.get("columns") or []),
                    refresh_cadence=r.get("refresh_cadence", ""),
                    location=r.get("location", ""),
                    notes=r.get("notes", ""),
                )
                for r in (d.get("rollups") or [])
            ],
            sampling_design=SamplingDesign.from_dict(sd) if sd else None,
            findings_log_path=d.get("findings_log_path", ""),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )
