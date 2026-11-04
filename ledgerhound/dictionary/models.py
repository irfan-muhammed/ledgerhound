"""
ledgerhound.dictionary.models
==========================

Dataclasses for the Schema Knowledge Base.

Naming convention
-----------------
* ``canonical_name`` is the **single source of truth** for a column.
  We pick one stable identifier (usually the dictionary name when present,
  otherwise the DB name) and map all other forms via aliases.
* ``db_alias`` is the actual column name as it appears in the database
  (e.g. ``txnid``).
* ``dict_name`` is the name as it appears in the data dictionary
  (e.g. ``txn_id``).
* These can differ -- the loader builds the alias graph automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# -- Enums --------------------------------------------------


class DictionaryStatus(str, Enum):
    """Status flag for a column from the source dictionary."""

    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    NEW = "new"               # New_Field -- pending UPI discussion
    DEPRECATED = "deprecated"
    UNKNOWN = "unknown"


class PopulationStatus(str, Enum):
    """Whether the column is actually populated correctly in production.

    Derived from the M / NM / NA codes in the organization data dictionary.
    """

    POPULATED = "populated"           # M  -- populated correctly
    NOT_POPULATED = "not_populated"   # NM -- populated wrongly or partially
    NOT_AVAILABLE = "not_available"   # NA -- column missing entirely
    UNKNOWN = "unknown"


class QualitySeverity(str, Enum):
    """How serious a known data-quality issue is for analytical use."""

    CRITICAL = "critical"   # column is unusable as-is
    HIGH = "high"           # column has wrong values, easy to misuse
    MEDIUM = "medium"       # mismatch between sources, partial data
    LOW = "low"             # cosmetic or rare issue
    INFO = "info"           # documented quirk, not blocking


class RelationKind(str, Enum):
    """Type of inter-column relationship."""

    SAME_ENTITY = "same_entity"     # two names for the same field (alias)
    JOIN_KEY = "join_key"           # FK-style join across tables
    DERIVED_FROM = "derived_from"   # one column computed from another
    SEMANTIC_PAIR = "semantic_pair" # e.g. payer_X / payee_X mirror pair


# -- Dataclasses --------------------------------------------


@dataclass
class ColumnInfo:
    """Everything the Schema KB knows about a single column.

    Parameters
    ----------
    canonical_name : Stable identifier we use everywhere internally.
    db_alias       : Actual DB column name (may differ from canonical).
    dict_name      : Name from the data dictionary (may differ from DB).
    dataset        : Dataset / table this column belongs to.
    semantic_type  : What kind of value it holds (id, timestamp, amount ...).
    role           : Role in fraud-modeling (entity_id, target ...).
    status         : Mandatory / Optional / New per dictionary.
    population     : Populated correctly / wrongly / missing in prod.
    description    : Human description.
    comments       : Raw analyst comment from dictionary (often quality-related).
    is_active      : Set False to mark as ignored by analysis.
    tags           : Free-form tags (e.g. ``["pii", "device", "geo"]``).
    """

    canonical_name: str
    db_alias: Optional[str] = None
    dict_name: Optional[str] = None
    dataset: Optional[str] = None
    semantic_type: Optional[str] = None
    role: Optional[str] = None
    status: DictionaryStatus = DictionaryStatus.UNKNOWN
    population: PopulationStatus = PopulationStatus.UNKNOWN
    description: str = ""
    comments: str = ""
    is_active: bool = True
    tags: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Best name to show a human."""
        return self.dict_name or self.db_alias or self.canonical_name

    @property
    def has_quality_concerns(self) -> bool:
        """True if known to have data-quality issues."""
        return self.population in (
            PopulationStatus.NOT_POPULATED,
            PopulationStatus.NOT_AVAILABLE,
        )


@dataclass
class QualityFlag:
    """A documented data-quality issue affecting a column.

    These come from analyst comments in the source dictionary.
    Examples:
        * "Server Ip is populating instead of server name"
        * "string 'no' is populating instead of N"
        * "millisecond missing"
    """

    canonical_name: str
    issue: str
    severity: QualitySeverity = QualitySeverity.MEDIUM
    source_quote: str = ""           # original analyst comment, verbatim
    source_file: str = ""            # which file/sheet it came from


@dataclass
class ReferenceCode:
    """A single entry in a code lookup (purpose codes, MCCs, etc.).

    For example:
        ReferenceCode(code="41", code_type="purpose_code",
                      description="UPI Lite Enablement & Add money",
                      product="UPI")
    """

    code: str
    code_type: str             # purpose_code | mcc | initiation_mode | ...
    description: str
    product: Optional[str] = None
    notes: str = ""


@dataclass
class Relationship:
    """Relationship between two columns in the KB.

    ``kind`` decides how the agent should treat the link:
    * SAME_ENTITY -> same column with two names; alias resolution
    * JOIN_KEY    -> FK-style; agent can auto-join
    * DERIVED_FROM-> one is a function of the other (e.g. *_nz from raw)
    * SEMANTIC_PAIR-> payer/payee mirror pair; useful for symmetry checks
    """

    col_a: str
    col_b: str
    kind: RelationKind
    notes: str = ""
