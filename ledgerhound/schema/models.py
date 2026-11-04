"""
ledgerhound.schema.models
======================

Core dataclasses representing a dataset schema.  Every other module
speaks in terms of these types.

Hierarchy::

    DatasetSchema
      +-- tables:  dict[alias -> TableSchema]
      |     +-- columns: dict[name -> ColumnSchema]
      +-- relationships: list[Relationship]
      +-- target: TargetConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# -- Enums --------------------------------------------------

class SemanticType(str, Enum):
    """What kind of data a column holds (independent of SQL type)."""
    IDENTIFIER  = "identifier"
    NUMERIC     = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL    = "temporal"
    BINARY      = "binary"
    TEXT        = "text"
    HASH        = "hash"
    UNKNOWN     = "unknown"


class ColumnRole(str, Enum):
    """The purpose of a column in the fraud-modeling context."""
    PRIMARY_KEY   = "primary_key"
    ENTITY_ID     = "entity_id"
    AMOUNT        = "amount"
    TIMESTAMP     = "timestamp"
    TARGET        = "target"
    FEATURE       = "feature"
    METADATA      = "metadata"
    RESPONSE_CODE = "response_code"
    STATUS        = "status"
    UNKNOWN       = "unknown"


class PIIStatus(str, Enum):
    """Whether the column contains personally identifiable information."""
    RAW      = "raw"
    HASHED   = "hashed"
    REDACTED = "redacted"
    NONE     = "none"


# -- Dataclasses --------------------------------------------

@dataclass
class ColumnSchema:
    """Schema for a single column.

    Parameters
    ----------
    name         : Exact column name in the data.
    dtype        : Raw data type string (VARCHAR, BIGINT, int64 ...).
    description  : Human description from data dictionary.
    semantic_type: Inferred or declared classification.
    role         : Purpose in fraud-modeling (amount, target, etc.).
    sample_values: Up to 5 representative values.
    transform    : Display transform (e.g. ``"divide_by_100"``).
    display_unit : Unit after transform (e.g. ``"INR"``).
    pii_status   : Whether values are hashed / raw / not PII.
    links_to     : FK target as ``"table.column"`` string.
    provided     : True if from user schema, False if auto-inferred.
    tags         : Free-form tags for knowledge-base retrieval.
    """
    name: str
    dtype: str
    description: str = ""
    semantic_type: SemanticType = SemanticType.UNKNOWN
    role: ColumnRole = ColumnRole.UNKNOWN
    sample_values: list[Any] = field(default_factory=list)
    transform: str | None = None
    display_unit: str | None = None
    pii_status: PIIStatus = PIIStatus.NONE
    links_to: str | None = None
    provided: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class TableSchema:
    """Schema for one table or file.

    Parameters
    ----------
    name        : Full SQL name (e.g. ``"upi_txn.urcs_ft_txns"``).
    alias       : Short key (e.g. ``"transactions"``).
    columns     : Column-name -> ColumnSchema mapping.
    primary_key : Name of the PK column, if known.
    row_grain   : What one row represents (``"one row per txn"``).
    """
    name: str
    alias: str = ""
    description: str = ""
    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    primary_key: str | None = None
    row_grain: str = ""

    @property
    def column_names(self) -> list[str]:
        return list(self.columns.keys())


@dataclass
class Relationship:
    """A foreign-key join between two tables.

    Direction: ``from_table.from_column -> to_table.to_column``
    (many-to-one by default).
    """
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    join_type: str = "many_to_one"
    nullable: bool = False
    description: str = ""


@dataclass
class TargetConfig:
    """How the fraud label is defined.

    Modes: explicit | proxy | external | none
    """
    mode: str = "none"
    column: str | None = None
    table: str | None = None
    positive_class: list[str] | None = None
    source_file: str | None = None
    join_key: str | None = None


@dataclass
class DatasetSchema:
    """Top-level container for the full dataset schema."""
    tables: dict[str, TableSchema] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    primary_table: str = ""
    target: TargetConfig = field(default_factory=TargetConfig)

    @property
    def primary(self) -> TableSchema | None:
        """The primary (fact) table, or None."""
        return self.tables.get(self.primary_table)

    def get_table_by_name(self, full_name: str) -> TableSchema | None:
        """Look up a table by its full SQL name."""
        for t in self.tables.values():
            if t.name == full_name:
                return t
        return None

    def relationships_for(self, table_alias: str) -> list[Relationship]:
        """All relationships involving a given table."""
        table = self.tables.get(table_alias)
        if not table:
            return []
        name = table.name
        return [r for r in self.relationships
                if r.from_table == name or r.to_table == name]
