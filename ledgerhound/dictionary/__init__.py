"""
ledgerhound.dictionary
===================

The **Schema Knowledge Base** (Schema KB).

A queryable, agent-callable store of *semantic knowledge* about columns,
data quality issues, reference codes (purpose / MCC / etc.), and
column-to-column relationships.

This is **distinct from** ``ledgerhound.schema`` (which holds runtime
DatasetSchema dataclasses). Think of it this way::

    ledgerhound.schema       -> "what columns exist and what is their type?"
    ledgerhound.dictionary   -> "what does each column MEAN, and what's wrong with it?"

The Schema KB seeds Phase-1 L3 org-knowledge directly from the org's hand-
curated artifacts:

* ``upi_ft_dictionary.csv``    -- column registry + quality flags
* ``upi_columns_all.txt``      -- actual DB column names (resolves naming drift)
* ``upi_purpose_codes.csv``    -- reference codes lookup

Public surface
--------------

>>> from ledgerhound.dictionary import SchemaKB, load_defaults
>>> kb = SchemaKB()
>>> load_defaults(kb)
>>> kb.lookup("payee_mobilenumber")
ColumnInfo(canonical_name='payee_mobilenumber', ..., quality_flags=[...])

For the agent, see ``ledgerhound.dictionary.tool.make_dictionary_tools``.
"""

from .models import (
    ColumnInfo,
    QualityFlag,
    ReferenceCode,
    Relationship as KBRelationship,
    QualitySeverity,
    DictionaryStatus,
)
from .store import SchemaKB
from .loader import (
    load_defaults,
    load_dictionary_csv,
    load_columns_txt,
    load_purpose_codes_csv,
    auto_align_aliases,
)
from .tool import make_dictionary_tools

__all__ = [
    "SchemaKB",
    "ColumnInfo",
    "QualityFlag",
    "ReferenceCode",
    "KBRelationship",
    "QualitySeverity",
    "DictionaryStatus",
    "load_defaults",
    "load_dictionary_csv",
    "load_columns_txt",
    "load_purpose_codes_csv",
    "auto_align_aliases",
    "make_dictionary_tools",
]
