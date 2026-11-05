"""
ledgerhound.kb
==============

The Knowledge Pack (design doc, "Knowledge Pack"): users' own fraud
knowledge, whose structured parts feed the guardrails directly.

    schema   entry types and strict validation
    pack     KnowledgePack: builtin / team / project scopes, add, documents
    cli      `ledgerhound kb init | add | list | search | validate`
"""

from ledgerhound.kb.pack import (
    SCOPES,
    WRITABLE,
    Entry,
    KBError,
    KnowledgePack,
    project_dir,
    team_dir,
)
from ledgerhound.kb.schema import TYPES, describe, validate

__all__ = ["KnowledgePack", "Entry", "KBError", "SCOPES", "WRITABLE", "TYPES",
           "validate", "describe", "project_dir", "team_dir"]
