"""
ledgerhound.analyst
===============

The analyst library layer: reusable capabilities the conversational shell (and
other callers) build on -- the server-aware ``save_output``, the Librarian
``books`` (understanding that compounds across sessions), ``subagent`` spawning,
and the ``Analyst`` composition helper. The primary UX is
``ledgerhound.shell.ConversationalAnalyst`` (LLM-driven).
"""

from ledgerhound.analyst.analyst import Analyst
from ledgerhound.analyst.books import Book, BookShelf, make_book
from ledgerhound.analyst.grounding import GroundingContract
from ledgerhound.analyst.hitl import HITLDecision, HITLGate, auto_yes
from ledgerhound.analyst.save import (
    SAVE_ENVELOPE,
    ReportPayload,
    SaveResult,
    envelope_text,
    save_output,
)
from ledgerhound.analyst.subagent import (
    SubagentResult,
    spawn_many,
    spawn_subagent,
)

__all__ = [
    "Analyst", "GroundingContract", "HITLGate", "HITLDecision", "auto_yes",
    "ReportPayload", "SaveResult", "save_output", "envelope_text", "SAVE_ENVELOPE",
    "spawn_subagent", "spawn_many", "SubagentResult",
    "Book", "BookShelf", "make_book",
]
