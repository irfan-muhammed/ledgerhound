"""V3-030 -- method-card ingestion into the Librarian's associative memory.

A method card (`docs/v3.0/METHOD_CARD_TEMPLATE.md`) is a markdown document,
not a code artifact -- nothing reads it by default. This module is the
thin bridge that makes "Librarian can recall the card from a relevant
query" a real, testable fact: it feeds a card's text into the same
`LongitudinalMemory` vector store `Librarian.recall()` already searches
(the `MemorySource.KNOWLEDGE_ENTRY` namespace, unused by any other caller
in this codebase -- confirmed before writing this: `ingest_run()` only
ever writes `source="finding"`), rather than inventing a new store.

Method cards are not tied to one physical dataset the way a `DatasetCard`
(`sentinel/knowledge/dataset_card.py`, a distinct and unrelated concept
despite the shared "card" word) is -- they describe a method, reusable
across datasets. They're bucketed under one reserved dataset key so a
caller can still scope `Librarian.recall(dataset=METHOD_CARD_DATASET, ...)`
to just cards if they want the full recall-bundle path rather than raw
vector search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

METHOD_CARD_DATASET = "_method_cards"


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def ingest_method_card(librarian, card_path, *, tags: Optional[list[str]] = None):
    """Read a method-card markdown file and remember it in `librarian`'s
    associative vector store, so a later `librarian.recall(...)` or
    `librarian.vectors().recall(...)` query can surface it.

    Returns the `MemoryRecord` written. Raises `FileNotFoundError` if
    `card_path` doesn't exist -- an ingestion call naming a missing card is
    a caller bug, not a recoverable "nothing to ingest" case.
    """
    path = Path(card_path)
    text = path.read_text(encoding="utf-8")
    title = _title(text, fallback=path.stem)

    from sentinel.memory.longitudinal import MemorySource

    vm = librarian.vectors()
    record = vm.remember(
        source=MemorySource.KNOWLEDGE_ENTRY,
        text=text,
        dataset=METHOD_CARD_DATASET,
        tags=["method_card", title] + list(tags or []),
        extra={"card_path": str(path), "title": title},
    )
    vm.save()
    return record
