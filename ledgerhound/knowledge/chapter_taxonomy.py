"""
ledgerhound.knowledge.chapter_taxonomy
====================================

V3-053: the Librarian's chapter names. A "chapter" is a declared APPROACH,
not free text and not a code hash -- per the user's own reasoning (chosen
over the alternatives): free text either never fires dedup (every phrasing
is "new") or over-fires (an LLM similarity judgment blocks legitimate
deepening), and a code-hash key means trivial formatting differences count
as a different chapter. Only an enumerated, finite vocabulary is
mechanically checkable.

Extensible on purpose -- this is a seed list, not a closed one -- but
extension is a deliberate edit to this file, never the model inventing a
new entry ad hoc at chapter-open time. That is what keeps "did we already
do this?" answerable by exact match instead of judgment.
"""

from __future__ import annotations

# name -> one-line description, shown in `/library`/`/chapter` output.
CHAPTER_TAXONOMY: dict[str, str] = {
    "univariate-profile": "per-column distribution, nulls, cardinality",
    "percentile-profile": "fraud vs non-fraud percentile table + separation callout (V3-051)",
    "interaction-sweep": "pairwise/multi-column interaction search",
    "temporal-drift": "cross-period stability of a prior finding (V3-051 C4)",
    "grain-change": "re-deriving features at a different unit of analysis",
    "dual-node-comparison": "remitter vs beneficiary side of the same feature (V3-053)",
    "leakage-audit": "target-leakage / point-in-time correctness check",
    "feature-catalog": "build_catalog / tier_features / export pass",
    "graph-structure": "account/device/network topology exploration",
    "outlier-review": "tail-value / anomaly inspection",
    "label-consistency": "checking the label definition holds across the frame",
    # V5-004: the only chapter whose purpose is to ATTACK a result rather than
    # produce one. Popper (arXiv 2502.09858) validates hypotheses by designing
    # the experiment that would refute them; everything else in this taxonomy
    # looks for support, which is the direction bias already runs in.
    "falsification": ("designing the analysis that would REFUTE a recorded "
                      "finding -- the attack, not the confirmation"),
}


def is_known_chapter(name: str) -> bool:
    return name in CHAPTER_TAXONOMY


def describe_chapter(name: str) -> str:
    return CHAPTER_TAXONOMY.get(name, "(not in the taxonomy)")


def taxonomy_block() -> str:
    """Human/LLM-readable listing, for /library and chapter-open guidance."""
    lines = ["Known chapter approaches (pick one, or ask to extend the list):"]
    for name, desc in sorted(CHAPTER_TAXONOMY.items()):
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
