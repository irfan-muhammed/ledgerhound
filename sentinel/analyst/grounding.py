"""
sentinel.analyst.grounding
==========================

The grounding on-ramp (Phase 2) -- the only *fixed* steps in the otherwise
fluid Analyst. Before any expensive analysis, three things must be known:

    goal            what are we trying to achieve
    data            where is the data (a path/folder, or a Trino source)
    output_format   how to save the result (default md)

Enforcement is ENFORCED-WITH-ESCAPE-HATCH (Phase 0 decision): the contract must
be satisfied before dispatch, unless the user runs ``/skip-grounding`` (expert
mode), which sets ``bypassed``. The contract itself is deterministic (no LLM);
the Analyst asks the user for whatever slot is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

# The slots, in the order the on-ramp asks for them.
SLOTS = ("goal", "data", "output_format")


@dataclass
class GroundingContract:
    goal: str = ""
    data: str = ""
    output_format: str = "md"        # has a safe default, so never blocks
    bypassed: bool = False           # set by /skip-grounding

    def set(self, slot: str, value: str) -> "GroundingContract":
        if slot not in SLOTS:
            raise ValueError(f"unknown grounding slot {slot!r}")
        setattr(self, slot, (value or "").strip() or getattr(self, slot))
        return self

    def missing(self) -> list[str]:
        """Slots still empty. output_format defaults to md so it never blocks."""
        out = []
        if not self.goal.strip():
            out.append("goal")
        if not self.data.strip():
            out.append("data")
        # output_format always has a default -> not blocking
        return out

    def is_grounded(self) -> bool:
        """True when dispatch may proceed (contract met, or expert bypass)."""
        return self.bypassed or not self.missing()

    def next_question(self) -> str | None:
        """The next prompt the on-ramp should ask, or None when grounded."""
        if self.is_grounded():
            return None
        slot = self.missing()[0]
        return {
            "goal": "What are we trying to achieve in this project?",
            "data": "Where is the data? (a file/folder path, or a Trino source)",
        }[slot]

    def skip(self) -> "GroundingContract":
        self.bypassed = True
        return self

    def summary(self) -> str:
        b = " (bypassed)" if self.bypassed else ""
        return (f"goal={self.goal or '?'} | data={self.data or '?'} | "
                f"output={self.output_format}{b}")
