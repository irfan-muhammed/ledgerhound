"""
sentinel.analyst.analyst
========================

``Analyst`` -- the composition seam for Phase 2. It is deliberately *not* a new
brain: it ties together the pieces the fluid loop needs and hands a ready
context + gates to the existing EDA agent engine (the ReAct loop in
``agents/eda_agent.py``), which does the heavy lifting.

Responsibilities:
  * inject project memory (SENTINEL.md) into the context preamble,
  * enforce the grounding on-ramp (with the /skip-grounding escape hatch),
  * route decisions through the HITL gate,
  * save the final report via the server-aware ``save_output``, honoring the
    project's default format and asking the user when appropriate.

Everything heavy (data, compute) stays in the engine/sandbox -- the Analyst
never pulls full data into its own context (big-data standing constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentinel.analyst.grounding import GroundingContract
from sentinel.analyst.hitl import HITLGate, auto_yes
from sentinel.analyst.save import (
    DEFAULT_FORMAT,
    SAVE_ENVELOPE,
    ReportPayload,
    SaveResult,
    envelope_text,
    save_output,
)


@dataclass
class Analyst:
    project: object = None                 # sentinel.project.Project (optional)
    grounding: GroundingContract = field(default_factory=GroundingContract)
    gate: HITLGate = field(default_factory=lambda: HITLGate(prompt=auto_yes))
    skills: object = None                  # sentinel.skills.SkillLibrary (optional)

    def __post_init__(self) -> None:
        # Seed output format from the project's default (Phase 0: md).
        if self.project is not None:
            try:
                self.grounding.output_format = self.project.config().save_format
            except Exception:
                pass
        # Load the authored skill library (lazy, non-fatal).
        if self.skills is None:
            try:
                from sentinel.skills import SkillLibrary
                self.skills = SkillLibrary().load()
            except Exception:
                self.skills = None

    # -- context assembly --------------------------------------------------
    def context_preamble(self) -> str:
        """The block prepended to the engine's context: project memory +
        grounding summary. Empty project memory contributes nothing."""
        parts = []
        if self.project is not None:
            block = self.project.memory_block()
            if block:
                parts.append(block)
        parts.append("# Session grounding\n\n" + self.grounding.summary())
        # Inject skills relevant to the goal (institutional method).
        if self.skills is not None and self.grounding.goal.strip():
            try:
                block = self.skills.injection_block(self.grounding.goal)
                if block:
                    parts.append(block)
            except Exception:
                pass
        return "\n\n".join(parts)

    # -- on-ramp -----------------------------------------------------------
    def ready(self) -> bool:
        return self.grounding.is_grounded()

    def next_onramp_question(self) -> Optional[str]:
        return self.grounding.next_question()

    def skip_grounding(self) -> None:
        self.grounding.skip()

    # -- save --------------------------------------------------------------
    def ask_save_format(self, ask: bool = True) -> str:
        """Return the format to save in. When ``ask`` and the gate is
        interactive, offer the server envelope; otherwise use the default."""
        if not ask:
            return self.grounding.output_format or DEFAULT_FORMAT
        options = list(SAVE_ENVELOPE.keys())
        d = self.gate.choose(
            "How should I save the report?\n" + envelope_text(), options)
        self.grounding.output_format = d.answer
        return d.answer

    def save(self, payload: ReportPayload, out_dir: str | Path = ".",
             fmt: Optional[str] = None, ask: bool = False) -> SaveResult:
        """Save the report. If ``fmt`` is given use it; else optionally ask;
        else fall back to the grounded/default format."""
        chosen = fmt or (self.ask_save_format(ask) if ask
                         else (self.grounding.output_format or DEFAULT_FORMAT))
        return save_output(payload, chosen, out_dir=out_dir)

    # -- books (Phase 5) ---------------------------------------------------
    def shelf(self):
        """The project's BookShelf, or None when there is no project."""
        if self.project is None:
            return None
        from sentinel.analyst.books import BookShelf
        return BookShelf(self.project)

    def recall_prior_understanding(self) -> str:
        """'Seen this dataset before' block for project open, keyed on goal+data."""
        sh = self.shelf()
        if sh is None:
            return ""
        query = f"{self.grounding.goal} {self.grounding.data}".strip()
        return sh.recall_block(query) if query else ""

    def offer_to_save_book(self, *, findings=None, report_path="",
                           digest_summary="", ask: bool = True):
        """At session end, ask (HITL confirm) whether to archive a book.
        Returns the saved Book or None. When ``ask`` is False, saves silently."""
        sh = self.shelf()
        if sh is None:
            return None
        if ask:
            d = self.gate.confirm(
                "Save this session's understanding to the project books?")
            if not d.approved:
                return None
        from sentinel.analyst.books import make_book
        book = make_book(self, findings=findings, report_path=report_path,
                         digest_summary=digest_summary)
        sh.save(book)
        return book
