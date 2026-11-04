"""
ledgerhound.skills.skill_doc
========================

``SkillDoc`` -- an *authored* skill: a procedure written as a markdown document
with a small YAML frontmatter. Skills encode institutional method ("how we
compare fraud vs non-fraud here") and are *injected into the Analyst's context
when relevant* -- they guide the model, they do not replace it. This is distinct
from ``memory.skill_memory`` (skills the system *learns* automatically).

The scaling unit of the redesign: adding a methodology = dropping one .md file.

Format::

    ---
    name: fraud_vs_nonfraud
    when_to_use: comparing a fraud cohort against a non-fraud population
    tags: [comparison, psi, sampling]
    ---
    # Fraud vs non-fraud comparison
    1. ... procedure ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SkillDoc:
    name: str
    when_to_use: str = ""
    tags: list[str] = field(default_factory=list)
    body: str = ""
    source: str = "builtin"        # builtin | user

    def _tokens(self) -> set[str]:
        blob = f"{self.name} {self.when_to_use} {' '.join(self.tags)}".lower()
        return {t for t in _split(blob) if t}

    def relevance(self, query: str) -> float:
        """Simple token-overlap score in [0,1] of ``query`` against triggers."""
        q = {t for t in _split(query.lower()) if t}
        if not q:
            return 0.0
        mine = self._tokens()
        if not mine:
            return 0.0
        return len(q & mine) / len(q)

    def injection_block(self) -> str:
        return f"## Skill: {self.name}\n(when: {self.when_to_use})\n\n{self.body.strip()}\n"

    # -- parsing -----------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, *, source: str = "builtin",
                  fallback_name: str = "skill") -> "SkillDoc":
        meta, body = _split_frontmatter(text)
        name = str(meta.get("name") or fallback_name)
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return cls(name=name, when_to_use=str(meta.get("when_to_use", "")),
                   tags=list(tags), body=body.strip(), source=source)

    @classmethod
    def from_file(cls, path: str | Path, *, source: str = "builtin") -> "SkillDoc":
        p = Path(path)
        return cls.from_text(p.read_text(encoding="utf-8"), source=source,
                             fallback_name=p.stem)


_STOP = {"a", "an", "the", "of", "to", "or", "and", "is", "in", "on", "for",
         "vs", "we", "do", "it", "my", "at", "be", "by", "with", "this", "that",
         "whether", "over", "time", "against"}


def _split(s: str) -> list[str]:
    toks = ("".join(c if c.isalnum() else " " for c in s)).split()
    out = []
    for t in toks:
        t = t.strip("-_")
        if len(t) >= 3 and t not in _STOP:
            out.append(t)
    return out


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata dict, body). Tolerates a missing frontmatter block."""
    t = text.lstrip("\ufeff")
    if t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except Exception:
                meta = {}
            if isinstance(meta, dict):
                return meta, parts[2]
    return {}, text
