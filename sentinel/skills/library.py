"""
sentinel.skills.library
=======================

``SkillLibrary`` -- discovers authored skill docs (builtin + user) and recalls
the ones relevant to a goal, producing a budgeted injection block for the
Analyst's context. User skills (``~/.sentinel/skills`` by default) shadow
builtins of the same name, mirroring the agent registry.
"""

from __future__ import annotations

from pathlib import Path

from sentinel.skills.skill_doc import SkillDoc

# Skill procedures live in the unified prompt tree:
# sentinel/prompts/skills/*.md -- see sentinel/prompts/README.md
_BUILTIN_DIR = Path(__file__).resolve().parents[1] / "prompts" / "skills"


def _user_dir() -> Path:
    return Path.home() / ".sentinel" / "skills"


class SkillLibrary:
    def __init__(self, builtin_dir: Path | None = None,
                 user_dir: Path | None = None):
        self._builtin_dir = builtin_dir or _BUILTIN_DIR
        self._user_dir = user_dir if user_dir is not None else _user_dir()
        self._skills: dict[str, SkillDoc] = {}

    def load(self) -> "SkillLibrary":
        self._skills.clear()
        self._load_dir(self._builtin_dir, "builtin")
        self._load_dir(self._user_dir, "user")   # user shadows builtin
        return self

    def _load_dir(self, directory: Path, source: str) -> None:
        if not directory or not directory.exists():
            return
        for f in sorted(directory.glob("*.md")):
            try:
                doc = SkillDoc.from_file(f, source=source)
                self._skills[doc.name] = doc
            except Exception:
                continue        # a broken skill never breaks the library

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> SkillDoc | None:
        return self._skills.get(name)

    def all(self) -> list[SkillDoc]:
        return list(self._skills.values())

    def recall(self, query: str, *, k: int = 2, min_score: float = 0.01
               ) -> list[SkillDoc]:
        """Top-k skills relevant to ``query`` (goal/request text)."""
        scored = [(s.relevance(query), s) for s in self._skills.values()]
        scored = [(sc, s) for sc, s in scored if sc > min_score]
        scored.sort(key=lambda t: (t[0], t[1].name), reverse=True)
        return [s for _, s in scored[:k]]

    def injection_block(self, query: str, *, k: int = 2) -> str:
        """A prompt-ready block of the relevant skills, or '' if none match."""
        hits = self.recall(query, k=k)
        if not hits:
            return ""
        body = "\n\n".join(s.injection_block() for s in hits)
        return "# Relevant skills (institutional method)\n\n" + body
