"""
ledgerhound.skills
==============

Authored skill documents (Phase 3): procedures written as markdown-with-
frontmatter, injected into the Analyst's context when relevant. Distinct from
``memory.skill_memory`` (learned/auto-distilled skills). Adding a methodology =
dropping one .md file into the builtins/ or ~/.ledgerhound/skills folder.
"""

from ledgerhound.skills.library import SkillLibrary
from ledgerhound.skills.skill_doc import SkillDoc

__all__ = ["SkillDoc", "SkillLibrary"]
