"""
sentinel.tools.digest_tools
===========================

``session_digest`` as a shell tool -- lets the model (and the user, via
``/digest``) see today's work summary; the same digest feeds git commit
drafts and Jira updates.
"""

from __future__ import annotations

from sentinel.reporting.session_digest import SessionDigest
from sentinel.tools.registry import ToolDef


def build_digest_tools(digest: SessionDigest) -> list[ToolDef]:
    def session_digest() -> str:
        return digest.to_markdown()

    return [ToolDef(
        name="session_digest",
        description=("Summarise this session's work so far: recorded findings, "
                     "actions taken, and today's commits. Use it to draft a "
                     "work update or before pushing code."),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=session_digest)]
