"""
ledgerhound.shell.tools.knowledge
=================================

The Knowledge Pack in the session: kb_add (and, from step 3, the search the
kb_search tool runs). A write to the TEAM pack is shared with everyone, so it
goes through the person gate; the project pack does not.
"""

from __future__ import annotations

TEAM_YES = "add to the team pack"
TEAM_NO = "no, keep it out"


class KnowledgeToolsMixin:
    """kb_add and the session's handle on the Knowledge Pack."""

    kb_actor: str = "analyst"
    _pack = None

    def knowledge_pack(self):
        if self._pack is None:
            from ledgerhound.kb import KnowledgePack
            self._pack = KnowledgePack()
        return self._pack

    def _tool_kb_add(self, entry: dict = None, document: str = "", scope: str = "project",
                     title: str = "", source: str = "") -> str:
        from ledgerhound.kb import KBError, validate
        from ledgerhound.kb.schema import describe
        if bool(entry) == bool(document):
            return "kb_add takes exactly one of entry= (a structured entry) or document= (a file path)."
        scope = (scope or "project").strip().lower()
        if entry:
            e, errs = validate(entry)
            if errs:
                return "REFUSED: entry not added: " + "; ".join(errs)
            preview = describe(e)
        else:
            preview = f"document {document}" + (f" ({source})" if source else "")
        by = self.kb_actor
        if scope == "team":
            q = ("Add this to the TEAM knowledge pack? Everyone on the team will see it, "
                 f"and it can change what Ledgerhound's checks do.\n\n{preview[:600]}")
            d = self.registry.gate.choose(q, [TEAM_YES, TEAM_NO])
            if d.answer != TEAM_YES:
                return "not added to the team pack: the user said no."
            by = f"{self.kb_actor}, team add approved by {getattr(d, 'approved_by', 'person')}"
        pack = self.knowledge_pack()
        try:
            if entry:
                added, replaced = pack.add(entry, scope=scope, added_by=by)
            else:
                added, replaced = pack.add_document(document, scope=scope, title=title,
                                                    source=source, added_by=by), False
        except (KBError, OSError) as err:
            return f"REFUSED: {err}"
        self._events.append("kb_add", scope=scope, type=added["type"], id=added["id"],
                            added_by=by, replaced=replaced)
        verb = "replaced" if replaced else "added"
        return (f"{verb} {added['type']} {added['id']} in the {scope} pack: "
                f"{describe(added.data)[:300]}\nReports quote knowledge-pack content as "
                "'from your knowledge base (unverified)' unless it is checked on the data.")
