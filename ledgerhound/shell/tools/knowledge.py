"""
ledgerhound.shell.tools.knowledge
=================================

The Knowledge Pack in the session: kb_add and kb_search. A write to the TEAM pack is shared with everyone, so it
goes through the person gate; the project pack does not.
"""

from __future__ import annotations

TEAM_YES = "add to the team pack"
TEAM_NO = "no, keep it out"


class KnowledgeToolsMixin:
    """kb_add, kb_search and the session's handle on the Knowledge Pack."""

    kb_actor: str = "analyst"
    _pack = None

    def knowledge_pack(self):
        if self._pack is None:
            from ledgerhound.kb import KnowledgePack
            self._pack = KnowledgePack()
        return self._pack

    def _dataset_names(self) -> list[str]:
        from ledgerhound.kb.policy import dataset_names
        return dataset_names(getattr(self, "_loaded_path", "") or "",
                             getattr(self, "_dataset_id", "") or "")

    def _pack_known_claims(self) -> list[tuple[str, str]]:
        try:
            from ledgerhound.kb.policy import known_claims
            return known_claims(self.knowledge_pack(), self._dataset_names())
        except Exception as e:  # noqa: BLE001 -- advisory, never blocks a run
            self._events.append("kb_error", op="known_claims", error=str(e)[:300])
            return []

    def _pack_ref(self, ref: str) -> str:
        """'type/id' of the entry `ref` names, or "" if there is none."""
        from ledgerhound.kb.policy import resolve_ref
        e = resolve_ref(self.knowledge_pack(), ref)
        return f"{e['type']}/{e['id']}" if e is not None else ""

    def _recorded_claims(self) -> list[str]:
        import json
        path = getattr(self, "_findings_path", None)
        out = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(str(json.loads(line).get("claim", "")))
                except ValueError:
                    continue
        return [c for c in out if c]

    def _tool_kb_search(self, query: str, max_hits: int = 8) -> str:
        """One ranked search over recorded findings (this session, earlier
        sessions on this dataset, team findings) and the Knowledge Pack."""
        from ledgerhound.kb.search import search_text
        findings = [("this session", c) for c in self._recorded_claims()]
        findings += [("earlier session on this dataset", c) for _, c in self._book_history_claims()]
        findings += [(f"team: {n}", c) for n, c in self._team_kb_claims()]
        out = search_text(query, self.knowledge_pack(), findings, k=max_hits)
        hits = sum(1 for ln in out.splitlines() if ln.startswith("- ["))
        self._events.append("kb_search", query=query[:300], hits=hits)
        return out

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
