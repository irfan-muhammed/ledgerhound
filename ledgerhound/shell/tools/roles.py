"""
ledgerhound.shell.tools.roles
=============================

Column roles in the session: proposed on load, confirmed through the person
gate, enforced on the steps that depend on them.

Exploration is never blocked: run_python works before confirmation, and the
report counts the cells that ran "roles unconfirmed". With
`require_confirmed_roles` on (the MCP server), the role-dependent steps
refuse until the roles are confirmed: record_finding, save_report, and
check_feature's leakage check (build_sample joins them in Phase 3 step 6).
"""

from __future__ import annotations

from pathlib import Path

YES = "yes, these roles are right"
NO = "no, some are wrong"
_MAX_QUESTION_LINES = 60


class RolesToolsMixin:
    """Column roles: propose on load, confirm through the gate, enforce."""

    _roles = None
    _roles_source: str = ""
    require_confirmed_roles: bool = False
    _cells_before_roles: int = 0

    # -- state ---------------------------------------------------------------
    @property
    def roles(self):
        return self._roles

    @property
    def roles_confirmed(self) -> bool:
        return self._roles is not None and self._roles.confirmed

    def _init_roles(self) -> str:
        """Called by load_data: reuse a confirmed map for this exact content,
        else infer one. Returns the line load_data shows."""
        from ledgerhound.roles import RoleStore, infer_roles
        self._cells_before_roles = 0
        self._reset_sample()
        cols = [str(c) for c in self._df.columns]
        stored = None
        try:
            stored = RoleStore().load(self._data_sha256, cols)
        except Exception:  # noqa: BLE001 -- a broken store never blocks a load
            stored = None
        if stored is not None:
            self._roles, self._roles_source = stored, "earlier session"
        else:
            self._roles = infer_roles(self._df, data_sha256=self._data_sha256,
                                      hints=self._pack_role_hints(cols))
            self._roles_source = "inferred"
        rm = self._roles
        self._events.append("roles_proposed", source=self._roles_source,
                            confirmed=rm.confirmed, confirmed_by=rm.confirmed_by,
                            roles={c: a.role for c, a in rm.columns.items()
                                   if a.role != "feature"})
        if rm.confirmed:
            who = "a person" if rm.confirmed_by == "person" else "the agent (unverified)"
            return (f"column roles: confirmed in an earlier session by {who} "
                    f"on {rm.confirmed_at[:10]} -- {rm.summary()}")
        return (f"column roles (proposed, NOT confirmed): {rm.summary()}.\n"
                "Call confirm_roles before recording findings: role-dependent "
                "steps (record_finding, the leakage check, the report) refuse "
                "until the user has confirmed them. Exploring with run_python "
                "is fine meanwhile.")

    def _pack_role_hints(self, cols: list) -> dict:
        try:
            from ledgerhound.kb.policy import role_hints
            return role_hints(self.knowledge_pack(), cols, self._dataset_names())
        except Exception as e:  # noqa: BLE001 -- a broken pack never blocks a load
            self._events.append("kb_error", op="role_hints", error=str(e)[:300])
            return {}

    def _note_cell_ran(self) -> None:
        if self._roles is not None and not self._roles.confirmed:
            self._cells_before_roles += 1

    # -- refusals ------------------------------------------------------------
    def _roles_refusal(self, step: str) -> str:
        if not self.require_confirmed_roles or self._roles is None:
            return ""
        if self._roles.confirmed:
            return ""
        return (f"REFUSED: {step} depends on the column roles, which are not "
                "confirmed yet. Call confirm_roles, ask the user the question it "
                "returns, then retry. Nothing was done.")

    def role_feature_refusal(self, features: list, label: str = "") -> str:
        """check_feature with confirmed roles: a column whose role makes it
        unusable as an input is refused outright, as is a label argument that
        is not the confirmed label."""
        from ledgerhound.roles import NOT_A_FEATURE, ROLES
        problems = []
        try:
            from ledgerhound.kb.policy import rule_outputs
            outs = rule_outputs(self.knowledge_pack(), self._dataset_names())
        except Exception:  # noqa: BLE001
            outs = {}
        for f in features:
            if str(f) in outs:
                problems.append(f"{f} is the output of rule {outs[str(f)]} in your knowledge "
                                "pack: it is computed downstream of the label")
        rm = self._roles
        if rm is None or not rm.confirmed:
            if problems:
                self._events.append("feature_refused_by_role", features=list(map(str, features)),
                                    label=label, problems=problems)
                return ("REFUSED: " + "; ".join(problems) + ". Nothing was computed.")
            return ""
        lab = rm.first("label")
        if label and lab and label != lab:
            problems.append(f"{label!r} is not the confirmed label column ({lab!r}); "
                            f"its role is {rm.role_of(label) or 'not a column'}")
        for f in features:
            r = rm.role_of(str(f))
            if r in NOT_A_FEATURE and str(f) not in outs:
                problems.append(f"{f} has role {r} ({ROLES[r]}) and cannot be a feature")
        if not problems:
            return ""
        self._events.append("feature_refused_by_role", features=list(map(str, features)),
                            label=label, problems=problems)
        return ("REFUSED: " + "; ".join(problems) + ". Nothing was computed. "
                "If a role is wrong, correct it with confirm_roles.")

    # -- the tool ------------------------------------------------------------
    def _tool_confirm_roles(self, roles: dict = None, absent: list = None) -> str:
        from ledgerhound.roles import PERSON_ONLY, ROLES, Assignment
        from ledgerhound.roles.model import SINGLE
        if self._df is None or self._roles is None:
            return "no data loaded -- call load_data first."
        cand = self._roles.copy()
        errors, changed = [], False
        for col, role in (roles or {}).items():
            col, role = str(col), str(role).strip().lower()
            if col not in cand.columns:
                errors.append(f"no column named {col!r}")
            elif role not in ROLES:
                errors.append(f"{role!r} (for {col}) is not a role")
            elif cand.columns[col].role != role or not cand.columns[col].settled:
                cand.columns[col] = Assignment(role, "high", "set by the user", "proposed")
                changed = True
        for r in absent or []:
            r = str(r).strip().lower()
            if r not in PERSON_ONLY:
                errors.append(f"absent= takes {', '.join(PERSON_ONLY)}; got {r!r}")
                continue
            had = cand.columns_with(r)
            if not had and cand.absent.get(r) in ("person", "agent"):
                continue                # already confirmed absent
            for c in had:
                cand.columns[c] = Assignment("feature", "high", "set by the user", "proposed")
            cand.absent[r] = "proposed"
            changed = True
        for r in SINGLE:
            if len(cand.columns_with(r)) > 1:
                errors.append(f"only one column can be {r}: {', '.join(cand.columns_with(r))}")
        if errors:
            return ("confirm_roles: " + "; ".join(errors) + ". Nothing was changed. "
                    "Roles: " + ", ".join(ROLES))
        if cand.confirmed and not changed:
            return ("column roles are already confirmed "
                    f"(by {cand.confirmed_by}, {cand.confirmed_at[:10]}): {cand.summary()}")
        if changed:
            cand.confirmed_by = ""
            for r in PERSON_ONLY:
                if cand.absent.get(r) in ("person", "agent") and not cand.columns_with(r):
                    cand.absent[r] = "proposed"

        decision = self.registry.gate.choose(self._roles_question(cand), [YES, NO])
        by = "agent" if getattr(decision, "approved_by", "person") == "agent" else "person"
        if decision.answer != YES:
            self._roles = cand          # keep the user's corrections as proposals
            self._events.append("roles_rejected", approved_by=getattr(decision, "approved_by", ""))
            return ("column roles NOT confirmed. Ask the user which roles are wrong, "
                    "then call confirm_roles again with roles={column: role} (and "
                    "absent=[...] for label, label_time or label_source the data "
                    "does not have).")
        cand.settle(by)
        self._roles = cand
        self._events.append("roles_confirmed", approved_by=by,
                            roles={c: a.role for c, a in cand.columns.items()
                                   if a.role != "feature"},
                            absent=sorted(r for r in PERSON_ONLY if not cand.columns_with(r)))
        self._persist_roles()
        closed = self._close_role_questions(cand)
        head = ("column roles CONFIRMED by the user" if by == "person" else
                "column roles CONFIRMED as relayed by the agent -- no person was "
                "verified to have checked them; the report will say so")
        tail = ""
        if not cand.columns_with("label_time"):
            tail = ("\nThere is no label-time column: label maturity cannot be "
                    "checked from the data, and build_sample will ask the user "
                    "for an assumed label lag.")
        if closed:
            tail += ("\nThese also answer the open data question(s): "
                     f"{', '.join(closed)}.")
        return f"{head}: {cand.summary()}{tail}"

    def _close_role_questions(self, rm) -> list[str]:
        """The schema questions asked at load (data_questions) that confirmed
        roles answer: which column is the label, the entity, the event time."""
        from ledgerhound.roles import ENTITY_ROLES
        answers = {"label": rm.first("label"), "event_time": rm.first("event_time"),
                   "entity": next((c for r in ENTITY_ROLES for c in rm.columns_with(r)), "")}
        open_qs = list(getattr(self, "_open_data_questions", []) or [])
        closed = [q.key for q in open_qs if answers.get(q.key)]
        if closed:
            self._open_data_questions = [q for q in open_qs if q.key not in closed]
            self._events.append("data_questions_answered_by_roles",
                                keys=closed, answers={k: answers[k] for k in closed})
        return closed

    def _roles_question(self, rm) -> str:
        from ledgerhound.roles import PERSON_ONLY
        ask = rm.needs_asking()
        lines = [f"Confirm the column roles for {Path(self._loaded_path or 'the data').name}. "
                 "Ledgerhound's guardrails use these roles, not the column names.", "",
                 "Needs your answer:"]
        lines += [ln for ln in rm.table(only=ask).splitlines()]
        for r in PERSON_ONLY:
            if not rm.columns_with(r):
                lines.append(f"  {r} -> not in the data")
        settled = [f"{c} -> {a.role}" for c, a in rm.columns.items()
                   if c not in ask and a.role != "feature"]
        if settled:
            lines += ["", "Set from strong evidence: " + "; ".join(settled)]
        n_feat = sum(1 for c, a in rm.columns.items() if c not in ask and a.role == "feature")
        if n_feat:
            lines.append(f"Plain features (no domain role): {n_feat} column(s).")
        if len(lines) > _MAX_QUESTION_LINES:
            extra = len(lines) - _MAX_QUESTION_LINES
            lines = lines[:_MAX_QUESTION_LINES] + [
                f"... and {extra} more line(s); the full list is in the ledgerhound://roles resource."]
        lines += ["", "Are these roles right?"]
        return "\n".join(lines)

    def _persist_roles(self) -> None:
        rm = self._roles
        try:
            from ledgerhound.roles import RoleStore
            RoleStore().save(rm)
        except Exception as e:  # noqa: BLE001 -- the session keeps its roles regardless
            self._events.append("roles_store_error", error=str(e)[:300])
        if self._card is not None:
            try:
                self._card.add_decision(f"column roles confirmed by {rm.confirmed_by}",
                                        rm.summary())
                self._save_card()
            except Exception:  # noqa: BLE001
                pass

    def _roles_provenance(self) -> list[str]:
        rm = self._roles
        if rm is None:
            return []
        out = []
        if rm.confirmed:
            who = ("a person" if rm.confirmed_by == "person"
                   else "**the agent, not verified by a person**")
            out.append(f"- Column roles confirmed by {who} on {rm.confirmed_at[:10]}: "
                       f"{rm.summary()}.")
            if not rm.columns_with("label_time"):
                out.append("- The data has no label-time column, so label maturity is "
                           "an assumption, not a check.")
        else:
            out.append("- **Column roles were not confirmed**; role-based checks did not run.")
        if self._cells_before_roles:
            out.append(f"- {self._cells_before_roles} analysis cell(s) ran before the "
                       "column roles were confirmed (roles unconfirmed).")
        return out
