"""
ledgerhound.shell.tools.workflow
================================

Methodology, plan steps and phases, the session goal and asking the user.
"""

from __future__ import annotations

import json





class WorkflowToolsMixin:
    """Methodology, plan steps and phases, the session goal and asking the user."""

    def _tool_consult_skill(self, query: str) -> str:
        try:
            from ledgerhound.skills import SkillLibrary
            hits = SkillLibrary().load().recall(query, k=1)
            if not hits:
                return f"no team skill matches '{query}' -- use your own judgment."
            s = hits[0]
            return (f"Skill '{s.name}' (when: {s.when_to_use}). Reference -- adapt, "
                    f"don't obey blindly:\n\n{s.body.strip()[:2500]}")
        except Exception as e:  # noqa: BLE001
            return f"could not load skills: {e}"

    def _tool_ask_user(self, question: str, options: list, multi: bool = False) -> str:
        """V3-032: a real, blocking MCQ/MSQ -- routes through the same
        HITLGate the high-tier save_report confirmation already uses, so
        this is a genuine pause for a real answer, not just text the model
        writes and hopes gets read. Low tier (see hitl.classify_action):
        the blocking happens HERE, inside the tool body via gate.choose[_multi],
        not via the tier wrapper around dispatch -- calling ask_user is itself
        already "asking", so there is nothing to additionally gate before it runs.
        """
        opts = [str(o) for o in (options or [])]
        if not opts:
            return "ask_user needs at least one option -- give the analyst something concrete to pick from."
        # I-04: an UNATTENDED run (--autonomous) has nobody to answer, and
        # the terminal prompt below would sit there for hours (a production
        # run once idled ~59 minutes on exactly this). The question
        # firewall only covers prose questions; the ask_user TOOL still
        # blocked. Take the first option -- the system prompt asks the model
        # to put its own recommendation among the options, and by the same
        # convention the HITL gate already uses (options[0] = the safe/
        # recommended one) -- and say so in the observation so the model
        # knows it was a default, not a human decision. Governance asks
        # (modelling authorisation, high-tier writes) are NOT routed here
        # and stay blocking.
        if self.autonomous:
            self._log("hitl", tier="ask_user", name="ask_user", approved=True,
                      answer=opts[0], auto=True)
            return (f"analyst picked: {opts[0]} (AUTONOMOUS RUN -- no human "
                    "was available, so the first option was taken as your "
                    "recommendation; state this assumption in your report "
                    "and revisit it if later evidence contradicts it)")
        gate = self.registry.gate
        if multi:
            decision = gate.choose_multi(question, opts)
            self._log("hitl", tier="ask_user", name="ask_user",
                      approved=True, answer=decision.answer)
            return f"analyst picked: {', '.join(decision.picked)}"
        decision = gate.choose(question, opts)
        self._log("hitl", tier="ask_user", name="ask_user",
                  approved=True, answer=decision.answer)
        return f"analyst picked: {decision.answer}"

    # -- methodology (V3-043) --------------------------------------------
    _AUTONOMOUS_DEFAULT_METHODOLOGY = "topdown_7phase"

    def _choose_methodology_once(self) -> str:
        """Ask the analyst which methodology to work, once, on first load.

        WHY ASK RATHER THAN PICK: the feedback that produced this feature
        was that each analyst follows their own sequence based on their own
        experience -- one works top-down, another lets phase 1 decide the
        sampling. That is a genuine judgment call about how THIS person
        works, not something inferable from the data, so the model must not
        guess it.

        Three ways it gets decided, in order:
          1. --methodology on the CLI      -> already set, never asked
          2. interactive session           -> asked here, once
          3. autonomous/unattended run     -> a stated default, NOT a
             blocking prompt (an earlier production run sat idle for ~59
             minutes waiting on a human for an ask_user; an unattended job
             must never do that again)
        """
        if self.methodology is not None or self._methodology_asked:
            return ""
        self._methodology_asked = True
        try:
            from ledgerhound.methodology import available_templates, load_template
            names = available_templates()
            if not names:
                return ""
            if self.autonomous:
                m = load_template(self._AUTONOMOUS_DEFAULT_METHODOLOGY) \
                    or load_template(names[0])
                if m is None:
                    return ""
                self.methodology = m
                if m.phases:
                    m.phases[0].status = "active"
                self._events.append("methodology_set", template=m.name,
                                    chosen_by="autonomous_default")
                self._save_methodology()
                self._refresh_schema_pin()
                return (f"[METHODOLOGY] autonomous run -- defaulted to "
                        f"'{m.name}'. Pass --methodology to choose a "
                        f"different one.\n\n{m.pin_block()}")

            # Only ask when a HUMAN can actually answer. HITLGate's default
            # prompt is `auto_yes`, which returns options[0] -- asking
            # through that would "choose" the first template on the
            # analyst's behalf, which is precisely the silent guess this
            # feature exists to prevent. With no interactive prompt we
            # leave the methodology unset and say so; set_methodology and
            # --methodology both remain available.
            # Same for an MCP host with no elicitation (RelayGate): nobody
            # here can be asked mid-load, so the agent is told to ask.
            from ledgerhound.analyst.hitl import auto_yes
            gate = self.registry.gate
            if (getattr(gate, "prompt", None) is auto_yes
                    or getattr(gate, "can_ask_person", True) is False):
                self._events.append("methodology_set", template=None,
                                    chosen_by="not_asked_non_interactive")
                return ("[METHODOLOGY] none adopted -- no interactive prompt "
                        "is available to choose one, and picking for you "
                        "would be a guess. Use --methodology, or ask the "
                        "user which one they want and call set_methodology, "
                        "to work a declared phase sequence.")

            options, by_label = [], {}
            for n in names:
                t = load_template(n)
                if t is None:
                    continue
                label = f"{n} ({len(t.phases)} phases, " \
                        f"{sum(len(p.steps) for p in t.phases)} steps)"
                options.append(label)
                by_label[label] = n
            options.append("no methodology -- free-form analysis")
            question = (
                "Which analysis methodology should this session follow?\n"
                "Each is an ordered phase sequence with its own EDA step "
                "checklist; it drives what each phase is allowed to "
                "produce. Analysts differ here, so this is your call:\n"
                "  - topdown_7phase: fraud-only EDA (both grains) -> "
                "non-fraud -> contrast -> aggregation plan -> features -> "
                "selection -> modelling\n"
                "  - sample_first_7phase: same, but phase 1 decides how "
                "non-fraud is sampled")
            answer = self.registry.gate.choose(question, options).answer
            name = by_label.get(answer)
            if not name:
                self._events.append("methodology_set", chosen_by="analyst",
                                    template=None)
                return ("[METHODOLOGY] none -- free-form analysis. Adopt one "
                        "later with set_methodology if you change your mind.")
            m = load_template(name)
            if m is None:
                return ""
            self.methodology = m
            if m.phases:
                m.phases[0].status = "active"
            self._events.append("methodology_set", template=name,
                                chosen_by="analyst")
            self._save_methodology()
            self._refresh_schema_pin()
            return f"[METHODOLOGY] {m.name}\n\n{m.pin_block()}"
        except Exception:  # noqa: BLE001 -- never break a data load
            return ""

    def _save_methodology(self) -> None:
        if self.methodology is None:
            return
        try:
            (self.run_dir / "methodology.json").write_text(
                json.dumps(self.methodology.to_dict(), indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001 -- persistence is best-effort
            pass

    def _tool_set_methodology(self, template: str = "", mode: str = "") -> str:
        from ledgerhound.methodology import available_templates, load_template
        names = available_templates()
        if not template:
            lines = ["available methodologies:"]
            for n in names:
                m = load_template(n)
                if m:
                    steps = sum(len(p.steps) for p in m.phases)
                    lines.append(f"  - {n}: {m.name} "
                                 f"({len(m.phases)} phases, {steps} steps)")
                    lines.append(f"      {' '.join(m.description.split())[:220]}")
            lines.append("Call set_methodology(template='...') to adopt one. "
                        "Optionally add mode='eda_only'|'fe_only'|'eda_fe_only' "
                        "-- a separate axis controlling how far the pipeline "
                        "goes, not the reasoning order.")
            return "\n".join(lines)
        m = load_template(template, mode=mode)
        if m is None:
            return (f"no methodology named {template!r}. Available: "
                    f"{', '.join(names)}")
        # NOTE: load_template already set the correct active phase (index 0
        # normally; the first feature/model phase when mode='fe_only', via
        # Methodology.apply_mode_start()) -- do not override it here.
        self.methodology = m
        self._events.append("methodology_set", template=template, mode=m.mode,
                            phases=len(m.phases),
                            steps=sum(len(p.steps) for p in m.phases))
        self._save_methodology()
        self._refresh_schema_pin()
        mode_note = f" [mode: {m.mode}]" if m.mode else ""
        return (f"adopted '{m.name}'{mode_note} -- {len(m.phases)} phases, "
                f"{sum(len(p.steps) for p in m.phases)} steps.\n\n"
                + m.pin_block())

    def _tool_complete_step(self, step_id: str, note: str = "",
                            skipped: bool = False) -> str:
        if self.methodology is None:
            return ("no methodology set -- call set_methodology first, or "
                    "work without one.")
        ph = self.methodology.current_phase()
        if ph is None:
            return "methodology has no phases."
        for s in ph.steps:
            if s.id == step_id:
                s.status = "skipped" if skipped else "done"
                s.note = note or s.note
                self._events.append("step_complete", phase=ph.id,
                                    step=step_id, skipped=bool(skipped),
                                    note=note[:1000])
                self._save_methodology()
                self._refresh_schema_pin()
                done, total = ph.progress()
                left = len(ph.pending_steps())
                tail = (f" {left} step(s) left in this phase."
                        if left else
                        " All steps done -- call advance_phase when ready.")
                return (f"{'skipped' if skipped else 'completed'} {step_id} "
                        f"[{done}/{total}].{tail}")
        ids = ", ".join(s.id for s in ph.steps[:12])
        return (f"no step {step_id!r} in phase '{ph.id}'. Steps here: {ids}"
                f"{' ...' if len(ph.steps) > 12 else ''}")

    def _tool_add_step(self, text: str, grain: str = "any") -> str:
        from ledgerhound.methodology.phases import Step
        if self.methodology is None:
            return "no methodology set -- call set_methodology first."
        ph = self.methodology.current_phase()
        if ph is None:
            return "methodology has no phases."
        n = sum(1 for s in ph.steps if s.source == "analyst") + 1
        sid = f"{ph.id.split('_')[0]}.x{n:02d}"
        ph.steps.append(Step(id=sid, text=text, grain=grain or "any",
                             source="analyst"))
        self._events.append("step_added", phase=ph.id, step=sid, text=text[:500])
        self._save_methodology()
        self._refresh_schema_pin()
        return f"added {sid} to phase '{ph.id}': {text}"

    def _tool_advance_phase(self, force: bool = False) -> str:
        if self.methodology is None:
            return "no methodology set -- call set_methodology first."

        # V3-045: a phase does not close without an artifact.
        #
        # Observed 2026-08-11: a run's output folder began at phase 3 --
        # phases 1 and 2 had run and left nothing saved. Advisory wording
        # cannot fix that; an agent under completion pressure advances. So
        # this REFUSES, consistent with the standing rule that governance
        # refuses rather than warns where a deliverable is at stake.
        #
        # Deliberately does NOT auto-write a stub on force. A file for a
        # phase whose report nobody wrote is worse than no file: it makes
        # the folder listing claim coverage that does not exist, which is
        # the exact illusion being fixed here. Forcing is legitimate and
        # recorded; the missing report is then the honest artifact.
        # Order matters: the methodology's own pending-steps refusal comes
        # first. Doing the steps precedes reporting on them, so telling an
        # analyst "no report saved" while six steps are still open points at
        # the wrong problem.
        ph = self.methodology.current_phase()
        if ph is not None and not force and ph.pending_steps():
            ok, msg = self.methodology.advance(force=False)   # refuses, no-op
            self._events.append("phase_advance", ok=ok, forced=False,
                                detail=msg[:1000])
            return msg

        if ph is not None:
            from ledgerhound.shell.guardrails import PhaseReportGate
            refusal = PhaseReportGate.check(ph.id, self._reports_by_phase, bool(force))
            if refusal:
                self._events.append("phase_advance_refused",
                                    phase=ph.id, reason="no_report_saved")
                return refusal
        if ph is not None and force and not self._reports_by_phase.get(ph.id):
            self._events.append("phase_closed_without_report", phase=ph.id)

        ok, msg = self.methodology.advance(force=bool(force))
        self._events.append("phase_advance", ok=ok, forced=bool(force),
                            detail=msg[:1000])
        self._save_methodology()
        self._refresh_schema_pin()
        if not ok:
            return msg
        return msg + "\n\n" + self.methodology.pin_block()

    def _tool_phase_status(self) -> str:
        if self.methodology is None:
            return ("no methodology set. set_methodology() lists the "
                    "available ones.")
        s = self.methodology.summary()
        lines = [f"methodology: {s['methodology']}",
                 f"phase {s['phase_index']}/{s['phase_count']}: {s['phase']}"]
        for p in s["phases"]:
            mark = {"done": "x", "active": ">", "pending": " "}.get(p["status"], " ")
            lines.append(f"  [{mark}] {p['id']:22s} {p['done']}/{p['total']} steps")
        return "\n".join(lines) + "\n\n" + self.methodology.pin_block()

    def _tool_set_goal(self, objective: str, unit_of_analysis: str = "",
                       entity_key: str = "", entity_key_secondary: str = "",
                       label_column: str = "",
                       notes: str = "") -> str:
        """Declare what the session is for. Pinned every turn and enforced."""
        from ledgerhound.shell.session_goal import GRAIN_ACCOUNT, normalise_grain
        g = self.goal
        if not g.origin_instruction and self._first_user_turn_text:
            g.origin_instruction = self._first_user_turn_text
        g.objective = objective or g.objective
        if unit_of_analysis:
            g.unit_of_analysis = normalise_grain(unit_of_analysis)
        if entity_key:
            g.entity_key = entity_key
        if entity_key_secondary:
            g.entity_key_secondary = entity_key_secondary
        if label_column:
            g.label_column = label_column
        if notes:
            g.notes = notes
        self._events.append("goal_set", **g.to_dict())
        g.save(self.run_dir / "session_goal.json")
        self._refresh_schema_pin()
        warn = ""
        if g.unit_of_analysis == GRAIN_ACCOUNT and not g.entity_key:
            warn = ("\nWARNING: unit_of_analysis is 'account' but no "
                    "entity_key was given -- the grain gate can only enforce "
                    "one-row-per-entity once you set it. Call set_goal again "
                    "with entity_key.")
        return f"goal recorded and pinned.\n\n{g.pin_block()}{warn}"

    def _tool_request_modeling_authorisation(self, summary: str) -> str:
        """Ask a human to move the session from analysis into modelling.
        Nothing else can flip this -- the model cannot authorise itself."""
        q = ("The analyst wants to move from analysis/feature work into MODEL "
             f"TRAINING.\n\nIts summary: {summary}\n\nAuthorise modelling?")
        opts = ["authorise modelling",
                "no -- keep doing feature/analysis work"]
        decision = self.registry.gate.choose(q, opts)
        approved = str(decision.answer).startswith("authorise")
        by = getattr(decision, "approved_by", "person")
        self.goal.modeling_authorised = approved
        if approved:
            self.goal.phase = "model"
        self._events.append("goal_gate", kind="modeling_authorisation",
                            approved=approved, approved_by=by,
                            summary=summary[:1000])
        self.goal.save(self.run_dir / "session_goal.json")
        self._refresh_schema_pin()
        if approved and by == "agent":
            return ("modelling AUTHORISED as relayed by the agent -- fit/train "
                    "calls are now allowed. No person was verified to have "
                    "approved this; the report will say so.")
        if approved:
            return ("modelling AUTHORISED by the analyst -- fit/train calls "
                    "are now allowed.")
        return ("modelling NOT authorised. Continue the analysis / feature "
                "work; do not attempt to train a model.")
