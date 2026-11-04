"""
ledgerhound.shell.tools.reporting
=================================

Reports, understanding notes, notebooks and the feature catalog.
"""

from __future__ import annotations

import json





class ReportingToolsMixin:
    """Reports, understanding notes, notebooks and the feature catalog."""

    def _findings_since_last_save(self) -> list[str]:
        """Claims recorded since the previous save_report, oldest first.

        Reads the run's findings.jsonl (the projection record_finding
        already maintains) and computes how far it would advance -- but
        does NOT commit that advance yet. V3-052: this used to mark
        findings consumed here, unconditionally, before save_output ran. If
        the save then failed, the findings were already marked read and
        NO later report could ever include them again -- a report-time
        failure silently and permanently dropped evidence. The consumed
        marker is now only committed by `_commit_findings_consumed()`,
        called from `_tool_save_report` after a save (or its fallback)
        actually persisted the findings text.
        """
        seen = getattr(self, "_findings_consumed", 0)
        out: list[str] = []
        self._findings_consumed_pending = seen
        try:
            if self._findings_path.exists():
                lines = self._findings_path.read_text(
                    encoding="utf-8").splitlines()
                for line in lines[seen:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    claim = (rec.get("claim") or "").strip()
                    if not claim:
                        continue
                    ev = (rec.get("evidence") or "").strip()
                    frame = (rec.get("sample_frame") or "").strip()
                    bits = [claim]
                    if ev:
                        bits.append(f"evidence: {ev}")
                    if frame:
                        bits.append(f"frame: {frame}")
                    out.append(" — ".join(bits))
                self._findings_consumed_pending = len(lines)
        except Exception:  # noqa: BLE001 -- never break a save over this
            pass
        return out

    def _commit_findings_consumed(self) -> None:
        """Advance the real consumed marker to what `_findings_since_last_
        save` staged -- call ONLY after the findings it returned have
        actually been written somewhere durable (V3-052)."""
        pending = getattr(self, "_findings_consumed_pending", None)
        if pending is not None:
            self._findings_consumed = pending

    def _tool_save_report(self, title: str, markdown: str,
                          iteration: int = 0) -> str:  # noqa: ARG002
        # `iteration` is accepted and IGNORED (V3-045). It is off the tool
        # schema now; the parameter stays so an older caller or a replayed
        # transcript that still sends it does not blow up.
        fmt = "md"
        if self.project is not None:
            try:
                fmt = self.project.config().save_format or "md"
            except Exception:
                pass
        # Sequence number is derived here, never taken from the model
        # (V3-045). The `iteration` argument used to set it, which gave two
        # competing numberings -- the model's idea and the folder's -- and
        # made a gap in the sequence unreadable: you could not tell a skipped
        # save from a mis-numbered one.
        self._report_iteration = getattr(self, "_report_iteration", 0) + 1
        it = self._report_iteration
        stem = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in title.lower())[:44] or "report"

        # Put the PHASE in the filename when a methodology is active. The
        # operator's complaint (2026-08-11) was that a run "started from
        # saving phase 3" -- with `iteration_01_...` as the only name, a
        # missing phase 1 and 2 is invisible in the folder listing. With the
        # phase in the stem, `ls` shows the gap.
        phase_key = ""
        if self.methodology is not None:
            ph = self.methodology.current_phase()
            if ph is not None:
                phase_key = ph.id
                stem = (f"phase_{self.methodology.current + 1}_"
                        f"{ph.id}__{it:02d}_{stem}")
        if not phase_key:
            stem = f"iteration_{it:02d}_{stem}"

        # Findings recorded SINCE THE LAST SAVE. The report used to carry
        # only the model's prose: `ReportPayload` has had `findings` and
        # `figures` fields all along and `_tool_save_report` passed neither,
        # so every number the analyst had actually recorded via
        # record_finding was absent from the artifact. Scoped to new findings
        # rather than all of them, or phase 5's report restates phases 1-4.
        new_findings = self._findings_since_last_save()
        # Carry a short pointer to the previous iteration into the document
        # so the continuity is in the artifact itself, not just the folder.
        prev = getattr(self, "_report_history", [])
        if prev:
            markdown = (
                f"*Iteration {it}. Builds on: "
                + ", ".join(f"`{p}`" for p in prev[-3:])
                + ".*\n\n" + markdown)
        self._report_history = prev + [f"{stem}.{fmt}"]
        try:
            from ledgerhound.analyst.save import ReportPayload, save_output
            res = save_output(
                ReportPayload(title=title, body_md=markdown,
                              findings=new_findings),
                fmt, out_dir=self.run_dir, stem=stem)
            self._commit_findings_consumed()  # V3-052: only after the write succeeded
            self._export_notebook()  # V3-014: best-effort, never blocks save_report
            if phase_key:
                self._reports_by_phase.setdefault(phase_key, []).append(
                    f"{stem}.{fmt}")
            # Say how much evidence went in. A "0 findings" report is the
            # signal that the analyst wrote prose without recording anything
            # -- visible to the model in the tool result, not buried.
            # J-07: a renderer that fell back (pdf without fpdf -> md) says
            # so in res.note; without it the model reported "saved pdf"
            # for a markdown file.
            fallback = (f" NOTE: {res.note}" if getattr(res, "note", "")
                        and not getattr(res, "ok", True) else "")
            return (f"saved {res.format} -> {res.path} "
                    f"({len(new_findings)} finding(s) recorded since the "
                    f"last save){fallback}")
        except Exception as e:  # noqa: BLE001
            # Do NOT silently claim success on a different format than asked
            # for -- surface the real error, then fall back to md so the
            # content is never lost, and say plainly that the fallback fired.
            # V3-052: the fallback used to write ONLY the model's prose,
            # with no findings section -- and the findings were about to be
            # marked consumed regardless, so they would vanish from every
            # future report too. The fallback now carries the same findings
            # text the primary save would have, and the consumed marker is
            # only advanced once this fallback write actually lands.
            try:
                findings_md = ""
                if new_findings:
                    findings_md = "\n\n## Findings\n\n" + "\n".join(
                        f"- {f}" for f in new_findings)
                path = self.run_dir / f"{stem}.md"
                path.write_text(f"# {title}\n\n{markdown}{findings_md}\n",
                                encoding="utf-8")
                self._commit_findings_consumed()
                return (f"'{fmt}' save failed ({type(e).__name__}: {e}); "
                        f"fell back to md -> {path} "
                        f"({len(new_findings)} finding(s) included)")
            except Exception as e2:  # noqa: BLE001
                # Total failure: do NOT commit -- these findings must remain
                # unconsumed so the NEXT save_report attempt still carries
                # them, instead of being silently lost forever.
                return (f"save FAILED entirely -- {fmt} error: "
                        f"{type(e).__name__}: {e}; md fallback also failed: "
                        f"{type(e2).__name__}: {e2}. Run folder: {self.run_dir}. "
                        f"{len(new_findings)} finding(s) NOT lost -- they "
                        f"remain unconsumed for the next save_report.")

    def _export_notebook(self) -> None:
        """V3-014: emit analysis.ipynb from the event stream on save_report.
        Best-effort -- a notebook-export failure must never break the
        report save it rides along with."""
        try:
            from ledgerhound.runtime.notebook_export import export_notebook
            export_notebook(self.run_dir)
        except Exception:
            pass

    def _tool_plan_features(self, entity_key: str = "", time_column: str = "",
                            max_features: int = 1000, windows: list = None) -> str:
        """Enumerate candidate entity-level features from the live schema.

        Deliberately returns CANDIDATES plus the code to compute them, not
        computed values: at crore scale the analyst must batch the work,
        and the vetting gate -- not this generator -- decides what is
        actually recommended."""
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        from ledgerhound.analysis.feature_factory import (
            emit_code,
            generate,
            infer_roles,
            summarise,
        )
        ek = entity_key or self.goal.entity_key
        if not ek:
            return ("no entity_key: pass one, or declare it with set_goal "
                    "(unit_of_analysis='account', entity_key='...'). Feature "
                    "generation needs to know what one row means.")
        cols = [str(c) for c in self._df.columns]
        if ek not in cols:
            return f"entity_key {ek!r} is not a column. Available: {cols[:40]}"
        dtypes = {str(c): str(t) for c, t in self._df.dtypes.items()}
        # Cardinality from whatever frame we hold. When that is only a schema
        # sample the ROLE split (identifier vs category) is an estimate --
        # say so rather than letting it read as measured.
        try:
            nun = {c: int(self._df[c].nunique()) for c in cols}
        except Exception:
            nun = {}
        tcol = time_column or next(
            (c for c, t in dtypes.items() if "date" in t.lower()
             or "time" in t.lower()), "")
        roles = infer_roles(dtypes, entity_key=ek,
                            label_column=self.goal.label_column,
                            nunique=nun, n_rows=len(self._df))
        specs = generate(roles, ek, time_column=tcol,
                         windows=tuple(windows) if windows else (1, 7, 30),
                         max_features=int(max_features or 1000))
        self._feature_specs = [s.to_dict() for s in specs]
        try:
            (self.run_dir / "feature_candidates.json").write_text(
                json.dumps(self._feature_specs, indent=2), encoding="utf-8")
        except Exception:
            pass
        code = emit_code(specs, ek, time_column=tcol)
        counts = summarise(specs)
        sample = ", ".join(s.name for s in specs[:12])
        note = ""
        if self._df_is_sample:
            note = ("\nNOTE: column roles were inferred from the schema "
                    "sample, so identifier-vs-category splits are estimates; "
                    "verify a couple before trusting the plan.\n")
        return (
            f"planned {counts['total']} candidate features at grain '{ek}'"
            f"{f' (time column: {tcol})' if tcol else ' (no time column -- no velocity features)'}.\n"
            f"by kind: {counts}\n"
            f"first few: {sample} ...\n"
            f"full list saved to feature_candidates.json\n{note}\n"
            "These are CANDIDATES. Compute them, then rank and prune with\n"
            "evaluate_features(feat, [...], target) and check_leakage(...)\n"
            "before recommending ANY of them -- real pilot feedback was that\n"
            "of a large proposed set only 6 of the top 50 mattered, so the\n"
            "deliverable must be the survivors, not this list.\n\n"
            "Starter code (batch it if the frame is large):\n"
            f"```python\n{code}\n```")

    def _tool_export_feature_catalog(self, evaluations: list = None,
                                     vetoed: list = None,
                                     redundant: list = None,
                                     iv_core: float = None,
                                     iv_supporting: float = None,
                                     title: str = "",
                                     percentiles: dict = None) -> str:
        """Write the documented, TIERED feature catalog to the run folder.

        Tiering rather than pruning: IV is univariate, so a feature that
        looks modest alone can still contribute through interactions in a
        GBM. Only leakage-vetoed, redundant, or genuinely signal-free
        features are dropped -- and even those stay in the catalog with
        their reason, so an excluded feature is explainable."""
        specs = getattr(self, "_feature_specs", None)
        if not specs:
            return ("no feature plan in this session -- call plan_features "
                    "first, then compute and evaluate them.")
        from ledgerhound.analysis.feature_factory import (
            IV_CORE,
            IV_SUPPORTING,
            build_catalog,
            build_dual_node_catalog,
            catalog_to_markdown,
            tier_features,
            tier_summary,
        )
        ek = self.goal.entity_key or "entity"
        ek2 = getattr(self.goal, "entity_key_secondary", "")
        if ek2:
            # V3-053: the goal declared BOTH sides of the payment -- build
            # every spec against both grains rather than defaulting to the
            # payer/remitter side alone (the testing-team-reported gap).
            rows = build_dual_node_catalog(specs, ek, ek2,
                                           evaluations=evaluations or [],
                                           percentiles=percentiles)
        else:
            rows = build_catalog(specs, ek, evaluations=evaluations or [],
                                 percentiles=percentiles)
        rows = tier_features(
            rows,
            iv_core=float(iv_core) if iv_core is not None else IV_CORE,
            iv_supporting=(float(iv_supporting) if iv_supporting is not None
                           else IV_SUPPORTING),
            vetoed=set(vetoed or ()), redundant=set(redundant or ()))
        tiers = tier_summary(rows)
        md = catalog_to_markdown(rows, ek, title=title or "Feature catalog")
        wrote = []
        try:
            import csv
            csv_path = self.run_dir / "feature_catalog.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            wrote.append(csv_path.name)
        except Exception as e:  # noqa: BLE001
            return f"catalog CSV failed: {type(e).__name__}: {e}"
        try:
            md_path = self.run_dir / "feature_catalog.md"
            md_path.write_text(md, encoding="utf-8")
            wrote.append(md_path.name)
        except Exception:
            pass
        n_meas = sum(1 for r in rows if r.get("measured"))
        n_sel = sum(1 for r in rows if r.get("selected"))
        self._events.append("feature_catalog", candidates=len(rows),
                            measured=n_meas, kept=n_sel, tiers=tiers)
        warn = ""
        if n_meas == 0:
            warn = ("\nWARNING: none of these are measured yet. Compute the "
                    "features and run evaluate_features(...), then call this "
                    "again -- an unvetted list is exactly what the pilot "
                    "rejected.")
        elif ek2:
            # K-04 follow-up: say which side is unmeasured, and how to fix it.
            n_b = sum(1 for r in rows if r.get("node") == "beneficiary"
                      and not r.get("measured"))
            if n_b:
                warn = (f"\nNOTE: {n_b} beneficiary-side feature(s) are "
                        "unmeasured. Plain-named evaluations apply to the "
                        "remitter side only; compute the features grouped on "
                        f"'{ek2}' and pass their evaluations as "
                        "'<name>__beneficiary'.")
        return (f"wrote {', '.join(wrote)} -- {len(rows)} candidates, "
                f"{n_meas} measured, {n_sel} kept.\ntiers: {tiers}\n"
                f"Each row carries: description, source columns, computed_at/"
                f"fit_scope, the exact reproducing code, IV/KS/AUC, its tier "
                f"and WHY it landed there.\n"
                f"Nothing was deleted -- 'dropped' rows stay in the catalog "
                f"with their reason. Adjust iv_supporting (default 0.02) if "
                f"you want to keep even more.{warn}")

    def _tool_save_understanding(self, title: str, summary: str,
                                 goal: str = "") -> str:
        try:
            from ledgerhound.analyst.books import Book, BookShelf
            findings = []
            if self._findings_path.exists():
                for line in self._findings_path.read_text(encoding="utf-8").splitlines():
                    try:
                        findings.append(json.loads(line).get("claim", "")[:120])
                    except Exception:
                        continue
            book = Book(title=title, summary=summary, goal=goal,
                        data_ref=self._loaded_path, finding_ids=findings,
                        report_path=str(self.run_dir))
            if self.project is not None:
                path = BookShelf(self.project).save(book)
                kb = self.team_kb()
                if kb is not None and self._dataset_id:
                    try:
                        kb.publish_book(self._dataset_id, book.to_dict())
                    except Exception as e:  # noqa: BLE001
                        self._events.append("team_kb_error", op="publish_book",
                                            error=str(e)[:300])
            else:  # no project -> keep it in the run folder
                path = self.run_dir / "book.md"
                path.write_text(book.to_markdown(), encoding="utf-8")
            return f"saved understanding to the books -> {path}"
        except Exception as e:  # noqa: BLE001
            return f"could not save the book: {e}"
