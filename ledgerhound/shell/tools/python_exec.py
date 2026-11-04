"""
ledgerhound.shell.tools.python_exec
===================================

run_python: code gates, exec/kernel execution and post-execution checks.
"""

from __future__ import annotations

import difflib
import io
import os
import re
from contextlib import redirect_stdout

import pandas as pd

from ledgerhound.shell.tools.common import _parse_traceback_signature



class PythonToolsMixin:
    """run_python: code gates, exec/kernel execution and post-execution checks."""

    def _tool_run_python(self, code: str) -> str:
        out = self._run_python_step(code)
        note = self._sync_outputs()
        out = f"{out}\n{note}" if note else out
        if self._open_data_questions and not self._questions_reminded:
            self._questions_reminded = True
            keys = ", ".join(q.key for q in self._open_data_questions[:8])
            out += ("\n[DATA QUESTIONS STILL OPEN: " + keys + " -- ask the user and "
                    "record_data_answer before drawing any conclusion that depends on them. "
                    "Do not assume.]")
        return out

    def _run_python_step(self, code: str) -> str:
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        # GOAL GATES (blocking, not advisory). The 2026-08-06 production run
        # showed an advisory grain warning being acknowledged in a code
        # comment and then ignored ("This is intentional"), and hours spent
        # training models that were never authorised. These REFUSE. See
        # ledgerhound.shell.guardrails for each gate.
        for gate in self._code_gates:
            refusal = gate.check(code)
            if refusal:
                self._events.append("goal_gate", kind=gate.kind, code=code[:500])
                return refusal
        if self._kernel_refusal:
            return self._kernel_refusal
        if self._kernel_active and self._kernel is not None:
            return self._run_python_kernel(code)
        if self.isolation is not None and self.isolation.mode == "require":
            return ("REFUSED: isolation is required (--require-isolation) but no "
                    "isolated kernel is running, and code never runs in-process "
                    "under that flag. Install the kernel dependencies "
                    "(pip install 'ledgerhound[mcp]') and reload the data.")
        refusal = self._exec_blocklist.check(code)
        if refusal:
            return refusal
        if self._ns is None:
            import numpy as np

            from ledgerhound.shell.feature_strength import (
                make_evaluate_feature_helper,
                make_evaluate_features_helper,
                make_percentile_table_helper,
                make_separation_callout_helper,
            )
            from ledgerhound.shell.leakage_checks import make_check_leakage_helper
            # V3-049: exec mode had NO chdir and no scratch/features_dir of
            # its own, so any relative df.to_csv(...)/plt.savefig(...) in
            # generated code landed wherever the process happened to be
            # launched from -- confirmed on disk as a stray `features.csv`
            # at the repo root. scratch/features_dir give the model
            # somewhere explicit to write; the chdir bracket around exec()
            # below (not here) makes relative paths land there too.
            scratch_dir = self.run_dir / "scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            from ledgerhound.shell.write_guard import link_features_into
            link_features_into(scratch_dir, self._features_dir())
            self._ns = {"pd": pd, "np": np, "df": self._df,
                       "scratch": scratch_dir,
                       "features_dir": self._features_dir(),
                       # V3-050: the GPU-awareness section of system.md is
                       # unconditional and tells the model to use `_xdf`/
                       # `to_pdf`/`safe_*` -- names that only really exist
                       # in kernel mode. In exec mode (no kernel) those were
                       # a guaranteed NameError. `_xdf` is pandas here (this
                       # IS the CPU path) and the safe helpers are no-ops/
                       # identity, so the same generated code works whether
                       # or not this session has a GPU kernel behind it.
                       "_xdf": pd,
                       "to_pdf": lambda x: x,
                       "safe_to_datetime": pd.to_datetime,
                       "safe_merge": pd.merge,
                       "safe_to_parquet": lambda d, path, **kw: d.to_parquet(path, **kw),
                       "check_leakage": make_check_leakage_helper(),
                       "evaluate_feature": make_evaluate_feature_helper(),
                       "evaluate_features": make_evaluate_features_helper(),
                       "percentile_table": make_percentile_table_helper(),
                       "separation_callout": make_separation_callout_helper()}
            # cached_agg / sample_frame -- same toolkit the kernel gets, so
            # the two execution paths expose the same helper surface.
            try:
                from ledgerhound.shell.workbench import WORKBENCH_SNIPPET
                exec(WORKBENCH_SNIPPET, self._ns)
            except Exception:  # noqa: BLE001 -- helpers are a bonus, not a gate
                pass
        buf = io.StringIO()
        # V3-049: bracket exec() with a chdir into this run's scratch dir so
        # a bare relative `df.to_csv("out.csv")` lands under the run folder
        # instead of wherever the shell process happened to be launched
        # from. Single-threaded REPL, one cell at a time -- safe to mutate
        # process CWD for the duration of one exec() call.
        cwd_before = os.getcwd()
        scratch_dir = self._ns.get("scratch") or (self.run_dir / "scratch")
        features_dir_ns = self._ns.get("features_dir") or scratch_dir
        try:
            os.chdir(scratch_dir)
        except OSError:
            pass
        # V3-063: chdir only protects a BARE RELATIVE write -- an absolute
        # path (or one built from the loaded data file's own directory,
        # e.g. os.path.join(os.path.dirname(data_path), "out.csv")) sails
        # straight through it. Confirmed in production: feature tables
        # landing in the ORIGINAL DATA FOLDER. Bracket per-cell, not
        # installed for the process lifetime -- exec mode shares this
        # interpreter with the rest of Ledgerhound, so a global monkeypatch
        # left in place would affect the host app's own pandas writes too.
        from ledgerhound.shell.write_guard import install as _install_write_guard

        def _wg_notice(orig: str, redirected: str) -> None:
            print(f"WRITE REDIRECTED: '{orig}' is outside the workspace "
                 f"-> '{redirected}'")
        _uninstall_wg = _install_write_guard(
            lambda: [scratch_dir, features_dir_ns],
            lambda: features_dir_ns, _wg_notice)
        try:
            try:
                with redirect_stdout(buf):
                    exec(code, self._ns)  # single namespace -> vars persist, funcs see pd/np
            except Exception as e:  # noqa: BLE001
                self._events.append("code_exec", code=code, executor="exec", success=False,
                                    error=f"{type(e).__name__}: {e}")
                return self._teach_error(e, buf.getvalue())
        finally:
            _uninstall_wg()
            try:
                os.chdir(cwd_before)
            except OSError:
                pass
        out = buf.getvalue().strip()
        res = self._ns.get("result")
        if not out and res is not None:
            out = str(res)
        # V3-017: `df` may have been reassigned in the sandbox namespace
        # (e.g. `df = df.dropna()`) -- resync self._df so the schema pin and
        # error-hint columns don't go stale, and check for a mutation delta
        # to surface to the model. Cheap (rows+columns) always; a null-rate
        # scan only fires when that already showed a real change.
        new_df = self._ns.get("df")
        delta_block = ""
        if new_df is not None and hasattr(new_df, "columns"):
            self._df = new_df
            delta_block = self._check_mutation_delta(new_df, code)
        # Data-comparability check on whatever frames the cell actually
        # created. Runs here, not at load_data, because code that reads its
        # own files (pd.read_csv inside run_python) never touches load_data
        # -- which is exactly how a 1-month-vs-1-day fraud/non-fraud pair
        # reached feature engineering unflagged.
        check_block = self._check_data_comparability(code)
        leakage_block = self._check_leakage(code)
        grain_block = self._check_output_grain(code)
        numeric_block = self._check_numeric_stability(code)
        ambiguity_block = self._check_ambiguous_columns()
        memory_block = self._check_memory_usage()
        # V3-012: session_code.py is now a projection of the event stream --
        # append the canonical code_exec event, then write the SAME view
        # function `ledgerhound run-views` uses for bulk regeneration, so the
        # live file and a regenerated one can never drift apart.
        self._log_code_exec(code, out, executor="exec")
        # V5-005: same 3500-char budget as before, but cut from the MIDDLE
        # instead of the tail. A bare out[:3500] drops the end of the output,
        # which on a DataFrame repr is where the shape summary lives
        # ("[1000 rows x 40 columns]") -- the single most useful line, and
        # the one the model needs to know it is looking at a truncated view.
        # The full text spills to the run folder either way.
        text = self._compact_cell_output(out)
        for block in (delta_block, check_block, leakage_block, grain_block,
                     numeric_block, ambiguity_block, memory_block,
                     self._stat_gate.advisory):
            if block:
                text = f"{text}\n\n{block}" if text else block
        return text if text.strip() else "(ran; no output -- print() what you want to see)"

    # Cell-output budget. Unchanged from the bare out[:3500] this replaced --
    # what changed is WHERE the cut falls.
    _CELL_OUTPUT_CHARS = 3500

    def _compact_cell_output(self, out: str) -> str:
        """V5-005: hold a cell's output to the budget without losing its end."""
        if not out:
            return ""
        try:
            from ledgerhound.shell.intra_compaction import compact_result
            outcome = compact_result(
                out, spill_dir=self.run_dir / "results",
                max_chars=self._CELL_OUTPUT_CHARS,
                head_chars=int(self._CELL_OUTPUT_CHARS * 0.68),
                tail_chars=int(self._CELL_OUTPUT_CHARS * 0.32))
            if outcome.compacted:
                self._events.append("intra_compaction", tool="run_python",
                                    original_chars=outcome.original_chars,
                                    saved_chars=outcome.saved_chars,
                                    spill_path=outcome.spill_path)
            return outcome.text
        except Exception:  # noqa: BLE001
            return out[:self._CELL_OUTPUT_CHARS]

    def _true_frame_rows(self) -> int | None:
        """Row count of the loaded frame, or None when only a schema sample
        is held (the real data lives in the kernel)."""
        if self._df is None or self._df_is_sample:
            return None
        try:
            return int(len(self._df))
        except Exception:  # noqa: BLE001
            return None

    def _check_data_comparability(self, code: str = "") -> str:
        """Mechanical time-window comparability check over the sandbox's
        frames (see ledgerhound.shell.data_checks). Best-effort: a failure here
        must never break a cell that otherwise ran fine. `code` lets
        unchanged frames skip their full-column scans (J-02)."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.data_checks import check_frames
            block = check_frames(self._ns, self._data_issues_seen, code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="time_window_mismatch",
                                detail=block[:2000])
        return block

    def _check_leakage(self, code: str = "") -> str:
        """Mechanical leakage check over the sandbox's frames (V3-032, see
        ledgerhound.shell.leakage_checks). Best-effort: a failure here must
        never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.leakage_checks import check_leakage
            block = check_leakage(self._ns, self._leakage_issues_seen, code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="target_leakage",
                                detail=block[:2000])
        return block

    def _check_output_grain(self, code: str) -> str:
        """Mechanical output-grain check, gated on an export call in `code`
        (V3-032 Tier 4, see ledgerhound.shell.grain_checks). Best-effort: a
        failure here must never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.grain_checks import check_output_grain
            block = check_output_grain(code, self._ns, self._grain_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="output_grain",
                                detail=block[:2000])
        return block

    def _check_numeric_stability(self, code: str = "") -> str:
        """Mechanical inf/-inf check over the sandbox's frames (V3-032, see
        ledgerhound.shell.numeric_checks). Best-effort: a failure here must
        never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.numeric_checks import check_numeric_stability
            block = check_numeric_stability(self._ns, self._numeric_issues_seen,
                                            code)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="numeric_stability",
                                detail=block[:2000])
        return block

    def _check_ambiguous_columns(self) -> str:
        """Mechanical operational/workflow-artifact-column check over the
        sandbox's frames (V3-032, see ledgerhound.shell.ambiguity_checks).
        Best-effort: a failure here must never break a cell that otherwise
        ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.ambiguity_checks import check_ambiguous_columns
            block = check_ambiguous_columns(self._ns, self._ambiguity_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="ambiguous_column",
                                detail=block[:2000])
        return block

    def _check_memory_usage(self) -> str:
        """Mechanical redundant-large-copy check over the sandbox's frames
        (V3-034, see ledgerhound.shell.memory_checks). Best-effort: a failure
        here must never break a cell that otherwise ran fine."""
        if self._ns is None:
            return ""
        try:
            from ledgerhound.shell.memory_checks import check_memory_usage
            block = check_memory_usage(self._ns, self._memory_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="redundant_memory_copy",
                                detail=block[:2000])
        return block

    def _check_mutation_delta(self, df, code: str) -> str:
        """V3-017: exec-mode delta check -- df is already in-process, so the
        null-rate scan (when needed) is a direct, synchronous computation."""
        from ledgerhound.shell.verification import fingerprint, fingerprints_differ, format_delta
        fp = fingerprint(len(df), list(df.columns))
        before = self._last_fp
        self._last_fp = fp
        if not fingerprints_differ(before, fp):
            return ""
        try:
            rates = {str(c): float(df[c].isna().mean()) for c in df.columns}
        except Exception:
            rates = {}
        prior_rates = self._last_null_rates
        self._last_null_rates = rates
        self._events.append("verification", code=code, before=before, after=fp,
                            null_rates=rates)
        return format_delta(before, fp, rates, prior_rates)

    def _log_code_exec(self, code: str, output: str, *, executor: str) -> None:
        """Record a successful code execution as a canonical event, and
        project it into session_code.py (see ledgerhound.runtime.views)."""
        # V5-002: the cheapest visible proxy for tests we cannot otherwise
        # see. Thirty cells against one recorded finding says most of what
        # was computed never reached the ledger -- which is exactly the
        # situation a multiplicity correction cannot account for on its own.
        try:
            self._effort.note_cell(getattr(self, "_current_chapter", ""))
        except Exception:  # noqa: BLE001
            pass
        import hashlib
        ev = self._events.append(
            "code_exec", code=code, executor=executor, success=True,
            output_preview=output[:2000],
            output_hash=hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
        )
        try:
            from ledgerhound.runtime.views import code_block
            with open(self._code_log, "a", encoding="utf-8") as f:
                f.write(code_block(ev))
        except Exception:
            pass

    def _run_python_kernel(self, code: str) -> str:
        """Execute in the real, persistent IPython kernel (V3-009) -- full
        Python, no `_BLOCKED` string-list (the kernel's own write-guard +
        process isolation cover that instead). Failures come back as a
        traceback string rather than a live exception, so we parse a
        (kind, msg) signature and reuse the same teaching-hint logic."""
        res = self._kernel.execute(code)
        if not res.success:
            kind, msg = _parse_traceback_signature(res.error or "")
            self._events.append("code_exec", code=code, executor="kernel", success=False,
                                error=f"{kind}: {msg}")
            return self._teach_error_text(kind, msg, res.output)
        out = (res.output or "").strip()
        # V3-017: the kernel's own execute() epilogue returns a cheap
        # (rows+columns) df_fingerprint on every call (see auto/kernel.py);
        # the null-rate scan is a SEPARATE, explicit kernel round-trip via
        # null_rates(), only paid for when that cheap check already showed
        # a change -- df itself never comes back to this process.
        delta_block = self._check_kernel_mutation_delta(res.df_fingerprint, code)
        memory_block = self._check_kernel_memory_usage(res.df_fingerprint)
        # The other four checks (leakage, numeric-stability, ambiguous-column,
        # output-grain) run against a compact JSON summary computed INSIDE the
        # kernel -- exec mode reads self._ns, which is empty here (audit
        # finding, 2026-08-06; see ledgerhound/shell/kernel_checks.py).
        kernel_blocks = self._run_kernel_checks(code)
        self._log_code_exec(code, out, executor="kernel")
        # V5-005: same 3500-char budget as before, but cut from the MIDDLE
        # instead of the tail. A bare out[:3500] drops the end of the output,
        # which on a DataFrame repr is where the shape summary lives
        # ("[1000 rows x 40 columns]") -- the single most useful line, and
        # the one the model needs to know it is looking at a truncated view.
        # The full text spills to the run folder either way.
        text = self._compact_cell_output(out)
        # V5-003's advisory is computed before dispatch, so it applies to both
        # execution paths -- a caveat that only appeared in exec mode would be
        # missing from exactly the GPU runs that matter most.
        for block in ([delta_block, memory_block] + kernel_blocks
                      + [self._stat_gate.advisory]):
            if block:
                text = f"{text}\n\n{block}" if text else block
        return text if text.strip() else "(ran; no output -- print() what you want to see)"

    def _export_frames(self) -> dict:
        """``{var: {"n_rows", "dup_keys": {entity_col: n}}}`` for the frames
        that exist right now -- what ExportGrainGate judges before a write.
        Kernel mode computes it inside the kernel; exec mode reads _ns."""
        frames: dict = {}
        if self._kernel_active and self._kernel is not None:
            from ledgerhound.shell.kernel_checks import collect_payload
            # force the (otherwise export-gated) duplicate scan
            return collect_payload(self._kernel, "to_csv(")
        if self._ns:
            from ledgerhound.shell.data_checks import _is_frame
            from ledgerhound.shell.grain_checks import _entity_key_col
            for k, v in self._ns.items():
                if str(k).startswith("_") or not _is_frame(v):
                    continue
                col = self.goal.entity_key or _entity_key_col(v)
                if not col or col not in getattr(v, "columns", []):
                    continue
                n = len(v)
                frames[str(k)] = {"n_rows": n,
                                  "dup_keys": {col: int(n - v[col].nunique())}}
        return frames

    def _run_kernel_checks(self, code: str) -> list[str]:
        """Kernel-mode counterpart to the four exec-mode checks that need
        real frame access. Best-effort: a failure here must never break a
        cell that otherwise ran fine."""
        if self._kernel is None:
            return []
        try:
            from ledgerhound.shell.kernel_checks import checks_from_payload, collect_payload
            payload = collect_payload(self._kernel, code)
            if not payload:
                return []
            seen = {"leakage": self._leakage_issues_seen,
                    "numeric": self._numeric_issues_seen,
                    "ambiguity": self._ambiguity_issues_seen,
                    "grain": self._grain_issues_seen}
            blocks = checks_from_payload(payload, code, seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return []
        for issue, block in blocks:
            self._events.append("data_check", issue=issue, detail=block[:2000])
        return [b for _, b in blocks]

    def _check_kernel_memory_usage(self, fp: dict) -> str:
        """Kernel-mode counterpart to _check_memory_usage (V3-034): `df`
        never comes back to this process in kernel mode, so this reads the
        `other_large_frames` {name: n_rows} map the kernel's own execute()
        epilogue already computes cheaply (auto/kernel.py) instead of
        needing a second round-trip. Best-effort: a failure here must never
        break a cell that otherwise ran fine."""
        if not fp:
            return ""
        try:
            from ledgerhound.shell.memory_checks import format_memory_block
            ref_len = int(fp.get("n_rows") or 0)
            pairs = fp.get("other_large_frames") or {}
            block = format_memory_block(pairs, ref_len, self._memory_issues_seen)
        except Exception:  # noqa: BLE001 -- advisory, never fatal
            return ""
        if block:
            self._events.append("data_check", issue="redundant_memory_copy",
                                detail=block[:2000])
        return block

    def _check_kernel_mutation_delta(self, fp: dict, code: str) -> str:
        from ledgerhound.shell.verification import fingerprints_differ, format_delta
        if not fp:
            return ""
        before = self._last_fp
        self._last_fp = fp
        if not fingerprints_differ(before, fp):
            return ""
        try:
            rates = self._kernel.null_rates()
        except Exception:
            rates = {}
        prior_rates = self._last_null_rates
        self._last_null_rates = rates
        self._events.append("verification", code=code, before=before, after=fp,
                            null_rates=rates)
        return format_delta(before, fp, rates, prior_rates)

    def _teach_error(self, e: Exception, stdout: str) -> str:
        """Turn a raw exception into a message that tells the model exactly
        what to try next, instead of a bare traceback it has to guess at.
        Local-exec path only (a real exception object); the kernel path
        (V3-009) only gets a traceback string, so it goes through
        `_teach_error_text` directly via a parsed (kind, msg) pair."""
        names = [str(k) for k in (self._ns or {}) if not k.startswith("_")]
        return self._teach_error_text(type(e).__name__, str(e), stdout, names)

    def _teach_error_text(self, kind: str, msg: str, stdout: str,
                          names: list[str] | None = None) -> str:
        """Fuzzy-matches missing column/name references against the live
        schema so a typo ('amnt') resolves to the real column ('amount') in
        one retry instead of a blind-guess loop. `names` is the list of live
        sandbox variables when known (exec mode); the kernel path (V3-009)
        has no cheap way to introspect kernel-side variables, so it passes
        an empty list -- the column-based hints still work either way."""
        names = names or []
        cols = [str(c) for c in self._df.columns] if self._df is not None else []

        hint = ""
        if kind == "KeyError":
            missing = msg.strip("'\"")
            close = difflib.get_close_matches(missing, cols, n=3, cutoff=0.5)
            if close:
                hint = f"column {missing!r} not found; closest matches: {close}. "
            hint += f"full columns ({len(cols)}): {cols[:40]}"
        elif kind == "NameError":
            m = re.search(r"name '(\w+)' is not defined", msg)
            if m:
                missing = m.group(1)
                close = difflib.get_close_matches(missing, names + cols, n=3, cutoff=0.5)
                if close:
                    hint = f"{missing!r} is not defined; closest matches: {close}. "
                hint += f"variables available: {names[:30]}"
        elif kind == "AttributeError":
            m = re.search(r"has no attribute '(\w+)'", msg)
            if m:
                missing = m.group(1)
                close = difflib.get_close_matches(missing, cols, n=3, cutoff=0.5)
                if close:
                    hint = (f"{missing!r} not found as an attribute; if you meant a "
                            f"column, use df[{close[0]!r}] -- closest matches: {close}")
        elif kind == "MemoryError":
            hint = ("out of memory -- work on a sample instead, e.g. "
                    "`sample = df.sample(n=100_000, random_state=0)`, and retry.")
        elif kind in ("ImportError", "ModuleNotFoundError"):
            hint = ("that library isn't available in this sandbox. Guaranteed "
                    "available: pandas (pd), numpy (np). Do not retry the same "
                    "import -- use a pandas/numpy-only approach instead.")

        out = f"Error: {kind}: {msg}"
        if hint:
            out += f"\n{hint}"
        tail = stdout[-1500:]
        if tail.strip():
            out += f"\n{tail}"
        return out
