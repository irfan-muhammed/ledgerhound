"""
ledgerhound.shell.tools.data
============================

Finding, loading and profiling data; the kernel; dataset cards and the team KB.
"""

from __future__ import annotations

import json
from pathlib import Path


from ledgerhound.shell.tools.common import _DATA_EXTS, _SHELL_LOAD_MAX_ROWS, _SHELL_SCHEMA_SAMPLE_ROWS, _dataset_parts, _estimate_rows, _is_parquet_dataset_dir, _read_any, _read_sample, _sha256_dataset_dir



class DataToolsMixin:
    """Finding, loading and profiling data; the kernel; dataset cards and the team KB."""

    def _tool_list_directory(self, path: str = ".") -> str:
        p = self._resolve_dir(path)
        if p is None or not p.exists():
            return f"path not found: {path}"
        if p.is_file():
            return f"{p} is a file ({p.stat().st_size} bytes)"
        entries = sorted(p.iterdir())
        lines = [f"{p}:"]
        for e in entries[:60]:
            tag = "/" if e.is_dir() else f"  {e.stat().st_size}B"
            lines.append(f"  {e.name}{tag}")
        return "\n".join(lines)

    def _tool_load_data(self, path: str) -> str:
        from ledgerhound.io import data_source as _ds
        fetched = None
        if _ds.is_s3(path):
            # Object store: one parallel bulk copy into the cache (RAM disk
            # on a pod), then everything below reads it like a local file.
            try:
                fetched = self._fetch_from_store(_ds.parse(path))
            except _ds.DataSourceError as e:
                self._events.append("load_data_failed", path=path, error=e.message)
                return f"could not load {path}: {e}"
            resolved = fetched.local_path
        else:
            resolved = self._resolve_data(path)
        if isinstance(resolved, list):  # a folder with choices
            if len(resolved) == 1:
                resolved = resolved[0]
            else:
                listing = "\n".join(f"  - {r}" for r in resolved[:40])
                return (f"'{path}' is a folder with multiple data files; "
                        f"which one?\n{listing}")
        if resolved is None:
            return (f"could not find data at '{path}'. Tried it as-given, "
                    f"./, ../, cwd and project root. Use list_directory to look around.")
        # In kernel mode the kernel loads the real data itself (cuDF on GPU).
        # Reading the WHOLE file into this process too was a pure duplicate:
        # full CPU RAM cost, paid unconditionally, BEFORE the compute router
        # was even consulted -- so "use the GPU" never actually avoided it,
        # and at 8-crore scale it risks OOM-ing the shell before any analysis
        # starts (audit finding, 2026-08-06). Above the threshold we read a
        # bounded SCHEMA sample only; the kernel still owns the full data.
        df = None
        sampled_from = None
        rows_exact = True
        if self.use_kernel:
            est, est_exact = _estimate_rows(resolved)
            if est is not None and est > _SHELL_LOAD_MAX_ROWS:
                df = _read_sample(resolved, _SHELL_SCHEMA_SAMPLE_ROWS)
                if df is not None:
                    sampled_from = est
                    rows_exact = est_exact
        if df is None:
            try:
                df = _read_any(resolved)
            except Exception as e:  # noqa: BLE001
                return f"found {resolved} but failed to load it: {e}"
        self._df = df
        # True row count for the schema pin / event log, so a sampled load
        # never misreports the dataset's real size.
        self._df_is_sample = sampled_from is not None
        self._true_n_rows = int(sampled_from) if sampled_from else len(df)
        self._true_rows_exact = rows_exact
        # V3-017: fresh baseline for the post-mutation delta check -- a new
        # load_data always starts a new "before" state, never diffs across
        # datasets.
        from ledgerhound.shell.verification import fingerprint as _fp
        self._last_fp = _fp(len(df), list(df.columns))
        self._last_null_rates = None
        self._undo_snapshot = None   # V3-016: fresh data -- no undo history
        # V3-012/V3-013: fingerprint the data file so `ledgerhound replay` can
        # detect if it changed since this session ran.
        try:
            if fetched is not None:
                # From the store's ETags: identifies the content without
                # re-reading GBs that were just downloaded.
                data_hash = fetched.content_id
            elif Path(resolved).is_dir():
                data_hash = _sha256_dataset_dir(Path(resolved))
            else:
                from ledgerhound.runtime.events import sha256_file
                data_hash = sha256_file(resolved)
        except Exception:
            data_hash = ""
        # V3-053: the Librarian keys stories off this. Stored on self (not
        # just the event payload) so /library, open_chapter, etc. can read
        # "what data version is loaded right now" without re-hashing.
        self._data_sha256 = data_hash
        self._events.append("load_data", path=str(resolved),
                            source=(fetched.source.uri if fetched else "local"),
                            rows=self._true_n_rows,
                            rows_exact=self._true_rows_exact,
                            cols=int(df.shape[1]),
                            shell_sampled=self._df_is_sample,
                            data_sha256=data_hash)
        # Outputs follow the source: decide features_dir (and the upload
        # target for store data) BEFORE the kernel starts, since the kernel
        # is handed features_dir at start.
        paths_note = self._set_run_paths(path, resolved, fetched)
        self._ns = None  # reset sandbox namespace for the new data
        self._digest = None  # force a fresh profile for the new data
        self._digest_summary = ""
        self._digest_from_card = False
        from ledgerhound.knowledge.team_kb import dataset_id_for
        self._dataset_id = dataset_id_for(
            name=self._dataset_name_override,
            s3_uri=fetched.source.uri if fetched is not None else "",
            content_sha256=data_hash)
        self._load_card_for(resolved, self._dataset_id, data_hash)  # V3-011: may pre-seed self._digest
        self._refresh_schema_pin()  # visible to the rest of THIS turn, not just the next
        kernel_note = self._start_kernel_for(resolved) if self.use_kernel else ""
        self._kernel_note = kernel_note
        cols = ", ".join(map(str, df.columns[:40]))
        n_disp = self._rows_display()
        msg = (f"loaded {resolved} -> df with {n_disp} rows x "
               f"{df.shape[1]} columns.\ncolumns: {cols}")
        if fetched is not None:
            msg = f"from the object store: {fetched.summary()}\n" + msg
            for n in fetched.notes:
                msg += f"\n({n})"
        if paths_note:
            msg += f"\n{paths_note}"
        if self._df_is_sample:
            msg += (f"\n(large dataset: the kernel holds all {n_disp} rows -- "
                    f"your run_python code sees the FULL data. This process "
                    f"kept only a {len(df):,}-row schema sample, so "
                    f"profile_data's summary stats are estimates from that "
                    f"sample; compute exact figures with run_python.)")
        if kernel_note:
            msg += f"\n{kernel_note}"
        # Recognise the dataset from its column names alone and PROPOSE the
        # goal immediately. The 2026-08-06 run had to rediscover "this is an
        # account-level mule problem" over hours, then shipped the wrong
        # grain anyway -- the schema said so from the first second.
        recognised = self._recognise_dataset(df.columns)
        if recognised:
            msg += f"\n\n{recognised}"
        chosen = self._choose_methodology_once()
        if chosen:
            msg += f"\n\n{chosen}"
        prior = self._recall_books(f"{Path(resolved).name} {cols}")
        if prior:
            msg += "\n\n" + prior
        return msg

    def team_kb(self):
        """The shared knowledge base, or None without a project (the
        adversarial debate agents and ad-hoc sessions have none)."""
        if self.project is None:
            return None
        if self._kb is None:
            try:
                from ledgerhound.knowledge.team_kb import TeamKB, enabled
                if not enabled():
                    return None
                self._kb = TeamKB()
                self._kb._kick()        # upload anything queued by an earlier session
            except Exception:  # noqa: BLE001 -- the KB is never allowed to break a session
                self._kb = None
        return self._kb

    def _kb_fetch_card(self, dataset_id: str):
        kb = self.team_kb()
        if kb is None or not dataset_id:
            return None
        try:
            return kb.fetch_card(dataset_id)
        except Exception:  # noqa: BLE001
            return None

    def _kb_publish_card(self) -> None:
        kb = self.team_kb()
        if kb is None or self._card is None or not getattr(self._card, "dataset_id", ""):
            return
        try:
            kb.publish_card(self._card)
        except Exception as e:  # noqa: BLE001
            self._events.append("team_kb_error", op="publish_card", error=str(e)[:300])

    def _kb_pull_extras(self, dataset_id: str) -> None:
        """Team findings (for the 'already known' check) and books (for
        recall) recorded on this dataset by anyone."""
        kb = self.team_kb()
        if kb is None or not dataset_id:
            return
        try:
            self._team_findings = kb.fetch_findings(dataset_id)
            from ledgerhound.analyst.books import BOOKS_DIR
            books_dir = Path(self.project.dotdir) / BOOKS_DIR
            for b in kb.fetch_books(dataset_id):
                bid = str(b.get("id") or "")
                if not bid or (books_dir / f"{bid}.json").exists():
                    continue
                from ledgerhound.analyst.books import Book
                b = {k: v for k, v in b.items() if not k.startswith("_") and k != "dataset_id"}
                book = Book.from_dict(b)
                books_dir.mkdir(parents=True, exist_ok=True)
                (books_dir / f"{bid}.json").write_text(json.dumps(book.to_dict(), indent=2),
                                                       encoding="utf-8")
                (books_dir / f"{bid}.md").write_text(book.to_markdown(), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self._events.append("team_kb_error", op="pull", error=str(e)[:300])

    def _save_card(self) -> None:
        """Save the dataset card locally, then share it (every card change
        -- a digest, a decision, a chapter page, an answer -- reaches the
        team automatically)."""
        from ledgerhound.knowledge.dataset_card import DatasetCardStore
        DatasetCardStore(self.project).save(self._card)
        self._kb_publish_card()

    def _load_card_for(self, resolved: Path, dataset_id: str = "",
                       data_sha256: str = "") -> None:
        """Load (or create) this dataset's durable card (V3-011). If a prior
        session already profiled this exact file, pre-seed the digest cache
        so profile_data does not recompute it. Requires a project --
        persistence across sessions is the entire point; with none, cards
        are skipped and behaviour is unchanged from before V3-011."""
        self._card = None
        self._team_findings = []
        if self.project is None:
            return
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore, merge_cards
            store = DatasetCardStore(self.project)
            card = store.load_or_create(resolved)
            had_digest = bool(card.digest)
            if dataset_id:
                card.dataset_id = dataset_id
            # Team knowledge base: fold in what everyone has recorded on this
            # dataset (stories, decisions, findings, answers to data
            # questions). Store unreachable -> local card only, never an error.
            remote = self._kb_fetch_card(dataset_id)
            if remote is not None:
                remote_digest_ok = (not data_sha256 or remote.data_sha256 == data_sha256)
                card = merge_cards(remote, card)
                if not had_digest and not remote_digest_ok:
                    card.digest = {}          # a teammate's profile of OTHER content
            card.add_run(self.run_dir.name)
            store.save(card)
            self._card = card
            self._kb_publish_card()
            self._kb_pull_extras(dataset_id)
            if card.digest:
                self._digest = card.digest
                self._digest_from_card = True
                try:
                    from ledgerhound.profiler.digest import Digest
                    self._digest_summary = Digest(**card.digest).summary()[:4000]
                except Exception:
                    self._digest_summary = ""
                    self._digest_from_card = False
        except Exception:  # noqa: BLE001
            self._card = None

    def _resolve_compute_engine(self, resolved: Path) -> str:
        """Which dataframe engine the kernel should use:
        "dask_cudf" | "cudf" | "pandas".

        Asks the SAME router the digest path already uses
        (ledgerhound.backends.compute_router), so the shell and the profiler
        agree on the machine instead of each deciding separately. The router
        accounts for GPU count, data size vs GPU memory, and whether a LOCAL
        LLM is occupying a GPU (a remote API like DeepSeek/gateway Kimi leaves
        every GPU free for cuDF).

        Advisory only: if the router raises, or cuDF turns out to be broken
        at kernel start, KernelExecutor falls back to pandas by itself. There
        is no path here that can make a GPU-less machine behave differently
        than it does today.
        """
        try:
            from ledgerhound.backends.compute_router import ComputeRouter
            router = ComputeRouter()
            # `llm` may be a test stub / any object -- never assume .config.
            base_url = getattr(getattr(getattr(self, "llm", None), "config", None), "base_url", "")
            profile = router.detect(
                str(resolved), llm_is_local=router.infer_llm_is_local(base_url))
            # Pass the router's choice STRAIGHT through. The previous line
            # was `"cudf" if choice == "cudf" else "pandas"`, which silently
            # rewrote a `dask_cudf` decision -- the multi-GPU path V3-038
            # added -- into CPU pandas. KernelExecutor has handled
            # "dask_cudf" since V3-038 but nothing ever passed it, so the
            # whole branch was unreachable and a 3xH100 box kept landing on
            # CPU no matter what the router worked out. Anything unexpected
            # still degrades to pandas.
            if profile.backend_choice in ("cudf", "dask_cudf"):
                return profile.backend_choice
            return "pandas"
        except Exception:  # noqa: BLE001 -- routing is advisory, never fatal
            return "pandas"

    def _record_gpu_fallback(self, stage: str, error_text: str,
                             fell_back_to: str, remedy: str = "",
                             error_class: str = "",
                             rmm_status: str = "") -> None:
        """Log a GPU->CPU downgrade and append it to the run's errornote.

        V3-047. Two destinations on purpose: `gpu_fallback` in the
        hash-chained event stream so it is replayable and auditable, and
        `gpu_fallback_rca.md` in the run folder so a human opening the
        output directory finds out why the run was slow without grepping
        JSON. Never raises -- an RCA that fails must not take the session
        with it.
        """
        if not error_class or not remedy:
            from ledgerhound.backends.compute_router import classify_gpu_error
            cls, rem = classify_gpu_error(error_text, rmm_status)
            error_class = error_class or cls
            remedy = remedy or rem
        try:
            self._events.append("gpu_fallback", stage=stage,
                                error_class=error_class,
                                error_text=(error_text or "")[:2000],
                                remedy=remedy, fell_back_to=fell_back_to,
                                rmm_status=rmm_status)
        except Exception:  # noqa: BLE001
            pass
        try:
            path = self.run_dir / "gpu_fallback_rca.md"
            new = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    f.write("# Why this run did not use the GPU\n\n"
                            "Each entry is one point where the GPU path was "
                            "abandoned, with the cause and the fix. Written "
                            "as it happened.\n")
                f.write(f"\n## {stage}\n\n"
                        f"- **Fell back to:** {fell_back_to}\n"
                        f"- **Cause:** `{error_class}`\n"
                        f"- **Fix:** {remedy}\n")
                if rmm_status:
                    f.write(f"- **RMM at the time:** {rmm_status}\n")
                if error_text:
                    f.write(f"\n```\n{(error_text or '')[:2000].strip()}\n```\n")
        except Exception:  # noqa: BLE001
            pass

    def _start_kernel_for(self, resolved: Path) -> str:
        """(Re)start the persistent IPython kernel for newly-loaded data
        (V3-009). Best-effort: any failure (missing jupyter_client/ipykernel,
        no working kernelspec, sandboxed subprocess restrictions) falls back
        to the local exec sandbox instead of breaking the load."""
        if self._kernel is not None:
            try:
                self._kernel.shutdown()
            except Exception:
                pass
            self._kernel = None
        self._kernel_active = False
        try:
            from ledgerhound.auto.kernel import KernelExecutor, kernels_available
            if not kernels_available():
                # The exec sandbox is pandas-only -- it has NO cuDF branch --
                # so this is a total GPU loss, not a degraded mode (V3-047).
                self._record_gpu_fallback(
                    "kernel_unavailable",
                    "jupyter_client/ipykernel not installed",
                    "CPU exec sandbox",
                    "The exec sandbox has no GPU code path at all, so no "
                    "GPU can be used in this session regardless of hardware. "
                    "Install the kernel dependencies: pip install "
                    "jupyter_client ipykernel")
                return ("(kernel mode requested but jupyter_client/ipykernel "
                        "aren't installed -- using the restricted exec sandbox "
                        "instead; NO GPU is reachable this way, see "
                        "gpu_fallback_rca.md)")
            engine = self._resolve_compute_engine(resolved)
            self._loaded_path = str(resolved)
            # I-10: the per-cell timeout used to be a hardcoded 120s. On a
            # crore-scale frame a single groupby on the CPU fallback takes
            # longer than that, the cell was interrupted, and the identical-
            # failure nudge then pushed the model toward shortcuts -- the
            # opposite of "hours are fine". Configurable via main_config's
            # kernel.cell_timeout_seconds (env LEDGERHOUND_CELL_TIMEOUT);
            # default 30 min, a runaway-cell net rather than a pace-setter.
            from ledgerhound.config import get_int as _get_int
            cell_timeout = _get_int("kernel", "cell_timeout_seconds",
                                    env="LEDGERHOUND_CELL_TIMEOUT", default=1800)
            if self.isolation is not None:
                cell_timeout = self.isolation.cell_timeout_s
            from ledgerhound.runtime.isolation import IsolationError
            try:
                self._kernel = KernelExecutor(
                    scratch_dir=self.run_dir / "scratch",
                    data_path=str(resolved), engine=engine,
                    timeout=max(30, cell_timeout),
                    features_dir=self._features_dir(),
                    isolation=self.isolation)
            except IsolationError as e:
                self._kernel_refusal = f"REFUSED: {e}"
                self._log("isolation", level="refused", reason=str(e))
                return f"({self._kernel_refusal})"
            self._kernel_refusal = ""
            self._kernel_active = True
            self.isolation_report = self._kernel.isolation_report
            self._log("isolation", **self.isolation_report.as_dict())
            # Drain whatever the kernel downgraded during startup into the
            # canonical log. KernelExecutor has no EventStream of its own.
            for note in getattr(self._kernel, "gpu_fallbacks", []):
                self._record_gpu_fallback(
                    note["stage"], note["error_text"], note["fell_back_to"],
                    note["remedy"], error_class=note["error_class"],
                    rmm_status=note.get("rmm_status", ""))
            # Define check_leakage/evaluate_feature/evaluate_features INSIDE
            # the kernel. Without this they exist only in the exec sandbox's
            # namespace, so in kernel mode the model follows the system
            # prompt's mandate to call them and gets a bare NameError (audit
            # finding, 2026-08-06 -- see ledgerhound/shell/kernel_helpers.py).
            # Inject, then VERIFY by asking the kernel whether the names
            # actually exist -- the 2026-08-06 production run showed the
            # "failed to load" warning on 2 of 5 kernel starts, i.e.
            # _exec_raw's success flag alone is not reliable here (a slow
            # first start races the 60s timeout). Retry once, longer, and
            # confirm by introspection rather than by exit status.
            helper_note = ""
            try:
                from ledgerhound.shell.kernel_helpers import ANALYST_HELPERS_SNIPPET

                def _helpers_present() -> bool:
                    # globals(), not dir(): inside a generator expression
                    # dir() lists the generator's own scope, so the check
                    # always said MISSING (a false warning on every start).
                    chk = self._kernel._exec_raw(
                        "print('HELPERS_OK' if all(n in globals() for n in "
                        "('check_leakage','evaluate_feature','evaluate_features')) "
                        "else 'HELPERS_MISSING')", timeout=60)
                    return "HELPERS_OK" in (getattr(chk, "output", "") or "")

                self._kernel._exec_raw(ANALYST_HELPERS_SNIPPET, timeout=180)
                if not _helpers_present():
                    self._kernel._exec_raw(ANALYST_HELPERS_SNIPPET, timeout=300)
                    if not _helpers_present():
                        helper_note = (
                            " -- WARNING: analyst helpers "
                            "(evaluate_feature/check_leakage) failed to load; "
                            "evaluate features with an explicit IV/AUC "
                            "computation instead of assuming")
            except Exception:  # noqa: BLE001 -- never break a working kernel
                helper_note = (" -- WARNING: analyst helpers "
                               "(evaluate_feature/check_leakage) unavailable")
            # The kernel may still downgrade cudf->pandas internally if the
            # CUDA context fails to initialise; report what it ACTUALLY ended
            # up with, not what we asked for.
            actual = getattr(self._kernel, "_engine", engine)
            gpu_note = (" -- data engine: cuDF (GPU)" if actual == "cudf"
                        else " -- data engine: pandas (CPU)")
            iso = self.isolation_report
            iso_note = f" -- {iso.summary()}" if iso and iso.level != "off" else ""
            return ("(real Python kernel started -- full language"
                    f"{gpu_note}{helper_note}{iso_note})")
        except Exception as e:  # noqa: BLE001
            self._kernel = None
            self._kernel_active = False
            return (f"(kernel start failed ({type(e).__name__}: {e}) -- "
                    f"using the restricted exec sandbox instead)")

    def _tool_profile_data(self) -> str:
        if self._df is None:
            return "no data loaded yet -- call load_data first."
        # Already profiled -- either earlier this session, or (V3-011) in a
        # PRIOR session on this exact file via the dataset card -- return the
        # cached digest instead of re-running the full profile (and
        # re-detecting GPUs) every time. Deterministic for fixed data;
        # recomputing wastes time and clutters the terminal.
        if self._digest is not None and self._digest_summary:
            if self._digest_from_card:
                note = ("(already profiled in a prior session on this exact "
                        "file -- returning the saved digest from its dataset "
                        "card; call again with fresh data if it has changed)")
            else:
                note = ("(already profiled -- returning the cached digest; the "
                        "data has not changed this session)")
            return note + "\n\n" + self._digest_summary + self._data_questions_section()
        summary = ""
        try:
            from ledgerhound.core import Ledgerhound
            # Profile the frame we ALREADY hold, instead of handing Ledgerhound
            # the path and making it read the same file a second time (this
            # session loaded it once in _read_any). The re-read bought
            # nothing: on a GPU box Ledgerhound's cuDF branch reads the file
            # with cudf and then immediately calls .to_pandas() before
            # profiling, so the digest was computed on CPU either way -- the
            # duplicate read was pure cost. `source=` keeps the digest's
            # provenance pointing at the real file, not the frame.
            eng = Ledgerhound(data=self._df, source=str(self._loaded_path or ""))
            digest = eng.profile()
            self._digest = digest.to_dict()
            summary = digest.summary()[:4000]
        except Exception:
            d = self._df
            miss = (d.isna().mean() * 100).round(1)
            top = miss.sort_values(ascending=False).head(8)
            self._digest = {"rows": len(d), "cols": int(d.shape[1])}
            summary = (f"{len(d):,} rows x {d.shape[1]} cols. dtypes: "
                       f"{d.dtypes.value_counts().to_dict()}. "
                       f"most-missing %: {top.to_dict()}")
        # A sampled shell-side frame must NEVER have its stats presented as
        # whole-dataset facts -- that would be a correctness bug worse than
        # the memory cost this sampling avoids. Label it loudly, on both the
        # text the model reads and the persisted digest.
        if self._df_is_sample:
            n_s = len(self._df) if self._df is not None else 0
            n_disp = self._rows_display()
            summary = (f"[SAMPLE-BASED ESTIMATE -- computed on {n_s:,} of "
                       f"{n_disp} rows, because the full dataset lives in the "
                       f"kernel. Distributions/rates below are ESTIMATES, not "
                       f"population values. Recompute anything you intend to "
                       f"report with run_python, which sees all {n_disp} "
                       f"rows.]\n\n" + summary)
            if isinstance(self._digest, dict):
                self._digest["sampled"] = True
                self._digest["sample_rows"] = n_s
                self._digest["true_rows"] = self._true_n_rows
        self._digest_summary = summary
        self._digest_from_card = False
        # Persist the digest into the run folder so the session is auditable /
        # reproducible (one digest, one place, tied to this run).
        try:
            (self.run_dir / "digest.json").write_text(
                json.dumps(self._digest, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        # V3-011: also persist onto the dataset card so the NEXT session on
        # this exact file skips re-profiling entirely. NOT done for a
        # sample-based digest -- a future session would load it as a full
        # profile and silently inherit sample estimates as population facts.
        if self._card is not None and not self._df_is_sample:
            try:
                self._card.set_digest(self._digest)
                self._save_card()
            except Exception:
                pass
        return summary + self._data_questions_section()

    def _answered_questions(self) -> dict:
        if self._card is None:
            return dict(getattr(self, "_session_answers", {}) or {})
        return {**(getattr(self, "_session_answers", {}) or {}),
                **self._card.current_clarifications()}

    def _data_questions_section(self) -> str:
        """Known answers + open questions about the loaded data, appended to
        the profile. The model is told to ask the open ones before any
        analysis (system prompt, 'Questions about the data')."""
        if self._df is None:
            return ""
        try:
            from ledgerhound.shell.data_questions import open_questions, questions_block
            known = self._answered_questions()
            qs = open_questions(self._df, set(known))
            self._open_data_questions = qs
            self._questions_reminded = False
            if qs:
                self._events.append("data_questions", keys=[q.key for q in qs])
            block = questions_block(qs, known)
            if block and self._df_is_sample:
                block += ("\n(class counts above come from the schema sample, "
                          "not the full data.)")
            return f"\n\n{block}" if block else ""
        except Exception:  # noqa: BLE001 -- advisory
            return ""

    def _tool_record_data_answer(self, key: str, answer: str, question: str = "") -> str:
        """Save the user's answer to a data question on the dataset card
        (shared with the team, so it is never asked again for this data)."""
        key, answer = (key or "").strip(), (answer or "").strip()
        if not key or not answer:
            return "record_data_answer needs the question's key and the user's answer."
        q = next((x for x in self._open_data_questions if x.key == key), None)
        question = question or (q.text if q else "")
        from ledgerhound.identity import current_user
        who = current_user()
        self._events.append("data_answer", key=key, answer=answer[:2000])
        self._open_data_questions = [x for x in self._open_data_questions if x.key != key]
        if self._card is None or self.autonomous:
            by = who if not self.autonomous else "autonomous run -- UNCONFIRMED"
            self._session_answers = {**(getattr(self, "_session_answers", {}) or {}),
                                     key: {"key": key, "answer": answer, "by": by}}
            why = ("no human confirmed it in this unattended run, so it is NOT saved "
                   "to the shared dataset card -- state it as an assumption in the report"
                   if self.autonomous else "no project, so it is not saved to a dataset card")
            return f"noted [{key}]: {answer} (this session only -- {why})."

        try:
            self._card.add_clarification(key, question, answer, by=who)
            self._save_card()
        except Exception as e:  # noqa: BLE001
            return f"could not save the answer: {e}"
        left = len(self._open_data_questions)
        return (f"recorded [{key}] on the dataset card and shared with the team."
                + (f" {left} question(s) still open." if left else " All data questions answered."))

    def _recognise_dataset(self, columns) -> str:
        """Match the loaded schema against the domain profiles and, when one
        fits, pre-seed the session goal with its proposal.

        Deliberately does NOT overwrite a goal the user or model already
        declared -- a profile is a prior, not an authority. Best-effort: any
        failure returns "" so recognition can never break data loading."""
        try:
            from ledgerhound.context.domain_profiles import (
                describe,
                match_profile,
                resolve_entity_key,
                resolve_label_column,
            )
            prof = match_profile(list(columns))
            if not prof:
                return ""
            # Kept so record_finding can tell an already-established claim
            # from a new one (V3-048).
            self._domain_profile = prof
            block = describe(prof, list(columns))
            self._events.append("dataset_recognised", profile=prof.get("name"),
                                detail=block[:2000])
            if not self.goal.is_set():
                from ledgerhound.shell.session_goal import normalise_grain
                g = (prof.get("goal") or {})
                self.goal.objective = " ".join(str(g.get("objective", "")).split())
                self.goal.unit_of_analysis = normalise_grain(
                    g.get("unit_of_analysis", ""))
                self.goal.entity_key = resolve_entity_key(prof, list(columns))
                self.goal.label_column = resolve_label_column(prof, list(columns))
                self.goal.notes = f"proposed by domain profile '{prof.get('name')}'"
                self.goal.save(self.run_dir / "session_goal.json")
                block += ("\n(The session goal has been PRE-SET from this "
                          "profile and is now pinned. Change it with set_goal "
                          "if the user wants something different.)")
            return block
        except Exception:  # noqa: BLE001 -- recognition must never break a load
            return ""

    def load_from_command(self, path: str, *, kind: str, as_one: bool = False,
                          name: str = "") -> str:
        """`/local --path` and `/s3 --path`: the same loader the model uses,
        with the source stated explicitly (so a path is never guessed as the
        wrong kind) and `--as-one` available for a folder of Parquet files
        that is one dataset. Raises DataSourceError for a bad argument."""
        from ledgerhound.io import data_source as _ds
        from ledgerhound.io.minio import shared_settings
        src = _ds.parse(path, kind=kind,
                        default_bucket=shared_settings().get("bucket", "") if kind == "s3" else "")
        self._force_one_dataset = as_one
        self._dataset_name_override = (name or "").strip()
        try:
            return self._tool_load_data(src.uri)
        finally:
            self._force_one_dataset = False
            self._dataset_name_override = ""

    def _fetch_from_store(self, src):
        """Download an s3:// source into the cache, with progress lines."""
        from ledgerhound.io import data_source as _ds
        from ledgerhound.io.minio import shared_settings
        s = shared_settings()
        self.render(f"fetching {src.uri} from the object store "
                    f"({s.get('endpoint') or 'endpoint not set'}) ...")
        last = {"pct": -10}

        def _progress(done_b, total_b, done_n, total_n):
            pct = int(100 * done_b / total_b) if total_b else 100
            if pct >= last["pct"] + 10 or done_n == total_n:
                last["pct"] = pct
                self.render(f"  {pct:3d}%  {done_n}/{total_n} files, "
                            f"{_ds._human(done_b)} of {_ds._human(total_b)}")
        fr = _ds.fetch(src, settings=s, progress=_progress,
                       as_one=self._force_one_dataset)
        self._events.append("data_fetched", source=src.uri, local=str(fr.local_path),
                            files=fr.files, bytes=fr.total_bytes,
                            downloaded=fr.downloaded_bytes,
                            seconds=round(fr.seconds, 1))
        return fr

    def _set_run_paths(self, asked: str, resolved, fetched) -> str:
        """Point outputs at the right place for this data source, write
        run_dir/paths.json, and return a line for the model/user."""
        from ledgerhound.io import data_source as _ds
        self._uploader = None
        self._features_override = None
        note = ""
        if fetched is not None:
            if _ds.is_ram_backed(fetched.cache_dir):
                self._features_override = (fetched.cache_dir / "runs" /
                                           self.run_dir.name / "features")
            try:
                bucket, prefix = _ds.outputs_prefix(self.run_dir.name)
                self._uploader = _ds.OutputUploader(self._features_dir(), bucket,
                                                    prefix + "features/")
                note = (f"outputs: save feature tables under features_dir "
                        f"({self._features_dir()}); each run_python step uploads "
                        f"new or changed files there to {self._uploader.uri}")
            except Exception as e:  # noqa: BLE001
                note = (f"outputs: features_dir is {self._features_dir()} -- NOT "
                        f"uploaded to the object store ({e}).")
        rp = _ds.RunPaths(
            source="s3" if fetched is not None else "local",
            data=fetched.source.uri if fetched is not None else str(resolved),
            local_copy=str(fetched.local_path) if fetched is not None else "",
            run_dir=str(self.run_dir), scratch_dir=str(self.run_dir / "scratch"),
            features_dir=str(self._features_dir()),
            features_upload=self._uploader.uri if self._uploader else "")
        try:
            rp.save(self.run_dir)
        except OSError:
            pass
        self._run_paths = rp
        return note

    def run_paths(self):
        """The current RunPaths (for /paths); a local placeholder before any
        data is loaded."""
        if self._run_paths is not None:
            return self._run_paths
        from ledgerhound.io.data_source import RunPaths
        return RunPaths(source="(no data loaded yet)", run_dir=str(self.run_dir),
                        scratch_dir=str(self.run_dir / "scratch"),
                        features_dir=str(self.run_dir / "features"))

    def _sync_outputs(self) -> str:
        """Upload new/changed feature files (object-store sessions only).
        Returns a short note for the tool result, '' when nothing moved."""
        if self._uploader is None:
            return ""
        try:
            done, errs = self._uploader.sync()
        except Exception as e:  # noqa: BLE001 -- never break a step on upload
            done, errs = [], [f"upload failed: {type(e).__name__}: {e}"]
        if done:
            self._events.append("outputs_uploaded", target=self._uploader.uri,
                                files=done[:200])
        if errs:
            self._events.append("outputs_upload_failed", target=self._uploader.uri,
                                errors=errs[:50])
        parts = []
        if done:
            shown = ", ".join(done[:8]) + (f", +{len(done) - 8} more" if len(done) > 8 else "")
            parts.append(f"[uploaded to {self._uploader.uri}: {shown}]")
        if errs:
            parts.append("[upload FAILED -- files are kept locally in features_dir "
                         "and retried after the next step: " + "; ".join(errs[:3]) + "]")
        return "\n".join(parts)

    def _candidate_roots(self) -> list[Path]:
        roots = [Path.cwd(), Path.cwd().parent]
        if self.project is not None:
            try:
                pr = Path(self.project.root)
                roots += [pr, pr.parent]        # data at ../ from the project
            except Exception:
                pass
        # de-dup, keep order
        seen, out = set(), []
        for r in roots:
            rp = r.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(r)
        return out

    def _resolve_dir(self, path: str) -> Path | None:
        p = Path(path)
        if p.exists():
            return p
        for root in self._candidate_roots():
            if (root / path).exists():
                return root / path
        return None

    def _resolve_data(self, path: str):
        """Return a resolved file Path, a list of candidate files (folder), or None."""
        tried: list[Path] = []
        p = Path(path)
        cands = [p] + [root / path for root in self._candidate_roots()]
        for c in cands:
            tried.append(c)
            if c.exists():
                if c.is_dir() and (_is_parquet_dataset_dir(c) or (
                        self._force_one_dataset and _dataset_parts(c) and not any(
                            f.suffix.lower() in _DATA_EXTS and f.suffix.lower() not in (".parquet", ".pq")
                            for f in c.rglob("*") if f.is_file()))):
                    self._loaded_path = str(c)
                    return c
                if c.is_dir():
                    files = [str(f) for f in sorted(c.iterdir())
                             if f.suffix.lower() in _DATA_EXTS]
                    return files or None
                self._loaded_path = str(c)
                return c
        # last resort: glob the basename under cwd/project (one level)
        name = p.name
        for root in self._candidate_roots():
            hits = list(root.glob(name)) + list(root.glob(f"*/{name}"))
            if len(hits) > 1:
                # I-19: several files share this basename -- silently
                # loading the first would analyse the wrong month/extract.
                # Return the choices, same as a folder listing does.
                return [str(h) for h in sorted(hits)]
            if hits:
                self._loaded_path = str(hits[0])
                return hits[0]
        return None
