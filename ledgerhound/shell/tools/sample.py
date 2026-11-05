"""
ledgerhound.shell.tools.sample
==============================

build_sample: the roles-driven sample and its construction certificate
(ledgerhound.pipeline.build). This mixin owns what needs the session: the
roles gate, the label-lag question to the person, files, events and the
kernel variables.
"""

from __future__ import annotations

LAG_YES = "yes, assume this label lag"
LAG_NO = "no, do not assume it"


class SampleToolsMixin:
    """build_sample + the current certificate."""

    _certificate = None
    _sample_meta: dict = None
    _lag_approval: tuple = None         # (days, stated, approved_by) for this dataset

    @property
    def certificate(self):
        return self._certificate

    def _reset_sample(self) -> None:
        self._certificate = None
        self._sample_meta = None
        self._lag_approval = None

    def _label_source_value(self, y) -> str:
        col = self._roles.first("label_source")
        if not col:
            return ""
        vals = self._df.loc[y == 1, col].dropna().astype(str).str.strip().str.lower()
        return str(vals.mode().iloc[0]) if len(vals) else ""

    def _tool_build_sample(self, design_prevalence="", label_lag: str = "", split: str = "",
                           entity: str = "", time_unit: str = "", test_fraction: float = 0.2,
                           embargo_days: int = 0, seed: int = 42) -> str:
        import pandas as pd

        from ledgerhound.pipeline.build import (
            BuildRefusal,
            LabelLag,
            build_sample,
            measured_lag,
            parse_assumption,
            summary_lines,
            to_times,
        )
        if self._df is None or self._roles is None:
            return "no data loaded -- call load_data first."
        rm = self._roles
        if not rm.confirmed:
            return ("REFUSED: build_sample is driven by the column roles, which are not "
                    "confirmed yet. Call confirm_roles first. Nothing was built.")
        label = rm.first("label")
        if not label:
            return ("REFUSED: the confirmed roles have no label column, so there is nothing "
                    "to sample against. Nothing was built.")
        df = self._df
        if not isinstance(df, pd.DataFrame):
            df = df.compute() if hasattr(df, "compute") else df
            df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)

        try:
            y = pd.to_numeric(df[label], errors="coerce")
            policy = None
            try:
                from ledgerhound.kb.policy import maturity_for
                policy = maturity_for(self.knowledge_pack(), self._label_source_value(y),
                                      self._dataset_names())
            except Exception as e:  # noqa: BLE001 -- the pack is advisory here
                self._events.append("kb_error", op="maturity_for", error=str(e)[:300])
            time_col = rm.first("event_time")
            times = to_times(df[time_col], time_unit) if time_col else None
            lt_col = rm.first("label_time")
            if lt_col:
                if times is None:
                    raise BuildRefusal("there is a label_time column but no event_time, so "
                                       "the label lag cannot be measured. Nothing was built.")
                lt, _ = to_times(df[lt_col], time_unit)
                lag = measured_lag(times[0], lt, y, policy)
            else:
                lag = self._assumed_lag(label_lag, policy, parse_assumption, LabelLag)
                if isinstance(lag, str):
                    return lag
            name = (self._dataset_names() or ["sample"])[0]
            result = build_sample(df, rm, lag, design_prevalence=design_prevalence,
                                  split=split, entity=entity, time_unit=time_unit,
                                  test_fraction=float(test_fraction),
                                  embargo_days=int(embargo_days or 0), seed=int(seed),
                                  name=f"{name} build_sample", times=times)
        except BuildRefusal as e:
            self._events.append("sample_refused", reason=str(e)[:500])
            return f"REFUSED: {e}"

        # under scratch/: the only run folder the sandboxed kernel can see
        out = self.run_dir / "scratch" / "sample"
        out.mkdir(parents=True, exist_ok=True)
        paths = {"train": out / "train.parquet", "test": out / "test.parquet",
                 "certificate": self.run_dir / "certificate.json"}
        result.train.to_parquet(paths["train"])
        result.test.to_parquet(paths["test"])
        cert = result.certificate
        cert.save(paths["certificate"])
        self._certificate = cert
        import hashlib
        files = {k: hashlib.sha256(paths[k].read_bytes()).hexdigest() for k in ("train", "test")}
        self._sample_meta = dict(result.meta, content_hash=cert.content_hash(),
                                 can_certify=cert.can_certify,
                                 train_sha256=files["train"], test_sha256=files["test"])
        self._events.append("sample_built", **{k: (str(v) if not isinstance(
            v, (int, float, bool, str)) else v) for k, v in self._sample_meta.items()})
        loaded = self._run_python_step(
            "import pandas as _lh_pd\n"
            f"train = _lh_pd.read_parquet({str(paths['train'])!r})\n"
            f"test = _lh_pd.read_parquet({str(paths['test'])!r})\n"
            "print(len(train), len(test))")
        where = ("  in the analysis kernel as `train` and `test`"
                 if loaded.strip().split()[:2] == [str(len(result.train)), str(len(result.test))]
                 else f"  (could not load into the kernel: {loaded.strip()[:200]})")
        return summary_lines(result, {k: str(v) for k, v in paths.items()}) + "\n" + where

    def _assumed_lag(self, stated: str, policy, parse_assumption, LabelLag):
        """The label-lag ASSUMPTION route: no label-time column, so a person
        states the lag (or that labels are final). Never silent."""
        pnote = policy.note() if policy is not None else ""
        if not (stated or "").strip():
            hint = (f" Your Knowledge Pack's label policy suggests {pnote}; "
                    "label_lag='policy' uses it." if pnote else "")
            return ("REFUSED: the data has no label-time column, so label maturity cannot be "
                    "checked. Ask the user how long fraud labels take to arrive, then call "
                    "build_sample again with label_lag=<days>, or label_lag='final' if the "
                    f"labels will not change.{hint} The certificate will record it as an "
                    "assumption, not a check. Nothing was built.")
        days = parse_assumption(stated, policy)
        if self._lag_approval and self._lag_approval[0] == days:
            by = self._lag_approval[2]
        else:
            what = "the labels are final" if days == 0 else f"fraud labels arrive within {days} days"
            q = (f"This data has no label-time column. Build the sample assuming {what}?\n"
                 "The certificate will record this as an ASSUMPTION, not a check: rows "
                 f"from the last {days} days are kept out of the test split.")
            if pnote:
                q += f"\nYour Knowledge Pack's label policy: {pnote}."
            decision = self.registry.gate.choose(q, [LAG_YES, LAG_NO])
            by = "agent" if getattr(decision, "approved_by", "person") == "agent" else "person"
            self._events.append("label_lag_assumption", days=days, stated=stated,
                                approved_by=by, accepted=decision.answer == LAG_YES)
            if decision.answer != LAG_YES:
                return ("label lag NOT assumed -- ask the user how long labels take to "
                        "arrive, then call build_sample again. Nothing was built.")
            self._lag_approval = (days, stated, by)
        return LabelLag(days, "assumed", stated=str(stated), approved_by=by, policy_note=pnote)

    def _sample_provenance(self) -> list[str]:
        m = self._sample_meta
        if not m:
            return []
        gate = "certified" if m["can_certify"] else "**NOT certified**"
        basis = ("measured from the label-time column" if m["maturity_basis"] == "measured"
                 else "**assumed** (no label-time column)")
        return [f"- Sample built by build_sample ({m['split']} split, {m['n_train']:,} train / "
                f"{m['n_test']:,} test rows): {gate}, certificate {m['content_hash'][:16]}; "
                f"label maturity {m['maturity_days']} days, {basis}."]
