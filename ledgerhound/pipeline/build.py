"""
ledgerhound.pipeline.build
==========================

build_sample: a train/test sample built from the **confirmed column roles**,
with no dataset-specific code, and a construction certificate that says
how it was made.

Roles drive everything:

    label        -> the target (must be confirmed, must be binary)
    event_time   -> the temporal split; a numeric time needs a stated unit
    label_time   -> label maturity *measured* (lag = label_time - event_time)
    entity roles -> entity-disjoint splitting (payer, account, card, ...)

Two refusals are the point (design doc, Phase 3):

* **Label lag unchecked.** With a label_time column the lag is measured.
  Without one, the caller must pass an assumption -- a lag in days, "final"
  (labels will not change) or "policy" (the Knowledge Pack's label_policy) --
  and the certificate records it as an ASSUMPTION, never as a check.
* **Prevalence unchecked.** The caller declares the design prevalence (or
  "natural" to keep the data's own rate, recorded as such). The train side is
  down-sampled to it if needed; the test side keeps the natural rate.

Grain is transaction only in Phase 3: account-grain samples need an
aggregation step that is not role-driven yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ledgerhound.pipeline.appropriateness import SEV_PASS, SEV_WARNING, Finding

EPOCH = "2000-01-01"
TIME_UNITS = {"seconds": 1 / 86400, "minutes": 1 / 1440, "hours": 1 / 24, "days": 1.0,
              "weeks": 7.0, "months": 30.4375}
ENTITY_PREFERENCE = ("payer", "account", "card", "device", "payee", "merchant")
SECTION_MATURITY = "label_maturity"


class BuildRefusal(ValueError):
    """build_sample cannot proceed; the message says what is needed."""


@dataclass
class LabelLag:
    """How label maturity was established."""
    days: int
    basis: str                  # "measured" | "assumed"
    stated: str = ""            # what the caller said ("final", "45", "policy")
    approved_by: str = ""       # person | agent (assumptions only)
    policy_note: str = ""       # the Knowledge Pack policy consulted, if any
    detail: dict = field(default_factory=dict)


@dataclass
class _Labelling:
    """What CertificateBuilder.with_labelling reads."""
    label_column: str
    n_fraud: int
    n_legit: int
    n_missing: int
    entities_without_report: int = 0
    unresolvable_window_entities: list = field(default_factory=list)
    n_via_default_rule: int = 0

    def to_dict(self) -> dict:
        return {"label_column": self.label_column, "n_fraud": self.n_fraud,
                "n_legit": self.n_legit, "n_missing_label": self.n_missing,
                "rule": "labels taken as given in the confirmed label column"}


@dataclass
class BuildResult:
    train: Any
    test: Any
    certificate: Any
    lines: list[str]
    meta: dict


# -- time ------------------------------------------------------------------- #

def _is_numeric(s) -> bool:
    import pandas as pd
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def to_times(s, unit: str = ""):
    """(datetime Series, relative?) for an event/label time column."""
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
    elif _is_numeric(s):
        u = (unit or "").strip().lower()
        if u not in TIME_UNITS:
            raise BuildRefusal(
                f"the time column {s.name!r} is numeric, so its unit must be stated: "
                f"time_unit is one of {', '.join(TIME_UNITS)}. Nothing was built.")
        days = pd.to_numeric(s, errors="coerce").astype(float) * TIME_UNITS[u]
        return pd.Timestamp(EPOCH) + pd.to_timedelta(days, unit="D"), True
    else:
        out = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    present = s.notna() & (s.astype(str).str.strip() != "")
    if present.any() and out[present].notna().mean() < 0.9:
        raise BuildRefusal(f"fewer than 90% of {s.name!r} parse as times. Nothing was built.")
    return out, False


# -- label lag -------------------------------------------------------------- #

def measured_lag(ts, lt, label, policy=None) -> LabelLag:
    import pandas as pd
    both = ts.notna() & lt.notna() & (label == 1)
    lag = (lt[both] - ts[both]).dt.total_seconds() / 86400.0
    if len(lag) < 5:
        raise BuildRefusal(
            f"only {len(lag)} fraud row(s) have both an event time and a label time, so the "
            "label lag cannot be measured. Nothing was built.")
    p50, p95, p99 = (float(lag.quantile(q)) for q in (0.5, 0.95, 0.99))
    negative = int((lag < 0).sum())
    days = max(0, math.ceil(p95))
    note = ""
    if policy is not None:
        note = f"your Knowledge Pack policy: {policy.note()}"
    return LabelLag(days, "measured", policy_note=note, detail={
        "n_measured": int(len(lag)), "p50_days": round(p50, 2), "p95_days": round(p95, 2),
        "p99_days": round(p99, 2), "negative_lags": negative,
        "policy_days": getattr(policy, "days", None), "rule": "maturity = ceil(p95 lag)",
        "observed_max": str(pd.Timestamp(lt.max()))})


def parse_assumption(stated: str, policy=None) -> int:
    s = (stated or "").strip().lower()
    if s in ("final", "labels are final", "0"):
        return 0
    if s == "policy":
        if policy is None:
            raise BuildRefusal("label_lag='policy' but your Knowledge Pack has no label_policy "
                               "that applies. Nothing was built.")
        return int(policy.days)
    s = s.removesuffix("days").removesuffix("day").strip()
    try:
        n = int(float(s))
    except ValueError:
        raise BuildRefusal(f"label_lag={stated!r} is not a number of days, 'final' or 'policy'. "
                           "Nothing was built.") from None
    if not 0 <= n <= 1000:
        raise BuildRefusal(f"label_lag must be 0-1000 days; got {n}. Nothing was built.")
    return n


# -- the build -------------------------------------------------------------- #

def pick_entity(roles, entity: str = "") -> str:
    if entity:
        if not roles.role_of(entity) or entity not in roles.columns:
            raise BuildRefusal(f"entity={entity!r} is not a column. Nothing was built.")
        return entity
    for r in ENTITY_PREFERENCE:
        cols = roles.columns_with(r)
        if cols:
            return cols[0]
    return ""


def parse_prevalence(value, natural: float) -> tuple[float, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise BuildRefusal(
            f"declare the design prevalence: the natural fraud rate of the mature training "
            f"rows is {natural:.6g} (1:{(1 - natural) / natural:.0f}). Pass "
            "design_prevalence as a fraction (e.g. 0.01 to down-sample legitimate rows to "
            "1:99) or 'natural' to keep the data's own rate. Nothing was built.")
    if isinstance(value, str) and value.strip().lower() == "natural":
        return natural, "natural (declared equal to the data's own rate)"
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise BuildRefusal(f"design_prevalence={value!r} is not a fraction or 'natural'. "
                           "Nothing was built.") from None
    if v >= 1 and v == int(v):
        v = 1.0 / (v + 1)       # "99" read as 1:99
    if not 0 < v < 1:
        raise BuildRefusal(f"design_prevalence must be strictly between 0 and 1; got {value!r}.")
    return v, "declared"


def build_sample(df, roles, lag: LabelLag, *, design_prevalence=None, split: str = "",
                 entity: str = "", time_unit: str = "", test_fraction: float = 0.2,
                 embargo_days: int = 0, seed: int = 42, name: str = "sample",
                 times=None) -> BuildResult:
    import pandas as pd

    from ledgerhound.pipeline.certificate import (
        DEFAULT_REQUIRED,
        SECTION_SPLIT,
        CertificateBuilder,
        CertificateSection,
        _worst,
    )
    from ledgerhound.pipeline.model_story import GRAIN_TRANSACTION, LabelRule, ModelStory
    from ledgerhound.pipeline.prevalence import build_population_correction, check_prevalence, measure_prevalence
    from ledgerhound.pipeline.splits import (
        MODE_ENTITY,
        MODE_GROUPED_TEMPORAL,
        MODE_TEMPORAL,
        SplitSpec,
        make_split,
        verify_split,
    )

    label_col = roles.first("label")
    y = pd.to_numeric(df[label_col], errors="coerce")
    if not set(y.dropna().unique()) <= {0, 1}:
        raise BuildRefusal(f"the label column {label_col!r} is not binary 0/1. Nothing was built.")
    time_col = roles.first("event_time")
    ent = pick_entity(roles, entity)

    mode = (split or "").strip().lower()
    if not mode:
        mode = (MODE_GROUPED_TEMPORAL if time_col and ent else
                MODE_TEMPORAL if time_col else MODE_ENTITY if ent else "")
    if mode and mode not in (MODE_TEMPORAL, MODE_ENTITY, MODE_GROUPED_TEMPORAL):
        raise BuildRefusal(f"split={split!r} is not temporal, entity or grouped_temporal. "
                           "Nothing was built.")
    if not mode:
        raise BuildRefusal("no event_time and no entity column among the confirmed roles, so "
                           "no honest split exists (a random row split would leak). Nothing "
                           "was built.")
    if mode in (MODE_TEMPORAL, MODE_GROUPED_TEMPORAL) and not time_col:
        raise BuildRefusal(f"split={mode!r} needs an event_time column. Nothing was built.")
    if mode in (MODE_ENTITY, MODE_GROUPED_TEMPORAL) and not ent:
        raise BuildRefusal(f"split={mode!r} needs an entity column. Nothing was built.")

    frame = df.copy()
    frame["_label"] = y
    ts, relative = (times if times is not None else
                    (to_times(df[time_col], time_unit) if time_col else (None, False)))
    tcol = ""
    obs = None
    if ts is not None:
        tcol = "_lh_time"
        frame[tcol] = ts
        obs = ts.max() + pd.Timedelta(seconds=1)
        if lag.basis == "measured" and lag.detail.get("observed_max"):
            obs = max(obs, pd.Timestamp(lag.detail["observed_max"]) + pd.Timedelta(seconds=1))
    frame = frame[frame["_label"].notna()]

    spec_kw: dict = dict(mode=mode, entity_keys=[ent] if ent else [], test_fraction=test_fraction,
                         seed=seed, embargo_days=int(embargo_days))
    natural_rows = frame
    if tcol:
        cutoff = obs - pd.Timedelta(days=lag.days)
        mature = frame[frame[tcol] < cutoff]
        if mature.empty:
            raise BuildRefusal(
                f"no row is older than the {lag.days}-day label maturity horizon "
                f"(data ends {obs.date()}), so no test row would have mature labels. "
                "Nothing was built.")
        natural_rows = mature
        test_start = mature[tcol].quantile(1 - test_fraction)
        spec_kw.update(timestamp_column=tcol, test_start=test_start,
                       train_end=test_start - pd.Timedelta(days=int(embargo_days)),
                       observation_date=obs, exclude_immature_from_test=True)
    else:
        spec_kw.update(exclude_immature_from_test=False)

    n1 = int((natural_rows["_label"] == 1).sum())
    natural = n1 / max(len(natural_rows), 1)
    if n1 == 0:
        raise BuildRefusal("there is no fraud among the label-mature rows. Nothing was built.")
    dp, dp_how = parse_prevalence(design_prevalence, natural)

    story = ModelStory(
        name=name, label_grain=GRAIN_TRANSACTION, fetch_keys=[ent or label_col],
        label_rules=[LabelRule("as_labelled", "reported_only",
                               notes=f"labels taken as given in {label_col}")],
        label_maturity_days=lag.days, grains_emitted=[GRAIN_TRANSACTION],
        design_prevalence=dp, design_prevalence_grain=GRAIN_TRANSACTION,
        population_prevalence=natural if 0 < natural < 1 else None,
        population_prevalence_grain=GRAIN_TRANSACTION,
        population_prevalence_source=("measured on the label-mature rows of this data"
                                      if 0 < natural < 1 else ""))
    spec = SplitSpec(**spec_kw)
    res = make_split(frame, spec, story)
    train, test = res.train, res.test

    # down-sample legitimate TRAIN rows to the design prevalence; test keeps its rate
    n_dropped = 0
    tr_rate = float((train["_label"] == 1).mean()) if len(train) else 0.0
    if len(train) and dp > tr_rate > 0:
        fraud = train[train["_label"] == 1]
        legit = train[train["_label"] == 0]
        keep = min(len(legit), int(round(len(fraud) * (1 - dp) / dp)))
        n_dropped = len(legit) - keep
        train = pd.concat([fraud, legit.sample(n=keep, random_state=seed)]).sort_index()
        res.train = train

    res.findings.extend(verify_split(
        train, test, entity_keys=[ent] if ent and mode != MODE_TEMPORAL else None,
        timestamp_column=tcol if mode != MODE_ENTITY else "",
        label_column="_label" if not n_dropped else "_no_label_drift_check",
        story=story if tcol else None, observation_date=obs if tcol else None))

    meas = measure_prevalence(train, grain=GRAIN_TRANSACTION, label_column="_label",
                              mature_column=None)
    meas.mature_only = bool(tcol)       # train ends before the maturity cutoff by construction
    check = check_prevalence(meas, story)
    correction = None
    if story.population_prevalence and meas.prevalence:
        try:
            correction = build_population_correction(meas, story)
        except Exception:  # noqa: BLE001 -- recorded as not checked instead
            correction = None

    required = tuple(DEFAULT_REQUIRED) + (SECTION_SPLIT, SECTION_MATURITY)
    b = CertificateBuilder(story, grain=GRAIN_TRANSACTION, required_sections=required)
    b.with_labelling(_Labelling(label_col, int((frame["_label"] == 1).sum()),
                                int((frame["_label"] == 0).sum()),
                                int(y.isna().sum())))
    b.with_prevalence(check, correction)
    b.with_split(res)
    notes = [f"design prevalence: {dp_how}"]
    if n_dropped:
        notes.append(f"{n_dropped:,} legitimate train rows were down-sampled away (seed {seed}); "
                     "the test split keeps the natural rate")
    if relative:
        notes.append(f"event time is relative ({time_col} in {time_unit}); dates in this "
                     f"certificate count from {EPOCH} and are not calendar dates")
    b.with_notes("; ".join(notes))
    cert = b.build()
    cert.sections[SECTION_MATURITY] = maturity_section(lag, CertificateSection, _worst)

    meta = {"label": label_col, "event_time": time_col, "entity": ent, "split": mode,
            "maturity_days": lag.days, "maturity_basis": lag.basis,
            "design_prevalence": dp, "natural_prevalence": natural,
            "n_train": len(train), "n_test": len(test), "n_dropped": n_dropped}
    drop = [c for c in ("_label", "_lh_time") if c in train.columns]
    return BuildResult(train.drop(columns=drop), test.drop(columns=drop), cert, [], meta)


def maturity_section(lag: LabelLag, Section, worst):
    findings = []
    if lag.basis == "measured":
        d = lag.detail
        if d.get("negative_lags"):
            findings.append(Finding(
                "NEGATIVE_LABEL_LAG", SEV_WARNING,
                f"{d['negative_lags']} fraud row(s) were labelled before the event happened -- "
                "check the label_time column", {}))
        pol = d.get("policy_days")
        if pol is not None and pol < d["p95_days"]:
            findings.append(Finding(
                "POLICY_SHORTER_THAN_DATA", SEV_WARNING,
                f"your Knowledge Pack's label policy says {pol} days but 5% of labels on "
                f"this data arrived later than {d['p95_days']:.0f} days; the data's figure "
                "was used", {}))
        not_checked = ["labels arriving after the data was extracted (beyond the p95 lag)"]
    else:
        who = "a person" if lag.approved_by == "person" else "the agent, not verified by a person"
        findings.append(Finding(
            "LABEL_LAG_ASSUMED", SEV_WARNING,
            f"the data has no label-time column; a {lag.days}-day label lag "
            f"({lag.stated!r}) was ASSUMED, stated by {who}. This is an assumption, not a "
            "check", {"approved_by": lag.approved_by}))
        not_checked = ["label maturity: assumed, not measured (no label-time column)"]
    status = worst(findings) if findings else SEV_PASS
    return Section(name=SECTION_MATURITY, status=status, findings=findings,
                   not_checked=not_checked,
                   detail={"days": lag.days, "basis": lag.basis, "stated": lag.stated,
                           "approved_by": lag.approved_by, "policy": lag.policy_note,
                           **lag.detail})


def summary_lines(result: BuildResult, paths: Optional[dict] = None) -> str:
    m, c = result.meta, result.certificate
    gate = "CERTIFIED" if c.can_certify else "NOT CERTIFIED"
    out = [f"sample built -- {gate} (certificate {c.content_hash()[:16]})",
           f"  split={m['split']}" + (f", entity={m['entity']}" if m["entity"] else "") +
           (f", time={m['event_time']}" if m["event_time"] else ""),
           f"  train={m['n_train']:,} rows, test={m['n_test']:,} rows"
           + (f" ({m['n_dropped']:,} legitimate train rows down-sampled away)"
              if m["n_dropped"] else ""),
           f"  label maturity: {m['maturity_days']} days, {m['maturity_basis']}",
           f"  prevalence: design {m['design_prevalence']:.6g}, natural "
           f"{m['natural_prevalence']:.6g}"]
    for name in sorted(c.sections):
        sec = c.sections[name]
        out.append(f"  [{sec.status:<12}] {name}")
        out.extend(f"      {f.severity}: {f.message}" for f in sec.findings
                   if f.severity != SEV_PASS)
    if not c.can_certify:
        out.append("  BLOCKED BY:")
        out.extend(f"    - {r}" for r in c.blocking_reasons)
    lim = c.all_not_checked
    if lim:
        out.append("  NOT CHECKED (a pass does not cover these):")
        out.extend(f"    - {n}" for n in lim)
    for k, v in (paths or {}).items():
        out.append(f"  {k}: {v}")
    return "\n".join(out)
