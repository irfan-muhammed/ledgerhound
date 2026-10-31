"""
sentinel.pipeline.labeller
===========================

**Apply label rules to fetched entity history** — the step that turns
``history_fetch``'s output into a labelled dataset.

The problem
-----------
We fetch history by *entity identity*: every transaction of every payee that
appears in a fraud report. Only the originally-reported transactions are
known-fraud. The other ~500k rows arrive **unlabelled**, and what they should
become depends entirely on the fraud typology:

* **mule** — the account exists to launder money and was never legitimate,
  so its whole history is fraud-side (``all_history``).
* **account takeover** — a real customer's account, compromised on a known
  date. Pre-compromise rows are genuine customer behaviour; only the
  post-compromise window is fraud (``window``).
* **phishing-induced payment** — the payer is a victim acting under
  deception. Only the specific reported transaction is fraud; the fraud
  entity is the payee receiving it (``reported_only``).

Same data, ~10x different fraud count depending on the rule. So "550k fraud
rows" is not a defined number until this is answered, and every downstream
rate, IV and KS figure moves with it.

Why the label is tri-state, not binary
--------------------------------------
This is the load-bearing design decision. Under ``reported_only``, a fraud
payee's other 500 transactions are **not known-legitimate** — they are
*unknown*. Some are probably unreported fraud. Collapsing them to ``0``
asserts an absence of fraud we never observed, which is precisely the
overclaim `NOVELTY.md` §5 and V4-025 forbid.

So:

===========  =====================================================
``1``        fraud — established by the rule
``0``        observed-legitimate — the rule genuinely establishes it
``<NA>``     unlabelled — we do not know, and will not pretend
===========  =====================================================

A ``0`` is only ever produced where a rule *positively establishes*
legitimacy (e.g. pre-compromise rows under ``window``, where the real
customer was demonstrably in control). Everywhere else the answer is
``<NA>``.

Label maturity folds in here
----------------------------
Fraud is reported 30–90 days late. A transaction from last week that has not
been reported is not "legitimate" — it is *too recent to know*. So when an
``observation_date`` is supplied, any row inside
``story.label_maturity_days`` of it has its ``0`` demoted to ``<NA>``. Fraud
labels are never demoted: a positive report is still positive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from sentinel.pipeline.model_story import LabelRule, ModelStory

log = logging.getLogger(__name__)

SCHEMA = "labelling/v1"

LABEL_COLUMN = "_label"
RULE_COLUMN = "_label_rule"
TYPOLOGY_COLUMN = "_typology"
MATURE_COLUMN = "_label_mature"
# True where the row was labelled through the story's DECLARED default rule
# rather than a typology-specific one -- an explicit fallback is defensible
# only because it stays visible in the output.
DEFAULT_RULE_COLUMN = "_label_via_default"

# How wide a fraud claim each scope makes, for resolving an entity that
# appears under conflicting typologies. Widest wins, and the conflict is
# recorded rather than silently absorbed.
_SCOPE_BREADTH = {"reported_only": 0, "window": 1, "all_history": 2}


class LabellingError(ValueError):
    """The labelling cannot be performed correctly as specified."""


@dataclass
class LabellingConfig:
    """Column bindings for the two input frames.

    Parameters
    ----------
    entity_keys
        Columns identifying an entity, present in BOTH frames and in the same
        order (normally ``ModelStory.fetch_keys``).
    history_timestamp
        Transaction time column in the history frame.
    typology_column
        Column in the reports frame naming the fraud type, used to select a
        :class:`~sentinel.pipeline.model_story.LabelRule`.
    report_timestamp
        When the fraud was reported. Required for ``window`` scope: it bounds
        the fraud interval.
    txn_id_column
        Transaction identifier, present in BOTH frames. Required for
        ``reported_only`` scope -- without it there is no way to say *which*
        transaction was reported.
    observation_date
        "Now" for maturity purposes. When set, recent rows have a ``0``
        demoted to ``<NA>`` (see module docstring).
    """

    entity_keys: list[str]
    history_timestamp: str
    typology_column: str
    report_timestamp: str = ""
    txn_id_column: str = ""
    observation_date: Any = None

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.entity_keys:
            problems.append("entity_keys must not be empty")
        if not self.history_timestamp.strip():
            problems.append("history_timestamp must not be empty")
        if not self.typology_column.strip():
            problems.append(
                "typology_column must not be empty -- without it no rule can "
                "be selected per entity"
            )
        return problems


@dataclass
class LabellingResult:
    """Labelled history plus an account of what was decided and what wasn't."""

    frame: Any
    n_fraud: int = 0
    n_legit: int = 0
    n_unlabelled: int = 0
    n_immature_demoted: int = 0
    n_via_default_rule: int = 0
    typologies_via_default: list = field(default_factory=list)
    per_typology: dict = field(default_factory=dict)
    entities_without_report: int = 0
    entities_with_conflicting_typologies: list = field(default_factory=list)
    unresolvable_window_entities: list = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def fraud_rate_over_labelled(self) -> Optional[float]:
        """Fraud rate over LABELLED rows only.

        Deliberately not over all rows: unlabelled rows are not denominators
        for a rate. Returns None when nothing is labelled, rather than 0.0.
        """
        labelled = self.n_fraud + self.n_legit
        return (self.n_fraud / labelled) if labelled else None

    def summary(self) -> str:
        lines = [
            f"Labelling ({SCHEMA}) -- {self.n_rows:,} history rows",
            f"  fraud (1)       : {self.n_fraud:,}",
            f"  legitimate (0)  : {self.n_legit:,}",
            f"  unlabelled (NA) : {self.n_unlabelled:,}",
        ]
        rate = self.fraud_rate_over_labelled
        lines.append(
            "  fraud rate over LABELLED rows: "
            + (f"{rate:.6f}" if rate is not None else "n/a (nothing labelled)")
        )
        if self.n_immature_demoted:
            lines.append(
                f"  demoted 0 -> NA as label-immature: "
                f"{self.n_immature_demoted:,}"
            )
        for typ, counts in sorted(self.per_typology.items()):
            lines.append(
                f"  [{typ}] rule={counts['rule']} entities={counts['entities']} "
                f"fraud={counts['fraud']:,} legit={counts['legit']:,} "
                f"unlabelled={counts['unlabelled']:,}"
            )
        if self.n_via_default_rule:
            lines.append(
                f"  {self.n_via_default_rule:,} rows labelled via the DECLARED "
                f"DEFAULT rule, covering typologies "
                f"{self.typologies_via_default} -- no typology-specific rule "
                f"existed for these")
        if self.entities_without_report:
            lines.append(
                f"  WARNING: {self.entities_without_report} entities in history "
                f"had no matching report -- left unlabelled"
            )
        if self.entities_with_conflicting_typologies:
            lines.append(
                f"  WARNING: {len(self.entities_with_conflicting_typologies)} "
                f"entities had conflicting typologies; widest fraud scope applied"
            )
        if self.unresolvable_window_entities:
            lines.append(
                f"  WARNING: {len(self.unresolvable_window_entities)} 'window' "
                f"entities lacked a usable compromise/report timestamp -- "
                f"left unlabelled rather than guessed"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "n_legit": self.n_legit,
            "n_unlabelled": self.n_unlabelled,
            "n_immature_demoted": self.n_immature_demoted,
            "n_via_default_rule": self.n_via_default_rule,
            "typologies_via_default": list(self.typologies_via_default),
            "fraud_rate_over_labelled": self.fraud_rate_over_labelled,
            "per_typology": self.per_typology,
            "entities_without_report": self.entities_without_report,
            "entities_with_conflicting_typologies":
                len(self.entities_with_conflicting_typologies),
            "unresolvable_window_entities":
                len(self.unresolvable_window_entities),
        }


def _require_columns(frame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c and c not in frame.columns]
    if missing:
        raise LabellingError(
            f"{where} is missing required columns {missing}; "
            f"available: {list(frame.columns)[:20]}"
        )


def _resolve_rules(reports, story: ModelStory, cfg: LabellingConfig):
    """Map each entity to the LabelRule its typology selects.

    Returns (entity_tuple -> (LabelRule, typology)), plus the list of
    entities whose reports disagreed on typology.
    """
    keys = list(cfg.entity_keys)
    resolved: dict[tuple, tuple[LabelRule, str]] = {}
    conflicts: list[tuple] = []

    seen_typologies: dict[tuple, set[str]] = {}
    for row in reports[keys + [cfg.typology_column]].itertuples(
            index=False, name=None):
        ent, typ = tuple(row[:len(keys)]), str(row[len(keys)])
        seen_typologies.setdefault(ent, set()).add(typ)

    unknown: set[str] = set()
    default = story.default_label_rule
    for ent, typs in seen_typologies.items():
        candidates = []
        for typ in typs:
            rule = story.rule_for(typ)
            if rule is None and default is not None:
                # An explicitly-declared fallback. The typology recorded on
                # the row stays the REAL one so the default is visible in the
                # output rather than hidden behind a placeholder name.
                rule = default
            if rule is None:
                unknown.add(typ)
                continue
            candidates.append((rule, typ))
        if not candidates:
            continue
        if len(candidates) > 1:
            conflicts.append(ent)
        # Widest fraud scope wins; ties broken deterministically by typology.
        rule, typ = max(
            candidates,
            key=lambda rt: (_SCOPE_BREADTH.get(rt[0].scope, -1), rt[1]),
        )
        resolved[ent] = (rule, typ)

    if unknown:
        raise LabellingError(
            f"reports contain typologies with no LabelRule in the model story: "
            f"{sorted(unknown)}. Add a rule for each, declare a default via "
            f"`typology.default_rule()`, or exclude those reports -- the "
            f"labeller will not guess which history rows they make fraud. "
            f"Run `typology.reconcile_typologies()` first to see the full gap "
            f"and its volume before deciding."
        )
    return resolved, conflicts


def apply_labels(
    history,
    reports,
    story: ModelStory,
    cfg: LabellingConfig,
) -> LabellingResult:
    """Label fetched history rows according to the story's per-typology rules.

    Does not mutate ``history``. Adds ``_label`` (nullable Int64, tri-state),
    ``_label_rule``, ``_typology``, and — when ``observation_date`` is set —
    ``_label_mature``.
    """
    import pandas as pd

    problems = cfg.validate()
    if problems:
        raise LabellingError("invalid LabellingConfig: " + "; ".join(problems))
    story_problems = story.validate()
    if story_problems:
        raise LabellingError(
            "model story is not valid, so labelling would be undefined: "
            + "; ".join(story_problems)
        )

    keys = list(cfg.entity_keys)
    _require_columns(history, keys + [cfg.history_timestamp], "history frame")
    _require_columns(reports, keys + [cfg.typology_column], "reports frame")

    resolved, conflicts = _resolve_rules(reports, story, cfg)
    scopes_used = {r.scope for r, _ in resolved.values()}

    if "reported_only" in scopes_used and not cfg.txn_id_column.strip():
        raise LabellingError(
            "a 'reported_only' rule applies, but txn_id_column is not set -- "
            "without a transaction identifier there is no way to say WHICH "
            "transaction was reported, and labelling the whole history would "
            "silently turn a phishing case into a mule case."
        )
    if "window" in scopes_used and not cfg.report_timestamp.strip():
        raise LabellingError(
            "a 'window' rule applies, but report_timestamp is not set -- it "
            "bounds the fraud interval, and without it the window is open-ended."
        )
    if cfg.txn_id_column:
        _require_columns(history, [cfg.txn_id_column], "history frame")
        _require_columns(reports, [cfg.txn_id_column], "reports frame")
    if cfg.report_timestamp:
        _require_columns(reports, [cfg.report_timestamp], "reports frame")

    out = history.copy()
    ts = pd.to_datetime(out[cfg.history_timestamp], errors="coerce")
    ent_series = list(out[keys].itertuples(index=False, name=None))

    label = pd.Series(pd.NA, index=out.index, dtype="Int64")
    rule_col = pd.Series(pd.NA, index=out.index, dtype="object")
    typ_col = pd.Series(pd.NA, index=out.index, dtype="object")

    # Reported transaction ids per entity, for reported_only.
    reported_ids: dict[tuple, set] = {}
    if cfg.txn_id_column:
        for row in reports[keys + [cfg.txn_id_column]].itertuples(
                index=False, name=None):
            reported_ids.setdefault(tuple(row[:len(keys)]), set()).add(
                row[len(keys)])

    # Per-entity report/compromise timestamps, for window.
    report_ts: dict[tuple, Any] = {}
    if cfg.report_timestamp:
        for row in reports[keys + [cfg.report_timestamp]].itertuples(
                index=False, name=None):
            ent = tuple(row[:len(keys)])
            val = pd.to_datetime(row[len(keys)], errors="coerce")
            # Earliest report bounds the window most conservatively.
            if ent not in report_ts or (pd.notna(val) and val < report_ts[ent]):
                report_ts[ent] = val

    compromise_ts: dict[tuple, Any] = {}
    window_fields = {r.window_start_field for r, _ in resolved.values()
                     if r.scope == "window" and r.window_start_field}
    for wf in window_fields:
        _require_columns(reports, [wf], f"reports frame (window_start_field {wf!r})")
        for row in reports[keys + [wf]].itertuples(index=False, name=None):
            ent = tuple(row[:len(keys)])
            val = pd.to_datetime(row[len(keys)], errors="coerce")
            if ent not in compromise_ts or (
                    pd.notna(val) and val < compromise_ts[ent]):
                compromise_ts[ent] = val

    # Index rows by entity so each rule is applied to its own slice.
    rows_by_entity: dict[tuple, list] = {}
    for pos, ent in enumerate(ent_series):
        rows_by_entity.setdefault(ent, []).append(out.index[pos])

    no_report = 0
    default = story.default_label_rule
    via_default_idx: list = []
    defaulted_typologies: set = set()
    unresolvable_window: list[tuple] = []
    per_typology: dict[str, dict] = {}

    for ent, idx in rows_by_entity.items():
        got = resolved.get(ent)
        if got is None:
            no_report += 1
            continue
        rule, typ = got
        bucket = per_typology.setdefault(
            typ, {"rule": rule.scope, "entities": 0,
                  "fraud": 0, "legit": 0, "unlabelled": 0})
        bucket["entities"] += 1
        rule_col.loc[idx] = rule.scope
        typ_col.loc[idx] = typ
        if default is not None and rule is default:
            via_default_idx.extend(idx)
            defaulted_typologies.add(typ)

        if rule.scope == "all_history":
            # The account was never legitimate: everything is fraud-side.
            label.loc[idx] = 1

        elif rule.scope == "reported_only":
            # Only the reported transactions. Everything else is UNKNOWN --
            # not legitimate. This entity is a known fraud actor, so its other
            # activity is exactly what we cannot vouch for.
            ids = reported_ids.get(ent, set())
            is_reported = out.loc[idx, cfg.txn_id_column].isin(ids)
            label.loc[[i for i in idx if is_reported.loc[i]]] = 1

        elif rule.scope == "window":
            start = compromise_ts.get(ent)
            end = report_ts.get(ent)
            if pd.isna(start) or pd.isna(end):
                unresolvable_window.append(ent)
                continue
            row_ts = ts.loc[idx]
            in_window = (row_ts >= start) & (row_ts < end)
            before = row_ts < start
            label.loc[[i for i in idx if in_window.loc[i]]] = 1
            # Pre-compromise the real customer was demonstrably in control,
            # so these are positively established as legitimate. Rows at or
            # after the report are left unknown: the account may be frozen,
            # remediated, or reopened.
            label.loc[[i for i in idx if before.loc[i]]] = 0

    out[LABEL_COLUMN] = label
    out[RULE_COLUMN] = rule_col
    out[TYPOLOGY_COLUMN] = typ_col
    out[DEFAULT_RULE_COLUMN] = out.index.isin(via_default_idx)
    if via_default_idx:
        log.warning(
            'labeller: %d rows across typologies %s were labelled by the '
            'DECLARED DEFAULT rule (%s), not a typology-specific one',
            len(via_default_idx), sorted(defaulted_typologies),
            default.scope if default else '?')

    # Maturity: a recent unreported transaction is "too new to know", not
    # legitimate. Demote 0 -> NA inside the maturity horizon; never demote 1.
    n_demoted = 0
    if cfg.observation_date is not None:
        horizon = pd.Timestamp(cfg.observation_date) - pd.Timedelta(
            days=story.label_maturity_days)
        mature = ts < horizon
        out[MATURE_COLUMN] = mature
        demote = (~mature) & (out[LABEL_COLUMN] == 0)
        n_demoted = int(demote.sum())
        if n_demoted:
            out.loc[demote, LABEL_COLUMN] = pd.NA
            log.warning(
                "labeller: demoted %d rows from 0 to NA -- inside the %d-day "
                "label-maturity horizon, so 'not reported' does not mean "
                "'not fraud'", n_demoted, story.label_maturity_days,
            )

    # Recount per typology after any demotion so the report matches the frame.
    for typ, bucket in per_typology.items():
        sel = out[TYPOLOGY_COLUMN] == typ
        lab = out.loc[sel, LABEL_COLUMN]
        bucket["fraud"] = int((lab == 1).sum())
        bucket["legit"] = int((lab == 0).sum())
        bucket["unlabelled"] = int(lab.isna().sum())

    return LabellingResult(
        frame=out,
        n_fraud=int((out[LABEL_COLUMN] == 1).sum()),
        n_legit=int((out[LABEL_COLUMN] == 0).sum()),
        n_unlabelled=int(out[LABEL_COLUMN].isna().sum()),
        n_immature_demoted=n_demoted,
        n_via_default_rule=len(via_default_idx),
        typologies_via_default=sorted(defaulted_typologies),
        per_typology=per_typology,
        entities_without_report=no_report,
        entities_with_conflicting_typologies=conflicts,
        unresolvable_window_entities=unresolvable_window,
    )
