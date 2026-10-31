"""
sentinel.pipeline.splits
=========================

**Train / test / validation splits that hold three properties at once**
(V4-022).

Each is separately well known; the interesting part is that they conflict,
and that "point-in-time correct" — the phrase people reach for — only covers
the first.

1. Point-in-time correct
------------------------
No feature may see data from after its own row's timestamp, and no training
row may sit after the test boundary. Feature stores commoditised this; it is
table stakes, not a differentiator.

2. Entity-disjoint
------------------
An account appearing in **both** train and test leaks. The model memorises
the entity rather than learning behaviour, and test performance is inflated
in a way no temporal check detects — the timestamps are perfectly ordered.
This matters especially here because the fraud side is drawn by entity and
carries every transaction that entity ever made.

**These two are in tension.** A pure temporal split cuts a busy entity in
half; a pure entity split lets a train row sit after a test row. Satisfying
both means discarding the rows that fall on the wrong side of one rule while
satisfying the other — a real cost, reported rather than hidden
(``n_discarded``).

3. Label-maturity aware
-----------------------
The one that silently biases results. Fraud is reported 30–90 days late, so
a temporally-later test split has systematically **less mature** labels than
train. Fraud in the test window that has not been reported yet reads as
legitimate, producing false negatives counted as true negatives — measured
precision comes out **higher** than reality, exactly on the most recent data
people trust most.

So the test window is closed at ``observation_date − label_maturity_days``,
not at "now". Rows past that are excluded by default and always counted.

Embargo
-------
Rolling-window features straddle a hard boundary: a 30-day velocity feature
on a test row at ``T+1`` reads transactions from before ``T``. ``embargo_days``
opens a gap between train end and test start so the windows cannot overlap.
Set it to at least the widest feature window in use. (Purged/embargoed
splitting, cf. López de Prado.)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from sentinel.pipeline.appropriateness import (
    SEV_FAIL,
    SEV_PASS,
    SEV_WARNING,
    Finding,
)
from sentinel.pipeline.model_story import ModelStory

log = logging.getLogger(__name__)

SCHEMA = "split/v1"

MODE_TEMPORAL = "temporal"
MODE_ENTITY = "entity"
MODE_GROUPED_TEMPORAL = "grouped_temporal"
_MODES = {MODE_TEMPORAL, MODE_ENTITY, MODE_GROUPED_TEMPORAL}


class SplitError(ValueError):
    """The split cannot be constructed or verified as specified."""


def _entity_bucket(key: tuple, seed: int, n_buckets: int = 10_000) -> int:
    """Deterministic bucket for an entity.

    Uses a stable digest rather than ``hash()``, whose salt changes per
    process — a split that silently differs between runs would be worse than
    no split at all.
    """
    payload = (repr(key) + "|" + str(seed)).encode("utf-8")
    return int(hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8], 16) % n_buckets


@dataclass
class SplitSpec:
    """How to cut the sample, and what guarantees are being asked for."""

    mode: str
    entity_keys: list[str] = field(default_factory=list)
    timestamp_column: str = ""
    train_end: Any = None
    test_start: Any = None
    embargo_days: int = 0
    test_fraction: float = 0.2
    seed: int = 42
    observation_date: Any = None
    exclude_immature_from_test: bool = True

    def validate(self, story: Optional[ModelStory] = None) -> list[str]:
        problems: list[str] = []
        if self.mode not in _MODES:
            problems.append(f"mode must be one of {sorted(_MODES)}; got {self.mode!r}")

        needs_time = self.mode in (MODE_TEMPORAL, MODE_GROUPED_TEMPORAL)
        needs_entity = self.mode in (MODE_ENTITY, MODE_GROUPED_TEMPORAL)

        if needs_time:
            if not self.timestamp_column.strip():
                problems.append(f"mode {self.mode!r} requires timestamp_column")
            if self.train_end is None or self.test_start is None:
                problems.append(
                    f"mode {self.mode!r} requires train_end and test_start")
        if needs_entity and not self.entity_keys:
            problems.append(f"mode {self.mode!r} requires entity_keys")

        if self.train_end is not None and self.test_start is not None:
            import pandas as pd
            te, ts_ = pd.Timestamp(self.train_end), pd.Timestamp(self.test_start)
            if ts_ < te:
                problems.append(
                    f"test_start {ts_} is before train_end {te} -- the test "
                    f"period would precede training, inverting time")
            gap = (ts_ - te).total_seconds() / 86400.0
            if self.embargo_days and gap < self.embargo_days:
                problems.append(
                    f"gap between train_end and test_start is {gap:.1f}d but "
                    f"embargo_days={self.embargo_days} -- rolling-window "
                    f"features would span the boundary")

        if not (0.0 < self.test_fraction < 1.0):
            problems.append(
                f"test_fraction must be in (0,1); got {self.test_fraction}")
        if self.embargo_days < 0:
            problems.append(f"embargo_days must be >= 0; got {self.embargo_days}")

        if self.exclude_immature_from_test and self.observation_date is None:
            problems.append(
                "exclude_immature_from_test is set but observation_date is "
                "None -- maturity cannot be determined without it, and "
                "silently skipping the exclusion would bias test metrics upward")
        return problems

    def maturity_cutoff(self, story: ModelStory):
        """Latest timestamp whose labels can be considered mature."""
        if self.observation_date is None:
            return None
        import pandas as pd
        return pd.Timestamp(self.observation_date) - pd.Timedelta(
            days=story.label_maturity_days)


@dataclass
class SplitResult:
    """The split, plus an account of what it cost and what it guarantees."""

    train: Any
    test: Any
    spec: SplitSpec
    n_input: int = 0
    n_discarded: int = 0
    n_immature_excluded: int = 0
    n_embargoed: int = 0
    findings: list[Finding] = field(default_factory=list)
    guarantees: list[str] = field(default_factory=list)
    not_guaranteed: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(f.severity == SEV_FAIL for f in self.findings):
            return SEV_FAIL
        if any(f.severity == SEV_WARNING for f in self.findings):
            return SEV_WARNING
        return SEV_PASS

    @property
    def is_usable(self) -> bool:
        return self.status != SEV_FAIL and len(self.train) > 0 and len(self.test) > 0

    def summary(self) -> str:
        lines = [
            f"Split ({SCHEMA}) mode={self.spec.mode} -- "
            f"train={len(self.train):,} test={len(self.test):,} "
            f"of {self.n_input:,} rows",
            f"  verdict: {self.status}",
        ]
        if self.n_discarded:
            lines.append(
                f"  discarded {self.n_discarded:,} rows to satisfy both "
                f"entity-disjointness and time ordering")
        if self.n_immature_excluded:
            lines.append(
                f"  excluded {self.n_immature_excluded:,} label-immature rows "
                f"from test")
        if self.n_embargoed:
            lines.append(f"  embargoed {self.n_embargoed:,} rows in the gap")
        for f in self.findings:
            lines.append(f"  [{f.severity}] {f.code}: {f.message}")
        if self.guarantees:
            lines.append("  GUARANTEES:")
            lines.extend(f"    + {g}" for g in self.guarantees)
        if self.not_guaranteed:
            lines.append("  NOT guaranteed:")
            lines.extend(f"    - {g}" for g in self.not_guaranteed)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "mode": self.spec.mode,
            "n_input": self.n_input,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "n_discarded": self.n_discarded,
            "n_immature_excluded": self.n_immature_excluded,
            "n_embargoed": self.n_embargoed,
            "status": self.status,
            "is_usable": self.is_usable,
            "findings": [f.to_dict() for f in self.findings],
            "guarantees": list(self.guarantees),
            "not_guaranteed": list(self.not_guaranteed),
        }


def _entity_tuples(frame, keys: list[str]) -> list[tuple]:
    return list(frame[keys].itertuples(index=False, name=None))


def make_split(frame, spec: SplitSpec, story: ModelStory) -> SplitResult:
    """Build a split honouring whichever guarantees ``spec.mode`` asks for."""
    import pandas as pd

    problems = spec.validate(story)
    if problems:
        raise SplitError("invalid SplitSpec: " + "; ".join(problems))
    for c in list(spec.entity_keys) + (
            [spec.timestamp_column] if spec.timestamp_column else []):
        if c not in frame.columns:
            raise SplitError(
                f"column {c!r} not in frame; available: {list(frame.columns)[:20]}")

    n_input = len(frame)
    ts = (pd.to_datetime(frame[spec.timestamp_column], errors="coerce")
          if spec.timestamp_column else None)

    train_mask = pd.Series(True, index=frame.index)
    test_mask = pd.Series(True, index=frame.index)
    guarantees: list[str] = []
    not_guaranteed: list[str] = []
    findings: list[Finding] = []

    # -- time ---------------------------------------------------------------
    n_embargoed = 0
    if spec.mode in (MODE_TEMPORAL, MODE_GROUPED_TEMPORAL):
        te, ts_start = pd.Timestamp(spec.train_end), pd.Timestamp(spec.test_start)
        train_mask &= ts < te
        test_mask &= ts >= ts_start
        n_embargoed = int(((ts >= te) & (ts < ts_start)).sum())
        guarantees.append(
            "point-in-time: every train row precedes train_end and every test "
            "row is at or after test_start")
        if spec.embargo_days:
            guarantees.append(
                f"embargo: {spec.embargo_days}d gap, so rolling-window "
                f"features cannot span the boundary")
        else:
            not_guaranteed.append(
                "no embargo -- a rolling-window feature on an early test row "
                "may read transactions from the train period")
    else:
        not_guaranteed.append(
            "no temporal ordering -- a train row may post-date a test row, so "
            "this measures generalisation across entities, not across time")

    # -- entity -------------------------------------------------------------
    if spec.mode in (MODE_ENTITY, MODE_GROUPED_TEMPORAL):
        ents = _entity_tuples(frame, spec.entity_keys)
        cut = int(spec.test_fraction * 10_000)
        buckets = pd.Series(
            [_entity_bucket(e, spec.seed) for e in ents], index=frame.index)
        in_test_group = buckets < cut
        train_mask &= ~in_test_group
        test_mask &= in_test_group
        guarantees.append(
            "entity-disjoint: no entity appears in both train and test")
    else:
        not_guaranteed.append(
            "NOT entity-disjoint -- the same account may appear in both train "
            "and test, which inflates test performance in a way no temporal "
            "check detects")

    # -- label maturity -----------------------------------------------------
    n_immature = 0
    cutoff = spec.maturity_cutoff(story)
    if cutoff is not None and ts is not None:
        immature = ts >= cutoff
        n_immature = int((test_mask & immature).sum())
        if spec.exclude_immature_from_test:
            test_mask &= ~immature
            guarantees.append(
                f"label maturity: test closes at {cutoff.date()} "
                f"(observation_date − {story.label_maturity_days}d), so test "
                f"labels have had time to arrive")
        elif n_immature:
            findings.append(Finding(
                "IMMATURE_TEST_ROWS", SEV_WARNING,
                f"{n_immature:,} test rows fall inside the "
                f"{story.label_maturity_days}-day maturity horizon. Unreported "
                f"fraud there reads as legitimate, so measured precision will "
                f"come out higher than reality.",
                {"n_immature": n_immature, "cutoff": str(cutoff.date())}))
    elif ts is not None:
        not_guaranteed.append(
            "label maturity not enforced -- test metrics may be biased upward "
            "by fraud that has not been reported yet")

    train = frame[train_mask]
    test = frame[test_mask]
    n_discarded = n_input - len(train) - len(test)

    if len(train) == 0:
        findings.append(Finding(
            "EMPTY_TRAIN", SEV_FAIL,
            "the train split is empty -- check train_end against the data's "
            "actual time range", {}))
    if len(test) == 0:
        findings.append(Finding(
            "EMPTY_TEST", SEV_FAIL,
            "the test split is empty -- check test_start and the maturity "
            "cutoff against the data's actual time range", {}))

    if spec.mode == MODE_GROUPED_TEMPORAL and n_discarded > 0.5 * n_input:
        findings.append(Finding(
            "HIGH_DISCARD", SEV_WARNING,
            f"{n_discarded:,}/{n_input:,} rows ({n_discarded / n_input:.0%}) "
            f"were discarded to satisfy entity-disjointness and time ordering "
            f"together. That is the honest cost of both guarantees, but check "
            f"the remaining sample is still large enough to model on.",
            {"n_discarded": n_discarded}))

    not_guaranteed.append(
        "unlabelled rows are carried into both splits as-is -- this does not "
        "decide whether they are usable")

    return SplitResult(
        train=train, test=test, spec=spec, n_input=n_input,
        n_discarded=n_discarded, n_immature_excluded=n_immature,
        n_embargoed=n_embargoed, findings=findings,
        guarantees=guarantees, not_guaranteed=not_guaranteed,
    )


def verify_split(
    train,
    test,
    *,
    entity_keys: Optional[list[str]] = None,
    timestamp_column: str = "",
    label_column: str = "_label",
    story: Optional[ModelStory] = None,
    observation_date: Any = None,
) -> list[Finding]:
    """Independently verify a split someone else produced.

    Same philosophy as `appropriateness.check_sample`: assume nothing about
    how it was made, check only what the two frames can falsify.
    """
    import pandas as pd

    findings: list[Finding] = []

    if len(train) == 0 or len(test) == 0:
        findings.append(Finding(
            "EMPTY_SPLIT", SEV_FAIL,
            f"train has {len(train):,} rows and test has {len(test):,} -- "
            f"an empty side cannot validate anything", {}))
        return findings

    # Entity overlap -- the leak a temporal check cannot see.
    if entity_keys:
        missing = [c for c in entity_keys
                   if c not in train.columns or c not in test.columns]
        if missing:
            findings.append(Finding(
                "MISSING_COLUMNS", SEV_FAIL,
                f"entity keys {missing} absent from one or both splits, so "
                f"entity-disjointness cannot be verified", {}))
        else:
            tr = set(_entity_tuples(train, entity_keys))
            te = set(_entity_tuples(test, entity_keys))
            shared = tr & te
            if shared:
                findings.append(Finding(
                    "ENTITY_LEAKAGE", SEV_FAIL,
                    f"{len(shared):,} entities appear in BOTH train and test "
                    f"({len(shared) / len(te):.0%} of test entities). The model "
                    f"can memorise these rather than learn behaviour, and no "
                    f"temporal check detects it because the timestamps are "
                    f"correctly ordered.",
                    {"n_shared_entities": len(shared),
                     "n_train_entities": len(tr), "n_test_entities": len(te)}))

    # Temporal ordering.
    if timestamp_column:
        if timestamp_column not in train.columns or \
                timestamp_column not in test.columns:
            findings.append(Finding(
                "MISSING_COLUMNS", SEV_FAIL,
                f"timestamp column {timestamp_column!r} absent from one or "
                f"both splits, so ordering cannot be verified", {}))
        else:
            tr_max = pd.to_datetime(train[timestamp_column], errors="coerce").max()
            te_min = pd.to_datetime(test[timestamp_column], errors="coerce").min()
            if pd.notna(tr_max) and pd.notna(te_min) and tr_max >= te_min:
                overlap_days = (tr_max - te_min).total_seconds() / 86400.0
                findings.append(Finding(
                    "TEMPORAL_OVERLAP", SEV_WARNING,
                    f"train extends to {tr_max.date()} but test starts at "
                    f"{te_min.date()} -- {overlap_days:.0f} days of overlap. "
                    f"Acceptable for a pure entity split; a leak if this was "
                    f"meant to be out-of-time.",
                    {"train_max": str(tr_max), "test_min": str(te_min),
                     "overlap_days": overlap_days}))

    # Maturity asymmetry -- the silent upward bias.
    if story is not None and observation_date is not None and timestamp_column \
            and timestamp_column in test.columns:
        cutoff = pd.Timestamp(observation_date) - pd.Timedelta(
            days=story.label_maturity_days)
        te_ts = pd.to_datetime(test[timestamp_column], errors="coerce")
        n_immature = int((te_ts >= cutoff).sum())
        if n_immature:
            findings.append(Finding(
                "IMMATURE_TEST_ROWS", SEV_WARNING,
                f"{n_immature:,}/{len(test):,} test rows sit inside the "
                f"{story.label_maturity_days}-day maturity horizon (after "
                f"{cutoff.date()}). Fraud there may not have been reported "
                f"yet, biasing measured precision upward on exactly the "
                f"recent data people trust most.",
                {"n_immature": n_immature, "cutoff": str(cutoff.date())}))

    # Prevalence drift between splits.
    if label_column in train.columns and label_column in test.columns:
        def _rate(f):
            lab = f[label_column]
            n1, n0 = int((lab == 1).sum()), int((lab == 0).sum())
            return (n1 / (n1 + n0)) if (n1 + n0) else None
        r_tr, r_te = _rate(train), _rate(test)
        if r_tr and r_te and r_tr > 0 and r_te > 0:
            fold = max(r_tr / r_te, r_te / r_tr)
            if fold >= 2.0:
                findings.append(Finding(
                    "SPLIT_PREVALENCE_DRIFT", SEV_WARNING,
                    f"fraud rate differs {fold:.1f}x between train "
                    f"({r_tr:.4g}) and test ({r_te:.4g}). Either the split is "
                    f"not random with respect to fraud, or the test period's "
                    f"labels are less mature.",
                    {"train_rate": r_tr, "test_rate": r_te, "fold": fold}))

    return findings
