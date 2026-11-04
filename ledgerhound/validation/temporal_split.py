"""
ledgerhound.validation.temporal_split
====================================

TemporalSplitAuthority -- V3-019: the ONLY way downstream fit/eval code is
meant to obtain a train/OOT split for a time-indexed fraud study. Computes
the split boundaries and label-maturity embargo gap ONCE per study and
returns an immutable `TemporalSplit` object; `require_temporal_split()` is
the typed guard a future fit/eval function calls first, so a random
K-fold / `train_test_split` split object is rejected before it silently
produces an over-optimistic (leaky) evaluation.

Why an embargo gap, not just a train/OOT cutoff
-------------------------------------------------
Fraud labels are not instantaneous -- a transaction flagged today might not
be confirmed as fraud for days or weeks (chargeback processing, investigation
lag). A naive cutoff (train = everything before date X, OOT = everything
after) lets training samples right at the boundary carry labels that were
not yet fully "matured" when the boundary was drawn, and lets the model
implicitly learn boundary-adjacent artifacts. The embargo gap is a buffer
immediately before `oot_start`: rows falling in it are dropped from BOTH
train and OOT entirely, not reassigned -- they are neither safe to train on
(label may still mature) nor safe to evaluate on (too close to the training
cutoff to be a clean out-of-time test).

Order dependency (roadmap, EPIC 5)
------------------------------------
This module must exist before V3-020 (feature selection) and V3-021
(modeler v1) -- both need an authoritative split so cross-validation folds,
feature-stability checks, and model evaluation are all computed against the
same boundaries, and so a leaky win (feature or model) can't sneak into the
knowledge base via a random split that happened to look good.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class NotATemporalSplitError(TypeError):
    """Raised when fit/eval code is handed something other than a
    TemporalSplit -- specifically, this is what blocks a random K-fold /
    train_test_split object from being used on time-indexed data."""


class EmptySplitError(ValueError):
    """Raised when the requested embargo gap consumes the entire pre-OOT
    span (or the OOT window itself is empty) -- i.e. the split authority
    would otherwise silently hand back a TemporalSplit with zero train or
    zero OOT rows. A study with a compressed time range and the default
    (or a too-large) embargo hits this; the fix is a smaller embargo_gap_days
    or more time-range coverage in the input data, not a downstream fit()
    that trains on nothing or fails with a confusing unrelated error."""


@dataclass
class TemporalSplit:
    """An immutable, already-computed train/OOT split with an embargo gap.
    Downstream fit/eval code is expected to accept ONLY this object (via
    `require_temporal_split`), never raw indices or a fold generator."""

    time_col: str
    train_end: str            # ISO timestamp -- train = time < train_end
    embargo_gap_days: float
    oot_start: str             # ISO timestamp -- OOT = time >= oot_start
    oot_end: Optional[str]
    train_index: list = field(default_factory=list)
    oot_index: list = field(default_factory=list)
    embargoed_index: list = field(default_factory=list)
    n_train: int = 0
    n_oot: int = 0
    n_embargoed: int = 0
    generated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train_df, oot_df) sliced from `df` by this split's
        index labels. `df` must share the index this split was computed on."""
        return df.loc[self.train_index], df.loc[self.oot_index]

    def summary_block(self) -> str:
        """Compact text for embedding in a future model card (V3-022)."""
        return (
            f"Temporal split on `{self.time_col}`: train < {self.train_end}, "
            f"embargo {self.embargo_gap_days:g}d, OOT >= {self.oot_start}"
            + (f" (< {self.oot_end})" if self.oot_end else "")
            + f". n_train={self.n_train:,} n_oot={self.n_oot:,} "
            f"n_embargoed={self.n_embargoed:,}."
        )


class TemporalSplitAuthority:
    """
    Derives a time axis from the data and computes a single train/OOT split
    with a label-maturity embargo gap.

    Parameters
    ----------
    oot_fraction    : Fraction of the time range reserved for OOT, measured
                      by row count quantile of `time_col` (default 0.2).
    embargo_gap_days: Buffer, in days, immediately before `oot_start`.
                      Rows in this window are dropped from both train and
                      OOT (default 7.0).
    """

    def __init__(self, oot_fraction: float = 0.2, embargo_gap_days: float = 7.0) -> None:
        if not (0.0 < oot_fraction < 1.0):
            raise ValueError(f"oot_fraction must be in (0, 1), got {oot_fraction}")
        if embargo_gap_days < 0:
            raise ValueError(f"embargo_gap_days must be >= 0, got {embargo_gap_days}")
        self.oot_fraction = oot_fraction
        self.embargo_gap_days = embargo_gap_days

    def split(self, df: pd.DataFrame, time_col: str) -> TemporalSplit:
        if time_col not in df.columns:
            raise KeyError(f"time_col '{time_col}' not found in DataFrame columns")

        times = pd.to_datetime(df[time_col])
        oot_start = times.quantile(1.0 - self.oot_fraction)
        embargo = pd.Timedelta(days=self.embargo_gap_days)
        train_end = oot_start - embargo

        train_mask = times < train_end
        oot_mask = times >= oot_start
        embargoed_mask = (~train_mask) & (~oot_mask)

        train_index = df.index[train_mask].tolist()
        oot_index = df.index[oot_mask].tolist()
        embargoed_index = df.index[embargoed_mask].tolist()

        if not train_index or not oot_index:
            raise EmptySplitError(
                f"Split produced n_train={len(train_index)}, n_oot={len(oot_index)} "
                f"for time_col='{time_col}' (data spans {times.min()} to {times.max()}, "
                f"embargo_gap_days={self.embargo_gap_days}). The embargo gap likely "
                f"consumes the entire pre-OOT span -- reduce embargo_gap_days or "
                f"widen the input data's time coverage."
            )

        return TemporalSplit(
            time_col=time_col,
            train_end=train_end.isoformat(),
            embargo_gap_days=self.embargo_gap_days,
            oot_start=oot_start.isoformat(),
            oot_end=times.max().isoformat() if len(times) else None,
            train_index=train_index,
            oot_index=oot_index,
            embargoed_index=embargoed_index,
            n_train=len(train_index),
            n_oot=len(oot_index),
            n_embargoed=len(embargoed_index),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


def require_temporal_split(split_obj) -> TemporalSplit:
    """Typed guard: the ONLY thing downstream fit/eval code should accept
    as a split. Raises NotATemporalSplitError for anything else -- a plain
    tuple of index arrays (what `sklearn.model_selection.train_test_split`
    or a K-fold generator produces) is rejected by construction rather than
    silently used as a random, non-temporal split."""
    if not isinstance(split_obj, TemporalSplit):
        raise NotATemporalSplitError(
            f"Expected a TemporalSplit (ledgerhound.validation.temporal_split), "
            f"got {type(split_obj).__name__}. Random K-fold / train_test_split "
            f"on time-indexed fraud data is not permitted -- construct a split "
            f"via TemporalSplitAuthority.split() instead."
        )
    return split_obj
