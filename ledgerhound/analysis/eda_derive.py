"""
ledgerhound.analysis.eda_derive
============================

Thin, SAFE, *exploratory* feature derivation for the EDA cycle (AUD-144).

Raw columns often hide a signal: you cannot see "fraud concentrates in young
accounts" unless ``account_age_days`` exists, which means deriving it from an
open-date. This module computes a small set of safe per-row transforms so
step 6 can compare them.

IMPORTANT -- this is NOT modelling feature engineering.
-------------------------------------------------------
These derivations are exploratory and **not leakage-validated**: they are
computed as-of the row with no point-in-time guarantee. They exist to *reveal
patterns* and seed a candidate-feature list for the feature-engineering team,
who must re-implement anything useful with proper leakage / point-in-time
rigor. Every derived column is reported tagged as such.

Supported kinds (deliberately few):
    date_diff  : whole days between two date columns      (to - from)
    ratio      : num / den   (div-by-zero -> NaN)
    log        : natural log1p of a non-negative column
    hour       : hour-of-day (0-23) from a timestamp column

No encodings, rolling windows, target stats, or aggregations -- those belong
to the FE team. Failures are non-fatal: a bad spec is skipped with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KINDS = {"date_diff", "ratio", "log", "hour"}


@dataclass
class Derivation:
    """One exploratory derived column."""

    name: str
    kind: str
    # kind-specific column refs:
    col: Optional[str] = None          # log / hour
    num: Optional[str] = None          # ratio
    den: Optional[str] = None          # ratio
    from_col: Optional[str] = None     # date_diff
    to_col: Optional[str] = None       # date_diff

    @classmethod
    def from_dict(cls, d: dict) -> "Derivation":
        return cls(
            name=d["name"],
            kind=d["kind"],
            col=d.get("col"),
            num=d.get("num"),
            den=d.get("den"),
            from_col=d.get("from") or d.get("from_col"),
            to_col=d.get("to") or d.get("to_col"),
        )


def load_derivations(path: Optional[str]) -> list[Derivation]:
    """Load a list of derivations from a JSON or YAML file."""
    if not path:
        return []
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text) or []
    else:
        import json
        raw = json.loads(text)
    return [Derivation.from_dict(d) for d in raw]


def apply_derivations(
    df: Any, derivations: list[Derivation]
) -> tuple[Any, list[str], list[str]]:
    """Add derived columns to a copy of ``df``.

    Returns ``(new_df, added_names, warnings)``. A derivation that references a
    missing column or errors is skipped (recorded in ``warnings``); it never
    raises, so a partial spec still produces a usable frame.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    added: list[str] = []
    warnings: list[str] = []

    for d in derivations:
        if d.kind not in _KINDS:
            warnings.append(f"{d.name}: unknown kind '{d.kind}' (skipped)")
            continue
        try:
            series = _compute(out, d, np, pd)
        except Exception as exc:  # noqa: BLE001 - exploratory, never fatal
            warnings.append(f"{d.name}: {exc} (skipped)")
            continue
        if series is None:
            warnings.append(f"{d.name}: missing source column(s) (skipped)")
            continue
        out[d.name] = series
        added.append(d.name)

    return out, added, warnings


def _compute(df: Any, d: Derivation, np, pd):
    cols = set(df.columns)
    if d.kind == "date_diff":
        if not (d.from_col in cols and d.to_col in cols):
            return None
        a = pd.to_datetime(df[d.from_col], errors="coerce")
        b = pd.to_datetime(df[d.to_col], errors="coerce")
        return (b - a).dt.days
    if d.kind == "ratio":
        if not (d.num in cols and d.den in cols):
            return None
        den = pd.to_numeric(df[d.den], errors="coerce")
        # Near-zero denominators (not just exact zero) produce extreme,
        # unstable ratios -- den=0.001 on a column whose typical scale is
        # ~100 yields a ~100,000x value that isn't NaN and passes straight
        # through. Real pilot feedback (V3-032): "Ratio-based features may
        # generate extreme values when denominators are close to zero."
        # Threshold is relative to this column's own scale (1% of the
        # median absolute non-zero value), not a fixed epsilon -- "close to
        # zero" means something different for a rupee-amount column than a
        # count column.
        abs_den = den.abs()
        nonzero = abs_den[abs_den > 0]
        threshold = max(float(nonzero.median()) * 0.01, 1e-9) if len(nonzero) else 1e-9
        den_safe = den.where(abs_den >= threshold, np.nan)
        return df[d.num] / den_safe
    if d.kind == "log":
        if d.col not in cols:
            return None
        x = pd.to_numeric(df[d.col], errors="coerce")
        return np.log1p(x.where(x >= 0))
    if d.kind == "hour":
        if d.col not in cols:
            return None
        return pd.to_datetime(df[d.col], errors="coerce").dt.hour
    return None
