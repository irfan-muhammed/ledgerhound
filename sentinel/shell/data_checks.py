"""
sentinel.shell.data_checks
===========================

Mechanical data-comparability checks that fire on whatever frames actually
exist in the sandbox -- not a tool the model may choose to call.

Why here, and why mechanical
----------------------------
Observed failure: a user gave a FULL MONTH of fraud and a SINGLE DAY of
non-fraud, then asked for feature engineering. Every downstream number --
fraud rate, any velocity/recency feature, day-of-week, the class ratio
itself -- is confounded by that, and a model trained on it learns "which
file did this row come from."

Two live DeepSeek runs on exactly that fixture DID notice. But:

  * they noticed by model competence, not by any mechanism -- nothing in
    the harness guarantees it, and the user hit a run where it stayed
    silent; and
  * even when noticed, it arrived as a closing caveat AFTER a full
    production recommendation ("auto-block at $900+..."), which is
    precisely backwards for a fact that invalidates the work.

So this mirrors V3-017's post-mutation delta: compute deterministically
from the data, inject an observation the model cannot skip, and let it
react -- rather than hoping it thinks to look. It warns; it does not block
(blocking legitimate work is its own failure mode).

Deliberately narrow. It answers two questions with real arithmetic:
  1. Do two frames cover very different time windows (or windows that
     don't overlap)?
  2. Inside ONE frame, do the label's classes cover different windows?
     (the concat case -- the mismatch survives the concat and is then
     invisible)

It is NOT a general "data quality AI" and does not try to be.
"""

from __future__ import annotations

from typing import Any, Optional

# Column-name hints, checked as substrings on the lowered name.
_DATE_HINTS = ("date", "time", "timestamp", "_dt", "dt_", "created",
               "booking", "posted", "txn_ts", "event")
_LABEL_HINTS = ("is_fraud", "fraud", "label", "target", "y", "class",
                "is_mule", "flag")

# A frame's window is "much wider" than another's at this ratio.
SPAN_RATIO_ALERT = 3.0
# Don't parse more than this many rows of a string column just to bound it.
_SAMPLE_ROWS = 5000


def _is_frame(obj: Any) -> bool:
    return hasattr(obj, "columns") and hasattr(obj, "dtypes") and hasattr(obj, "__len__")


# -- per-cell scan gating (J-01/J-02) ------------------------------------------
#
# Every check in this family used to re-scan every frame after every cell,
# because `seen` only remembered HITS -- a clean 8-crore-row frame was
# re-scanned (inf per column, column-vs-label per column, date bounds) on
# every single cell for the whole session. A frame is re-scanned only when
# its shape changed, or the cell's code assigned into it (a column overwrite
# keeps the shape but can introduce an inf or a label copy).

_FP_PREFIX = "_fp:"


def frame_fingerprint(df) -> tuple:
    return (int(len(df)), tuple(str(c) for c in df.columns))


def _assign_re(name: str):
    import re
    n = re.escape(name)
    return re.compile(
        rf"(?:\b{n}\s*(?:\[[^\n]*?\]\s*)?=(?!=))"          # df = ..., df['x'] = ...
        rf"|(?:\b{n}\.(?:loc|iloc|at|iat)\b)"              # df.loc[...] = ...
        rf"|(?:\b{n}\.\w+\([^\n]*inplace\s*=\s*True)")     # df.fillna(..., inplace=True)


def code_assigns_to(code: str, name: str) -> bool:
    """Does this cell's source assign into the variable `name`?"""
    if not code or not name:
        return False
    try:
        return _assign_re(name).search(code) is not None
    except Exception:  # noqa: BLE001
        return True   # if in doubt, rescan


def frame_key(fname: str, df) -> tuple | None:
    """The ONE identity used by both the scan gate and the window cache:
    name, shape, object id, and a cheap content sentinel (first + last
    row). id() alone is a recycled address -- after `df = df.dropna()`
    frees the old frame, a later same-shape frame can land on it -- so the
    edge rows are what actually discriminate a different frame."""
    try:
        fp = frame_fingerprint(df)
    except Exception:  # noqa: BLE001
        return None
    try:
        edge = (str(df.iloc[0].values.tolist()) if fp[0] else "",
                str(df.iloc[-1].values.tolist()) if fp[0] else "")
    except Exception:  # noqa: BLE001
        edge = ("", "")
    return (fname, fp[0], hash(fp[1]), id(df), hash(edge))


def frame_needs_scan(fname: str, df, seen: set, code: str = "") -> bool:
    """True when `df` should be scanned this cell: first sighting, a
    different frame (see frame_key), or a cell whose source assigns into it.

    Failure direction, deliberately: a same-shape, same-edge-rows mutation
    the assignment regex does not recognise -- `f(df)` where f overwrites a
    column, `df.update(other)`, writes through `.values` -- is NOT rescanned,
    so an inf/leakage warning it introduces can be missed until the frame
    next changes. That is the price of not re-scanning every crore-scale
    frame on every cell; unknown -> skip, on purpose."""
    k = frame_key(fname, df)
    if k is None:
        return True
    key = f"{_FP_PREFIX}{fname}:{hash(k)}"
    if key in seen and not code_assigns_to(code, fname):
        return False
    for old in [k for k in seen if isinstance(k, str)
                and k.startswith(f"{_FP_PREFIX}{fname}:")]:
        seen.discard(old)
    seen.add(key)
    return True


# Windows already computed for a frame at a given fingerprint, so the
# pairwise comparison still sees every frame without re-bounding the
# unchanged ones. Small: one tuple per (name, shape).
_WINDOW_CACHE: dict = {}
_WINDOW_CACHE_MAX = 256


def _coerce_dates(s: Any):
    """Return a parsed datetime Series, or None. Samples big string columns:
    bounding a window doesn't need every row, and parsing 1.5M strings on
    every cell would make the check cost more than the analysis."""
    import pandas as pd

    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.dropna()
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            return None
        sample = s.dropna()
        if len(sample) == 0:
            return None
        if len(sample) > _SAMPLE_ROWS:
            sample = sample.sample(_SAMPLE_ROWS, random_state=0)
        # A non-date column reaching here is expected (we're probing), and
        # pandas' "could not infer format" UserWarning would surface as noise
        # in the user's terminal for a check that correctly stays silent.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        # Mostly-unparseable => not a date column (e.g. free text, ids).
        if parsed.notna().mean() < 0.8:
            return None
        return parsed.dropna()
    except Exception:
        return None


def _date_col(df) -> Optional[str]:
    """The most plausible date column: prefer a real datetime dtype, then a
    name hint that actually parses."""
    import pandas as pd

    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return str(c)
        except Exception:
            continue
    for c in df.columns:
        if any(h in str(c).lower() for h in _DATE_HINTS):
            if _coerce_dates(df[c]) is not None:
                return str(c)
    return None


def _window(df, col: str) -> Optional[tuple]:
    """(min, max, span_days) for a frame's date column."""
    d = _coerce_dates(df[col])
    if d is None or len(d) == 0:
        return None
    lo, hi = d.min(), d.max()
    try:
        span = (hi - lo).total_seconds() / 86400.0
    except Exception:
        return None
    return (lo, hi, span)


def _label_col(df) -> Optional[str]:
    """A binary label column, if one is obvious."""
    for c in df.columns:
        if not any(h in str(c).lower() for h in _LABEL_HINTS):
            continue
        try:
            u = df[c].dropna().unique()
            if 1 < len(u) <= 2:
                return str(c)
        except Exception:
            continue
    return None


def _fmt_span(days: float) -> str:
    if days < 1:
        return f"{days * 24:.0f}h"
    return f"{days:.0f}d"


def _fmt_win(lo, hi, span) -> str:
    try:
        return f"{lo.date()} -> {hi.date()} ({_fmt_span(span)})"
    except Exception:
        return f"{lo} -> {hi} ({_fmt_span(span)})"


def _compare(a_name: str, a_win: tuple, b_name: str, b_win: tuple) -> Optional[tuple[str, str]]:
    """Compare two windows. Returns (issue_key, message) or None."""
    a_lo, a_hi, a_span = a_win
    b_lo, b_hi, b_span = b_win

    overlaps = (a_lo <= b_hi) and (b_lo <= a_hi)
    wide, narrow = (a_span, b_span) if a_span >= b_span else (b_span, a_span)
    ratio = (wide / narrow) if narrow > 0 else float("inf")

    if not overlaps:
        return (f"nooverlap:{a_name}:{b_name}",
                f"'{a_name}' [{_fmt_win(*a_win)}] and '{b_name}' "
                f"[{_fmt_win(*b_win)}] cover time windows that DO NOT OVERLAP "
                f"at all. Any comparison between them confounds the effect you "
                f"want with whatever changed between the two periods.")

    if ratio >= SPAN_RATIO_ALERT:
        wider, narrower = ((a_name, a_win), (b_name, b_win)) \
            if a_span >= b_span else ((b_name, b_win), (a_name, a_win))
        return (f"span:{a_name}:{b_name}",
                f"'{wider[0]}' spans {_fmt_span(wider[1][2])} "
                f"[{_fmt_win(*wider[1])}] but '{narrower[0]}' spans only "
                f"{_fmt_span(narrower[1][2])} [{_fmt_win(*narrower[1])}] "
                f"-- {ratio:.0f}x narrower.")
    return None


def _guidance() -> str:
    return (
        "This is a SAMPLING DESIGN problem, not a modelling one. Before "
        "engineering features or reporting any rate:\n"
        "  - Any time-derived feature (velocity, recency, day-of-week, "
        "hour-of-month, rolling counts) will separate the classes by "
        "CONSTRUCTION rather than by fraud behaviour.\n"
        "  - The class ratio, and therefore any 'fraud rate', is an artifact "
        "of how the two samples were drawn -- it is not a population rate.\n"
        "  - Say this to the user and agree a sampling frame before "
        "proceeding: either restrict every dataset to the common overlapping "
        "window, or obtain the wider sample for the narrow side. If you "
        "proceed anyway, tag findings 'cross_stratum' and say plainly which "
        "results are unusable."
    )


def _cached_window(name: str, df, code: str):
    """(date_col, window) for a frame, recomputed only when the frame
    changed (J-02) -- bounding a datetime column is a full pass."""
    key = frame_key(name, df)
    if key is None:
        return None, None
    hit = _WINDOW_CACHE.get(key)
    if hit is not None and not code_assigns_to(code, name):
        return hit
    col = _date_col(df)
    w = _window(df, col) if col else None
    if len(_WINDOW_CACHE) >= _WINDOW_CACHE_MAX:
        _WINDOW_CACHE.clear()
    _WINDOW_CACHE[key] = (col, w)
    return col, w


def check_frames(ns: dict, seen: set, code: str = "") -> str:
    """Inspect DataFrames in a sandbox namespace; return a [DATA CHECK]
    block for issues not already reported, else "".

    `seen` accumulates issue keys so a warning fires once, not on every
    subsequent cell. `code` (the cell that just ran) lets an unchanged
    frame skip its full-column date scan (J-02).
    """
    frames = {str(k): v for k, v in ns.items()
              if not str(k).startswith("_") and _is_frame(v)}
    if not frames:
        return ""

    msgs: list[str] = []

    # 1. Pairwise across separately-loaded frames (fraud_df vs nonfraud_df).
    windows: dict[str, tuple] = {}
    for name, df in frames.items():
        try:
            if len(df) == 0:
                continue
            col, w = _cached_window(name, df, code)
            if not col:
                continue
            if w and w[2] is not None:
                windows[name] = w
        except Exception:
            continue

    names = sorted(windows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Frames that are views/copies of each other produce a duplicate
            # warning; identical windows compare clean anyway.
            got = _compare(a, windows[a], b, windows[b])
            if got and got[0] not in seen:
                seen.add(got[0])
                msgs.append(got[1])

    # 2. Within one frame: do the label's classes cover different windows?
    #    This is the concat case -- after pd.concat the mismatch is still
    #    there but no longer visible as two separate frames.
    for name, df in frames.items():
        try:
            if len(df) == 0:
                continue
            # The per-class split is a full pass -- only for frames that
            # changed this cell (J-02); an unchanged frame's verdict is
            # already in `seen`.
            if not frame_needs_scan(name, df, seen, code):
                continue
            lab = _label_col(df)
            col, _ = _cached_window(name, df, code)
            if not lab or not col:
                continue
            vals = list(df[lab].dropna().unique())
            if len(vals) != 2:
                continue
            wins = {}
            for v in vals:
                sub = df[df[lab] == v]
                if len(sub) == 0:
                    continue
                w = _window(sub, col)
                if w:
                    wins[f"{name}[{lab}={v}]"] = w
            if len(wins) != 2:
                continue
            (an, aw), (bn, bw) = list(wins.items())
            got = _compare(an, aw, bn, bw)
            if got and got[0] not in seen:
                seen.add(got[0])
                msgs.append(got[1])
        except Exception:
            continue

    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[DATA CHECK] Time-window mismatch detected:\n{body}\n\n{_guidance()}"
