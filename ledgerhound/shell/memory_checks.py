"""
ledgerhound.shell.memory_checks
===============================

Mechanical check for redundant large data copies sitting in memory at the
same time -- fires on whatever variables actually exist in the sandbox,
not a tool the model may choose to call. Sibling to
``data_checks``/``leakage_checks``/``grain_checks``/``numeric_checks``/
``ambiguity_checks``, same shape.

Why (real feedback, same session as the deep-exploration budget fix)
----------------------------------------------------------------------
Once a session is encouraged to explore for hours across many columns and
correlations on an 8-crore-row table (see the "Go deep" system-prompt
section), the generated code naturally does more transforms -- and nothing
previously discouraged `backup = df.copy()` / `df_orig = df.copy()` /
`snapshot = df.copy()` style patterns that quietly hold a second (or
third) full-size copy of the working frame alive for the rest of the
session. At crore-scale that is real memory, not a style nit -- worth a
mechanical nudge the same way an inf/-inf value or a leaky feature is.

Deliberately a heuristic, not a hard rule: a same-sized frame is not
ALWAYS wasteful (a genuine before/after comparison sometimes needs both
kept around briefly), so this warns once per variable name and lets the
model's own judgment decide whether to `del` it -- same non-blocking,
advisory posture as every other check in this family.
"""

from __future__ import annotations

from ledgerhound.shell.data_checks import _is_frame

# Below this many rows the memory cost of a second copy is noise --
# nothing here is worth flagging on small/dev-scale data.
_MIN_ROWS = 10_000
# A second frame at or above this fraction of the reference frame's row
# count reads as "basically a full copy," not a genuinely smaller
# derived/aggregated result.
_NEAR_FULL_RATIO = 0.5


def _guidance(name: str, n_rows: int, ref_len: int) -> str:
    pct = n_rows / ref_len if ref_len else 0.0
    return (
        f"'{name}' holds {n_rows:,} rows -- {pct:.0%} the size of `df` "
        f"({ref_len:,} rows). Keeping a second large copy alive burns real "
        f"memory at this scale. If you no longer need it, `del {name}`; if "
        "you were about to transform it further, prefer reassigning in "
        "place (e.g. `df = df.dropna()`) or working on a column subset "
        "instead of copying the whole frame."
    )


def format_memory_block(pairs: dict, ref_len: int, seen: set) -> str:
    """Shared formatter: `pairs` is {var_name: row_count} for candidate
    large frames (already collected by either the exec-sandbox or the
    kernel-mode caller). Returns a [MEMORY CHECK] block for names not
    already in `seen`, else "" -- same once-per-issue contract as the
    sibling checks."""
    if ref_len < _MIN_ROWS:
        return ""
    msgs: list[str] = []
    for name, n_rows in pairs.items():
        try:
            n_rows = int(n_rows)
        except (TypeError, ValueError):
            continue
        if n_rows < _MIN_ROWS or n_rows < _NEAR_FULL_RATIO * ref_len:
            continue
        key = f"dupe:{name}"
        if key in seen:
            continue
        seen.add(key)
        msgs.append(_guidance(name, n_rows, ref_len))
    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[MEMORY CHECK] Multiple large data copies in memory:\n{body}"


def check_memory_usage(ns: dict, seen: set) -> str:
    """Inspect a sandbox namespace (exec-mode `self._ns`) for other
    DataFrame-like variables that are near-full-size copies of `df`."""
    df = ns.get("df")
    if df is None or not _is_frame(df):
        return ""
    try:
        ref_len = len(df)
    except Exception:
        return ""
    pairs: dict = {}
    for k, v in ns.items():
        name = str(k)
        if name.startswith("_") or name == "df" or not _is_frame(v):
            continue
        try:
            pairs[name] = len(v)
        except Exception:
            continue
    return format_memory_block(pairs, ref_len, seen)
