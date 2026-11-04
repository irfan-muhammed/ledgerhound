"""
ledgerhound.shell.numeric_checks
================================

Mechanical numeric-stability check that fires on whatever frames actually
exist in the sandbox -- not a tool the model may choose to call. Fourth
sibling to ``data_checks``/``leakage_checks``/``grain_checks``, same shape.

Why (V3-032 gap, found auditing whether prior fixes actually reached this
engine)
------------------------------------------------------------------------
Real pilot feedback: "Ratio-based features may generate extreme values
when denominators are close to zero." Tier 1 of this same initiative fixed
exactly this in ``ledgerhound/analysis/eda_derive.py`` -- but that module has
zero references from ``ledgerhound/shell/`` (confirmed by grep, not assumed),
so the fix never reached ``ConversationalAnalyst``, whose ``run_python``
computes a ratio as arbitrary freeform pandas code (``df['a'] / df['b']``)
with no guard at all.

Deliberately narrow: this checks for literal ``inf``/``-inf`` values,
which is the exact, unambiguous signature of an unguarded division by
EXACT zero (``5.0 / 0.0 == inf`` in numpy/pandas, not a raised exception).
That is a CERTAIN problem worth a hard warning, unlike a near-zero (but
not exactly zero) denominator, which is a judgment call about whether the
resulting large-but-finite values are a genuine signal or numerical noise
-- real transaction-amount data is often legitimately heavy-tailed, and a
generic "this value looks too big" check would cry wolf on correct data
constantly. A near-zero-but-nonzero case is exactly the kind of ambiguous
call better routed through ``ask_user`` by the model's own judgment (see
the system prompt) than hard-coded here as a false-positive-prone rule.
"""

from __future__ import annotations

from ledgerhound.shell.data_checks import _is_frame, frame_needs_scan


def _guidance(col: str, fname: str, n_inf: int) -> str:
    return (
        f"'{col}' in '{fname}' has {n_inf} inf/-inf value(s) -- almost "
        "always an unguarded division by exactly zero (e.g. a ratio "
        "feature whose denominator hit 0 for some rows). NaN those rows "
        "out (or drop them) rather than leaving inf in a feature -- most "
        "downstream stats/models silently break or badly distort on it."
    )


def check_numeric_stability(ns: dict, seen: set, code: str = "") -> str:
    """Inspect DataFrames in a sandbox namespace for inf/-inf in a numeric
    column; return a [NUMERIC CHECK] block for issues not already
    reported, else "".

    `seen` accumulates issue keys so a warning fires once per column, not
    on every subsequent cell -- same contract as the sibling checks. The
    inf scan is a full pass per numeric column, so an unchanged frame is
    skipped (J-02).
    """
    import numpy as np
    import pandas as pd

    frames = {str(k): v for k, v in ns.items()
              if not str(k).startswith("_") and _is_frame(v)}
    if not frames:
        return ""

    msgs: list[str] = []
    for fname, df in frames.items():
        try:
            if len(df) == 0:
                continue
            if not frame_needs_scan(fname, df, seen, code):
                continue
            for col in df.columns:
                series = df[col]
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                key = f"inf:{fname}:{col}"
                if key in seen:
                    continue
                n_inf = int(np.isinf(series.to_numpy(dtype="float64", na_value=0.0)).sum())
                if n_inf <= 0:
                    continue
                seen.add(key)
                msgs.append(_guidance(str(col), fname, n_inf))
        except Exception:
            continue

    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[NUMERIC CHECK] Non-finite values found:\n{body}"
