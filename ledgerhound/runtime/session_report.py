"""
ledgerhound.runtime.session_report
==================================

Turns collected session_log files (ledgerhound.runtime.session_log /
session_collector) into a quality report -- built to run inside a plain
Jupyter notebook cell (pandas table + inline matplotlib), since a Kubeflow
notebook environment has no way to serve a real web app (no JS runtime,
no proxy for an arbitrary server -- Streamlit/Langfuse/Phoenix all need
one). No server, no extra process: `report(archive_dir)` in a cell is the
whole tool.

Grain: session_log writes one line per TURN, so a session's own file/
archive has many lines sharing one `run_id`. This module collapses that to
one row per `run_id` (the latest line, i.e. that session's current/final
state) before summarising -- exactly mirroring how session_log itself says
its own file should be read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_all(path: str | Path, pattern: str = "*.jsonl"):
    """Read every matching file under `path` (or just `path` itself if it's
    a single file) into a pandas DataFrame, one row per RUN (latest line
    per run_id wins -- earlier lines are that session's own history)."""
    import pandas as pd

    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob(pattern))
    rows = []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001 -- one bad line must not sink the report
                    continue
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        return pd.DataFrame(columns=["run_id", "user", "ts"])
    df = pd.DataFrame(rows)
    if "run_id" not in df.columns:
        return df
    df = df.sort_values("ts").drop_duplicates(subset="run_id", keep="last")
    return df.reset_index(drop=True)


def summarize(df) -> dict:
    """Deployment-wide numbers: how many sessions, by whom, how often
    something needed a human or got refused. `df` is load_all()'s output."""
    if df.empty:
        return {"n_sessions": 0}

    def _col_sum(name: str) -> int:
        return int(df[name].fillna(0).sum()) if name in df.columns else 0

    flags_flat = [f for row in (df["flags"] if "flags" in df.columns else [])
                 for f in (row or [])]
    from collections import Counter
    flag_counts = Counter(flags_flat)
    agg_cols = {c: (c, "sum") for c in
               ("tool_calls", "errors", "refusals", "hitl_high_tier")
               if c in df.columns}
    by_user = df.groupby("user").agg(sessions=("run_id", "count"), **agg_cols) \
        if "user" in df.columns else None
    return {
        "n_sessions": int(len(df)),
        "n_users": int(df["user"].nunique()) if "user" in df.columns else 0,
        "total_tool_calls": _col_sum("tool_calls"),
        "total_errors": _col_sum("errors"),
        "total_refusals": _col_sum("refusals"),
        "sessions_with_any_flag": int(df["flags"].map(bool).sum())
            if "flags" in df.columns else 0,
        "flag_counts": dict(flag_counts),
        "by_user": by_user,
    }


def report(path: str | Path, pattern: str = "*.jsonl", plot: bool = True):
    """The one call to make from a notebook cell: load, summarise, and
    render -- a printed summary plus (if a display/plotting backend is
    available) a couple of inline charts. Safe to call from a plain
    terminal too -- plotting is best-effort and never raises."""
    df = load_all(path, pattern=pattern)
    s = summarize(df)

    print(f"sessions: {s.get('n_sessions', 0)}   users: {s.get('n_users', 0)}   "
         f"tool calls: {s.get('total_tool_calls', 0)}   "
         f"errors: {s.get('total_errors', 0)}   refusals: {s.get('total_refusals', 0)}")
    if s.get("flag_counts"):
        print("\nflags across all sessions:")
        for flag, n in sorted(s["flag_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {flag}: {n}")
    by_user = s.get("by_user")
    if by_user is not None and not by_user.empty:
        try:
            from IPython.display import display
            print("\nper user:")
            display(by_user)
        except Exception:  # noqa: BLE001 -- plain terminal, no IPython
            print("\nper user:")
            print(by_user.to_string())

    if plot and not df.empty:
        _maybe_plot(df, s)
    return df


def _maybe_plot(df, summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 -- matplotlib not installed; text report still stood
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "ts" in df.columns:
            import pandas as pd
            counts = pd.to_datetime(df["ts"]).dt.date.value_counts().sort_index()
            axes[0].bar(counts.index.astype(str), counts.values)
            axes[0].set_title("sessions per day")
            axes[0].tick_params(axis="x", rotation=45)
        flag_counts = summary.get("flag_counts") or {}
        if flag_counts:
            names = list(flag_counts.keys())
            axes[1].barh(names, [flag_counts[n] for n in names])
            axes[1].set_title("flags across all sessions")
        fig.tight_layout()
        plt.show()
    except Exception:  # noqa: BLE001 -- a plotting failure must not sink the report
        pass
