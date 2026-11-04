"""
ledgerhound.shell.verification
=============================

V3-017: auto-verification after a mutating transform.

A cheap (row count + column list only) fingerprint is compared before and
after every code_exec cell. Only when that comparison shows a real change
(rows or columns differ) does anything more expensive run -- a null-rate
scan is a full pass over every column, the wrong thing to pay for on every
cell at crore-scale, including the many cells that are pure reads/computes
and never touch `df` at all.

Pure functions over plain dicts -- no pandas import, no I/O -- so both the
exec-mode and kernel-mode callers in ``ledgerhound.shell.conversation`` share
one tested implementation instead of two hand-rolled ones.
"""

from __future__ import annotations

from typing import Optional


def fingerprint(n_rows, columns) -> dict:
    # `columns` is deliberately NOT annotated `list`: callers pass a pandas /
    # cuDF Index (df.columns), and a Cython-compiled build enforces a `list`
    # annotation exactly ("Argument 'columns' has incorrect type (expected
    # list, got Index)") -- which made load_data and every run_python fail
    # on the pod while plain Python ran fine. Any iterable of names works.
    return {"n_rows": int(n_rows), "columns": [str(c) for c in columns]}


def fingerprints_differ(before: Optional[dict], after: dict) -> bool:
    """False when there's no prior baseline yet (nothing to compare) or the
    two fingerprints match."""
    if not before:
        return False
    return (before.get("n_rows") != after.get("n_rows") or
            before.get("columns") != after.get("columns"))


def format_delta(before: dict, after: dict, null_rates: Optional[dict] = None,
                 prior_null_rates: Optional[dict] = None) -> str:
    """Build the observation-injected verification block. `null_rates` (if
    given) is the CURRENT null-rate-per-column, computed by the caller only
    because a fingerprint change was already detected. `prior_null_rates`,
    if the caller has one cached from a previous mutation, turns that into
    a genuine delta rather than just a current snapshot."""
    n0, n1 = before.get("n_rows", 0), after.get("n_rows", 0)
    cols0 = before.get("columns", [])
    cols1 = after.get("columns", [])
    pct = f"{(n1 - n0) / n0 * 100:+.1f}%" if n0 else "n/a"
    lines = [f"[VERIFICATION] rows: {n0:,} -> {n1:,} ({pct})"]

    dropped = [c for c in cols0 if c not in cols1]
    added = [c for c in cols1 if c not in cols0]
    if dropped:
        lines.append(f"dropped columns: {dropped}")
    if added:
        lines.append(f"added columns: {added}")

    if null_rates:
        if prior_null_rates:
            deltas = []
            for c in sorted(set(null_rates) & set(prior_null_rates)):
                d = null_rates[c] - prior_null_rates[c]
                if abs(d) >= 0.001:
                    deltas.append(f"{c}: {d * 100:+.1f}%")
            if deltas:
                lines.append("null-rate change: " + ", ".join(deltas[:8]))
        else:
            top = sorted(null_rates.items(), key=lambda kv: -kv[1])[:5]
            top = [(c, r) for c, r in top if r > 0]
            if top:
                lines.append("null rates now: " +
                             ", ".join(f"{c}={r * 100:.1f}%" for c, r in top))

    return "\n".join(lines)
