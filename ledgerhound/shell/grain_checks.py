"""
ledgerhound.shell.grain_checks
=============================

Mechanical output-grain check that fires when the sandbox code EXPORTS a
frame (``.to_csv(`` / ``.to_parquet(``) -- not a tool the model may choose
to call. Same shape and rationale as ``ledgerhound.shell.data_checks`` and
``ledgerhound.shell.leakage_checks``: a warning the model cannot skip, injected
right after the code that created the problem.

Why gated on export, specifically (V3-032 Tier 4)
----------------------------------------------------
Real pilot feedback: "For each Customer Reference Number or Account, feature
values should be unique and consistent. We require an aggregated
account-level transaction dataset based on Payee Account, rather than a
dataset containing duplicate Customer Reference Numbers."

A raw transaction-level frame legitimately has many rows per account --
that is correct and expected while it is still being explored. Flagging
every frame with a repeated account ID would cry wolf on ordinary,
correct intermediate work (exactly the failure mode
``data_checks.py``'s docstring warns against: a check that cries wolf on
legitimate data is worse than no check, because people learn to ignore
it). Gating on an actual export call (".to_csv(" / ".to_parquet(") ties
the check to the moment that matters: the model has decided THIS frame is
a deliverable, which is exactly when the pilot's complaint would
materialise -- a feature sheet handed to the FE team with duplicate
account rows.
"""

from __future__ import annotations

import re
from typing import Optional

from ledgerhound.shell.data_checks import _date_col, _is_frame

_EXPORT_HINTS = (".to_csv(", ".to_parquet(", ".to_excel(")

# Column-name hints for an entity/account key, checked as substrings on the
# lowered name. `payee_accno` is named explicitly in the real feedback.
_ENTITY_KEY_HINTS = (
    "accno", "account", "acct", "cust_ref", "customer_reference",
    "payee_accno", "payer_accno", "beneficiary_account", "crn",
)


def _looks_like_export(code: str) -> bool:
    return any(h in code for h in _EXPORT_HINTS)


_EXPORTED_NAME_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\.\s*to_(?:csv|parquet|excel|feather|json|pickle)\s*\(")
_HELPER_EXPORT_RE = re.compile(r"\bsafe_to_parquet\s*\(\s*([A-Za-z_]\w*)\b")


def exported_frame_names(code: str) -> set[str]:
    """Names of the frames this cell actually saves (`feat.to_parquet(...)`,
    `safe_to_parquet(feat, ...)`). 2026-09-19: the grain check used to scan
    EVERY frame in scope whenever a cell exported anything -- so saving an
    account-level `feat` reported the raw transaction `df` (which repeats
    accounts by design) as "a frame you are exporting", on every save."""
    names = set(_EXPORTED_NAME_RE.findall(code or ""))
    names |= set(_HELPER_EXPORT_RE.findall(code or ""))
    return names


def _entity_key_col(df) -> Optional[str]:
    """The most plausible entity/account-key column, by name hint."""
    for c in df.columns:
        if any(h in str(c).lower() for h in _ENTITY_KEY_HINTS):
            return str(c)
    return None


def _guidance(col: str, dupes: int, total: int) -> str:
    return (
        f"'{col}' has {dupes} duplicate value(s) across {total} rows in a "
        "frame you are exporting. If this is meant to be an account-level "
        "feature sheet, each account should appear exactly once -- "
        "aggregate to one row per account before exporting (real pilot "
        "feedback: \"we require an aggregated account-level transaction "
        "dataset... rather than a dataset containing duplicate Customer "
        "Reference Numbers\"). If this frame is intentionally "
        "transaction-level (not a feature deliverable), ignore this."
    )


def check_output_grain(code: str, ns: dict, seen: set) -> str:
    """Inspect DataFrames in a sandbox namespace for a duplicated entity key,
    ONLY when `code` looks like it's exporting one. Returns a
    [OUTPUT GRAIN] block for issues not already reported, else "".

    `seen` accumulates issue keys so a warning fires once per column, not
    on every subsequent export -- same contract as the sibling checks.
    """
    if not _looks_like_export(code):
        return ""

    exported = exported_frame_names(code)
    frames = {str(k): v for k, v in ns.items()
              if not str(k).startswith("_") and _is_frame(v) and str(k) in exported}
    if not frames:
        return ""

    msgs: list[str] = []
    for fname, df in frames.items():
        try:
            if len(df) == 0:
                continue
            # A frame with an obvious per-transaction timestamp column is
            # expected to repeat an account many times -- that's correct
            # transaction grain, not the bug being checked for.
            if _date_col(df) is not None:
                continue
            key_col = _entity_key_col(df)
            if not key_col:
                continue
            check_key = f"grain:{fname}:{key_col}"
            if check_key in seen:
                continue   # J-06: consult `seen` BEFORE the nunique() pass
            n_total = len(df)
            n_unique = df[key_col].nunique(dropna=True)
            n_dupes = n_total - n_unique
            if n_dupes <= 0:
                continue
            seen.add(check_key)
            msgs.append(_guidance(key_col, n_dupes, n_total))
        except Exception:
            continue

    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[OUTPUT GRAIN] Possible duplicate-account rows in an exported frame:\n{body}"
