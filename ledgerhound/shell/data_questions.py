"""
ledgerhound.shell.data_questions
===============================

What must be ASKED about a dataset once it is profiled, instead of assumed.

Real UPI extracts are full of things a model will otherwise guess: a
`fraud_flag` with four classes (which of them are fraud?), a second label
(`is_mule`), a payer and a payee id (which entity is the analysis about?),
two timestamps (which is the event time?), response / purpose / MCC codes
whose values mean nothing without the business key, amounts that may be in
paise. A wrong guess on any of these silently poisons everything after it.

`open_questions()` turns the loaded frame's schema (plus the profiler's
digest) into concrete questions. Answers are recorded on the dataset card
(`record_data_answer`), which the team shares -- so a question answered
once for a dataset is never asked again, by anyone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL_HINTS = ("fraud", "label", "target", "is_mule", "mule", "flag", "class",
                "chargeback", "dispute", "is_bad", "bad_flag")
_ENTITY_HINTS = ("vpa", "accno", "account", "acct", "cust", "customer", "crn",
                 "payer", "payee", "remitter", "beneficiary", "merchant", "mid",
                 "device", "mobile", "card")
_AMOUNT_HINTS = ("amt", "amount", "value", "inr", "txn_val")
_CODE_NAME_HINTS = ("code", "_cd", "cd_", "type", "status", "resp", "reason",
                    "purpose", "mcc", "channel", "mode", "category", "cat", "flag")
_MAX_CODE_QUESTIONS = 6
_MAX_CLASSES_SHOWN = 12


@dataclass
class Question:
    key: str          # stable id -- the same question on the same data has the same key
    text: str

    def line(self) -> str:
        return f"[{self.key}] {self.text}"


def _nunique(df, col) -> int:
    try:
        return int(df[col].nunique(dropna=True))
    except Exception:  # noqa: BLE001
        return -1


def _classes(df, col, n=_MAX_CLASSES_SHOWN) -> str:
    try:
        vc = df[col].value_counts(dropna=True).head(n)
        return ", ".join(f"{k} ({int(v):,})" for k, v in vc.items())
    except Exception:  # noqa: BLE001
        return ""


def _is_datetime(df, col) -> bool:
    try:
        import pandas as pd
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        low = str(col).lower()
        return low.endswith(("_ts", "_dt", "_date", "_time", "timestamp")) or low in ("ts", "date", "time")
    except Exception:  # noqa: BLE001
        return False


def _looks_like_codes(df, col) -> bool:
    """Short opaque values: '00', 'U30', 'Z9', 'P2M', '5411' -- meaningless
    without the business key."""
    try:
        vals = [str(v) for v in df[col].dropna().astype(str).unique()[:50]]
    except Exception:  # noqa: BLE001
        return False
    if not vals:
        return False
    opaque = [v for v in vals if len(v) <= 6 and re.fullmatch(r"[A-Za-z0-9_\-]+", v)
              and not v.isalpha() or (len(v) <= 4 and v.isupper())]
    return len(opaque) >= max(2, int(0.6 * len(vals)))


def open_questions(df, answered: set[str] | None = None) -> list[Question]:
    """Questions about `df` that have not been answered yet, most
    important first."""
    answered = answered or set()
    cols = [str(c) for c in getattr(df, "columns", [])]
    low = {c: c.lower() for c in cols}
    qs: list[Question] = []

    # -- the label ---------------------------------------------------------
    labels = [c for c in cols if any(h in low[c] for h in _LABEL_HINTS)
              and 0 < _nunique(df, c) <= 20]
    if not labels:
        qs.append(Question("label", "No fraud-label column is recognisable. Which "
                           "column is the label -- or is this data unlabelled?"))
    elif len(labels) > 1:
        qs.append(Question("label", f"Several label-like columns: {', '.join(labels)}. "
                           "Which one is the target for this work, and what is each of "
                           "the others?"))
    for c in labels:
        n = _nunique(df, c)
        if n > 2:
            qs.append(Question(f"label_classes:{c}",
                               f"`{c}` has {n} classes: {_classes(df, c)}. What does each "
                               "class mean, and which of them count as fraud?"))
        elif n == 2:
            qs.append(Question(f"label_meaning:{c}",
                               f"`{c}` values: {_classes(df, c)}. What does the positive "
                               "value mean -- confirmed fraud, reported, or suspected? "
                               "Since when is it reliable?"))

    # -- the entity / grain ------------------------------------------------
    entities = [c for c in cols if any(h in low[c] for h in _ENTITY_HINTS)
                and c not in labels and _nunique(df, c) > 20]
    if len(entities) > 1:
        qs.append(Question("entity", f"Several identifier columns: {', '.join(entities[:8])}. "
                           "Which entity is this analysis about (e.g. the payer account, "
                           "the payee/beneficiary, the merchant) -- and is one row one "
                           "transaction?"))
    elif not entities:
        qs.append(Question("entity", "No account/customer identifier is recognisable. "
                           "What does one row represent, and which column identifies "
                           "the entity?"))

    # -- the event time ----------------------------------------------------
    times = [c for c in cols if _is_datetime(df, c)]
    if len(times) > 1:
        qs.append(Question("event_time", f"Several time columns: {', '.join(times)}. "
                           "Which one is the event time to analyse by (and what are "
                           "the others)?"))

    # -- amounts -----------------------------------------------------------
    for c in cols:
        if any(h in low[c] for h in _AMOUNT_HINTS) and c not in labels:
            try:
                import pandas as pd
                if pd.api.types.is_numeric_dtype(df[c]):
                    qs.append(Question(f"units:{c}", f"Is `{c}` in rupees or paise?"))
            except Exception:  # noqa: BLE001
                pass

    # -- opaque code columns -----------------------------------------------
    n_code = 0
    for c in cols:
        if c in labels or n_code >= _MAX_CODE_QUESTIONS:
            continue
        n = _nunique(df, c)
        if 2 <= n <= 60 and (any(h in low[c] for h in _CODE_NAME_HINTS)
                            or _looks_like_codes(df, c)) and _looks_like_codes(df, c):
            qs.append(Question(f"codes:{c}",
                               f"What do the values of `{c}` mean ({n} classes: "
                               f"{_classes(df, c)})? Any that must be excluded or grouped?"))
            n_code += 1

    return [q for q in qs if q.key not in answered]


def questions_block(open_qs: list[Question], known: dict[str, dict]) -> str:
    """The text appended to the profile result."""
    parts = []
    if known:
        parts.append("KNOWN ABOUT THIS DATA (answered earlier -- use these, do not re-ask):")
        for k, c in known.items():
            who = f" -- {c.get('by')}" if c.get("by") else ""
            parts.append(f"  [{k}] {c.get('answer')}{who}")
    if open_qs:
        if parts:
            parts.append("")
        parts.append(
            "QUESTIONS TO ASK THE USER NOW -- before any analysis. Do NOT assume an "
            "answer to any of these. Ask them (ask_user for choices, or a plain "
            "question), then record each answer with record_data_answer(key, answer):")
        for i, q in enumerate(open_qs, 1):
            parts.append(f"  {i}. {q.line()}")
    return "\n".join(parts)
