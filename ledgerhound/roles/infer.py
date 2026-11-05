"""
ledgerhound.roles.infer
=======================

Propose a role for every column from generic evidence only: the words in its
name (split on case, underscores and digits) and the shape of its values
(dtype, cardinality, binary-ness). No dataset's column names appear here --
that is the Phase 3 gate ("public datasets run with no schema-specific code").

Confidence is honest rather than eager: "high" settles a non-label role
without asking, so it is only given when name AND values agree. Anything
less goes to the person. Unknown columns are "feature" with high confidence
when nothing in the name suggests a role (e.g. anonymised V1..V28).
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ledgerhound.roles.model import PERSON_ONLY, SINGLE, Assignment, RoleMap

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokens(name: str) -> list[str]:
    """'nameOrig' -> [name, orig]; 'TransactionAmt' -> [transaction, amt];
    'fraud_bool' -> [fraud, bool]; 'card1' -> [card, 1]."""
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", str(name)):
        out.extend(t.lower() for t in _CAMEL.findall(part))
    return out


# -- lexicons (generic fraud vocabulary, not column names) ------------------ #
_LABEL_WORDS = {"fraud", "fraudulent", "label", "target", "class", "chargeback", "cb",
                "y", "scam", "mule", "bad"}
_LABEL_FILLER = {"is", "bool", "flag", "ind", "indicator", "label", "target", "class",
                 "fraud", "fraudulent", "chargeback", "cb", "y", "scam", "mule", "bad",
                 "confirmed", "reported", "txn", "transaction", "tx", "binary"}
_AGGREGATE = {"count", "cnt", "num", "n", "rate", "pct", "ratio", "sum", "avg", "mean",
              "distinct", "days", "history", "hist", "prior", "past", "prev", "last",
              "total", "velocity", "share"}
_RULE_WORDS = {"flagged", "rule", "alert", "alerted", "hit", "blocked", "declined",
               "triggered", "flagger"}
_SCORE_WORDS = {"score", "prob", "probability", "risk", "propensity"}
_TIME_WORDS = {"time", "timestamp", "ts", "date", "datetime", "dt", "dttm", "created"}
_WEAK_TIME_WORDS = {"step", "period", "month", "week", "day"}
_NOT_TIME = {"birth", "dob", "since", "elapsed", "duration", "length", "minutes",
             "seconds", "hours", "expiry", "expiration"}
_LABEL_TIME_HINT = {"report", "reported", "chargeback", "cb", "dispute", "fraud",
                    "label", "confirmed", "investigation", "resolved"}
_SOURCE_WORDS = {"source", "reason", "origin", "type", "channel"}
_AMOUNT_WORDS = {"amount", "amt"}
_ID_WORDS = {"id", "no", "num", "number", "key", "ref", "hash", "fp", "fingerprint", "token"}
_TXN_WORDS = {"txn", "transaction", "trans", "tx", "trx", "payment", "event", "row", "record"}
_PAYER = {"payer", "sender", "orig", "origin", "originator", "src", "source", "from",
          "remitter", "debtor", "debit"}
_PAYEE = {"payee", "receiver", "recipient", "beneficiary", "dest", "destination", "dst",
          "to", "creditor", "credit", "benef"}
_ACCOUNT = {"account", "acct", "acc", "customer", "cust", "user", "client", "member",
            "applicant", "holder"}
_CARD = {"card", "pan", "bin"}
_DEVICE = {"device", "imei", "fingerprint"}
_MERCHANT = {"merchant", "mcc", "store", "shop", "terminal", "mid"}
_CHANNEL = {"channel", "rail", "product", "method", "mode", "instrument", "type",
            "network", "scheme"}
_PII = {"name", "phone", "mobile", "email", "mail", "address", "addr", "ssn",
        "aadhaar", "passport", "dob", "birth", "zip", "postcode", "ip"}
_NOT_ENTITY = {"balance", "bal", "amount", "amt", "count", "cnt", "age", "limit",
               "os", "type", "category", "country", "city", "state", "len", "length",
               "domain", "valid", "velocity", "distinct", "days", "ratio"}


def _is_binary(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    if len(vals) != 2:
        return False
    try:
        return set(pd.Series(vals).astype(float).tolist()) == {0.0, 1.0}
    except (TypeError, ValueError):
        low = {str(v).strip().lower() for v in vals}
        return low in ({"true", "false"}, {"yes", "no"}, {"y", "n"})


def _positive_rate(s: pd.Series) -> Optional[float]:
    try:
        x = s.dropna().astype(float)
        return float(x.mean()) if len(x) else None
    except (TypeError, ValueError):
        low = s.dropna().astype(str).str.strip().str.lower()
        return float(low.isin({"true", "yes", "y"}).mean()) if len(low) else None


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.9 and sample.str.len().min() >= 6
    return False


def _unique_ratio(s: pd.Series) -> float:
    n = s.notna().sum()
    return float(s.nunique(dropna=True)) / n if n else 0.0


def _id_like(s: pd.Series, t: set[str]) -> bool:
    """An identifier: string or integer, many distinct values, not a
    measured quantity."""
    if t & _NOT_ENTITY:
        return False
    if pd.api.types.is_float_dtype(s):
        # float ids exist (IDs read with NaNs), but only if integral
        x = s.dropna()
        if len(x) == 0 or not (x == x.round()).all():
            return False
    n = s.notna().sum()
    if n == 0:
        return False
    nu = s.nunique(dropna=True)
    # 20 distinct values on real data; fewer on a tiny sample
    return nu >= min(20, max(2, int(0.2 * n))) and nu / n >= 0.01


def _propose(name: str, s: pd.Series, n_rows: int) -> Assignment:
    t = tokens(name)
    ts = set(t)
    binary = _is_binary(s)

    # rule outputs and scores: downstream of the label
    if ts & _RULE_WORDS and (binary or not _is_numeric(s) or s.nunique() <= 20):
        return Assignment("rule_output", "high" if binary else "medium",
                          "name says a rule/alert fired")
    if ts & _SCORE_WORDS and _is_numeric(s) and not binary:
        return Assignment("score", "medium", "numeric column named as a score/probability")

    # label: a binary column whose name is (mostly) label words
    if binary and ts & _LABEL_WORDS and not ts & _AGGREGATE:
        rate = _positive_rate(s)
        rate_s = f"{rate:.2%} positive" if rate is not None else "binary"
        conf = "high" if ts <= _LABEL_FILLER else "medium"
        return Assignment("label", conf, f"binary, {rate_s}, name says fraud/label")

    # label_time / label_source
    if ts & _TIME_WORDS and ts & _LABEL_TIME_HINT:
        return Assignment("label_time", "medium", "a date tied to the fraud report or label")
    if ts & _SOURCE_WORDS and ts & (_LABEL_WORDS - {"y"}) and not binary:
        return Assignment("label_source", "medium", "where the fraud label came from")

    # row / transaction identifier
    ur = _unique_ratio(s)
    if (ts & _ID_WORDS or t == ["id"]) and (ts & _TXN_WORDS or t == ["id"]) and ur > 0.98:
        return Assignment("transaction_id", "high", "unique per row, named as an id")

    # amount
    if ts & _AMOUNT_WORDS and _is_numeric(s):
        return Assignment("amount", "high", "numeric, named as an amount")

    # event time (not an aggregate or a birth date that merely says "date")
    timeish = not ts & (_AGGREGATE | _NOT_TIME)
    if timeish and _is_datetime_like(s) and not ts & _LABEL_TIME_HINT:
        return Assignment("event_time", "high" if ts & _TIME_WORDS else "medium",
                          "values parse as timestamps")
    if timeish and ts & _TIME_WORDS and _is_numeric(s):
        return Assignment("event_time", "medium", "numeric time offset (name says time)")
    if ts & _WEAK_TIME_WORDS and _is_numeric(s) and len(t) == 1:
        return Assignment("event_time", "low", "numeric period index (name says step/period)")

    # entities: name words AND identifier-shaped values
    idl = _id_like(s, ts)
    for words, role in ((_PAYER, "payer"), (_PAYEE, "payee"), (_CARD, "card"),
                        (_DEVICE, "device"), (_MERCHANT, "merchant"),
                        (_ACCOUNT, "account")):
        if ts & words and idl:
            conf = "high" if (ts & _ID_WORDS or not _is_numeric(s)) else "medium"
            return Assignment(role, conf, f"identifier-shaped, name says {role}")

    # pii (names, phones, emails...) -- string-valued
    if ts & _PII and not _is_numeric(s) and not ts & {"domain"}:
        return Assignment("pii", "medium", "name suggests personal data")

    # channel: low-cardinality categorical named as a type/product/channel
    if ts & _CHANNEL and not _is_numeric(s) and 1 < s.nunique() <= 30:
        return Assignment("channel", "medium", "few categories, name says type/channel")

    # nothing domain-specific in the name
    if any(ts & w for w in (_LABEL_WORDS - {"y", "class", "bad"}, _RULE_WORDS, _SCORE_WORDS)):
        return Assignment("feature", "medium",
                          "name mentions fraud/rules but values are not a label -- "
                          "check it is computed only from data before each event")
    return Assignment("feature", "high", "no domain role")


def infer_roles(df: pd.DataFrame, *, hints: Optional[dict[str, str]] = None,
                data_sha256: str = "") -> RoleMap:
    """Propose a RoleMap for `df`. `hints` (column -> role, e.g. from a
    Knowledge Pack dictionary) override inference at medium confidence: a
    dictionary is trusted, but the person still sees it."""
    rm = RoleMap(data_sha256=data_sha256)
    n = len(df)
    for col in df.columns:
        name = str(col)
        if hints and name in hints:
            rm.columns[name] = Assignment(hints[name], "medium", "from the data dictionary")
            continue
        try:
            rm.columns[name] = _propose(name, df[col], n)
        except Exception as e:  # noqa: BLE001 -- one odd column never breaks inference
            rm.columns[name] = Assignment("feature", "low", f"could not inspect ({type(e).__name__})")
    _one_per_role(rm, df)
    for r in PERSON_ONLY:
        if not rm.columns_with(r):
            rm.absent[r] = "proposed"
    rm.auto_settle()
    return rm


_CONF_RANK = {"high": 2, "medium": 1, "low": 0}


def _one_per_role(rm: RoleMap, df: pd.DataFrame) -> None:
    """Single-valued roles keep their best candidate; the others drop to
    medium-confidence feature so the person sees them."""
    for role in SINGLE:
        cols = rm.columns_with(role)
        if len(cols) <= 1:
            continue

        def rank(c: str):
            a = rm.columns[c]
            extra = 0
            if role == "label":
                # prefer the column whose name is only label words
                extra = 1 if set(tokens(c)) <= _LABEL_FILLER else 0
            return (_CONF_RANK[a.confidence], extra, -list(df.columns).index(c))
        keep = max(cols, key=rank)
        for c in cols:
            if c != keep:
                rm.columns[c] = Assignment(
                    "feature", "medium",
                    f"also looked like {role}, but {keep} was preferred -- confirm")
