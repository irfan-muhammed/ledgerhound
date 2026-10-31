"""
sentinel.shell.leakage_checks
==============================

Mechanical leakage checks that fire on whatever frames actually exist in the
sandbox -- not a tool the model may choose to call. Mirrors
``sentinel.shell.data_checks``'s shape and rationale exactly (see that
module's docstring for the full argument); the short version: a warning the
model cannot skip, injected right after the code that created the problem,
beats hoping the model thinks to check.

Why this exists (V3-032)
-------------------------
Real pilot feedback on this exact engine (``ConversationalAnalyst`` /
``sentinel shell``, confirmed as the tool that produced the reviewed
reports): features were generated directly from the fraud label column
itself, and a Risk Score field -- a downstream MODEL/pipeline output, not a
genuine analytical signal -- was treated as an analytical feature, which a
reviewer said "impacts one of the major conclusions" of a report.

``sentinel.validation.leakage.LeakageSentinel`` already solves this
correctly, but it lives in the OTHER agent engine
(``sentinel/agents/tools/phase2_tools.py``, reached via ``sentinel agent
run`` / ``sentinel study run``) which ``ConversationalAnalyst`` has no
connection to at all -- confirmed by grep, not assumed. Rather than build a
second implementation, this module reuses ``LeakageSentinel``'s lexicon
directly (import, not copy) so the two engines can never silently drift
apart on what counts as a leaky name.

Two checks, both narrow and mechanical
---------------------------------------
1. Column NAME matches the target-leak lexicon (Risk Score, chargeback,
   fraud-rate-style aggregates, ...).
2. Column VALUES are suspiciously identical to a detected label column --
   catches a feature built directly from the label under a name the
   lexicon would never match (the pilot's literal complaint: "avoid
   generating feature columns derived from Fraud_Flag").

Deliberately narrow. It does not try to detect every leakage failure mode
(a genuinely subtle one needs the fuller AUROC-gap/point-in-time machinery
in LeakageSentinel, reachable via the check_leakage helper this module also
injects into the sandbox for anyone who wants it) -- it catches the two
concrete, real, named-in-feedback mistakes cheaply and every time.
"""

from __future__ import annotations

from typing import Any

from sentinel.shell.data_checks import _is_frame, _label_col, frame_needs_scan
from sentinel.validation.leakage import ComputedAt, FeatureSpec, LeakageSentinel, LeakVerdict


def _lexicon_hits(columns: list[str]) -> list[tuple[str, str]]:
    """[(column, reason), ...] for names matching LeakageSentinel's lexicon.

    Goes through the public vet() API rather than reaching into
    LeakageSentinel's internals, so this module can't drift out of sync
    with whatever the sentinel actually checks. computed_at is declared
    PRE_EVENT for every column here on purpose -- these are candidate
    COLUMN NAMES already sitting in a loaded frame, not yet-to-be-computed
    features, so the point-in-time check (which this call would otherwise
    challenge as UNKNOWN) isn't the thing being asked about; only a lexicon
    match should surface from this call.
    """
    specs = [FeatureSpec(name=c, computed_at=ComputedAt.PRE_EVENT) for c in columns]
    report = LeakageSentinel().vet(specs)
    return [
        (v.feature, "; ".join(v.reasons))
        for v in report.feature_verdicts
        if v.verdict == LeakVerdict.VETOED
    ]


def _looks_derived_from_label(df, label_col: str, other_col: str) -> bool:
    """True if `other_col`'s values are suspiciously identical to the label's.

    Deliberately simple: exact equality after aligning on the label's own
    dtype (handles the common "same values, different dtype" case, e.g. a
    0/1 int label vs a 0.0/1.0 float copy) -- not a fuzzy correlation
    threshold, which would need a numeric-only assumption this can't make
    for an arbitrary column.
    """
    import pandas as pd

    try:
        label = df[label_col]
        other = df[other_col]
        if len(label) != len(other):
            return False
        aligned = pd.to_numeric(other, errors="coerce") if pd.api.types.is_numeric_dtype(label) \
            else other.astype(str)
        base = label if pd.api.types.is_numeric_dtype(label) else label.astype(str)
        return bool((aligned == base).all())
    except Exception:
        return False


def _guidance() -> str:
    return (
        "Do not use this as a feature. If it is genuinely needed, confirm "
        "with the user first:\n"
        "  - A column matching a leak-lexicon pattern (risk score, fraud "
        "rate, chargeback, ...) is very often a DOWNSTREAM pipeline/model "
        "output, not raw behavioural signal -- using it as an input to a "
        "new model is circular.\n"
        "  - A column built from the label column is the answer, not a "
        "feature -- it will look perfect here and fail completely in "
        "production.\n"
        "  - For a real check beyond names (point-in-time, dev-vs-OOT "
        "AUROC gap), call the check_leakage(...) helper available in this "
        "sandbox."
    )


def check_leakage(ns: dict, seen: set, code: str = "") -> str:
    """Inspect DataFrames in a sandbox namespace; return a [LEAKAGE CHECK]
    block for issues not already reported, else "".

    `seen` accumulates issue keys so a warning fires once per column, not
    on every subsequent cell -- same contract as
    ``data_checks.check_frames``. The column-vs-label comparison is a full
    pass per column, so an unchanged frame is skipped (J-02).
    """
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
            columns = [str(c) for c in df.columns]

            for col, reason in _lexicon_hits(columns):
                key = f"lexicon:{fname}:{col}"
                if key in seen:
                    continue
                seen.add(key)
                msgs.append(f"'{col}' in '{fname}': {reason}")

            lab = _label_col(df)
            if lab:
                for col in columns:
                    if col == lab:
                        continue
                    key = f"derived:{fname}:{col}"
                    if key in seen:
                        continue
                    if _looks_derived_from_label(df, lab, col):
                        seen.add(key)
                        msgs.append(
                            f"'{col}' in '{fname}' has values identical to "
                            f"label column '{lab}' -- built from the target."
                        )
        except Exception:
            continue

    if not msgs:
        return ""
    body = "\n".join(f"  - {m}" for m in msgs)
    return f"[LEAKAGE CHECK] Possible target leakage:\n{body}\n\n{_guidance()}"


# -- sandbox helper (V3-032) -------------------------------------------------
#
# Exposed inside run_python's namespace so the model can run the FULLER
# LeakageSentinel checks (point-in-time, dev-vs-OOT AUROC gap) on demand,
# not just the two mechanical ones above that fire automatically.


def make_check_leakage_helper():
    """Return a callable to inject into the sandbox as `check_leakage`.

    Signature designed to be easy for a model to call from ordinary code:
        check_leakage(["feat_a", "feat_b"], label_column="Fraud_Flag")
    `computed_at`/`inputs`/etc. can be passed per-feature as a dict instead
    of a bare name when the model has that detail; a bare string defaults
    to computed_at="unknown" (challenged, not vetoed) -- honest about what
    wasn't declared rather than assuming it's safe.
    """
    def _check_leakage(features, label_column=None, **kw):
        specs = []
        for f in features:
            if isinstance(f, dict):
                specs.append(FeatureSpec(
                    name=f["name"],
                    computed_at=f.get("computed_at", "unknown"),
                    as_of_note=f.get("as_of_note", ""),
                    inputs=list(f.get("inputs", []) or []),
                    is_group_aggregate=bool(f.get("is_group_aggregate", False)),
                    fit_scope=f.get("fit_scope", "unknown"),
                ))
            else:
                specs.append(FeatureSpec(name=str(f)))
        report = LeakageSentinel().vet(specs, label_column=label_column, **kw)
        report.display()
        # report.to_dict() (asdict()) omits vetoed_features -- it's a
        # method, not a dataclass field. Same explicit shape as
        # Phase2Tools.check_leakage's JSON return, so a user familiar with
        # one gets the other's keys for free.
        d = report.to_dict()
        d["vetoed_features"] = report.vetoed_features()
        return d

    return _check_leakage
