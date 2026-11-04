"""
ledgerhound.pipeline.appropriateness
==================================

**Is this sample appropriate for the model story it claims to serve?** (V4-021)

The scenario: instead of running the fetch, the analyst already has the data
and hands over a path. Nothing about how it was drawn is known — only what
can be read off the frame itself. So the question is not "is this data
clean" (Great Expectations answers that) but **"was this sample constructed
the way the model story says it was?"**

Refusal is the deliverable
--------------------------
Every comparable tool emits advisory warnings a user clicks through. Here a
``FAIL`` means ``can_certify`` is ``False`` and the certificate is not
issued. A sample that cannot be shown to match its declared construction
does not get a certificate saying it does — that refusal, not the report, is
the product.

What is checkable from the frame alone
--------------------------------------
Deliberately limited to things the data can actually falsify:

============================  ==================================================
``MISSING_COLUMNS``           declared keys/timestamp absent — nothing else can run
``GRAIN_MISMATCH``            declared account grain, but entities repeat — this
                              is transaction-grain data, or a broken aggregate
``DUPLICATE_KEYS``            same entity+time appears twice — double counting
``WINDOW_TOO_SHORT``          span covers far less than the declared lookback
``CLASS_WINDOW_MISMATCH``     the classes cover different periods — the original
                              reported bug (a month of fraud vs a day of
                              non-fraud), now caught on a handed-over file
``PREVALENCE``                delegated to `prevalence.check_prevalence`
``UNLABELLED_FRACTION``       so much is unlabelled the sample cannot support a rate
``IMMATURE_TAIL``             extends into the label-maturity horizon, so recent
                              fraud has not been reported yet
============================  ==================================================

Things that need provenance we do not have — was the draw stratified, what
sampling fractions were used, is it entity-disjoint from a test split — are
**not** guessed at. `appropriateness` reports what it can prove and says
plainly what it could not check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ledgerhound.pipeline.model_story import GRAIN_ACCOUNT, ModelStory
from ledgerhound.pipeline.prevalence import (
    STATUS_FAIL,
    STATUS_INCOMPARABLE,
    STATUS_PASS,
    STATUS_UNDECLARED,
    STATUS_WARNING,
    check_prevalence,
    measure_prevalence,
)

log = logging.getLogger(__name__)

SCHEMA = "appropriateness/v1"

SEV_PASS = "PASS"
SEV_WARNING = "WARNING"
SEV_FAIL = "FAIL"
_ORDER = {SEV_PASS: 0, SEV_WARNING: 1, SEV_FAIL: 2}

# A class window this many times wider than the other is a design problem,
# not sampling noise. Matches `shell/data_checks.SPAN_RATIO_ALERT`.
CLASS_SPAN_RATIO_ALERT = 3.0
# Coverage below this fraction of the declared lookback is a failure.
MIN_WINDOW_COVERAGE = 0.5
# Unlabelled share above this cannot support a prevalence claim.
MAX_UNLABELLED_FRACTION = 0.5


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "detail": self.detail}


@dataclass
class AppropriatenessReport:
    """Findings, a verdict, and an explicit list of what could NOT be checked."""

    story_name: str
    grain: str
    n_rows: int
    findings: list[Finding] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.findings:
            return SEV_PASS
        return max((f.severity for f in self.findings), key=lambda s: _ORDER[s])

    @property
    def can_certify(self) -> bool:
        """A FAIL blocks certification. This is the gate, not a suggestion."""
        return self.status != SEV_FAIL

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEV_FAIL]

    def summary(self) -> str:
        lines = [
            f"Sample appropriateness ({SCHEMA}) -- story={self.story_name!r} "
            f"grain={self.grain} rows={self.n_rows:,}",
            f"  verdict: {self.status}  "
            f"(certificate {'may' if self.can_certify else 'WILL NOT'} be issued)",
        ]
        for f in sorted(self.findings, key=lambda f: -_ORDER[f.severity]):
            lines.append(f"  [{f.severity}] {f.code}: {f.message}")
        if self.not_checked:
            lines.append("  NOT CHECKED (needs provenance the frame cannot carry):")
            lines.extend(f"    - {n}" for n in self.not_checked)
        return "\n".join(lines)

    def block(self) -> str:
        """`[DATA CHECK]` block for injection into the shell, matching
        `shell/data_checks.py`'s pattern."""
        if self.status == SEV_PASS:
            return ""
        body = "\n".join(
            f"  - [{f.severity}] {f.message}"
            for f in sorted(self.findings, key=lambda f: -_ORDER[f.severity])
            if f.severity != SEV_PASS
        )
        head = (
            "[DATA CHECK] This sample does not match its declared construction"
            if not self.can_certify
            else "[DATA CHECK] Sample construction concerns"
        )
        tail = (
            "\n\nNo certificate will be issued for this sample until the FAIL "
            "items are resolved. Feature engineering on it would optimise "
            "against a sampling artifact."
            if not self.can_certify else ""
        )
        return f"{head}:\n{body}{tail}"

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "story_name": self.story_name,
            "grain": self.grain,
            "n_rows": self.n_rows,
            "status": self.status,
            "can_certify": self.can_certify,
            "findings": [f.to_dict() for f in self.findings],
            "not_checked": list(self.not_checked),
        }


def _span_days(series) -> Optional[float]:
    import pandas as pd
    s = pd.to_datetime(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float((s.max() - s.min()).total_seconds() / 86400.0)


def check_sample(
    frame,
    story: ModelStory,
    *,
    grain: str = "",
    timestamp_column: str = "",
    label_column: str = "_label",
    mature_column: Optional[str] = "_label_mature",
    entity_keys: Optional[list[str]] = None,
    observation_date: Any = None,
) -> AppropriatenessReport:
    """Assess a handed-over sample against the model story it claims to serve.

    Only the frame is inspected; nothing about the draw is assumed.
    """
    import pandas as pd

    grain = grain or story.label_grain
    keys = list(entity_keys if entity_keys is not None else story.fetch_keys)
    report = AppropriatenessReport(
        story_name=story.name, grain=grain, n_rows=len(frame))

    # Always state the limits of this check, so a PASS is not over-read.
    report.not_checked = [
        "whether the draw was stratified as designed (needs the sampling design)",
        "per-class sampling fractions (needs draw provenance, not in the frame)",
        "entity-disjointness from any train/test split (needs the other split)",
        "whether unlabelled rows hide unreported fraud (unknowable by construction)",
    ]

    if len(frame) == 0:
        report.findings.append(Finding(
            "EMPTY", SEV_FAIL,
            "the sample is empty -- nothing can be certified about it"))
        return report

    # -- required columns ---------------------------------------------------
    missing = [c for c in keys if c not in frame.columns]
    if timestamp_column and timestamp_column not in frame.columns:
        missing.append(timestamp_column)
    if missing:
        report.findings.append(Finding(
            "MISSING_COLUMNS", SEV_FAIL,
            f"declared columns are absent from the sample: {missing}. The "
            f"model story names these as entity keys / timestamp, so this "
            f"frame cannot be the thing the story describes.",
            {"missing": missing, "available": list(frame.columns)[:30]}))
        return report          # nothing further is meaningful

    # -- grain -------------------------------------------------------------
    if keys:
        n_entities = len(frame[keys].drop_duplicates())
        rows_per_entity = len(frame) / max(n_entities, 1)
        if grain == GRAIN_ACCOUNT and n_entities < len(frame):
            report.findings.append(Finding(
                "GRAIN_MISMATCH", SEV_FAIL,
                f"declared grain is {GRAIN_ACCOUNT!r} (one row per entity) but "
                f"{len(frame):,} rows cover only {n_entities:,} entities "
                f"({rows_per_entity:.1f} rows each). This is transaction-grain "
                f"data, or an aggregate with duplicates -- either way every "
                f"per-entity rate computed from it would be wrong.",
                {"n_rows": len(frame), "n_entities": n_entities,
                 "rows_per_entity": rows_per_entity}))
        elif grain != GRAIN_ACCOUNT and rows_per_entity == 1.0:
            report.findings.append(Finding(
                "GRAIN_MISMATCH", SEV_WARNING,
                f"declared grain is {grain!r} but every entity appears exactly "
                f"once -- this may already be aggregated to account grain.",
                {"n_entities": n_entities}))

    # -- duplicates ---------------------------------------------------------
    if keys and timestamp_column:
        dup_cols = keys + [timestamp_column]
        n_dup = int(frame.duplicated(subset=dup_cols).sum())
        if n_dup:
            report.findings.append(Finding(
                "DUPLICATE_KEYS", SEV_FAIL,
                f"{n_dup:,} rows repeat the same {dup_cols} combination. "
                f"Duplicated transactions double-count in every aggregate and "
                f"inflate whichever class they fall in.",
                {"n_duplicates": n_dup, "on": dup_cols}))

    # -- window coverage ----------------------------------------------------
    span = _span_days(frame[timestamp_column]) if timestamp_column else None
    if span is not None and story.lookback_months > 0:
        expected = story.lookback_months * 30.44
        coverage = span / expected if expected else None
        if coverage is not None and coverage < MIN_WINDOW_COVERAGE:
            report.findings.append(Finding(
                "WINDOW_TOO_SHORT", SEV_FAIL,
                f"the sample spans {span:.0f} days but the story declares a "
                f"{story.lookback_months}-month lookback (~{expected:.0f} days) "
                f"-- {coverage:.0%} coverage. Any recency, velocity or "
                f"rolling-window feature will be computed over a shorter "
                f"history than the design assumes.",
                {"span_days": span, "expected_days": expected,
                 "coverage": coverage}))

    # -- per-class windows (the originally reported bug) --------------------
    if timestamp_column and label_column in frame.columns:
        spans = {}
        for cls in (1, 0):
            sub = frame[frame[label_column] == cls]
            if len(sub):
                s = _span_days(sub[timestamp_column])
                if s is not None:
                    spans[cls] = s
        if len(spans) == 2 and min(spans.values()) >= 0:
            wide, narrow = max(spans.values()), min(spans.values())
            ratio = (wide / narrow) if narrow > 0 else float("inf")
            if ratio >= CLASS_SPAN_RATIO_ALERT:
                wide_cls = max(spans, key=lambda c: spans[c])
                report.findings.append(Finding(
                    "CLASS_WINDOW_MISMATCH", SEV_FAIL,
                    f"the two classes cover very different periods: "
                    f"label={wide_cls} spans {wide:.0f}d, label="
                    f"{1 - wide_cls} spans {narrow:.0f}d ({ratio:.0f}x "
                    f"narrower). Every time-derived feature will separate the "
                    f"classes by construction rather than by fraud behaviour, "
                    f"and the class ratio is an artifact of the draw.",
                    {"spans_days": {str(k): v for k, v in spans.items()},
                     "ratio": ratio}))

    # -- labels and prevalence ---------------------------------------------
    if label_column not in frame.columns:
        report.findings.append(Finding(
            "UNLABELLED_FRACTION", SEV_WARNING,
            f"no {label_column!r} column -- the sample is unlabelled, so no "
            f"prevalence claim can be made about it. Run the labeller first.",
            {}))
    else:
        n_na = int(pd.isna(frame[label_column]).sum())
        frac = n_na / len(frame)
        if frac > MAX_UNLABELLED_FRACTION:
            report.findings.append(Finding(
                "UNLABELLED_FRACTION", SEV_WARNING,
                f"{frac:.0%} of rows are unlabelled ({n_na:,}/{len(frame):,}). "
                f"A prevalence computed over the remainder describes a small "
                f"labelled subset, not this sample.",
                {"n_unlabelled": n_na, "fraction": frac}))

        measurement = measure_prevalence(
            frame, grain=grain, label_column=label_column,
            mature_column=mature_column)
        chk = check_prevalence(measurement, story)
        sev = {
            STATUS_PASS: SEV_PASS, STATUS_WARNING: SEV_WARNING,
            STATUS_FAIL: SEV_FAIL, STATUS_UNDECLARED: SEV_WARNING,
            STATUS_INCOMPARABLE: SEV_WARNING,
        }[chk.status]
        if sev != SEV_PASS:
            report.findings.append(Finding(
                "PREVALENCE", sev, chk.message, chk.to_dict()))

    # -- label maturity -----------------------------------------------------
    if observation_date is not None and timestamp_column:
        horizon = pd.Timestamp(observation_date) - pd.Timedelta(
            days=story.label_maturity_days)
        ts = pd.to_datetime(frame[timestamp_column], errors="coerce")
        n_immature = int((ts >= horizon).sum())
        if n_immature:
            report.findings.append(Finding(
                "IMMATURE_TAIL", SEV_WARNING,
                f"{n_immature:,} rows fall inside the "
                f"{story.label_maturity_days}-day label-maturity horizon "
                f"(after {horizon.date()}). Fraud there may simply not have "
                f"been reported yet, so those rows under-state fraud and must "
                f"not be treated as legitimate.",
                {"n_immature": n_immature, "horizon": str(horizon.date())}))

    return report
