"""
ledgerhound.validation.validator
================================

V3-022: A10 Independent Validator + model card.

"Fresh-context agent, never the modeler"
-------------------------------------------
Realized the same way V3-018's "separate agent context" was: not via
actual LLM sub-agent orchestration, but via a hard architectural rule a
shared reasoning trace couldn't talk itself around. Here the rule is
RECOMPUTE, DON'T COPY -- `ModelValidator` re-scores the fitted booster
against OOT data itself and derives every metric from those scores. It
never lifts `TrainReport.metrics` into the card. A card populated from the
modeler's own self-report would look complete while failing the entire
point of an independent check (the modeler grading its own homework). The
one place the self-report is consulted at all is a cross-check: the
validator's own OOT PR-AUC/ROC-AUC is compared against
`train_report.metrics["oot"]`, and a mismatch is recorded as a finding,
not silently accepted.

Refuse vs. report -- read literally from the roadmap's two separate clauses
-----------------------------------------------------------------------------
"refuses a model lacking an OOT split object or leakage verdict" (ABSENCE)
is a different requirement from "card contains ... leakage verdicts"
(PRESENCE, reported as-is). This validator:
  - refuses (raises `ValidatorRefused`, or the underlying
    `NotATemporalSplitError` for a bad split) when the split object or the
    leakage verdict is MISSING;
  - does NOT refuse when a present leakage verdict happens to be
    `leakage_blocked=True` -- that gets written into the card's leakage
    section honestly, not hidden and not treated as a hard failure.
This distinction matters concretely: V3-021's own real-PaySim run carries
a `leakage_blocked=True` verdict that V3-021's closure diagnosed as a
FALSE POSITIVE (the AUROC-gap check firing on ordinary OOT drift, not a
real leak -- see the v3 roadmap's V3-021 section). If this validator
hard-failed on any blocked verdict, it would be unable to validate the
only real model this session has trained, and the honest fix (loosening
the gap check's tolerance) was explicitly deferred, not rushed here.

A model card is honest, not necessarily flattering
------------------------------------------------------
"A human DS signs off without edits" means the CARD accurately describes
the model -- not that the model is good. A weak, miscalibrated model with
a disputed leakage flag, reported plainly with all three facts visible,
is a passing close-out for the validator. Nothing here tunes a model to
look better; the calibration check is supposed to flag a miscalibrated
model, and it does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ledgerhound.validation.temporal_split import TemporalSplit, require_temporal_split

log = logging.getLogger(__name__)


class ValidatorRefused(RuntimeError):
    """Raised when the validator is missing something it cannot proceed
    without: a fitted model, or a leakage verdict. (A missing/invalid
    split raises NotATemporalSplitError directly, from
    require_temporal_split -- not wrapped, so the caller sees the same
    typed error V3-019/V3-021 already use.)"""


@dataclass
class SliceMetric:
    slice_col: str
    slice_value: str
    n: int
    n_positive: int
    pr_auc: Optional[float]
    roc_auc: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationBin:
    bin_index: int
    n: int
    predicted_mean: float
    observed_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    generated_at: str
    backend: str
    n_oot: int
    n_oot_positive: int
    pr_auc: float
    roc_auc: float
    precision_at_k: dict            # {"10": 0.3, "50": 0.12, ...}
    cost_curve: list                # [{"threshold":.., "cost":.., "n_flagged":..}]
    calibration_bins: list          # [CalibrationBin.to_dict(), ...]
    calibration_flagged: bool
    calibration_flag_reason: str
    slices: list                    # [SliceMetric.to_dict(), ...]
    slice_notes: list               # e.g. ["geo: not available in this dataset"]
    threshold: Optional[float]
    threshold_rationale: str
    cost_matrix: dict
    leakage_verdict: dict           # {"leakage_blocked": bool, "pre_fit": {...}, "post_fit": {...}}
    unstable_features: list
    selection_report_provided: bool
    n_candidate_features: Optional[int]
    feature_cols: list
    split_summary: str
    seeds_versions: dict
    self_report_divergence: Optional[dict]
    findings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_MANDATORY_CARD_SECTIONS = [
    "Data", "Split", "Features & Stability", "Metrics & Slices",
    "Threshold Rationale", "Leakage Verdicts", "Seeds & Versions",
]


class ModelValidator:
    """Independent, self-scoring validator for a V3-021 `TrainReport`.

    Parameters
    ----------
    top_k_list          : counts for precision@top-k (e.g. "investigate
                          the top 100 riskiest OOT transactions").
    calibration_n_bins  : quantile bins for the reliability curve.
    calibration_tolerance: max allowed |predicted_mean - observed_rate| in
                          any bin with >= calibration_min_bin_n samples
                          before the model is flagged miscalibrated.
    calibration_min_bin_n: bins smaller than this are ignored by the
                          miscalibration check (too few samples for the
                          observed rate to mean anything).
    """

    def __init__(
        self,
        top_k_list: tuple = (10, 50, 100),
        calibration_n_bins: int = 10,
        calibration_tolerance: float = 0.15,
        calibration_min_bin_n: int = 20,
    ) -> None:
        self.top_k_list = top_k_list
        self.calibration_n_bins = calibration_n_bins
        self.calibration_tolerance = calibration_tolerance
        self.calibration_min_bin_n = calibration_min_bin_n

    def validate(
        self,
        train_report,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        split,
        selection_report=None,
        slice_cols: Optional[list[str]] = None,
        dataset_name: str = "unnamed dataset",
    ) -> ValidationReport:
        split = require_temporal_split(split)   # real caller #3

        if getattr(train_report, "leakage_pre_fit", None) is None:
            raise ValidatorRefused(
                "train_report has no leakage_pre_fit verdict -- refusing to "
                "validate a model with no leakage assessment on record."
            )
        booster = getattr(train_report, "booster", None)
        if booster is None:
            raise ValidatorRefused(
                "train_report carries no fitted booster -- nothing to "
                "independently re-score."
            )

        from ledgerhound.modeling.gbm import _BACKENDS
        _, predict_fn = _BACKENDS[train_report.backend]

        train_df, oot_df = split.apply(df)
        y_oot = oot_df[label_col].reset_index(drop=True)
        scores_oot = np.asarray(predict_fn(booster, oot_df[feature_cols]))

        from sklearn.metrics import average_precision_score, roc_auc_score

        pr_auc = float(average_precision_score(y_oot, scores_oot))
        roc_auc = float(roc_auc_score(y_oot, scores_oot)) if y_oot.nunique() > 1 else float("nan")

        precision_at_k = self._precision_at_k(y_oot, scores_oot)
        cost_matrix = getattr(train_report, "cost_matrix", None) or {"fp_cost": 1.0, "fn_cost": 25.0}
        cost_curve = self._cost_curve(y_oot, scores_oot, cost_matrix)
        calibration_bins, flagged, reason = self._calibration_check(y_oot, scores_oot)
        slices, slice_notes = self._slice_analysis(oot_df, y_oot, scores_oot, feature_cols, slice_cols)

        # Carried over as-is (not recomputed -- these are the trainer's own
        # process notes, e.g. V3-023's "no L0 framing captured" warning,
        # not a metric "never the modeler" would need to re-derive).
        findings = list(getattr(train_report, "findings", None) or [])
        self_report_divergence = None
        self_metrics = getattr(train_report, "metrics", None) or {}
        oot_self = self_metrics.get("oot") if isinstance(self_metrics, dict) else None
        if oot_self:
            pr_diff = abs(oot_self.get("pr_auc", pr_auc) - pr_auc)
            roc_diff = abs(oot_self.get("roc_auc", roc_auc) - roc_auc) if roc_auc == roc_auc else 0.0
            self_report_divergence = {"pr_auc_diff": pr_diff, "roc_auc_diff": roc_diff}
            if pr_diff > 1e-6 or roc_diff > 1e-6:
                findings.append(
                    f"Independent OOT re-scoring diverges from the modeler's self-report "
                    f"(pr_auc diff={pr_diff:.6f}, roc_auc diff={roc_diff:.6f}) -- investigate."
                )

        leakage_verdict = {
            "leakage_blocked": bool(getattr(train_report, "leakage_blocked", False)),
            "pre_fit": train_report.leakage_pre_fit,
            "post_fit": getattr(train_report, "leakage_post_fit", None),
        }
        if leakage_verdict["leakage_blocked"]:
            findings.append(
                "leakage_blocked=True on this model -- reported, not suppressed. "
                "See the training run's leakage_post_fit detail for the diagnosed cause "
                "before treating this as either a real leak or a false positive."
            )

        unstable_features = list(getattr(selection_report, "unstable_features", []) or [])
        selection_report_provided = selection_report is not None
        n_candidate_features = None
        if selection_report_provided and getattr(selection_report, "stages", None):
            first_stage = selection_report.stages[0]
            n_candidate_features = len(first_stage.kept) + len(first_stage.dropped)

        report = ValidationReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            backend=train_report.backend,
            n_oot=len(y_oot),
            n_oot_positive=int(y_oot.sum()),
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            precision_at_k=precision_at_k,
            cost_curve=cost_curve,
            calibration_bins=[b.to_dict() for b in calibration_bins],
            calibration_flagged=flagged,
            calibration_flag_reason=reason,
            slices=[s.to_dict() for s in slices],
            slice_notes=slice_notes,
            threshold=getattr(train_report, "threshold", None),
            threshold_rationale=(
                f"Calibrated on OOT scores to minimize "
                f"fp_cost*FP + fn_cost*FN given cost_matrix={cost_matrix}. "
                + (
                    "Cost matrix source: L0 framing captured at study start "
                    "(ledgerhound.framing.l0.L0Framing, source='captured')."
                    if getattr(train_report, "framing_source", "default") == "captured"
                    else "Cost matrix source: DOCUMENTED DEFAULT -- no L0 framing "
                    "was captured for this study (see Findings)."
                )
            ),
            cost_matrix=cost_matrix,
            leakage_verdict=leakage_verdict,
            unstable_features=unstable_features,
            selection_report_provided=selection_report_provided,
            n_candidate_features=n_candidate_features,
            feature_cols=list(feature_cols),
            split_summary=split.summary_block(),
            seeds_versions={
                "random_state": getattr(train_report, "random_state", None),
                "split_id": getattr(train_report, "split_id", None),
                "backend": train_report.backend,
                "params": getattr(train_report, "params", {}),
                "dataset_name": dataset_name,
            },
            self_report_divergence=self_report_divergence,
            findings=findings,
        )
        if flagged:
            report.findings.append(f"Calibration check flagged this model: {reason}")
        return report

    # -- metrics ---------------------------------------------------------------

    def _precision_at_k(self, y_true: pd.Series, scores: np.ndarray) -> dict:
        order = np.argsort(-scores)
        y_sorted = y_true.values[order]
        out = {}
        for k in self.top_k_list:
            kk = min(k, len(y_sorted))
            out[str(k)] = float(y_sorted[:kk].sum() / kk) if kk else None
        return out

    def _cost_curve(self, y_true: pd.Series, scores: np.ndarray, cost_matrix: dict,
                     n_points: int = 30) -> list:
        y = y_true.values
        candidates = np.quantile(scores, np.linspace(0, 1, n_points))
        curve = []
        for t in np.unique(candidates):
            pred = (scores >= t).astype(int)
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            cost = cost_matrix["fp_cost"] * fp + cost_matrix["fn_cost"] * fn
            curve.append({"threshold": float(t), "n_flagged": int(pred.sum()),
                         "fp": fp, "fn": fn, "cost": float(cost)})
        return curve

    def _calibration_check(self, y_true: pd.Series, scores: np.ndarray):
        y = y_true.values
        try:
            bin_idx = pd.qcut(scores, self.calibration_n_bins, labels=False, duplicates="drop")
        except ValueError:
            bin_idx = np.zeros(len(scores), dtype=float)
        bin_idx = np.asarray(bin_idx, dtype=float)
        # A collapsed/duplicate-edge qcut can leave some rows unassigned
        # (NaN bin code) when too many scores tie at a bin boundary --
        # those rows are excluded from the reliability curve rather than
        # crashing on int(nan).
        valid = ~np.isnan(bin_idx)
        bin_idx, y, scores = bin_idx[valid], y[valid], scores[valid]

        bins = []
        flagged, reason = False, "within tolerance"
        if len(bin_idx) == 0:
            return bins, flagged, "no valid calibration bins (all scores tied)"
        for b in sorted(pd.unique(bin_idx)):
            mask = bin_idx == b
            n = int(mask.sum())
            predicted_mean = float(scores[mask].mean())
            observed_rate = float(y[mask].mean())
            bins.append(CalibrationBin(bin_index=int(b), n=n,
                                       predicted_mean=predicted_mean, observed_rate=observed_rate))
            if n >= self.calibration_min_bin_n:
                gap = abs(predicted_mean - observed_rate)
                if gap > self.calibration_tolerance:
                    flagged = True
                    reason = (f"bin {b}: predicted mean {predicted_mean:.3f} vs "
                             f"observed rate {observed_rate:.3f} (gap {gap:.3f} > "
                             f"tolerance {self.calibration_tolerance:.3f}, n={n})")
        return bins, flagged, reason

    def _slice_analysis(self, oot_df: pd.DataFrame, y_true: pd.Series, scores: np.ndarray,
                        feature_cols: list[str], slice_cols: Optional[list[str]]):
        from sklearn.metrics import average_precision_score, roc_auc_score

        slices: list[SliceMetric] = []
        notes: list[str] = []
        requested = slice_cols or []
        for col in requested:
            if col not in oot_df.columns:
                notes.append(f"{col}: not available in this dataset")
                continue
            series = oot_df[col].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 6:
                try:
                    buckets = pd.qcut(series, 4, duplicates="drop")
                except ValueError:
                    buckets = series.astype(str)
            else:
                buckets = series.astype(str)

            for value in pd.unique(buckets.astype(str)):
                mask = (buckets.astype(str) == value).values
                n = int(mask.sum())
                n_pos = int(y_true.values[mask].sum())
                if n == 0:
                    continue
                pr = (float(average_precision_score(y_true.values[mask], scores[mask]))
                     if 0 < n_pos < n else None)
                roc = (float(roc_auc_score(y_true.values[mask], scores[mask]))
                      if 0 < n_pos < n else None)
                slices.append(SliceMetric(slice_col=col, slice_value=str(value), n=n,
                                          n_positive=n_pos, pr_auc=pr, roc_auc=roc))

        return slices, notes

    # -- model card --------------------------------------------------------------

    def render_model_card(self, report: ValidationReport) -> str:
        lines = ["# Model Card", "", f"_Generated {report.generated_at}_", ""]

        lines += ["## Data", "",
                  f"- Dataset: {report.seeds_versions.get('dataset_name', 'unnamed dataset')}",
                  f"- OOT rows: {report.n_oot:,} ({report.n_oot_positive:,} positive)",
                  f"- Features: {', '.join(report.feature_cols)}", ""]

        lines += ["## Split", "", report.split_summary, ""]

        lines += ["## Features & Stability", ""]
        if not report.selection_report_provided:
            lines.append("- No SelectionReport was provided to the validator -- "
                         "feature stability was NOT checked for this run.")
        else:
            if report.n_candidate_features is not None:
                lines.append(f"- Selection stack kept {len(report.feature_cols)} of "
                             f"{report.n_candidate_features} candidate features.")
            if report.unstable_features:
                lines.append(f"- Checked; flagged unstable by cross-time stability: "
                             f"{', '.join(report.unstable_features)}")
            else:
                lines.append("- Checked; no features flagged unstable.")
        lines.append("")

        prevalence = report.n_oot_positive / report.n_oot if report.n_oot else 0.0
        lines += ["## Metrics & Slices", "",
                  f"- OOT prevalence (base rate): {prevalence:.4%} "
                  f"({report.n_oot_positive:,} / {report.n_oot:,}) -- "
                  f"anchor for reading PR-AUC below (a non-informative model scores "
                  f"PR-AUC ≈ prevalence).",
                  f"- PR-AUC (primary): {report.pr_auc:.4f}",
                  f"- ROC-AUC (secondary): {report.roc_auc:.4f}",
                  f"- Precision@k: " + ", ".join(
                      f"top-{k}={v:.3f}" if v is not None else f"top-{k}=n/a"
                      for k, v in report.precision_at_k.items()),
                  ""]
        if report.slices:
            lines.append("| Slice | Value | n | n_positive | PR-AUC | ROC-AUC |")
            lines.append("|---|---|---|---|---|---|")
            for s in report.slices:
                pr = f"{s['pr_auc']:.3f}" if s["pr_auc"] is not None else "n/a"
                roc = f"{s['roc_auc']:.3f}" if s["roc_auc"] is not None else "n/a"
                lines.append(f"| {s['slice_col']} | {s['slice_value']} | {s['n']} | "
                             f"{s['n_positive']} | {pr} | {roc} |")
            lines.append("")
        for note in report.slice_notes:
            lines.append(f"- {note}")
        lines.append(f"- Calibration: {'FLAGGED -- ' + report.calibration_flag_reason if report.calibration_flagged else 'within tolerance'}")
        if len(report.calibration_bins) <= 1:
            lines.append(f"- NOTE: the reliability curve collapsed to "
                         f"{len(report.calibration_bins)} bin(s) instead of the requested "
                         f"quantile split -- scores are too saturated/tied to divide "
                         f"further. Still caught gross miscalibration here, but this is a "
                         f"single point, not a curve.")
        lines.append("")

        lines += ["## Threshold Rationale", "",
                  f"- Threshold: {report.threshold}",
                  f"- Cost matrix: {report.cost_matrix}",
                  f"- {report.threshold_rationale}", ""]

        lines += ["## Leakage Verdicts", "",
                  f"- leakage_blocked: {report.leakage_verdict['leakage_blocked']}",
                  f"- Pre-fit: {report.leakage_verdict['pre_fit']}",
                  f"- Post-fit (AUROC gap): {report.leakage_verdict['post_fit']}", ""]

        lines += ["## Seeds & Versions", ""]
        for k, v in report.seeds_versions.items():
            lines.append(f"- {k}: {v}")
        lines.append("- NOTE: this section is a summary, not yet a complete replay "
                     "recipe -- it does not record the input data's hash (the way "
                     "`ledgerhound replay`'s events.jsonl does). Reproducing this exact "
                     "run requires the same DataFrame the validator was called with.")
        lines.append("")

        if report.findings:
            lines += ["## Findings", ""]
            for f in report.findings:
                lines.append(f"- {f}")
            lines.append("")

        return "\n".join(lines)

    def save_model_card(self, report: ValidationReport, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_model_card(report), encoding="utf-8")
