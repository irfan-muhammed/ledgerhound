"""
sentinel.monitor.drift_sentinel
================================

V3-024 -- A12 Drift Sentinel wired as an agent.

Wires two libraries that already existed but were never connected
(the roadmap's "In-place check": PSI/monitor libraries exist but nothing
schedules them, nothing reads card baselines, nothing re-opens a plan):

  - ``sentinel.sampling.psi`` -- PSI/KS computation, population-vs-sample.
  - ``sentinel.knowledge.dataset_card.DatasetCard.reference_distributions``
    -- a field that existed since V3-011 with a setter
    (``set_reference_distribution``) but zero callers anywhere in the
    codebase until ``capture_baseline`` below.

``DriftSentinel.sweep()`` is one deterministic unit of work, structurally
mirroring ``sentinel.monitor.monitor.DriftMonitor.run_cycle()`` (config-like
thresholds on the instance, one call per cycle, always returns a result
record even when nothing fired): it compares a current data sample +
current eval scores against the dataset card's captured baseline, and --
only when evidence crosses a real threshold -- injects a ``PlanGraph``
node, records a ``Finding`` (audit trail), and writes a ``Lesson`` (the
"what fixed it last time" procedural memory the roadmap asks for, recalled
into this and future episodes via ``LessonMemory.find``).

Routing table (evidence-routed, per the roadmap's three named rules) --
priority order is a deliberate design decision, not arbitrary, since more
than one condition can fire in the same sweep:

  1. **calibration decayed** (newly flagged vs. baseline) -> ``re_threshold``.
     Checked first: it is the cheapest fix and doesn't implicate the
     features or the model itself.
  2. **key-feature PSI fail** with eval score stable -> ``re_engineer_feature``.
     Inputs drifted but the model is still performing -- the specific
     feature family needs work, not a full retrain.
  3. **eval score dropped** (>= ``eval_drop_threshold``) with watched
     features stable -> ``full_retrain``. Concept drift: no feature to
     blame, the model itself needs refreshing.
  4. **both** a key-feature PSI fail and an eval drop fire together ->
     ``full_retrain`` (folded into rule 3's route, but reported with its
     own reason string). This is the roadmap's literal "trigger = PSI>0.2
     AND eval drop" phrase, read as the top-severity compound case rather
     than a fourth independent action.

Below every threshold -> ``route=None``: no plan node, no finding, no
lesson. This is deliberate, not an oversight -- the close-out explicitly
requires "no alert fatigue" on ordinary sub-threshold drift.

Calibration "decay" is defined as a transition (baseline not-flagged ->
current flagged), not "current flagged" in isolation -- V3-022's closure
disclosed that ``calibration_flagged`` fires on nearly every honest
imbalanced-fraud model this pipeline trains, so treating "flagged" alone
as decay would fire the re-threshold route on every single sweep of a
dataset whose baseline was already (correctly) flagged. Requiring a
transition catches genuine *new* decay while not re-alarming on a known,
already-disclosed condition. Documented v1 limitation, not fixed: a model
that was already flagged and gets *more* miscalibrated produces no new
decay signal here.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sentinel.knowledge.dataset_card import DatasetCard
from sentinel.plan.graph import PlanGraph
from sentinel.sampling.psi import PSICalculator, PSIStatus, compute_bucket_bounds

ROUTE_TO_NODE_TYPE = {
    "re_threshold": "verification",
    "re_engineer_feature": "analysis",
    "full_retrain": "model",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_id() -> str:
    return f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _numeric_population_pct(series: pd.Series, bounds: list[float]) -> dict[int, float]:
    clean = series.dropna().astype(float)
    if len(clean) == 0 or len(bounds) < 2:
        return {}
    bins = pd.cut(clean, bins=bounds, include_lowest=True, right=False, labels=False)
    counts = bins.value_counts(normalize=True)
    return {int(idx): float(pct) for idx, pct in counts.items() if pd.notna(idx)}


def capture_baseline(
    card: DatasetCard,
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    categorical_cols: Optional[list[str]] = None,
    eval_scores: Optional[dict] = None,
    n_buckets: int = 10,
) -> None:
    """Populate ``card.reference_distributions`` in exactly the shape
    ``PSICalculator.numeric``/``.categorical`` need (population_dist +
    bucket bounds), plus ``card.last_eval`` for the eval-drop half of the
    trigger. Nothing did this before V3-024: ``set_reference_distribution``
    existed since V3-011 with no caller anywhere in the codebase.

    Call once to establish "what normal looks like" for a dataset; a later
    ``DriftSentinel.sweep()`` compares a new sample against what this call
    stored. Re-calling replaces the baseline for the given columns (e.g.
    after a confirmed, intentional distribution shift) -- it does not
    merge or average with the previous baseline.
    """
    for col in (numeric_cols or []):
        if col not in df.columns:
            continue
        bounds = compute_bucket_bounds(df[col], n_buckets=n_buckets)
        dist = _numeric_population_pct(df[col], bounds)
        card.set_reference_distribution(col, {
            "feature_type": "numeric",
            "population_dist": dist,
            "bounds": bounds,
            "n_population": int(df[col].notna().sum()),
            "captured_at": _now(),
        })
    for col in (categorical_cols or []):
        if col not in df.columns:
            continue
        vc = df[col].dropna().astype(str).value_counts(normalize=True)
        card.set_reference_distribution(col, {
            "feature_type": "categorical",
            "population_dist": {str(k): float(v) for k, v in vc.items()},
            "n_population": int(df[col].notna().sum()),
            "captured_at": _now(),
        })
    if eval_scores is not None:
        card.set_last_eval(eval_scores)


@dataclass
class DriftEpisode:
    """One sweep's outcome -- always produced, even when nothing fired
    (mirrors ``MonitorRunResult``'s "a result record every cycle" shape)."""

    episode_id: str = field(default_factory=_episode_id)
    dataset_name: str = ""
    triggered_at: str = field(default_factory=_now)
    route: Optional[str] = None   # "re_threshold" | "re_engineer_feature" | "full_retrain" | None
    reason: str = ""
    psi_summary: dict = field(default_factory=dict)       # {feature: {psi, status}}
    failing_features: list[str] = field(default_factory=list)
    eval_drop: Optional[float] = None
    calibration_decayed: bool = False
    plan_node_id: Optional[str] = None
    finding_id: Optional[str] = None
    past_lessons: list[str] = field(default_factory=list)  # recalled BEFORE this episode's own lesson is written
    status: str = "OK"           # "OK" | "ERROR"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save_to_log(self, log_path: str | Path) -> None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")


class DriftSentinel:
    """Evidence-routed drift sweep. See module docstring for the routing
    table and why calibration "decay" requires a transition, not a bare
    flag."""

    def __init__(
        self,
        psi_fail_threshold: float = 0.20,
        eval_drop_threshold: float = 0.03,   # 3 points -- low end of the roadmap's "3-5pt" band
        eval_metric: str = "pr_auc",
        key_features: Optional[list[str]] = None,
        findings_log: Any = None,     # sentinel.findings.log.FindingsLog, optional
        lesson_memory: Any = None,    # sentinel.memory.lesson_memory.LessonMemory, optional
    ) -> None:
        self.psi_fail_threshold = psi_fail_threshold
        self.eval_drop_threshold = eval_drop_threshold
        self.eval_metric = eval_metric
        self.key_features = list(key_features) if key_features else None
        self.findings_log = findings_log
        self.lesson_memory = lesson_memory
        self._psi = PSICalculator()

    def sweep(
        self,
        card: DatasetCard,
        plan: PlanGraph,
        current_df: pd.DataFrame,
        current_eval: Optional[dict] = None,
        dataset_name: str = "",
        run_id: str = "",
    ) -> DriftEpisode:
        episode = DriftEpisode(dataset_name=dataset_name)

        numeric_pop, cat_pop, bounds = self._baselines_from_card(card)
        if not numeric_pop and not cat_pop:
            episode.status = "ERROR"
            episode.error = ("no PSI baseline on this dataset card -- call "
                             "capture_baseline() before the first sweep")
            return episode

        weights = {f: 3.0 for f in self.key_features} if self.key_features else None
        psi_result = self._psi.batch(current_df, numeric_pop, cat_pop, bounds, weights=weights)
        episode.psi_summary = {f.feature: {"psi": round(f.psi, 6), "status": f.status.value}
                               for f in psi_result.features}

        watched = set(self.key_features) if self.key_features else {f.feature for f in psi_result.features}
        if self.psi_fail_threshold == 0.20:
            failing = [f.feature for f in psi_result.features
                      if f.feature in watched and f.status == PSIStatus.FAIL]
        else:
            # a caller-supplied threshold overrides psi.py's fixed 0.20
            # bucket boundary
            failing = [f.feature for f in psi_result.features
                      if f.feature in watched and f.psi > self.psi_fail_threshold]
        episode.failing_features = failing
        key_feature_fail = bool(failing)

        eval_drop = None
        eval_dropped = False
        baseline_eval = dict(card.last_eval or {})
        if current_eval is not None and baseline_eval:
            baseline_score = baseline_eval.get(self.eval_metric)
            current_score = current_eval.get(self.eval_metric)
            if baseline_score is not None and current_score is not None:
                eval_drop = float(baseline_score) - float(current_score)
                eval_dropped = eval_drop >= self.eval_drop_threshold
        episode.eval_drop = eval_drop

        calibration_decayed = bool(
            current_eval and current_eval.get("calibration_flagged")
            and not baseline_eval.get("calibration_flagged")
        )
        episode.calibration_decayed = calibration_decayed

        route, reason = self._route(calibration_decayed, key_feature_fail, eval_dropped,
                                    failing, eval_drop)
        episode.route = route
        episode.reason = reason

        if route is None:
            return episode  # no plan node, no finding, no lesson -- no alert fatigue

        if self.lesson_memory is not None:
            past = self.lesson_memory.find(dataset_name, query=f"{route} drift {' '.join(failing)}")
            episode.past_lessons = [l.text for l in past]

        node = plan.add_node(text=f"Drift response ({route}): {reason}",
                             type=ROUTE_TO_NODE_TYPE[route])
        if isinstance(node, str):
            episode.status = "ERROR"
            episode.error = f"plan_graph.add_node failed: {node}"
        else:
            episode.plan_node_id = node.id

        if self.findings_log is not None:
            self._record_finding(episode, dataset_name, route, reason, failing, eval_drop)

        if self.lesson_memory is not None:
            self.lesson_memory.add(
                text=f"{route} was triggered on {dataset_name}: {reason}",
                dataset=dataset_name, run_id=run_id or episode.episode_id,
                tags=["drift_episode", route],
            )
            # LessonMemory.add() only mutates the in-process index --
            # without this, "surfaced by recall on the next drift" only
            # holds within one process, not across scheduled cycles (a
            # fresh DriftSentinel/LessonMemory next cron run would find
            # nothing on disk). Mirrors FindingsLog.append's self-persist.
            self.lesson_memory.save()

        return episode

    def _route(self, calibration_decayed: bool, key_feature_fail: bool, eval_dropped: bool,
              failing: list[str], eval_drop: Optional[float]) -> tuple[Optional[str], str]:
        if calibration_decayed:
            return "re_threshold", ("calibration newly flagged vs. baseline -- "
                                    "re-threshold before touching features/model")
        if key_feature_fail and eval_dropped:
            return "full_retrain", (f"compound drift: PSI fail on {failing} AND "
                                    f"{self.eval_metric} dropped {eval_drop:.3f} -- "
                                    f"most severe case, full retrain")
        if key_feature_fail:
            return "re_engineer_feature", (f"PSI fail on {failing} with {self.eval_metric} "
                                           f"stable -- re-engineer that feature family")
        if eval_dropped:
            return "full_retrain", (f"{self.eval_metric} dropped {eval_drop:.3f} with all "
                                    f"watched features stable -- concept drift, full retrain")
        return None, "no feature or eval evidence crossed threshold -- no trigger"

    def _record_finding(self, episode: DriftEpisode, dataset_name: str, route: str,
                        reason: str, failing: list[str], eval_drop: Optional[float]) -> None:
        from sentinel.findings.models import Evidence, Finding, SampleFrame
        finding = Finding(
            claim=f"Drift episode on {dataset_name}: {reason}",
            evidence=Evidence(summary=f"PSI: {episode.psi_summary}; eval_drop={eval_drop}"),
            sample_frame=SampleFrame.UNSPECIFIED,
            confidence=0.9,
            dataset=dataset_name,
            tags=(["drift_episode", f"route:{route}"]
                 + [f"feature:{f}" for f in failing]),
        )
        self.findings_log.append(finding)
        episode.finding_id = finding.id

    def _baselines_from_card(self, card: DatasetCard) -> tuple[dict, dict, dict]:
        numeric_pop: dict = {}
        cat_pop: dict = {}
        bounds: dict = {}
        for col, stats in (card.reference_distributions or {}).items():
            if not isinstance(stats, dict):
                continue
            if stats.get("feature_type") == "numeric":
                # JSON round-trip turns numeric-dict keys into strings --
                # convert back, since PSICalculator.numeric indexes by int.
                numeric_pop[col] = {int(k): v for k, v in stats.get("population_dist", {}).items()}
                bounds[col] = stats.get("bounds", [])
            elif stats.get("feature_type") == "categorical":
                cat_pop[col] = stats.get("population_dist", {})
        return numeric_pop, cat_pop, bounds
