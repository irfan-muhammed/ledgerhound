"""
ledgerhound.modeling.registry
===========================

V3-025 -- A11 Champion/challenger registry.

Nothing existed here before this (roadmap's "In-place check": "nothing
exists"). ``ModelRegistry`` is a passive log + comparison + promotion
gate, deliberately NOT an inference server: "shadow-deploy config (scores
logged, not served)" is read literally -- the registry never scores a
transaction itself. A caller (the real serving path, or a backtest
script) computes both the champion's and the challenger's score for a
record and hands both to ``log_shadow_prediction``; only the champion's
score is ever actually served/acted on, the challenger's is logged
alongside it purely for later comparison.

Where inputs come from
-----------------------
``register()`` takes a ``ValidationReport`` (``ledgerhound.validation.
validator``), not a ``TrainReport`` -- consistent with V3-022's
"recompute, don't copy" principle: the registry should trust the
INDEPENDENT validator's numbers, not the modeler's self-report. It reads
``backend``, ``threshold``, ``pr_auc``, ``roc_auc``, ``calibration_flagged``,
``seeds_versions["params"]``, and ``leakage_verdict["leakage_blocked"]``
by attribute/key access (duck-typed against those exact names), so any
object with that shape works, not only a literal ``ValidationReport``
instance.

Matured labels
--------------
No "label maturation" concept existed anywhere in the codebase before
this (confirmed by search) -- ``TemporalSplit.embargo_gap_days`` treats
label maturity as a prose concept baked into the embargo gap, not a
reusable API. Here it is made explicit and caller-driven: a shadow log
entry's ``matured_label`` starts ``None`` (label not yet known -- exactly
today's transaction) and is backfilled later via ``mature_label()`` (a
later job -- chargeback resolved, investigation closed) supplying the
real outcome. ``compare()`` only computes PR-AUC/ROC-AUC over rows where
a label has actually matured, and reports ``n_total`` vs. ``n_matured``
honestly rather than silently dropping the unmatured majority.

Promotion is HITL-gated; rollback and bootstrap are not
----------------------------------------------------------
The true invariant (not "promote() is the only method that touches
``active_champion``" -- ``register()``'s bootstrap and ``rollback()``
both legitimately reassign it too, and are named exceptions below, not
oversights): **no model can be promoted OVER an existing champion
without going through ``promote()``'s gate.** ``promote()`` always calls
``HITLGate.high()`` and raises ``PromotionDeclined`` if the decision is
not approved, leaving the registry state untouched -- this is the only
code path that replaces a standing champion with a challenger.
``HITLGate.high()`` itself never raises on decline (confirmed against
``ledgerhound.analyst.hitl``: it always returns an ``HITLDecision``, and the
caller must check ``.approved``), so "promotion without HITL approval
impossible" is enforced by this module's own control flow, not by the
gate.

The two deliberate exceptions: ``register()``'s bootstrap only sets
``active_champion`` when a dataset has NONE yet (there is no standing
champion to displace, so there is nothing to gate); ``rollback()`` only
reverts to a champion that was itself already approved by an earlier
``promote()`` call, and has no gate of its own -- it is the safety valve
for an already-approved-but-now-regretted promotion, and a break-glass
action should not itself be blockable. This asymmetry is a deliberate
design decision, not an oversight -- the roadmap's close-out only names
the promote direction as required-gated.

Two things this module does NOT enforce, worth naming rather than
implying otherwise: (1) under the codebase-wide non-interactive default
``auto_yes`` (always returns ``options[0]="proceed"``), ``promote()``
proceeds even when ``comparison.recommendation`` is ``"keep_champion"``
or ``"insufficient_matured_labels"`` -- the recommendation only shapes
the decision PACKAGE shown to a human/gate, it does not itself block a
bad promotion; only a real prompt function that can return the decline
option does that (same contract every other high-tier action in this
codebase already has). (2) ``promote()`` does not require the challenger
to have gone through ``enter_shadow()`` first -- register -> shadow ->
compare -> promote is the tested happy path, not an enforced state
machine; a challenger can be promoted straight from "registered" status.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ledgerhound.analyst.hitl import HITLDecision, HITLGate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_id() -> str:
    return f"m_{uuid.uuid4().hex[:10]}"


class PromotionDeclined(RuntimeError):
    """Raised when a promote() call's HITL decision was not approved.
    The registry is left exactly as it was before the call."""


@dataclass
class RegisteredModel:
    model_id: str
    dataset_name: str
    backend: str
    params: dict
    threshold: Optional[float]
    pr_auc: float
    roc_auc: float
    calibration_flagged: bool
    leakage_blocked: bool
    registered_at: str
    status: str = "registered"     # registered | shadow | champion | retired
    validation_report_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RegisteredModel":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ShadowLogEntry:
    record_id: str
    champion_id: str
    challenger_id: str
    champion_score: float
    challenger_score: float
    matured_label: Optional[int] = None
    logged_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowLogEntry":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ComparisonResult:
    champion_id: Optional[str]
    challenger_id: str
    n_total: int
    n_matured: int
    champion_pr_auc: Optional[float]
    challenger_pr_auc: Optional[float]
    champion_roc_auc: Optional[float]
    challenger_roc_auc: Optional[float]
    recommendation: str    # "promote" | "keep_champion" | "insufficient_matured_labels"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """One JSON file per registry (mirrors ``ledgerhound.workspace.registry``'s
    append-only-JSON pattern): a dict of ``RegisteredModel`` keyed by
    model_id, an ``active_champion`` pointer per dataset, an append-only
    shadow prediction log, and an append-only promotion history (what
    ``rollback()`` reads)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.models: dict[str, RegisteredModel] = {}
        self.active_champion: dict[str, str] = {}
        self.shadow_log: list[ShadowLogEntry] = []
        self.promotion_history: list[dict] = []
        self._load()

    # -- register / shadow -----------------------------------------------

    def register(
        self,
        validation_report: Any,
        dataset_name: str,
        validation_report_path: Optional[str | Path] = None,
    ) -> RegisteredModel:
        """Add a model to the registry. The FIRST model ever registered
        for a given dataset becomes champion automatically (bootstrap --
        otherwise a fresh dataset could never get its first champion,
        since shadow/promote both require an existing champion to compare
        against). Every subsequent registration starts as plain
        "registered" and needs an explicit enter_shadow() + promote()."""
        leakage_verdict = getattr(validation_report, "leakage_verdict", {}) or {}
        seeds_versions = getattr(validation_report, "seeds_versions", {}) or {}
        rm = RegisteredModel(
            model_id=_model_id(),
            dataset_name=dataset_name,
            backend=validation_report.backend,
            params=dict(seeds_versions.get("params", {})),
            threshold=validation_report.threshold,
            pr_auc=validation_report.pr_auc,
            roc_auc=validation_report.roc_auc,
            calibration_flagged=bool(validation_report.calibration_flagged),
            leakage_blocked=bool(leakage_verdict.get("leakage_blocked", False)),
            registered_at=_now(),
            status="champion" if dataset_name not in self.active_champion else "registered",
            validation_report_path=str(validation_report_path) if validation_report_path else None,
        )
        self.models[rm.model_id] = rm
        if dataset_name not in self.active_champion:
            self.active_champion[dataset_name] = rm.model_id
        self._save()
        return rm

    def enter_shadow(self, model_id: str) -> None:
        """Mark a registered (non-champion) model as shadow -- requires an
        existing champion, since "shadow" means "scored alongside, never
        served, compared against the champion"."""
        rm = self._require(model_id)
        champion_id = self.active_champion.get(rm.dataset_name)
        if champion_id is None:
            raise ValueError(f"no champion registered for dataset {rm.dataset_name!r} yet "
                             f"-- shadow requires a champion to compare against")
        if champion_id == model_id:
            raise ValueError(f"{model_id} is already the champion for {rm.dataset_name!r}")
        rm.status = "shadow"
        self._save()

    def log_shadow_prediction(
        self,
        challenger_id: str,
        record_id: str,
        champion_score: float,
        challenger_score: float,
        matured_label: Optional[int] = None,
    ) -> ShadowLogEntry:
        """Log one paired (champion, challenger) score for a record. Only
        the champion's score is ever served -- this is a log, not a
        serving decision. ``matured_label`` is None by default (not yet
        known); backfill it later via ``mature_label()``."""
        rm = self._require(challenger_id)
        if rm.status != "shadow":
            raise ValueError(f"{challenger_id} is not in shadow status "
                             f"(status={rm.status!r}) -- call enter_shadow() first")
        champion_id = self.active_champion.get(rm.dataset_name)
        entry = ShadowLogEntry(
            record_id=record_id, champion_id=champion_id, challenger_id=challenger_id,
            champion_score=float(champion_score), challenger_score=float(challenger_score),
            matured_label=matured_label,
        )
        self.shadow_log.append(entry)
        self._save()
        return entry

    def mature_label(self, record_id: str, challenger_id: str, label: int) -> None:
        """Backfill a label that has arrived after the fact (chargeback
        resolved, investigation closed) onto an already-logged shadow
        prediction. Raises if no matching entry exists -- silently
        no-oping would hide a record_id/challenger_id typo."""
        found = False
        for e in self.shadow_log:
            if e.record_id == record_id and e.challenger_id == challenger_id:
                e.matured_label = int(label)
                found = True
        if not found:
            raise ValueError(f"no shadow log entry for record_id={record_id!r}, "
                             f"challenger_id={challenger_id!r}")
        self._save()

    # -- compare -----------------------------------------------------------

    def compare(self, challenger_id: str, min_matured: int = 30) -> ComparisonResult:
        """PR-AUC/ROC-AUC for champion vs. challenger, computed ONLY over
        rows whose label has matured. Below ``min_matured`` matured rows,
        returns "insufficient_matured_labels" rather than a noisy
        AUC estimate on a handful of points."""
        rm = self._require(challenger_id)
        champion_id = self.active_champion.get(rm.dataset_name)
        entries = [e for e in self.shadow_log if e.challenger_id == challenger_id]
        matured = [e for e in entries if e.matured_label is not None]
        n_total, n_matured = len(entries), len(matured)

        def _insufficient(reason: str) -> ComparisonResult:
            return ComparisonResult(
                champion_id=champion_id, challenger_id=challenger_id,
                n_total=n_total, n_matured=n_matured,
                champion_pr_auc=None, challenger_pr_auc=None,
                champion_roc_auc=None, challenger_roc_auc=None,
                recommendation="insufficient_matured_labels", reason=reason,
            )

        if n_matured < min_matured:
            return _insufficient(f"only {n_matured} matured labels (need >= {min_matured}) "
                                 f"-- cannot compare yet")

        y = [e.matured_label for e in matured]
        if len(set(y)) < 2:
            return _insufficient(f"{n_matured} matured labels are single-class "
                                 f"-- cannot compute PR-AUC/ROC-AUC")

        from sklearn.metrics import average_precision_score, roc_auc_score
        champ_scores = [e.champion_score for e in matured]
        chall_scores = [e.challenger_score for e in matured]
        champ_pr = float(average_precision_score(y, champ_scores))
        chall_pr = float(average_precision_score(y, chall_scores))
        champ_roc = float(roc_auc_score(y, champ_scores))
        chall_roc = float(roc_auc_score(y, chall_scores))

        if chall_pr > champ_pr:
            rec = "promote"
            reason = (f"challenger PR-AUC {chall_pr:.4f} > champion PR-AUC {champ_pr:.4f} "
                     f"on {n_matured} matured labels")
        else:
            rec = "keep_champion"
            reason = (f"challenger PR-AUC {chall_pr:.4f} <= champion PR-AUC {champ_pr:.4f} "
                     f"on {n_matured} matured labels")

        return ComparisonResult(
            champion_id=champion_id, challenger_id=challenger_id,
            n_total=n_total, n_matured=n_matured,
            champion_pr_auc=champ_pr, challenger_pr_auc=chall_pr,
            champion_roc_auc=champ_roc, challenger_roc_auc=chall_roc,
            recommendation=rec, reason=reason,
        )

    # -- promote / rollback --------------------------------------------------

    def promote(
        self,
        challenger_id: str,
        gate: "HITLGate",
        comparison: Optional[ComparisonResult] = None,
    ) -> "HITLDecision":
        """The only code path that can replace a STANDING champion with a
        challenger. Always routes through ``gate.high()``; a non-approved
        decision raises ``PromotionDeclined`` and leaves the registry
        unchanged. (``register()``'s bootstrap and ``rollback()`` also
        assign ``active_champion``, but neither displaces an existing,
        gate-approved champion -- see module docstring.)"""
        rm = self._require(challenger_id)
        champion_id = self.active_champion.get(rm.dataset_name)
        comparison = comparison or self.compare(challenger_id)

        decision = gate.high(
            question=f"Promote {challenger_id} to champion for dataset {rm.dataset_name!r}?",
            findings_so_far=[comparison.reason,
                            f"leakage_blocked={rm.leakage_blocked}, "
                            f"calibration_flagged={rm.calibration_flagged}"],
            recommendation="proceed" if comparison.recommendation == "promote" else "cancel",
            consequence=(f"{challenger_id} becomes the served model for {rm.dataset_name!r}; "
                        f"{champion_id} is retired (reversible via rollback())."),
        )
        if not decision.approved:
            raise PromotionDeclined(f"HITL declined promotion of {challenger_id}: "
                                    f"answer={decision.answer!r}")

        if champion_id is not None and champion_id in self.models:
            self.models[champion_id].status = "retired"
        rm.status = "champion"
        self.active_champion[rm.dataset_name] = challenger_id
        self.promotion_history.append({
            "dataset_name": rm.dataset_name, "from_champion": champion_id,
            "to_challenger": challenger_id, "promoted_at": _now(),
            "comparison": comparison.to_dict(), "rollback": False,
        })
        self._save()
        return decision

    def rollback(self, dataset_name: str) -> str:
        """Revert to the champion active before the most recent promotion
        for this dataset. No HITL gate -- see module docstring. Returns
        the model_id that is now champion again."""
        history = [p for p in self.promotion_history if p["dataset_name"] == dataset_name]
        if not history:
            raise ValueError(f"no promotion history for {dataset_name!r} -- nothing to roll back")
        last = history[-1]
        previous_champion = last["from_champion"]
        if previous_champion is None:
            raise ValueError(f"the first promotion for {dataset_name!r} had no earlier "
                             f"champion -- nothing to roll back to")

        current_champion = self.active_champion.get(dataset_name)
        if current_champion in self.models:
            self.models[current_champion].status = "retired"
        self.models[previous_champion].status = "champion"
        self.active_champion[dataset_name] = previous_champion
        self.promotion_history.append({
            "dataset_name": dataset_name, "from_champion": current_champion,
            "to_challenger": previous_champion, "promoted_at": _now(),
            "comparison": None, "rollback": True,
        })
        self._save()
        return previous_champion

    # -- lookup / persistence ------------------------------------------------

    def _require(self, model_id: str) -> RegisteredModel:
        rm = self.models.get(model_id)
        if rm is None:
            raise ValueError(f"unknown model_id {model_id!r}")
        return rm

    def get(self, model_id: str) -> Optional[RegisteredModel]:
        return self.models.get(model_id)

    def champion_for(self, dataset_name: str) -> Optional[RegisteredModel]:
        model_id = self.active_champion.get(dataset_name)
        return self.models.get(model_id) if model_id else None

    def to_dict(self) -> dict:
        return {
            "models": {mid: m.to_dict() for mid, m in self.models.items()},
            "active_champion": dict(self.active_champion),
            "shadow_log": [e.to_dict() for e in self.shadow_log],
            "promotion_history": list(self.promotion_history),
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.models = {mid: RegisteredModel.from_dict(m) for mid, m in d.get("models", {}).items()}
        self.active_champion = dict(d.get("active_champion", {}))
        self.shadow_log = [ShadowLogEntry.from_dict(e) for e in d.get("shadow_log", [])]
        self.promotion_history = list(d.get("promotion_history", []))
