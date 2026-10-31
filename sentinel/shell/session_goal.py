"""
sentinel.shell.session_goal
==============================

A declared goal the session is GROUNDED to -- unit of analysis, entity key,
label column, and which phase of work is currently allowed.

Why this exists (Kubeflow run 2026-08-06, 7.8 h, 74.5M rows)
--------------------------------------------------------------
Two failures in that run were not "the model was dumb", they were "nothing
in the system tied the stated goal to what got built":

1. **Grain drift.** The analyst's OWN report concluded "the `flag` is
   effectively an account-level mule label", then shipped a
   transaction-level model (test_transactions=18,376,931) and demoted
   account-level to a *future recommendation*. The output-grain check
   fired 4x -- including "999,971 duplicate values across 1,000,000 rows"
   -- and the model wrote, verbatim:
       "# The output grain warnings are because we are exporting
        transaction-level frames. # This is intentional"
   An advisory warning is something a model under time pressure will
   rationalise past. The user then said mid-run "the analysis u did is
   transactional level" and the final artifact STILL was.

2. **Phase drift.** The stated goal was aggregated / account-level FEATURE
   engineering. The run spent hours training LightGBM models and tuning
   AUC instead -- modelling was never authorised.

So this module is deliberately not another advisory check. `check_export`
and `check_modeling` return a REFUSAL that the caller turns into a failed
tool call: the write does not happen, the model gets told why, and the
only way past it is to fix the grain or get the phase authorised by a
human. Warnings were tried; this is the version with teeth.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

GRAIN_ACCOUNT = "account"
GRAIN_TRANSACTION = "transaction"
GRAIN_UNSET = ""

PHASE_EXPLORE = "explore"     # EDA, pattern finding -- always allowed
PHASE_FEATURES = "features"   # building/exporting the feature sheet
PHASE_MODEL = "model"         # training models -- requires authorisation

_PHASES = (PHASE_EXPLORE, PHASE_FEATURES, PHASE_MODEL)

# Libraries whose use means "training a model", not "exploring data".
_MODELLING_RE = re.compile(
    r"\b(?:import\s+(?:lightgbm|xgboost|catboost)"
    r"|from\s+(?:lightgbm|xgboost|catboost|sklearn\.(?:ensemble|linear_model|tree))"
    r"|LGBMClassifier|XGBClassifier|CatBoostClassifier"
    r"|RandomForestClassifier|GradientBoostingClassifier|LogisticRegression"
    r"|\.fit\s*\(|cross_val_score|GridSearchCV|RandomizedSearchCV)\b"
)

_EXPORT_RE = re.compile(r"\.to_(?:csv|parquet|excel|feather|pickle)\s*\(")
# Which VARIABLE is being exported: `acct.to_csv(...)` -> "acct". Needed so
# the grain gate judges only the frame actually being written, not every
# frame in scope -- `df` itself is legitimately transaction-level and must
# never trip the gate just by existing.
_EXPORT_TARGET_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\.to_(?:csv|parquet|excel|feather|pickle)\s*\(")


def exported_names(code: str) -> set:
    """Variable names this cell writes to disk. Empty set means 'could not
    tell' -- callers should then fall back to checking nothing rather than
    everything, so an unparseable expression never blocks legitimate work."""
    return set(_EXPORT_TARGET_RE.findall(code or ""))


@dataclass
class SessionGoal:
    """What this session is FOR. Pinned into the prompt every turn."""

    objective: str = ""
    unit_of_analysis: str = GRAIN_UNSET     # "account" | "transaction" | ""
    entity_key: str = ""                    # e.g. "payer_account_number"
    # V3-053: the OTHER side of a payment. Optional -- most non-payment
    # datasets have no second node. When set, feature generation can be run
    # against BOTH grains (remitter and beneficiary), which testing-team
    # feedback flagged as missing: aggregation was payer-only, while the
    # domain profile's own established findings say the payee side is where
    # the account-level verdict actually lives.
    entity_key_secondary: str = ""          # e.g. "payee_account_number"
    label_column: str = ""
    phase: str = PHASE_EXPLORE
    # Set only by a real human authorisation (HITL), never by the model.
    modeling_authorised: bool = False
    notes: str = ""
    # V6.3 postmortem "Filter 1" (run_outputs/2026-09-09_08-05-10): the
    # model's own `objective` string is a LOSSY distillation of the actual
    # prompt -- it dropped detail from a 2,387-char instruction down to a
    # compact JSON field, and only the compressed version survived into the
    # rest of the run. This holds the ORIGINAL user turn verbatim, captured
    # by the harness (not written by the model, so it cannot be re-compressed
    # by a later set_goal call) and pinned every turn -- see pin_block().
    origin_instruction: str = ""

    # ---------------------------------------------------------------- #
    def is_set(self) -> bool:
        return bool(self.objective or self.unit_of_analysis)

    def pin_block(self) -> str:
        """The always-current goal banner injected alongside the schema pin."""
        if not self.is_set():
            return ("=== SESSION GOAL (not yet declared) ===\n"
                    "No goal declared. Call set_goal(objective=..., "
                    "unit_of_analysis='account'|'transaction', entity_key=..., "
                    "label_column=...) before building or exporting anything. "
                    "Exploration is allowed meanwhile.")
        lines = ["=== SESSION GOAL (pinned -- everything must serve this) ==="]
        lines.append(f"objective: {self.objective or '(unstated)'}")
        lines.append(f"unit of analysis: {self.unit_of_analysis or '(unset)'}"
                     + (f"   entity key: {self.entity_key}" if self.entity_key else ""))
        if self.entity_key_secondary:
            lines.append(
                f"secondary entity key (the OTHER side of the payment): "
                f"{self.entity_key_secondary} -- build_feature_catalog can "
                "target either grain; features are not payer-only by default")
        if self.label_column:
            lines.append(f"label column: {self.label_column}")
        lines.append(f"phase: {self.phase}"
                     + ("  (modelling AUTHORISED)" if self.modeling_authorised
                        else "  (modelling NOT authorised)"))
        if self.unit_of_analysis == GRAIN_ACCOUNT:
            lines.append(
                "GRAIN RULE: the deliverable is ACCOUNT-level. Any frame you "
                "export must have exactly ONE row per "
                f"{self.entity_key or 'entity key'}. Transaction-level "
                "exploration is fine and expected -- but the exported feature "
                "sheet must be aggregated. This is ENFORCED: an export that "
                "violates it is refused, not warned about.")
        if not self.modeling_authorised:
            lines.append(
                "PHASE RULE: model training is NOT authorised yet. Do "
                "exploration and feature work first; when you believe the "
                "features are ready, ask the user to authorise modelling. "
                "This is ENFORCED: fit/train calls are refused until then.")
        if self.notes:
            lines.append(f"notes: {self.notes}")
        if self.origin_instruction:
            lines.append(
                "--- ORIGINAL INSTRUCTION (verbatim, captured by the harness "
                "-- check the objective above against this; if you drifted "
                "from it, that is Filter-1 drift, fix it with set_goal) ---")
            lines.append(self.origin_instruction[:4000])
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    def check_modeling(self, code: str, methodology=None) -> str:
        """Return a refusal string if `code` trains a model before the
        methodology says it is time, else "" .

        With a methodology active, the phase decides: the modelling phase
        declares a model output, so training there needs no separate
        authorisation. Earlier phases still block -- the 2026-08-06 run
        spent hours tuning LightGBM when the goal was feature engineering
        -- but the message names the phase rather than pretending
        modelling is forbidden outright.
        """
        if self.modeling_authorised or self.phase == PHASE_MODEL:
            return ""
        if not _MODELLING_RE.search(code):
            return ""
        if methodology is not None:
            try:
                if methodology.allows_modeling():
                    return ""
                ph = methodology.current_phase()
                if ph is not None:
                    done, total = ph.progress()
                    return (
                        f"REFUSED: this cell trains a model, but the "
                        f"methodology is at phase '{ph.id}' ({ph.name}) "
                        f"-- {done}/{total} steps done -- and that phase "
                        f"does not produce a model.\n"
                        "Modelling comes after the analysis and feature "
                        "phases; that ordering is the point of the method. "
                        "Finish this phase and advance_phase, or call "
                        "request_modeling_authorisation if you genuinely "
                        "need to jump ahead and a human agrees.")
            except Exception:  # noqa: BLE001
                pass
        return (
            "REFUSED: this cell trains/fits a model, but modelling is not "
            "authorised for this session yet.\n"
            f"The declared goal is: {self.objective or '(unstated)'} "
            f"(phase='{self.phase}').\n"
            "Real feedback from the last run: it spent hours training models "
            "when the goal was deep exploration and account-level feature "
            "engineering. Finish the analysis/feature work first.\n"
            "When the features genuinely are ready, call ask_user to get "
            "explicit authorisation (options like ['authorise modelling', "
            "'keep doing feature work']) -- a human must say yes. Do not "
            "route around this by renaming the import."
        )

    def check_export(self, code: str, frames: dict,
                     methodology=None) -> str:
        """Return a refusal string if `code` exports a frame at a grain the
        CURRENT PHASE does not declare, else "".

        Senior feedback (2026-08-07): transaction-level output is a
        legitimate part of several fraud methodologies -- an analyst's
        phase 5 deliberately produces BOTH grains into separate files. So
        this must never mean "account grain only for the whole session".

        When a methodology is active its current phase decides: a phase
        that declares `transaction` allows a transaction-grain export
        outright. The check that survives is the one that caught the real
        2026-08-06 defect -- UNDECLARED drift, i.e. shipping a grain the
        current phase never said it would produce.

        `frames` maps var-name -> {"n_rows": int, "dup_keys": {col: dupes}}
        as computed by the kernel/exec check payload.
        """
        # A phase that permits transaction grain permits this export.
        if methodology is not None:
            try:
                if methodology.allows_grain(GRAIN_TRANSACTION):
                    return ""
            except Exception:  # noqa: BLE001 -- never let the gate crash a cell
                pass
        if self.unit_of_analysis != GRAIN_ACCOUNT:
            return ""
        if not _EXPORT_RE.search(code):
            return ""
        # Judge ONLY the frames this cell actually writes. `frames` may
        # legitimately contain transaction-level working frames (df itself,
        # intermediates) that are nobody's deliverable.
        targets = exported_names(code)
        offenders = []
        for name, info in (frames or {}).items():
            if targets and name not in targets:
                continue
            for col, dupes in (info.get("dup_keys") or {}).items():
                if self.entity_key and col != self.entity_key:
                    continue
                if int(dupes) > 0:
                    offenders.append((name, col, int(dupes),
                                      int(info.get("n_rows") or 0)))
        if not offenders:
            return ""
        detail = "\n".join(
            f"  - '{name}': '{col}' has {d:,} duplicate values across "
            f"{n:,} rows" for name, col, d, n in offenders)
        return (
            "REFUSED: the declared unit of analysis is ACCOUNT-level, but the "
            "frame you are exporting has more than one row per entity.\n"
            f"{detail}\n"
            "This is the exact defect the pilot rejected once already: \"we "
            "require an aggregated account-level dataset, rather than a "
            "dataset containing duplicate Customer Reference Numbers\".\n"
            "Aggregate to one row per "
            f"{self.entity_key or 'entity key'} (groupby + agg) and export "
            "that. If this frame is genuinely meant to be transaction-level "
            "scratch output, write it under a different name that is not the "
            "feature deliverable, or change the goal with set_goal -- which "
            "requires stating why."
        )

    # ---------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        try:
            Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                                  encoding="utf-8")
        except Exception:  # noqa: BLE001 -- persistence is best-effort
            pass

    @classmethod
    def load(cls, path: Path) -> "SessionGoal | None":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in d.items()
                          if k in cls.__dataclass_fields__})
        except Exception:  # noqa: BLE001
            return None


def normalise_grain(value: str) -> str:
    """Map free-text grain wording onto the two canonical values."""
    v = (value or "").strip().lower()
    if not v:
        return GRAIN_UNSET
    if any(t in v for t in ("account", "acct", "customer", "entity", "payer",
                            "payee", "party", "mule")):
        return GRAIN_ACCOUNT
    if any(t in v for t in ("transaction", "txn", "event", "payment", "row")):
        return GRAIN_TRANSACTION
    return GRAIN_UNSET
