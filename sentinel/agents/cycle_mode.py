"""
sentinel.agents.cycle_mode
==========================

Human-in-the-loop modes and inter-step checkpoints for the fraud x non-fraud
EDA cycle (``FraudNonFraudCycle``).

Motivation
----------
The cycle has six major steps (fraud EDA -> non-fraud sampling design ->
SQL plan -> fetch -> non-fraud EDA -> joint EDA -> comparison). An analyst
(or a compliance policy) wants to make a decision AFTER each major step --
"do I sample non-fraud now, or do another EDA pass on the fraud data?" --
and, at the fetch boundary, "does the system run the query on Trino, or do
I run it myself?".

Two kinds of decision, kept deliberately separate
-------------------------------------------------
* **Routing decisions** -- which step comes next. Reversible and cheap. The
  *mode* governs how often we pause to ask.
* **Action gates** -- running a query against production Trino. Irreversible
  and costly. These stay hard-gated REGARDLESS of mode (the AUD-098
  invariant): even ``auto`` mode never runs a prod query without the
  existing ``TrinoClient`` approval prompt.

Three modes (chosen at the start of a run, like a CLI permission mode)
---------------------------------------------------------------------
* ``plan`` -- run the read-only steps (fraud EDA, design, SQL generation),
  present the strategy + SQL, then STOP. Executes nothing, fetches nothing.
* ``hitl`` -- (default) pause at every checkpoint and ask what to do next.
* ``auto`` -- follow the canonical 1..6 sequence without asking routing
  questions; still hard-gate prod Trino execution.

This module is pure logic + JSON I/O (no LLM, no rich): the checkpoint
*rendering and input* is the CLI's job (inject a ``reader`` callable), so
everything here is deterministic and unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class CycleMode(str, Enum):
    """How much the cycle pauses for a human between steps."""

    PLAN = "plan"    # read-only: run cheap steps, present, execute nothing
    HITL = "hitl"    # pause at every checkpoint (default)
    AUTO = "auto"    # follow the canonical sequence; still gate prod Trino

    @classmethod
    def parse(cls, value: str) -> "CycleMode":
        try:
            return cls(value.strip().lower())
        except ValueError as exc:  # noqa: TRY003
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unknown cycle mode {value!r}. Valid: {valid}."
            ) from exc


# The canonical spine, in order. ``fetch`` is the human/Trino boundary.
CYCLE_STEPS: tuple[str, ...] = (
    "fraud_eda",        # step 1
    "nonfraud_design",  # step 2
    "sql_plan",         # step 3 (generate the SQL)
    "fetch",            # step 3.5 (export vs execute vs skip)
    "nonfraud_eda",     # step 4
    "joint_eda",        # step 5
    "compare",          # step 6
)

# ``plan`` mode runs the read-only spine up to and including this step, then
# always exports the SQL and stops (never fetches, never runs step 4+).
PLAN_MODE_LAST_STEP = "sql_plan"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Persisted cycle state
# ---------------------------------------------------------------------------


@dataclass
class CycleState:
    """Resumable cursor over the six-step cycle.

    Persisted to ``.sentinel/cycle_state.json`` so every checkpoint is a real
    save-point: stop after step 1, resume days later, mode and decisions
    remembered.
    """

    mode: str = CycleMode.HITL.value
    current_step: str = CYCLE_STEPS[0]
    completed_steps: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    data_dir: str = ""
    findings_path: str = ""
    objective: str = ""
    # Run config captured on a fresh start (source_table, strategy, budget,
    # ...), so --resume continues the SAME run instead of silently falling
    # back to CLI defaults (e.g. a different --data-dir).
    params: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    # -- persistence --------------------------------------------------------

    @staticmethod
    def path_for(root: str | Path = ".sentinel") -> Path:
        return Path(root) / "cycle_state.json"

    @classmethod
    def load(cls, root: str | Path = ".sentinel") -> Optional["CycleState"]:
        p = cls.path_for(root)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, root: str | Path = ".sentinel") -> Path:
        if not self.created_at:
            self.created_at = _now()
        self.updated_at = _now()
        p = self.path_for(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def clear(cls, root: str | Path = ".sentinel") -> None:
        cls.path_for(root).unlink(missing_ok=True)

    # -- cursor -------------------------------------------------------------

    def record(self, step: str, choice: str) -> None:
        """Log the decision taken at ``step`` and mark the step complete."""
        self.decisions.append({"step": step, "choice": choice, "ts": _now()})
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def advance_to(self, step: str) -> None:
        if step not in CYCLE_STEPS:
            raise ValueError(f"Unknown step {step!r}.")
        self.current_step = step

    def is_done(self) -> bool:
        return "compare" in self.completed_steps


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    is_default: bool = False
    # ``gated`` choices trigger a downstream hard approval (e.g. Trino
    # execution). Marked so ``auto`` mode never SILENTLY selects them as a
    # default -- see ``Checkpoint.default_key``.
    gated: bool = False


@dataclass(frozen=True)
class Checkpoint:
    """A routing/fetch decision point between cycle steps."""

    step: str
    prompt: str
    choices: tuple[Choice, ...]

    def default_key(self) -> str:
        for c in self.choices:
            if c.is_default:
                return c.key
        return self.choices[0].key

    def key_set(self) -> set[str]:
        return {c.key for c in self.choices}

    def is_gated(self, key: str) -> bool:
        for c in self.choices:
            if c.key == key:
                return c.gated
        return False

    def resolve(
        self,
        mode: CycleMode,
        reader: Optional[Callable[["Checkpoint"], str]] = None,
    ) -> str:
        """Return the chosen key for this checkpoint under ``mode``.

        * ``AUTO`` / ``PLAN`` -> the default choice, without prompting. The
          default is always a NON-gated option, so ``auto`` never silently
          triggers a production Trino run (AUD-098 invariant): the ``fetch``
          checkpoint defaults to ``export``, not ``execute``.
        * ``HITL`` -> call ``reader`` to obtain the human's choice; validate
          it against the checkpoint's keys and fall back to the default if
          the reader returns something unrecognised.
        """
        if mode in (CycleMode.AUTO, CycleMode.PLAN):
            return self.default_key()
        if reader is None:
            raise ValueError("HITL mode requires a reader callable.")
        chosen = (reader(self) or "").strip().lower()
        return chosen if chosen in self.key_set() else self.default_key()


# -- the standard checkpoints -----------------------------------------------
#
# Defaults are deliberately the safe/forward option, and never a gated one.

def _routing(step: str, next_label: str, more_label: str) -> Checkpoint:
    return Checkpoint(
        step=step,
        prompt=f"Step complete ({step}). What next?",
        choices=(
            Choice("proceed", next_label, is_default=True),
            Choice("more_eda", more_label),
            Choice("stop", "Stop here (state saved; resume later)"),
        ),
    )


CHECKPOINTS: dict[str, Checkpoint] = {
    "fraud_eda": _routing(
        "fraud_eda",
        "Proceed to non-fraud sampling design (step 2)",
        "Another EDA pass on the fraud data",
    ),
    # After the SQL is generated: who fetches the non-fraud data?
    "fetch": Checkpoint(
        step="fetch",
        prompt="Sampling plan + SQL ready. How should non-fraud data be fetched?",
        choices=(
            Choice("export", "Export SQL -- you run it in Trino, drop the CSV in",
                   is_default=True),
            Choice("execute", "Run now via the HITL gate (you approve after cost)",
                   gated=True),
            Choice("skip", "Skip -- non-fraud already fetched"),
            Choice("stop", "Stop here (state saved; resume later)"),
        ),
    ),
    "nonfraud_eda": _routing(
        "nonfraud_eda",
        "Proceed to joint EDA (step 5)",
        "Another EDA pass on the non-fraud data",
    ),
    "joint_eda": _routing(
        "joint_eda",
        "Proceed to the fraud-vs-non-fraud comparison (step 6)",
        "Another joint EDA pass",
    ),
}


def next_step_after(step: str) -> Optional[str]:
    """The canonical step that follows ``step`` (None if terminal)."""
    idx = CYCLE_STEPS.index(step)
    return CYCLE_STEPS[idx + 1] if idx + 1 < len(CYCLE_STEPS) else None


def plan_mode_should_stop(step: str) -> bool:
    """In ``plan`` mode, stop once the read-only spine is done."""
    return CYCLE_STEPS.index(step) >= CYCLE_STEPS.index(PLAN_MODE_LAST_STEP)
