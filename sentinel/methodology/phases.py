"""
sentinel.methodology.phases
==============================

A DECLARED, analyst-chosen sequence of analysis phases -- each with its own
data scope, allowed grains, and a checklist of EDA steps.

Why this replaces the single-grain session goal (senior feedback, 2026-08-07)
------------------------------------------------------------------------------
Sentinel previously took one `unit_of_analysis` for the whole session and
REFUSED any export at another grain. That was wrong. The fraud team runs
several legitimate methodologies, and analysts differ by experience:

  Analyst A works top-down -- fraud-only EDA at transaction AND account
  level to form hypotheses ("think like a fraudster"), the same lens on
  non-fraud, then a fraud-vs-non-fraud contrast at both grains, then an
  aggregation plan, then features at BOTH grains saved to SEPARATE files,
  then selection, then modelling.

  Analyst B starts fraud-only and lets phase 1 decide HOW to sample
  non-fraud -- sampling strategy is an OUTPUT of the analysis, not a fixed
  input.

Nothing is right or wrong until the end goal -- optimal fraud-model
precision -- is reached. So the tool must not impose a grain or an order.

What the gate becomes
-----------------------
The original 2026-08-06 failure was NOT "transaction-level analysis". It was
**undeclared drift**: the report concluded the label was account-level, then
shipped a transaction-level model without ever saying it had switched. The
defect was the silence.

So the check changes from "enforce one grain" to "are you doing what THIS
phase declared?" -- phase 5 legitimately produces both grains; phase 4
declares account only, so a transaction export there is drift worth
flagging. Declaring the change is always available and always allowed.

Steps
-------
Each phase carries 10-50 concrete EDA steps. They are a floor, not a
ceiling: `add_step` lets the analyst append steps the data actually
demands, which is the point of exploratory work. Steps are tracked so
"phase 3 complete" means something checkable rather than asserted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# Data scope for a phase.
SCOPE_FRAUD = "fraud_only"
SCOPE_NONFRAUD = "nonfraud_only"
SCOPE_BOTH = "both"

# Grains a phase may legitimately produce.
GRAIN_TXN = "transaction"
GRAIN_ACCOUNT = "account"
GRAIN_ANY = "any"

# V3-062: mode is a SEPARATE axis from which methodology (topdown/
# sample_first/bottom_up) is running -- it controls how FAR the pipeline
# goes, not the reasoning order within it. Any methodology can run in any
# mode. "" (the default) means no restriction -- the full phase sequence,
# unchanged from before this existed.
MODE_FULL = ""
MODE_EDA_ONLY = "eda_only"          # stops before feature construction
MODE_FE_ONLY = "fe_only"            # skips straight to feature construction
MODE_EDA_FE_ONLY = "eda_fe_only"    # stops before modelling authorisation
_VALID_MODES = (MODE_FULL, MODE_EDA_ONLY, MODE_FE_ONLY, MODE_EDA_FE_ONLY)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _phase_is_feature(p) -> bool:
    return any("feature" in str(o).lower() for o in (p.outputs or []))


def _phase_is_model(p) -> bool:
    return any("model" in str(o).lower() for o in (p.outputs or []))


@dataclass
class Step:
    """One concrete EDA step inside a phase."""

    id: str
    text: str
    grain: str = GRAIN_ANY          # which grain this step operates at
    status: str = "pending"         # pending | done | skipped
    note: str = ""                  # what was found, or why skipped
    source: str = "template"        # template | analyst (AI-added)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase:
    """One phase: a scope, the grains it may produce, and its steps."""

    id: str
    name: str
    purpose: str = ""
    data_scope: str = SCOPE_BOTH
    grains: list = field(default_factory=lambda: [GRAIN_ANY])
    outputs: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    status: str = "pending"         # pending | active | done

    # -- progress ----------------------------------------------------- #
    def done_steps(self) -> list:
        return [s for s in self.steps if s.status in ("done", "skipped")]

    def pending_steps(self) -> list:
        return [s for s in self.steps if s.status == "pending"]

    def progress(self) -> tuple:
        return len(self.done_steps()), len(self.steps)

    def allows_grain(self, grain: str) -> bool:
        return GRAIN_ANY in self.grains or grain in self.grains

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() if isinstance(s, Step) else s
                      for s in self.steps]
        return d


@dataclass
class Methodology:
    """An ordered phase sequence. Templates ship; analysts may edit."""

    name: str = ""
    description: str = ""
    phases: list = field(default_factory=list)
    current: int = 0                # index into phases
    mode: str = MODE_FULL           # "" | eda_only | fe_only | eda_fe_only

    # -- navigation ---------------------------------------------------- #
    def current_phase(self):
        if not self.phases:
            return None
        i = max(0, min(self.current, len(self.phases) - 1))
        return self.phases[i]

    def phase_by_id(self, pid: str):
        for p in self.phases:
            if p.id == pid:
                return p
        return None

    def advance(self, force: bool = False) -> tuple:
        """Move to the next phase. Returns (ok, message).

        Refuses to skip ahead while steps are outstanding UNLESS forced --
        and a forced skip is recorded, because "we did phase 3" should be
        checkable, not asserted. Skipping is legitimate (an analyst may
        judge a phase unnecessary for this dataset); doing it silently is
        not.
        """
        ph = self.current_phase()
        if ph is None:
            return False, "no methodology set"
        pending = ph.pending_steps()
        if pending and not force:
            names = ", ".join(s.id for s in pending[:6])
            return False, (
                f"phase '{ph.id}' still has {len(pending)} pending step(s): "
                f"{names}{' ...' if len(pending) > 6 else ''}.\n"
                "Complete them with complete_step(...), or skip individual "
                "steps with a reason, or call advance_phase(force=True) to "
                "move on deliberately -- the skip is recorded either way.")
        if pending and force:
            for s in pending:
                s.status = "skipped"
                s.note = s.note or "skipped when advancing the phase"
        if self.current >= len(self.phases) - 1:
            ph.status = "done"
            return True, f"phase '{ph.id}' complete -- that was the last phase."
        nxt = self.phases[self.current + 1]
        # V3-062: mode is a SCOPE BOUNDARY, not a real gate -- force=True
        # overrides it exactly like an outstanding-steps refusal, because a
        # human deciding to go past the mode's declared scope is legitimate,
        # doing it silently is not.
        block_reason = self._mode_blocks(nxt)
        if block_reason and not force:
            return False, (
                f"mode '{self.mode}' stops before phase '{nxt.id}' "
                f"({block_reason}). This is the mode's scope boundary -- "
                "call advance_phase(force=True) to continue past it "
                "deliberately, or stop here if that was the intent.")
        ph.status = "done"
        self.current += 1
        nxt.status = "active"
        return True, f"phase '{ph.id}' complete -> now in '{nxt.id}': {nxt.name}"

    def _mode_blocks(self, phase) -> str:
        """Reason string if `mode` says the pipeline should not enter
        `phase`, else ""."""
        if self.mode == MODE_EDA_ONLY and (_phase_is_feature(phase)
                                           or _phase_is_model(phase)):
            return "eda_only stops before feature construction"
        if self.mode == MODE_EDA_FE_ONLY and _phase_is_model(phase):
            return "eda_fe_only stops before modelling"
        return ""

    def apply_mode_start(self) -> None:
        """fe_only only: jump straight to the first feature-building
        phase, marking the pure-exploration phases before it as skipped
        BY MODE (not by the analyst) -- recorded, not silent. No-op for
        every other mode."""
        if self.mode != MODE_FE_ONLY or not self.phases:
            return
        for i, p in enumerate(self.phases):
            if _phase_is_feature(p) or _phase_is_model(p):
                for skipped in self.phases[:i]:
                    skipped.status = "done"
                    for s in skipped.steps:
                        if s.status == "pending":
                            s.status = "skipped"
                            s.note = s.note or "skipped: fe_only mode starts at feature construction"
                self.current = i
                p.status = "active"
                return
        # No feature/model phase found at all -- leave current_phase as-is
        # (whatever from_dict/load_template already set active).

    # -- gates --------------------------------------------------------- #
    def allows_grain(self, grain: str) -> bool:
        ph = self.current_phase()
        return True if ph is None else ph.allows_grain(grain)

    def allows_modeling(self) -> bool:
        """True only in a phase that declares a model output."""
        ph = self.current_phase()
        return False if ph is None else _phase_is_model(ph)

    # -- reporting ----------------------------------------------------- #
    def pin_block(self) -> str:
        """Always-current banner: where we are, and what this phase allows."""
        ph = self.current_phase()
        if ph is None:
            return ""
        done, total = ph.progress()
        mode_tag = f"  [mode: {self.mode}]" if self.mode else ""
        lines = [f"=== METHODOLOGY: {self.name} "
                 f"(phase {self.current + 1} of {len(self.phases)}){mode_tag} ==="]
        lines.append(f"phase: {ph.id} -- {ph.name}   [{done}/{total} steps done]")
        if ph.purpose:
            lines.append(f"purpose: {ph.purpose}")
        lines.append(f"data scope: {ph.data_scope}   grains allowed: "
                     f"{', '.join(ph.grains)}")
        if ph.outputs:
            lines.append(f"expected outputs: {', '.join(map(str, ph.outputs))}")
        nxt = ph.pending_steps()[:5]
        if nxt:
            lines.append("next steps:")
            lines.extend(f"  [ ] {s.id}: {s.text}" for s in nxt)
        remaining = len(ph.pending_steps())
        if remaining > 5:
            lines.append(f"  ... and {remaining - 5} more in this phase")
        lines.append(
            "Work THIS phase's steps. Add steps the data demands with "
            "add_step -- the checklist is a floor, not a ceiling. Move on "
            "with advance_phase when the phase is genuinely done.")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "methodology": self.name,
            "phase": (self.current_phase().id if self.current_phase() else None),
            "phase_index": self.current + 1,
            "phase_count": len(self.phases),
            "phases": [{"id": p.id, "status": p.status,
                        "done": p.progress()[0], "total": p.progress()[1]}
                       for p in self.phases],
        }

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "current": self.current, "mode": self.mode,
                "phases": [p.to_dict() for p in self.phases]}


# ---------------------------------------------------------------------- #
def from_dict(d: dict) -> Methodology:
    phases = []
    known = Step.__dataclass_fields__
    for p in (d.get("phases") or []):
        # J-08: an extra key in a hand-edited template (or a methodology.json
        # written by a newer version) must not turn into a TypeError that
        # load_template swallows into "no methodology named ...".
        steps = [Step(**{k: v for k, v in s.items() if k in known})
                 if isinstance(s, dict) else s
                 for s in (p.get("steps") or [])]
        phases.append(Phase(
            id=p.get("id", ""), name=p.get("name", ""),
            purpose=p.get("purpose", ""),
            data_scope=p.get("data_scope", SCOPE_BOTH),
            grains=list(p.get("grains") or [GRAIN_ANY]),
            outputs=list(p.get("outputs") or []),
            steps=steps, status=p.get("status", "pending")))
    mode = d.get("mode", MODE_FULL) or MODE_FULL
    if mode not in _VALID_MODES:
        mode = MODE_FULL
    m = Methodology(name=d.get("name", ""),
                    description=d.get("description", ""),
                    phases=phases, current=int(d.get("current", 0) or 0),
                    mode=mode)
    if m.phases and all(p.status == "pending" for p in m.phases):
        m.phases[0].status = "active"
    m.apply_mode_start()
    return m


def available_templates() -> list:
    return sorted(p.stem for p in _TEMPLATE_DIR.glob("*.yaml"))


def available_modes() -> tuple:
    return _VALID_MODES


def load_template(name: str, mode: str = MODE_FULL) -> Methodology | None:
    """Load a shipped methodology template by stem name. `mode` is a
    SEPARATE axis from which template this is (V3-062) -- eda_only stops
    before feature construction, fe_only jumps straight to it, eda_fe_only
    stops before modelling. "" (default) is the full, unrestricted
    sequence, unchanged from before mode existed."""
    import yaml
    if "/" in name or "\\" in name or ".." in name:
        return None
    path = _TEMPLATE_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if mode:
            data["mode"] = mode
        return from_dict(data)
    except Exception:  # noqa: BLE001
        return None
