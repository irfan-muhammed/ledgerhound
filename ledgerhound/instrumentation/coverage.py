"""
ledgerhound.instrumentation.coverage
=================================

Process-local coverage telemetry for Ledgerhound.

What it captures
----------------
- **Columns examined** by tool (e.g. cluster_typologies looked at
  sim_change_count, device_change_count, ...).
- **Steps executed** in the playbook (consistency, typology,
  hard_negatives, ...) along with their status.
- **Schema-KB consultations** (``dict_lookup`` calls) before column
  analysis -- used to compute "% of analysed columns that were KB-checked".
- **Quality-flag handling** events (CRITICAL flag -> column skipped,
  HIGH -> caveat noted) so the scorecard can verify obedience.
- **Strata visited** -- when a tool operation is bound to a named stratum.

Why a module-level singleton
----------------------------
Coverage events are emitted from many call sites (Phase 2 tools, CLI
runners, the agent dispatch loop).  Threading a tracker through every
function would bloat signatures.  A process-local singleton keeps the
write side a one-liner.  For multi-session use, the runner calls
``reset_coverage()`` at the start of each run.

Read API
--------
``coverage().report()`` returns a dict ready for the Scorecard.  Includes:
- ``columns_examined``       set[str]
- ``steps_executed``         dict[step_name -> status]
- ``kb_lookups``             set[str]
- ``quality_flag_handled``   dict[severity -> list[col]]
- ``strata_visited``         set[str]
- ``column_coverage_pct``    float (requires ``set_dataset_columns()``)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Iterable, Optional

log = logging.getLogger(__name__)


@dataclass
class CoverageTracker:
    """Process-local coverage state.

    All fields are write-once-add-only.  ``reset()`` clears them at the
    start of a new run.
    """

    columns_examined: set[str] = field(default_factory=set)
    columns_examined_by_tool: dict[str, set[str]] = field(default_factory=dict)
    steps_executed: dict[str, str] = field(default_factory=dict)   # step_name -> status
    kb_lookups: set[str] = field(default_factory=set)
    quality_flag_handled: dict[str, list[str]] = field(
        default_factory=lambda: {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
    )
    strata_visited: set[str] = field(default_factory=set)
    dataset_columns: Optional[set[str]] = None
    declared_strata: Optional[set[str]] = None
    mandatory_steps: set[str] = field(
        default_factory=lambda: {
            "consistency", "typology", "typology_evolution",
            "hard_negatives", "psi_certification", "signal_health",
            "memory_deposit",
        }
    )

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- Write API ---------------------------------------------------------

    def record_columns(self, tool_name: str, columns: Iterable[str]) -> None:
        cols = {str(c) for c in columns if c}
        if not cols:
            return
        with self._lock:
            self.columns_examined.update(cols)
            self.columns_examined_by_tool.setdefault(tool_name, set()).update(cols)

    def record_step(self, step_name: str, status: str) -> None:
        with self._lock:
            # If a step ran multiple times, keep the worst status (FAIL > WARNING > PASS)
            order = {"ERROR": 0, "FAIL": 1, "WARNING": 2, "SKIPPED": 3, "PASS": 4}
            existing = self.steps_executed.get(step_name)
            if existing is None or order.get(status, 9) < order.get(existing, 9):
                self.steps_executed[step_name] = status

    def record_kb_lookup(self, column: str) -> None:
        with self._lock:
            self.kb_lookups.add(str(column))

    def record_quality_flag_handled(
        self,
        severity: str,
        column: str,
    ) -> None:
        sev = severity.upper().strip()
        if sev not in self.quality_flag_handled:
            self.quality_flag_handled[sev] = []
        with self._lock:
            self.quality_flag_handled[sev].append(str(column))

    def record_stratum(self, stratum: str) -> None:
        if not stratum:
            return
        with self._lock:
            self.strata_visited.add(str(stratum))

    # -- Configuration -----------------------------------------------------

    def set_dataset_columns(self, columns: Iterable[str]) -> None:
        """Tell the tracker the full column set so column_coverage_pct is meaningful."""
        with self._lock:
            self.dataset_columns = {str(c) for c in columns if c}

    def set_declared_strata(self, strata: Iterable[str]) -> None:
        with self._lock:
            self.declared_strata = {str(s) for s in strata if s}

    def set_mandatory_steps(self, steps: Iterable[str]) -> None:
        with self._lock:
            self.mandatory_steps = {str(s) for s in steps if s}

    def reset(self) -> None:
        with self._lock:
            self.columns_examined.clear()
            self.columns_examined_by_tool.clear()
            self.steps_executed.clear()
            self.kb_lookups.clear()
            for v in self.quality_flag_handled.values():
                v.clear()
            self.strata_visited.clear()
            self.dataset_columns = None
            self.declared_strata = None

    # -- Read API ----------------------------------------------------------

    def column_coverage_pct(self) -> Optional[float]:
        if not self.dataset_columns:
            return None
        if not len(self.dataset_columns):
            return None
        return len(self.columns_examined & self.dataset_columns) / len(self.dataset_columns)

    def stratum_coverage_pct(self) -> Optional[float]:
        if not self.declared_strata:
            return None
        if not len(self.declared_strata):
            return None
        return len(self.strata_visited & self.declared_strata) / len(self.declared_strata)

    def step_completeness_pct(self) -> float:
        if not self.mandatory_steps:
            return 1.0
        ran_ok = {
            s for s, status in self.steps_executed.items()
            if status not in ("ERROR", "SKIPPED")
        }
        return len(self.mandatory_steps & ran_ok) / len(self.mandatory_steps)

    def kb_consultation_pct(self) -> Optional[float]:
        if not self.columns_examined:
            return None
        return len(self.kb_lookups & self.columns_examined) / len(self.columns_examined)

    def quality_flag_critical_skipped(self) -> int:
        return len(self.quality_flag_handled.get("CRITICAL", []))

    def report(self) -> dict:
        """One dict with everything the Scorecard needs."""
        with self._lock:
            return {
                "columns_examined":          sorted(self.columns_examined),
                "n_columns_examined":        len(self.columns_examined),
                "columns_by_tool":           {
                    k: sorted(v) for k, v in self.columns_examined_by_tool.items()
                },
                "steps_executed":            dict(self.steps_executed),
                "n_steps":                   len(self.steps_executed),
                "kb_lookups":                sorted(self.kb_lookups),
                "n_kb_lookups":              len(self.kb_lookups),
                "quality_flag_handled":      {
                    k: list(v) for k, v in self.quality_flag_handled.items()
                },
                "strata_visited":            sorted(self.strata_visited),
                "dataset_columns":           (
                    sorted(self.dataset_columns) if self.dataset_columns else None
                ),
                "n_dataset_columns":         (
                    len(self.dataset_columns) if self.dataset_columns else None
                ),
                "declared_strata":           (
                    sorted(self.declared_strata) if self.declared_strata else None
                ),
                "mandatory_steps":           sorted(self.mandatory_steps),
                # Derived metrics
                "column_coverage_pct":       self.column_coverage_pct(),
                "stratum_coverage_pct":      self.stratum_coverage_pct(),
                "step_completeness_pct":     self.step_completeness_pct(),
                "kb_consultation_pct":       self.kb_consultation_pct(),
                "n_critical_flags_skipped":  self.quality_flag_critical_skipped(),
            }


# -- Singleton + module-level helpers ---------------------------------------

_GLOBAL_TRACKER = CoverageTracker()


def coverage() -> CoverageTracker:
    """Return the process-local CoverageTracker."""
    return _GLOBAL_TRACKER


def record_columns(tool_name: str, columns: Iterable[str]) -> None:
    _GLOBAL_TRACKER.record_columns(tool_name, columns)


def record_step(step_name: str, status: str) -> None:
    _GLOBAL_TRACKER.record_step(step_name, status)


def record_kb_lookup(column: str) -> None:
    _GLOBAL_TRACKER.record_kb_lookup(column)


def record_quality_flag_handled(severity: str, column: str) -> None:
    _GLOBAL_TRACKER.record_quality_flag_handled(severity, column)


def reset_coverage() -> None:
    _GLOBAL_TRACKER.reset()
