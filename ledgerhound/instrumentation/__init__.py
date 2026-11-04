"""
ledgerhound.instrumentation
========================

Telemetry primitives that observe what the EDA actually did -- *not* what
it found.  Coverage, step execution, schema-KB consultation rates, and
similar process-quality signals are recorded here so the Scorecard can
roll them up.

The collectors are deliberately lightweight: they are dict-backed,
process-local singletons that any tool can write to without taking a
dependency on the agent loop.

Modules
-------
- ``coverage`` -- CoverageTracker + module-level helper API
                 (``record_columns``, ``record_step``, ``record_kb_lookup``,
                 ``record_quality_flag_handled``).
"""

from ledgerhound.instrumentation.coverage import (
    CoverageTracker,
    coverage,
    record_columns,
    record_step,
    record_kb_lookup,
    record_quality_flag_handled,
    reset_coverage,
)

__all__ = [
    "CoverageTracker",
    "coverage",
    "record_columns",
    "record_step",
    "record_kb_lookup",
    "record_quality_flag_handled",
    "reset_coverage",
]
