"""
ledgerhound.shell.guardrails
============================

One class per guardrail. A guardrail REFUSES rather than advises: each
check returns ``""`` to allow, or the refusal text the tool hands back to
the model instead of running (the text names the fix).

They take their inputs explicitly (code, claims, providers for the session
goal / methodology / frames) rather than reaching into the analyst, so the
same rules can back any front door -- the CLI shell today, the MCP server
next.

Code gates (checked by run_python before a cell executes, in this order):

    ModelingGate               model training before it is authorised
    ExportGrainGate            exporting a non-account-grain deliverable
    StatisticalAssumptionsGate a test that cannot be valid as written
    ExecBlocklist              exec-mode-only substring blocklist

Finding and workflow gates:

    RestatementGuard           re-recording a known finding without a delta
    PredicateConsistencyCheck  same quantity, different predicate (advisory)
    ChapterDedupeGate          redoing a chapter on unchanged data
    PhaseReportGate            advancing a phase that saved no report
"""

from ledgerhound.shell.guardrails.code_gates import (
    CodeGate,
    ExecBlocklist,
    ExportGrainGate,
    ModelingGate,
    StatisticalAssumptionsGate,
)
from ledgerhound.shell.guardrails.findings import (
    PredicateConsistencyCheck,
    RestatementGuard,
    claim_tokens,
    claims_match,
)
from ledgerhound.shell.guardrails.workflow import ChapterDedupeGate, PhaseReportGate

__all__ = [
    "CodeGate", "ModelingGate", "ExportGrainGate", "StatisticalAssumptionsGate",
    "ExecBlocklist", "RestatementGuard", "PredicateConsistencyCheck",
    "ChapterDedupeGate", "PhaseReportGate", "claim_tokens", "claims_match",
]
