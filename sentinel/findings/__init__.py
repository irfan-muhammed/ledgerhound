"""
sentinel.findings
=================

Structured findings -- the unit of analytical truth in Sentinel.

A Finding is **not** a paragraph in a report. It's a record with:

* **claim**         -- what the finding says
* **evidence**      -- reproducible code / SQL / chart that backs it
* **sample_frame**  -- under what sampling regime the claim holds
* **confidence**    -- how strongly we believe it
* **priors_tested** -- which org-knowledge priors this finding stress-tested
* **priors_result** -- survives | collapses | shifts | untested
* **derived_from**  -- IDs of findings this one builds on
* **enables_sampling_for** -- Phase 2 hook: which datasets this finding
                             could re-shape the sampling strategy for

This is the seed of the hypothesis graph (Phase 2 will add cross-finding
falsification and counter-evidence). Phase 1 just emits records reliably.

Why structure? Two reasons:

1. **Auditability** -- every claim has a re-runnable trail.
2. **Compounding** -- next session reads findings as priors, not as prose.
   Prose is one-shot. Structured records are forever.
"""

from .models import (
    Evidence,
    Finding,
    PriorResult,
    SampleFrame,
)
from .log import FindingsLog
from .render import render_finding_md, render_findings_log_md
from .tool import FINDINGS_TOOL_SPECS, make_findings_tools
from .promotion import promote_findings, PromotionResult

__all__ = [
    "Evidence",
    "Finding",
    "PriorResult",
    "SampleFrame",
    "FindingsLog",
    "render_finding_md",
    "render_findings_log_md",
    "make_findings_tools",
    "FINDINGS_TOOL_SPECS",
    "promote_findings",
    "PromotionResult",
]
