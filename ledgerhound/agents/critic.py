"""
ledgerhound.agents.critic
=========================

V3-027 -- the "optional critic agent (fresh context, read-only) reviewing
conclusions ... before report save" half of the item.

Why this is a deterministic recompute, not a live LLM call
-------------------------------------------------------------
Confirmed before writing any code: no generic "fresh-context agent reviews
another agent's conclusions" mechanism exists anywhere in this codebase.
The one close analog, `ledgerhound.validation.validator.ModelValidator`
(V3-022), achieves "fresh context" NOT via a second LLM call but via a
hard rule -- RECOMPUTE, DON'T COPY -- because there's a numeric ground
truth (the fitted booster) to recompute against. This module generalizes
that exact principle to Findings broadly: a finding's `evidence.code` is
the reproducible claim; the critic RE-EXECUTES it against fresh data and
compares the result to `evidence.result`, the same way `ModelValidator`
re-scores a booster instead of trusting `TrainReport.metrics`. This is
real verification with no LLM involved -- same "no live-LLM evidence"
disclosure this session has made for every deterministic module since
V3-018 (a live LLM critic -- an actual second model call under a
read-only AgentSpec/tool-whitelist -- is architecturally straightforward
to add later via `ledgerhound.agents.factory.load_agent`/
`ledgerhound.analyst.subagent.spawn_subagent`, both of which already exist
and were confirmed reusable for this purpose; not built here).

Scope, stated precisely (a finding the critic CANNOT mechanically check
is reported as "unverifiable", never silently "confirmed" -- matching
this codebase's refuse-vs-report discipline)
------------------------------------------------------------------------
A finding is only mechanically checkable when it carries BOTH
`evidence.code` (a Python/pandas expression, re-executed via `eval` --
the same execution shape `evidence.code` already has everywhere else in
this codebase, e.g. `"df.groupby('age_bucket')['is_fraud'].mean()"`) AND
a scalar `evidence.result` (a single number the claim asserts). Findings
backed only by `evidence.query` (SQL -- no live DB in this context),
`evidence.chart_path`, prose with no evidence at all, or a
non-scalar/structured `evidence.result`, are "unverifiable" -- the
critic does not attempt NLP claim-extraction from `Finding.claim` text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ledgerhound.findings.models import Evidence, Finding, SampleFrame

# Same conservative substring blocklist ledgerhound.shell.conversation's
# _tool_run_python already applies before exec()'ing model-authored code
# -- mirrored here (not imported, to avoid an agents->shell dependency)
# since evidence.code is exactly that shape of untrusted-ish snippet.
_BLOCKED_PATTERNS = ("import os", "import sys", "import subprocess", "import shutil",
                     "open(", "__import__", "eval(", "exec(", "requests.", "socket.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_unsafe(code: str) -> bool:
    return any(p in code for p in _BLOCKED_PATTERNS)


def _to_scalar(value: Any) -> Optional[float]:
    """None for anything that isn't a single, unambiguous number --
    booleans excluded deliberately (True/False silently reading as 1/0
    would let a boolean claim look like it "matched" a numeric one)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


@dataclass
class CriticVerdict:
    finding_id: str
    claim: str
    verdict: str    # "confirmed" | "challenged" | "unverifiable"
    recomputed_value: Optional[float]
    claimed_value: Optional[float]
    reason: str
    generated_at: str = field(default_factory=_now)
    new_finding_id: Optional[str] = None   # id of the Finding this verdict was recorded as

    def to_dict(self) -> dict:
        return asdict(self)


def review_finding(finding: Finding, df: pd.DataFrame, tolerance: float = 1e-4) -> CriticVerdict:
    """Re-execute `finding.evidence.code` against `df` and compare to
    `finding.evidence.result`. `tolerance` is checked BOTH as an absolute
    difference and as a relative difference (whichever is looser) so
    tiny-magnitude and large-magnitude claims both get a sane check."""
    code = finding.evidence.code
    claimed = finding.evidence.result

    if not code or claimed is None:
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="unverifiable",
            recomputed_value=None, claimed_value=None,
            reason="no evidence.code + evidence.result pair to recompute against",
        )
    if _looks_unsafe(code):
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="unverifiable",
            recomputed_value=None, claimed_value=None,
            reason="evidence.code contains a disallowed pattern -- critic will not execute it",
        )
    claimed_scalar = _to_scalar(claimed)
    if claimed_scalar is None:
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="unverifiable",
            recomputed_value=None, claimed_value=None,
            reason="evidence.result is not a single numeric value -- critic v1 only checks scalar claims",
        )

    try:
        recomputed = eval(code, {"pd": pd, "np": np}, {"df": df})   # noqa: S307 -- see module docstring
    except Exception as e:  # noqa: BLE001
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="unverifiable",
            recomputed_value=None, claimed_value=claimed_scalar,
            reason=f"re-execution failed: {type(e).__name__}: {e}",
        )
    recomputed_scalar = _to_scalar(recomputed)
    if recomputed_scalar is None:
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="unverifiable",
            recomputed_value=None, claimed_value=claimed_scalar,
            reason="recomputed value is not a single numeric scalar (e.g. a Series/DataFrame result)",
        )

    diff = abs(recomputed_scalar - claimed_scalar)
    rel = diff / max(abs(claimed_scalar), 1e-9)
    if diff <= tolerance or rel <= tolerance:
        return CriticVerdict(
            finding_id=finding.id, claim=finding.claim, verdict="confirmed",
            recomputed_value=recomputed_scalar, claimed_value=claimed_scalar,
            reason=f"recomputed {recomputed_scalar:.6g} matches claimed {claimed_scalar:.6g} within tolerance",
        )
    return CriticVerdict(
        finding_id=finding.id, claim=finding.claim, verdict="challenged",
        recomputed_value=recomputed_scalar, claimed_value=claimed_scalar,
        reason=(f"recomputed {recomputed_scalar:.6g} diverges from claimed "
               f"{claimed_scalar:.6g} (abs diff={diff:.6g}, rel diff={rel:.4g})"),
    )


def review_findings(
    findings: list[Finding],
    df: pd.DataFrame,
    findings_log: Any = None,
    dataset_name: Optional[str] = None,
    tolerance: float = 1e-4,
) -> list[CriticVerdict]:
    """review_finding() over a batch. When findings_log is given, each
    verdict is ALSO written as its own Finding (tags=["critic_verdict",
    verdict], derived_from=[original.id]) -- this is "critic verdicts
    appear as findings," the roadmap's own phrase, and reuses the exact
    FindingsLog.append() sink V3-024's drift episodes and V3-026's merge
    already write to."""
    verdicts: list[CriticVerdict] = []
    for f in findings:
        v = review_finding(f, df, tolerance=tolerance)
        if findings_log is not None:
            verdict_finding = Finding(
                claim=f"Critic {v.verdict} finding {f.id}: {v.reason}",
                evidence=Evidence(summary=v.reason),
                sample_frame=SampleFrame.UNSPECIFIED,
                confidence=0.95 if v.verdict != "unverifiable" else 0.3,
                dataset=dataset_name or f.dataset,
                derived_from=[f.id],
                tags=["critic_verdict", v.verdict],
            )
            findings_log.append(verdict_finding)
            v.new_finding_id = verdict_finding.id
        verdicts.append(v)
    return verdicts


@dataclass
class CriticGateResult:
    proceed: bool
    verdicts: list[CriticVerdict]
    n_challenged: int
    reason: str

    def to_dict(self) -> dict:
        return {"proceed": self.proceed, "n_challenged": self.n_challenged,
                "reason": self.reason, "verdicts": [v.to_dict() for v in self.verdicts]}


def gate_high_stakes_save(
    findings: list[Finding],
    df: pd.DataFrame,
    findings_log: Any = None,
    dataset_name: Optional[str] = None,
    tolerance: float = 1e-4,
) -> CriticGateResult:
    """The "before report save" decision unit: review every finding,
    block (proceed=False) if ANY is challenged. Callable independently of
    any interactive shell. NOT wired into
    ledgerhound.shell.conversation.ConversationalAnalyst's live
    save_report dispatch in this closure -- a documented scope decision,
    same "API contract before agent wrapper" shape as V3-019/021/024/026
    at their own closures, made deliberately rather than editing a live,
    heavily-tested interactive-shell subsystem this session did not
    otherwise touch, for a capability the roadmap's own close-out
    doesn't test end-to-end through that shell."""
    verdicts = review_findings(findings, df, findings_log=findings_log,
                               dataset_name=dataset_name, tolerance=tolerance)
    n_challenged = sum(1 for v in verdicts if v.verdict == "challenged")
    proceed = n_challenged == 0
    reason = ("no challenged conclusions -- safe to save" if proceed else
             f"{n_challenged} conclusion(s) challenged by the critic -- review before saving")
    return CriticGateResult(proceed=proceed, verdicts=verdicts,
                            n_challenged=n_challenged, reason=reason)
