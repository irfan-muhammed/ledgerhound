"""
ledgerhound.scorecard.components
=============================

Five component scorers, each returning a ``ComponentScore`` with a
0..1 score, a list of named sub-metrics, and a list of remediation
suggestions.

These are pure functions of the inputs they receive -- no I/O, no
side effects.  The aggregator (``Scorecard``) is responsible for
gathering inputs and weighting components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# -- Container --------------------------------------------------------------


@dataclass
class ComponentScore:
    """One component of the Ledgerhound scorecard."""

    name: str
    score: float                           # 0..1
    weight: float                          # weight in the composite
    sub_metrics: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "score":       round(self.score, 4),
            "weight":      round(self.weight, 4),
            "sub_metrics": self.sub_metrics,
            "suggestions": self.suggestions,
            "notes":       self.notes,
        }


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


# -- 1. Completeness --------------------------------------------------------


def completeness_score(
    coverage_report: dict,
    weight: float = 0.25,
) -> ComponentScore:
    """
    Score from 0..1 capturing how thorough the EDA was.

    Sub-metrics
    -----------
    - step_completeness_pct       fraction of mandatory steps that ran OK
    - column_coverage_pct         fraction of dataset columns examined
    - stratum_coverage_pct        fraction of declared strata visited
    - kb_consultation_pct         fraction of analysed cols looked up in KB
    - n_critical_flags_skipped    quality-flag obedience (0+ = good)
    """
    step_pct = float(coverage_report.get("step_completeness_pct") or 0.0)
    col_pct  = coverage_report.get("column_coverage_pct")
    str_pct  = coverage_report.get("stratum_coverage_pct")
    kb_pct   = coverage_report.get("kb_consultation_pct")

    # Component sub-scores; missing inputs (None) drop out and don't penalise
    parts: list[tuple[float, float]] = []  # (sub_score, weight)
    parts.append((step_pct, 0.40))
    if col_pct is not None:
        parts.append((float(col_pct), 0.25))
    if str_pct is not None:
        parts.append((float(str_pct), 0.15))
    if kb_pct is not None:
        parts.append((float(kb_pct), 0.20))

    if not parts:
        return ComponentScore(
            name="completeness", score=0.0, weight=weight,
            notes="No coverage data -- ran without instrumentation.",
        )
    total_w = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / total_w
    score = _clip(score)

    # Suggestions
    suggestions: list[str] = []
    if step_pct < 1.0:
        ran = set(coverage_report.get("steps_executed", {}).keys())
        mand = set(coverage_report.get("mandatory_steps", []))
        missing = sorted(mand - ran)
        if missing:
            suggestions.append(
                f"Mandatory steps not run: {missing}. "
                f"Use `ledgerhound playbook run` to execute the full chain."
            )
    if col_pct is not None and col_pct < 0.50:
        suggestions.append(
            f"Only {col_pct:.0%} of dataset columns were examined -- "
            f"consider running cluster_typologies / measure_signal_decay "
            f"on additional features."
        )
    if str_pct is not None and str_pct < 1.0:
        suggestions.append(
            "Some declared strata were never analysed -- sample-frame "
            "claims may be biased toward visited strata only."
        )
    if kb_pct is not None and kb_pct < 0.50:
        suggestions.append(
            "Schema KB was consulted for less than half the analysed "
            "columns -- call dict_lookup before analysing each new column."
        )

    return ComponentScore(
        name="completeness",
        score=score,
        weight=weight,
        sub_metrics={
            "step_completeness_pct":     step_pct,
            "column_coverage_pct":       col_pct,
            "stratum_coverage_pct":      str_pct,
            "kb_consultation_pct":       kb_pct,
            "n_critical_flags_skipped":  coverage_report.get("n_critical_flags_skipped", 0),
            "steps_executed":            coverage_report.get("steps_executed"),
            "n_columns_examined":        coverage_report.get("n_columns_examined"),
        },
        suggestions=suggestions,
    )


# -- 2. Discipline ----------------------------------------------------------


def discipline_score(
    findings: list,
    weight: float = 0.30,
) -> ComponentScore:
    """
    Score the epistemic discipline of the recorded findings.

    Sub-metrics
    -----------
    - n_findings
    - sample_frame_validated_pct    fraction with validation.is_valid == True
    - evidence_attached_pct         fraction with evidence_query OR evidence_code
    - sample_frame_unspecified_pct  fraction with sample_frame == 'unspecified'
                                    (lower is better)
    - high_confidence_pct           fraction with confidence >= 0.80
    - n_population_estimate         count of population_estimate claims
                                    (tracked separately -- more = riskier)
    - n_priors_tested
    - n_priors_collapsed            count where priors_result == 'collapses'
    """
    n = len(findings)
    if n == 0:
        return ComponentScore(
            name="discipline", score=0.0, weight=weight,
            notes="No findings recorded.",
            suggestions=[
                "Use record_finding to capture every defensible insight. "
                "An EDA with zero structured findings can't be audited."
            ],
        )

    n_validated     = 0
    n_unspecified   = 0
    n_high_conf     = 0
    n_pop_est       = 0
    n_priors_tested = 0
    n_priors_collapse = 0
    n_with_evidence = 0
    n_validation_warnings = 0
    n_validation_errors   = 0

    for f in findings:
        # Sample-frame discipline
        sf = getattr(f, "sample_frame", None)
        sf_val = sf.value if hasattr(sf, "value") else str(sf)
        if sf_val == "unspecified":
            n_unspecified += 1
        if sf_val == "population_estimate":
            n_pop_est += 1

        # Validation result (auto-attached by FindingsLog if validator wired)
        v = getattr(f, "validation", None) or {}
        if v.get("is_valid") and not v.get("errors"):
            n_validated += 1
        if v.get("warnings"):
            n_validation_warnings += 1
        if v.get("errors"):
            n_validation_errors += 1

        # Evidence
        ev = getattr(f, "evidence", None)
        if ev is not None:
            if getattr(ev, "query", None) or getattr(ev, "code", None):
                n_with_evidence += 1

        # Confidence
        conf = float(getattr(f, "confidence", 0.0) or 0.0)
        if conf >= 0.80:
            n_high_conf += 1

        # Priors
        pt = getattr(f, "priors_tested", None) or []
        if pt:
            n_priors_tested += 1
            pr = getattr(f, "priors_result", None)
            pr_val = pr.value if hasattr(pr, "value") else str(pr)
            if pr_val == "collapses":
                n_priors_collapse += 1

    validated_pct   = n_validated / n
    evidence_pct    = n_with_evidence / n
    unspecified_pct = n_unspecified / n
    high_conf_pct   = n_high_conf / n

    # Score: weighted combination favouring validation + evidence,
    # penalising unspecified frames and validation errors
    score = (
        0.40 * validated_pct
        + 0.30 * evidence_pct
        + 0.15 * (1.0 - unspecified_pct)
        + 0.15 * high_conf_pct
    )
    # Hard penalty for any validation errors
    if n > 0 and n_validation_errors > 0:
        err_rate = n_validation_errors / n
        score = score * max(0.0, 1.0 - 2.0 * err_rate)
    score = _clip(score)

    suggestions: list[str] = []
    if validated_pct < 0.90:
        suggestions.append(
            f"{n - n_validated} of {n} findings failed sample-frame "
            f"validation -- review and fix tags or downgrade claims."
        )
    if evidence_pct < 0.90:
        suggestions.append(
            f"{n - n_with_evidence} of {n} findings have no "
            f"evidence_query / evidence_code -- they are unauditable."
        )
    if unspecified_pct > 0.10:
        suggestions.append(
            f"{n_unspecified} findings have sample_frame=UNSPECIFIED -- "
            f"every claim should declare its frame."
        )
    if n_pop_est > 0 and validated_pct < 1.0:
        suggestions.append(
            f"{n_pop_est} POPULATION_ESTIMATE claim(s) -- verify the "
            f"sampling design has an i.i.d. reservoir."
        )

    return ComponentScore(
        name="discipline",
        score=score,
        weight=weight,
        sub_metrics={
            "n_findings":                  n,
            "sample_frame_validated_pct":  round(validated_pct, 4),
            "evidence_attached_pct":       round(evidence_pct, 4),
            "sample_frame_unspecified_pct": round(unspecified_pct, 4),
            "high_confidence_pct":         round(high_conf_pct, 4),
            "n_population_estimate":       n_pop_est,
            "n_priors_tested":             n_priors_tested,
            "n_priors_collapsed":          n_priors_collapse,
            "n_validation_warnings":       n_validation_warnings,
            "n_validation_errors":         n_validation_errors,
        },
        suggestions=suggestions,
    )


# -- 3. Corroboration -------------------------------------------------------


def corroboration_score(
    alerts_by_tool: dict[str, set[str]],
    weight: float = 0.15,
) -> ComponentScore:
    """
    Score how well alerts are corroborated by independent tools.

    Parameters
    ----------
    alerts_by_tool : dict
        ``{tool_name: set(account_ids flagged at ALERT severity)}``
        For example: {'detect_mule_motifs': {ACC_X, ACC_Y},
                      'fingerprint_accounts': {ACC_X, ACC_Z},
                      'find_mule_rings': {ACC_X}}.

    Definition
    ----------
    For every alerted account, count how many tools agree.  An alert
    flagged by >=2 independent tools is "corroborated".

    Score = corroborated_alerts / total_alerts.
    """
    # Vacuous case -- no tool flagged any account. We score 1.0 to avoid
    # punishing the overall score, but flag `vacuous=True` and add a clear
    # note so the user knows this is not real corroboration.
    if not alerts_by_tool or all(not accs for accs in alerts_by_tool.values()):
        return ComponentScore(
            name="corroboration", score=1.0, weight=weight,
            sub_metrics={
                "vacuous":                True,
                "total_alerted_accounts": 0,
                "corroborated_alerts":    0,
                "singleton_alerts":       0,
                "tools_used":             sorted(alerts_by_tool.keys()) if alerts_by_tool else [],
                "max_agreement":          0,
            },
            notes="VACUOUS: no tool emitted any account-level alert. "
                  "Run motif / fingerprint / community detection to populate.",
            suggestions=[
                "Run `ledgerhound graph motifs`, `ledgerhound graph fingerprint`, and "
                "`ledgerhound graph rings` in the same workspace to gather "
                "corroborating evidence."
            ],
        )

    # Build {account: count_of_tools_flagging_it}
    account_tool_count: dict[str, int] = {}
    for tool, accounts in alerts_by_tool.items():
        for acc in accounts:
            account_tool_count[acc] = account_tool_count.get(acc, 0) + 1

    total_alerts = len(account_tool_count)
    if total_alerts == 0:
        return ComponentScore(
            name="corroboration", score=1.0, weight=weight,
            sub_metrics={"vacuous": True, "total_alerted_accounts": 0,
                         "corroborated_alerts": 0, "singleton_alerts": 0,
                         "tools_used": sorted(alerts_by_tool.keys()),
                         "max_agreement": 0},
            notes="VACUOUS: tools fired but produced no account-level alerts.",
        )
    corroborated = sum(1 for n in account_tool_count.values() if n >= 2)
    singletons   = total_alerts - corroborated
    score = corroborated / total_alerts

    suggestions: list[str] = []
    if singletons > 0:
        suggestions.append(
            f"{singletons} of {total_alerts} alerted account(s) flagged by "
            f"only one tool -- consider running additional graph tools or "
            f"manually reviewing these singletons."
        )

    return ComponentScore(
        name="corroboration",
        score=score,
        weight=weight,
        sub_metrics={
            "vacuous":                   False,
            "total_alerted_accounts":    total_alerts,
            "corroborated_alerts":       corroborated,
            "singleton_alerts":          singletons,
            "tools_used":                sorted(alerts_by_tool.keys()),
            "max_agreement":             max(account_tool_count.values())
                                          if account_tool_count else 0,
        },
        suggestions=suggestions,
    )


# -- 4. Data quality --------------------------------------------------------


def data_quality_score(
    consistency_report: Optional[dict] = None,
    psi_report: Optional[dict] = None,
    signal_health_report: Optional[dict] = None,
    weight: float = 0.15,
) -> ComponentScore:
    """
    Score data quality based on the three quality-relevant tools.

    Sub-metrics
    -----------
    - consistency_pass_rate        # PASS checks / total checks
    - psi_overall_status           PASS | WARNING | FAIL
    - signal_grade_avg             A=1.0, B=0.8, C=0.6, D=0.4, F=0.2
    """
    parts: list[tuple[float, float]] = []
    sub: dict[str, Any] = {}

    if consistency_report:
        n_pass = int(consistency_report.get("n_pass") or 0)
        n_warn = int(consistency_report.get("n_warning") or 0)
        n_fail = int(consistency_report.get("n_fail") or 0)
        n_skip = int(consistency_report.get("n_skipped") or 0)
        total = n_pass + n_warn + n_fail
        consistency_score = (
            n_pass / total if total > 0 else 0.5
        )
        # WARN counts as 0.5 success
        if total > 0:
            consistency_score = (n_pass + 0.5 * n_warn) / total
        parts.append((consistency_score, 0.40))
        sub["consistency_pass_rate"] = round(consistency_score, 4)
        sub["consistency_n_pass"]    = n_pass
        sub["consistency_n_warning"] = n_warn
        sub["consistency_n_fail"]    = n_fail
        sub["consistency_n_skipped"] = n_skip

    if psi_report:
        status = str(psi_report.get("overall_status", "")).upper()
        psi_score = {
            "PASS": 1.0, "WARNING": 0.5, "FAIL": 0.0,
        }.get(status, 0.5)
        parts.append((psi_score, 0.30))
        sub["psi_overall_status"] = status or "UNKNOWN"
        sub["psi_weighted"]       = psi_report.get("weighted_psi")

    if signal_health_report:
        # Map A->1.0, B->0.8, C->0.6, D->0.4, F->0.2
        grade_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "F": 0.2, "?": 0.0}
        top = signal_health_report.get("top_features") or []
        grades = [
            grade_map.get(str(row.get("grade", "?")).upper(), 0.0)
            for row in top
        ]
        if grades:
            sig_score = sum(grades) / len(grades)
            parts.append((sig_score, 0.30))
            sub["signal_grade_avg"]  = round(sig_score, 4)
            sub["signal_n_features"] = len(grades)

    if not parts:
        return ComponentScore(
            name="data_quality", score=0.0, weight=weight,
            notes="No data-quality reports available.",
            suggestions=[
                "Run validate_consistency, psi_certify, and "
                "measure_signal_decay to populate this component."
            ],
        )
    total_w = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / total_w
    score = _clip(score)

    suggestions: list[str] = []
    if consistency_report and consistency_report.get("n_fail", 0) > 0:
        suggestions.append(
            f"{consistency_report['n_fail']} consistency checks FAILED -- "
            f"data quality issues will cascade through downstream analysis."
        )
    psi_status = sub.get("psi_overall_status")
    if psi_status == "FAIL":
        suggestions.append(
            "PSI status FAIL -- sample is not representative of population. "
            "Re-sample or document weights."
        )
    if sub.get("signal_grade_avg") is not None and sub["signal_grade_avg"] < 0.6:
        suggestions.append(
            "Average signal grade below C -- feature discriminative power is "
            "weak; review labels and feature engineering."
        )

    return ComponentScore(
        name="data_quality",
        score=score,
        weight=weight,
        sub_metrics=sub,
        suggestions=suggestions,
    )


# -- 5. Process -------------------------------------------------------------


def process_score(
    n_tool_calls: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
    n_tools_used: Optional[int] = None,
    n_total_tools_available: int = 13,
    n_artifacts_saved: Optional[int] = None,
    n_knowledge_entries_deposited: Optional[int] = None,
    weight: float = 0.15,
) -> ComponentScore:
    """
    Score the EDA's process diversity & efficiency.

    Sub-metrics
    -----------
    - tool_diversity_pct      n_tools_used / total_tools (use a variety)
    - n_tool_calls
    - elapsed_minutes
    - n_artifacts_saved
    - n_knowledge_entries
    """
    sub: dict[str, Any] = {}
    parts: list[tuple[float, float]] = []

    if n_tools_used is not None and n_total_tools_available > 0:
        diversity_pct = min(1.0, n_tools_used / n_total_tools_available)
        parts.append((diversity_pct, 0.50))
        sub["tool_diversity_pct"] = round(diversity_pct, 4)
        sub["n_tools_used"]       = n_tools_used
        sub["n_total_tools_available"] = n_total_tools_available

    if n_tool_calls is not None:
        # 8+ tool calls = full score; less = ramp
        call_score = min(1.0, n_tool_calls / 8.0)
        parts.append((call_score, 0.20))
        sub["n_tool_calls"] = n_tool_calls

    if n_artifacts_saved is not None:
        # 10+ artifacts = full score
        art_score = min(1.0, n_artifacts_saved / 10.0)
        parts.append((art_score, 0.20))
        sub["n_artifacts_saved"] = n_artifacts_saved

    if n_knowledge_entries_deposited is not None:
        # 5+ entries = full
        kn_score = min(1.0, n_knowledge_entries_deposited / 5.0)
        parts.append((kn_score, 0.10))
        sub["n_knowledge_entries"] = n_knowledge_entries_deposited

    if elapsed_ms is not None:
        sub["elapsed_minutes"] = round(elapsed_ms / 60_000.0, 2)

    if not parts:
        return ComponentScore(
            name="process", score=0.0, weight=weight,
            notes="No process telemetry available.",
        )
    total_w = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / total_w
    score = _clip(score)

    suggestions: list[str] = []
    if sub.get("tool_diversity_pct") is not None and sub["tool_diversity_pct"] < 0.50:
        suggestions.append(
            f"Only {sub['n_tools_used']}/{sub['n_total_tools_available']} "
            f"tools used -- broaden the analysis with the unused tools."
        )

    return ComponentScore(
        name="process",
        score=score,
        weight=weight,
        sub_metrics=sub,
        suggestions=suggestions,
    )
