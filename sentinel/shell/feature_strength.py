"""
sentinel.shell.feature_strength
=================================

Sandbox helper (V3-033) exposing ``sentinel.analysis.feature_eval`` --
Information Value / KS / AUC, the modelling-grade feature-strength metrics
the OTHER agent engine's Feature-Engineering spec already uses -- inside
``run_python``, so a feature can be checked for real predictive value
before it is recommended, not only discovered weak after a full model run.

Why (V3-032/033 gap)
----------------------
Real pilot feedback: of a large proposed feature set, only 6 of the top 50
model features came from it; specific categories (Risk-Score aggregates,
amount-bucket booleans) were named as having "no predictive value" or "no
meaningful contribution." That is an empirical fact about one past run
this repository cannot retroactively change. What it CAN change is the
structural cause available to fix: those features were recommended without
ever being checked against the label first. ``evaluate_feature`` is exactly
that check -- pure numpy/pandas, deterministic, already built and tested
(``sentinel/analysis/feature_eval.py``, used by the Feature-Engineering
agent spec) -- and this module's only job is making it reachable from
``ConversationalAnalyst``'s sandbox, the same way Tier 3 did for
``check_leakage``.

Not a mechanical auto-fired check (unlike this package's other sibling
modules): there is no reliable way to guess which sandbox columns the
model considers "candidate features" for a specific target without asking
it, so this is exposed as an on-demand helper + a system-prompt
instruction to use it, exactly like ``check_leakage``.
"""

from __future__ import annotations


def make_evaluate_feature_helper():
    """Return a callable to inject into the sandbox as `evaluate_feature`.

    Signature: evaluate_feature(df, "feature_col", "target_col") -> dict
    Thin wrapper over sentinel.analysis.feature_eval.evaluate_feature that
    converts the returned dataclass to a plain dict (nicer to print/inspect
    from generated code) and returns an explicit "not evaluable" dict
    instead of None on failure, so the model gets an explicit reason
    rather than a bare None to puzzle over.
    """
    def _evaluate_feature(df, feature, target, n_bins: int = 10):
        from sentinel.analysis.feature_eval import evaluate_feature as _ev
        result = _ev(df, feature, target, n_bins=n_bins)
        if result is None:
            return {"feature": feature, "ok": False,
                    "reason": f"'{feature}' or '{target}' not in df.columns, "
                             "or the feature is entirely null"}
        d = result.to_dict()
        d["ok"] = True
        return d

    return _evaluate_feature


def make_evaluate_features_helper():
    """Return a callable to inject into the sandbox as `evaluate_features`.

    Signature: evaluate_features(df, ["f1", "f2", ...], "target_col") -> dict
    Batch version: ranks by IV, prunes redundant-correlated features, and
    reports which candidates were skipped (missing/unusable) -- exactly the
    ranking the pilot's "only 6 of 50 mattered" result needed to have been
    checked against before those 50 were handed over as a recommendation.
    """
    def _evaluate_features(df, features, target, n_bins: int = 10,
                           corr_threshold: float = 0.9):
        from sentinel.analysis.feature_eval import evaluate_features as _ev
        report = _ev(df, list(features), target, n_bins=n_bins,
                     corr_threshold=corr_threshold)
        return report.to_dict()

    return _evaluate_features


def make_percentile_table_helper():
    """Return a callable to inject into the sandbox as `percentile_table`.

    Signature: percentile_table(fraud_series, nonfraud_series) -> dict
    V3-051 (testing-team report §2): IV/KS/AUC collapse a distribution to
    one scalar. This exposes the side-by-side quantile view directly in the
    sandbox -- average, p1..p99 -- so a feature can be inspected for
    reversed directionality or a crossing tail without leaving run_python.
    """
    def _percentile_table(fraud, nonfraud):
        from sentinel.analysis.feature_eval import percentile_table as _pt
        return _pt(fraud, nonfraud)

    return _percentile_table


def make_separation_callout_helper():
    """Return a callable to inject into the sandbox as `separation_callout`.

    Signature: separation_callout(percentile_table(...)) -> dict
    Mechanically -- not LLM-narrated -- reports where the two populations
    diverge, overlap, or cross, plus a pruning-threshold value at the tail
    where they differ. Companion to `percentile_table` above.
    """
    def _separation_callout(table):
        from sentinel.analysis.feature_eval import separation_callout as _sc
        return _sc(table)

    return _separation_callout
