"""
sentinel.eval
=============

Evaluation harness for Sentinel's mule-detection tools against benchmarked
ground-truth labels.

Modules
-------
- ``account_eval``   account-level mule classification metrics
                     (precision, recall, F1, AUC)
- ``motif_eval``     per-motif-type precision/recall/coverage
- ``report``         aggregated EvaluationReport (display + save)

Usage
-----
    from sentinel.benchmarks import generate_paysim_like
    from sentinel.eval import evaluate_motif_detector, evaluate_account_classifier

    bench = generate_paysim_like()
    motif_metrics = evaluate_motif_detector(bench)
    account_metrics = evaluate_account_classifier(bench)
"""

from sentinel.eval.account_eval import (
    AccountClassificationMetrics,
    score_accounts_from_motifs,
    evaluate_account_classifier,
)
from sentinel.eval.motif_eval import (
    MotifEvalMetrics,
    PerMotifMetrics,
    evaluate_motif_detector,
)
from sentinel.eval.report import EvaluationReport

# Golden-dataset evaluation harness (groundedness via recomputation)
from sentinel.eval.golden import (
    GoldenTask, GoldenFact, GoldenFinding, GoldenDataset, load_golden,
)
from sentinel.eval.golden_scorer import Scorecard, score_run, aggregate
from sentinel.eval.scorecard import build_run_scorecard, write_run_scorecard
from sentinel.eval.runner import evaluate_golden
from sentinel.eval.trend import collect_scorecards, build_trend, render_console

__all__ = [
    "AccountClassificationMetrics",
    "score_accounts_from_motifs",
    "evaluate_account_classifier",
    "MotifEvalMetrics",
    "PerMotifMetrics",
    "evaluate_motif_detector",
    "EvaluationReport",
    # golden harness
    "GoldenTask", "GoldenFact", "GoldenFinding", "GoldenDataset",
    "load_golden", "Scorecard", "score_run", "aggregate",
    "build_run_scorecard", "write_run_scorecard", "evaluate_golden",
]
