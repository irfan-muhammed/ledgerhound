"""
ledgerhound.eval
=============

Evaluation harness for Ledgerhound's mule-detection tools against benchmarked
ground-truth labels.

Modules
-------
- ``account_eval``   account-level mule classification metrics
                     (precision, recall, F1, AUC)
- ``motif_eval``     per-motif-type precision/recall/coverage
- ``report``         aggregated EvaluationReport (display + save)

Usage
-----
    from ledgerhound.benchmarks import generate_paysim_like
    from ledgerhound.eval import evaluate_motif_detector, evaluate_account_classifier

    bench = generate_paysim_like()
    motif_metrics = evaluate_motif_detector(bench)
    account_metrics = evaluate_account_classifier(bench)
"""

from ledgerhound.eval.account_eval import (
    AccountClassificationMetrics,
    score_accounts_from_motifs,
    evaluate_account_classifier,
)
from ledgerhound.eval.motif_eval import (
    MotifEvalMetrics,
    PerMotifMetrics,
    evaluate_motif_detector,
)
from ledgerhound.eval.report import EvaluationReport

# Golden-dataset evaluation harness (groundedness via recomputation)
from ledgerhound.eval.golden import (
    GoldenTask, GoldenFact, GoldenFinding, GoldenDataset, load_golden,
)
from ledgerhound.eval.golden_scorer import Scorecard, score_run, aggregate
from ledgerhound.eval.scorecard import build_run_scorecard, write_run_scorecard
from ledgerhound.eval.runner import evaluate_golden
from ledgerhound.eval.trend import collect_scorecards, build_trend, render_console

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
