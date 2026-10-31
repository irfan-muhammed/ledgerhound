"""
sentinel.eval.account_eval
==========================

Account-level mule classification evaluation.

The pipeline
------------
1. Build the AccountGraph from the benchmark's txn_df.
2. Run MotifDetector and CommunityFinder.
3. Aggregate per-account "suspicion scores" from motif/ring membership.
4. Compare to ground-truth labels.

We compute:
  - Precision, Recall, F1    at a fixed threshold (default 0.30)
  - AUC                      (ROC-AUC over the per-account scores)
  - PR-AUC                   (Average Precision)
  - Lift                     (mule rate among top-k flagged / base rate)

These numbers are the headline metrics for the workshop paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sentinel.benchmarks.paysim import BenchmarkDataset

log = logging.getLogger(__name__)


@dataclass
class AccountClassificationMetrics:
    """Metrics for the account-level mule classification task."""

    n_accounts: int
    n_mules:    int
    base_rate:  float

    threshold:  float
    n_flagged:  int
    n_true_positive:  int
    n_false_positive: int
    n_false_negative: int
    n_true_negative:  int

    precision:  float
    recall:     float
    f1:         float
    accuracy:   float

    roc_auc:    Optional[float] = None
    pr_auc:     Optional[float] = None
    lift_at_k:  dict = field(default_factory=dict)   # {k: lift}

    score_stats: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("base_rate", "precision", "recall", "f1", "accuracy",
                  "roc_auc", "pr_auc", "threshold"):
            v = d.get(k)
            if v is not None and isinstance(v, float):
                d[k] = round(v, 4)
        d["lift_at_k"] = {str(k): round(v, 3) for k, v in d.get("lift_at_k", {}).items()}
        return d


# -- Score aggregation ------------------------------------------------------


def score_accounts_from_motifs(
    motif_result,
    community_result=None,
    fingerprint_result=None,
) -> dict[str, float]:
    """
    Aggregate a per-account "mule score" from motif / community / fingerprint
    detector outputs.

    Parameters
    ----------
    motif_result : MotifDetectionResult
    community_result : CommunityResult, optional
    fingerprint_result : FingerprintResult, optional

    Returns
    -------
    dict[account_id -> score in 0..1]
        Score is max(motif_score)  ring_score  fingerprint_likelihood
        for any account that participates in any flagged structure.
        Accounts not flagged anywhere get score = 0.
    """
    scores: dict[str, float] = {}

    # Motif contributions
    if motif_result is not None:
        for m in motif_result.matches:
            for node in m.nodes:
                scores[node] = max(scores.get(node, 0.0), float(m.score))

    # Community contributions
    if community_result is not None:
        for com in community_result.communities:
            for node in com.nodes:
                scores[node] = max(scores.get(node, 0.0), float(com.mule_score))

    # Fingerprint contributions
    if fingerprint_result is not None:
        for fp in fingerprint_result.fingerprints:
            scores[fp.account_id] = max(
                scores.get(fp.account_id, 0.0),
                float(fp.mule_likelihood),
            )

    return scores


# -- Evaluation -----------------------------------------------------------------


def evaluate_account_classifier(
    bench: BenchmarkDataset,
    threshold: float = 0.30,
    use_motifs: bool = True,
    use_community: bool = True,
    use_fingerprint: bool = True,
    k_values: tuple[int, ...] = (10, 25, 50, 100),
) -> AccountClassificationMetrics:
    """
    Run the full Sentinel mule pipeline on a benchmark and compute
    account-level classification metrics.

    Parameters
    ----------
    bench : BenchmarkDataset
        Loaded benchmark with txn_df + labels.
    threshold : float
        Score threshold for binary precision/recall.
    use_motifs, use_community, use_fingerprint : bool
        Which detectors to include in the score aggregation.
    k_values : iterable[int]
        K-values for lift@k computation.
    """
    from sentinel.graph.builder import AccountGraphBuilder
    from sentinel.graph.motifs import MotifDetector
    from sentinel.graph.community import CommunityFinder
    from sentinel.graph.fingerprint import BehavioralFingerprinter

    # Build graph from txn_df
    builder = AccountGraphBuilder()
    graph = builder.build(bench.txn_df, period=bench.name)

    motif_result = (
        MotifDetector().detect(graph) if use_motifs else None
    )
    community_result = (
        CommunityFinder().find(graph) if use_community else None
    )
    fingerprint_result = (
        BehavioralFingerprinter(device_col=None).fingerprint_all(
            bench.txn_df, period=bench.name,
        )
        if use_fingerprint else None
    )

    # Aggregate per-account scores
    scores_dict = score_accounts_from_motifs(
        motif_result, community_result, fingerprint_result,
    )

    # Align with the label table
    labels_df = bench.labels.copy()
    labels_df["score"] = labels_df["account_id"].map(
        lambda a: float(scores_dict.get(str(a), 0.0))
    ).fillna(0.0)

    return _compute_metrics(
        labels_df, threshold=threshold, k_values=k_values,
    )


def _compute_metrics(
    labels_df: pd.DataFrame,
    threshold: float,
    k_values: tuple[int, ...],
) -> AccountClassificationMetrics:
    """Compute all metrics from a (account_id, is_mule, score) frame."""
    y_true = labels_df["is_mule"].astype(int).values
    y_score = labels_df["score"].astype(float).values

    n = len(labels_df)
    n_mules = int(y_true.sum())
    base_rate = n_mules / n if n else 0.0

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (tp + tn) / n if n else 0.0

    # AUC and PR-AUC via sklearn (already a dependency for typology)
    roc_auc = None
    pr_auc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if n_mules > 0 and n_mules < n:
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc = float(average_precision_score(y_true, y_score))
    except Exception as exc:
        log.warning("AUC computation failed: %s", exc)

    # Lift at k
    sorted_idx = np.argsort(-y_score)
    lift_at_k: dict[int, float] = {}
    for k in k_values:
        if k > n or n_mules == 0:
            continue
        topk_mules = y_true[sorted_idx[:k]].sum()
        lift = (topk_mules / k) / base_rate if base_rate > 0 else 0.0
        lift_at_k[int(k)] = float(lift)

    return AccountClassificationMetrics(
        n_accounts=n,
        n_mules=n_mules,
        base_rate=base_rate,
        threshold=threshold,
        n_flagged=int(y_pred.sum()),
        n_true_positive=tp,
        n_false_positive=fp,
        n_false_negative=fn,
        n_true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        lift_at_k=lift_at_k,
        score_stats={
            "min": float(y_score.min()),
            "max": float(y_score.max()),
            "mean": float(y_score.mean()),
            "median": float(np.median(y_score)),
            "n_nonzero": int((y_score > 0).sum()),
        },
    )
