"""
ledgerhound.eval.report
====================

Aggregated EvaluationReport -- one place that holds account-level + motif
metrics for a single benchmark run.

This is what becomes a row in the workshop paper's results table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ledgerhound.eval.account_eval import AccountClassificationMetrics
from ledgerhound.eval.motif_eval import MotifEvalMetrics


def _ledgerhound_version() -> str:
    """Read the canonical version at instantiation time.

    These records stamp WHICH Ledgerhound produced a saved artifact, so a value
    that silently lags a release mislabels provenance. It was hardcoded in
    three separate dataclasses and had to be hand-edited every bump; imported
    lazily here because `ledgerhound/__init__` imports `ledgerhound.core`, so a
    module-level import would risk a cycle.
    """
    from ledgerhound import __version__
    return __version__



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationReport:
    """One full benchmark evaluation, ready to save / display / cite."""

    benchmark_name: str
    generated_at: str = field(default_factory=_now)
    benchmark_metadata: dict = field(default_factory=dict)
    account_metrics: Optional[AccountClassificationMetrics] = None
    motif_metrics:   Optional[MotifEvalMetrics] = None
    ledgerhound_version: str = field(default_factory=_ledgerhound_version)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark_name":     self.benchmark_name,
            "generated_at":       self.generated_at,
            "ledgerhound_version":   self.ledgerhound_version,
            "benchmark_metadata": self.benchmark_metadata,
            "account_metrics":    (
                self.account_metrics.to_dict() if self.account_metrics else None
            ),
            "motif_metrics":      (
                self.motif_metrics.to_dict() if self.motif_metrics else None
            ),
            "notes":              self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8",
        )

    def display(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        c = Console()

        # Headline panel
        am = self.account_metrics
        mm = self.motif_metrics

        head_lines = [
            f"[bold]Ledgerhound evaluation -- {self.benchmark_name}[/bold]",
            f"[dim]generated: {self.generated_at}[/dim]",
        ]
        if am:
            head_lines.append("")
            head_lines.append(
                f"  Accounts={am.n_accounts:,}  "
                f"Mules={am.n_mules:,} "
                f"({am.base_rate:.1%})"
            )
        if mm:
            head_lines.append(
                f"  Motif overall: precision={mm.overall_precision:.3f}  "
                f"recall={mm.overall_recall:.3f}  F1={mm.overall_f1:.3f}"
            )
        if am:
            head_lines.append(
                f"  Account-level: precision={am.precision:.3f}  "
                f"recall={am.recall:.3f}  F1={am.f1:.3f}"
                + (f"  AUC={am.roc_auc:.3f}" if am.roc_auc is not None else "")
                + (f"  PR-AUC={am.pr_auc:.3f}" if am.pr_auc is not None else "")
            )
        c.print(Panel("\n".join(head_lines), border_style="cyan"))

        # Per-motif breakdown
        if mm and mm.per_pattern:
            t = Table(
                "Pattern", "Detected", "Labeled", "Recovered", "FP",
                "Precision", "Recall", "F1",
                title="Per-pattern (account-level)",
                show_header=True, header_style="bold dim",
            )
            for p in mm.per_pattern:
                t.add_row(
                    p.pattern_name,
                    str(p.n_detected_accounts),
                    str(p.n_labeled_accounts),
                    str(p.n_recovered_accounts),
                    str(p.n_false_positive_accounts),
                    f"{p.precision:.3f}",
                    f"{p.recall:.3f}",
                    f"{p.f1:.3f}",
                )
            c.print(t)

        # Account-level confusion + lift
        if am:
            t2 = Table(
                "Metric", "Value",
                title="Account-level classification",
                show_header=True, header_style="bold dim",
            )
            t2.add_row("Threshold",          f"{am.threshold:.2f}")
            t2.add_row("Flagged",            f"{am.n_flagged:,}")
            t2.add_row("True positives",     str(am.n_true_positive))
            t2.add_row("False positives",    str(am.n_false_positive))
            t2.add_row("False negatives",    str(am.n_false_negative))
            t2.add_row("True negatives",     str(am.n_true_negative))
            t2.add_row("Precision",          f"{am.precision:.4f}")
            t2.add_row("Recall",             f"{am.recall:.4f}")
            t2.add_row("F1",                 f"{am.f1:.4f}")
            t2.add_row("Accuracy",           f"{am.accuracy:.4f}")
            if am.roc_auc is not None:
                t2.add_row("ROC-AUC",        f"{am.roc_auc:.4f}")
            if am.pr_auc is not None:
                t2.add_row("PR-AUC",         f"{am.pr_auc:.4f}")
            c.print(t2)

            if am.lift_at_k:
                t3 = Table(
                    "k", "Lift",
                    title="Lift @ k (vs base rate)",
                    show_header=True, header_style="bold dim",
                )
                for k, v in sorted(am.lift_at_k.items()):
                    t3.add_row(str(k), f"{v:.2f}x")
                c.print(t3)

        c.print()
