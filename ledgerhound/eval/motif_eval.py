"""
ledgerhound.eval.motif_eval
========================

Per-motif evaluation: how well does each detector recover the canonical
mule typologies labeled in the benchmark?

For each motif pattern we compute:
  - precision     : of detected matches involving labeled mule accounts,
                    what fraction overlap with the *correct* mule type?
  - recall        : of labeled mule instances of this type, what fraction
                    were detected by ANY motif of the matching name?
  - coverage      : of the labeled accounts of this type, what fraction
                    appear in at least one detected match of this pattern?

Because PaySim-like benchmarks label per-account (not per-motif-instance)
we approximate "instance recall" by ground-truth mule type -> detector
pattern equivalence:

    fan_in_collector  <->  fan_in_collector
    fan_out_distributor <->  fan_out_distributor
    layering_chain     <->  layering_chain
    cyclic_ring        <->  cyclic_ring
    passthrough        <->  pass_through  (note: detector spelling)
    velocity_hub       <->  velocity_hub  (no ground-truth equivalent in
                                         PaySim -- ignored unless dataset
                                         provides it)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ledgerhound.benchmarks.paysim import BenchmarkDataset

log = logging.getLogger(__name__)


# Canonical type -> detector-pattern map.
TYPE_TO_PATTERN: dict[str, str] = {
    "fan_in_collector":     "fan_in_collector",
    "fan_in_source":        "fan_in_collector",
    "fan_out_distributor":  "fan_out_distributor",
    "fan_out_dest":         "fan_out_distributor",
    "layering_chain":       "layering_chain",
    "cyclic_ring":          "cyclic_ring",
    "passthrough":          "pass_through",
    "passthrough_source":   "pass_through",
    "passthrough_dest":     "pass_through",
}


@dataclass
class PerMotifMetrics:
    pattern_name: str
    n_detected_matches: int
    n_detected_accounts: int
    n_labeled_accounts: int
    n_recovered_accounts: int     # detected  labeled
    n_false_positive_accounts: int
    precision: float
    recall: float
    f1: float
    coverage: float                 # = recall by account intersection
    n_alert: int = 0
    n_warning: int = 0
    n_info: int = 0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("precision", "recall", "f1", "coverage"):
            v = d.get(k)
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class MotifEvalMetrics:
    """Aggregated motif evaluation report."""

    benchmark_name: str
    n_total_accounts: int
    n_total_mules: int

    overall_precision: float
    overall_recall: float
    overall_f1: float

    n_alerts:   int
    n_warnings: int
    n_info:     int

    per_pattern: list[PerMotifMetrics] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark_name":    self.benchmark_name,
            "n_total_accounts":  self.n_total_accounts,
            "n_total_mules":     self.n_total_mules,
            "overall_precision": round(self.overall_precision, 4),
            "overall_recall":    round(self.overall_recall, 4),
            "overall_f1":        round(self.overall_f1, 4),
            "n_alerts":          self.n_alerts,
            "n_warnings":        self.n_warnings,
            "n_info":            self.n_info,
            "per_pattern":       [p.to_dict() for p in self.per_pattern],
            "notes":             self.notes,
        }

    def display(self) -> None:
        from rich.console import Console
        from rich.rule import Rule
        from rich.table import Table
        c = Console()
        c.print(Rule(f"[bold]Motif evaluation -- {self.benchmark_name}[/bold]"))
        c.print(
            f"  Accounts     : {self.n_total_accounts:,}\n"
            f"  Labeled mules: {self.n_total_mules:,} "
            f"({self.n_total_mules / max(1, self.n_total_accounts):.1%})\n"
            f"  Detector hits: "
            f"[red]{self.n_alerts} ALERT[/red]  "
            f"[yellow]{self.n_warnings} WARNING[/yellow]  "
            f"[dim]{self.n_info} INFO[/dim]\n\n"
            f"  [bold]Overall[/bold] -- precision={self.overall_precision:.3f}  "
            f"recall={self.overall_recall:.3f}  F1={self.overall_f1:.3f}"
        )

        t = Table("Pattern", "Detected", "Labeled",
                  "Recovered", "FP", "Precision", "Recall", "F1",
                  title="Per-pattern", show_header=True, header_style="bold dim")
        for p in self.per_pattern:
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
        c.print()


def evaluate_motif_detector(
    bench: BenchmarkDataset,
    type_to_pattern: dict[str, str] = None,
) -> MotifEvalMetrics:
    """
    Run MotifDetector on the benchmark and compute per-pattern metrics
    against ground-truth ``mule_type`` labels.

    Parameters
    ----------
    bench : BenchmarkDataset
    type_to_pattern : dict, optional
        Override the default ground-truth-type -> detector-pattern map.
    """
    from ledgerhound.graph.builder import AccountGraphBuilder
    from ledgerhound.graph.motifs import MotifDetector

    type_map = dict(type_to_pattern or TYPE_TO_PATTERN)

    builder = AccountGraphBuilder()
    graph = builder.build(bench.txn_df, period=bench.name)

    detector = MotifDetector()
    result = detector.detect(graph)

    # Index labels: {pattern_name: set(account_ids)} -- many-to-one ground-truth
    labeled_by_pattern: dict[str, set] = {}
    for _, row in bench.labels.iterrows():
        if row["is_mule"] != 1:
            continue
        gt_type = str(row["mule_type"])
        det_pattern = type_map.get(gt_type)
        if not det_pattern:
            continue
        labeled_by_pattern.setdefault(det_pattern, set()).add(str(row["account_id"]))

    # Index detector matches by pattern: {pattern_name: set(account_ids)}
    detected_by_pattern: dict[str, set] = {}
    pattern_severity_counts: dict[str, dict[str, int]] = {}
    for m in result.matches:
        accounts = set(str(n) for n in m.nodes)
        detected_by_pattern.setdefault(m.pattern_name, set()).update(accounts)
        pattern_severity_counts.setdefault(
            m.pattern_name, {"ALERT": 0, "WARNING": 0, "INFO": 0}
        )[m.severity] += 1

    # Compute per-pattern metrics
    per_pattern: list[PerMotifMetrics] = []
    all_patterns = set(detected_by_pattern.keys()) | set(labeled_by_pattern.keys())
    for pat in sorted(all_patterns):
        det = detected_by_pattern.get(pat, set())
        lab = labeled_by_pattern.get(pat, set())
        recovered = det & lab
        false_pos = det - lab
        precision = (
            len(recovered) / len(det) if det else 0.0
        )
        recall = (
            len(recovered) / len(lab) if lab else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        sev_counts = pattern_severity_counts.get(
            pat, {"ALERT": 0, "WARNING": 0, "INFO": 0}
        )
        n_detected_matches = sum(
            1 for m in result.matches if m.pattern_name == pat
        )
        per_pattern.append(PerMotifMetrics(
            pattern_name=pat,
            n_detected_matches=n_detected_matches,
            n_detected_accounts=len(det),
            n_labeled_accounts=len(lab),
            n_recovered_accounts=len(recovered),
            n_false_positive_accounts=len(false_pos),
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=recall,
            n_alert=sev_counts["ALERT"],
            n_warning=sev_counts["WARNING"],
            n_info=sev_counts["INFO"],
        ))

    # Overall (micro-averaged on accounts)
    detected_all: set = set()
    labeled_all: set = set()
    for pat in all_patterns:
        detected_all.update(detected_by_pattern.get(pat, set()))
        labeled_all.update(labeled_by_pattern.get(pat, set()))
    recovered_all = detected_all & labeled_all
    overall_p = len(recovered_all) / len(detected_all) if detected_all else 0.0
    overall_r = len(recovered_all) / len(labeled_all) if labeled_all else 0.0
    overall_f = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0 else 0.0
    )

    return MotifEvalMetrics(
        benchmark_name=bench.name,
        n_total_accounts=int(bench.labels["account_id"].nunique()),
        n_total_mules=int(bench.labels["is_mule"].sum()),
        overall_precision=overall_p,
        overall_recall=overall_r,
        overall_f1=overall_f,
        n_alerts=result.n_alerts,
        n_warnings=result.n_warnings,
        n_info=result.n_info,
        per_pattern=per_pattern,
    )
