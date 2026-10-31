"""
sentinel.profiler.diff
======================

Compare two Digest objects and produce a structured DiffReport showing
what changed when the population was filtered.

Usage::

    sentinel diff digest_before.yaml digest_after.yaml

This is the backbone of what-if analysis: profile the full dataset,
filter to a sub-population, reprofile, then diff to see what shifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentinel.profiler.digest import Digest


@dataclass
class ColumnDiff:
    """Changes in a single column between two digests."""
    name: str
    status: str = "unchanged"  # "added" | "removed" | "changed" | "unchanged"
    changes: list[dict] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return self.status != "unchanged"


@dataclass
class DiffReport:
    """Structured comparison between two digest snapshots."""
    before_source: str = ""
    after_source: str = ""
    overview_changes: list[dict] = field(default_factory=list)
    quality_changes: list[dict] = field(default_factory=list)
    priority_changes: list[dict] = field(default_factory=list)
    column_diffs: dict[str, ColumnDiff] = field(default_factory=dict)
    summary_stats: dict = field(default_factory=dict)

    @property
    def changed_columns(self) -> list[str]:
        return [name for name, d in self.column_diffs.items() if d.has_changes]

    def to_markdown(self) -> str:
        """Generate a markdown report of the diff."""
        lines = ["# Sentinel Digest Diff\n"]
        lines.append(f"**Before:** {self.before_source}")
        lines.append(f"**After:** {self.after_source}\n")

        # Overview changes
        if self.overview_changes:
            lines.append("## Overview Changes\n")
            lines.append("| Metric | Before | After | Change |")
            lines.append("|--------|--------|-------|--------|")
            for c in self.overview_changes:
                delta = c.get("delta", "")
                delta_str = f"{delta:+,}" if isinstance(delta, (int, float)) and delta != 0 else str(delta)
                lines.append(f"| {c['metric']} | {c['before']} | {c['after']} | {delta_str} |")
            lines.append("")

        # Quality changes
        if self.quality_changes:
            lines.append("## Quality Changes\n")
            for c in self.quality_changes:
                lines.append(f"- **{c['metric']}**: {c['before']} -> {c['after']}")
            lines.append("")

        # Priority shifts
        if self.priority_changes:
            lines.append("## Priority Shifts\n")
            for c in self.priority_changes:
                lines.append(f"- **{c['column']}**: {c['before']} -> {c['after']}")
            lines.append("")

        # Column-level diffs
        changed = [d for d in self.column_diffs.values() if d.has_changes]
        if changed:
            lines.append("## Column Changes\n")
            for cd in sorted(changed, key=lambda x: x.name):
                lines.append(f"### {cd.name} ({cd.status})\n")
                if cd.changes:
                    lines.append("| Metric | Before | After | Delta |")
                    lines.append("|--------|--------|-------|-------|")
                    for ch in cd.changes:
                        delta = ch.get("delta", "")
                        if isinstance(delta, float):
                            delta = f"{delta:+.4f}" if delta != 0 else "0"
                        elif isinstance(delta, int):
                            delta = f"{delta:+,}" if delta != 0 else "0"
                        lines.append(f"| {ch['metric']} | {ch.get('before', '-')} | {ch.get('after', '-')} | {delta} |")
                    lines.append("")

        # Summary
        lines.append("## Summary\n")
        s = self.summary_stats
        lines.append(f"- Columns compared: {s.get('total_columns', 0)}")
        lines.append(f"- Changed: {s.get('changed', 0)}")
        lines.append(f"- Added: {s.get('added', 0)}")
        lines.append(f"- Removed: {s.get('removed', 0)}")
        lines.append(f"- Unchanged: {s.get('unchanged', 0)}")

        return "\n".join(lines)

    def save(self, path: str):
        from pathlib import Path as P
        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")


# -- differ -------------------------------------------------

class DigestDiffer:
    """Compare two Digests and produce a DiffReport."""

    # Thresholds for flagging changes
    MEAN_CHANGE_THRESHOLD = 0.05       # 5% relative change in mean
    KS_CHANGE_THRESHOLD = 0.02         # absolute change in KS statistic
    QUALITY_SCORE_THRESHOLD = 5        # absolute change in quality score

    def diff(self, before: Digest, after: Digest) -> DiffReport:
        report = DiffReport(
            before_source=before.overview.get("source", "before"),
            after_source=after.overview.get("source", "after"),
        )

        self._diff_overview(report, before, after)
        self._diff_quality(report, before, after)
        self._diff_priorities(report, before, after)
        self._diff_columns(report, before, after)
        self._compute_summary(report)

        return report

    def _diff_overview(self, report: DiffReport, before: Digest, after: Digest):
        """Compare high-level dataset metrics."""
        for metric in ("rows", "columns", "target"):
            bval = before.overview.get(metric)
            aval = after.overview.get(metric)
            if bval != aval:
                delta = ""
                if isinstance(bval, (int, float)) and isinstance(aval, (int, float)):
                    delta = aval - bval
                report.overview_changes.append({
                    "metric": metric, "before": bval, "after": aval, "delta": delta
                })

    def _diff_quality(self, report: DiffReport, before: Digest, after: Digest):
        """Compare quality scores and issue counts."""
        bq, aq = before.data_quality, after.data_quality
        bscore = bq.get("score", 0)
        ascore = aq.get("score", 0)

        if abs(ascore - bscore) >= self.QUALITY_SCORE_THRESHOLD:
            report.quality_changes.append({
                "metric": "quality_score",
                "before": f"{bscore}/100",
                "after": f"{ascore}/100",
                "delta": ascore - bscore,
            })

        bcrit = len(bq.get("critical", []))
        acrit = len(aq.get("critical", []))
        if bcrit != acrit:
            report.quality_changes.append({
                "metric": "critical_issues",
                "before": bcrit, "after": acrit,
            })

        bempty = len(bq.get("empty_columns", []))
        aempty = len(aq.get("empty_columns", []))
        if bempty != aempty:
            report.quality_changes.append({
                "metric": "empty_columns",
                "before": bempty, "after": aempty,
            })

    def _diff_priorities(self, report: DiffReport, before: Digest, after: Digest):
        """Detect columns that shifted priority tiers."""
        b_map = self._priority_map(before.column_priority)
        a_map = self._priority_map(after.column_priority)

        all_cols = set(b_map) | set(a_map)
        for col in sorted(all_cols):
            btier = b_map.get(col, "absent")
            atier = a_map.get(col, "absent")
            if btier != atier:
                report.priority_changes.append({
                    "column": col, "before": btier, "after": atier,
                })

    @staticmethod
    def _priority_map(cp: dict) -> dict[str, str]:
        """Invert priority dict to column -> tier."""
        result = {}
        for tier in ("high", "medium", "low", "drop_candidates", "identifiers", "hashed_pii"):
            for col in cp.get(tier, []):
                result[col] = tier
        return result

    def _diff_columns(self, report: DiffReport, before: Digest, after: Digest):
        """Compare per-column profiles."""
        b_cols = set(before.column_profiles.keys())
        a_cols = set(after.column_profiles.keys())

        # Added columns
        for col in sorted(a_cols - b_cols):
            report.column_diffs[col] = ColumnDiff(name=col, status="added")

        # Removed columns
        for col in sorted(b_cols - a_cols):
            report.column_diffs[col] = ColumnDiff(name=col, status="removed")

        # Shared columns -- compare stats
        for col in sorted(b_cols & a_cols):
            bp = before.column_profiles[col]
            ap = after.column_profiles[col]
            cd = self._compare_column(col, bp, ap)
            report.column_diffs[col] = cd

    def _compare_column(self, name: str, bp: dict, ap: dict) -> ColumnDiff:
        """Deep compare two column profiles."""
        changes = []

        bs = bp.get("stats", {})
        as_ = ap.get("stats", {})

        # Compare numeric stats
        for metric in ("mean", "median", "std", "min", "max", "skew",
                        "zeros", "null_count", "null_pct"):
            bv = bs.get(metric)
            av = as_.get(metric)
            if bv is not None and av is not None:
                if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                    delta = av - bv
                    # Flag if relative change exceeds threshold
                    rel_change = abs(delta) / max(abs(bv), 1e-9)
                    if rel_change > self.MEAN_CHANGE_THRESHOLD or (metric == "null_count" and delta != 0):
                        changes.append({
                            "metric": metric,
                            "before": round(bv, 4) if isinstance(bv, float) else bv,
                            "after": round(av, 4) if isinstance(av, float) else av,
                            "delta": round(delta, 4) if isinstance(delta, float) else delta,
                        })

        # Compare categorical top values
        b_top = bs.get("top_values", {})
        a_top = as_.get("top_values", {})
        if b_top and a_top:
            b_mode = next(iter(b_top), None)
            a_mode = next(iter(a_top), None)
            if b_mode != a_mode:
                changes.append({
                    "metric": "mode",
                    "before": b_mode,
                    "after": a_mode,
                    "delta": "mode shifted",
                })

        # Compare unique count
        b_uniq = bs.get("unique")
        a_uniq = as_.get("unique")
        if b_uniq is not None and a_uniq is not None and b_uniq != a_uniq:
            changes.append({
                "metric": "unique_count",
                "before": b_uniq, "after": a_uniq,
                "delta": a_uniq - b_uniq,
            })

        # Compare KS test
        bks = bp.get("ks_test", {})
        aks = ap.get("ks_test", {})
        if bks and aks:
            b_stat = bks.get("statistic", 0)
            a_stat = aks.get("statistic", 0)
            if abs(a_stat - b_stat) > self.KS_CHANGE_THRESHOLD:
                changes.append({
                    "metric": "ks_statistic",
                    "before": round(b_stat, 4),
                    "after": round(a_stat, 4),
                    "delta": round(a_stat - b_stat, 4),
                })

        status = "changed" if changes else "unchanged"
        return ColumnDiff(name=name, status=status, changes=changes)

    def _compute_summary(self, report: DiffReport):
        """Fill in summary stats."""
        total = len(report.column_diffs)
        report.summary_stats = {
            "total_columns": total,
            "changed": sum(1 for d in report.column_diffs.values() if d.status == "changed"),
            "added": sum(1 for d in report.column_diffs.values() if d.status == "added"),
            "removed": sum(1 for d in report.column_diffs.values() if d.status == "removed"),
            "unchanged": sum(1 for d in report.column_diffs.values() if d.status == "unchanged"),
        }
