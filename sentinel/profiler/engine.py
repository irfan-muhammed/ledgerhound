"""
sentinel.profiler.engine
========================

Orchestrates the profiler pipeline with dramatic step-by-step display.
Each detection phase shows what it found in real-time.
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

from sentinel.profiler.detectors.column_classifier import ColumnClassifier
from sentinel.profiler.detectors.quality import QualityDetector
from sentinel.profiler.detectors.statistics import StatisticalProfiler
from sentinel.profiler.detectors.priority_scorer import PriorityScorer
from sentinel.profiler.digest import Digest, DigestFormatter
from sentinel.profiler.parallel import profile_statistics_parallel
from sentinel.schema.models import DatasetSchema

console = Console()


class ProfilerEngine:
    """Run the full profiler pipeline with live progress display."""

    def __init__(self, backend, schema: DatasetSchema):
        self.backend = backend
        self.schema = schema

    def run(self, target_col: str | None = None, source: str = "",
            parallel: bool = False, n_workers: int = 3) -> Digest:
        primary = self.schema.primary
        if not primary:
            raise ValueError("No primary table in schema")

        columns = primary.columns
        col_infos = {ci.name: ci for ci in self.backend.get_columns()}
        classifier = ColumnClassifier()
        t0 = time.perf_counter()

        # Data overview
        n_rows = self.backend.count_rows()
        n_cols = len(col_infos)
        console.print(f"\n  [bold] Data loaded:[/bold] [cyan]{n_rows:,}[/cyan] rows x [cyan]{n_cols}[/cyan] columns")
        if target_col:
            console.print(f"  [bold] Target:[/bold] [cyan]{target_col}[/cyan]")
        console.print()

        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TimeElapsedColumn(),
            console=console,
        ) as prog:

            # 1. Classify
            t = prog.add_task("[cyan] Classifying columns...", total=n_cols)
            classified = {}
            for i, cname in enumerate(col_infos):
                ci = col_infos[cname]
                samples = self.backend.sample_values(cname, 5)
                classified[cname] = classifier.classify(ci, samples, columns.get(cname))
                prog.update(t, advance=1)
            prog.update(t, description=f"[green]OK Classified {len(classified)} columns")

            # Show classification highlights
            hashes = [c for c, s in classified.items() if s.semantic_type.value == "hash"]
            temporal = [c for c, s in classified.items() if s.semantic_type.value == "temporal"]
            targets = [c for c, s in classified.items() if s.role.value == "target"]
            if hashes:
                prog.console.print(f"    [dim]Found {len(hashes)} hashed PII columns[/dim]")
            if temporal:
                prog.console.print(f"    [dim]Found {len(temporal)} temporal columns[/dim]")
            if targets:
                prog.console.print(f"    [dim]Target detected: {targets[0]}[/dim]")

            # 2. Quality
            t = prog.add_task("[cyan] Detecting quality issues...", total=None)
            quality = QualityDetector().detect(
                self.backend, classified, schema=self.schema)
            n_issues = len(quality.issues)
            n_broken = len(quality.broken_joins)
            n_empty = len(quality.empty_columns)
            msg = f"[green]OK Quality: {quality.score}/100"
            if n_issues:
                msg += f" ({n_issues} issues)"
            prog.update(t, description=msg)

            if n_empty:
                prog.console.print(f"    [yellow]! {n_empty} dead columns: {', '.join(quality.empty_columns[:5])}[/yellow]")
            if n_broken:
                prog.console.print(f"    [red]! {n_broken} broken joins detected[/red]")

            # 3. Statistics -- V3-027: optionally fanned out across a
            # process pool by column group (see sentinel.profiler.parallel
            # for why only this step is parallelized and why it's
            # pandas-only). Same StatisticalProfiler().profile() call
            # either way -- parallel=False reproduces the exact prior
            # behavior with zero code-path change.
            t = prog.add_task("[cyan] Computing statistics...", total=n_cols)
            if parallel:
                stats = profile_statistics_parallel(self.backend, classified, target_col,
                                                    n_workers=n_workers)
            else:
                stats = StatisticalProfiler().profile(self.backend, classified, target_col)
            prog.update(t, completed=n_cols, description=f"[green]OK Profiled {len(stats.columns)} columns")

            # Show KS highlights
            if target_col:
                ks_cols = [(c, s.ks_test["statistic"]) for c, s in stats.columns.items()
                           if s.ks_test and "statistic" in s.ks_test]
                ks_cols.sort(key=lambda x: x[1], reverse=True)
                if ks_cols:
                    top3 = ks_cols[:3]
                    prog.console.print(f"    [dim]Top signals: {', '.join(f'{c}(KS={v:.3f})' for c,v in top3)}[/dim]")

            # 4. Priority
            t = prog.add_task("[cyan] Ranking column importance...", total=None)
            priority = PriorityScorer().score(
                classified, stats.columns,
                empty_columns=quality.empty_columns)
            prog.update(t, description=(
                f"[green]OK Priority: [bold]{len(priority.high)}[/bold] high, "
                f"{len(priority.medium)} medium, {len(priority.low)} low"))

            # 5. Assemble
            t = prog.add_task("[cyan] Assembling digest...", total=None)
            for cname, cc in classified.items():
                if cname in primary.columns:
                    primary.columns[cname] = cc

            digest = DigestFormatter().format(
                schema=self.schema, quality=quality, stats=stats,
                priority=priority, source=source, target_col=target_col)
            elapsed = time.perf_counter() - t0
            prog.update(t, description=f"[green]OK Digest ready ({elapsed:.1f}s)")

        return digest
