"""
sentinel.graph.result
=====================

Typed result objects for graph EDA.

All classes are JSON-serialisable (via to_dict / asdict + default=str)
so they can travel through the Phase 2 tool dispatch loop and land in
DatasetKnowledge as evidence_ref payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Motif results ----------------------------------------------------------


@dataclass
class MotifMatch:
    """One concrete instance of a mule motif found in the graph."""

    pattern_name: str                       # e.g. "fan_in_collector"
    severity: str                           # "INFO" | "WARNING" | "ALERT"
    score: float                            # 0..1, higher = stronger evidence
    nodes: list[str]                        # account ids participating
    edges: list[tuple[str, str]] = field(   # ordered (payer, payee) list
        default_factory=list
    )
    n_transactions: int = 0
    total_amount: float = 0.0
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # short label used in the agent observation
    @property
    def headline(self) -> str:
        return (
            f"{self.pattern_name} [{self.severity}] "
            f"{len(self.nodes)} nodes, {self.n_transactions} txns, "
            f"Rs {self.total_amount:,.0f}"
        )

    def to_dict(self) -> dict:
        return {
            "pattern_name":      self.pattern_name,
            "severity":          self.severity,
            "score":             round(self.score, 4),
            "nodes":             self.nodes,
            "edges":             [list(e) for e in self.edges],
            "n_transactions":    self.n_transactions,
            "total_amount":      round(self.total_amount, 2),
            "time_window_start": self.time_window_start,
            "time_window_end":   self.time_window_end,
            "metadata":          self.metadata,
        }


@dataclass
class MotifDetectionResult:
    """Output of a full motif detection sweep across all patterns."""

    period: str
    generated_at: str
    n_nodes: int
    n_edges: int
    n_transactions: int
    total_amount: float
    matches: list[MotifMatch] = field(default_factory=list)
    matches_by_pattern: dict[str, int] = field(default_factory=dict)
    n_alerts: int = 0
    n_warnings: int = 0
    n_info: int = 0
    elapsed_ms: float = 0.0
    notes: str = ""

    # -- Convenience -------------------------------------------------------

    def alerts(self) -> list[MotifMatch]:
        return [m for m in self.matches if m.severity == "ALERT"]

    def by_pattern(self, pattern_name: str) -> list[MotifMatch]:
        return [m for m in self.matches if m.pattern_name == pattern_name]

    def top_by_score(self, n: int = 10) -> list[MotifMatch]:
        return sorted(self.matches, key=lambda m: -m.score)[:n]

    def to_dict(self) -> dict:
        return {
            "period":             self.period,
            "generated_at":       self.generated_at,
            "n_nodes":            self.n_nodes,
            "n_edges":            self.n_edges,
            "n_transactions":     self.n_transactions,
            "total_amount":       round(self.total_amount, 2),
            "n_matches":          len(self.matches),
            "n_alerts":           self.n_alerts,
            "n_warnings":         self.n_warnings,
            "n_info":             self.n_info,
            "matches_by_pattern": self.matches_by_pattern,
            "matches":            [m.to_dict() for m in self.matches],
            "elapsed_ms":         round(self.elapsed_ms, 1),
            "notes":              self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def display(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        c = Console()
        c.print(Rule(f"[bold]Motif Detection -- {self.period}[/bold]"))
        c.print(
            f"  Graph    : {self.n_nodes:,} nodes, {self.n_edges:,} edges, "
            f"{self.n_transactions:,} txns, Rs {self.total_amount:,.0f}\n"
            f"  Matches  : "
            f"[red]{self.n_alerts} ALERT[/red]  "
            f"[yellow]{self.n_warnings} WARNING[/yellow]  "
            f"[dim]{self.n_info} INFO[/dim]"
        )
        if not self.matches:
            c.print("  [dim]No motifs above threshold.[/dim]\n")
            return
        t = Table("Pattern", "Sev", "Score", "Nodes", "Txns",
                  "Rs  Amount", "Highlight",
                  show_header=True, header_style="bold dim")
        sev_colors = {"ALERT": "red", "WARNING": "yellow", "INFO": "dim"}
        for m in sorted(self.matches, key=lambda x: (-{"ALERT":0,"WARNING":1,"INFO":2}.get(x.severity, 9), -x.score))[:25]:
            sc = sev_colors.get(m.severity, "white")
            highlight = m.metadata.get("highlight", "") or ", ".join(m.nodes[:3])
            t.add_row(
                m.pattern_name,
                f"[{sc}]{m.severity}[/{sc}]",
                f"{m.score:.3f}",
                str(len(m.nodes)),
                f"{m.n_transactions:,}",
                f"{m.total_amount:,.0f}",
                highlight[:50] + ("..." if len(highlight) > 50 else ""),
            )
        c.print(t)
        c.print()


# -- Centrality results -----------------------------------------------------


@dataclass
class AccountCentrality:
    """Per-account centrality scores."""

    account_id: str
    pagerank:        Optional[float] = None
    betweenness:     Optional[float] = None
    eigenvector:     Optional[float] = None
    in_degree:       int = 0
    out_degree:      int = 0
    in_amount:       float = 0.0
    out_amount:      float = 0.0
    flow_balance:    Optional[float] = None   # |in-out| / max(in,out)
    k_core:          int = 0

    def to_dict(self) -> dict:
        return {
            "account_id":   self.account_id,
            "pagerank":     round(self.pagerank, 6) if self.pagerank is not None else None,
            "betweenness":  round(self.betweenness, 6) if self.betweenness is not None else None,
            "eigenvector":  round(self.eigenvector, 6) if self.eigenvector is not None else None,
            "in_degree":    self.in_degree,
            "out_degree":   self.out_degree,
            "in_amount":    round(self.in_amount, 2),
            "out_amount":   round(self.out_amount, 2),
            "flow_balance": round(self.flow_balance, 4) if self.flow_balance is not None else None,
            "k_core":       self.k_core,
        }


@dataclass
class CentralityResult:
    """Output of CentralityComputer.compute()."""

    period: str
    generated_at: str
    n_nodes: int
    n_edges: int
    measures_computed: list[str] = field(default_factory=list)
    accounts: list[AccountCentrality] = field(default_factory=list)
    elapsed_ms: float = 0.0
    notes: str = ""

    # -- Convenience -------------------------------------------------------

    def top_pagerank(self, n: int = 10) -> list[AccountCentrality]:
        return sorted(
            [a for a in self.accounts if a.pagerank is not None],
            key=lambda a: -(a.pagerank or 0),
        )[:n]

    def top_betweenness(self, n: int = 10) -> list[AccountCentrality]:
        return sorted(
            [a for a in self.accounts if a.betweenness is not None],
            key=lambda a: -(a.betweenness or 0),
        )[:n]

    def passthrough_candidates(
        self,
        max_flow_imbalance: float = 0.10,
        min_in_degree: int = 3,
    ) -> list[AccountCentrality]:
        """Accounts with money_in ~ money_out and many sources -- classic mule signature."""
        out = []
        for a in self.accounts:
            if a.flow_balance is None:
                continue
            if (
                a.flow_balance <= max_flow_imbalance
                and a.in_degree >= min_in_degree
                and a.out_amount > 0
                and a.in_amount > 0
            ):
                out.append(a)
        return sorted(out, key=lambda a: -(a.in_amount + a.out_amount))

    def to_dict(self) -> dict:
        return {
            "period":            self.period,
            "generated_at":      self.generated_at,
            "n_nodes":           self.n_nodes,
            "n_edges":           self.n_edges,
            "measures_computed": self.measures_computed,
            "n_accounts":        len(self.accounts),
            "accounts":          [a.to_dict() for a in self.accounts],
            "elapsed_ms":        round(self.elapsed_ms, 1),
            "notes":             self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def display(self, top_n: int = 10) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        c = Console()
        c.print(Rule(f"[bold]Centrality -- {self.period}[/bold]"))
        c.print(
            f"  Graph      : {self.n_nodes:,} nodes, {self.n_edges:,} edges\n"
            f"  Measured   : {', '.join(self.measures_computed)}\n"
            f"  Accounts   : {len(self.accounts):,}"
        )
        if "pagerank" in self.measures_computed:
            t = Table(
                "Account", "PageRank", "in/out deg",
                "Rs  in", "Rs  out", "balance", "k-core",
                title=f"Top {top_n} by PageRank -- possible hubs",
                show_header=True, header_style="bold dim",
            )
            for a in self.top_pagerank(top_n):
                t.add_row(
                    a.account_id,
                    f"{a.pagerank:.4f}" if a.pagerank else "--",
                    f"{a.in_degree}/{a.out_degree}",
                    f"{a.in_amount:,.0f}",
                    f"{a.out_amount:,.0f}",
                    f"{a.flow_balance:.3f}" if a.flow_balance is not None else "--",
                    str(a.k_core),
                )
            c.print(t)

        cands = self.passthrough_candidates()
        if cands:
            c.print(
                f"\n[bold red]Pass-through mule candidates: "
                f"{len(cands)}[/bold red]  "
                "[dim](in~out, >=3 sources)[/dim]"
            )
            for a in cands[:8]:
                c.print(
                    f"  [red]{a.account_id}[/red]: "
                    f"in_deg={a.in_degree}, Rs in={a.in_amount:,.0f}, "
                    f"Rs out={a.out_amount:,.0f}, bal={a.flow_balance:.3f}"
                )
        c.print()
