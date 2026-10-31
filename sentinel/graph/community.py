"""
sentinel.graph.community
========================

Louvain community detection for mule-ring discovery.

Why Louvain over GNN-based community detection
-----------------------------------------------
- Pure CPU.  Linear-ish complexity in graph size.  No training.
- Well-understood theoretical properties (modularity maximisation).
- Output is interpretable: each community is a concrete set of accounts
  with quantifiable internal density and inter-community flow.
- Defended in regulatory review: "we found these accounts because they
  cluster densely under modularity Q=0.42" beats "the GNN said so."

Mule scoring of a community
---------------------------
A community is a *candidate mule ring* if it shows ALL of:
  - small-to-medium size (3..50 nodes)
  - high internal density (>0.30)
  - balanced flow inside (in ~ out per node)
  - rapid internal txns (median gap < threshold)
  - dominantly internal flow (>70% of edges stay inside)

We compute a 0..1 ``mule_score`` from a weighted combination of these.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sentinel.graph.builder import AccountGraph

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Dataclasses ------------------------------------------------------------


@dataclass
class Community:
    """One detected community with mule-relevant statistics."""

    community_id: int
    nodes: list[str]
    n_nodes: int
    n_internal_edges: int
    density: float                  # 2E / (N(N-1)) for directed
    internal_flow_ratio: float       # internal_edges / total_edges_touching_community
    total_internal_amount: float
    median_gap_minutes: Optional[float] = None
    flow_balance_avg: Optional[float] = None  # mean |in-out|/max(in,out) per node
    mule_score: float = 0.0
    severity: str = "INFO"           # INFO | WARNING | ALERT
    label: str = ""                  # auto-generated descriptor
    top_nodes_by_pagerank: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mule_score"] = round(self.mule_score, 4)
        d["density"] = round(self.density, 4)
        d["internal_flow_ratio"] = round(self.internal_flow_ratio, 4)
        d["total_internal_amount"] = round(self.total_internal_amount, 2)
        if self.median_gap_minutes is not None:
            d["median_gap_minutes"] = round(self.median_gap_minutes, 2)
        if self.flow_balance_avg is not None:
            d["flow_balance_avg"] = round(self.flow_balance_avg, 4)
        return d


@dataclass
class CommunityResult:
    """Output of CommunityFinder.find()."""

    period: str
    generated_at: str
    n_nodes: int
    n_edges: int
    n_communities_total: int
    n_communities_kept: int
    modularity: Optional[float] = None
    communities: list[Community] = field(default_factory=list)
    elapsed_ms: float = 0.0
    notes: str = ""

    # -- Convenience -------------------------------------------------------

    def alerts(self) -> list[Community]:
        return [c for c in self.communities if c.severity == "ALERT"]

    def warnings(self) -> list[Community]:
        return [c for c in self.communities if c.severity == "WARNING"]

    def top_by_score(self, n: int = 10) -> list[Community]:
        return sorted(self.communities, key=lambda c: -c.mule_score)[:n]

    def to_dict(self) -> dict:
        return {
            "period":              self.period,
            "generated_at":        self.generated_at,
            "n_nodes":             self.n_nodes,
            "n_edges":             self.n_edges,
            "n_communities_total": self.n_communities_total,
            "n_communities_kept":  self.n_communities_kept,
            "modularity":          round(self.modularity, 4) if self.modularity is not None else None,
            "n_alerts":            sum(1 for c in self.communities if c.severity == "ALERT"),
            "n_warnings":          sum(1 for c in self.communities if c.severity == "WARNING"),
            "communities":         [c.to_dict() for c in self.communities],
            "elapsed_ms":          round(self.elapsed_ms, 1),
            "notes":               self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )

    def display(self, top_n: int = 15) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        c = Console()
        c.print(Rule(f"[bold]Community Detection -- {self.period}[/bold]"))
        c.print(
            f"  Graph        : {self.n_nodes:,} nodes, {self.n_edges:,} edges\n"
            f"  Communities  : {self.n_communities_kept} kept "
            f"(of {self.n_communities_total} total)"
            + (f"  |  modularity Q={self.modularity:.3f}"
               if self.modularity is not None else "")
        )
        if not self.communities:
            c.print("  [dim]No mule-candidate communities above threshold.[/dim]\n")
            return
        sev_colors = {"ALERT": "red", "WARNING": "yellow", "INFO": "dim"}
        t = Table(
            "ID", "Sev", "Score", "Nodes", "Density",
            "Internal Rs ", "Internal flow", "Label",
            title=f"Top {top_n} candidate mule rings",
            show_header=True, header_style="bold dim",
        )
        for com in self.top_by_score(top_n):
            sc = sev_colors.get(com.severity, "white")
            t.add_row(
                str(com.community_id),
                f"[{sc}]{com.severity}[/{sc}]",
                f"{com.mule_score:.3f}",
                str(com.n_nodes),
                f"{com.density:.2f}",
                f"{com.total_internal_amount:,.0f}",
                f"{com.internal_flow_ratio:.0%}",
                (com.label[:40] + "...") if len(com.label) > 40 else com.label,
            )
        c.print(t)
        c.print()


# -- Detector ---------------------------------------------------------------


class CommunityFinder:
    """
    Louvain community detection, filtered and scored for mule-ring relevance.

    Parameters
    ----------
    min_size : int
        Communities smaller than this are dropped (typical: 3).
    max_size : int
        Communities larger than this are likely organic clusters, not mule
        rings (typical: 50).
    min_density : float
        Minimum directed density to consider as a candidate (typical: 0.10).
    min_internal_flow_ratio : float
        Minimum fraction of edges that stay inside the community.
    resolution : float
        Louvain resolution parameter -- higher = more, smaller communities.
    random_state : int
        Reproducibility.
    """

    def __init__(
        self,
        min_size: int = 3,
        max_size: int = 50,
        min_density: float = 0.10,
        min_internal_flow_ratio: float = 0.30,
        min_mule_score: float = 0.30,
        resolution: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.min_size                = int(min_size)
        self.max_size                = int(max_size)
        self.min_density             = float(min_density)
        self.min_internal_flow_ratio = float(min_internal_flow_ratio)
        self.min_mule_score          = float(min_mule_score)
        self.resolution              = float(resolution)
        self.random_state            = int(random_state)

    # -- Main API ----------------------------------------------------------

    def find(self, graph: AccountGraph) -> CommunityResult:
        import networkx as nx

        t0 = time.perf_counter()
        simple = graph.to_simple()
        und = simple.to_undirected()
        und.remove_edges_from(nx.selfloop_edges(und))

        if und.number_of_nodes() == 0 or und.number_of_edges() == 0:
            return CommunityResult(
                period=graph.period,
                generated_at=_now(),
                n_nodes=graph.n_nodes,
                n_edges=graph.n_edges,
                n_communities_total=0,
                n_communities_kept=0,
            )

        # Louvain
        try:
            partitions = nx.community.louvain_communities(
                und,
                weight="weight",
                resolution=self.resolution,
                seed=self.random_state,
            )
        except Exception as exc:
            log.warning("Louvain failed: %s -- falling back to label propagation", exc)
            partitions = list(nx.community.label_propagation_communities(und))

        # Modularity (best-effort)
        modularity = None
        try:
            modularity = nx.community.modularity(und, partitions, weight="weight")
        except Exception:
            pass

        # Pre-compute PageRank for top-node selection
        try:
            pagerank = nx.pagerank(simple, alpha=0.85, weight="weight")
        except Exception:
            pagerank = {}

        # Pre-compute per-node in/out amounts for flow balance
        in_amt: dict[str, float] = {}
        out_amt: dict[str, float] = {}
        for u, v, data in simple.edges(data=True):
            w = float(data.get("weight", 0.0))
            in_amt[v]  = in_amt.get(v, 0.0) + w
            out_amt[u] = out_amt.get(u, 0.0) + w

        # Build edge timestamp map per node-pair (best ts) for gap computation
        # We use the original multi-graph for raw txn timestamps.
        edge_ts: dict[tuple[str, str], list[pd.Timestamp]] = {}
        for u, v, data in graph.graph.edges(data=True):
            ts = data.get("ts")
            if ts is None:
                continue
            ts = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
            if pd.isna(ts):
                continue
            edge_ts.setdefault((u, v), []).append(ts)

        communities: list[Community] = []
        kept = 0
        for cid, part in enumerate(partitions):
            nodes = list(part)
            if len(nodes) < self.min_size or len(nodes) > self.max_size:
                continue
            community = self._score_community(
                cid=cid,
                nodes=nodes,
                directed=simple,
                multigraph=graph.graph,
                in_amt=in_amt,
                out_amt=out_amt,
                pagerank=pagerank,
                edge_ts=edge_ts,
            )
            if community is None:
                continue
            if community.density < self.min_density:
                continue
            if community.internal_flow_ratio < self.min_internal_flow_ratio:
                continue
            if community.mule_score < self.min_mule_score:
                continue
            communities.append(community)
            kept += 1

        # Sort by score desc and assign severity bands
        communities.sort(key=lambda c: -c.mule_score)
        for c in communities:
            if c.mule_score >= 0.70:
                c.severity = "ALERT"
            elif c.mule_score >= 0.50:
                c.severity = "WARNING"
            else:
                c.severity = "INFO"

        elapsed = (time.perf_counter() - t0) * 1000
        return CommunityResult(
            period=graph.period,
            generated_at=_now(),
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            n_communities_total=len(partitions),
            n_communities_kept=kept,
            modularity=modularity,
            communities=communities,
            elapsed_ms=round(elapsed, 1),
        )

    # -- Scoring helper ----------------------------------------------------

    def _score_community(
        self,
        cid: int,
        nodes: list[str],
        directed,
        multigraph,
        in_amt: dict,
        out_amt: dict,
        pagerank: dict,
        edge_ts: dict,
    ) -> Optional[Community]:
        node_set = set(nodes)
        n = len(nodes)
        if n < 2:
            return None

        # Internal vs external edges (directed)
        internal_edges = 0
        external_edges = 0
        internal_amount = 0.0
        for u, v, data in directed.edges(data=True):
            if u in node_set and v in node_set:
                internal_edges += 1
                internal_amount += float(data.get("weight", 0.0))
            elif u in node_set or v in node_set:
                external_edges += 1
        total_touching = internal_edges + external_edges
        internal_flow_ratio = (
            internal_edges / total_touching if total_touching > 0 else 0.0
        )
        # Directed density
        density = internal_edges / (n * (n - 1)) if n > 1 else 0.0

        # Per-node flow balance among internal flows
        balances = []
        for node in nodes:
            in_ = in_amt.get(node, 0.0)
            out_ = out_amt.get(node, 0.0)
            if max(in_, out_) > 0:
                balances.append(abs(in_ - out_) / max(in_, out_))
        balance_avg = float(np.mean(balances)) if balances else None

        # Median internal gap (across all internal edge timestamps, sorted)
        internal_ts: list[pd.Timestamp] = []
        for (u, v), ts_list in edge_ts.items():
            if u in node_set and v in node_set:
                internal_ts.extend(ts_list)
        median_gap = None
        if len(internal_ts) >= 2:
            sorted_ts = sorted(internal_ts)
            gaps = pd.Series(sorted_ts).diff().dropna().dt.total_seconds() / 60.0
            median_gap = float(gaps.median()) if len(gaps) > 0 else None

        # Top nodes by PageRank within community
        top_pr = sorted(
            [(node, pagerank.get(node, 0.0)) for node in nodes],
            key=lambda x: -x[1],
        )[:5]
        top_nodes = [n[0] for n in top_pr]

        # Composite mule score
        # - density (0.30) - dense communities are more suspicious
        # - internal_flow_ratio (0.30) - inward-focused flow
        # - flow balance (0.20) - balanced in/out per node
        # - velocity (0.20) - rapid internal txns
        density_score = min(1.0, density / 0.40)
        flow_score    = min(1.0, internal_flow_ratio / 0.80)
        balance_score = (
            max(0.0, 1.0 - balance_avg / 0.30) if balance_avg is not None else 0.0
        )
        velocity_score = (
            max(0.0, 1.0 - median_gap / 60.0) if median_gap is not None else 0.0
        )
        mule_score = round(
            0.30 * density_score
            + 0.30 * flow_score
            + 0.20 * balance_score
            + 0.20 * velocity_score,
            4,
        )

        # Auto-label
        descriptors = []
        if density >= 0.40:
            descriptors.append("dense")
        if internal_flow_ratio >= 0.70:
            descriptors.append("inward-focused")
        if balance_avg is not None and balance_avg <= 0.20:
            descriptors.append("balanced-flow")
        if median_gap is not None and median_gap <= 30:
            descriptors.append("high-velocity")
        if not descriptors:
            descriptors.append("loose-cluster")
        label = " / ".join(descriptors) + f" ({n} nodes)"

        return Community(
            community_id=cid,
            nodes=nodes,
            n_nodes=n,
            n_internal_edges=internal_edges,
            density=density,
            internal_flow_ratio=internal_flow_ratio,
            total_internal_amount=internal_amount,
            median_gap_minutes=median_gap,
            flow_balance_avg=balance_avg,
            mule_score=mule_score,
            label=label,
            top_nodes_by_pagerank=top_nodes,
        )
