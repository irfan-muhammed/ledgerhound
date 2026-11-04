"""
ledgerhound.graph.centrality
=========================

Centrality computation for an AccountGraph.

Measures
--------
- **PageRank**       hub-likelihood under random-walk model
- **Betweenness**    proportion of shortest paths that cross this node
                     -- the canonical "money-conduit" measure for mules
- **Eigenvector**    importance via importance-of-neighbours
- **k-core**         structural redundancy / cohesion
- **In/out degree** + **flow_balance**

For the agent's mule-finding workflow, the most informative measures are:
  - betweenness  -> finds money-flow conduits (mules)
  - flow_balance -> finds passthrough accounts
  - k-core       -> finds dense mule rings

Performance
-----------
Exact betweenness is O(N.E) -- fine up to ~10K nodes / ~50K edges.  For
larger graphs we use approximate betweenness with k-sample sources.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ledgerhound.graph.builder import AccountGraph
from ledgerhound.graph.result import AccountCentrality, CentralityResult

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CentralityComputer:
    """
    Compute centrality measures for all nodes in an AccountGraph.

    Parameters
    ----------
    measures : iterable[str]
        Subset of {"pagerank", "betweenness", "eigenvector", "k_core"}.
        Defaults to all four.
    betweenness_k : int | None
        If set and ``n_nodes > approx_threshold``, use k-sample approximate
        betweenness (NetworkX ``betweenness_centrality(k=...)``) for speed.
    approx_threshold : int
        Above this node count, switch to approximate betweenness.
    """

    DEFAULT_MEASURES = ("pagerank", "betweenness", "k_core")

    def __init__(
        self,
        measures: tuple[str, ...] = DEFAULT_MEASURES,
        betweenness_k: Optional[int] = None,
        approx_threshold: int = 5_000,
        pagerank_alpha: float = 0.85,
    ) -> None:
        self.measures          = tuple(measures)
        self.betweenness_k     = betweenness_k
        self.approx_threshold  = int(approx_threshold)
        self.pagerank_alpha    = float(pagerank_alpha)

    # -- Public API ---------------------------------------------------------

    def compute(self, graph: AccountGraph) -> CentralityResult:
        import networkx as nx
        t0 = time.perf_counter()

        simple = graph.to_simple()
        nodes = list(simple.nodes())

        # Pre-compute degree + amount aggregates
        in_deg  = dict(simple.in_degree())
        out_deg = dict(simple.out_degree())
        in_amt: dict[str, float]  = {}
        out_amt: dict[str, float] = {}
        for u, v, data in simple.edges(data=True):
            w = float(data.get("weight", 0.0))
            in_amt[v]  = in_amt.get(v, 0.0) + w
            out_amt[u] = out_amt.get(u, 0.0) + w

        pagerank: dict = {}
        betweenness: dict = {}
        eigenvector: dict = {}
        kcore: dict = {}

        if "pagerank" in self.measures:
            try:
                pagerank = nx.pagerank(
                    simple, alpha=self.pagerank_alpha, weight="weight",
                )
            except Exception as exc:
                log.warning("pagerank failed: %s", exc)

        if "betweenness" in self.measures:
            try:
                if self.betweenness_k is not None:
                    k = self.betweenness_k
                elif len(nodes) > self.approx_threshold:
                    k = min(self.approx_threshold, max(100, len(nodes) // 50))
                else:
                    k = None
                betweenness = nx.betweenness_centrality(
                    simple, k=k, weight="weight", normalized=True,
                )
            except Exception as exc:
                log.warning("betweenness failed: %s", exc)

        if "eigenvector" in self.measures:
            try:
                eigenvector = nx.eigenvector_centrality_numpy(
                    simple, weight="weight",
                )
            except Exception as exc:
                log.warning("eigenvector failed: %s", exc)

        if "k_core" in self.measures:
            try:
                # core_number requires undirected graph
                und = simple.to_undirected()
                und.remove_edges_from(nx.selfloop_edges(und))
                kcore = nx.core_number(und)
            except Exception as exc:
                log.warning("k_core failed: %s", exc)

        # Build per-account result
        accounts: list[AccountCentrality] = []
        for n in nodes:
            in_a  = in_amt.get(n, 0.0)
            out_a = out_amt.get(n, 0.0)
            balance = None
            if max(in_a, out_a) > 0:
                balance = abs(in_a - out_a) / max(in_a, out_a)
            accounts.append(AccountCentrality(
                account_id=str(n),
                pagerank=pagerank.get(n) if pagerank else None,
                betweenness=betweenness.get(n) if betweenness else None,
                eigenvector=eigenvector.get(n) if eigenvector else None,
                in_degree=int(in_deg.get(n, 0)),
                out_degree=int(out_deg.get(n, 0)),
                in_amount=in_a,
                out_amount=out_a,
                flow_balance=balance,
                k_core=int(kcore.get(n, 0)),
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "Centrality computed in %.0fms for %d nodes (%s)",
            elapsed, len(nodes), ", ".join(self.measures),
        )

        return CentralityResult(
            period=graph.period,
            generated_at=_now(),
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            measures_computed=list(self.measures),
            accounts=accounts,
            elapsed_ms=round(elapsed, 1),
        )
