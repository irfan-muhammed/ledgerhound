"""
ledgerhound.graph.flow
===================

Money-flow tracing: given a tainted source (a known fraud account, a
mule, an account flagged by signal decay) trace where the money went --
or where it came from.

Three operations
----------------
1. ``trace_downstream``   BFS forward from a source account, capturing all
                          accounts reachable within ``max_hops`` and the
                          edges (txns) used.  Annotates each reachable
                          node with: distance, total amount inflow on that
                          path, holding time before forwarding.
2. ``trace_upstream``     mirror -- find everyone who sent money to a sink.
3. ``shortest_paths``     k-shortest simple paths from source to sink,
                          ordered by hop count (ties broken by amount).

These are the canonical AML forensic primitives.  None of them require
GPU; all run on CPU via NetworkX.

Honest limits
-------------
- We trace UPI-internal flows.  Cash-out at ATMs and cross-rail flows
  (NEFT, IMPS, RTGS) terminate the trace.
- ``trace_downstream`` returns nodes reachable in any number of hops
  <= ``max_hops``; for diamond-shaped flows the same node may appear at
  multiple distances -- we keep the *shortest* distance.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ledgerhound.graph.builder import AccountGraph

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Dataclasses ------------------------------------------------------------


@dataclass
class FlowPath:
    """One concrete path through the graph."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    hops: int
    total_amount: float
    min_amount: float                # bottleneck amount along the path
    holding_time_hours: float        # span between first and last txn
    timestamps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes":              self.nodes,
            "edges":              [list(e) for e in self.edges],
            "hops":               self.hops,
            "total_amount":       round(self.total_amount, 2),
            "min_amount":         round(self.min_amount, 2),
            "holding_time_hours": round(self.holding_time_hours, 2),
            "timestamps":         self.timestamps,
        }


@dataclass
class FlowResult:
    """Output of MoneyFlowTracer.trace_*()."""

    source: str
    direction: str                                         # "downstream" | "upstream" | "path"
    sink: Optional[str] = None                             # only for path
    period: str = "current"
    generated_at: str = field(default_factory=_now)

    n_reachable: int = 0
    nodes_by_distance: dict[int, list[str]] = field(default_factory=dict)
    paths: list[FlowPath] = field(default_factory=list)
    total_amount_traced: float = 0.0
    elapsed_ms: float = 0.0
    notes: str = ""

    # -- Convenience -------------------------------------------------------

    def reachable_nodes(self) -> list[str]:
        return [n for nodes in self.nodes_by_distance.values() for n in nodes]

    def to_dict(self) -> dict:
        return {
            "source":              self.source,
            "direction":           self.direction,
            "sink":                self.sink,
            "period":              self.period,
            "generated_at":        self.generated_at,
            "n_reachable":         self.n_reachable,
            "nodes_by_distance":   {str(k): v for k, v in self.nodes_by_distance.items()},
            "n_paths":             len(self.paths),
            "paths":               [p.to_dict() for p in self.paths],
            "total_amount_traced": round(self.total_amount_traced, 2),
            "elapsed_ms":          round(self.elapsed_ms, 1),
            "notes":               self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )

    def display(self, max_paths: int = 10, max_per_distance: int = 8) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        c = Console()
        arrow = "" if self.direction == "downstream" else (
            "" if self.direction == "upstream" else "->"
        )
        title = (
            f"Flow Trace [{self.direction}] {arrow} from [bold]{self.source}[/bold]"
            + (f"  to [bold]{self.sink}[/bold]" if self.sink else "")
        )
        c.print(Rule(title))
        c.print(
            f"  Reachable    : {self.n_reachable:,} accounts\n"
            f"  Total Rs       : {self.total_amount_traced:,.0f}\n"
            f"  Paths found  : {len(self.paths)}"
        )

        if self.nodes_by_distance:
            t = Table("Distance", "Count", "Sample accounts",
                      title="Nodes by hop distance",
                      show_header=True, header_style="bold dim")
            for d in sorted(self.nodes_by_distance.keys()):
                nodes = self.nodes_by_distance[d]
                sample = ", ".join(nodes[:max_per_distance]) + (
                    f"  ...+{len(nodes) - max_per_distance}"
                    if len(nodes) > max_per_distance else ""
                )
                t.add_row(str(d), str(len(nodes)), sample[:80])
            c.print(t)

        if self.paths:
            c.print(f"\n[bold]Top {min(max_paths, len(self.paths))} paths:[/bold]")
            for i, p in enumerate(self.paths[:max_paths], 1):
                chain = " -> ".join(p.nodes[:6]) + ("..." if len(p.nodes) > 6 else "")
                c.print(
                    f"  [{i}] hops={p.hops}  "
                    f"Rs min={p.min_amount:,.0f}  "
                    f"Rs total={p.total_amount:,.0f}  "
                    f"hold={p.holding_time_hours:.1f}h\n"
                    f"      {chain}"
                )
        c.print()


# -- Tracer -----------------------------------------------------------------


class MoneyFlowTracer:
    """
    BFS-based forensic money-flow tracer.

    Parameters
    ----------
    max_hops : int
        Maximum BFS depth (default: 6).  Above 6 the noise dominates.
    max_paths : int
        Maximum simple paths to return for shortest_paths (default: 20).
    min_amount : float
        Edges below this amount are ignored (default: 0).
    enforce_time_order : bool
        If True, only follow edges with ts > parent edge's ts.  Required
        for valid money-flow chains.  Default True.
    """

    def __init__(
        self,
        max_hops: int = 6,
        max_paths: int = 20,
        min_amount: float = 0.0,
        enforce_time_order: bool = True,
    ) -> None:
        self.max_hops          = int(max_hops)
        self.max_paths         = int(max_paths)
        self.min_amount        = float(min_amount)
        self.enforce_time_order = bool(enforce_time_order)

    # -- Downstream ---------------------------------------------------------

    def trace_downstream(
        self,
        graph: AccountGraph,
        source: str,
        max_hops: Optional[int] = None,
    ) -> FlowResult:
        return self._trace(graph, source, max_hops, direction="downstream")

    def trace_upstream(
        self,
        graph: AccountGraph,
        sink: str,
        max_hops: Optional[int] = None,
    ) -> FlowResult:
        return self._trace(graph, sink, max_hops, direction="upstream")

    def _trace(
        self,
        graph: AccountGraph,
        root: str,
        max_hops: Optional[int],
        direction: str,
    ) -> FlowResult:
        t0 = time.perf_counter()
        if root not in graph.graph:
            return FlowResult(
                source=root,
                direction=direction,
                period=graph.period,
                notes=f"Account '{root}' not in graph.",
            )

        max_h = self.max_hops if max_hops is None else int(max_hops)

        # BFS
        visited: dict[str, int] = {root: 0}      # node -> shortest distance
        node_first_ts: dict[str, pd.Timestamp] = {}  # earliest arrival time
        amount_received: dict[str, float] = {}   # money flowing in
        queue = deque()
        queue.append((root, 0, None))            # (node, distance, parent_ts)

        edges_used: list[tuple] = []
        total_amount = 0.0

        while queue:
            node, dist, parent_ts = queue.popleft()
            if dist >= max_h:
                continue

            # Pick edges in the right direction
            if direction == "downstream":
                neighbors_iter = graph.graph.out_edges(node, data=True)
                for u, v, data in neighbors_iter:
                    self._maybe_advance(
                        v, dist, parent_ts, data,
                        visited, node_first_ts, amount_received,
                        queue, edges_used,
                    )
            else:
                neighbors_iter = graph.graph.in_edges(node, data=True)
                for u, v, data in neighbors_iter:
                    self._maybe_advance(
                        u, dist, parent_ts, data,
                        visited, node_first_ts, amount_received,
                        queue, edges_used,
                    )

        # Aggregate
        nodes_by_distance: dict[int, list[str]] = {}
        for node, d in visited.items():
            if node == root:
                continue
            nodes_by_distance.setdefault(d, []).append(node)

        total_amount = sum(amount_received.values())
        n_reachable = sum(len(v) for v in nodes_by_distance.values())

        elapsed = (time.perf_counter() - t0) * 1000
        result = FlowResult(
            source=root,
            direction=direction,
            period=graph.period,
            n_reachable=n_reachable,
            nodes_by_distance=nodes_by_distance,
            total_amount_traced=total_amount,
            elapsed_ms=round(elapsed, 1),
        )
        # Best-path-per-leaf paths (one per terminal node, max depth)
        result.paths = self._reconstruct_paths(
            graph, root, visited, edges_used,
            direction=direction, max_paths=self.max_paths,
        )
        return result

    def _maybe_advance(
        self,
        nxt: str,
        dist: int,
        parent_ts: Optional[pd.Timestamp],
        data: dict,
        visited: dict,
        node_first_ts: dict,
        amount_received: dict,
        queue: deque,
        edges_used: list,
    ) -> None:
        amt = float(data.get("amount", 0.0))
        if amt < self.min_amount:
            return
        ts = data.get("ts")
        if ts is not None and not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if self.enforce_time_order and parent_ts is not None and ts is not None and ts <= parent_ts:
            return

        new_dist = dist + 1
        if nxt not in visited or visited[nxt] > new_dist:
            visited[nxt] = new_dist
            node_first_ts[nxt] = ts
            amount_received[nxt] = amount_received.get(nxt, 0.0) + amt
            queue.append((nxt, new_dist, ts))
            edges_used.append((dist, nxt, amt, ts))

    def _reconstruct_paths(
        self,
        graph: AccountGraph,
        root: str,
        visited: dict,
        edges_used: list,
        direction: str,
        max_paths: int,
    ) -> list[FlowPath]:
        # We didn't track parent pointers in BFS to keep it light.  Reconstruct
        # with networkx shortest_simple_paths from root to each terminal node,
        # capped at max_paths total.
        import networkx as nx
        simple = graph.to_simple()
        if direction == "upstream":
            simple = simple.reverse(copy=True)
        if root not in simple:
            return []

        # Pick terminal nodes: deepest distance per "tree"
        max_d = max(visited.values()) if visited else 0
        terminals = [n for n, d in visited.items() if d == max_d and n != root]
        if not terminals:
            terminals = [n for n, d in visited.items() if d > 0]
        terminals = terminals[:max_paths]

        paths: list[FlowPath] = []
        for term in terminals:
            try:
                path_iter = nx.shortest_simple_paths(simple, root, term, weight=None)
                node_path = next(path_iter)
            except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
                continue
            edges = list(zip(node_path[:-1], node_path[1:]))

            # Compute path metrics from the multigraph
            total_amt = 0.0
            min_amt = float("inf")
            timestamps: list[pd.Timestamp] = []
            for u, v in edges:
                if direction == "upstream":
                    src, dst = v, u
                else:
                    src, dst = u, v
                if not graph.graph.has_edge(src, dst):
                    continue
                # Pick the smallest amount (bottleneck) for min, sum amounts
                edge_amts = []
                edge_ts: list[pd.Timestamp] = []
                for _, _, data in graph.graph.out_edges(src, data=True):
                    if _ != src:
                        continue
                for _, dst_node, data in graph.graph.out_edges(src, data=True):
                    if dst_node != dst:
                        continue
                    a = float(data.get("amount", 0.0))
                    edge_amts.append(a)
                    ts = data.get("ts")
                    if ts is not None:
                        edge_ts.append(pd.Timestamp(ts))
                if edge_amts:
                    avg = sum(edge_amts) / len(edge_amts)
                    total_amt += avg
                    min_amt = min(min_amt, min(edge_amts))
                    if edge_ts:
                        timestamps.append(min(edge_ts))
            if min_amt == float("inf"):
                min_amt = 0.0
            holding = 0.0
            if len(timestamps) >= 2:
                holding = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0

            paths.append(FlowPath(
                nodes=list(node_path),
                edges=edges,
                hops=len(edges),
                total_amount=total_amt,
                min_amount=min_amt,
                holding_time_hours=holding,
                timestamps=[str(t) for t in sorted(timestamps)],
            ))

        # Sort by hops desc, then total_amount desc -- most "interesting" first
        paths.sort(key=lambda p: (-p.hops, -p.total_amount))
        return paths[:max_paths]

    # -- Source-to-sink k-shortest -------------------------------------------

    def shortest_paths(
        self,
        graph: AccountGraph,
        source: str,
        sink: str,
        k: Optional[int] = None,
    ) -> FlowResult:
        import networkx as nx
        t0 = time.perf_counter()
        if source not in graph.graph or sink not in graph.graph:
            return FlowResult(
                source=source, direction="path", sink=sink,
                period=graph.period,
                notes="Source or sink not in graph.",
            )

        simple = graph.to_simple()
        try:
            path_iter = nx.shortest_simple_paths(simple, source, sink)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return FlowResult(
                source=source, direction="path", sink=sink,
                period=graph.period,
                notes="No path from source to sink.",
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

        max_k = self.max_paths if k is None else int(k)
        paths: list[FlowPath] = []
        total_amt_traced = 0.0
        for i, node_path in enumerate(path_iter):
            if i >= max_k:
                break
            edges = list(zip(node_path[:-1], node_path[1:]))
            total_amt = 0.0
            min_amt = float("inf")
            timestamps: list[pd.Timestamp] = []
            for u, v in edges:
                amts = []
                ts_list: list[pd.Timestamp] = []
                for _, dst, data in graph.graph.out_edges(u, data=True):
                    if dst != v:
                        continue
                    amts.append(float(data.get("amount", 0.0)))
                    ts = data.get("ts")
                    if ts is not None:
                        ts_list.append(pd.Timestamp(ts))
                if amts:
                    avg = sum(amts) / len(amts)
                    total_amt += avg
                    min_amt = min(min_amt, min(amts))
                    if ts_list:
                        timestamps.append(min(ts_list))
            if min_amt == float("inf"):
                min_amt = 0.0
            holding = 0.0
            if len(timestamps) >= 2:
                holding = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
            total_amt_traced += total_amt
            paths.append(FlowPath(
                nodes=list(node_path),
                edges=edges,
                hops=len(edges),
                total_amount=total_amt,
                min_amount=min_amt,
                holding_time_hours=holding,
                timestamps=[str(t) for t in sorted(timestamps)],
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        return FlowResult(
            source=source,
            sink=sink,
            direction="path",
            period=graph.period,
            n_reachable=len({n for p in paths for n in p.nodes}),
            paths=paths,
            total_amount_traced=total_amt_traced,
            elapsed_ms=round(elapsed, 1),
        )
