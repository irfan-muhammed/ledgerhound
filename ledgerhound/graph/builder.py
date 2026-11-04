"""
ledgerhound.graph.builder
======================

Build a directed multigraph of account-to-account money flows from a
transaction DataFrame.

Design notes
------------
- Multi-edge: every transaction is its own edge.  Aggregation (sum amount,
  count edges) is done lazily by the consumer.
- Timestamp-aware: edge attribute ``ts`` carries the parsed datetime so
  motif detectors can apply time windows.
- Status-aware: by default we only include SUCCESS transactions.  Failed
  / reversed txns are useful in their own right (signal feature) but
  contaminate motif counts.
- VPA-vs-account: if a ``payer_account_col`` AND ``payee_account_col`` are
  passed, the builder collapses VPAs to their underlying bank account.
  Otherwise the graph is at VPA layer (the honest default for the organization data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

log = logging.getLogger(__name__)


# -- Wrapper class ---------------------------------------------------------


@dataclass
class AccountGraph:
    """Wrapper around a NetworkX MultiDiGraph carrying provenance metadata."""

    graph: "nx.MultiDiGraph"
    period: str = "current"
    n_transactions: int = 0
    total_amount: float = 0.0
    payer_col: str = "payer"
    payee_col: str = "payee"
    amount_col: str = "amount"
    timestamp_col: str = "timestamp"

    # -- Convenience accessors ----------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def neighbors_in(self, node: str) -> list[str]:
        return list(self.graph.predecessors(node))

    def neighbors_out(self, node: str) -> list[str]:
        return list(self.graph.successors(node))

    def in_degree(self, node: str) -> int:
        return self.graph.in_degree(node)

    def out_degree(self, node: str) -> int:
        return self.graph.out_degree(node)

    def in_amount(self, node: str) -> float:
        total = 0.0
        for u, _, data in self.graph.in_edges(node, data=True):
            total += float(data.get("amount", 0.0))
        return total

    def out_amount(self, node: str) -> float:
        total = 0.0
        for _, v, data in self.graph.out_edges(node, data=True):
            total += float(data.get("amount", 0.0))
        return total

    def edges_between(self, u: str, v: str) -> list[dict]:
        if not self.graph.has_edge(u, v):
            return []
        return [data for _, _, data in self.graph.edges([u], data=True) if _ == u and data]

    def to_simple(self) -> "nx.DiGraph":
        """Collapse multi-edges into a simple DiGraph weighted by sum amount."""
        import networkx as nx
        simple = nx.DiGraph()
        simple.add_nodes_from(self.graph.nodes(data=True))
        for u, v, data in self.graph.edges(data=True):
            amt = float(data.get("amount", 0.0))
            if simple.has_edge(u, v):
                simple[u][v]["weight"] += amt
                simple[u][v]["count"] += 1
            else:
                simple.add_edge(u, v, weight=amt, count=1)
        return simple

    def info(self) -> str:
        return (
            f"AccountGraph(period={self.period!r}, "
            f"nodes={self.n_nodes:,}, edges={self.n_edges:,}, "
            f"txns={self.n_transactions:,}, Rs {self.total_amount:,.0f})"
        )

    __repr__ = info


# -- Builder ----------------------------------------------------------------


class AccountGraphBuilder:
    """
    Build an AccountGraph from a transaction DataFrame.

    Parameters
    ----------
    payer_col, payee_col : str
        Column names for the source and destination of each transaction.
        Defaults match the synthetic UPI generator: ``account_id`` and
        ``beneficiary_id``.
    amount_col, timestamp_col : str
        Columns for amount and timestamp.
    status_col : str | None
        If provided, transactions whose status is not in ``success_values``
        are dropped.
    success_values : iterable[str]
        Status values to keep (default: ``("SUCCESS",)``).
    payer_account_col, payee_account_col : str | None
        If both are provided, the graph is collapsed from VPA to underlying
        bank account.  Otherwise the graph is at VPA layer.
    min_amount : float
        Edges below this threshold are dropped (default 0 = keep all).
    """

    def __init__(
        self,
        payer_col: str = "account_id",
        payee_col: str = "beneficiary_id",
        amount_col: str = "txn_amount",
        timestamp_col: str = "txn_date",
        status_col: Optional[str] = "status",
        success_values: tuple[str, ...] = ("SUCCESS",),
        payer_account_col: Optional[str] = None,
        payee_account_col: Optional[str] = None,
        min_amount: float = 0.0,
    ) -> None:
        self.payer_col          = payer_col
        self.payee_col          = payee_col
        self.amount_col         = amount_col
        self.timestamp_col      = timestamp_col
        self.status_col         = status_col
        self.success_values     = set(success_values)
        self.payer_account_col  = payer_account_col
        self.payee_account_col  = payee_account_col
        self.min_amount         = float(min_amount)

    # -- Build -------------------------------------------------------------

    def build(
        self,
        txn_df: pd.DataFrame,
        period: str = "current",
    ) -> AccountGraph:
        import networkx as nx

        df = self._prepare(txn_df)
        if df.empty:
            return AccountGraph(
                graph=nx.MultiDiGraph(),
                period=period,
                payer_col=self.payer_col,
                payee_col=self.payee_col,
                amount_col=self.amount_col,
                timestamp_col=self.timestamp_col,
            )

        # Decide node layer (VPA vs account)
        if self.payer_account_col and self.payee_account_col:
            payer_id = df[self.payer_account_col].astype(str)
            payee_id = df[self.payee_account_col].astype(str)
        else:
            payer_id = df[self.payer_col].astype(str)
            payee_id = df[self.payee_col].astype(str)

        amounts = df[self.amount_col].astype(float)
        ts = df[self.timestamp_col]

        g = nx.MultiDiGraph()

        # Bulk-add nodes for speed
        all_nodes = pd.unique(pd.concat([payer_id, payee_id], ignore_index=True))
        g.add_nodes_from(all_nodes.tolist())

        # Build edge list
        edges = []
        for u, v, amt, t in zip(
            payer_id.tolist(),
            payee_id.tolist(),
            amounts.tolist(),
            ts.tolist(),
        ):
            edges.append((u, v, {"amount": amt, "ts": t}))
        g.add_edges_from(edges)

        log.info(
            "AccountGraph built: %d nodes, %d edges, %d txns",
            g.number_of_nodes(),
            g.number_of_edges(),
            len(df),
        )

        return AccountGraph(
            graph=g,
            period=period,
            n_transactions=int(len(df)),
            total_amount=float(amounts.sum()),
            payer_col=self.payer_col,
            payee_col=self.payee_col,
            amount_col=self.amount_col,
            timestamp_col=self.timestamp_col,
        )

    # -- Preparation -------------------------------------------------------

    def _prepare(self, txn_df: pd.DataFrame) -> pd.DataFrame:
        required = [self.payer_col, self.payee_col, self.amount_col, self.timestamp_col]
        missing = [c for c in required if c not in txn_df.columns]
        if missing:
            raise ValueError(
                f"AccountGraphBuilder: required columns missing: {missing}. "
                f"Available: {list(txn_df.columns)[:30]}"
            )

        df = txn_df[
            [c for c in (
                self.payer_col, self.payee_col, self.amount_col, self.timestamp_col,
                self.status_col, self.payer_account_col, self.payee_account_col,
            ) if c]
        ].copy()

        # Filter by status
        if self.status_col and self.status_col in df.columns:
            df = df[df[self.status_col].astype(str).str.upper().isin(
                {s.upper() for s in self.success_values}
            )]

        # Drop nulls in critical fields
        df = df.dropna(subset=[self.payer_col, self.payee_col, self.amount_col])

        # Filter by amount
        if self.min_amount > 0:
            df = df[df[self.amount_col].astype(float) >= self.min_amount]

        # Parse timestamps
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col], errors="coerce", utc=False,
            )
        df = df.dropna(subset=[self.timestamp_col])

        return df.reset_index(drop=True)
