"""
ledgerhound.graph.motifs
=====================

Mule motif catalogue.

Each pattern is a small class implementing ``detect(graph) -> list[MotifMatch]``.
The catalogue is grounded in real AML typologies that FIU-IND, RBI, and FATF
explicitly call out -- none of these patterns require demographic data.

Patterns
--------
1. ``FanInCollector``     N->1 in short window  (smurfing collector)
2. ``FanOutDistributor``  1->N rapid distribution (smurfing distributor)
3. ``LayeringChain``      A->B->C->D... long unidirectional pass-through
4. ``CyclicRing``         A->B->C->A round-tripping
5. ``VelocityHub``        anomalously bursty txn pattern
6. ``PassThrough``        money_in ~ money_out within hold-time T

Severity scale
--------------
INFO     suggestive but inconclusive (single short fan-out, low velocity)
WARNING  pattern matches but small magnitude or wide window
ALERT    high magnitude, tight window, and matches multiple criteria

Scores
------
Score is a 0..1 normalisation per pattern (defined inside each detector).
The agent should treat ``score >= 0.7`` as actionable.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from ledgerhound.graph.builder import AccountGraph
from ledgerhound.graph.result import MotifMatch, MotifDetectionResult

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Base class -------------------------------------------------------------


class MotifPattern:
    """Base interface for all mule motif detectors."""

    name: str = "base"
    description: str = ""

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        raise NotImplementedError


# -- 1. Fan-in collector ----------------------------------------------------


class FanInCollector(MotifPattern):
    """Many sources converge on one account in a short time window.

    Classic smurfing collector signature: N small inbound transfers from
    distinct accounts inside ``window_hours``, summing to >= ``min_total_amount``.

    Score = clip(n_distinct_sources / target_sources, 0, 1) weighted by
    amount and tightness of the time window.
    """

    name = "fan_in_collector"
    description = "Many sources -> 1 account, short window"

    def __init__(
        self,
        min_distinct_sources: int = 5,
        target_sources: int = 10,
        window_hours: float = 24.0,
        min_total_amount: float = 50_000.0,
        min_score: float = 0.30,
    ) -> None:
        self.min_distinct_sources = int(min_distinct_sources)
        self.target_sources       = int(target_sources)
        self.window_hours         = float(window_hours)
        self.min_total_amount     = float(min_total_amount)
        self.min_score            = float(min_score)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        out: list[MotifMatch] = []
        win = pd.Timedelta(hours=self.window_hours)

        for node in graph.graph.nodes():
            in_edges = list(graph.graph.in_edges(node, data=True))
            if len(in_edges) < self.min_distinct_sources:
                continue

            # Build a sorted list of (src, amt, ts) for inbound txns
            rows = []
            for u, _, data in in_edges:
                ts = pd.Timestamp(data.get("ts"))
                if pd.isna(ts):
                    continue
                rows.append((u, float(data.get("amount", 0.0)), ts))
            if len(rows) < self.min_distinct_sources:
                continue
            rows.sort(key=lambda r: r[2])

            # Two-pointer sliding window -- manual, avoids pandas string-rolling issues
            best_window = self._best_window(rows, win)
            if best_window is None:
                continue
            window_start, window_end, sources, total_amt, n_txns = best_window

            n_sources = len(sources)
            if n_sources < self.min_distinct_sources or total_amt < self.min_total_amount:
                continue

            src_score = min(1.0, n_sources / self.target_sources)
            amt_score = min(1.0, total_amt / (5 * self.min_total_amount))
            score = round(0.65 * src_score + 0.35 * amt_score, 4)

            if score < self.min_score:
                continue

            severity = (
                "ALERT" if score >= 0.70 else
                "WARNING" if score >= 0.45 else
                "INFO"
            )

            sources_list = list(sources)
            edges = [(s, node) for s in sources_list]

            out.append(MotifMatch(
                pattern_name=self.name,
                severity=severity,
                score=score,
                nodes=[node] + sources_list,
                edges=edges,
                n_transactions=n_txns,
                total_amount=total_amt,
                time_window_start=str(window_start),
                time_window_end=str(window_end),
                metadata={
                    "collector":     node,
                    "n_sources":     n_sources,
                    "highlight":     f"collector={node}, {n_sources} sources, Rs {total_amt:,.0f}",
                    "window_hours":  self.window_hours,
                },
            ))
        return out

    @staticmethod
    def _best_window(
        rows: list[tuple[str, float, pd.Timestamp]],
        win: pd.Timedelta,
    ):
        """Two-pointer sweep: find time window with most distinct first-elements.

        Each row is (party_id, amount, ts).  Returns (start_ts, end_ts,
        set(party_ids), total_amount, n_txns) for the best window, or None.
        """
        if not rows:
            return None
        best = None
        from collections import defaultdict
        counts: dict[str, int] = defaultdict(int)
        amount_sum = 0.0
        i = 0  # left pointer
        for j in range(len(rows)):
            party_j, amt_j, ts_j = rows[j]
            counts[party_j] += 1
            amount_sum += amt_j
            # Shrink window from the left until span <= win
            while ts_j - rows[i][2] > win:
                party_i, amt_i, _ = rows[i]
                counts[party_i] -= 1
                if counts[party_i] == 0:
                    del counts[party_i]
                amount_sum -= amt_i
                i += 1
            n_unique = len(counts)
            if best is None or n_unique > best[0] or (
                n_unique == best[0] and amount_sum > best[1]
            ):
                best = (n_unique, amount_sum, i, j, set(counts.keys()), amount_sum)
        if best is None:
            return None
        n_unique, _, i_left, j_right, parties, total_amt = best
        return (rows[i_left][2], rows[j_right][2], parties, total_amt, j_right - i_left + 1)


# -- 2. Fan-out distributor ------------------------------------------------


class FanOutDistributor(MotifPattern):
    """One account distributes to many destinations rapidly.

    Mirror of FanInCollector -- same scoring shape, opposite direction.
    """

    name = "fan_out_distributor"
    description = "1 account -> many destinations, short window"

    def __init__(
        self,
        min_distinct_destinations: int = 5,
        target_destinations: int = 10,
        window_hours: float = 24.0,
        min_total_amount: float = 50_000.0,
        min_score: float = 0.30,
    ) -> None:
        self.min_distinct_destinations = int(min_distinct_destinations)
        self.target_destinations       = int(target_destinations)
        self.window_hours              = float(window_hours)
        self.min_total_amount          = float(min_total_amount)
        self.min_score                 = float(min_score)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        out: list[MotifMatch] = []
        win = pd.Timedelta(hours=self.window_hours)

        for node in graph.graph.nodes():
            out_edges = list(graph.graph.out_edges(node, data=True))
            if len(out_edges) < self.min_distinct_destinations:
                continue

            rows = []
            for _, v, data in out_edges:
                ts = pd.Timestamp(data.get("ts"))
                if pd.isna(ts):
                    continue
                rows.append((v, float(data.get("amount", 0.0)), ts))
            if len(rows) < self.min_distinct_destinations:
                continue
            rows.sort(key=lambda r: r[2])

            best_window = FanInCollector._best_window(rows, win)
            if best_window is None:
                continue
            window_start, window_end, dests, total_amt, n_txns = best_window

            n_dest = len(dests)
            if n_dest < self.min_distinct_destinations or total_amt < self.min_total_amount:
                continue

            dest_score = min(1.0, n_dest / self.target_destinations)
            amt_score  = min(1.0, total_amt / (5 * self.min_total_amount))
            score = round(0.65 * dest_score + 0.35 * amt_score, 4)

            if score < self.min_score:
                continue

            severity = (
                "ALERT" if score >= 0.70 else
                "WARNING" if score >= 0.45 else
                "INFO"
            )

            dests_list = list(dests)
            edges = [(node, d) for d in dests_list]

            out.append(MotifMatch(
                pattern_name=self.name,
                severity=severity,
                score=score,
                nodes=[node] + dests_list,
                edges=edges,
                n_transactions=n_txns,
                total_amount=total_amt,
                time_window_start=str(window_start),
                time_window_end=str(window_end),
                metadata={
                    "distributor":    node,
                    "n_destinations": n_dest,
                    "highlight":      f"distributor={node}, {n_dest} destinations, Rs {total_amt:,.0f}",
                    "window_hours":   self.window_hours,
                },
            ))
        return out


# -- 3. Layering chain ------------------------------------------------------


class LayeringChain(MotifPattern):
    """Long unidirectional pass-through path A->B->C->D... with short hold times.

    A layering chain is a directed simple path where each intermediate node
    receives funds and forwards a similar amount within ``max_hold_hours``
    to the next node in the chain.

    We enumerate paths up to length ``max_hops`` rooted at every node.
    Scored by chain length, hold-time tightness, and amount preservation.
    """

    name = "layering_chain"
    description = "Long unidirectional pass-through chain"

    def __init__(
        self,
        min_hops: int = 3,
        max_hops: int = 6,
        max_hold_hours: float = 48.0,
        amount_preservation: float = 0.70,  # next-amt >= this x prev-amt
        min_score: float = 0.30,
        max_root_outdegree: int = 5,        # don't enumerate from highly-connected hubs
    ) -> None:
        self.min_hops             = int(min_hops)
        self.max_hops             = int(max_hops)
        self.max_hold_hours       = float(max_hold_hours)
        self.amount_preservation  = float(amount_preservation)
        self.min_score            = float(min_score)
        self.max_root_outdegree   = int(max_root_outdegree)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        out: list[MotifMatch] = []
        seen_chains: set[tuple] = set()

        # Pre-collect outgoing edges per node sorted by ts
        out_idx: dict[str, list[tuple[str, float, pd.Timestamp]]] = {}
        for u, v, data in graph.graph.edges(data=True):
            ts = pd.Timestamp(data.get("ts"))
            if pd.isna(ts):
                continue
            out_idx.setdefault(u, []).append((v, float(data.get("amount", 0.0)), ts))
        for u in out_idx:
            out_idx[u].sort(key=lambda x: x[2])

        max_hold = pd.Timedelta(hours=self.max_hold_hours)

        # DFS chain extension; we treat each (node, prev_amt, prev_ts) state as a frontier
        for root in graph.graph.nodes():
            # Skip very out-connected roots (too many false positives)
            if graph.graph.out_degree(root) == 0:
                continue
            if graph.graph.out_degree(root) > self.max_root_outdegree:
                continue

            for first_dst, first_amt, first_ts in out_idx.get(root, []):
                self._extend_chain(
                    graph=graph,
                    out_idx=out_idx,
                    chain=[root, first_dst],
                    edges=[(root, first_dst)],
                    amounts=[first_amt],
                    timestamps=[first_ts],
                    last_amt=first_amt,
                    last_ts=first_ts,
                    max_hold=max_hold,
                    out=out,
                    seen_chains=seen_chains,
                )
        return out

    def _extend_chain(
        self,
        graph: AccountGraph,
        out_idx: dict,
        chain: list[str],
        edges: list[tuple],
        amounts: list[float],
        timestamps: list[pd.Timestamp],
        last_amt: float,
        last_ts: pd.Timestamp,
        max_hold: pd.Timedelta,
        out: list[MotifMatch],
        seen_chains: set,
    ) -> None:
        # Try to extend; record chain if length sufficient
        current = chain[-1]
        candidates = []
        for nxt, amt, ts in out_idx.get(current, []):
            if nxt in chain:
                continue  # avoid cycles within layering chain
            if ts <= last_ts:
                continue
            if ts - last_ts > max_hold:
                continue
            if amt < self.amount_preservation * last_amt:
                continue
            candidates.append((nxt, amt, ts))

        # Record current chain if long enough (before further extension)
        if len(chain) >= self.min_hops + 1:  # nodes = hops+1
            key = tuple(chain)
            if key not in seen_chains:
                seen_chains.add(key)
                hops = len(chain) - 1
                hold_total = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
                avg_hold = hold_total / max(1, hops - 1) if hops > 1 else 0.0

                # Score factors
                hop_score = min(1.0, hops / self.max_hops)
                hold_score = max(
                    0.0,
                    1.0 - (avg_hold / self.max_hold_hours)
                ) if avg_hold > 0 else 1.0
                preservation = (amounts[-1] / amounts[0]) if amounts[0] > 0 else 0.0
                preservation_score = min(1.0, preservation / 1.0)
                score = round(
                    0.45 * hop_score + 0.30 * hold_score + 0.25 * preservation_score, 4
                )
                if score >= self.min_score:
                    severity = (
                        "ALERT" if score >= 0.70 else
                        "WARNING" if score >= 0.50 else
                        "INFO"
                    )
                    out.append(MotifMatch(
                        pattern_name=self.name,
                        severity=severity,
                        score=score,
                        nodes=list(chain),
                        edges=list(edges),
                        n_transactions=len(edges),
                        total_amount=float(sum(amounts)),
                        time_window_start=str(timestamps[0]),
                        time_window_end=str(timestamps[-1]),
                        metadata={
                            "hops":                hops,
                            "avg_hold_hours":      round(avg_hold, 2),
                            "amount_preservation": round(preservation, 3),
                            "highlight":           f"chain {' -> '.join(chain[:5])}{'...' if len(chain) > 5 else ''} ({hops} hops)",
                        },
                    ))

        # Stop extension if at max length
        if len(chain) - 1 >= self.max_hops:
            return

        # Recurse
        for nxt, amt, ts in candidates:
            self._extend_chain(
                graph, out_idx,
                chain + [nxt],
                edges + [(current, nxt)],
                amounts + [amt],
                timestamps + [ts],
                last_amt=amt,
                last_ts=ts,
                max_hold=max_hold,
                out=out,
                seen_chains=seen_chains,
            )


# -- 4. Cyclic ring ---------------------------------------------------------


class CyclicRing(MotifPattern):
    """Closed cycle A->B->C->A with money round-tripping inside ``max_cycle_hours``.

    Detected via Tarjan's strongly-connected components, then within each
    non-trivial SCC we enumerate elementary cycles up to ``max_cycle_length``.
    """

    name = "cyclic_ring"
    description = "Closed cycle round-tripping money"

    def __init__(
        self,
        min_cycle_length: int = 3,
        max_cycle_length: int = 6,
        max_cycle_hours: float = 72.0,
        min_total_amount: float = 25_000.0,
        min_score: float = 0.30,
    ) -> None:
        self.min_cycle_length = int(min_cycle_length)
        self.max_cycle_length = int(max_cycle_length)
        self.max_cycle_hours  = float(max_cycle_hours)
        self.min_total_amount = float(min_total_amount)
        self.min_score        = float(min_score)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        import networkx as nx
        out: list[MotifMatch] = []
        simple = graph.to_simple()

        # Restrict to SCCs of size >= min_cycle_length
        sccs = [c for c in nx.strongly_connected_components(simple)
                if len(c) >= self.min_cycle_length]

        for scc in sccs:
            sub = simple.subgraph(scc).copy()
            try:
                cycles = list(nx.simple_cycles(sub, length_bound=self.max_cycle_length))
            except TypeError:
                # older networkx -- no length_bound
                cycles = []
                for c in nx.simple_cycles(sub):
                    if self.min_cycle_length <= len(c) <= self.max_cycle_length:
                        cycles.append(c)

            seen_keys: set = set()
            for cyc in cycles:
                if not (self.min_cycle_length <= len(cyc) <= self.max_cycle_length):
                    continue
                # Canonical form for dedup: rotation-invariant tuple
                key = tuple(sorted(cyc)) + (len(cyc),)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                m = self._score_cycle(graph, cyc)
                if m and m.score >= self.min_score:
                    out.append(m)
        return out

    def _score_cycle(self, graph: AccountGraph, cycle: list[str]) -> Optional[MotifMatch]:
        """Try every rotation of the cycle and pick the best time-ordered traversal."""
        n = len(cycle)
        best_match: Optional[MotifMatch] = None
        for start in range(n):
            rotated = cycle[start:] + cycle[:start]
            m = self._traverse_rotation(graph, rotated)
            if m is None:
                continue
            if best_match is None or m.score > best_match.score:
                best_match = m
        return best_match

    def _traverse_rotation(
        self, graph: AccountGraph, cycle: list[str]
    ) -> Optional[MotifMatch]:
        nodes = list(cycle) + [cycle[0]]  # close it
        edges_used: list[tuple] = []
        timestamps: list[pd.Timestamp] = []
        amounts: list[float] = []

        last_ts: Optional[pd.Timestamp] = None
        for u, v in zip(nodes[:-1], nodes[1:]):
            if not graph.graph.has_edge(u, v):
                return None
            best_ts = None
            best_amt = None
            for _, dst, data in graph.graph.out_edges(u, data=True):
                if dst != v:
                    continue
                ts = pd.Timestamp(data.get("ts"))
                if pd.isna(ts):
                    continue
                if last_ts is not None and ts <= last_ts:
                    continue
                if best_ts is None or ts < best_ts:
                    best_ts = ts
                    best_amt = float(data.get("amount", 0.0))
            if best_ts is None:
                return None
            edges_used.append((u, v))
            timestamps.append(best_ts)
            amounts.append(best_amt)
            last_ts = best_ts

        cycle_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
        if cycle_hours > self.max_cycle_hours:
            return None
        total_amt = float(sum(amounts))
        if total_amt < self.min_total_amount:
            return None

        len_score = min(1.0, len(cycle) / self.max_cycle_length)
        time_score = max(0.0, 1.0 - cycle_hours / self.max_cycle_hours)
        amt_score = min(1.0, total_amt / (4 * self.min_total_amount))
        score = round(0.40 * len_score + 0.35 * time_score + 0.25 * amt_score, 4)

        severity = (
            "ALERT" if score >= 0.70 else
            "WARNING" if score >= 0.45 else
            "INFO"
        )
        return MotifMatch(
            pattern_name=self.name,
            severity=severity,
            score=score,
            nodes=list(cycle),
            edges=edges_used,
            n_transactions=len(edges_used),
            total_amount=total_amt,
            time_window_start=str(timestamps[0]),
            time_window_end=str(timestamps[-1]),
            metadata={
                "cycle_length":   len(cycle),
                "cycle_hours":    round(cycle_hours, 2),
                "highlight":      f"cycle {' -> '.join(list(cycle))} -> {cycle[0]}",
            },
        )


# -- 5. Velocity hub -------------------------------------------------------


class VelocityHub(MotifPattern):
    """Account with anomalously bursty transaction pattern.

    Computed via inter-arrival-time distribution: an account whose median
    gap between transactions is ``< max_median_gap_minutes`` AND has at
    least ``min_txn_count`` transactions is flagged.

    Also flags accounts whose burst rate (max txns inside any
    ``burst_window_minutes`` window) exceeds ``min_burst_count``.
    """

    name = "velocity_hub"
    description = "Anomalously bursty account txn pattern"

    def __init__(
        self,
        min_txn_count: int = 20,
        max_median_gap_minutes: float = 10.0,
        burst_window_minutes: float = 60.0,
        min_burst_count: int = 15,
        min_score: float = 0.30,
    ) -> None:
        self.min_txn_count           = int(min_txn_count)
        self.max_median_gap_minutes  = float(max_median_gap_minutes)
        self.burst_window_minutes    = float(burst_window_minutes)
        self.min_burst_count         = int(min_burst_count)
        self.min_score               = float(min_score)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        out: list[MotifMatch] = []
        burst_win = pd.Timedelta(minutes=self.burst_window_minutes)

        for node in graph.graph.nodes():
            # Combine in + out edges, keyed by ts
            ts_amt: list[tuple[pd.Timestamp, float, str]] = []
            for u, _, data in graph.graph.in_edges(node, data=True):
                t = pd.Timestamp(data.get("ts"))
                if not pd.isna(t):
                    ts_amt.append((t, float(data.get("amount", 0.0)), "in"))
            for _, v, data in graph.graph.out_edges(node, data=True):
                t = pd.Timestamp(data.get("ts"))
                if not pd.isna(t):
                    ts_amt.append((t, float(data.get("amount", 0.0)), "out"))

            if len(ts_amt) < self.min_txn_count:
                continue

            ts_amt.sort(key=lambda x: x[0])
            ts_series = pd.Series([t[0] for t in ts_amt])
            gaps = ts_series.diff().dt.total_seconds() / 60.0
            gaps = gaps.dropna()
            if gaps.empty:
                continue
            median_gap = float(gaps.median())

            df = pd.DataFrame(ts_amt, columns=["ts", "amt", "dir"]).set_index("ts")
            burst = (
                df["amt"].rolling(burst_win, closed="both").count().max()
            )
            max_burst = int(burst) if not pd.isna(burst) else 0

            # Score factors
            gap_score = max(0.0, 1.0 - median_gap / self.max_median_gap_minutes) \
                if median_gap < self.max_median_gap_minutes * 2 else 0.0
            burst_score = min(1.0, max_burst / max(1, self.min_burst_count))

            score = round(0.50 * gap_score + 0.50 * burst_score, 4)
            if score < self.min_score:
                continue
            if median_gap >= self.max_median_gap_minutes and max_burst < self.min_burst_count:
                continue

            severity = (
                "ALERT" if score >= 0.70 else
                "WARNING" if score >= 0.45 else
                "INFO"
            )
            total_amt = float(df["amt"].sum())
            out.append(MotifMatch(
                pattern_name=self.name,
                severity=severity,
                score=score,
                nodes=[node],
                edges=[],
                n_transactions=len(ts_amt),
                total_amount=total_amt,
                time_window_start=str(ts_series.iloc[0]),
                time_window_end=str(ts_series.iloc[-1]),
                metadata={
                    "median_gap_minutes": round(median_gap, 2),
                    "max_burst":          max_burst,
                    "burst_window_min":   self.burst_window_minutes,
                    "highlight":          f"node={node}, median_gap={median_gap:.1f}min, burst={max_burst}",
                },
            ))
        return out


# -- 6. Pass-through (proper mule) ------------------------------------------


class PassThrough(MotifPattern):
    """Account whose money_in ~ money_out within ``max_hold_hours``.

    This is the canonical mule signature: funds arrive, get forwarded
    quickly, with little net residue.
    """

    name = "pass_through"
    description = "money_in ~ money_out within hold window"

    def __init__(
        self,
        min_in_amount: float = 25_000.0,
        max_imbalance: float = 0.10,        # |in-out|/max(in,out) <= 0.10
        max_hold_hours: float = 48.0,
        min_in_degree: int = 2,
        min_score: float = 0.30,
    ) -> None:
        self.min_in_amount   = float(min_in_amount)
        self.max_imbalance   = float(max_imbalance)
        self.max_hold_hours  = float(max_hold_hours)
        self.min_in_degree   = int(min_in_degree)
        self.min_score       = float(min_score)

    def detect(self, graph: AccountGraph) -> list[MotifMatch]:
        out: list[MotifMatch] = []
        for node in graph.graph.nodes():
            in_edges  = list(graph.graph.in_edges(node, data=True))
            out_edges = list(graph.graph.out_edges(node, data=True))
            if len(in_edges) < self.min_in_degree or not out_edges:
                continue

            in_amt  = sum(float(d.get("amount", 0.0)) for _, _, d in in_edges)
            out_amt = sum(float(d.get("amount", 0.0)) for _, _, d in out_edges)
            if in_amt < self.min_in_amount:
                continue

            imb = abs(in_amt - out_amt) / max(in_amt, out_amt)
            if imb > self.max_imbalance:
                continue

            in_ts  = [pd.Timestamp(d.get("ts")) for _, _, d in in_edges if not pd.isna(d.get("ts"))]
            out_ts = [pd.Timestamp(d.get("ts")) for _, _, d in out_edges if not pd.isna(d.get("ts"))]
            if not in_ts or not out_ts:
                continue

            avg_in_ts  = pd.Timestamp(pd.Series(in_ts).mean())
            avg_out_ts = pd.Timestamp(pd.Series(out_ts).mean())
            hold = (avg_out_ts - avg_in_ts).total_seconds() / 3600.0
            # money should leave AFTER it arrives
            if hold < 0 or hold > self.max_hold_hours:
                continue

            balance_score = max(0.0, 1.0 - imb / self.max_imbalance)
            hold_score    = max(0.0, 1.0 - hold / self.max_hold_hours)
            amt_score     = min(1.0, in_amt / (4 * self.min_in_amount))
            score = round(
                0.40 * balance_score + 0.35 * hold_score + 0.25 * amt_score, 4
            )
            if score < self.min_score:
                continue

            severity = (
                "ALERT" if score >= 0.70 else
                "WARNING" if score >= 0.45 else
                "INFO"
            )
            sources      = [u for u, _, _ in in_edges]
            destinations = [v for _, v, _ in out_edges]

            out.append(MotifMatch(
                pattern_name=self.name,
                severity=severity,
                score=score,
                nodes=[node] + list(set(sources)) + list(set(destinations)),
                edges=[(u, node) for u in set(sources)] +
                      [(node, v) for v in set(destinations)],
                n_transactions=len(in_edges) + len(out_edges),
                total_amount=in_amt + out_amt,
                time_window_start=str(min(in_ts + out_ts)),
                time_window_end=str(max(in_ts + out_ts)),
                metadata={
                    "passthrough":     node,
                    "in_amount":       round(in_amt, 2),
                    "out_amount":      round(out_amt, 2),
                    "imbalance":       round(imb, 4),
                    "hold_hours":      round(hold, 2),
                    "n_sources":       len(set(sources)),
                    "n_destinations":  len(set(destinations)),
                    "highlight":       (
                        f"node={node}, Rs in={in_amt:,.0f}, Rs out={out_amt:,.0f}, "
                        f"imb={imb:.1%}, hold={hold:.1f}h"
                    ),
                },
            ))
        return out


# -- Coordinator -----------------------------------------------------------


class MotifDetector:
    """
    Run all configured motif patterns on an AccountGraph.

    Defaults include all 6 patterns from the canonical AML catalogue.
    """

    DEFAULT_PATTERNS = (
        FanInCollector,
        FanOutDistributor,
        LayeringChain,
        CyclicRing,
        VelocityHub,
        PassThrough,
    )

    def __init__(
        self,
        patterns: Optional[Iterable[MotifPattern]] = None,
    ) -> None:
        if patterns is None:
            self.patterns = [P() for P in self.DEFAULT_PATTERNS]
        else:
            self.patterns = list(patterns)

    def detect(self, graph: AccountGraph) -> MotifDetectionResult:
        t0 = time.perf_counter()
        all_matches: list[MotifMatch] = []
        per_pattern: dict[str, int] = {}

        for pat in self.patterns:
            try:
                matches = pat.detect(graph)
                per_pattern[pat.name] = len(matches)
                all_matches.extend(matches)
                log.info("Motif %s: %d matches", pat.name, len(matches))
            except Exception as exc:
                log.warning("Motif %s failed: %s", pat.name, exc)
                per_pattern[pat.name] = 0

        n_alerts   = sum(1 for m in all_matches if m.severity == "ALERT")
        n_warnings = sum(1 for m in all_matches if m.severity == "WARNING")
        n_info     = sum(1 for m in all_matches if m.severity == "INFO")

        elapsed = (time.perf_counter() - t0) * 1000
        return MotifDetectionResult(
            period=graph.period,
            generated_at=_now(),
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            n_transactions=graph.n_transactions,
            total_amount=graph.total_amount,
            matches=all_matches,
            matches_by_pattern=per_pattern,
            n_alerts=n_alerts,
            n_warnings=n_warnings,
            n_info=n_info,
            elapsed_ms=round(elapsed, 1),
        )
