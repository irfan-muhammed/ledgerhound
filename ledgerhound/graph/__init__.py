"""
ledgerhound.graph
==============

CPU-only graph layer for mule and money-flow EDA on UPI / payment data.

Design rules
------------
- **No GPU.** Classical graph primitives only: motifs, centrality,
  community detection, path tracing.  We do not train GNNs.
- **Payments-feasible.** Operates on transaction-level data -- payer, payee,
  amount, timestamp, status, mode.  No demographics required.
- **Sample-frame aware.** Motif counts and centrality scores carry
  sample-frame tags so the agent can reason about whether a finding
  generalises.
- **Tier-routable.** LOCAL up to ~10^5 nodes / 10^6 edges in 2-4 GB.
  Larger graphs delegate to TRINO recursive CTEs.

Public surface
--------------
- ``AccountGraphBuilder``    build a directed multigraph from a txn DataFrame
- ``AccountGraph``           thin wrapper around a NetworkX MultiDiGraph
- ``MotifDetector``          catalogue of 6 mule motif patterns
- ``CentralityComputer``     PageRank, betweenness, k-core, eigenvector
- ``CommunityFinder``        Louvain ring detection with mule scoring
- ``MoneyFlowTracer``        BFS forensic tracing + k-shortest paths
- ``BehavioralFingerprinter`` per-account behavioral mule fingerprints
"""

from ledgerhound.graph.result import (
    MotifMatch,
    MotifDetectionResult,
    CentralityResult,
    AccountCentrality,
)
from ledgerhound.graph.builder import AccountGraph, AccountGraphBuilder
from ledgerhound.graph.motifs import (
    MotifDetector,
    MotifPattern,
    FanInCollector,
    FanOutDistributor,
    LayeringChain,
    CyclicRing,
    VelocityHub,
    PassThrough,
)
from ledgerhound.graph.centrality import CentralityComputer
from ledgerhound.graph.community import (
    CommunityFinder,
    Community,
    CommunityResult,
)
from ledgerhound.graph.flow import (
    MoneyFlowTracer,
    FlowPath,
    FlowResult,
)
from ledgerhound.graph.fingerprint import (
    BehavioralFingerprinter,
    AccountFingerprint,
    FingerprintResult,
)

__all__ = [
    # Result types
    "MotifMatch", "MotifDetectionResult",
    "CentralityResult", "AccountCentrality",
    "Community", "CommunityResult",
    "FlowPath", "FlowResult",
    "AccountFingerprint", "FingerprintResult",
    # Builder
    "AccountGraph", "AccountGraphBuilder",
    # Motifs
    "MotifDetector", "MotifPattern",
    "FanInCollector", "FanOutDistributor", "LayeringChain",
    "CyclicRing", "VelocityHub", "PassThrough",
    # Centrality
    "CentralityComputer",
    # Community
    "CommunityFinder",
    # Flow
    "MoneyFlowTracer",
    # Fingerprint
    "BehavioralFingerprinter",
]
