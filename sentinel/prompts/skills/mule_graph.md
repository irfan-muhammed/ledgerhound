---
name: mule_graph
when_to_use: detecting mule accounts, money-laundering rings, or graph/network patterns in transactions
tags: [graph, mule, rings, centrality, motifs, network, laundering]
---
# Mule / transaction-graph analysis (institutional method)

1. **Build the account graph** from transactions (payer -> payee edges,
   amount/time as attributes). Keep it on the compute tier that fits the data
   size -- check_compute() before materialising a large graph.
2. **Detect motifs**: fan-in / fan-out, pass-through (rapid in-then-out),
   cycles. These are the mule fingerprints.
3. **Score centrality** to surface hub accounts; a mule ring shows as a dense
   community (Louvain) with high internal flow.
4. **Trace money flow** along suspected chains to confirm pass-through timing.
5. **Fingerprint accounts** behaviourally so a typology can be recalled/compared
   across datasets and over time.
6. **Frame discipline still applies**: a motif rate measured on a filtered
   subgraph is a within-subgraph estimate, not a population claim.

Deliverable: ranked suspect accounts/rings with evidence (motif, centrality,
flow trace), each frame-tagged.
