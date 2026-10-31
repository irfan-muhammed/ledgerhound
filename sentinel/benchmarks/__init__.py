"""
sentinel.benchmarks
===================

Public benchmark adapters for Sentinel.

The mule-detection literature uses a small set of standard datasets.
Sentinel ships adapters that map each into the canonical raw_txn schema
(``account_id, beneficiary_id, txn_amount, txn_date, status, ...``) so the
existing graph + Phase 2 toolset can be evaluated against published
baselines.

Currently shipping
------------------
- **PaySim** (Lopez-Rojas et al., 2016)
  Synthetic mobile-money txns with binary fraud labels.
  ``load_paysim()`` reads the canonical Kaggle CSV.
  ``generate_paysim_like()`` produces a self-contained synthetic
  PaySim-style dataset with explicit mule motifs embedded -- useful for
  CI tests and reproducibility when the real CSV isn't available.

Roadmap
-------
- IBM AMLSim (per-typology mule labels)
- Elliptic Bitcoin (graph-only, anonymised)
- the organization public benchmarks (when released)

All adapters share a common ``BenchmarkDataset`` shape:
  - ``txn_df``   : raw_txn-compatible DataFrame
  - ``labels``   : per-account ground-truth labels
                   (account_id, is_mule, mule_type)
  - ``metadata`` : provenance dict
"""

from sentinel.benchmarks.paysim import (
    BenchmarkDataset,
    load_paysim,
    generate_paysim_like,
    paysim_to_sentinel_schema,
)

__all__ = [
    "BenchmarkDataset",
    "load_paysim",
    "generate_paysim_like",
    "paysim_to_sentinel_schema",
]
