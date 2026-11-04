"""
ledgerhound.compute.tier
=====================

TierClassifier -- routes every operation to the right compute tier
before the LLM or agent attempts it.

Two tiers:
  LOCAL   H100 GPU with cuDF (or CPU pandas as fallback).
          Use for: confirmed fraud sample, summary table after Trino pull,
          any dataset that fits in memory (<= local_max_rows).

  TRINO   KubeFlow TrinoSQL cluster. Async, hours of wait.
          Use for: population profiling, non-fraud sample pulls, any
          dataset too large to fit locally.

The classifier is called before every analysis step. Its decision is
logged into the agent trace so the audit trail shows WHY a query went
where it did.

Design principles
-----------------
- When use_trino=False, everything routes LOCAL regardless of size.
- The classifier is conservative: if uncertain, it routes TRINO to avoid
  silent OOM or timeout on the H100.
- The agent reads TierDecision.reason to explain routing to the user.
- Mule-track and fraud-track datasets are treated identically. No bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ComputeTier(str, Enum):
    LOCAL = "local"   # H100/cuDF or pandas
    TRINO = "trino"   # KubeFlow TrinoSQL


@dataclass
class TierDecision:
    tier: ComputeTier
    reason: str
    estimated_rows: Optional[int] = None
    local_feasible: bool = True         # False means LOCAL is risky/impossible
    warning: Optional[str] = None       # displayed to user if set

    def is_local(self) -> bool:
        return self.tier == ComputeTier.LOCAL

    def is_trino(self) -> bool:
        return self.tier == ComputeTier.TRINO


# -- Known dataset classifications ------------------------------------------
#
# Named datasets whose tier is known in advance. Used to short-circuit
# the row-count check when the agent already knows the dataset identity.
#
# Key: canonical dataset name (lowercase, underscores)
# Value: (tier, rationale)

_KNOWN_DATASETS: dict[str, tuple[ComputeTier, str]] = {
    # Confirmed fraud / mule labels -- always small, always local
    "fraud_labels":              (ComputeTier.LOCAL,  "Confirmed fraud labels: thousands to lakhs of rows, always fits on H100"),
    "confirmed_fraud":           (ComputeTier.LOCAL,  "Confirmed fraud cases: always local"),
    "confirmed_mule":            (ComputeTier.LOCAL,  "Confirmed mule accounts: always local"),
    "historical_fraud":          (ComputeTier.LOCAL,  "Historical confirmed fraud: always local"),
    # UPI summary table -- medium, but population is Trino; local after pull
    "upi_summary_sample":        (ComputeTier.LOCAL,  "UPI summary sample: pulled via Trino, now local"),
    "upi_summary_fraud":         (ComputeTier.LOCAL,  "Fraud-filtered summary: small after label join, local"),
    # Raw transactions -- always Trino
    "upi_raw_transactions":      (ComputeTier.TRINO,  "Raw UPI transactions: billions of rows, KubeFlow Trino only"),
    "upi_transactions":          (ComputeTier.TRINO,  "Raw UPI transactions: billions of rows, KubeFlow Trino only"),
    # Full summary population -- Trino for population profiling
    "upi_summary":               (ComputeTier.TRINO,  "Full UPI summary population: hundreds of millions of rows, use Trino for profiling"),
    "upi_summary_population":    (ComputeTier.TRINO,  "Full UPI summary population: Trino only"),
    # Non-fraud samples -- local after Trino pull
    "nonfr_sample":              (ComputeTier.LOCAL,  "Non-fraud sample: pulled via Trino, now local"),
    "nonfr_hard_negatives":      (ComputeTier.LOCAL,  "Hard negative sample: pulled via Trino, now local"),
}

# Operations that are always Trino regardless of dataset
_TRINO_OPERATIONS = frozenset({
    "population_profile",
    "population_count",
    "full_table_scan",
    "non_fraud_sample_pull",
    "hard_negative_pull",
    "temporal_population_scan",
    "cross_year_scan",
})

# Operations that are always local regardless of dataset
_LOCAL_OPERATIONS = frozenset({
    "fraud_distribution",
    "fraud_clustering",
    "fraud_typology",
    "hard_negative_boundary",
    "sample_certification",
    "signal_decay_local",
    "mule_distribution",
    "chart_generation",
    "finding_record",
})


class TierClassifier:
    """
    Routes datasets and operations to LOCAL or TRINO.

    Parameters
    ----------
    use_trino       : Master toggle. When False, everything routes LOCAL.
    local_max_rows  : Row count above which LOCAL is considered infeasible.
                      Default 50M -- conservative for H100 with cuDF.
    """

    def __init__(
        self,
        use_trino: bool = True,
        local_max_rows: int = 50_000_000,
    ) -> None:
        self.use_trino = use_trino
        self.local_max_rows = local_max_rows

    def classify_dataset(
        self,
        dataset_name: str,
        n_rows: Optional[int] = None,
    ) -> TierDecision:
        """
        Determine compute tier for a named dataset.

        Checks known dataset registry first, then falls back to row-count
        heuristic. When use_trino=False, always returns LOCAL.
        """
        if not self.use_trino:
            return TierDecision(
                tier=ComputeTier.LOCAL,
                reason="use_trino=False -- all operations routed to local compute",
                estimated_rows=n_rows,
                local_feasible=True,
            )

        # Known dataset lookup (case-insensitive, strip common suffixes)
        key = _normalise_name(dataset_name)
        if key in _KNOWN_DATASETS:
            tier, rationale = _KNOWN_DATASETS[key]
            return TierDecision(
                tier=tier,
                reason=rationale,
                estimated_rows=n_rows,
                local_feasible=(tier == ComputeTier.LOCAL),
            )

        # Row-count heuristic
        if n_rows is not None:
            if n_rows > self.local_max_rows:
                return TierDecision(
                    tier=ComputeTier.TRINO,
                    reason=f"Dataset has {n_rows:,} rows (>{self.local_max_rows:,} local threshold) -- route to Trino",
                    estimated_rows=n_rows,
                    local_feasible=False,
                    warning=f"Large dataset: {n_rows:,} rows. Trino query will be async (hours).",
                )
            return TierDecision(
                tier=ComputeTier.LOCAL,
                reason=f"Dataset has {n_rows:,} rows (<= local threshold) -- local compute feasible",
                estimated_rows=n_rows,
                local_feasible=True,
            )

        # Unknown name and no row count -- conservative: route to Trino
        return TierDecision(
            tier=ComputeTier.TRINO,
            reason=f"Unknown dataset '{dataset_name}' with no row count -- routing conservatively to Trino",
            estimated_rows=None,
            local_feasible=False,
            warning=f"Dataset '{dataset_name}' is not in the known dataset registry. Verify before submitting.",
        )

    def classify_operation(
        self,
        operation: str,
        dataset_name: Optional[str] = None,
        n_rows: Optional[int] = None,
    ) -> TierDecision:
        """
        Determine compute tier for a named operation.

        Operation names are matched against known LOCAL/TRINO sets first.
        Falls back to dataset classification if the operation is unknown.
        """
        if not self.use_trino:
            return TierDecision(
                tier=ComputeTier.LOCAL,
                reason="use_trino=False -- all operations routed to local compute",
                local_feasible=True,
            )

        op_key = operation.lower().strip()

        if op_key in _TRINO_OPERATIONS:
            return TierDecision(
                tier=ComputeTier.TRINO,
                reason=f"Operation '{operation}' requires full population access -- Trino only",
                local_feasible=False,
            )

        if op_key in _LOCAL_OPERATIONS:
            return TierDecision(
                tier=ComputeTier.LOCAL,
                reason=f"Operation '{operation}' works on local confirmed fraud/sample data",
                local_feasible=True,
            )

        # Fall back to dataset classification
        if dataset_name:
            return self.classify_dataset(dataset_name, n_rows)

        # Unknown operation, no dataset context
        return TierDecision(
            tier=ComputeTier.LOCAL,
            reason=f"Operation '{operation}' unknown -- defaulting to local (safer than accidental Trino pull)",
            local_feasible=True,
            warning=f"Operation '{operation}' not in tier registry. Verify it doesn't need Trino.",
        )

    def explain(self, dataset_name: str, n_rows: Optional[int] = None) -> str:
        """One-line human-readable routing explanation."""
        d = self.classify_dataset(dataset_name, n_rows)
        suffix = f" ! {d.warning}" if d.warning else ""
        return f"[{d.tier.value.upper()}] {d.reason}{suffix}"


# -- Helpers ----------------------------------------------------------------


def _normalise_name(name: str) -> str:
    """Lowercase, replace hyphens with underscores, strip version suffixes."""
    import re
    n = name.lower().replace("-", "_").strip()
    # Strip trailing _v1, _v2, _2024, _sample_v3, etc.
    n = re.sub(r"_v\d+$", "", n)
    n = re.sub(r"_\d{4}(q\d)?$", "", n)
    return n
