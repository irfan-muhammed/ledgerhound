"""
ledgerhound.benchmarks.paysim
==========================

PaySim adapter and self-contained synthetic-PaySim generator.

PaySim schema (Lopez-Rojas et al. 2016, Kaggle release)
-------------------------------------------------------
::

    step              hour offset from epoch (int)
    type              PAYMENT | TRANSFER | CASH_OUT | DEBIT | CASH_IN
    amount            float
    nameOrig          payer account
    oldbalanceOrg     payer balance before
    newbalanceOrig    payer balance after
    nameDest          payee account
    oldbalanceDest    payee balance before
    newbalanceDest    payee balance after
    isFraud           1 if labelled fraud transaction
    isFlaggedFraud    1 if flagged by simple rule (>200K transfer)

Ledgerhound mapping
----------------
We adapt PaySim -> ledgerhound.raw_txn:
  account_id      <- nameOrig
  beneficiary_id  <- nameDest
  txn_amount      <- amount
  txn_date        <- base_date + step hours
  txn_type        <- type
  status          <- 'SUCCESS'  (PaySim has no failure column)
  device_id       <- None
  bank_code       <- None

The label table is per-account: any account that appears as nameOrig or
nameDest of an isFraud=1 transaction is marked is_mule=1.  We additionally
classify the mule_type by inspecting the local motif structure.

Synthetic generator
-------------------
``generate_paysim_like(n_legit=200, embed_motifs=True, ...)`` produces
a PaySim-schema DataFrame with explicit motifs embedded for every
mule pattern in Ledgerhound's catalogue.  Use this in CI and when you
don't have the real Kaggle file handy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -- Container --------------------------------------------------------------


@dataclass
class BenchmarkDataset:
    """Loaded benchmark in Ledgerhound's canonical raw_txn schema."""

    name: str
    txn_df: pd.DataFrame
    labels: pd.DataFrame             # cols: account_id, is_mule, mule_type
    metadata: dict = field(default_factory=dict)

    def info(self) -> str:
        n_acc = self.labels["account_id"].nunique()
        n_mule = int(self.labels["is_mule"].sum())
        return (
            f"BenchmarkDataset(name={self.name!r}, "
            f"txns={len(self.txn_df):,}, accounts={n_acc:,}, "
            f"mules={n_mule:,} ({n_mule / max(1, n_acc):.1%}))"
        )

    __repr__ = info


# -- PaySim -> Ledgerhound schema converter -------------------------------------


def paysim_to_ledgerhound_schema(
    df: pd.DataFrame,
    base_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Convert a PaySim-format DataFrame to Ledgerhound's raw_txn schema.

    Parameters
    ----------
    df : pd.DataFrame
        PaySim-format DataFrame with at least columns step, type, amount,
        nameOrig, nameDest.
    base_date : str
        Anchor date for the synthetic timestamps.  ``txn_date = base_date
        + step hours`` for each row.
    """
    base = pd.Timestamp(base_date)
    out = pd.DataFrame({
        "txn_id":         [f"PAYSIM_{i:09d}" for i in range(len(df))],
        "account_id":     df["nameOrig"].astype(str),
        "beneficiary_id": df["nameDest"].astype(str),
        "txn_amount":     df["amount"].astype(float),
        "txn_date":       (base + pd.to_timedelta(df["step"].astype(int), unit="h"))
                          .dt.strftime("%Y-%m-%d %H:%M:%S"),
        "txn_type":       df["type"].astype(str),
        "status":         "SUCCESS",
    })
    return out


# -- Real PaySim CSV loader -------------------------------------------------


def load_paysim(
    csv_path: str | Path,
    base_date: str = "2024-01-01",
    n_rows: Optional[int] = None,
) -> BenchmarkDataset:
    """
    Load the canonical PaySim CSV from Kaggle into a BenchmarkDataset.

    Parameters
    ----------
    csv_path : str | Path
        Path to ``PS_20174392719_1491204439457_log.csv`` or similar.
    base_date : str
        Anchor date for synthetic timestamps.
    n_rows : int, optional
        Read only the first N rows (useful for smoke tests).
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found: {p}. "
            f"Download from Kaggle (search 'PaySim') or use "
            f"generate_paysim_like() for a synthetic stand-in."
        )

    log.info("Loading PaySim from %s", p)
    df = pd.read_csv(p, nrows=n_rows)

    txn_df = paysim_to_ledgerhound_schema(df, base_date=base_date)
    # Carry the canonical per-transaction fraud label through. This is
    # PaySim's ground truth (8,213 txns / 0.129% in the full file) and the
    # only label comparable to published PaySim baselines. Account-level
    # mule labels (below) are a DERIVED view, not a replacement.
    if "isFraud" in df.columns:
        txn_df["is_fraud"] = df["isFraud"].astype(int).values

    # Per-account labels: account is mule if it appears in any isFraud row
    fraud_rows = df[df.get("isFraud", 0) == 1]
    mule_accounts = set(fraud_rows["nameOrig"].astype(str)) | \
                    set(fraud_rows["nameDest"].astype(str))
    all_accounts = pd.unique(
        pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
    )
    labels = pd.DataFrame({
        "account_id": all_accounts.astype(str),
        "is_mule":    [int(a in mule_accounts) for a in all_accounts.astype(str)],
        "mule_type":  ["unknown" if a in mule_accounts else "" for a in all_accounts.astype(str)],
    })

    return BenchmarkDataset(
        name="paysim",
        txn_df=txn_df,
        labels=labels,
        metadata={
            "source":     str(p),
            "base_date":  base_date,
            "n_rows":     int(len(df)),
            "n_fraud":    int(len(fraud_rows)),
            "n_mules":    int(len(mule_accounts)),
        },
    )


# -- Synthetic PaySim-like generator with embedded motifs -------------------


def generate_paysim_like(
    n_legit_accounts:    int = 200,
    n_legit_txns:        int = 1500,
    n_fan_in:            int = 5,    # collectors
    fan_in_sources:      int = 8,
    n_fan_out:           int = 5,    # distributors
    fan_out_dests:       int = 8,
    n_layering_chains:   int = 3,
    chain_length:        int = 5,
    n_cyclic_rings:      int = 2,
    ring_length:         int = 4,
    n_passthrough:       int = 5,
    passthrough_sources: int = 5,
    passthrough_dests:   int = 4,
    base_date:           str = "2024-01-01",
    random_state:        int = 42,
) -> BenchmarkDataset:
    """
    Generate a self-contained PaySim-style benchmark with embedded mule motifs.

    Every motif from Ledgerhound's catalogue is represented, so a perfect
    motif detector should recover them all.  The legit-account population
    is sized so motif accounts are a clear minority -- you can compute
    realistic precision/recall.

    Returns
    -------
    BenchmarkDataset
        ``txn_df`` is in Ledgerhound's raw_txn schema (already adapted).
        ``labels`` carries per-account ``is_mule`` (1) and ``mule_type``
        (one of: 'fan_in_collector', 'fan_in_source', 'fan_out_distributor',
        'fan_out_dest', 'layering_chain', 'cyclic_ring', 'passthrough').
    """
    rng = np.random.RandomState(random_state)
    base = pd.Timestamp(base_date)

    rows: list[dict] = []
    label_rows: list[dict] = []
    txn_id_counter = 0
    next_step = [0]   # hours from base; mutable closure

    def _emit(payer: str, payee: str, amount: float, step: int,
              ttype: str = "TRANSFER"):
        nonlocal txn_id_counter
        rows.append({
            "txn_id":         f"PAYSIM_{txn_id_counter:09d}",
            "account_id":     payer,
            "beneficiary_id": payee,
            "txn_amount":     round(float(amount), 2),
            "txn_date":       (base + pd.Timedelta(hours=int(step)))
                              .strftime("%Y-%m-%d %H:%M:%S"),
            "txn_type":       ttype,
            "status":         "SUCCESS",
        })
        txn_id_counter += 1

    def _label(account: str, mule_type: str = "", is_mule: int = 1):
        label_rows.append({
            "account_id": account,
            "is_mule":    int(is_mule),
            "mule_type":  mule_type,
        })

    # -- 1. Legit population ---------------------------------------------
    legit_accounts = [f"LEGIT_{i:05d}" for i in range(n_legit_accounts)]
    for acc in legit_accounts:
        _label(acc, mule_type="", is_mule=0)

    for _ in range(n_legit_txns):
        u, v = rng.choice(legit_accounts, size=2, replace=False)
        amt = float(rng.uniform(50, 5_000))
        step = int(rng.randint(0, 24 * 90))   # over 90 days
        _emit(u, v, amt, step,
              ttype=str(rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"])))

    # -- 2. Fan-in collectors --------------------------------------------
    base_step = 24 * 30   # collectors operate in month 2
    for c_idx in range(n_fan_in):
        collector = f"MULE_FANIN_C{c_idx:03d}"
        _label(collector, mule_type="fan_in_collector")
        sources = [f"MULE_FANIN_C{c_idx:03d}_S{s:02d}" for s in range(fan_in_sources)]
        c_start = base_step + c_idx * 48
        for j, src in enumerate(sources):
            _label(src, mule_type="fan_in_source")
            _emit(
                src, collector,
                amount=float(rng.uniform(8_000, 18_000)),
                step=c_start + j,                                 # within ~8h
            )

    # -- 3. Fan-out distributors -----------------------------------------
    base_step = 24 * 31
    for d_idx in range(n_fan_out):
        distributor = f"MULE_FANOUT_D{d_idx:03d}"
        _label(distributor, mule_type="fan_out_distributor")
        dests = [f"MULE_FANOUT_D{d_idx:03d}_T{t:02d}" for t in range(fan_out_dests)]
        d_start = base_step + d_idx * 48
        for j, dst in enumerate(dests):
            _label(dst, mule_type="fan_out_dest")
            _emit(
                distributor, dst,
                amount=float(rng.uniform(7_000, 17_000)),
                step=d_start + j,
            )

    # -- 4. Layering chains ----------------------------------------------
    base_step = 24 * 33
    for chain_idx in range(n_layering_chains):
        chain = [f"MULE_CHAIN_C{chain_idx:03d}_N{n:02d}" for n in range(chain_length)]
        for acc in chain:
            _label(acc, mule_type="layering_chain")
        amount = float(rng.uniform(80_000, 150_000))
        step = base_step + chain_idx * 24
        for u, v in zip(chain[:-1], chain[1:]):
            _emit(u, v, amount * float(rng.uniform(0.92, 0.99)), step)
            step += int(rng.uniform(2, 8))    # 2-8h hold time

    # -- 5. Cyclic rings -------------------------------------------------
    base_step = 24 * 35
    for ring_idx in range(n_cyclic_rings):
        ring = [f"MULE_RING_R{ring_idx:03d}_N{n:02d}" for n in range(ring_length)]
        for acc in ring:
            _label(acc, mule_type="cyclic_ring")
        step = base_step + ring_idx * 24
        for i in range(ring_length):
            u = ring[i]
            v = ring[(i + 1) % ring_length]
            _emit(u, v, float(rng.uniform(30_000, 50_000)), step)
            step += int(rng.uniform(4, 12))

    # -- 6. Pass-through mules -------------------------------------------
    base_step = 24 * 38
    for p_idx in range(n_passthrough):
        passthrough = f"MULE_PT_P{p_idx:03d}"
        _label(passthrough, mule_type="passthrough")
        sources = [f"MULE_PT_P{p_idx:03d}_S{s:02d}" for s in range(passthrough_sources)]
        dests   = [f"MULE_PT_P{p_idx:03d}_T{t:02d}" for t in range(passthrough_dests)]
        in_amount = 0.0
        in_step = base_step + p_idx * 48
        for j, src in enumerate(sources):
            _label(src, mule_type="passthrough_source")
            amt = float(rng.uniform(15_000, 25_000))
            in_amount += amt
            _emit(src, passthrough, amt, in_step + j)
        # Forward ~ same total in 4 chunks within hold window
        out_step = in_step + 6   # 6h hold
        per_dest = in_amount / passthrough_dests
        for j, dst in enumerate(dests):
            _label(dst, mule_type="passthrough_dest")
            _emit(passthrough, dst,
                  amount=per_dest * float(rng.uniform(0.97, 1.03)),
                  step=out_step + j)

    # -- Assemble --------------------------------------------------------
    txn_df = pd.DataFrame(rows)
    # De-dup label rows (an account may be labeled multiple times during
    # generation; keep the most-specific = first encountered mule tag)
    labels_df = (
        pd.DataFrame(label_rows)
        .drop_duplicates(subset=["account_id"], keep="first")
        .reset_index(drop=True)
    )

    n_mules = int(labels_df["is_mule"].sum())
    n_acc = labels_df["account_id"].nunique()

    return BenchmarkDataset(
        name="paysim_like",
        txn_df=txn_df,
        labels=labels_df,
        metadata={
            "source":             "synthetic",
            "random_state":       int(random_state),
            "base_date":          base_date,
            "n_legit_accounts":   int(n_legit_accounts),
            "n_legit_txns":       int(n_legit_txns),
            "n_fan_in":           int(n_fan_in),
            "n_fan_out":          int(n_fan_out),
            "n_layering_chains":  int(n_layering_chains),
            "n_cyclic_rings":     int(n_cyclic_rings),
            "n_passthrough":      int(n_passthrough),
            "n_accounts":         int(n_acc),
            "n_mules":            int(n_mules),
            "mule_rate":          round(n_mules / n_acc, 4) if n_acc else 0.0,
        },
    )
