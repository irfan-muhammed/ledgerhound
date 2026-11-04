"""
ledgerhound.data.synthetic
========================

UpiSyntheticGenerator -- creates realistic synthetic UPI fraud datasets
for integration testing and demos. No PII, no real data.

Schema matches the UPI data dictionary conventions used throughout
the Ledgerhound codebase. Statistical distributions are calibrated to
reflect real UPI behavior patterns described in public the organization reports.

Generated datasets
------------------
summary_df      : One row per account. Account-level aggregated features.
                  50,000 accounts (configurable), ~1:100 fraud ratio.
fraud_labels_df : Confirmed fraud accounts with typology tags.
raw_txn_df      : Sample of raw transactions (5 per account, ~250k rows).
                  Subset of accounts only -- not full population.

Fraud typologies (4 distinct clusters with different feature signatures)
------------------------------------------------------------------------
1. SIM_SWAP          High sim_change_count + device_change_count.
                     Sudden spike in new beneficiaries post-SIM change.

2. SOCIAL_ENGINEERING High avg_txn_amount, older trusted accounts,
                      high new_benef_rate, single large outward transfer.

3. ACCOUNT_TAKEOVER  New accounts (low account_age_days), high night
                     activity, device changes, rapid large transfers.

4. FAKE_MERCHANT     High unique_beneficiaries, low p2p_ratio (looks
                     like merchant), moderate transaction amounts.

Hard negatives (200 accounts)
-------------------------------
Legitimate accounts that share features with fraud -- the cases where
models fail. These are in the non-fraud population but sit close to
the fraud cluster boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FRAUD_TYPES = ["sim_swap", "social_engineering", "account_takeover", "fake_merchant"]

BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB", "BOI", "CANARA", "UNION", "IDFC"]
STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "AP", "TG", "KL", "MP"]


@dataclass
class SyntheticDataset:
    summary_df: pd.DataFrame
    fraud_labels_df: pd.DataFrame
    raw_txn_df: pd.DataFrame
    n_accounts: int
    n_fraud: int
    n_hard_negatives: int

    @property
    def fraud_ratio(self) -> float:
        return (self.n_accounts - self.n_fraud) / self.n_fraud

    def save(self, workspace_dir: Path) -> dict[str, Path]:
        """Save all datasets to workspace directory."""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        p = workspace_dir / "upi_summary.parquet"
        self.summary_df.to_parquet(p, index=False)
        paths["summary"] = p

        p = workspace_dir / "fraud_labels.parquet"
        self.fraud_labels_df.to_parquet(p, index=False)
        paths["fraud_labels"] = p

        p = workspace_dir / "raw_txn_sample.parquet"
        self.raw_txn_df.to_parquet(p, index=False)
        paths["raw_txn"] = p

        return paths

    def info(self) -> str:
        return (
            f"SyntheticDataset\n"
            f"  Accounts       : {self.n_accounts:,}\n"
            f"  Fraud          : {self.n_fraud:,} ({self.n_fraud/self.n_accounts:.2%})\n"
            f"  Hard negatives : {self.n_hard_negatives}\n"
            f"  Fraud ratio    : 1:{self.fraud_ratio:.0f}\n"
            f"  Summary cols   : {list(self.summary_df.columns)}\n"
            f"  Raw txn rows   : {len(self.raw_txn_df):,}\n"
        )


class UpiSyntheticGenerator:
    """
    Generates synthetic UPI fraud datasets with realistic distributions.

    Parameters
    ----------
    n_accounts      : Total account count (fraud + non-fraud).
    fraud_rate      : Fraction of accounts that are fraud. Default 0.01 (1:100).
    n_hard_negatives: Legitimate accounts engineered to look like fraud.
    random_state    : Reproducibility seed.
    """

    def __init__(
        self,
        n_accounts: int = 50_000,
        fraud_rate: float = 0.01,
        n_hard_negatives: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_accounts = n_accounts
        self.fraud_rate = fraud_rate
        self.n_hard_negatives = n_hard_negatives
        self.rng = np.random.RandomState(random_state)

    def generate(self) -> SyntheticDataset:
        n_fraud = int(self.n_accounts * self.fraud_rate)
        n_nonfr = self.n_accounts - n_fraud

        fraud_ids = [f"ACC_FR_{i:06d}" for i in range(n_fraud)]
        nonfr_ids = [f"ACC_NF_{i:06d}" for i in range(n_nonfr)]
        all_ids   = fraud_ids + nonfr_ids

        # -- Non-fraud baseline features ------------------------------------
        nonfr_df = self._gen_nonfr_accounts(nonfr_ids)

        # -- Fraud accounts (4 typologies) ----------------------------------
        per_type = n_fraud // len(FRAUD_TYPES)
        extras   = n_fraud - per_type * len(FRAUD_TYPES)
        type_counts = [per_type + (1 if i < extras else 0) for i in range(len(FRAUD_TYPES))]

        fraud_parts = []
        fraud_label_parts = []
        offset = 0
        for ftype, count in zip(FRAUD_TYPES, type_counts):
            ids = fraud_ids[offset:offset + count]
            fdf, ldf = self._gen_fraud_typology(ids, ftype)
            fraud_parts.append(fdf)
            fraud_label_parts.append(ldf)
            offset += count

        fraud_df     = pd.concat(fraud_parts, ignore_index=True)
        labels_df    = pd.concat(fraud_label_parts, ignore_index=True)

        # -- Engineer hard negatives into non-fraud pool --------------------
        hn_indices = self.rng.choice(len(nonfr_df), self.n_hard_negatives, replace=False)
        nonfr_df = self._inject_hard_negatives(nonfr_df, hn_indices)

        # -- Combine and label ----------------------------------------------
        fraud_df["is_fraud"] = 1
        nonfr_df["is_fraud"] = 0
        summary_df = pd.concat([fraud_df, nonfr_df], ignore_index=True)
        summary_df = summary_df.sample(frac=1, random_state=self.rng.randint(9999)).reset_index(drop=True)

        # -- Raw transaction sample -----------------------------------------
        raw_txn_df = self._gen_raw_transactions(summary_df.head(10_000))

        return SyntheticDataset(
            summary_df=summary_df,
            fraud_labels_df=labels_df,
            raw_txn_df=raw_txn_df,
            n_accounts=self.n_accounts,
            n_fraud=n_fraud,
            n_hard_negatives=self.n_hard_negatives,
        )

    # -- Feature generators -------------------------------------------------

    def _base_features(self, ids: list[str]) -> pd.DataFrame:
        """Baseline feature distributions for normal UPI accounts."""
        n = len(ids)
        r = self.rng
        return pd.DataFrame({
            "account_id":          ids,
            "account_age_days":    r.randint(180, 2190, n),          # 6mo-6yr
            "total_txn_count":     r.randint(20, 800, n),
            "total_txn_amount":    r.exponential(50_000, n) + 5_000, # INR
            "avg_txn_amount":      r.exponential(2_500, n) + 500,
            "max_txn_amount":      r.exponential(10_000, n) + 1_000,
            "unique_beneficiaries":r.randint(2, 50, n),
            "new_benef_rate":      r.beta(2, 8, n),                  # ~0.20
            "night_txn_rate":      r.beta(1, 9, n),                  # ~0.10
            "weekend_txn_rate":    r.beta(3, 7, n),                  # ~0.30
            "p2p_ratio":           r.beta(6, 4, n),                  # ~0.60
            "txn_velocity_7d":     r.randint(0, 15, n),
            "avg_txn_gap_hours":   r.exponential(12, n) + 1,
            "device_change_count": r.poisson(0.3, n),
            "sim_change_count":    r.poisson(0.05, n),
            "failed_txn_rate":     r.beta(1, 19, n),                 # ~0.05
            "bank_code":           r.choice(BANKS, n),
            "state_code":          r.choice(STATES, n),
            "registration_year":   r.randint(2019, 2025, n),
        })

    def _gen_nonfr_accounts(self, ids: list[str]) -> pd.DataFrame:
        return self._base_features(ids)

    def _gen_fraud_typology(
        self, ids: list[str], ftype: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(ids)
        r = self.rng
        df = self._base_features(ids)

        if ftype == "sim_swap":
            # Hallmark: very high SIM + device changes, sudden new beneficiaries
            df["sim_change_count"]    = r.randint(3, 9, n)
            df["device_change_count"] = r.randint(5, 12, n)
            df["new_benef_rate"]      = r.beta(7, 3, n)              # ~0.70
            df["night_txn_rate"]      = r.beta(3, 7, n)              # ~0.30
            df["avg_txn_amount"]      = r.exponential(8_000, n) + 2_000
            df["account_age_days"]    = r.randint(300, 1500, n)      # established accounts

        elif ftype == "social_engineering":
            # Hallmark: victim sends large amount once to new beneficiary
            df["avg_txn_amount"]      = r.exponential(25_000, n) + 10_000
            df["max_txn_amount"]      = df["avg_txn_amount"] * r.uniform(2, 8, n)
            df["new_benef_rate"]      = r.beta(8, 2, n)              # ~0.80
            df["account_age_days"]    = r.randint(500, 3000, n)      # trusted old accounts
            df["p2p_ratio"]           = r.beta(9, 1, n)              # ~0.90 -- all P2P
            df["total_txn_count"]     = r.randint(5, 30, n)          # few txns

        elif ftype == "account_takeover":
            # Hallmark: new account, burst of night activity, device changes
            df["account_age_days"]    = r.randint(7, 90, n)          # very new
            df["night_txn_rate"]      = r.beta(5, 5, n)              # ~0.50
            df["device_change_count"] = r.randint(3, 8, n)
            df["txn_velocity_7d"]     = r.randint(10, 50, n)         # high velocity
            df["failed_txn_rate"]     = r.beta(3, 7, n)              # ~0.30 -- many fails before success
            df["avg_txn_amount"]      = r.exponential(5_000, n) + 1_000

        elif ftype == "fake_merchant":
            # Hallmark: high unique beneficiaries, low p2p, moderate amounts
            df["unique_beneficiaries"] = r.randint(80, 300, n)
            df["p2p_ratio"]            = r.beta(1, 9, n)             # ~0.10 -- mostly merchant
            df["avg_txn_amount"]       = r.normal(3_000, 800, n).clip(500, 15_000)
            df["total_txn_count"]      = r.randint(200, 1000, n)
            df["new_benef_rate"]       = r.beta(5, 5, n)             # ~0.50

        # Common fraud signals across all typologies
        df["failed_txn_rate"] = df["failed_txn_rate"].clip(0.08, 0.5)

        # Fraud labels
        base_date = pd.Timestamp("2024-01-01")
        labels_df = pd.DataFrame({
            "account_id":           ids,
            "fraud_type":           ftype,
            "fraud_confirmed_date": [
                str(base_date + pd.Timedelta(days=int(d)))
                for d in r.randint(0, 365, n)
            ],
            "fraud_amount":         (df["max_txn_amount"].values * r.uniform(0.5, 1.5, n)).clip(1000),
            "case_id":              [f"CASE_{ftype[:3].upper()}_{i:06d}" for i in range(n)],
        })

        return df, labels_df

    def _inject_hard_negatives(
        self, nonfr_df: pd.DataFrame, indices: np.ndarray
    ) -> pd.DataFrame:
        """
        Make selected non-fraud accounts look fraud-like (the hard negatives).
        These are legitimate accounts with unusual but legal patterns.
        """
        r = self.rng
        n = len(indices)
        df = nonfr_df.copy()

        # Mix of SIM-swap-like and account-takeover-like patterns
        df.loc[df.index[indices[:n//2]], "device_change_count"] = r.randint(3, 7, n//2)
        df.loc[df.index[indices[:n//2]], "sim_change_count"]    = r.randint(1, 4, n//2)
        df.loc[df.index[indices[n//2:]], "night_txn_rate"]      = r.beta(4, 6, n - n//2)
        df.loc[df.index[indices[n//2:]], "new_benef_rate"]      = r.beta(6, 4, n - n//2)
        df.loc[df.index[indices],        "is_hard_negative"]    = True

        # Tag (not used in modelling -- only for validation)
        if "is_hard_negative" not in df.columns:
            df["is_hard_negative"] = False
        df["is_hard_negative"] = df["is_hard_negative"].fillna(False)
        return df

    # -- Raw transaction generator ------------------------------------------

    def _gen_raw_transactions(self, sample_accounts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ~5 raw transactions per account (for consistency validation).
        Not a full population -- just enough for cross-dataset checks.
        """
        r = self.rng
        rows = []
        base_date = pd.Timestamp("2024-01-01")

        for _, acc in sample_accounts.iterrows():
            n_txns = max(1, int(acc.get("total_txn_count", 10) // 30))
            n_txns = min(n_txns, 20)  # cap for performance
            for _ in range(n_txns):
                is_night = r.random() < acc.get("night_txn_rate", 0.1)
                hour = r.randint(22, 30) % 24 if is_night else r.randint(8, 22)
                rows.append({
                    "account_id":     acc["account_id"],
                    "txn_id":         f"TXN_{r.randint(100_000_000, 999_999_999)}",
                    "txn_amount":     max(10, r.normal(
                        acc.get("avg_txn_amount", 2500),
                        acc.get("avg_txn_amount", 2500) * 0.5,
                    )),
                    "txn_date":       str(base_date + pd.Timedelta(
                        days=int(r.randint(0, 365)),
                        hours=int(hour),
                    )),
                    "beneficiary_id": f"BEN_{r.randint(1000, 99999)}",
                    "txn_type":       "P2P" if r.random() < acc.get("p2p_ratio", 0.6) else "MERCHANT",
                    "status":         "FAILED" if r.random() < acc.get("failed_txn_rate", 0.05) else "SUCCESS",
                    "device_id":      f"DEV_{r.randint(1000, 9999)}",
                    "bank_code":      acc.get("bank_code", "HDFC"),
                    "state_code":     acc.get("state_code", "MH"),
                })

        return pd.DataFrame(rows)
