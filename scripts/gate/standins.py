"""Synthetic stand-ins for the four gate datasets: the real column names and
value shapes, random values. They let run_gate.py be dry-run (and tested in
CI) without the Kaggle downloads. They say nothing about how well role
inference does on the real data -- only the real samples do.

    python scripts/gate/standins.py <dir>     # writes <dir>/samples/*.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def paysim(n: int, rng) -> pd.DataFrame:
    amount = rng.gamma(1.5, 9000, n).round(2)
    old = rng.gamma(1.2, 50000, n).round(2)
    fraud = (rng.random(n) < 0.03).astype(int)
    return pd.DataFrame({
        "step": rng.integers(1, 744, n), "type": rng.choice(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], n),
        "amount": amount, "nameOrig": [f"C{x}" for x in rng.integers(1e8, 2e9, n)],
        "oldbalanceOrg": old, "newbalanceOrig": np.maximum(old - amount, 0).round(2),
        "nameDest": [("M" if rng.random() < 0.3 else "C") + str(x)
                     for x in rng.integers(1e8, 2e9, n)],
        "oldbalanceDest": rng.gamma(1.2, 80000, n).round(2),
        "newbalanceDest": rng.gamma(1.2, 90000, n).round(2),
        "isFraud": fraud, "isFlaggedFraud": (fraud & (amount > 30000)).astype(int)})


def ulb(n: int, rng) -> pd.DataFrame:
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172792, n)).round(0)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(0, 1.5, n)
    df["Amount"] = rng.gamma(1.0, 88, n).round(2)
    df["Class"] = (rng.random(n) < 0.02).astype(int)
    return df


def ieee(n: int, rng) -> pd.DataFrame:
    df = pd.DataFrame({
        "TransactionID": np.arange(2987000, 2987000 + n),
        "isFraud": (rng.random(n) < 0.035).astype(int),
        "TransactionDT": np.sort(rng.integers(86400, 15811131, n)),
        "TransactionAmt": rng.gamma(1.3, 100, n).round(3),
        "ProductCD": rng.choice(list("WHCSR"), n, p=[.74, .06, .12, .02, .06]),
        "card1": rng.integers(1000, 18397, n),
        "card2": rng.choice(np.arange(100, 601, dtype=float), n),
        "card3": rng.choice([150.0, 185.0, 106.0], n),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n),
        "card5": rng.choice([226.0, 224.0, 166.0, 102.0, 117.0], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice(np.arange(100, 541, dtype=float), n),
        "addr2": rng.choice([87.0, 60.0, 96.0], n),
        "dist1": rng.gamma(1.0, 100, n).round(0),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", "anonymous.com", None], n)})
    for c in ("C1", "C2", "C13", "D1", "D15", "V258", "V294"):
        df[c] = rng.poisson(3, n).astype(float)
    for c in ("M4", "M6"):
        df[c] = rng.choice(["T", "F", None], n)
    return df


def baf(n: int, rng) -> pd.DataFrame:
    return pd.DataFrame({
        "fraud_bool": (rng.random(n) < 0.011).astype(int),
        "income": rng.choice([0.1, 0.3, 0.6, 0.9], n),
        "name_email_similarity": rng.random(n),
        "prev_address_months_count": rng.integers(-1, 300, n),
        "current_address_months_count": rng.integers(-1, 400, n),
        "customer_age": rng.choice([20, 30, 40, 50, 60], n),
        "days_since_request": rng.exponential(1.0, n),
        "intended_balcon_amount": rng.normal(8, 20, n),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD"], n),
        "zip_count_4w": rng.integers(1, 6000, n),
        "velocity_6h": rng.gamma(2, 2500, n), "velocity_24h": rng.gamma(2, 2500, n),
        "bank_branch_count_8w": rng.integers(0, 2400, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
        "employment_status": rng.choice(["CA", "CB", "CC"], n),
        "credit_risk_score": rng.integers(-170, 390, n),
        "email_is_free": rng.integers(0, 2, n),
        "housing_status": rng.choice(["BA", "BB", "BC"], n),
        "phone_home_valid": rng.integers(0, 2, n), "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.choice([200.0, 500.0, 1500.0], n),
        "foreign_request": rng.integers(0, 2, n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[.99, .01]),
        "session_length_in_minutes": rng.gamma(1.5, 5, n),
        "device_os": rng.choice(["windows", "linux", "macintosh", "other", "x11"], n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.integers(0, 3, n),
        "device_fraud_count": np.zeros(n, dtype=int),
        "month": rng.integers(0, 8, n)})


MAKERS = {"paysim": paysim, "ulb": ulb, "ieee": ieee, "baf": baf}


def write(root: Path, n: int = 4000, seed: int = 20261105) -> Path:
    out = Path(root) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, make in MAKERS.items():
        make(n, np.random.default_rng(seed)).to_parquet(out / f"{name}.parquet", index=False)
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("standins")))
