"""Column-role inference, the RoleMap rules, and the store.

Frames here are shaped like common public fraud schemas (synthetic values),
so the generic rules are pinned before any real data is involved. The
product code contains no dataset-specific names; these tests are where such
names are allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ledgerhound.roles import RoleMap, RoleStore, infer_roles, tokens

RNG = np.random.default_rng(7)
N = 400


def _ids(prefix: str, n: int = N, k: int = 300) -> list[str]:
    return [f"{prefix}{i}" for i in RNG.integers(1_000_000, 1_000_000 + k, n)]


def _bin(rate: float = 0.05) -> np.ndarray:
    return (RNG.random(N) < rate).astype(int)


def roles_of(df: pd.DataFrame) -> dict[str, str]:
    return {c: a.role for c, a in infer_roles(df).columns.items()}


def test_tokens():
    assert tokens("nameOrig") == ["name", "orig"]
    assert tokens("TransactionAmt") == ["transaction", "amt"]
    assert tokens("isFlaggedFraud") == ["is", "flagged", "fraud"]
    assert tokens("fraud_bool") == ["fraud", "bool"]
    assert tokens("card1") == ["card", "1"]
    assert tokens("ProductCD") == ["product", "cd"]


def test_simulated_mobile_money_schema():
    df = pd.DataFrame({
        "step": RNG.integers(1, 700, N), "type": RNG.choice(["CASH_OUT", "TRANSFER", "PAYMENT"], N),
        "amount": RNG.gamma(2, 5000, N), "nameOrig": _ids("C"),
        "oldbalanceOrg": RNG.gamma(2, 9000, N), "newbalanceOrig": RNG.gamma(2, 9000, N),
        "nameDest": _ids("M"), "oldbalanceDest": RNG.gamma(2, 9000, N),
        "isFraud": _bin(), "isFlaggedFraud": _bin(0.01)})
    r = roles_of(df)
    assert r["isFraud"] == "label"
    assert r["isFlaggedFraud"] == "rule_output"
    assert r["nameOrig"] == "payer" and r["nameDest"] == "payee"
    assert r["amount"] == "amount" and r["type"] == "channel"
    assert r["step"] == "event_time"
    assert r["oldbalanceOrg"] == r["newbalanceOrig"] == "feature"


def test_anonymised_card_schema_is_mostly_plain_features():
    cols = {"Time": np.sort(RNG.integers(0, 170_000, N)).astype(float)}
    cols.update({f"V{i}": RNG.normal(size=N) for i in range(1, 29)})
    cols.update({"Amount": RNG.gamma(2, 50, N), "Class": _bin(0.002 * 50)})
    rm = infer_roles(pd.DataFrame(cols))
    assert rm.role_of("Class") == "label" and rm.role_of("Amount") == "amount"
    assert rm.role_of("Time") == "event_time"
    assert all(rm.role_of(f"V{i}") == "feature" for i in range(1, 29))
    assert all(rm.columns[f"V{i}"].status == "auto" for i in range(1, 29))


def test_ecommerce_schema():
    df = pd.DataFrame({
        "TransactionID": np.arange(N) + 2_987_000, "isFraud": _bin(0.035),
        "TransactionDT": np.sort(RNG.integers(86_400, 15_000_000, N)),
        "TransactionAmt": RNG.gamma(2, 60, N), "ProductCD": RNG.choice(list("WHCSR"), N),
        "card1": RNG.integers(1000, 18000, N), "card4": RNG.choice(["visa", "mastercard"], N),
        "addr1": RNG.integers(100, 540, N).astype(float), "P_emaildomain": RNG.choice(["a.com", "b.com"], N),
        "C1": RNG.integers(0, 5, N), "D1": RNG.integers(0, 600, N), "M4": RNG.choice(["M0", "M1"], N)})
    r = roles_of(df)
    assert r["TransactionID"] == "transaction_id" and r["isFraud"] == "label"
    assert r["TransactionDT"] == "event_time" and r["TransactionAmt"] == "amount"
    assert r["ProductCD"] == "channel" and r["card1"] == "card"
    assert r["card4"] == r["addr1"] == r["P_emaildomain"] == r["C1"] == r["D1"] == "feature"


def test_application_schema_avoids_aggregate_traps():
    df = pd.DataFrame({
        "fraud_bool": _bin(0.011), "income": RNG.random(N), "customer_age": RNG.integers(18, 90, N),
        "intended_balcon_amount": RNG.normal(size=N), "payment_type": RNG.choice(["AA", "AB", "AC"], N),
        "date_of_birth_distinct_emails_4w": RNG.integers(0, 30, N),
        "device_fraud_count": np.zeros(N, dtype=int), "device_os": RNG.choice(["linux", "windows"], N),
        "credit_risk_score": RNG.normal(100, 50, N), "month": RNG.integers(0, 8, N),
        "source": RNG.choice(["INTERNET", "TELEAPP"], N)})
    r = roles_of(df)
    assert r["fraud_bool"] == "label"
    assert r["date_of_birth_distinct_emails_4w"] == "feature"     # not event_time
    assert r["device_fraud_count"] == "feature"                    # not a second label
    assert r["month"] == "event_time" and r["payment_type"] == "channel"
    assert r["device_os"] == r["source"] == r["customer_age"] == "feature"
    # an internal score is put to the person, not silently accepted as a feature
    assert r["credit_risk_score"] == "score"


def test_two_label_candidates_keep_one_and_ask_about_the_other():
    df = pd.DataFrame({"is_fraud": _bin(), "chargeback": _bin(), "amt": RNG.random(N)})
    rm = infer_roles(df)
    assert rm.columns_with("label") == ["is_fraud"]
    assert rm.columns["chargeback"].role == "feature"
    assert rm.columns["chargeback"].confidence == "medium"
    assert "chargeback" in rm.needs_asking()


def test_label_time_and_source_are_found():
    df = pd.DataFrame({
        "txn_ts": pd.date_range("2024-01-01", periods=N, freq="h").astype(str),
        "is_fraud": _bin(), "chargeback_date": pd.date_range("2024-02-01", periods=N, freq="h").astype(str),
        "fraud_source": RNG.choice(["chargeback", "complaint", "rule"], N)})
    r = roles_of(df)
    assert r["txn_ts"] == "event_time" and r["chargeback_date"] == "label_time"
    assert r["fraud_source"] == "label_source"


def test_rolemap_confirmation_rules():
    df = pd.DataFrame({"txn_id": range(N), "amount": RNG.random(N), "is_fraud": _bin()})
    rm = infer_roles(df, data_sha256="abc")
    assert not rm.confirmed
    assert rm.columns["txn_id"].status == rm.columns["amount"].status == "auto"
    assert rm.absent == {"label_time": "proposed", "label_source": "proposed"}
    rm.settle("agent")
    assert rm.confirmed and rm.confirmed_by == "agent"
    assert rm.columns["is_fraud"].status == "agent"
    assert rm.absent == {"label_time": "agent", "label_source": "agent"}
    back = RoleMap.from_dict(rm.to_dict())
    assert back.confirmed and back.role_of("is_fraud") == "label"


def test_dictionary_hints_are_proposals_not_settlements():
    df = pd.DataFrame({"x1": RNG.random(N), "y": _bin()})
    rm = infer_roles(df, hints={"x1": "score"})
    assert rm.role_of("x1") == "score" and rm.columns["x1"].status == "proposed"


def test_store_reuses_only_a_confirmed_map_with_the_same_columns(tmp_path):
    df = pd.DataFrame({"amount": RNG.random(N), "is_fraud": _bin()})
    store = RoleStore(tmp_path)
    rm = infer_roles(df, data_sha256="sha-1")
    store.save(rm)
    assert store.load("sha-1", list(df.columns)) is None        # not confirmed
    rm.settle("person")
    store.save(rm)
    assert store.load("sha-1", list(df.columns)).confirmed
    assert store.load("sha-1", ["amount"]) is None              # columns changed
    assert store.load("sha-2", list(df.columns)) is None


@pytest.mark.parametrize("bad", [pd.Series([None] * N), pd.Series([[1]] * N)])
def test_odd_columns_never_break_inference(bad):
    df = pd.DataFrame({"amount": RNG.random(N), "odd": bad})
    assert infer_roles(df).role_of("amount") == "amount"
