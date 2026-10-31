"""
sentinel.adversarial.sampling
================================

Pick which fraud cases the adversarial deep-research loop spends its
(very expensive -- up to 150 LLM turns per case) iteration budget on, and
carve out each case's own data slice.

Reads the label/key columns directly off the SOURCE data file rather than
whatever the parent shell session currently has loaded (which may be a
schema-only sample above the shell's load-size ceiling for a crore-scale
file) -- fraud rows are typically a tiny fraction of the whole file, so
scanning just the label+key columns for them is cheap even when loading
the whole file for analysis is not.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd

# Values treated as "this row is fraud" when the label column isn't a
# clean 0/1 -- covers the common encodings this codebase has seen
# (upi_mule.yaml and friends use "flag"/"is_fraud" as 0/1 ints, but a
# hand-labelled export can come back as strings).
FRAUD_VALUES = frozenset({"1", "true", "fraud", "yes", "y"})


def _read_columns(path: str | Path, columns: list) -> pd.DataFrame:
    path = Path(path)
    cols = [c for c in dict.fromkeys(columns) if c]  # de-dupe, keep order
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, usecols=cols)


def _is_fraud(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(FRAUD_VALUES)


def sample_fraud_cases(path: str | Path, *, label_column: str, entity_key: str,
                       entity_key_secondary: str = "", n: int = 25,
                       seed: Optional[int] = None) -> list:
    """Up to `n` random, distinct fraud entities from `path` -- the unit
    sentinel.adversarial.engine spends its per-case iteration budget on.

    Returns [{"entity_key": <payer id>, "entity_key_secondary": <payee id
    from that same row, or None>}, ...]. Empty list (never raises) if the
    label column has no fraud rows, is missing, or the file can't be
    read -- an adversarial run over nothing is a no-op, not a crash.
    """
    try:
        df = _read_columns(path, [label_column, entity_key, entity_key_secondary])
    except Exception:  # noqa: BLE001
        return []
    if label_column not in df.columns or entity_key not in df.columns:
        return []
    fraud_df = df[_is_fraud(df[label_column])]
    if fraud_df.empty:
        return []
    fraud_df = fraud_df.drop_duplicates(subset=[entity_key])
    rows = fraud_df.to_dict("records")
    random.Random(seed).shuffle(rows)
    picked = rows[:n]
    return [
        {"entity_key": r.get(entity_key),
         "entity_key_secondary": (r.get(entity_key_secondary)
                                  if entity_key_secondary else None)}
        for r in picked
    ]


# I-07: rows per batch when STREAMING the source file. The whole point is
# that the shell process never holds the full crore-scale file -- one
# batch at a time, keep the matching rows, move on.
_STREAM_BATCH_ROWS = 1_000_000


def _iter_batches(source_path: Path):
    """Yield the source file as pandas frames of ~_STREAM_BATCH_ROWS rows.
    Parquet streams by row group via pyarrow; CSV via pandas chunksize."""
    if source_path.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(source_path)
        for batch in pf.iter_batches(batch_size=_STREAM_BATCH_ROWS):
            yield batch.to_pandas()
    else:
        sep = "\t" if source_path.suffix.lower() == ".tsv" else ","
        for chunk in pd.read_csv(source_path, sep=sep,
                                 chunksize=_STREAM_BATCH_ROWS):
            yield chunk


def _extract_matching(source_path: str | Path, out_path: str | Path,
                      matcher) -> int:
    """Stream `source_path`, keep the rows `matcher(batch) -> bool mask`
    selects, write them to `out_path` (parquet). Returns the row count.
    Memory is bounded by one batch plus the (small) matched slice --
    previously the ENTIRE file was read into pandas per call, twice per
    case, which at 8-crore rows meant 50 full reads in the shell process
    for a 25-case run (I-07)."""
    source_path = Path(source_path)
    kept = []
    for batch in _iter_batches(source_path):
        mask = matcher(batch)
        if mask is not None and bool(mask.any()):
            kept.append(batch[mask])
    slice_df = (pd.concat(kept, ignore_index=True) if kept
                else next(_iter_batches(source_path)).iloc[0:0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    slice_df.to_parquet(out_path, index=False)
    return len(slice_df)


def extract_case_slice(source_path: str | Path, out_path: str | Path, *,
                       account_id, payer_col: str, payee_col: str = "") -> int:
    """Write every row where `account_id` appears as EITHER payer or
    payee in `source_path` to `out_path` (parquet) -- "one fraud entity
    and all related accounts" as one small file each debate agent can
    `load_data` independently. Returns the row count written (0 if
    nothing matched -- callers should skip a case rather than hand two
    agents an empty file to debate). Streams the source (I-07)."""
    def matcher(df):
        mask = df[payer_col] == account_id
        if payee_col and payee_col in df.columns:
            mask = mask | (df[payee_col] == account_id)
        return mask
    return _extract_matching(source_path, out_path, matcher)


def extract_related_slice(source_path: str | Path, out_path: str | Path, *,
                          account_ids: list, payer_col: str,
                          payee_col: str = "") -> int:
    """Like extract_case_slice, but for the PAYEE-side round: every row
    where any of `account_ids` (the fraud payer's counterparties) appears
    as payer or payee. Writes an empty-select (0 rows) rather than raising
    when `account_ids` is empty -- callers should check the returned
    count, not assume a nonzero write. Streams the source (I-07)."""
    if not account_ids:
        return 0
    ids = set(account_ids)

    def matcher(df):
        mask = df[payer_col].isin(ids)
        if payee_col and payee_col in df.columns:
            mask = mask | df[payee_col].isin(ids)
        return mask
    return _extract_matching(source_path, out_path, matcher)


def related_accounts(slice_df: pd.DataFrame, account_id, *,
                     payer_col: str, payee_col: str = "") -> list:
    """The OTHER party of every transaction `account_id` appears in --
    the payee-side accounts a payer-side debate should then examine."""
    others = set()
    if payee_col and payee_col in slice_df.columns:
        payer_rows = slice_df[slice_df[payer_col] == account_id]
        others.update(payer_rows[payee_col].dropna().unique().tolist())
    if payer_col in slice_df.columns:
        payee_rows = (slice_df[slice_df[payee_col] == account_id]
                      if payee_col and payee_col in slice_df.columns
                      else slice_df.iloc[0:0])
        others.update(payee_rows[payer_col].dropna().unique().tolist())
    others.discard(account_id)
    return sorted(others, key=str)
