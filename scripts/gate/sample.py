"""Write a uniform random sample of each raw gate dataset to
<data>/samples/<name>.parquet. Uniform, not fraud-enriched: role inference
and prevalence checks must see the real class balance.

    python scripts/gate/sample.py <data_dir> <rows> [name ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SEED = 20261105


def main(argv: list[str]) -> int:
    data, rows, names = Path(argv[0]), int(argv[1]), argv[2:]
    raw = data / "raw"
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        if names and d.name not in names:
            continue
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            print(f"[{d.name}] no csv, skipped")
            continue
        out = data / "samples" / f"{d.name}.parquet"
        df = pd.read_csv(csvs[0], low_memory=False)
        n = min(rows, len(df))
        s = df.sample(n=n, random_state=SEED).sort_index()
        s.to_parquet(out, index=False)
        print(f"[{d.name}] {len(df):,} rows -> sample {n:,} rows x {s.shape[1]} cols -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
