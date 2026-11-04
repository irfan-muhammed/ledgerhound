"""
ledgerhound.backends.pandas_backend
================================

File-based backend using pandas.  Supports CSV, TSV, Parquet, and Excel.
All operations are read-only -- the original DataFrame is never mutated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ledgerhound.backends.base import (
    CategoricalProfile, ColumnInfo, NumericProfile,
)


class PandasBackend:
    """Read-only data backend for CSV / Parquet / Excel files.

    Parameters
    ----------
    source : file path or ``pd.DataFrame``
    """

    _LOADERS = {
        ".csv":     pd.read_csv,
        ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
        ".parquet": pd.read_parquet,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
    }

    def __init__(self, source: str | Path | pd.DataFrame):
        if isinstance(source, pd.DataFrame):
            self.df: pd.DataFrame = source
            self._source_path = "<DataFrame>"
        else:
            path = Path(source)
            loader = self._LOADERS.get(path.suffix.lower())
            if loader is None:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. "
                    f"Supported: {list(self._LOADERS)}"
                )
            self.df = loader(path)
            self._source_path = str(source)

    @property
    def source_path(self) -> str:
        return self._source_path

    # -- core interface -------------------------------------

    def count_rows(self) -> int:
        return len(self.df)

    def get_columns(self) -> list[ColumnInfo]:
        """Return lightweight metadata for every column."""
        return [
            ColumnInfo(
                name=col,
                dtype=str(self.df[col].dtype),
                non_null=int(self.df[col].notna().sum()),
                null_count=int(self.df[col].isna().sum()),
                unique=int(self.df[col].nunique()),
            )
            for col in self.df.columns
        ]

    def sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Return up to *n* non-null sample values (deterministic)."""
        series = self.df[column].dropna()
        if len(series) == 0:
            return []
        return series.sample(min(n, len(series)), random_state=42).tolist()

    def null_count(self, column: str) -> int:
        return int(self.df[column].isna().sum())

    def distinct_count(self, column: str) -> int:
        return int(self.df[column].nunique())

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.df.head(n)

    # -- numeric profiling ----------------------------------

    def numeric_profile(self, column: str) -> NumericProfile:
        """Compute descriptive statistics for a numeric column."""
        series = self.df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        clean = numeric.dropna()

        if len(clean) == 0:
            return NumericProfile(count=len(series), null_count=int(series.isna().sum()))

        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        pctls = clean.quantile(quantiles)
        pctl_dict = {f"p{int(k*100)}": round(float(v), 4) for k, v in pctls.items()}

        return NumericProfile(
            count=len(series),
            non_null=len(clean),
            null_count=int(series.isna().sum()),
            null_pct=round(float(series.isna().mean()) * 100, 2),
            mean=round(float(clean.mean()), 4),
            median=round(float(clean.median()), 4),
            std=round(float(clean.std()), 4),
            min_val=round(float(clean.min()), 4),
            max_val=round(float(clean.max()), 4),
            skew=round(float(clean.skew()), 4),
            kurtosis=round(float(clean.kurtosis()), 4),
            zeros=int((clean == 0).sum()),
            negatives=int((clean < 0).sum()),
            percentiles=pctl_dict,
        )

    # -- categorical profiling ------------------------------

    def categorical_profile(self, column: str, top_n: int = 20) -> CategoricalProfile:
        """Compute statistics for a categorical column."""
        series = self.df[column]
        vc = series.value_counts(dropna=True).head(top_n)
        total = int(series.notna().sum())

        # Shannon entropy (base-2)
        entropy = 0.0
        if total > 0:
            probs = series.value_counts(normalize=True, dropna=True).values
            entropy = round(float(sp_stats.entropy(probs, base=2)), 4)

        mode_val = str(vc.index[0]) if len(vc) > 0 else ""
        mode_count = int(vc.iloc[0]) if len(vc) > 0 else 0

        return CategoricalProfile(
            count=len(series),
            non_null=total,
            null_count=int(series.isna().sum()),
            null_pct=round(float(series.isna().mean()) * 100, 2),
            unique=int(series.nunique()),
            top_values={str(k): int(v) for k, v in vc.items()},
            entropy=entropy,
            mode=mode_val,
            mode_pct=round(mode_count / max(total, 1) * 100, 2),
        )

    def value_counts(self, column: str, top_n: int = 20) -> dict[str, int]:
        vc = self.df[column].value_counts(dropna=True).head(top_n)
        return {str(k): int(v) for k, v in vc.items()}

    # -- target-aware analysis ------------------------------

    def by_target(self, column: str, target_col: str) -> dict[str, NumericProfile]:
        """Numeric profile split by each class of *target_col*."""
        result = {}
        for cls_val, group in self.df.groupby(target_col):
            sub = PandasBackend(group)
            result[str(cls_val)] = sub.numeric_profile(column)
        return result

    def ks_test(self, column: str, target_col: str) -> dict:
        """Two-sample Kolmogorov-Smirnov test between target classes.

        BUG FIX: uses ``.copy()`` to avoid SettingWithCopyWarning.
        """
        subset = self.df[[column, target_col]].copy()
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna()

        classes = subset[target_col].unique()
        if len(classes) != 2:
            return {"error": f"KS test needs binary target, got {len(classes)} classes"}

        a = subset.loc[subset[target_col] == classes[0], column]
        b = subset.loc[subset[target_col] == classes[1], column]
        if len(a) == 0 or len(b) == 0:
            return {"error": "One or both groups are empty"}

        stat, pval = sp_stats.ks_2samp(a, b)
        return {
            "statistic": round(float(stat), 6),
            "p_value": float(pval),
            "class_a": str(classes[0]),
            "class_b": str(classes[1]),
            "n_a": len(a),
            "n_b": len(b),
        }
