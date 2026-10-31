"""
sentinel.backends.base
======================

Protocol (interface) that every data backend must implement.

The profiler and agent never know whether data lives in a Parquet file
or a Trino cluster -- they call the same methods either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass
class NumericProfile:
    """Descriptive statistics for a numeric column."""
    count: int = 0
    non_null: int = 0
    null_count: int = 0
    null_pct: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    zeros: int = 0
    negatives: int = 0
    percentiles: dict[str, float] = field(default_factory=dict)


@dataclass
class CategoricalProfile:
    """Descriptive statistics for a categorical column."""
    count: int = 0
    non_null: int = 0
    null_count: int = 0
    null_pct: float = 0.0
    unique: int = 0
    top_values: dict[str, int] = field(default_factory=dict)
    entropy: float = 0.0
    mode: str = ""
    mode_pct: float = 0.0


@dataclass
class ColumnInfo:
    """Lightweight column metadata returned by ``get_columns()``."""
    name: str
    dtype: str
    non_null: int = 0
    null_count: int = 0
    unique: int = 0


@runtime_checkable
class DataBackend(Protocol):
    """Interface that all storage backends implement."""

    def count_rows(self) -> int: ...
    def get_columns(self) -> list[ColumnInfo]: ...
    def sample_values(self, column: str, n: int = 5) -> list[Any]: ...
    def null_count(self, column: str) -> int: ...
    def distinct_count(self, column: str) -> int: ...
    def numeric_profile(self, column: str) -> NumericProfile: ...
    def categorical_profile(self, column: str, top_n: int = 20) -> CategoricalProfile: ...
    def value_counts(self, column: str, top_n: int = 20) -> dict[str, int]: ...
    def head(self, n: int = 5) -> pd.DataFrame: ...
