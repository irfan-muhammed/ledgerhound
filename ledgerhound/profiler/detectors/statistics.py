"""
ledgerhound.profiler.detectors.statistics
=======================================

Computes per-column statistics, optionally split by a target column.

BUG FIX: ``_fraud_rate_by_value`` now accesses ``backend.df`` directly
instead of the broken ``backend.head(backend.count_rows())`` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ledgerhound.backends.base import NumericProfile
from ledgerhound.schema.models import ColumnSchema, SemanticType


@dataclass
class ColumnStats:
    """Profiled statistics for a single column."""
    name: str
    semantic_type: str
    profile: dict = field(default_factory=dict)
    by_target: dict | None = None    # per-class profiles
    ks_test: dict | None = None      # two-sample KS result


@dataclass
class StatsReport:
    """Collection of per-column stats."""
    columns: dict[str, ColumnStats] = field(default_factory=dict)


class StatisticalProfiler:
    """Profile every column, with optional target-aware splits."""

    def profile(self, backend, columns: dict[str, ColumnSchema],
                target_col: str | None = None) -> StatsReport:
        report = StatsReport()

        for cname, scol in columns.items():
            st = scol.semantic_type

            # Skip identifiers and hashes (stats are meaningless)
            if st in (SemanticType.IDENTIFIER, SemanticType.HASH):
                report.columns[cname] = ColumnStats(
                    name=cname, semantic_type=st.value,
                    profile={"unique_count": backend.distinct_count(cname),
                             "note": "Identifier/Hash -- stats skipped"},
                )
                continue

            # Numeric columns
            if st == SemanticType.NUMERIC:
                report.columns[cname] = self._profile_numeric(
                    backend, cname, scol, target_col)
                continue

            # Categorical / Binary / Temporal
            if st in (SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.TEMPORAL):
                report.columns[cname] = self._profile_categorical(
                    backend, cname, scol, target_col)
                continue

            # Fallback
            report.columns[cname] = ColumnStats(
                name=cname, semantic_type=st.value,
                profile={"unique": backend.distinct_count(cname)},
            )

        return report

    # -- helpers --------------------------------------------

    def _profile_numeric(self, backend, cname, scol, target_col):
        np = backend.numeric_profile(cname)
        prof = {
            "count": np.count, "non_null": np.non_null,
            "null_count": np.null_count, "null_pct": np.null_pct,
            "mean": np.mean, "median": np.median, "std": np.std,
            "min": np.min_val, "max": np.max_val,
            "skew": np.skew, "kurtosis": np.kurtosis,
            "zeros": np.zeros, "negatives": np.negatives,
            "percentiles": np.percentiles,
        }
        # Display transform (e.g. paise -> INR)
        if scol.transform == "divide_by_100":
            prof["display_stats"] = {
                "mean": round(np.mean / 100, 2),
                "median": round(np.median / 100, 2),
                "min": round(np.min_val / 100, 2),
                "max": round(np.max_val / 100, 2),
                "unit": scol.display_unit or "INR",
            }

        cs = ColumnStats(name=cname, semantic_type=scol.semantic_type.value, profile=prof)

        # Target-aware analysis
        if target_col and hasattr(backend, "by_target"):
            try:
                bt = backend.by_target(cname, target_col)
                cs.by_target = {
                    k: {"mean": v.mean, "median": v.median, "std": v.std, "count": v.non_null}
                    for k, v in bt.items()
                }
            except Exception:
                pass
            try:
                cs.ks_test = backend.ks_test(cname, target_col)
            except Exception:
                pass

        return cs

    def _profile_categorical(self, backend, cname, scol, target_col):
        cp = backend.categorical_profile(cname)
        prof = {
            "count": cp.count, "non_null": cp.non_null,
            "null_count": cp.null_count, "null_pct": cp.null_pct,
            "unique": cp.unique, "entropy": cp.entropy,
            "mode": cp.mode, "mode_pct": cp.mode_pct,
            "top_values": cp.top_values,
        }
        cs = ColumnStats(name=cname, semantic_type=scol.semantic_type.value, profile=prof)

        # Fraud rate per category value
        if target_col:
            cs.by_target = self._fraud_rate_by_value(backend, cname, target_col)

        return cs

    @staticmethod
    def _fraud_rate_by_value(backend, column: str, target_col: str) -> dict:
        """Compute fraud rate per categorical value.

        Uses ``backend.df`` directly (not a copy via ``head()``).
        For very large data we sample 1M rows to keep it fast.
        """
        try:
            df = backend.df
            if len(df) > 5_000_000:
                df = df.sample(1_000_000, random_state=42)

            grouped = df.groupby(column)[target_col].agg(["mean", "count"])
            grouped = grouped.sort_values("mean", ascending=False).head(20)

            return {
                "fraud_rate_by_value": {
                    str(idx): {"fraud_rate": round(float(r["mean"]), 6), "count": int(r["count"])}
                    for idx, r in grouped.iterrows()
                }
            }
        except Exception:
            return {}
