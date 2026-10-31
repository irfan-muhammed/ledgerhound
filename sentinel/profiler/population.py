"""
sentinel.profiler.population
=============================

PopulationProfiler -- generates Trino push-down SQL for population-level
statistics. Returns kilobytes of aggregate statistics, never rows.

This is the foundation of all sampling certification and bias detection.
The SQL is generated here, executed by TrinoClient, and the resulting
stats are consumed by PSICalculator and SamplingCertificate.

Design principle: statistics-first, rows-never.
-------------------------------------------------
Every query generated here is an aggregation. No query returns individual
rows. The population's distributions are characterised entirely through
aggregate statistics (counts, means, percentiles, bucket distributions)
that can be computed in a single Trino pass over billions of rows.

Flow
----
1. PopulationProfiler.build_column_stats_query()
   -> Fast overview: COUNT, AVG, STDDEV, MIN, MAX, null rate per column.
   -> Used to understand the population shape before committing to PSI buckets.

2. PopulationProfiler.build_bucket_distribution_query()
   -> Bucket distribution for one continuous feature using caller-supplied bounds.
   -> Bounds come from the local sample's quantiles -- ensures both distributions
     use the same bucket boundaries for an apples-to-apples PSI comparison.

3. PopulationProfiler.build_categorical_distribution_query()
   -> Top-K value counts for a categorical feature.

4. PopulationProfiler.build_full_profile_query()
   -> Combined query for multiple numeric columns in one Trino round-trip.
     Returns one row per (feature, bucket) -- efficient but harder to parse.
     Use for batch profiling when you have many features to certify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class BucketBounds:
    """Boundary values for a numeric feature's PSI buckets."""
    feature: str
    bounds: list[float]     # sorted boundary values; len = n_buckets + 1
    source: str = "sample"  # "sample" or "population"

    @property
    def n_buckets(self) -> int:
        return len(self.bounds) - 1


class PopulationProfiler:
    """
    Generates Trino SQL for population distribution profiling.

    All methods return SQL strings -- nothing is executed here.
    Pass the returned SQL to TrinoClient.execute_stats_only().
    """

    def __init__(
        self,
        catalog: str = "",
        schema: str = "",
        approx_percentiles: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        catalog / schema  : Trino catalog and schema (prepended to table names).
        approx_percentiles: Use APPROX_PERCENTILE (fast, slightly imprecise) vs
                            PERCENTILE_CONT (exact, slow). Default True for speed.
        """
        self.catalog = catalog
        self.schema = schema
        self.approx_percentiles = approx_percentiles

    # -- 1. Column overview stats -------------------------------------------

    def build_column_stats_query(
        self,
        table: str,
        features: list[str],
        where_clause: str = "",
        sample_pct: Optional[float] = None,
    ) -> str:
        """
        Fast overview of population stats per column.

        Returns one row per feature with: count, null_count, null_rate,
        mean, stddev, min, max, approx_distinct.

        Parameters
        ----------
        table         : Table name (qualified if catalog/schema set).
        features      : Column names to profile.
        where_clause  : Optional WHERE clause (no leading WHERE keyword).
        sample_pct    : If set (e.g. 1.0), use TABLESAMPLE for speed.
        """
        full_table = self._qualify(table, sample_pct)
        where = f"WHERE {where_clause}" if where_clause else ""

        selects = []
        for feat in features:
            selects.append(f"""
    SELECT
        '{feat}'                                  AS feature,
        COUNT(*)                                  AS total_rows,
        COUNT({feat})                             AS non_null_count,
        COUNT(*) - COUNT({feat})                  AS null_count,
        ROUND(
            1.0 * (COUNT(*) - COUNT({feat})) / NULLIF(COUNT(*), 0), 6
        )                                         AS null_rate,
        AVG(TRY_CAST({feat} AS DOUBLE))           AS mean,
        STDDEV(TRY_CAST({feat} AS DOUBLE))        AS stddev,
        MIN(TRY_CAST({feat} AS DOUBLE))           AS min_val,
        MAX(TRY_CAST({feat} AS DOUBLE))           AS max_val,
        APPROX_DISTINCT({feat})                   AS approx_distinct
    FROM {full_table} {where}""")

        union_sql = "\nUNION ALL\n".join(selects)
        return f"-- Population column stats: {', '.join(features)}\n{union_sql}\nORDER BY feature"

    # -- 2. Bucket distribution for PSI ------------------------------------

    def build_bucket_distribution_query(
        self,
        table: str,
        bounds: BucketBounds,
        where_clause: str = "",
        sample_pct: Optional[float] = None,
    ) -> str:
        """
        Count rows per bucket for a continuous feature.

        Bucket boundaries come from BucketBounds (typically derived from the
        local sample's quantiles). This guarantees PSI buckets are identical
        in both population and sample.

        Returns columns: feature, bucket_idx, lo, hi, population_count,
                         population_total, population_pct.
        """
        full_table = self._qualify(table, sample_pct)
        where = f"AND ({where_clause})" if where_clause else ""
        feat = bounds.feature
        n = bounds.n_buckets
        lo_vals = bounds.bounds[:-1]
        hi_vals = bounds.bounds[1:]

        # Build CASE WHEN for each bucket
        when_clauses = []
        for i, (lo, hi) in enumerate(zip(lo_vals, hi_vals)):
            if i == 0:
                when_clauses.append(
                    f"    WHEN TRY_CAST({feat} AS DOUBLE) < {hi} THEN {i}"
                )
            elif i == n - 1:
                when_clauses.append(
                    f"    WHEN TRY_CAST({feat} AS DOUBLE) >= {lo} THEN {i}"
                )
            else:
                when_clauses.append(
                    f"    WHEN TRY_CAST({feat} AS DOUBLE) >= {lo} "
                    f"AND TRY_CAST({feat} AS DOUBLE) < {hi} THEN {i}"
                )

        case_expr = "CASE\n" + "\n".join(when_clauses) + "\n    ELSE -1\nEND"

        return f"""
-- Population bucket distribution: {feat} ({n} buckets)
WITH bucketed AS (
    SELECT {case_expr} AS bucket_idx
    FROM {full_table}
    WHERE {feat} IS NOT NULL {where}
),
counts AS (
    SELECT
        bucket_idx,
        COUNT(*) AS cnt
    FROM bucketed
    WHERE bucket_idx >= 0
    GROUP BY bucket_idx
),
total AS (SELECT SUM(cnt) AS grand_total FROM counts)
SELECT
    '{feat}'                                    AS feature,
    c.bucket_idx,
    c.cnt                                       AS population_count,
    t.grand_total                               AS population_total,
    ROUND(1.0 * c.cnt / NULLIF(t.grand_total, 0), 8) AS population_pct
FROM counts c
CROSS JOIN total t
ORDER BY c.bucket_idx
""".strip()

    # -- 3. Categorical distribution ----------------------------------------

    def build_categorical_distribution_query(
        self,
        table: str,
        feature: str,
        top_k: int = 50,
        where_clause: str = "",
        sample_pct: Optional[float] = None,
    ) -> str:
        """
        Top-K value counts for a categorical feature.
        Returns: feature, value, population_count, population_pct.
        """
        full_table = self._qualify(table, sample_pct)
        where = f"WHERE {where_clause}" if where_clause else ""

        return f"""
-- Population categorical distribution: {feature} (top {top_k})
WITH counts AS (
    SELECT
        CAST({feature} AS VARCHAR) AS val,
        COUNT(*) AS cnt
    FROM {full_table}
    {where}
    GROUP BY {feature}
    ORDER BY cnt DESC
    LIMIT {top_k}
),
total AS (SELECT SUM(cnt) AS grand_total FROM counts)
SELECT
    '{feature}'                                  AS feature,
    c.val                                        AS value,
    c.cnt                                        AS population_count,
    t.grand_total                                AS population_total,
    ROUND(1.0 * c.cnt / NULLIF(t.grand_total, 0), 8) AS population_pct
FROM counts c
CROSS JOIN total t
ORDER BY c.cnt DESC
""".strip()

    # -- 4. Batch profile (multiple features, one query) --------------------

    def build_full_profile_query(
        self,
        table: str,
        numeric_features: list[str],
        n_buckets: int = 20,
        where_clause: str = "",
        sample_pct: Optional[float] = None,
    ) -> str:
        """
        Combined overview stats for multiple numeric columns.
        Efficient: one Trino round-trip for all features.
        Returns one row per feature.
        """
        return self.build_column_stats_query(
            table=table,
            features=numeric_features,
            where_clause=where_clause,
            sample_pct=sample_pct,
        )

    # -- 5. Percentile query (for deriving bucket bounds) ------------------

    def build_percentile_query(
        self,
        table: str,
        feature: str,
        percentiles: Optional[list[float]] = None,
        where_clause: str = "",
        sample_pct: Optional[float] = None,
    ) -> str:
        """
        Compute quantile values for a feature in the population.
        Used when you want population-derived bucket bounds instead of sample-derived.
        Returns one row with one column per percentile.
        """
        pcts = percentiles or [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        full_table = self._qualify(table, sample_pct)
        where = f"WHERE {where_clause}" if where_clause else ""

        if self.approx_percentiles:
            pct_exprs = [
                f"APPROX_PERCENTILE(TRY_CAST({feature} AS DOUBLE), {p}) AS p{int(p*100):02d}"
                for p in pcts
            ]
        else:
            pct_exprs = [
                f"PERCENTILE_CONT({p}) WITHIN GROUP (ORDER BY TRY_CAST({feature} AS DOUBLE)) AS p{int(p*100):02d}"
                for p in pcts
            ]

        return f"""
-- Population percentiles: {feature}
SELECT
    '{feature}' AS feature,
    COUNT(*) AS n,
    {',\n    '.join(pct_exprs)}
FROM {full_table}
{where}
""".strip()

    # -- 6. Simple count ----------------------------------------------------

    def build_count_query(self, table: str, where_clause: str = "") -> str:
        """Simple COUNT(*) -- fast population size check."""
        full_table = self._qualify(table)
        where = f"WHERE {where_clause}" if where_clause else ""
        return f"SELECT COUNT(*) AS population_count FROM {full_table} {where}".strip()

    # -- Internal -----------------------------------------------------------

    def _qualify(self, table: str, sample_pct: Optional[float] = None) -> str:
        """Build fully-qualified table reference, optionally with TABLESAMPLE."""
        parts = []
        if self.catalog:
            parts.append(self.catalog)
        if self.schema:
            parts.append(self.schema)
        parts.append(table)
        qualified = ".".join(parts)
        if sample_pct is not None:
            return f"{qualified} TABLESAMPLE BERNOULLI ({sample_pct})"
        return qualified


# -- Helpers for consuming Trino results -----------------------------------


def parse_column_stats(rows: list[dict]) -> dict[str, dict]:
    """
    Convert raw rows from build_column_stats_query into a clean dict.
    Returns: {feature_name: {mean, stddev, min_val, max_val, null_rate, ...}}
    """
    return {row["feature"]: {k: v for k, v in row.items() if k != "feature"} for row in rows}


def parse_bucket_distribution(rows: list[dict]) -> dict[int, float]:
    """
    Convert raw rows from build_bucket_distribution_query into {bucket_idx: pct}.
    """
    return {int(row["bucket_idx"]): float(row["population_pct"] or 0) for row in rows}


def parse_categorical_distribution(rows: list[dict]) -> dict[str, float]:
    """
    Convert raw rows from build_categorical_distribution_query into {value: pct}.
    """
    return {str(row["value"]): float(row["population_pct"] or 0) for row in rows}
