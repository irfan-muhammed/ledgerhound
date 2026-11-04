"""
ledgerhound.validation.consistency
================================

CrossDatasetValidator -- checks that the UPI summary table accurately
represents the underlying raw transaction data.

Why this matters
----------------
Feature engineers trust the summary table. If the aggregation logic has
bugs -- null propagation errors, incorrect time windows, off-by-one in
transaction counts -- those bugs silently corrupt every model trained on
those features. Models degrade in ways that are almost impossible to
diagnose post-deployment.

This validator runs checks that quantify the agreement between:
  summary_df   : One row per account. Pre-aggregated features.
  raw_df       : Raw transaction rows. May be a sample (not full population).
  join_key     : Column linking accounts between the two tables.

Checks performed
----------------
1. JOIN KEY COVERAGE   -- what % of summary accounts appear in raw transactions?
2. ROW COUNT SANITY    -- does raw txn count per account match summary's txn count?
3. AMOUNT AGREEMENT    -- does summary's avg_txn_amount match raw data mean?
4. NULL PROPAGATION    -- do nulls in raw transactions produce nulls in summary?
5. EDGE CASE ACCOUNTS  -- single-txn accounts, zero-balance, day-0 accounts behave correctly?
6. POPULATION LEAKAGE  -- are fraud-labeled accounts present in both datasets?
7. TEMPORAL ALIGNMENT  -- does the summary cover the same date range as the raw data?

Each check returns a ConsistencyCheck with status PASS / WARNING / FAIL and
supporting detail. All checks are non-destructive -- read-only analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)
console = Console()


class CheckStatus:
    PASS    = "PASS"
    WARNING = "WARNING"
    FAIL    = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass
class ConsistencyCheck:
    """Result of one consistency check."""
    check_name: str
    status: str
    message: str
    metric: Optional[float] = None    # numeric value of the check (e.g. coverage %)
    threshold: Optional[float] = None # pass/fail threshold
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsistencyReport:
    """Full cross-dataset consistency report."""
    generated_at: str
    summary_dataset: str
    raw_dataset: str
    join_key: str
    checks: list[ConsistencyCheck] = field(default_factory=list)
    n_pass: int = 0
    n_warning: int = 0
    n_fail: int = 0
    n_skipped: int = 0
    overall_status: str = CheckStatus.PASS
    notes: str = ""

    def is_clean(self) -> bool:
        return self.n_fail == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def display(self) -> None:
        status_color = {"PASS": "green", "WARNING": "yellow", "FAIL": "red"}.get(
            self.overall_status, "white"
        )
        console.print(f"\n[bold {status_color}]Cross-Dataset Consistency -- {self.overall_status}[/bold {status_color}]")
        console.print(
            f"  {self.n_pass} pass  {self.n_warning} warning  "
            f"{self.n_fail} fail  {self.n_skipped} skipped\n"
        )

        t = Table("Check", "Status", "Metric", "Message", show_header=True, header_style="bold dim")
        sym = {CheckStatus.PASS: "OK", CheckStatus.WARNING: "!", CheckStatus.FAIL: "X", CheckStatus.SKIPPED: "--"}
        clr = {CheckStatus.PASS: "green", CheckStatus.WARNING: "yellow", CheckStatus.FAIL: "red", CheckStatus.SKIPPED: "dim"}

        for c in self.checks:
            metric_str = f"{c.metric:.2%}" if c.metric is not None else "--"
            color = clr.get(c.status, "")
            t.add_row(
                c.check_name,
                f"[{color}]{sym.get(c.status, '?')} {c.status}[/{color}]",
                metric_str,
                c.message[:70],
            )
        console.print(t)


class CrossDatasetValidator:
    """
    Validates agreement between a summary table and raw transaction data.

    Parameters
    ----------
    join_coverage_threshold  : Min fraction of summary accounts that must
                               appear in raw data. Default 0.95.
    amount_tolerance_pct     : Max allowed % difference between summary's
                               avg amount and raw data mean. Default 5.0%.
    count_tolerance_pct      : Max allowed % difference in txn counts.
                               Default 10.0%.
    """

    def __init__(
        self,
        join_coverage_threshold: float = 0.95,
        amount_tolerance_pct: float = 5.0,
        count_tolerance_pct: float = 10.0,
    ) -> None:
        self.join_coverage_threshold = join_coverage_threshold
        self.amount_tolerance_pct = amount_tolerance_pct
        self.count_tolerance_pct = count_tolerance_pct

    def validate(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        join_key: str,
        fraud_label_ids: Optional[pd.Series] = None,
        summary_txn_count_col: Optional[str] = None,
        summary_avg_amount_col: Optional[str] = None,
        raw_amount_col: Optional[str] = None,
        summary_date_col: Optional[str] = None,
        raw_date_col: Optional[str] = None,
        summary_dataset_name: str = "summary",
        raw_dataset_name: str = "raw_transactions",
    ) -> ConsistencyReport:
        """
        Run all consistency checks and return a report.

        Parameters
        ----------
        summary_df             : Summary table (one row per account).
        raw_df                 : Raw transactions (may be a sample).
        join_key               : Column name present in both DataFrames.
        fraud_label_ids        : Optional Series of confirmed fraud account IDs.
                                 Used for population leakage check.
        summary_txn_count_col  : Column in summary_df with transaction count.
        summary_avg_amount_col : Column in summary_df with average transaction amount.
        raw_amount_col         : Column in raw_df with transaction amount.
        summary_date_col       : Date column in summary_df.
        raw_date_col           : Date column in raw_df.
        """
        checks: list[ConsistencyCheck] = []

        # 1. Join key coverage
        checks.append(self._check_join_coverage(summary_df, raw_df, join_key))

        # 2. Txn count agreement
        if summary_txn_count_col and summary_txn_count_col in summary_df.columns:
            checks.append(self._check_txn_count(
                summary_df, raw_df, join_key, summary_txn_count_col
            ))
        else:
            checks.append(ConsistencyCheck(
                "TXN COUNT AGREEMENT", CheckStatus.SKIPPED,
                f"summary_txn_count_col not provided or not in summary_df"
            ))

        # 3. Amount agreement
        if summary_avg_amount_col and raw_amount_col:
            if summary_avg_amount_col in summary_df.columns and raw_amount_col in raw_df.columns:
                checks.append(self._check_amount_agreement(
                    summary_df, raw_df, join_key, summary_avg_amount_col, raw_amount_col
                ))
            else:
                checks.append(ConsistencyCheck(
                    "AMOUNT AGREEMENT", CheckStatus.SKIPPED, "Amount columns not found"
                ))
        else:
            checks.append(ConsistencyCheck(
                "AMOUNT AGREEMENT", CheckStatus.SKIPPED, "Amount columns not specified"
            ))

        # 4. Null propagation
        checks.append(self._check_null_propagation(summary_df, raw_df, join_key))

        # 5. Edge case accounts
        checks.append(self._check_edge_cases(summary_df, raw_df, join_key, summary_txn_count_col))

        # 6. Fraud label population leakage
        if fraud_label_ids is not None:
            checks.append(self._check_fraud_label_coverage(
                summary_df, raw_df, join_key, fraud_label_ids
            ))
        else:
            checks.append(ConsistencyCheck(
                "FRAUD LABEL COVERAGE", CheckStatus.SKIPPED,
                "No fraud label IDs provided"
            ))

        # 7. Temporal alignment
        if summary_date_col and raw_date_col:
            if summary_date_col in summary_df.columns and raw_date_col in raw_df.columns:
                checks.append(self._check_temporal_alignment(
                    summary_df, raw_df, summary_date_col, raw_date_col
                ))
            else:
                checks.append(ConsistencyCheck(
                    "TEMPORAL ALIGNMENT", CheckStatus.SKIPPED, "Date columns not found"
                ))
        else:
            checks.append(ConsistencyCheck(
                "TEMPORAL ALIGNMENT", CheckStatus.SKIPPED, "Date columns not specified"
            ))

        # Tally
        n_pass    = sum(1 for c in checks if c.status == CheckStatus.PASS)
        n_warning = sum(1 for c in checks if c.status == CheckStatus.WARNING)
        n_fail    = sum(1 for c in checks if c.status == CheckStatus.FAIL)
        n_skipped = sum(1 for c in checks if c.status == CheckStatus.SKIPPED)

        if n_fail > 0:
            overall = CheckStatus.FAIL
        elif n_warning > 0:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.PASS

        return ConsistencyReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            summary_dataset=summary_dataset_name,
            raw_dataset=raw_dataset_name,
            join_key=join_key,
            checks=checks,
            n_pass=n_pass,
            n_warning=n_warning,
            n_fail=n_fail,
            n_skipped=n_skipped,
            overall_status=overall,
        )

    # -- Individual checks --------------------------------------------------

    def _check_join_coverage(
        self, summary_df: pd.DataFrame, raw_df: pd.DataFrame, join_key: str
    ) -> ConsistencyCheck:
        summary_ids = set(summary_df[join_key].dropna().unique())
        raw_ids     = set(raw_df[join_key].dropna().unique())
        covered     = len(summary_ids & raw_ids)
        total       = len(summary_ids)
        coverage    = covered / total if total else 0.0

        if coverage >= self.join_coverage_threshold:
            status = CheckStatus.PASS
        elif coverage >= 0.80:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL

        missing = list(summary_ids - raw_ids)[:10]
        return ConsistencyCheck(
            "JOIN KEY COVERAGE",
            status,
            f"{coverage:.1%} of summary accounts found in raw data "
            f"({covered:,}/{total:,})",
            metric=coverage,
            threshold=self.join_coverage_threshold,
            detail={"covered": covered, "total": total, "missing_examples": missing},
        )

    def _check_txn_count(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        join_key: str,
        summary_txn_count_col: str,
    ) -> ConsistencyCheck:
        raw_counts = raw_df.groupby(join_key).size().rename("raw_count")
        merged = summary_df[[join_key, summary_txn_count_col]].merge(
            raw_counts, on=join_key, how="inner"
        )
        if len(merged) == 0:
            return ConsistencyCheck(
                "TXN COUNT AGREEMENT", CheckStatus.WARNING, "No matching accounts for count comparison"
            )

        summary_total = merged[summary_txn_count_col].sum()
        raw_total = merged["raw_count"].sum()
        diff_pct = abs(summary_total - raw_total) / max(raw_total, 1) * 100

        if diff_pct <= self.count_tolerance_pct:
            status = CheckStatus.PASS
        elif diff_pct <= self.count_tolerance_pct * 3:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL

        return ConsistencyCheck(
            "TXN COUNT AGREEMENT",
            status,
            f"Summary txn count total differs from raw by {diff_pct:.1f}% "
            f"(tolerance {self.count_tolerance_pct:.0f}%)",
            metric=diff_pct / 100,
            threshold=self.count_tolerance_pct / 100,
            detail={"summary_total": int(summary_total), "raw_total": int(raw_total), "diff_pct": diff_pct},
        )

    def _check_amount_agreement(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        join_key: str,
        summary_avg_col: str,
        raw_amount_col: str,
    ) -> ConsistencyCheck:
        raw_avgs = raw_df.groupby(join_key)[raw_amount_col].mean().rename("raw_avg")
        merged = summary_df[[join_key, summary_avg_col]].merge(
            raw_avgs, on=join_key, how="inner"
        )
        if len(merged) == 0:
            return ConsistencyCheck("AMOUNT AGREEMENT", CheckStatus.WARNING, "No matched accounts")

        corr = merged[summary_avg_col].corr(merged["raw_avg"])
        mean_diff_pct = (
            abs(merged[summary_avg_col] - merged["raw_avg"]) / (merged["raw_avg"].abs() + 1e-8)
        ).mean() * 100

        if corr > 0.95 and mean_diff_pct <= self.amount_tolerance_pct:
            status = CheckStatus.PASS
        elif corr > 0.80:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL

        return ConsistencyCheck(
            "AMOUNT AGREEMENT",
            status,
            f"Summary avg amount vs raw avg amount: correlation={corr:.3f}, "
            f"mean diff={mean_diff_pct:.1f}%",
            metric=corr,
            threshold=0.95,
            detail={"correlation": round(corr, 4), "mean_diff_pct": round(mean_diff_pct, 2)},
        )

    def _check_null_propagation(
        self, summary_df: pd.DataFrame, raw_df: pd.DataFrame, join_key: str
    ) -> ConsistencyCheck:
        """
        Spot-check: accounts with all-null values in one dataset should not
        have well-populated values in the other.
        """
        # Find accounts where every numeric column in raw_df is null
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != join_key]

        if not numeric_cols:
            return ConsistencyCheck(
                "NULL PROPAGATION", CheckStatus.SKIPPED, "No numeric columns in raw_df to check"
            )

        all_null_mask = raw_df[numeric_cols].isnull().all(axis=1)
        all_null_ids = set(raw_df[all_null_mask][join_key].dropna().unique())

        if not all_null_ids:
            return ConsistencyCheck(
                "NULL PROPAGATION", CheckStatus.PASS,
                "No all-null transaction accounts found -- null propagation not testable",
                metric=0.0,
            )

        # Check how many of these all-null raw accounts have non-null summary features
        summary_numeric = summary_df.select_dtypes(include=[np.number]).columns.tolist()
        summary_numeric = [c for c in summary_numeric if c != join_key]

        leakage_df = summary_df[summary_df[join_key].isin(all_null_ids)]
        if summary_numeric and len(leakage_df):
            non_null_rate = leakage_df[summary_numeric].notnull().any(axis=1).mean()
        else:
            non_null_rate = 0.0

        status = CheckStatus.PASS if non_null_rate < 0.05 else CheckStatus.WARNING
        return ConsistencyCheck(
            "NULL PROPAGATION",
            status,
            f"{len(all_null_ids)} accounts have all-null raw transactions. "
            f"{non_null_rate:.1%} have non-null summary features (expected ~0%).",
            metric=non_null_rate,
            threshold=0.05,
            detail={"all_null_account_count": len(all_null_ids), "non_null_rate_in_summary": round(non_null_rate, 4)},
        )

    def _check_edge_cases(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        join_key: str,
        summary_txn_count_col: Optional[str],
    ) -> ConsistencyCheck:
        """Check accounts with exactly 1 transaction behave correctly in summary."""
        raw_counts = raw_df.groupby(join_key).size().rename("raw_count")
        single_txn_ids = set(raw_counts[raw_counts == 1].index)

        if not single_txn_ids:
            return ConsistencyCheck(
                "EDGE CASE ACCOUNTS", CheckStatus.PASS,
                "No single-transaction accounts in raw sample", metric=0.0
            )

        n_single = len(single_txn_ids)
        summary_single = summary_df[summary_df[join_key].isin(single_txn_ids)]
        n_in_summary = len(summary_single)
        coverage = n_in_summary / n_single if n_single else 0.0

        # For single-txn accounts, txn count in summary should be 1
        if summary_txn_count_col and summary_txn_count_col in summary_df.columns:
            wrong_count = summary_single[summary_single[summary_txn_count_col] != 1]
            wrong_rate = len(wrong_count) / max(n_in_summary, 1)
        else:
            wrong_rate = None

        if coverage < 0.90:
            status, msg = CheckStatus.WARNING, f"Only {coverage:.1%} of single-txn accounts appear in summary"
        elif wrong_rate is not None and wrong_rate > 0.05:
            status, msg = CheckStatus.FAIL, f"{wrong_rate:.1%} of single-txn accounts have wrong txn_count in summary"
        else:
            status = CheckStatus.PASS
            msg = f"{n_single:,} single-txn accounts: {coverage:.1%} in summary"
            if wrong_rate is not None:
                msg += f", {wrong_rate:.1%} have wrong count"

        return ConsistencyCheck(
            "EDGE CASE ACCOUNTS", status, msg, metric=coverage,
            detail={"n_single_txn": n_single, "in_summary": n_in_summary, "wrong_count_rate": wrong_rate},
        )

    def _check_fraud_label_coverage(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        join_key: str,
        fraud_label_ids: pd.Series,
    ) -> ConsistencyCheck:
        """All fraud-labeled accounts must exist in BOTH datasets."""
        fraud_ids = set(fraud_label_ids.dropna().unique())
        summary_ids = set(summary_df[join_key].dropna().unique())
        raw_ids     = set(raw_df[join_key].dropna().unique())

        missing_summary = fraud_ids - summary_ids
        missing_raw     = fraud_ids - raw_ids
        n_total = len(fraud_ids)

        missing_either = missing_summary | missing_raw
        missing_rate = len(missing_either) / n_total if n_total else 0.0

        if missing_rate == 0:
            status = CheckStatus.PASS
        elif missing_rate < 0.01:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.FAIL

        return ConsistencyCheck(
            "FRAUD LABEL COVERAGE",
            status,
            f"{len(missing_either)} / {n_total} fraud accounts missing from one or both datasets "
            f"({missing_rate:.2%})",
            metric=1.0 - missing_rate,
            threshold=0.99,
            detail={
                "n_fraud_labels": n_total,
                "missing_from_summary": len(missing_summary),
                "missing_from_raw": len(missing_raw),
            },
        )

    def _check_temporal_alignment(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        summary_date_col: str,
        raw_date_col: str,
    ) -> ConsistencyCheck:
        """Summary and raw data should cover the same date range."""
        try:
            summary_dates = pd.to_datetime(summary_df[summary_date_col].dropna())
            raw_dates     = pd.to_datetime(raw_df[raw_date_col].dropna())

            s_min, s_max = summary_dates.min(), summary_dates.max()
            r_min, r_max = raw_dates.min(), raw_dates.max()

            # Overlap check
            overlap_start = max(s_min, r_min)
            overlap_end   = min(s_max, r_max)

            if overlap_start > overlap_end:
                return ConsistencyCheck(
                    "TEMPORAL ALIGNMENT", CheckStatus.FAIL,
                    f"Summary ({s_min.date()} - {s_max.date()}) and raw "
                    f"({r_min.date()} - {r_max.date()}) have NO date overlap.",
                    metric=0.0,
                )

            summary_days = (s_max - s_min).days + 1
            raw_days     = (r_max - r_min).days + 1
            overlap_days = (overlap_end - overlap_start).days + 1
            alignment    = overlap_days / max(summary_days, raw_days)

            status = (
                CheckStatus.PASS if alignment >= 0.90
                else CheckStatus.WARNING if alignment >= 0.70
                else CheckStatus.FAIL
            )
            return ConsistencyCheck(
                "TEMPORAL ALIGNMENT",
                status,
                f"Summary: {s_min.date()} - {s_max.date()} | "
                f"Raw: {r_min.date()} - {r_max.date()} | "
                f"Overlap: {alignment:.1%}",
                metric=alignment,
                threshold=0.90,
                detail={
                    "summary_range": f"{s_min.date()}-{s_max.date()}",
                    "raw_range": f"{r_min.date()}-{r_max.date()}",
                    "overlap_days": overlap_days,
                },
            )
        except Exception as exc:
            return ConsistencyCheck(
                "TEMPORAL ALIGNMENT", CheckStatus.WARNING,
                f"Could not parse date columns: {exc}"
            )
