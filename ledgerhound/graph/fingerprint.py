"""
ledgerhound.graph.fingerprint
==========================

Per-account behavioral fingerprints for mule-likelihood scoring.

These are pure DataFrame computations -- no graph required, but we live in
the graph package because the consumer is the same agent loop.  A mule
account's behavioral signature differs from a normal account in:

- **Time-of-day distribution** -- mules are bursty, often nocturnal
- **Inter-arrival gaps** -- mules show short, consistent gaps; normals are
  long-tailed, irregular
- **Amount patterns** -- mules use round-number amounts; normals don't
- **Beneficiary diversity** -- mules send to many distinct beneficiaries
- **Device diversity** -- mules cycle through devices

We compute a 0..1 ``mule_likelihood`` score from a weighted combination.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Dataclasses ------------------------------------------------------------


@dataclass
class AccountFingerprint:
    """Behavioral fingerprint for one account."""

    account_id: str
    n_txns: int

    # Time signals
    night_ratio: float = 0.0          # 22:00-05:00 UTC fraction
    weekend_ratio: float = 0.0
    median_gap_minutes: float = 0.0
    burst_max_5min: int = 0
    inter_arrival_cv: float = 0.0     # CV of gap times -- low CV -> robotic

    # Amount signals
    amount_median: float = 0.0
    amount_skew: float = 0.0
    round_amount_ratio: float = 0.0   # fraction with amount % 100 == 0
    amount_entropy: float = 0.0       # Shannon entropy over amount buckets

    # Diversity signals
    unique_beneficiaries: int = 0
    unique_devices: int = 0
    beneficiary_concentration: float = 0.0   # max single-benef share

    # Composite
    burstiness_score: float = 0.0
    timing_anomaly: float = 0.0
    mule_likelihood: float = 0.0
    severity: str = "INFO"            # INFO | WARNING | ALERT

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("night_ratio", "weekend_ratio", "median_gap_minutes",
                  "inter_arrival_cv", "amount_median", "amount_skew",
                  "round_amount_ratio", "amount_entropy",
                  "beneficiary_concentration", "burstiness_score",
                  "timing_anomaly", "mule_likelihood"):
            v = d.get(k)
            if v is not None and isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class FingerprintResult:
    """Output of BehavioralFingerprinter.fingerprint_all()."""

    period: str
    generated_at: str
    n_accounts: int
    fingerprints: list[AccountFingerprint] = field(default_factory=list)
    n_alerts: int = 0
    n_warnings: int = 0
    elapsed_ms: float = 0.0
    notes: str = ""

    # -- Convenience -------------------------------------------------------

    def alerts(self) -> list[AccountFingerprint]:
        return [f for f in self.fingerprints if f.severity == "ALERT"]

    def top_by_mule_likelihood(self, n: int = 25) -> list[AccountFingerprint]:
        return sorted(self.fingerprints, key=lambda f: -f.mule_likelihood)[:n]

    def to_dict(self) -> dict:
        return {
            "period":       self.period,
            "generated_at": self.generated_at,
            "n_accounts":   self.n_accounts,
            "n_alerts":     self.n_alerts,
            "n_warnings":   self.n_warnings,
            "fingerprints": [f.to_dict() for f in self.fingerprints],
            "elapsed_ms":   round(self.elapsed_ms, 1),
            "notes":        self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def display(self, top_n: int = 15) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        c = Console()
        c.print(Rule(f"[bold]Behavioral Fingerprints -- {self.period}[/bold]"))
        c.print(
            f"  Accounts     : {self.n_accounts:,}\n"
            f"  Suspicious   : "
            f"[red]{self.n_alerts} ALERT[/red]  "
            f"[yellow]{self.n_warnings} WARNING[/yellow]"
        )
        if not self.fingerprints:
            return
        sev = {"ALERT": "red", "WARNING": "yellow", "INFO": "dim"}
        t = Table(
            "Account", "Sev", "Likelihood",
            "Txns", "Night%", "Median gap (m)",
            "Round %", "Burst5", "Unique benef",
            title=f"Top {top_n} by mule likelihood",
            show_header=True, header_style="bold dim",
        )
        for f in self.top_by_mule_likelihood(top_n):
            sc = sev.get(f.severity, "white")
            t.add_row(
                f.account_id,
                f"[{sc}]{f.severity}[/{sc}]",
                f"{f.mule_likelihood:.3f}",
                str(f.n_txns),
                f"{f.night_ratio:.0%}",
                f"{f.median_gap_minutes:.1f}",
                f"{f.round_amount_ratio:.0%}",
                str(f.burst_max_5min),
                str(f.unique_beneficiaries),
            )
        c.print(t)
        c.print()


# -- Fingerprinter ----------------------------------------------------------


class BehavioralFingerprinter:
    """
    Compute per-account behavioral fingerprints from a transaction DataFrame.

    Required columns
    ----------------
    account_col, beneficiary_col, amount_col, timestamp_col

    Optional columns
    ----------------
    device_col, status_col

    Parameters
    ----------
    night_start, night_end : int
        Hour-of-day bounds for "night" txns (default 22..5).
    burst_window_minutes : float
        Window for burst-count metric (default 5.0).
    min_txns : int
        Drop accounts with fewer than this many txns (default 5).
    alert_threshold, warning_threshold : float
        mule_likelihood thresholds for severity (default 0.70, 0.50).
    """

    def __init__(
        self,
        account_col: str = "account_id",
        beneficiary_col: str = "beneficiary_id",
        amount_col: str = "txn_amount",
        timestamp_col: str = "txn_date",
        device_col: Optional[str] = "device_id",
        status_col: Optional[str] = "status",
        success_only: bool = True,
        night_start: int = 22,
        night_end: int = 5,
        burst_window_minutes: float = 5.0,
        min_txns: int = 5,
        alert_threshold: float = 0.70,
        warning_threshold: float = 0.50,
    ) -> None:
        self.account_col          = account_col
        self.beneficiary_col      = beneficiary_col
        self.amount_col           = amount_col
        self.timestamp_col        = timestamp_col
        self.device_col           = device_col
        self.status_col           = status_col
        self.success_only         = bool(success_only)
        self.night_start          = int(night_start)
        self.night_end            = int(night_end)
        self.burst_window_minutes = float(burst_window_minutes)
        self.min_txns             = int(min_txns)
        self.alert_threshold      = float(alert_threshold)
        self.warning_threshold    = float(warning_threshold)

    # -- Main API ----------------------------------------------------------

    def fingerprint_all(
        self,
        txn_df: pd.DataFrame,
        period: str = "current",
    ) -> FingerprintResult:
        t0 = time.perf_counter()
        df = self._prepare(txn_df)
        if df.empty:
            return FingerprintResult(
                period=period,
                generated_at=_now(),
                n_accounts=0,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

        fps: list[AccountFingerprint] = []
        for account_id, group in df.groupby(self.account_col, sort=False):
            if len(group) < self.min_txns:
                continue
            fp = self._fingerprint_one(account_id, group)
            fps.append(fp)

        # Severity assignment
        n_alerts = 0
        n_warnings = 0
        for fp in fps:
            if fp.mule_likelihood >= self.alert_threshold:
                fp.severity = "ALERT"
                n_alerts += 1
            elif fp.mule_likelihood >= self.warning_threshold:
                fp.severity = "WARNING"
                n_warnings += 1
            else:
                fp.severity = "INFO"

        elapsed = (time.perf_counter() - t0) * 1000
        return FingerprintResult(
            period=period,
            generated_at=_now(),
            n_accounts=len(fps),
            fingerprints=fps,
            n_alerts=n_alerts,
            n_warnings=n_warnings,
            elapsed_ms=round(elapsed, 1),
        )

    def fingerprint_account(
        self,
        txn_df: pd.DataFrame,
        account_id: str,
    ) -> Optional[AccountFingerprint]:
        df = self._prepare(txn_df)
        sub = df[df[self.account_col] == account_id]
        if len(sub) < self.min_txns:
            return None
        fp = self._fingerprint_one(account_id, sub)
        if fp.mule_likelihood >= self.alert_threshold:
            fp.severity = "ALERT"
        elif fp.mule_likelihood >= self.warning_threshold:
            fp.severity = "WARNING"
        return fp

    # -- Per-account computation -------------------------------------------

    def _fingerprint_one(
        self,
        account_id: str,
        group: pd.DataFrame,
    ) -> AccountFingerprint:
        ts = group[self.timestamp_col]
        amt = group[self.amount_col].astype(float)
        n = len(group)

        # Time-of-day
        hours = ts.dt.hour
        if self.night_start <= self.night_end:
            night_mask = hours.between(self.night_start, self.night_end)
        else:
            night_mask = (hours >= self.night_start) | (hours <= self.night_end)
        night_ratio = float(night_mask.mean())
        weekend_ratio = float((ts.dt.dayofweek >= 5).mean())

        # Inter-arrival
        sorted_ts = ts.sort_values()
        gaps = sorted_ts.diff().dt.total_seconds().dropna() / 60.0
        median_gap = float(gaps.median()) if len(gaps) else 0.0
        gap_cv = (
            float(gaps.std() / gaps.mean()) if len(gaps) > 1 and gaps.mean() > 0
            else 0.0
        )

        # Burst count in 5-minute windows
        burst_max = 0
        if len(sorted_ts) > 1:
            burst_df = pd.DataFrame({"ts": sorted_ts.values}).set_index("ts")
            try:
                burst_max = int(
                    burst_df.assign(c=1)["c"]
                    .rolling(pd.Timedelta(minutes=self.burst_window_minutes), closed="both")
                    .count().max()
                )
            except Exception:
                burst_max = 0

        # Amount statistics
        amount_median = float(amt.median()) if n else 0.0
        # Skew
        if n >= 3 and amt.std() > 0:
            amount_skew = float(((amt - amt.mean()) ** 3).mean() / (amt.std() ** 3))
        else:
            amount_skew = 0.0
        # Round-amount ratio
        rounded = (amt % 100 == 0).sum()
        round_ratio = float(rounded / n) if n else 0.0

        # Amount Shannon entropy over log-bucketed amounts
        entropy = self._log_amount_entropy(amt)

        # Beneficiary diversity
        benefs = group[self.beneficiary_col].astype(str)
        unique_benef = int(benefs.nunique())
        if unique_benef > 0:
            top_share = float(benefs.value_counts(normalize=True).iloc[0])
        else:
            top_share = 0.0

        # Device diversity
        unique_devices = 0
        if self.device_col and self.device_col in group.columns:
            unique_devices = int(group[self.device_col].astype(str).nunique())

        # Composite signals
        burstiness_score = self._burstiness_score(median_gap, gap_cv, burst_max)
        timing_anomaly = self._timing_anomaly(night_ratio, weekend_ratio)

        # Mule likelihood -- weighted combination
        diversity_score = min(1.0, unique_benef / 30.0)
        round_score = round_ratio  # already 0..1
        mule_likelihood = round(
            0.30 * burstiness_score
            + 0.20 * round_score
            + 0.20 * diversity_score
            + 0.15 * timing_anomaly
            + 0.15 * (1.0 - top_share),  # diffuse beneficiary pattern
            4,
        )

        return AccountFingerprint(
            account_id=str(account_id),
            n_txns=n,
            night_ratio=night_ratio,
            weekend_ratio=weekend_ratio,
            median_gap_minutes=median_gap,
            burst_max_5min=burst_max,
            inter_arrival_cv=gap_cv,
            amount_median=amount_median,
            amount_skew=amount_skew,
            round_amount_ratio=round_ratio,
            amount_entropy=entropy,
            unique_beneficiaries=unique_benef,
            unique_devices=unique_devices,
            beneficiary_concentration=top_share,
            burstiness_score=burstiness_score,
            timing_anomaly=timing_anomaly,
            mule_likelihood=mule_likelihood,
        )

    # -- Helpers ------------------------------------------------------------

    def _prepare(self, txn_df: pd.DataFrame) -> pd.DataFrame:
        required = [
            self.account_col, self.beneficiary_col,
            self.amount_col, self.timestamp_col,
        ]
        missing = [c for c in required if c not in txn_df.columns]
        if missing:
            raise ValueError(
                f"BehavioralFingerprinter: missing columns {missing}. "
                f"Available: {list(txn_df.columns)[:30]}"
            )

        df = txn_df.copy()
        if self.success_only and self.status_col and self.status_col in df.columns:
            df = df[df[self.status_col].astype(str).str.upper() == "SUCCESS"]
        df = df.dropna(subset=required)
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col], errors="coerce",
            )
        df = df.dropna(subset=[self.timestamp_col])
        return df.reset_index(drop=True)

    @staticmethod
    def _log_amount_entropy(amt: pd.Series, n_buckets: int = 10) -> float:
        if len(amt) == 0:
            return 0.0
        log_amt = np.log10(amt.replace(0, 1))
        try:
            buckets = pd.qcut(log_amt, q=min(n_buckets, len(log_amt.unique())),
                              duplicates="drop")
        except Exception:
            return 0.0
        probs = buckets.value_counts(normalize=True)
        ent = -float(sum(p * math.log2(p) for p in probs if p > 0))
        # Normalise to 0..1 by dividing by log2(n_buckets)
        return ent / math.log2(n_buckets) if n_buckets > 1 else 0.0

    def _burstiness_score(
        self, median_gap: float, gap_cv: float, burst_max: int,
    ) -> float:
        """High when gaps are short AND consistent."""
        if median_gap <= 0:
            return 0.0
        # Short median gap -> suspicious (under 30 min)
        gap_short = max(0.0, 1.0 - median_gap / 30.0)
        # Low CV -> robotic
        cv_score = max(0.0, 1.0 - gap_cv / 1.5) if gap_cv > 0 else 0.0
        # High burst -> suspicious
        burst_score = min(1.0, burst_max / 15.0)
        return round(0.40 * gap_short + 0.30 * cv_score + 0.30 * burst_score, 4)

    def _timing_anomaly(
        self, night_ratio: float, weekend_ratio: float,
    ) -> float:
        """High when an account does most txns at night or weekends."""
        # Night ratio > 0.30 is anomalous (real users mostly sleep)
        n = max(0.0, (night_ratio - 0.20) / 0.50)
        n = min(1.0, n)
        # Weekend ratio > 0.40 is mildly anomalous
        w = max(0.0, (weekend_ratio - 0.30) / 0.30)
        w = min(1.0, w)
        return round(0.70 * n + 0.30 * w, 4)
