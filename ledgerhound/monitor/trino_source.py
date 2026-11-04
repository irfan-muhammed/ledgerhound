"""
ledgerhound.monitor.trino_source
==============================

Pulls the Drift Monitor's fraud / non-fraud / raw-txn slices directly
from Trino instead of static parquet/csv files -- no manual export step,
samples straight from the warehouse on every scheduled cycle.

Safety model, different from the interactive TrinoClient used in the chat
shell (``ledgerhound.trino.client``): DriftMonitor runs unattended
(cron/daemon), so there is no human present to approve each query the
way ``TrinoClient._hitl_approve()`` requires there. Instead, this module
uses a PRE-APPROVED, HASH-PINNED SQL model: the SQL text is reviewed and
signed off by a human ONCE (``ledgerhound monitor approve-sql``, which
records who approved it and when). Every unattended cycle re-runs that
exact, unchanged SQL -- if the SQL is ever edited after approval, its
hash no longer matches and the cycle refuses to run until a human
re-approves it. This mirrors the "reviewed once, then trusted" shape
already used for the V3-029 package gate, applied to unattended queries
instead of packages.

A hard row cap (``max_rows``) is always enforced via a wrapping
subquery, as a second safety net independent of what the approved SQL
itself contains -- a scheduled job can never accidentally pull an
unbounded table even if the approved SQL has no LIMIT of its own.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


class TrinoSourceNotApproved(RuntimeError):
    """Raised when configured SQL doesn't match its recorded approval hash."""


def sql_hash(sql: str) -> str:
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


_SQL_FIELDS = ("fraud_sql", "nonfr_sql", "raw_txn_sql")


@dataclass
class TrinoSourceConfig:
    """Optional Trino-backed data source for one DriftMonitorConfig.

    When ``enabled`` is False (default), the monitor loads from
    ``fraud_data``/``nonfr_data``/``raw_txn_data`` file paths exactly as
    before -- this is purely additive.
    """

    enabled: bool = False
    fraud_sql: str = ""
    nonfr_sql: str = ""
    raw_txn_sql: str = ""
    max_rows: int = 2_000_000

    # Populated only by approve(); never hand-edit these.
    approved_sql_sha256: dict[str, str] = field(default_factory=dict)
    approved_by: str = ""
    approved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TrinoSourceConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def approve(self, approved_by: str) -> None:
        """Record the CURRENT sql text's hash as approved.

        Call this explicitly (CLI: ``ledgerhound monitor approve-sql``) after
        a human has actually read the SQL. Never call this automatically
        from a code path that also generates or edits the SQL.
        """
        self.approved_sql_sha256 = {
            f: sql_hash(getattr(self, f)) for f in _SQL_FIELDS if getattr(self, f)
        }
        self.approved_by = approved_by
        self.approved_at = datetime.now(timezone.utc).isoformat()

    def check_approved(self) -> list[str]:
        """Return problems; empty list = every configured SQL field matches
        its recorded approval hash (or is unset, which is fine)."""
        problems: list[str] = []
        for f in _SQL_FIELDS:
            sql = getattr(self, f)
            if not sql:
                continue
            approved_hash = self.approved_sql_sha256.get(f)
            if approved_hash is None:
                problems.append(
                    f"trino.{f} has never been approved -- run `ledgerhound monitor approve-sql`"
                )
            elif approved_hash != sql_hash(sql):
                problems.append(
                    f"trino.{f} has changed since it was approved -- "
                    f"re-run `ledgerhound monitor approve-sql` after review"
                )
        return problems

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.enabled:
            return issues
        if not self.fraud_sql:
            issues.append("trino.fraud_sql is required when trino.enabled=true")
        if not self.nonfr_sql:
            issues.append("trino.nonfr_sql is required when trino.enabled=true")
        if self.max_rows <= 0:
            issues.append("trino.max_rows must be positive")
        issues.extend(self.check_approved())
        return issues


def _capped(sql: str, max_rows: int) -> str:
    """Wrap SQL with a hard row cap via a subquery -- applies regardless
    of whether the inner SQL already has its own LIMIT."""
    stripped = sql.strip().rstrip(";")
    return f"SELECT * FROM (\n{stripped}\n) AS _ledgerhound_monitor_capped LIMIT {int(max_rows)}"


class TrinoMonitorSource:
    """Pulls DataFrames for one DriftMonitor cycle from Trino, using
    pre-approved, hash-pinned SQL and a hard row cap.

    Never prompts -- this is the unattended path. ``TrinoClient``'s
    interactive HITL gate is for the chat shell, where a human is
    present; a scheduled cycle has nobody to answer a y/N prompt.
    """

    def __init__(self, trino_config, source_config: TrinoSourceConfig) -> None:
        self.trino_config = trino_config
        self.source_config = source_config

    def load(self):
        """Returns (fraud_df, nonfr_df, raw_txn_df_or_None).

        Raises TrinoSourceNotApproved if any configured SQL's hash
        doesn't match what was last approved -- this fires BEFORE any
        network connection is attempted.
        """
        problems = self.source_config.check_approved()
        if problems:
            raise TrinoSourceNotApproved("; ".join(problems))

        import pandas as pd

        if not self.trino_config.use_trino:
            log.info("TrinoConfig.use_trino=False -- TrinoMonitorSource returning empty DataFrames")
            return pd.DataFrame(), pd.DataFrame(), None

        import trino
        conn = trino.dbapi.connect(**self.trino_config.to_connection_kwargs())

        def _run(sql: str) -> "pd.DataFrame":
            capped = _capped(sql, self.source_config.max_rows)
            cur = conn.cursor()
            cur.execute(capped)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)

        fraud_df = _run(self.source_config.fraud_sql)
        nonfr_df = _run(self.source_config.nonfr_sql)
        raw_txn_df = _run(self.source_config.raw_txn_sql) if self.source_config.raw_txn_sql else None
        return fraud_df, nonfr_df, raw_txn_df
