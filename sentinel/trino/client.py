"""
sentinel.trino.client
=====================

TrinoClient -- the gatekeeper for all Trino interaction.

Three hard rules enforced here:

1. HITL before every execution
   Every query is shown to the user in full. The estimated row count and
   runtime are shown. The user must type 'y' to proceed. No query ever
   executes automatically.

2. COUNT estimation before large pulls
   Before any row-returning query, a COUNT(*) variant runs first to
   estimate rows and runtime. The user sees this before approving.

3. use_trino toggle
   When TrinoConfig.use_trino is False, all methods return empty/mock
   results immediately. The agent can run in local-only mode without
   any code changes.
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import Optional, Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from sentinel.trino.config import TrinoConfig
from sentinel.trino.jobs import AsyncJobRegistry, TrinoJob
from sentinel.workspace.layout import WorkspaceLayout

log = logging.getLogger(__name__)
console = Console()

# Rows above this threshold trigger a warning and mandatory COUNT estimation
_LARGE_QUERY_ROW_WARNING = 10_000_000


class TrinoDisabledError(RuntimeError):
    """Raised when a Trino operation is attempted with use_trino=False."""


class TrinoQueryDenied(RuntimeError):
    """Raised when the user denies a query in the HITL prompt."""


class TrinoClient:
    def __init__(
        self,
        config: TrinoConfig,
        workspace: WorkspaceLayout,
        job_registry: Optional[AsyncJobRegistry] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.session_id = session_id or "untracked"
        self._job_registry = job_registry or AsyncJobRegistry(
            workspace.trino_jobs_dir / "registry.json"
        )

    # -- Public API ---------------------------------------------------------

    def execute_stats_only(self, sql: str, context: str = "") -> dict[str, Any]:
        """
        Run an aggregation query and return results as a dict.

        Intended for population profiling: SELECT feature, COUNT(*), AVG()...
        Returns kilobytes of statistics, not rows. No parquet written.
        Uses HITL -- user must approve before execution.
        """
        if not self.config.use_trino:
            log.info("use_trino=False -- returning empty stats.")
            return {}

        self._check_config()
        self._show_sql_panel(sql, context, query_type="STATS QUERY (aggregation only -- no rows returned)")

        if not self._hitl_approve():
            raise TrinoQueryDenied("User denied the stats query.")

        log.info("Executing stats query...")
        start = time.time()
        rows = self._run_query(sql)
        elapsed = time.time() - start
        log.info(f"Stats query completed in {elapsed:.1f}s -- {len(rows)} result rows.")

        return {"rows": rows, "elapsed_seconds": elapsed, "sql": sql}

    def submit_async(
        self,
        sql: str,
        context: str = "",
        result_filename: str = "result.parquet",
    ) -> TrinoJob:
        """
        HITL-gated async job submission.

        Shows SQL + COUNT estimate to user. On approval, registers the job
        and returns a TrinoJob the caller can track. The actual execution
        is the caller's responsibility (run execute_job(job) when ready).

        This separation lets the agent checkpoint state before the long
        wait begins.
        """
        if not self.config.use_trino:
            log.info("use_trino=False -- returning mock job.")
            return self._mock_job(sql, context)

        self._check_config()

        # Cost estimation itself runs a COUNT(*) over the (possibly heavy)
        # query against PRODUCTION Trino. That is a real prod touch, so it
        # must be consented to -- previously it ran before any approval, so
        # a user who ultimately DENIED the query had still triggered a full
        # COUNT scan they never agreed to. Show the SQL first and gate the
        # estimate.
        estimated_rows = None
        est_min = est_max = None
        self._show_sql_panel(sql, context, query_type="ROW-RETURNING QUERY")
        if self._hitl_approve(
                prompt="Run a COUNT(*) cost estimate against production? "
                       "(this executes a scan) [y/N]"):
            estimated_rows = self._count_estimate(sql)
            est_min, est_max = self._estimate_runtime(estimated_rows)
            self._show_sql_panel(
                sql, context, query_type="ROW-RETURNING QUERY",
                estimated_rows=estimated_rows, est_min=est_min, est_max=est_max)

        if not self._hitl_approve(
                prompt="Approve and register this query for execution? [y/N]"):
            raise TrinoQueryDenied("User denied the query.")

        # Register and persist the job
        job_dir = None
        job = self._job_registry.submit(
            sql=sql,
            session_id=self.session_id,
            dataset_context=context,
            estimated_rows=estimated_rows,
            estimated_seconds_min=est_min,
            estimated_seconds_max=est_max,
            workspace=self.workspace,
        )

        console.print(f"[green]Job registered:[/green] [bold]{job.job_id}[/bold]")
        console.print(f"[dim]Resume with: sentinel resume --job {job.job_id}[/dim]")
        return job

    def execute_job(self, job: TrinoJob) -> Path:
        """
        Execute a previously submitted and approved job.
        Returns the path to the result parquet file.
        """
        if not self.config.use_trino:
            return Path("/dev/null")

        self._job_registry.update_status(job.job_id, "running")
        result_dir = self.workspace.trino_job_result_dir(job.job_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "result.parquet"

        start = time.time()
        try:
            import pandas as pd
            rows = self._run_query(job.sql)
            df = pd.DataFrame(rows)
            df.to_parquet(result_path, index=False)
            elapsed = time.time() - start

            self._job_registry.update_status(
                job.job_id,
                "done",
                result_path=str(result_path.relative_to(self.workspace.root)),
                actual_runtime_seconds=elapsed,
            )
            console.print(
                f"[green]Job {job.job_id} complete[/green] -- "
                f"{len(df):,} rows in {elapsed:.0f}s -> {result_path}"
            )
            return result_path

        except Exception as exc:
            self._job_registry.update_status(job.job_id, "failed", error=str(exc))
            raise

    def cancel_job(self, job_id: str) -> None:
        self._job_registry.update_status(job_id, "cancelled")
        console.print(f"[yellow]Job {job_id} cancelled.[/yellow]")

    # -- COUNT estimation ---------------------------------------------------

    def _count_estimate(self, sql: str) -> Optional[int]:
        """
        Wrap the query in SELECT COUNT(*) FROM (...) and run it.
        Returns None if the wrapping fails (e.g. complex CTEs).
        """
        count_sql = _wrap_as_count(sql)
        if count_sql is None:
            return None
        try:
            console.print("[dim]Estimating row count...[/dim]")
            rows = self._run_query(count_sql)
            if rows and rows[0]:
                return int(list(rows[0].values())[0])
        except Exception as exc:
            log.warning(f"COUNT estimation failed: {exc}")
        return None

    def _estimate_runtime(self, row_count: Optional[int]) -> tuple[int, int]:
        """
        Return (min_seconds, max_seconds) based on row count and cluster calibration.
        Uses a 2x safety band around the throughput estimate.
        """
        if row_count is None:
            return (60, 3600)
        rps = self.config.rows_per_second_estimate
        mid = max(30, row_count // rps)
        return (max(30, mid // 2), mid * 2)

    # -- HITL display -------------------------------------------------------

    def _show_sql_panel(
        self,
        sql: str,
        context: str,
        query_type: str = "QUERY",
        estimated_rows: Optional[int] = None,
        est_min: Optional[int] = None,
        est_max: Optional[int] = None,
    ) -> None:
        console.print()
        console.rule(f"[bold yellow]TRINO {query_type}[/bold yellow]")

        if context:
            console.print(f"[dim]Context:[/dim] {context}")

        console.print(Syntax(sql.strip(), "sql", theme="monokai", line_numbers=True))

        if estimated_rows is not None:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_row("[dim]Estimated rows[/dim]", f"[bold]{estimated_rows:,}[/bold]")
            if est_min and est_max:
                table.add_row(
                    "[dim]Estimated runtime[/dim]",
                    f"[bold]{_fmt_seconds(est_min)} - {_fmt_seconds(est_max)}[/bold]",
                )
            if estimated_rows > _LARGE_QUERY_ROW_WARNING:
                table.add_row(
                    "[dim]Warning[/dim]",
                    f"[bold red]Large query -- review SQL carefully before approving[/bold red]",
                )
            console.print(table)

        console.rule()

    def _hitl_approve(self, prompt: str | None = None) -> bool:
        """Prompt user for explicit approval. Returns True if approved."""
        msg = prompt or "Approve and execute this Trino query? [y/N/cancel]"
        try:
            answer = console.input(
                f"[bold yellow]{msg}: [/bold yellow]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # -- Internal execution -------------------------------------------------

    def _run_query(self, sql: str) -> list[dict]:
        """Execute SQL against Trino and return list of row dicts."""
        import trino
        conn = trino.dbapi.connect(**self.config.to_connection_kwargs())
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _check_config(self) -> None:
        problems = self.config.validate()
        if problems:
            raise ValueError(f"TrinoConfig invalid: {'; '.join(problems)}")

    def _mock_job(self, sql: str, context: str) -> "TrinoJob":
        from sentinel.trino.jobs import TrinoJob
        return TrinoJob(
            job_id="mock_disabled",
            sql=sql,
            session_id=self.session_id,
            dataset_context=context,
            status="cancelled",
            approved_by_user=False,
        )


# -- Helpers ----------------------------------------------------------------


def _wrap_as_count(sql: str) -> Optional[str]:
    """Wrap a SELECT query as SELECT COUNT(*) FROM (...) alias."""
    stripped = sql.strip().rstrip(";")
    # Avoid wrapping if it already is a COUNT or is a DDL/DML
    upper = stripped.upper()
    if upper.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")):
        return None
    if re.match(r"^\s*SELECT\s+COUNT\s*\(", stripped, re.IGNORECASE):
        return None
    return f"SELECT COUNT(*) AS _row_count FROM (\n{stripped}\n) AS _sentinel_count"


def _fmt_seconds(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"
