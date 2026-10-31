"""
sentinel.profiler.detectors.quality
====================================

Detects data-quality issues: nulls, constants, empty strings,
zero-value amounts, duplicate rows, and join health.

Phase B fixes
-------------
* **Empty column detection upgraded**: if ALL samples are empty string,
  severity is ``"critical"`` and column is flagged for dropping.
  Previously only triggered at 50% empty (severity "warning").
* **Join health check added**: for each schema relationship, samples
  FK values and PK values.  If formats don't match (e.g. hashed FK
  vs plain-text PK), the join is flagged as ``"broken"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sentinel.backends.base import ColumnInfo
from sentinel.schema.models import ColumnSchema, DatasetSchema, Relationship


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


@dataclass
class QualityIssue:
    column: str
    severity: str          # "critical" | "warning" | "info"
    issue: str
    detail: str = ""
    action: str = ""


@dataclass
class JoinHealth:
    """Health status for a single FK -> PK relationship."""
    from_col: str
    to_table: str
    to_col: str
    status: str            # "healthy" | "broken" | "unknown"
    reason: str = ""
    fk_sample: str = ""
    pk_sample: str = ""


@dataclass
class QualityReport:
    total_rows: int = 0
    total_columns: int = 0
    duplicate_rows: int = 0
    empty_columns: list[str] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    join_health: list[JoinHealth] = field(default_factory=list)
    score: int = 100

    @property
    def critical(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def info(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def broken_joins(self) -> list[JoinHealth]:
        return [j for j in self.join_health if j.status == "broken"]


class QualityDetector:
    """Run quality checks and return a scored report."""

    _DUPE_LIMIT = 1_000_000

    def detect(self, backend, columns: dict[str, ColumnSchema],
               schema: DatasetSchema | None = None) -> QualityReport:
        """Run all quality checks.

        Parameters
        ----------
        backend : DataBackend
        columns : classified column schemas
        schema  : full dataset schema (needed for join health check)
        """
        report = QualityReport(
            total_rows=backend.count_rows(),
            total_columns=len(columns),
        )
        col_infos = {ci.name: ci for ci in backend.get_columns()}

        for cname, scol in columns.items():
            ci = col_infos.get(cname)
            if ci is None:
                continue
            self._check_nulls(report, ci)
            self._check_constant(report, ci)
            self._check_empty_column(report, ci, backend)
            self._check_zeros(report, ci, scol, backend)

        self._check_duplicates(report, backend)

        # Join health (requires full schema with relationships)
        if schema and schema.relationships:
            self._check_joins(report, backend, schema)

        self._compute_score(report)
        return report

    # -- individual checks ----------------------------------

    def _check_nulls(self, rpt: QualityReport, ci: ColumnInfo):
        if ci.null_count == 0:
            return
        pct = ci.null_count / max(rpt.total_rows, 1) * 100
        if pct >= 90:
            sev, msg = "critical", f"Almost entirely null ({pct:.1f}%)"
        elif pct >= 30:
            sev, msg = "warning", f"High null rate ({pct:.1f}%)"
        else:
            sev, msg = "info", f"Has nulls ({pct:.1f}%)"
        rpt.issues.append(QualityIssue(
            column=ci.name, severity=sev, issue=msg,
            detail=f"{ci.null_count:,} nulls",
            action="Impute or create missing indicator",
        ))

    def _check_constant(self, rpt: QualityReport, ci: ColumnInfo):
        if ci.unique <= 1 and ci.non_null > 0:
            rpt.issues.append(QualityIssue(
                column=ci.name, severity="warning",
                issue="Constant column (zero variance)",
                detail=f"Only {ci.unique} unique value(s)",
                action="Drop -- provides no information for modeling",
            ))

    def _check_empty_column(self, rpt: QualityReport, ci: ColumnInfo, backend):
        """Detect empty-string columns (different from null).

        Phase B upgrade: if ALL samples are empty string, escalate to
        critical and add to empty_columns list.  The priority scorer
        will route these to drop_candidates.
        """
        if "object" not in str(ci.dtype) and "str" not in str(ci.dtype):
            return

        samples = backend.sample_values(ci.name, 10)
        if not samples:
            return

        empty = sum(1 for s in samples if isinstance(s, str) and s.strip() == "")

        if empty == len(samples):
            # ALL samples empty -- dead column
            rpt.empty_columns.append(ci.name)
            rpt.issues.append(QualityIssue(
                column=ci.name, severity="critical",
                issue="Completely empty (all values are empty string)",
                detail="Every sampled value is ''",
                action="Drop -- dead column, contains no data",
            ))
        elif empty >= len(samples) // 2:
            # >50% empty -- still worth flagging
            rpt.issues.append(QualityIssue(
                column=ci.name, severity="warning",
                issue=f"Mostly empty strings ({empty}/{len(samples)} samples)",
                action="Replace empty strings with null before analysis",
            ))

    def _check_zeros(self, rpt: QualityReport, ci: ColumnInfo,
                     scol: ColumnSchema, backend):
        if scol.role.value != "amount":
            return
        profile = backend.numeric_profile(ci.name)
        if profile.zeros > 0:
            pct = profile.zeros / max(profile.count, 1) * 100
            rpt.issues.append(QualityIssue(
                column=ci.name,
                severity="warning" if pct > 1 else "info",
                issue=f"Zero-value amounts ({pct:.2f}%)",
                detail=f"{profile.zeros:,} zero-amount rows",
                action="Filter -- likely test/reversal transactions",
            ))

    def _check_duplicates(self, rpt: QualityReport, backend):
        if rpt.total_rows > self._DUPE_LIMIT:
            return
        try:
            dupes = int(backend.df.duplicated().sum())
            if dupes > 0:
                rpt.duplicate_rows = dupes
                rpt.issues.append(QualityIssue(
                    column="<all>", severity="warning",
                    issue=f"{dupes:,} exact duplicate rows",
                    action="Deduplicate before modeling",
                ))
        except Exception:
            pass

    # -- join health check ----------------------------------

    def _check_joins(self, rpt: QualityReport, backend, schema: DatasetSchema):
        """For each relationship, check if FK and PK value formats match.

        Detects broken joins like hashed IFSC codes in txn table vs
        plain-text IFSC codes in the location master.
        """
        primary = schema.primary
        if not primary:
            return

        for rel in schema.relationships:
            fk_col = rel.from_column
            pk_col = rel.to_column

            # We can only check FK columns that exist in the primary table data
            if fk_col not in primary.columns:
                continue

            # Sample FK values from the data
            try:
                fk_samples = backend.sample_values(fk_col, 5)
            except Exception:
                rpt.join_health.append(JoinHealth(
                    from_col=fk_col, to_table=rel.to_table,
                    to_col=pk_col, status="unknown",
                    reason="Could not sample FK values"))
                continue

            # Get PK sample values from the schema (if available)
            to_table = schema.get_table_by_name(rel.to_table)
            pk_samples = []
            if to_table and pk_col in to_table.columns:
                pk_samples = to_table.columns[pk_col].sample_values

            health = self._assess_join(fk_col, fk_samples, rel.to_table, pk_col, pk_samples)
            rpt.join_health.append(health)

            if health.status == "broken":
                rpt.issues.append(QualityIssue(
                    column=fk_col, severity="warning",
                    issue=f"Broken join to {rel.to_table}.{pk_col}",
                    detail=health.reason,
                    action=f"Join will return 0 rows -- {rel.to_table} enrichment unavailable",
                ))

    @staticmethod
    def _assess_join(fk_col: str, fk_samples: list, to_table: str,
                     pk_col: str, pk_samples: list) -> JoinHealth:
        """Compare FK and PK sample formats to detect mismatches."""
        fk_valid = [str(s).strip() for s in fk_samples if s and str(s).strip()]
        pk_valid = [str(s).strip() for s in pk_samples if s and str(s).strip()]

        fk_str = fk_valid[0] if fk_valid else ""
        pk_str = pk_valid[0] if pk_valid else ""

        # If no FK values, column is empty -- join is trivially broken
        if not fk_valid:
            return JoinHealth(
                from_col=fk_col, to_table=to_table, to_col=pk_col,
                status="broken", reason="FK column is empty",
                fk_sample="(empty)", pk_sample=pk_str)

        # If no PK samples available, we can't assess
        if not pk_valid:
            return JoinHealth(
                from_col=fk_col, to_table=to_table, to_col=pk_col,
                status="unknown", reason="No PK sample values in schema",
                fk_sample=fk_str)

        # Format comparison: check if FK is hashed but PK is not
        fk_is_hex = all(
            _HEX_RE.match(s) and len(s) in {32, 40, 64, 128}
            for s in fk_valid
        )
        pk_is_hex = all(
            _HEX_RE.match(s) and len(s) in {32, 40, 64, 128}
            for s in pk_valid
        )

        if fk_is_hex and not pk_is_hex:
            return JoinHealth(
                from_col=fk_col, to_table=to_table, to_col=pk_col,
                status="broken",
                reason=f"FK is hashed ({len(fk_str)}-char hex) but PK is plain text",
                fk_sample=fk_str[:20] + "...", pk_sample=pk_str[:20])

        if not fk_is_hex and pk_is_hex:
            return JoinHealth(
                from_col=fk_col, to_table=to_table, to_col=pk_col,
                status="broken",
                reason="FK is plain text but PK is hashed",
                fk_sample=fk_str[:20], pk_sample=pk_str[:20] + "...")

        # Length mismatch (both non-hex) could indicate different formats
        fk_lens = {len(s) for s in fk_valid}
        pk_lens = {len(s) for s in pk_valid}
        if fk_lens and pk_lens and not fk_lens & pk_lens:
            if max(fk_lens) > 3 * max(pk_lens) or max(pk_lens) > 3 * max(fk_lens):
                return JoinHealth(
                    from_col=fk_col, to_table=to_table, to_col=pk_col,
                    status="broken",
                    reason=f"FK length ({fk_lens}) incompatible with PK length ({pk_lens})",
                    fk_sample=fk_str[:20], pk_sample=pk_str[:20])

        return JoinHealth(
            from_col=fk_col, to_table=to_table, to_col=pk_col,
            status="healthy", fk_sample=fk_str[:20], pk_sample=pk_str[:20])

    def _compute_score(self, rpt: QualityReport):
        score = 100
        for i in rpt.issues:
            score -= {"critical": 15, "warning": 5, "info": 1}.get(i.severity, 0)
        rpt.score = max(0, score)
