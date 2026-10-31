"""Preflight data validation.

Runs the moment data files are selected, BEFORE any LLM call or full load,
and reports schema / grain problems up front with a clear message instead of
letting them surface as a 200-line pyarrow traceback after 25M rows are read
(the AUD-191 class of failure).

Design goals:
  * CHEAP: parquet schema + row count come from file metadata (no data read);
    CSV dtypes are inferred from a small header sample, never the whole file.
  * PURE: takes a list of paths, returns a report. No console, no I/O side
    effects, fully unit-testable.
  * ADVISORY by default: the caller decides whether a WARN blocks. Only truly
    broken inputs (empty / unreadable) are ERROR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Column names that, if present, mark a row as fraud/non-fraud. Used only to
# hint when a "fraud" selection carries no obvious label column.
_LABEL_CANDIDATES = ("is_fraud", "fraud", "label", "typology", "fraud_flag",
                     "is_fraud_flag", "target")

# Row-count ratio above which two files are almost certainly a different grain
# (e.g. 25M-row transaction history vs 350K-row per-account snapshot).
_GRAIN_RATIO = 50


@dataclass
class FileSchema:
    """Cheap schema snapshot for one file."""
    name: str
    path: str
    columns: list  # ordered column names
    dtypes: dict   # col -> dtype string (inferred/sampled)
    n_rows: Optional[int]  # None when not cheaply knowable (CSV)
    kind: str      # "parquet" | "csv"
    error: Optional[str] = None


@dataclass
class PreflightIssue:
    level: str   # "error" | "warn" | "info"
    code: str
    message: str


@dataclass
class PreflightReport:
    label: str
    files: list = field(default_factory=list)          # list[FileSchema]
    issues: list = field(default_factory=list)          # list[PreflightIssue]
    common_columns: list = field(default_factory=list)
    dropped_columns: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.level == "warn"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


def _read_parquet_schema(p: Path) -> FileSchema:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(p))
    arrow = pf.schema_arrow
    cols = list(arrow.names)
    dtypes = {n: str(t) for n, t in zip(arrow.names, arrow.types)}
    return FileSchema(name=p.name, path=str(p), columns=cols, dtypes=dtypes,
                      n_rows=pf.metadata.num_rows, kind="parquet")


def _read_csv_schema(p: Path, sample_rows: int = 500) -> FileSchema:
    import pandas as pd
    head = pd.read_csv(p, nrows=sample_rows)
    cols = list(head.columns)
    dtypes = {c: str(head[c].dtype) for c in cols}
    return FileSchema(name=p.name, path=str(p), columns=cols, dtypes=dtypes,
                      n_rows=None, kind="csv")


def _schema_for(p: Path) -> FileSchema:
    try:
        if p.suffix.lower() in (".parquet", ".pq"):
            return _read_parquet_schema(p)
        return _read_csv_schema(p)
    except Exception as exc:  # noqa: BLE001
        return FileSchema(name=p.name, path=str(p), columns=[], dtypes={},
                          n_rows=None, kind=p.suffix.lower().lstrip("."),
                          error=f"{type(exc).__name__}: {exc}")


def preflight(files, label: str, *, expect_label: bool = False) -> PreflightReport:
    """Scan ``files`` and return a :class:`PreflightReport`.

    ``expect_label`` -- set True for datasets that should carry a fraud label
    column (e.g. a mixed fraud/non-fraud extract); leave False for all-fraud
    shards where the label is implicit.
    """
    paths = [Path(f) for f in files]
    schemas = [_schema_for(p) for p in paths]
    rep = PreflightReport(label=label, files=schemas)

    ok = [s for s in schemas if s.error is None]

    # --- unreadable / empty files -> ERROR ---------------------------------
    for s in schemas:
        if s.error is not None:
            rep.issues.append(PreflightIssue(
                "error", "unreadable", f"{s.name}: cannot read ({s.error})"))
        elif not s.columns:
            rep.issues.append(PreflightIssue(
                "error", "empty", f"{s.name}: no columns / empty file"))
        elif s.n_rows == 0:
            rep.issues.append(PreflightIssue(
                "warn", "no_rows", f"{s.name}: 0 rows"))

    if not ok:
        return rep

    # --- schema agreement across files -------------------------------------
    common = set(ok[0].columns)
    union = set()
    for s in ok:
        common &= set(s.columns)
        union |= set(s.columns)
    rep.common_columns = [c for c in ok[0].columns if c in common]
    rep.dropped_columns = sorted(union - common)

    if len(ok) > 1 and rep.dropped_columns:
        # Which files diverge, so the message is actionable.
        odd = [s.name for s in ok if set(s.columns) != common]
        rep.issues.append(PreflightIssue(
            "warn", "schema_mismatch",
            f"{len(ok)} files disagree on schema: {len(rep.common_columns)} "
            f"common of {len(union)} columns; {len(rep.dropped_columns)} would "
            f"be dropped ({', '.join(rep.dropped_columns[:6])}"
            f"{'...' if len(rep.dropped_columns) > 6 else ''}). "
            f"Diverging: {', '.join(odd[:4])}"
            f"{'...' if len(odd) > 4 else ''}."))

    # --- dtype conflicts on shared columns ---------------------------------
    conflicts = []
    for col in rep.common_columns:
        seen = {s.dtypes.get(col) for s in ok}
        seen.discard(None)
        # Normalise pandas 'object'/'string' + parquet 'string' as one family
        # so we only flag genuinely different families (e.g. timestamp vs str).
        fams = {_dtype_family(d) for d in seen}
        if len(fams) > 1:
            conflicts.append((col, sorted(seen)))
    if conflicts:
        shown = "; ".join(f"{c} ({'/'.join(d)})" for c, d in conflicts[:5])
        rep.issues.append(PreflightIssue(
            "warn", "dtype_conflict",
            f"{len(conflicts)} column(s) have conflicting dtypes across files "
            f"(will be coerced to string): {shown}"
            f"{'...' if len(conflicts) > 5 else ''}."))

    # --- grain mismatch heuristic ------------------------------------------
    counted = [s for s in ok if s.n_rows]
    if len(counted) > 1:
        lo = min(s.n_rows for s in counted)
        hi = max(s.n_rows for s in counted)
        if lo > 0 and hi / lo >= _GRAIN_RATIO:
            small = min(counted, key=lambda s: s.n_rows)
            big = max(counted, key=lambda s: s.n_rows)
            rep.issues.append(PreflightIssue(
                "warn", "grain_mismatch",
                f"row counts span {hi / lo:,.0f}x ({small.name}={small.n_rows:,} "
                f"vs {big.name}={big.n_rows:,}) -- these are likely DIFFERENT "
                f"datasets (e.g. per-account snapshot vs transaction history), "
                f"not shards of the same one."))

    # --- fraud label presence ----------------------------------------------
    if expect_label:
        has_label = any(c.lower() in _LABEL_CANDIDATES
                        for c in rep.common_columns)
        if not has_label:
            rep.issues.append(PreflightIssue(
                "warn", "no_label",
                f"no fraud-label column found among common columns "
                f"(looked for {', '.join(_LABEL_CANDIDATES[:4])}...). "
                f"If this is a mixed extract, the non-fraud rows cannot be "
                f"distinguished."))

    # --- per-file constant / all-null (sampled) ----------------------------
    # Only meaningful, and only reported at INFO, since it is a sampled hint.
    return rep


def _dtype_family(dtype: str) -> str:
    """Collapse a dtype string to a coarse family for conflict detection."""
    d = (dtype or "").lower()
    if any(k in d for k in ("int", "float", "double", "decimal", "number")):
        return "numeric"
    if any(k in d for k in ("time", "date")):
        return "datetime"
    if any(k in d for k in ("bool",)):
        return "bool"
    if any(k in d for k in ("object", "string", "str", "utf8", "large_string",
                            "category")):
        return "string"
    return d or "unknown"


def render_report(rep: PreflightReport) -> str:
    """Human-readable summary (ASCII only) for the CLI."""
    lines = [f"Preflight: {rep.label} ({len(rep.files)} file(s))"]
    for s in rep.files:
        if s.error:
            lines.append(f"  - {s.name}: UNREADABLE ({s.error})")
        else:
            rows = f"{s.n_rows:,} rows" if s.n_rows is not None else "rows n/a"
            lines.append(f"  - {s.name}: {len(s.columns)} cols, {rows}")
    if not rep.issues:
        lines.append("  OK: schemas agree, no issues found.")
        return "\n".join(lines)
    for lvl in ("error", "warn", "info"):
        for i in rep.issues:
            if i.level == lvl:
                tag = {"error": "ERROR", "warn": "WARN", "info": "INFO"}[lvl]
                lines.append(f"  [{tag}] {i.message}")
    return "\n".join(lines)
