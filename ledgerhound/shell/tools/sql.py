"""
ledgerhound.shell.tools.sql
===========================

sql: a read-only DuckDB query over the loaded data, run inside the analysis
kernel (raw rows never leave it; only the result table is returned).

Read-only is enforced twice:

1. Here, before anything runs: the query is parsed with DuckDB's own parser
   and must be exactly one SELECT (or EXPLAIN) statement.
2. In the kernel: the connection is opened with external access disabled
   and its configuration locked, so a SELECT cannot read other files
   (read_csv('/etc/passwd'), 'x.parquet'), COPY out, ATTACH, load
   extensions or switch any of that back on. The only table is `df`.

Trino is deferred (Phase 3 decision).
"""

from __future__ import annotations

MAX_ROWS = 200
ALLOWED = ("SELECT", "EXPLAIN")

_CELL = '''\
import duckdb as _lh_ddb
import pandas as _lh_pd
_lh_con = _lh_ddb.connect(config={{"enable_external_access": False, "lock_configuration": True}})
try:
    _lh_src = df
    if type(df).__module__.split(".")[0] in ("cudf", "dask_cudf"):
        _lh_src = (df.compute() if hasattr(df, "compute") else df).to_arrow()
    _lh_con.register("df", _lh_src)
    try:
        _lh_cur = _lh_con.execute({q})
    except _lh_ddb.Error as _lh_e:
        raise RuntimeError("sql: " + str(_lh_e).splitlines()[0]) from None
    _lh_cols = [d[0] for d in _lh_cur.description]
    _lh_rows = _lh_cur.fetchmany({n} + 1)
    _lh_more = len(_lh_rows) > {n}
    _lh_res = _lh_pd.DataFrame(_lh_rows[:{n}], columns=_lh_cols)
    with _lh_pd.option_context("display.max_columns", 50, "display.width", 200):
        print(_lh_res.to_string(index=False, max_colwidth=60) if len(_lh_res) else "(no rows)")
    print(f"[{{len(_lh_res)}} row(s){{' -- truncated at {n}; aggregate or add LIMIT' if _lh_more else ''}}]")
finally:
    _lh_con.close()
    del _lh_con
'''


def check_read_only(query: str) -> tuple[str, str]:
    """(clean query, refusal). The refusal is "" when the query may run."""
    q = (query or "").strip().rstrip(";").strip()
    if not q:
        return q, "sql needs a query."
    import duckdb
    try:
        stmts = duckdb.extract_statements(q)
    except Exception as e:  # noqa: BLE001 -- parser errors are the user's to fix
        return q, f"sql: the query does not parse: {str(e).splitlines()[0][:300]}"
    if len(stmts) != 1:
        return q, (f"REFUSED: sql runs one statement per call; this has {len(stmts)}. "
                   "Nothing was run.")
    kind = str(stmts[0].type).rsplit(".", 1)[-1]
    if kind not in ALLOWED:
        return q, (f"REFUSED: sql is read-only; {kind} statements are not allowed. "
                   "Only SELECT (and EXPLAIN) over the loaded data, which is the table "
                   "`df`. Nothing was run.")
    return q, ""


class SqlToolsMixin:
    """sql: read-only DuckDB over `df`, in the kernel."""

    def _tool_sql(self, query: str, max_rows: int = 50) -> str:
        if self._df is None:
            return "no data loaded -- call load_data first."
        q, refusal = check_read_only(query)
        if refusal:
            self._events.append("sql_refused", query=str(query)[:500], reason=refusal[:300])
            return refusal
        try:
            n = max(1, min(int(max_rows or 50), MAX_ROWS))
        except (TypeError, ValueError):
            n = 50
        self._events.append("sql", query=q[:2000], max_rows=n)
        return self._tool_run_python(_CELL.format(q=repr(q), n=n))
