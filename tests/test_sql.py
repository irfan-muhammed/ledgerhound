"""sql: read-only DuckDB over the loaded data, enforced by the parser and by
the kernel connection's own configuration."""

from __future__ import annotations

import pytest

from ledgerhound.shell.tools.sql import check_read_only


@pytest.mark.parametrize("q", [
    "select is_fraud, avg(amount) from df group by 1",
    "SELECT 1;",
    "with a as (select * from df) select count(*) from a",
    "explain select * from df",
    "describe df",
])
def test_read_only_queries_pass_the_parser(q):
    assert check_read_only(q)[1] == ""


@pytest.mark.parametrize("q,needle", [
    ("insert into df values (1)", "INSERT statements are not allowed"),
    ("delete from df", "DELETE statements are not allowed"),
    ("create table t as select * from df", "CREATE statements are not allowed"),
    ("copy df to 'out.csv'", "COPY statements are not allowed"),
    ("attach 'other.db'", "ATTACH statements are not allowed"),
    ("set enable_external_access = true", "SET statements are not allowed"),
    ("install httpfs", "statements are not allowed"),
    ("pragma version", "PRAGMA statements are not allowed"),
    ("select 1; drop table df", "one statement per call"),
    ("selec 1", "does not parse"),
    ("  ;  ", "needs a query"),
])
def test_everything_else_is_refused_before_it_runs(q, needle):
    assert needle in check_read_only(q)[1]


def test_exec_mode_tool_layer(tmp_path, fraud_csv):
    """The CLI shell without a kernel: same refusals, same results."""
    from ledgerhound.shell.tools import AnalystTools
    t = AnalystTools(out_root=tmp_path / "runs", renderer=lambda s: None)
    try:
        assert "no data loaded" in t._run_tool("sql", {"query": "select 1"})
        t._run_tool("load_data", {"path": str(fraud_csv)})
        out = t._run_tool("sql", {"query": "select is_fraud, round(avg(amount), 1) as m "
                                           "from df group by 1 order by 1"})
        assert "550.0" in out and "[2 row(s)]" in out
        assert t._run_tool("sql", {"query": "drop table df"}).startswith("REFUSED")
        capped = t._run_tool("sql", {"query": "select * from df", "max_rows": 3})
        assert "[3 row(s) -- truncated at 3" in capped
        blocked = t._run_tool("sql", {"query": "select * from read_csv('/etc/passwd')"})
        assert "Permission Error" in blocked and "root:" not in blocked
    finally:
        t.close()


mcp = pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_sql_over_mcp_runs_in_the_kernel(tmp_path, fraud_csv):
    from mcp import Client

    from ledgerhound.mcp.server import LedgerhoundSession, build_server

    def text(r):
        return "\n".join(getattr(c, "text", "") for c in r.content)
    s = LedgerhoundSession(tmp_path / "runs")
    try:
        async with Client(build_server(s)) as c:
            await c.call_tool("load_dataset", {"path": str(fraud_csv)})
            ok = text(await c.call_tool("sql", {"query": "select payer_id, count(*) n, sum(is_fraud) f "
                                                         "from df group by 1 order by 1"}))
            assert "A" in ok and "B" in ok and "[2 row(s)]" in ok
            ro = text(await c.call_tool("sql", {"query": "update df set amount = 0"}))
            assert ro.startswith("REFUSED") and "UPDATE" in ro
            leak = text(await c.call_tool("sql", {"query": f"select * from '{fraud_csv}'"}))
            assert "Permission Error" in leak
            # the kernel's own df is untouched and no connection lingers
            chk = text(await c.call_tool("run_analysis", {
                "code": "print(float(df.amount.sum()), '_lh_con' in globals())"}))
            assert "1240.0 False" in chk
    finally:
        s.tools.close()
