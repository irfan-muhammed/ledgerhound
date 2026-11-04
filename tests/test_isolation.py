"""ADR 0001: the analysis kernel cannot open network connections, writes only
where it is allowed to, starts with no secrets, and is stopped at its memory
cap and its wall-clock cell timeout. The report names only what is enforced."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import ledgerhound
from ledgerhound.runtime import isolation as iso
from ledgerhound.runtime.isolation import IsolationError, IsolationPolicy

kernels = pytest.importorskip("jupyter_client")
needs_bwrap = pytest.mark.skipif(not iso.bwrap_works(), reason="bubblewrap unavailable")


# -- the plan, no kernel ---------------------------------------------------- #

def test_kernel_env_drops_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("LEDGERHOUND_LLM_API_KEY", "z")
    monkeypatch.setenv("LEDGERHOUND_CUDF_PANDAS", "1")
    monkeypatch.setenv("SOME_RANDOM_VAR", "1")
    env = iso.kernel_env(tmp_path)
    assert "PATH" in env and env["HOME"] == str(tmp_path)
    assert env["LEDGERHOUND_CUDF_PANDAS"] == "1"
    for gone in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
                 "LEDGERHOUND_LLM_API_KEY", "SOME_RANDOM_VAR"):
        assert gone not in env


def test_bwrap_private_tmp_comes_before_binds(tmp_path):
    args = iso.bwrap_args([tmp_path])
    assert args.index("/tmp") < args.index(str(tmp_path.resolve()))
    assert "--unshare-net" in args and args[-1] == "--"


def test_no_sandbox_is_reported_partial_not_sandboxed(monkeypatch, tmp_path):
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    plan = IsolationPolicy(mode="best").plan([tmp_path], tmp_path)
    r = plan.report
    assert (r.level, r.network, r.writes, r.env) == ("partial", "open", "unrestricted", "allow-listed")
    assert r.missing and plan.prefix[:1] != ["bwrap"]
    assert "not enforced" in r.summary()


def test_require_refuses_below_sandboxed(monkeypatch, tmp_path):
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    with pytest.raises(IsolationError, match="require-isolation"):
        IsolationPolicy(mode="require").plan([tmp_path], tmp_path)


def test_off_claims_nothing(tmp_path):
    r = IsolationPolicy(mode="off").plan([tmp_path], tmp_path).report
    assert (r.level, r.network, r.env, r.memory) == ("off", "open", "inherited", "none")


def test_no_address_space_or_cpu_rlimits():
    # RLIMIT_AS kills cuDF/Arrow at startup; RLIMIT_CPU kills legitimate long runs.
    src = Path(iso.__file__).read_text()
    assert "setrlimit" not in src and "import resource" not in src


# -- a real sandboxed kernel ------------------------------------------------ #

def _executor(tmp_path, policy, timeout=60):
    from ledgerhound.auto.kernel import KernelExecutor
    return KernelExecutor(scratch_dir=tmp_path / "scratch", timeout=timeout,
                          isolation=policy)


@pytest.fixture(scope="module")
def sandboxed(tmp_path_factory):
    if not iso.bwrap_works():
        pytest.skip("bubblewrap unavailable")
    tmp = tmp_path_factory.mktemp("iso")
    ex = _executor(tmp, IsolationPolicy(mode="best"))
    yield ex, tmp
    ex.shutdown()


def _run(ex, code):
    res = ex.execute(code)
    return (res.output or "") + (res.error or "")


@needs_bwrap
def test_sandboxed_report(sandboxed):
    r = sandboxed[0].isolation_report
    assert (r.level, r.network, r.writes, r.env) == ("sandboxed", "blocked", "confined", "allow-listed")
    assert r.memory.startswith(("cgroup", "rss-poll"))


@needs_bwrap
def test_kernel_cannot_open_network_connections(sandboxed):
    out = _run(sandboxed[0],
               "import socket\n"
               "try:\n"
               "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
               "except OSError as e:\n"
               "    print('blocked', e.errno)")
    assert "blocked" in out and "CONNECTED" not in out


@needs_bwrap
def test_loopback_still_works_inside(sandboxed):
    # dask-cuda's LocalCUDACluster talks over loopback inside the kernel.
    out = _run(sandboxed[0],
               "import socket\ns = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
               "c = socket.create_connection(s.getsockname()); print('LOOPBACK_OK')")
    assert "LOOPBACK_OK" in out


@needs_bwrap
def test_writes_are_confined(sandboxed):
    ex, tmp = sandboxed
    pkg_probe = Path(ledgerhound.__file__).parent / "_lh_isolation_probe.txt"
    host_probe = tmp / "outside.txt"     # tmp is not bound: /tmp is private inside
    try:
        out = _run(ex,
                   f"from pathlib import Path\n"
                   f"for p in (r'{pkg_probe}', r'{host_probe}'):\n"
                   f"    try:\n"
                   f"        _real_open(p, 'w').write('x'); print('WROTE', p)\n"
                   f"    except OSError as e:\n"
                   f"        print('blocked', type(e).__name__)\n"
                   f"(scratch / 'ok.txt').write_text('ok'); print('SCRATCH_OK')")
        assert "SCRATCH_OK" in out
        assert not pkg_probe.exists() and not host_probe.exists()
    finally:
        pkg_probe.unlink(missing_ok=True)


@needs_bwrap
def test_no_escape_through_host_unix_sockets(sandboxed):
    # The user D-Bus would let a cell `systemd-run --user` a process outside
    # the sandbox; WSL interop would let it run a Windows .exe with network.
    out = _run(sandboxed[0],
               "import os, glob, subprocess\n"
               "print('BUS', [p for p in glob.glob('/run/user/*/bus') + "
               "['/run/dbus/system_bus_socket', '/var/run/docker.sock'] if os.path.exists(p)])\n"
               "print('WSL', glob.glob('/run/WSL/*'))\n"
               "exe = '/mnt/c/Windows/System32/cmd.exe'\n"
               "if os.path.exists(exe):\n"
               "    try:\n"
               "        r = subprocess.run([exe, '/c', 'echo', 'ESCAPED'], capture_output=True, timeout=10)\n"
               "        print('EXE', r.returncode, r.stdout)\n"
               "    except Exception as e:\n"
               "        print('EXE blocked', type(e).__name__)")
    assert "BUS []" in out and "WSL []" in out and "ESCAPED" not in out


@needs_bwrap
def test_kernel_env_has_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    ex = _executor(tmp_path, IsolationPolicy(mode="best"))
    try:
        out = _run(ex, "import os; print(sorted(k for k in os.environ if 'KEY' in k))")
        assert "OPENAI_API_KEY" not in out and "sk-should-not-leak" not in out
    finally:
        ex.shutdown()


# -- runaway cells ---------------------------------------------------------- #

def test_timeout_is_wall_clock_not_gap(tmp_path):
    # A cell that prints every 0.3s never goes quiet, so a gap timeout would
    # never fire. The deadline must stop it anyway, and the kernel survives.
    ex = _executor(tmp_path, IsolationPolicy(mode="off"), timeout=3)
    try:
        t0 = time.monotonic()
        res = ex.execute("import time\nfor i in range(200):\n    print(i, flush=True); time.sleep(0.3)")
        assert not res.success and "TimeoutError" in res.error
        assert time.monotonic() - t0 < 20
        assert ex.execute("print(6 * 7)").output.strip().startswith("42")
    finally:
        ex.shutdown()


_HOG = "x = b'x' * (900 * 2**20)\nimport time\ntime.sleep(30)\nprint('SURVIVED')"


def test_memory_cap_by_rss_polling(monkeypatch, tmp_path):
    monkeypatch.setattr(iso, "cgroup_scope_works", lambda: False)
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    ex = _executor(tmp_path, IsolationPolicy(mode="best", memory_max_bytes=600 * 2**20))
    try:
        assert ex.isolation_report.memory.startswith("rss-poll")
        t0 = time.monotonic()
        res = ex.execute(_HOG)
        assert not res.success and "MemoryError" in res.error and "SURVIVED" not in res.output
        assert time.monotonic() - t0 < 25
        again = ex.execute("print(1)")
        assert not again.success and "MemoryError" in again.error
    finally:
        ex.shutdown()


@pytest.mark.skipif(not iso.cgroup_scope_works(), reason="no user cgroup scope")
def test_memory_cap_by_cgroup(tmp_path):
    ex = _executor(tmp_path, IsolationPolicy(mode="best", memory_max_bytes=600 * 2**20))
    try:
        assert ex.isolation_report.memory.startswith("cgroup")
        res = ex.execute(_HOG)
        assert not res.success and "SURVIVED" not in res.output
    finally:
        ex.shutdown()


# -- the tool layer --------------------------------------------------------- #

def test_required_isolation_refuses_instead_of_falling_back(monkeypatch, tmp_path, fraud_csv):
    from ledgerhound.shell.tools import AnalystTools
    monkeypatch.setattr(iso, "bwrap_works", lambda: False)
    s = AnalystTools(out_root=tmp_path / "runs", renderer=lambda _: None,
                     use_kernel=True, isolation=IsolationPolicy(mode="require"))
    try:
        loaded = s._run_tool("load_data", {"path": str(fraud_csv)})
        assert "REFUSED" in loaded
        assert s._run_tool("run_python", {"code": "print(1)"}).startswith("REFUSED")
    finally:
        s.close()
