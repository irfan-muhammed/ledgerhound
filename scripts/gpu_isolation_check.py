"""
Check that the sandboxed kernel (ADR 0001) still gets the GPU.

Run on a Linux or WSL machine with an NVIDIA GPU, RAPIDS installed
(pip install --extra-index-url=https://pypi.nvidia.com ".[gpu,mcp]") and
bubblewrap available:

    python scripts/gpu_isolation_check.py              # synthetic 2M-row frame
    python scripts/gpu_isolation_check.py data.parquet # your own data

Every line is PASS or FAIL; exit status is 0 only if all pass. Paste the
whole output into the issue or PR if anything fails.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str, detail: str = "") -> bool:
    results.append((ok, what))
    print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    from ledgerhound.runtime import isolation as iso

    smi = shutil.which("nvidia-smi")
    gpus = subprocess.run([smi, "-L"], capture_output=True, text=True).stdout.strip() if smi else ""
    if not check(bool(gpus), "nvidia-smi sees a GPU", gpus.splitlines()[0] if gpus else "no nvidia-smi"):
        return 1
    if not check(iso.bwrap_works(), "bubblewrap can create a network namespace"):
        return 1

    work = Path(tempfile.mkdtemp(prefix="lh-gpucheck-"))
    if len(sys.argv) > 1:
        data = Path(sys.argv[1]).resolve()
    else:
        import numpy as np
        import pandas as pd
        n = 2_000_000
        rng = np.random.default_rng(0)
        data = work / "synthetic.parquet"
        pd.DataFrame({"account": rng.integers(0, 50_000, n),
                      "amount": rng.gamma(2.0, 500.0, n),
                      "is_fraud": rng.random(n) < 0.002}).to_parquet(data)
    print(f"      data: {data}")

    from ledgerhound.auto.kernel import KernelExecutor
    from ledgerhound.runtime.isolation import IsolationError, IsolationPolicy
    try:
        ex = KernelExecutor(scratch_dir=work / "scratch", data_path=str(data),
                            engine="cudf", timeout=600,
                            isolation=IsolationPolicy(mode="require"))
    except IsolationError as e:
        check(False, "kernel starts under --require-isolation", str(e))
        return 1
    try:
        r = ex.isolation_report
        check(r.level == "sandboxed", "isolation level is sandboxed", r.summary())
        check(ex._engine == "cudf", "kernel kept the cuDF engine", f"engine={ex._engine}")
        check(not ex.gpu_fallbacks, "no GPU fallbacks during startup",
              "; ".join(f"{f['stage']}: {f['error_class']}" for f in ex.gpu_fallbacks))

        out = ex.execute("import cupy\nprint('DEVICES', cupy.cuda.runtime.getDeviceCount())\n"
                         "print('DF', type(df).__module__)")
        text = (out.output or "") + (out.error or "")
        check(out.success and "DEVICES 0" not in text and "DEVICES" in text,
              "CUDA devices visible inside the sandbox", text.splitlines()[0] if text else "")
        check("cudf" in text, "df is a cuDF frame", text.strip().splitlines()[-1] if text else "")

        cols = ex.execute("print(list(df.columns)[:3])").output
        gb = ex.execute("c = df.columns[0]\nprint(df.groupby(c).size().sort_values().tail(1))")
        check(gb.success, "a groupby runs on the GPU", (gb.error or cols or "").strip()[:200])

        net = ex.execute("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)"
                         "; print('CONNECTED')\nexcept OSError:\n    print('BLOCKED')")
        check("BLOCKED" in (net.output or ""), "network still blocked with the GPU in use")
    finally:
        ex.shutdown()
        shutil.rmtree(work, ignore_errors=True)

    failed = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
