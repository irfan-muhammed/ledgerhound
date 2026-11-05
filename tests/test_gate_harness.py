"""The Phase 3 gate harness, dry-run on synthetic stand-ins with the four
datasets' real column names (scripts/gate/standins.py). The real gate runs
the same script on Kaggle samples; this keeps the harness itself working."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jupyter_client")

GATE = Path(__file__).resolve().parents[1] / "scripts" / "gate"


@pytest.fixture(scope="module")
def gate():
    sys.path.insert(0, str(GATE))
    try:
        import run_gate
        import standins
        yield run_gate, standins
    finally:
        sys.path.remove(str(GATE))


def test_the_gate_names_no_columns():
    """No schema code: run_gate.py never names a dataset column; the only
    per-dataset inputs are the person's answers and the expected roles."""
    import re
    src = (GATE / "run_gate.py").read_text(encoding="utf-8")
    # role lookups (rm.first("amount")) name a ROLE, not a column
    src = re.sub(r'first\("[a-z_]+"\)', "", src)
    expected = json.loads((GATE / "expected_roles.json").read_text(encoding="utf-8"))
    cols = {c for name, roles in expected.items() if not name.startswith("_") for c in roles}
    named = sorted(c for c in cols if f'"{c}"' in src or f"'{c}'" in src)
    assert not named, named


def test_all_four_pass_on_standins(gate, tmp_path):
    run_gate, standins = gate
    standins.write(tmp_path, n=1500)
    md = tmp_path / "gate.md"
    code = run_gate.main([str(tmp_path), "--md", str(md)])
    text = md.read_text(encoding="utf-8")
    assert code == 0, text
    for name in ("paysim", "ulb", "ieee", "baf"):
        assert f"| {name} | 1,500 | PASS |" in text
    assert "FAIL" not in text
    assert "ieee" in text and "down-sampled" not in text.split("## ieee")[1].split("## baf")[0]
