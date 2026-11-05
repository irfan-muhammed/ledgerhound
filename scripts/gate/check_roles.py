"""Phase 3 gate, roles part: run the product's generic role inference on each
gate sample and score it against scripts/gate/expected_roles.json.

    python scripts/gate/check_roles.py [~/lh-data] [--md out.md]

Per dataset it reports:
  correct        inferred role is an allowed one
  soft           allowed, but not the first (preferred) answer
  wrong-asked    wrong, but not settled automatically: the person sees it
                 in confirm_roles and can correct it
  wrong-settled  wrong AND settled without asking (high confidence) --
                 the dangerous kind; the gate wants zero of these
  label roles    always asked, so never "settled"

"feature, no domain role" counts as correct where the column has no role
(the ULB V1..V28 case). Exit code 1 if any wrong-settled cell exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ledgerhound.roles import infer_roles  # noqa: E402


def score(name: str, df: pd.DataFrame, expected: dict) -> dict:
    rm = infer_roles(df)
    rows, counts = [], {"correct": 0, "soft": 0, "wrong-asked": 0, "wrong-settled": 0}
    for col, a in rm.columns.items():
        allowed = expected.get(col, ["feature"])
        if a.role == allowed[0]:
            verdict = "correct"
        elif a.role in allowed:
            verdict = "soft"
        elif a.status == "auto":
            verdict = "wrong-settled"
        else:
            verdict = "wrong-asked"
        counts[verdict] += 1
        if verdict != "correct" or a.role != "feature":
            rows.append((col, a.role, a.confidence, a.status, "/".join(allowed), verdict, a.reason))
    asked = rm.needs_asking()
    return {"name": name, "rows": len(df), "cols": df.shape[1], "counts": counts,
            "detail": rows, "asked": len(asked), "summary": rm.summary()}


def render(results: list[dict]) -> str:
    out = ["# Phase 3 role check", "",
           "| dataset | columns | correct | soft | wrong, asked | wrong, settled | asked in confirm_roles |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        c = r["counts"]
        out.append(f"| {r['name']} | {r['cols']} | {c['correct']} | {c['soft']} | "
                   f"{c['wrong-asked']} | {c['wrong-settled']} | {r['asked']} |")
    for r in results:
        out += ["", f"## {r['name']} ({r['rows']:,} sampled rows)", "", f"Inferred: {r['summary']}", "",
                "| column | inferred | confidence | status | expected | verdict |",
                "| --- | --- | --- | --- | --- | --- |"]
        for col, role, conf, status, exp, verdict, _reason in r["detail"]:
            out.append(f"| {col} | {role} | {conf} | {status} | {exp} | {verdict} |")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    md_out = None
    if "--md" in argv:
        i = argv.index("--md")
        md_out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    data = Path(argv[0]).expanduser() if argv else Path.home() / "lh-data"
    expected = json.loads((HERE / "expected_roles.json").read_text(encoding="utf-8"))
    results = []
    for name in ("paysim", "ulb", "ieee", "baf"):
        p = data / "samples" / f"{name}.parquet"
        if not p.exists():
            print(f"[{name}] no sample at {p}; run scripts/gate/fetch_samples.sh {name}")
            continue
        results.append(score(name, pd.read_parquet(p), expected.get(name, {})))
    text = render(results)
    print(text)
    if md_out:
        md_out.write_text(text, encoding="utf-8")
    bad = sum(r["counts"]["wrong-settled"] for r in results)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
