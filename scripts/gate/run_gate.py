"""Phase 3 gate: the four public datasets run end to end with no schema code.

    python scripts/gate/run_gate.py [~/lh-data] [--md out.md] [--only paysim,ulb]

For each sample in <data>/samples/<name>.parquet, over the real MCP server
(in-process client, sandboxed kernel):

  1. load_dataset, then score the proposed roles (check_roles.score)
  2. confirm_roles with the ground-truth roles from expected_roles.json --
     standing in for the person's corrections, relayed by the agent (so the
     report flags them as not verified by a person)
  3. sql: fraud count by label
  4. check_feature on the amount column; a NOT_A_FEATURE column is refused
  5. build_sample on the label-lag ASSUMPTION route (answers.json)
  6. falsify + record_finding, then report

The dataset passes when: no role is wrong AND settled without asking, the
sample is CERTIFIED with label maturity recorded as assumed, the role
refusal fires, and the report is saved. Per-dataset inputs are only the
person's answers in answers.json; nothing in this file names a column.
Exit code 1 if any dataset fails.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from check_roles import score  # noqa: E402

ROLES_YES = "yes, these roles are right"
LAG_YES = "yes, assume this label lag"
DATASETS = ("paysim", "ulb", "ieee", "baf")


def _text(r) -> str:
    return "\n".join(getattr(c, "text", "") for c in r.content)


async def run_one(name: str, path: Path, expected: dict, answers: dict, out_root: Path) -> dict:
    from mcp import Client

    from ledgerhound.mcp.server import LedgerhoundSession, build_server

    res: dict = {"name": name, "checks": {}, "notes": []}
    df = pd.read_parquet(path)
    res["roles"] = score(name, df, expected)
    ok = res["checks"]
    ok["no role wrong and settled"] = res["roles"]["counts"]["wrong-settled"] == 0

    s = LedgerhoundSession(out_root / name)
    t0 = time.time()
    try:
        async with Client(build_server(s)) as c:
            async def call(tool, **args):
                return _text(await c.call_tool(tool, args))

            load = await call("load_dataset", path=str(path))
            ok["loads"] = "error" not in load[:200].lower()
            roles = {col: allowed[0] for col, allowed in expected.items()
                     if col in df.columns and not col.startswith("_")}
            conf = await call("confirm_roles", roles=roles,
                              absent=["label_time", "label_source"], confirm=ROLES_YES)
            ok["roles confirmed"] = "CONFIRMED" in conf
            rm = s.tools.roles
            label, amount = rm.first("label"), rm.first("amount")

            q = await call("sql", query=f'select "{label}" as label, count(*) as n '
                                        "from df group by 1 order by 1")
            ok["sql"] = "[2 row(s)]" in q
            if amount:
                feat = await call("check_feature", features=[amount], label=label)
                ok["check_feature runs"] = not feat.startswith(("REFUSED", "error"))
            naf = answers.get("not_a_feature")
            if naf:
                ref = await call("check_feature", features=[naf], label=label)
                ok[f"{naf} refused as a feature"] = ref.startswith("REFUSED")

            built = await call("build_sample", time_unit=answers["time_unit"],
                               label_lag=answers["label_lag"],
                               design_prevalence=answers["design_prevalence"],
                               confirm=LAG_YES)
            ok["sample CERTIFIED"] = "sample built -- CERTIFIED" in built
            ok["maturity recorded as assumed"] = "assumed" in built
            res["sample"] = built.splitlines()[:6]
            if not ok["sample CERTIFIED"]:
                res["notes"].append(built[:1500])

            claim = f"the fraud rate in {name} differs across {amount or 'transaction-size'} quartiles"
            await call("falsify", claim=claim)
            rec = await call("record_finding", claim=claim, sample_frame="population_estimate",
                             evidence="gate smoke finding")
            ok["finding recorded"] = rec.startswith("recorded finding")
            saved = await call("report", title=f"Phase 3 gate: {name}",
                               markdown=f"Gate run on a {len(df):,}-row sample of {name}.")
            m = re.search(r"-> (\S+)", saved)
            report = Path(m.group(1)).read_text(encoding="utf-8") if m else ""
            ok["report saved"] = bool(report)
            ok["report says maturity assumed"] = "**assumed** (no label-time column)" in report
    except Exception as e:  # noqa: BLE001 -- a crash is a failed gate, reported as such
        res["checks"]["no crash"] = False
        res["notes"].append(f"{type(e).__name__}: {e}")
    finally:
        s.tools.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["passed"] = all(res["checks"].values())
    return res


def render(results: list[dict]) -> str:
    out = ["# Phase 3 gate: four public datasets, no schema code", "",
           "| dataset | rows | result | role check (correct / soft / wrong-asked / "
           "wrong-settled) | seconds |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        c = r["roles"]["counts"]
        out.append(f"| {r['name']} | {r['roles']['rows']:,} | "
                   f"{'PASS' if r['passed'] else '**FAIL**'} | {c['correct']} / {c['soft']} / "
                   f"{c['wrong-asked']} / {c['wrong-settled']} | {r['seconds']} |")
    out += ["", "Label maturity: none of the four has a label-time column, so every sample "
            "is built on the assumption route (labels stated final by the person) and the "
            "certificate records it as an assumption, not a check."]
    for r in results:
        out += ["", f"## {r['name']}", ""]
        out += [f"- {'ok ' if v else 'FAIL'} {k}" for k, v in r["checks"].items()]
        if r.get("sample"):
            out += ["", "```", *r["sample"], "```"]
        for n in r["notes"]:
            out += ["", "```", n, "```"]
    return "\n".join(out) + "\n"


@contextlib.contextmanager
def isolate(root: Path):
    """Every gate run starts from nothing: its own role store (so roles a
    previous run confirmed are not silently reused) and empty project/team
    Knowledge Packs (so the user's packs do not steer the result). Nothing
    is written to the user's library, and everything is restored on exit, so
    an in-process caller (the test suite) is left exactly as it was."""
    import os

    import ledgerhound.paths as paths
    saved_lib = paths.library_dir
    keys = ("LEDGERHOUND_KB_DIR", "LEDGERHOUND_TEAM_KB_DIR")
    saved_env = {k: os.environ.get(k) for k in keys}
    lib = root / "library"
    paths.library_dir = lambda: lib
    os.environ["LEDGERHOUND_KB_DIR"] = str(root / "kb-project")
    os.environ["LEDGERHOUND_TEAM_KB_DIR"] = str(root / "kb-team")
    try:
        yield
    finally:
        paths.library_dir = saved_lib
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main(argv: list[str]) -> int:
    import anyio
    md_out, only = None, None
    if "--md" in argv:
        i = argv.index("--md")
        md_out, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
    if "--only" in argv:
        i = argv.index("--only")
        only, argv = set(argv[i + 1].split(",")), argv[:i] + argv[i + 2:]
    data = Path(argv[0]).expanduser() if argv else Path.home() / "lh-data"
    expected = json.loads((HERE / "expected_roles.json").read_text(encoding="utf-8"))
    answers = json.loads((HERE / "answers.json").read_text(encoding="utf-8"))
    out_root = Path(tempfile.mkdtemp(prefix="lh-gate-"))
    results = []
    with isolate(out_root):
        for name in DATASETS:
            if only and name not in only:
                continue
            p = data / "samples" / f"{name}.parquet"
            if not p.exists():
                print(f"[{name}] no sample at {p}; run scripts/gate/fetch_samples.sh {name}")
                continue
            print(f"[{name}] running ...", flush=True)
            results.append(anyio.run(run_one, name, p, expected[name], answers[name], out_root))
    text = render(results)
    print(text)
    if md_out:
        md_out.write_text(text, encoding="utf-8")
    missing = [n for n in DATASETS if (not only or n in only)
               and n not in {r["name"] for r in results}]
    return 1 if missing or not all(r["passed"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
