"""
Checks that can only run inside the org's network or on a GPU box.

    python scripts/prod_checks.py            # everything
    python scripts/prod_checks.py gateway store tokenizer
    python scripts/prod_checks.py gpu
    python scripts/prod_checks.py adversarial   # makes ~10 real LLM calls

Each check prints one line -- PASS / WARN / FAIL / SKIP -- with the reason,
and everything is saved to prod_checks_<time>.json so the result can be
sent back as one file. Exit code 1 if anything FAILED.

It reads the same main_config.yaml the shell uses (run it from where you
run `ledgerhound shell`, or set LEDGERHOUND_MAIN_CONFIG). It never prints a key.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RESULTS: list[dict] = []


def report(check: str, status: str, detail: str) -> None:
    RESULTS.append({"check": check, "status": status, "detail": detail})
    print(f"{status:5s} {check:28s} {detail}", flush=True)


def _llm():
    """The shell's own LLM client, from main_config.yaml."""
    from ledgerhound.llm.client import LLMClient, _load_master_config, normalise_method
    method = normalise_method(str((_load_master_config().get("llm") or {})
                                  .get("method") or "api"))
    cfg = Path(f"configs/{method}.yaml")
    if not cfg.exists():
        cfg = Path(__file__).resolve().parent.parent / "configs" / f"{method}.yaml"
    return LLMClient.from_config(str(cfg))


# -- gateway -----------------------------------------------------------------

def check_gateway() -> None:
    try:
        llm = _llm()
    except Exception as e:  # noqa: BLE001
        report("gateway: config", "FAIL", f"cannot build the LLM client: {e}")
        return
    report("gateway: config", "PASS", f"{llm.config.base_url}  model={llm.config.model}")

    listing = llm.list_models_detailed()
    if listing:
        ids = [m.get("id") for m in listing]
        from ledgerhound.llm.catalog import context_window_from_listing
        cw = context_window_from_listing(llm.config.model, listing)
        report("gateway: /models", "PASS",
               f"{len(ids)} model(s): {', '.join(map(str, ids[:6]))}"
               + ("" if llm.config.model in ids else
                  f"  -- NOTE configured model {llm.config.model!r} is not listed"))
        report("gateway: context window", "PASS" if cw else "WARN",
               f"gateway reports {cw:,} for {llm.config.model}" if cw else
               "gateway does not report max_model_len -- the shell caps its "
               "assumption at 262,144; set context_window under llm.<method> "
               "if the deployment serves less")
    else:
        report("gateway: /models", "WARN",
               "no model list (unreachable, or the gateway has no /models)")

    try:
        t0 = time.time()
        r = llm.generate([{"role": "user", "content": "Reply with the word OK."}],
                         max_tokens=16)
        usage = r.usage or {}
        report("gateway: chat", "PASS" if (r.content or "").strip() else "WARN",
               f"{(time.time() - t0):.1f}s, reply={(r.content or '')[:40]!r}")
        report("gateway: usage.prompt_tokens",
               "PASS" if usage.get("prompt_tokens") else "WARN",
               f"prompt_tokens={usage.get('prompt_tokens')} (calibration needs it)"
               if usage.get("prompt_tokens") else
               "the gateway does not report usage -- token counts stay "
               "estimates unless a tokenizer is available")
    except Exception as e:  # noqa: BLE001
        from ledgerhound.llm.client import LLMAuthError
        kind = "API key rejected" if isinstance(e, LLMAuthError) else type(e).__name__
        report("gateway: chat", "FAIL", f"{kind}: {str(e)[:200]}")
        return

    tool = [{"type": "function", "function": {
        "name": "echo", "description": "Echo a value back.",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                       "required": ["x"]}}}]
    try:
        r = llm.generate([{"role": "user",
                           "content": "Call the echo tool with x='42'. Do not answer in text."}],
                         tools=tool, tool_choice="auto", max_tokens=256)
        ok = any((c.get("function") or {}).get("name") == "echo" for c in (r.tool_calls or []))
        report("gateway: tool calling", "PASS" if ok else "FAIL",
               "tool call returned" if ok else
               "no tool call came back -- the inference server needs a tool-call "
               "parser for this model (e.g. vLLM --tool-call-parser); the agent "
               "cannot run analysis without it")
    except Exception as e:  # noqa: BLE001
        report("gateway: tool calling", "FAIL", f"{type(e).__name__}: {str(e)[:200]}")


# -- object store ------------------------------------------------------------

def check_store() -> None:
    from ledgerhound.io.minio import shared_settings
    from ledgerhound.shell.minio_cmd import check_store as _check
    s = shared_settings()
    ok, code, message, fix = _check(s)
    report("object store", "PASS" if ok else "FAIL",
           message if ok else f"[{code}] {message} Fix: {fix}")


def check_kb() -> None:
    """Team knowledge base: write two revisions of a probe card under
    kb/_probe/, read them back merged, confirm nothing sensitive went up,
    then remove the probe (a missing delete right is a WARN: compaction
    will simply keep old revisions)."""
    import json as _json
    import tempfile as _tf
    from ledgerhound.io.minio import shared_settings
    from ledgerhound.knowledge.dataset_card import DatasetCard
    from ledgerhound.knowledge.team_kb import TeamKB
    s = shared_settings()
    kb = TeamKB(settings=s, prefix="kb/_probe", outbox=Path(_tf.mkdtemp()) / "outbox",
                background=False)
    if not kb.configured():
        report("team kb", "SKIP", "object store not fully configured (/minio shows what is missing)")
        return
    dsid = "name:prod-check-probe"
    for who in ("a", "b"):
        c = DatasetCard.new("probe.parquet")
        c.dataset_id = dsid
        c.add_clarification("probe", "q", f"answer {who} 9876543210", by=who)
        kb.publish_card(c)
    sent = kb.flush()
    if kb.pending():
        report("team kb: write", "FAIL", f"{kb.pending()} probe revision(s) not uploaded: {kb.last_error}")
        return
    merged = kb.fetch_card(dsid)
    n = len(merged.clarifications) if merged else 0
    names = [o.object_name for o in kb._c().list(kb.bucket, "kb/_probe/")]
    leaked = any("9876543210" in kb._c().get_bytes(kb.bucket, k).decode("utf-8", "ignore")
                 for k in names)
    report("team kb: write + merge", "PASS" if (sent == 2 and n == 2 and not leaked) else "FAIL",
           f"{sent} revisions written, {n} answers merged, identifiers redacted: {not leaked}")
    try:
        for k in names:
            kb._c().delete(kb.bucket, k)
        report("team kb: delete (compaction)", "PASS", "probe removed")
    except Exception as e:  # noqa: BLE001
        report("team kb: delete (compaction)", "WARN",
               f"no delete right ({type(e).__name__}) -- nothing is lost, old card "
               "revisions are just never compacted; ask for delete on kb/ if it grows")


# -- tokenizer ---------------------------------------------------------------

def _configured_model() -> str:
    from ledgerhound.llm.client import _load_master_config, normalise_method
    sect = _load_master_config().get("llm") or {}
    method = normalise_method(str(sect.get("method") or "api"))
    return str(((sect.get(method) or {}).get("model")) or "")


def check_tokenizer() -> None:
    # Loading needs only the model NAME and the store -- not the gateway key.
    model = _configured_model()
    if not model:
        report("tokenizer: load", "SKIP", "no llm.<method>.model in main_config.yaml")
        return
    from ledgerhound.llm.tokenizer import resolve_tokenizer
    counter, st = resolve_tokenizer(model)
    if not st.ok:
        report("tokenizer: load", "FAIL", f"[{st.code}] {st.message} Fix: {st.fix}")
        return
    report("tokenizer: load", "PASS", f"{st.matched} from {st.source}")
    try:
        llm = _llm()
    except Exception as e:  # noqa: BLE001
        report("tokenizer: vs gateway", "SKIP", f"no gateway to compare with: {str(e)[:120]}")
        return
    text = "payer_account_number,amount,flag\n9876543210@ybl,12500.50,1\n" * 300
    try:
        r = llm.generate([{"role": "user", "content": text + "\nReply OK."}], max_tokens=8)
        gw = int((r.usage or {}).get("prompt_tokens") or 0)
    except Exception as e:  # noqa: BLE001
        report("tokenizer: vs gateway", "SKIP", f"gateway call failed: {str(e)[:120]}")
        return
    if not gw:
        report("tokenizer: vs gateway", "SKIP", "gateway reports no usage to compare with")
        return
    mine = counter.count(text + "\nReply OK.")
    diff = (gw - mine) / gw
    report("tokenizer: vs gateway", "PASS" if 0 <= diff < 0.10 else "WARN",
           f"tokenizer {mine:,} vs gateway {gw:,} (template overhead {gw - mine:,})"
           + ("" if 0 <= diff < 0.10 else " -- large gap: check the manifest entry"))


# -- GPU kernel paths ----------------------------------------------------------

def check_gpu() -> None:
    from ledgerhound.backends.compute_router import ComputeRouter
    prof = ComputeRouter().detect()
    if prof.gpu_count == 0:
        report("gpu: detect", "SKIP", "no GPU on this machine")
        return
    report("gpu: detect", "PASS",
           f"{prof.gpu_count} GPU(s) via {prof.gpu_detected_via}: {', '.join(prof.gpu_names)}")
    try:
        from ledgerhound.auto.kernel import KernelExecutor, kernels_available
    except Exception as e:  # noqa: BLE001
        report("gpu: kernel", "FAIL", f"cannot import the kernel: {e}")
        return
    if not kernels_available():
        report("gpu: kernel", "FAIL", "jupyter_client/ipykernel not installed -- "
               "pip install jupyter_client ipykernel")
        return
    import numpy as np
    import pandas as pd
    work = Path(tempfile.mkdtemp())
    data = work / "synthetic.parquet"
    n = 2_000_000
    rng = np.random.default_rng(0)
    pd.DataFrame({"payer": rng.integers(0, 50_000, n).astype(str),
                  "amount": rng.gamma(2.0, 400.0, n),
                  "flag": (rng.random(n) < 0.002).astype("int8")}).to_parquet(data)
    engines = ["cudf"] + (["dask_cudf"] if prof.gpu_count > 1 else [])
    for engine in engines:
        k = None
        try:
            t0 = time.time()
            k = KernelExecutor(scratch_dir=work / f"scratch_{engine}", data_path=str(data),
                               engine=engine, timeout=600)
            actual = getattr(k, "_engine", engine)
            r = k.execute("g = df.groupby('payer')['amount'].agg(['sum','count'])\n"
                          "print(type(df).__module__, len(g))")
            good = r.success and actual == ("cudf" if engine == "cudf" else actual) \
                and "cudf" in (r.output or "")
            report(f"gpu: {engine}", "PASS" if good else "FAIL",
                   f"{time.time() - t0:.0f}s, engine={actual}, out={(r.output or r.error)[:120]!r}"
                   + ("" if not k.gpu_fallbacks else
                      f"; fallbacks={[f['error_class'] for f in k.gpu_fallbacks]}"))
        except Exception as e:  # noqa: BLE001
            report(f"gpu: {engine}", "FAIL", f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            if k is not None:
                k.shutdown()


# -- /adversarial end to end -------------------------------------------------

def check_adversarial() -> None:
    import numpy as np
    import pandas as pd
    from ledgerhound.adversarial import engine as E
    from ledgerhound.adversarial.run import run_from_session
    from ledgerhound.shell.conversation import ConversationalAnalyst
    try:
        llm = _llm()
    except Exception as e:  # noqa: BLE001
        report("adversarial", "SKIP", f"no LLM config: {e}")
        return
    work = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        payer = f"ACC{rng.integers(0, 60):04d}"
        rows.append({"payer": payer, "payee": f"ACC{rng.integers(60, 120):04d}",
                     "amount": float(rng.gamma(2, 300)), "hour": int(rng.integers(0, 24)),
                     "flag": int(payer in ("ACC0001", "ACC0002"))})
    data = work / "fraud_sample.parquet"
    pd.DataFrame(rows).to_parquet(data)
    a = ConversationalAnalyst(llm, out_root=work / "runs", renderer=lambda s: None,
                              autonomous=True)
    a._run_tool("load_data", {"path": str(data)})
    a.goal.label_column, a.goal.entity_key, a.goal.entity_key_secondary = "flag", "payer", "payee"
    saved = (E.PAYER_ROUNDS, E.PAYEE_ROUNDS)
    E.PAYER_ROUNDS, E.PAYEE_ROUNDS = 2, 2          # a smoke test, not a study
    try:
        t0 = time.time()
        results, report_path = run_from_session(a, cases=1, seed=7,
                                                say=lambda s: print(f"      {s}"))
        r = results[0]
        ok = bool(r.verdict_label) and Path(report_path).exists()
        report("adversarial", "PASS" if ok else "FAIL",
               f"{time.time() - t0:.0f}s, verdict={r.verdict_label or r.verdict_text[:80]!r}, "
               f"report={report_path}")
    except Exception as e:  # noqa: BLE001
        report("adversarial", "FAIL", f"{type(e).__name__}: {str(e)[:240]}")
    finally:
        E.PAYER_ROUNDS, E.PAYEE_ROUNDS = saved
        a.close()


CHECKS = {"gateway": check_gateway, "store": check_store, "kb": check_kb,
          "tokenizer": check_tokenizer,
          "gpu": check_gpu, "adversarial": check_adversarial}


def main(argv: list[str]) -> int:
    wanted = argv or list(CHECKS)
    unknown = [w for w in wanted if w not in CHECKS]
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}. Choose from: {', '.join(CHECKS)}")
        return 2
    from ledgerhound.config import resolved_path
    print(f"main_config.yaml: {resolved_path() or '(none found)'}\n")
    for name in wanted:
        try:
            CHECKS[name]()
        except Exception as e:  # noqa: BLE001 -- one check must not hide the rest
            report(name, "FAIL", f"check crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
    out = Path(f"prod_checks_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json")
    out.write_text(json.dumps({"when": datetime.now(timezone.utc).isoformat(),
                               "results": RESULTS}, indent=2), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in RESULTS)
              for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print(f"\n{counts}  -> {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
