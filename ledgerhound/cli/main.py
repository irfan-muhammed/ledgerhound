"""
ledgerhound.cli.main -- CLI entry point with auto-report and compute display.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

if sys.argv[1:2] == ["mcp"]:
    # `ledgerhound mcp`: stdout is the MCP protocol channel from the first
    # byte, so divert it before typer, rich or any command module loads.
    from ledgerhound.mcp._stdio import protect_stdout
    protect_stdout()

import typer  # noqa: E402
from rich.console import Console
from rich.panel import Panel


def _make_console_utf8_safe() -> None:
    """Force stdout/stderr to UTF-8 so rich output never crashes on a legacy
    Windows cp1252 console (AUD-166). ``errors='replace'`` degrades any stray
    non-ASCII glyph to '?' instead of raising UnicodeEncodeError mid-run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:
            pass


_make_console_utf8_safe()

app = typer.Typer(
    name="ledgerhound",
    help=(
        "Ledgerhound -- governed agentic EDA for fraud & risk analytics.\n\n"
        "START HERE:  ledgerhound shell   (the conversational analyst -- just "
        "describe your data and goal; the model drives, computes in a sandbox, "
        "and records evidence-backed findings). Reports land inside the "
        "project's run_outputs/ folder.\n\n"
        "The commands below are the scripted / power-user surface that the "
        "shell composes for you; you rarely need them directly.\n\n"
        "Setup:  ledgerhound doctor  (check env/keys)  .  "
        "ledgerhound shell --llm-config configs/gateway.yaml"
    ),
    epilog=(
        "Quick start:\n"
        "  ledgerhound doctor\n"
        "  ledgerhound shell --project ./my-analysis --llm-config configs/gateway.yaml\n"
        "Docs: docs/USER_GUIDE.md"
    ),
    no_args_is_help=True)
console = Console()


@app.callback()
def _startup() -> None:
    """Runs before every command: move pre-rename ./.sentinel and
    ./sentinel_workspace to their Ledgerhound names (see ledgerhound._compat)."""
    from ledgerhound._compat import migrate_cwd
    migrate_cwd()


@app.command("mcp")
def mcp_server(
    out_root: str = typer.Option("run_outputs", "--out-root",
                                 help="where run folders go"),
    require_isolation: bool = typer.Option(
        False, "--require-isolation",
        help="refuse to run code unless the kernel is fully sandboxed"),
) -> None:
    """Serve Ledgerhound over MCP (stdio) for Claude Code, Codex, OpenCode
    or any MCP client:  claude mcp add ledgerhound -- ledgerhound mcp"""
    from ledgerhound.mcp.server import main as mcp_main
    args = ["--out-root", out_root] + (["--require-isolation"] if require_isolation else [])
    raise typer.Exit(mcp_main(args))


def _read_user_turn(console: Console, prompt: str) -> str:
    """Read one user turn, but don't lose a pasted multi-line block.

    A plain `input()`/`console.input()` returns the instant it sees a
    newline -- fine for typing, but a terminal WITHOUT bracketed-paste
    support (common in a JupyterLab/Kubeflow terminal proxied over
    websockets) delivers a big pasted block as a burst of already-buffered
    lines, each with its own newline. The naive read only picks up the
    FIRST line and starts processing before the rest even arrives,
    reproducing exactly "only takes 2 lines" -- the paste isn't truncated
    upstream, it's abandoned here.

    Fix: after the first line, drain whatever is ALREADY sitting in
    stdin's buffer (non-blocking) and join it in -- a real keystroke-typed
    Enter has nothing queued behind it, so single-line typing is
    unaffected; a paste's remaining lines are all present near-instantly
    and get pulled in as one turn. POSIX-only (`select` doesn't support
    stdin on Windows) -- falls back to single-line reads there, unchanged
    from prior behaviour.
    """
    first = console.input(prompt)
    lines = [first]
    try:
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            nxt = sys.stdin.readline()
            if not nxt:
                break
            lines.append(nxt.rstrip("\n"))
    except Exception:  # noqa: BLE001 -- Windows, or any select failure: single-line only
        pass
    return "\n".join(lines)


def _check_file(path: str, label: str):
    if not Path(path).exists():
        console.print(f"[red]Error: {label} not found: {path}[/red]")
        if label == "Schema file":
            console.print("[dim]Hint: use --preset upi or --preset bank_fraud[/dim]")
        raise typer.Exit(1)


@app.command()
def profile(
    data:   str = typer.Option(..., "--data", "-d", help="Data file"),
    schema: Optional[str] = typer.Option(None, "--schema", "-s"),
    preset: Optional[str] = typer.Option(None, "--preset", help="upi | bank_fraud"),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    output: str = typer.Option("digest.yaml", "--output", "-o"),
):
    """Profile a dataset -- no LLM needed."""
    from ledgerhound.core import Ledgerhound
    _check_file(data, "Data file")
    if schema: _check_file(schema, "Schema file")

    console.print(Panel(f"[bold]Ledgerhound Profile[/bold]\nData: {data}\n"
                        f"Schema: {schema or preset or 'auto'}\n"
                        f"Target: {target or 'none'}", border_style="cyan"))
    try:
        engine = Ledgerhound(data=data, schema=schema, preset=preset, target=target)
        digest = engine.profile()
        output = _default_out(output, "digest.yaml")
        digest.save(output)

        # Beautiful summary display
        try:
            from ledgerhound.output.banner import print_banner, print_profile_summary
            print_banner()
            print_profile_summary(digest.to_dict())
        except Exception:
            console.print(f"\n{digest.summary()}")

        console.print(f"\n[bold green]OK Digest saved: {output}[/bold green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def eda(
    digest_path: str = typer.Option(..., "--digest"),
    data:   Optional[str] = typer.Option(None, "--data", "-d"),
    schema: Optional[str] = typer.Option(None, "--schema", "-s"),
    preset: Optional[str] = typer.Option(None, "--preset"),
    model:  Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    ask:      Optional[str] = typer.Option(None, "--ask", "-a"),
    playbook: Optional[str] = typer.Option(None, "--playbook", "-p"),
    report:   Optional[str] = typer.Option(None, "--report", "-r"),
    interactive: bool = typer.Option(False, "--interactive", "-i"),
    knowledge_dir: Optional[str] = typer.Option(None, "--knowledge", "-k"),
    # -- P1.H: Phase-1 stack integration ----------------------------------
    memory: Optional[str] = typer.Option(
        None, "--memory",
        help="Dataset name for memory lookup (default: data file stem). "
             "Auto-loads .ledgerhound/memory/<name>.yaml if it exists.",
    ),
    findings_path: Optional[str] = typer.Option(
        None, "--findings",
        help="Override findings JSONL path "
             "(default: .ledgerhound/findings/<name>.jsonl)",
    ),
    kb_path: Optional[str] = typer.Option(
        None, "--kb",
        help="Override Schema KB path (default: .ledgerhound/upi_kb.duckdb)",
    ),
    journal_dir: Optional[str] = typer.Option(
        None, "--journal-dir",
        help="Override notebook journal output dir "
             "(default: .ledgerhound/journals)",
    ),
    no_journal: bool = typer.Option(
        False, "--no-journal",
        help="Skip notebook journal writing for this run",
    ),
):
    """Run agentic EDA with KB tools, structured findings, and notebook journal.

    The full Phase-1 stack auto-engages when artifacts are present:

    * `.ledgerhound/upi_kb.duckdb` -> Schema KB tools (P1.A)
    * `.ledgerhound/memory/<name>.yaml` -> DatasetMemory + sample-frame discipline (P1.D/E/I)
    * `.ledgerhound/findings/<name>.jsonl` -> structured-finding emission (P1.B)
    * `.ledgerhound/journals/` -> notebook journal per question (P1.C)

    Override any path with --kb / --memory / --findings / --journal-dir.
    """
    from ledgerhound.core import Ledgerhound
    from ledgerhound.profiler.digest import Digest

    _check_file(digest_path, "Digest file")
    if data: _check_file(data, "Data file")
    if llm_config: _check_file(llm_config, "LLM config file")

    if not model and not llm_config:
        console.print("[red]Provide --model or --llm-config[/red]")
        raise typer.Exit(1)

    try:
        loaded = Digest.load(digest_path)
    except Exception as e:
        console.print(f"[red]Error loading digest: {e}[/red]")
        console.print("[dim]Run 'ledgerhound profile' first.[/dim]")
        raise typer.Exit(1)

    engine = Ledgerhound(data=data, schema=schema, preset=preset,
                      model=model, model_name=model_name, api_key=api_key,
                      llm_config=llm_config, knowledge_dir=knowledge_dir)
    engine.eda.set_digest(loaded)

    # -- P1.H: Auto-wire the Phase-1 stack --------------------------------
    dataset_name = memory or (Path(data).stem if data else None)
    _wire_phase1_stack(
        agent=engine.eda,
        dataset_name=dataset_name,
        explicit_findings_path=findings_path,
        explicit_kb_path=kb_path,
        explicit_journal_dir=journal_dir,
        write_journal=not no_journal,
    )

    h = engine._get_llm().health_check()
    if h["status"] != "healthy":
        console.print(f"[red]LLM not available: {h.get('error', h)}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]OK LLM: {h.get('models',['?'])[0]}[/green]")

    if ask:
        console.print(f"\n[bold]Q: {ask}[/bold]\n")
        r = engine.eda.ask(ask)
        r.display()
        console.print(f"[dim]({r.tool_calls} tools, {r.total_tokens:,} tok, "
                      f"{r.elapsed_ms/1000:.1f}s, {len(r.charts)} charts, "
                      f"{len(r.finding_ids)} findings)[/dim]")
    elif playbook:
        pb = (engine.knowledge.load_playbook(playbook) if Path(playbook).exists()
              else None)
        res = engine.eda.run_playbook(playbook_name=playbook, playbook=pb,
                                      report_path=report)
        if not report:
            res.display()
    elif interactive:
        _interactive(engine, loaded)
    else:
        console.print("[yellow]Use --ask, --playbook, or --interactive[/yellow]")


def _wire_phase1_stack(
    *,
    agent,
    dataset_name: Optional[str],
    explicit_findings_path: Optional[str] = None,
    explicit_kb_path: Optional[str] = None,
    explicit_journal_dir: Optional[str] = None,
    write_journal: bool = True,
):
    """Auto-wire Schema KB, DatasetMemory, FindingsLog, and journal dir.

    Each artifact is opt-in: present on disk -> loaded; absent -> silently
    skipped (the agent already handles missing gracefully via its lazy
    loaders).
    """
    # -- Schema KB --------------------------------------------------------
    kb_path = Path(explicit_kb_path or ".ledgerhound/upi_kb.duckdb")
    if kb_path.exists():
        try:
            from ledgerhound.dictionary import SchemaKB
            agent.set_schema_kb(SchemaKB(str(kb_path)))
            console.print(f"[green]OK Schema KB:[/green] {kb_path}")
        except Exception as exc:
            console.print(f"[yellow]! Schema KB load failed: {exc}[/yellow]")
    elif explicit_kb_path:
        console.print(f"[yellow]! Schema KB not found at {kb_path} -- "
                      f"run 'ledgerhound dictionary load'[/yellow]")

    # -- Dataset Memory ---------------------------------------------------
    if dataset_name:
        try:
            from ledgerhound.memory import load_memory
            mem = load_memory(dataset_name)
            if mem is not None:
                agent.set_dataset_memory(mem)
                priors = (mem.sampling_design.priors_encoded
                          if mem.sampling_design else [])
                priors_str = (", ".join(priors[:3]) if priors
                               else "[dim](none)[/dim]")
                console.print(
                    f"[green]OK Memory:[/green] {dataset_name}  "
                    f"[dim](role={mem.role.value}, priors={priors_str})[/dim]"
                )
            else:
                console.print(
                    f"[yellow]! No memory for '{dataset_name}'.[/yellow]\n"
                    f"[dim]  Bootstrap one with: "
                    f"ledgerhound memory init {dataset_name}[/dim]"
                )
        except Exception as exc:
            console.print(f"[yellow]! Memory load failed: {exc}[/yellow]")

    # -- Findings log -----------------------------------------------------
    if explicit_findings_path or dataset_name:
        try:
            from ledgerhound.findings import FindingsLog
            findings_p = Path(
                explicit_findings_path
                or f".ledgerhound/findings/{dataset_name}.jsonl"
            )
            agent.set_findings_log(FindingsLog(findings_p))
            console.print(f"[green]OK Findings log:[/green] {findings_p}")
        except Exception as exc:
            console.print(f"[yellow]! Findings log setup failed: {exc}[/yellow]")

    # -- Journal directory ------------------------------------------------
    if explicit_journal_dir:
        agent._journal_dir = explicit_journal_dir
    elif dataset_name:
        # Park journals under the canonical .ledgerhound root
        agent._journal_dir = ".ledgerhound/journals"
    agent._write_journal = bool(write_journal)
    if write_journal:
        console.print(f"[green]OK Journals:[/green] {agent._journal_dir}")


@app.command("eda-multi")
def eda_multi(
    fraud_data: str = typer.Option(..., "--fraud-data", help="Fraud / txn parquet"),
    payee_data: str = typer.Option(..., "--payee-data", help="Payee / dim parquet"),
    join_key: str = typer.Option("payee_id", "--join-key",
                                  help="Column to merge on (default: payee_id)"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model:  Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    fraud_memory: Optional[str] = typer.Option(
        None, "--fraud-memory",
        help="Memory name for the fraud dataset (default: fraud-data file stem)",
    ),
    payee_memory: Optional[str] = typer.Option(
        None, "--payee-memory",
        help="Memory name for the payee dataset (default: payee-data file stem)",
    ),
    findings_path: Optional[str] = typer.Option(
        None, "--findings",
        help="Shared findings JSONL path "
             "(default: .ledgerhound/findings/<fraud-stem>__merged.jsonl)",
    ),
    kb_path: Optional[str] = typer.Option(
        None, "--kb", help="Schema KB path (default: .ledgerhound/upi_kb.duckdb)",
    ),
    journal_dir: Optional[str] = typer.Option(
        None, "--journal-dir",
        help="Notebook journal directory (default: .ledgerhound/journals)",
    ),
    no_journal: bool = typer.Option(False, "--no-journal"),
    no_synthesise: bool = typer.Option(
        False, "--no-synthesise",
        help="Skip the final executive-summary LLM call",
    ),
    report: Optional[str] = typer.Option(
        None, "--report", "-r",
        help="Markdown report path (incrementally saved between phases)",
    ),
):
    """Run the three-phase fraud x payee x merged-verdict playbook.

    Phase A runs fraud-side questions on the txn dataset.
    Phase B runs payee-side questions on the dimension dataset.
    Phase C joins them and uses earlier findings as priors.

    All three phases share one FindingsLog so structured findings compound.
    """
    from ledgerhound.agents.two_dataset import build_two_dataset_playbook

    _check_file(fraud_data, "Fraud data file")
    _check_file(payee_data, "Payee data file")
    if llm_config:
        _check_file(llm_config, "LLM config file")
    if not model and not llm_config:
        console.print("[red]Provide --model or --llm-config[/red]")
        raise typer.Exit(1)

    # Build LLM (re-use the same factory as `ledgerhound eda`)
    if llm_config:
        from ledgerhound.llm.client import LLMClient
        llm = LLMClient.from_config(llm_config)
    else:
        from ledgerhound.llm.client import LLMClient
        llm = LLMClient(
            base_url=model, model=model_name or "default",
            api_key=api_key or "EMPTY",
        )

    h = llm.health_check()
    if h.get("status") != "healthy":
        console.print(f"[red]LLM not available: {h.get('error', h)}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]OK LLM: {h.get('models', ['?'])[0]}[/green]")

    # Resolve memories + KB + findings log
    fraud_mem_name = fraud_memory or Path(fraud_data).stem
    payee_mem_name = payee_memory or Path(payee_data).stem

    fraud_mem = payee_mem = None
    try:
        from ledgerhound.memory import load_memory
        fraud_mem = load_memory(fraud_mem_name)
        payee_mem = load_memory(payee_mem_name)
        if fraud_mem:
            console.print(f"[green]OK Fraud memory:[/green] {fraud_mem_name}")
        else:
            console.print(
                f"[yellow]! No fraud memory for {fraud_mem_name!r}; "
                f"sample-frame discipline will be relaxed.[/yellow]"
            )
        if payee_mem:
            console.print(f"[green]OK Payee memory:[/green] {payee_mem_name}")
    except Exception as exc:
        console.print(f"[yellow]! Memory load failed: {exc}[/yellow]")

    schema_kb = None
    kb_p = Path(kb_path or ".ledgerhound/upi_kb.duckdb")
    if kb_p.exists():
        try:
            from ledgerhound.dictionary import SchemaKB
            schema_kb = SchemaKB(str(kb_p))
            console.print(f"[green]OK Schema KB:[/green] {kb_p}")
        except Exception as exc:
            console.print(f"[yellow]! Schema KB load failed: {exc}[/yellow]")

    findings_p = Path(
        findings_path
        or f".ledgerhound/findings/{fraud_mem_name}__merged.jsonl"
    )
    from ledgerhound.findings import FindingsLog
    findings_log = FindingsLog(findings_p)
    console.print(f"[green]OK Findings log:[/green] {findings_p}")

    journal_d = journal_dir or ".ledgerhound/journals"
    if not no_journal:
        console.print(f"[green]OK Journals:[/green] {journal_d}")

    console.print(Panel(
        f"[bold]Two-dataset playbook[/bold]\n"
        f"  fraud: [cyan]{fraud_data}[/cyan]\n"
        f"  payee: [cyan]{payee_data}[/cyan]\n"
        f"  join:  [cyan]{join_key}[/cyan]",
        border_style="cyan",
    ))

    try:
        playbook = build_two_dataset_playbook(
            fraud_data=fraud_data,
            payee_data=payee_data,
            join_key=join_key,
            llm=llm,
            fraud_memory=fraud_mem,
            payee_memory=payee_mem,
            findings_log=findings_log,
            schema_kb=schema_kb,
            journal_dir=journal_d,
            write_journal=not no_journal,
            interactive=True,
        )
    except KeyError as exc:
        console.print(f"[red]X Join error: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]X Playbook setup failed: {exc}[/red]")
        raise typer.Exit(1)

    if no_synthesise:
        playbook.synthesise = False

    result = playbook.run(report_path=report)

    if not report:
        result.display()

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{result.total_findings} findings . "
        f"{result.total_charts} charts . "
        f"{result.total_warnings} warnings . "
        f"{result.elapsed_ms / 1000:.1f}s"
    )


def _interactive(engine, digest):
    console.print(Panel("[bold]Ledgerhound EDA -- Interactive[/bold]\n"
                        "Type questions. 'quit' to exit. 'summary' for overview.",
                        border_style="blue"))
    console.print(f"\n{digest.summary()}\n")
    while True:
        try:
            q = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q.lower() in ("quit","exit","q"): break
        if q.lower() == "summary": console.print(digest.summary()); continue
        try:
            r = engine.eda.ask(q)
            r.display()
            console.print(f"[dim]({r.tool_calls} tools, {r.total_tokens:,} tok)[/dim]\n")
        except Exception as e:
            console.print(f"[red]{e}[/red]\n")


@app.command()
def health(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
):
    """Check LLM health."""
    if llm_config:
        _check_file(llm_config, "LLM config")
        from ledgerhound.core import _build_llm_from_config
        client = _build_llm_from_config(Path(llm_config))
    elif model:
        from ledgerhound.llm.client import LLMClient
        client = LLMClient(base_url=model, api_key=api_key or "EMPTY")
    else:
        console.print("[red]Provide --model or --llm-config[/red]")
        raise typer.Exit(1)

    r = client.health_check()
    if r["status"] == "healthy":
        console.print("[green]OK Healthy[/green]")
        for m in r.get("models",[]): console.print(f"  {m}")
    else:
        console.print(f"[red]X {r.get('error', r)}[/red]")


@app.command()
def diff(
    before: str = typer.Argument(..., help="Path to the BEFORE digest YAML"),
    after:  str = typer.Argument(..., help="Path to the AFTER digest YAML"),
    output: Optional[str] = typer.Option(None, "--output", "-o",
                                          help="Save diff report to file"),
):
    """Compare two digests -- shows what changed when you filtered the data."""
    from ledgerhound.profiler.diff import DigestDiffer
    from ledgerhound.profiler.digest import Digest

    _check_file(before, "Before digest")
    _check_file(after, "After digest")

    try:
        d_before = Digest.load(before)
        d_after = Digest.load(after)
    except Exception as e:
        console.print(f"[red]Error loading digests: {e}[/red]")
        raise typer.Exit(1)

    differ = DigestDiffer()
    report = differ.diff(d_before, d_after)

    # Display summary
    console.print(Panel(
        f"[bold]Ledgerhound Diff[/bold]\n"
        f"Before: {report.before_source}\n"
        f"After:  {report.after_source}\n"
        f"Changed: {report.summary_stats.get('changed', 0)} columns  |  "
        f"Added: {report.summary_stats.get('added', 0)}  |  "
        f"Removed: {report.summary_stats.get('removed', 0)}",
        border_style="yellow",
    ))

    # Overview changes
    if report.overview_changes:
        from rich.table import Table
        t = Table(title="Overview Changes", border_style="yellow")
        t.add_column("Metric"); t.add_column("Before"); t.add_column("After"); t.add_column("Delta")
        for c in report.overview_changes:
            delta = c.get("delta", "")
            delta_str = f"{delta:+,}" if isinstance(delta, (int, float)) and delta != 0 else str(delta)
            t.add_row(str(c["metric"]), str(c["before"]), str(c["after"]), delta_str)
        console.print(t)

    # Quality changes
    if report.quality_changes:
        console.print("\n[bold]Quality Changes:[/bold]")
        for c in report.quality_changes:
            console.print(f"  {c['metric']}: {c['before']} -> {c['after']}")

    # Priority shifts
    if report.priority_changes:
        console.print(f"\n[bold]Priority Shifts ({len(report.priority_changes)}):[/bold]")
        for c in report.priority_changes[:10]:
            console.print(f"  {c['column']}: [dim]{c['before']}[/dim] -> [bold]{c['after']}[/bold]")
        if len(report.priority_changes) > 10:
            console.print(f"  ... and {len(report.priority_changes) - 10} more")

    # Top column changes
    changed = [d for d in report.column_diffs.values() if d.status == "changed"]
    if changed:
        console.print(f"\n[bold]Column Stats Changed ({len(changed)}):[/bold]")
        for cd in sorted(changed, key=lambda x: len(x.changes), reverse=True)[:10]:
            metrics = ", ".join(c["metric"] for c in cd.changes[:3])
            console.print(f"  [cyan]{cd.name}[/cyan]: {metrics}")

    # Save
    if output:
        report.save(output)
        console.print(f"\n[green]Diff report saved: {output}[/green]")
    elif not report.overview_changes and not report.quality_changes and not changed:
        console.print("\n[dim]No significant changes detected.[/dim]")


# ============================================================
# `ledgerhound dictionary` -- Schema Knowledge Base management
# ============================================================

dict_app = typer.Typer(
    name="dictionary",
    help="Schema KB: load, look up columns, decode codes, see quality flags.",
    no_args_is_help=True,
)
app.add_typer(dict_app, name="dictionary")


_DEFAULT_KB_PATH = ".ledgerhound/upi_kb.duckdb"


def _open_kb(path: str):
    """Open the SchemaKB at ``path``, creating the parent dir if needed."""
    from ledgerhound.dictionary import SchemaKB

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SchemaKB(path)


@dict_app.command("load")
def dict_load(
    db_path: str = typer.Option(_DEFAULT_KB_PATH, "--db", help="DuckDB path for the KB"),
    dictionary_csv: Optional[str] = typer.Option(None, "--dict-csv",
                                                  help="Override path to data dictionary CSV"),
    columns_txt: Optional[str] = typer.Option(None, "--columns-txt",
                                              help="Override path to columns_all.txt"),
    purpose_codes_csv: Optional[str] = typer.Option(None, "--purpose-csv",
                                                    help="Override path to purpose codes CSV"),
    dataset: str = typer.Option("upi_ft", "--dataset", help="Dataset label for these columns"),
):
    """Load the organization artifacts (dictionary + DB columns + purpose codes) into the KB.

    Idempotent -- safe to re-run when source files change.
    """
    from ledgerhound.dictionary.loader import (
        DEFAULT_COLUMNS_TXT,
        DEFAULT_DICTIONARY_CSV,
        DEFAULT_PURPOSE_CODES_CSV,
        load_defaults,
    )

    kb = _open_kb(db_path)
    paths = {
        "dictionary_csv":    dictionary_csv    or DEFAULT_DICTIONARY_CSV,
        "columns_txt":       columns_txt       or DEFAULT_COLUMNS_TXT,
        "purpose_codes_csv": purpose_codes_csv or DEFAULT_PURPOSE_CODES_CSV,
    }

    console.print(Panel(
        f"[bold]Loading Schema KB[/bold]\n"
        f"db:       {db_path}\n"
        f"dataset:  {dataset}\n"
        f"dict_csv: {paths['dictionary_csv']}\n"
        f"db_cols:  {paths['columns_txt']}\n"
        f"purpose:  {paths['purpose_codes_csv']}",
        border_style="cyan",
    ))

    try:
        counts = load_defaults(kb, dataset=dataset, **paths)
    except FileNotFoundError as e:
        console.print(f"[red]X {e}[/red]")
        console.print("[dim]Place the file at the path shown, or pass --dict-csv / "
                      "--columns-txt / --purpose-csv to override.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]X Load failed: {e}[/red]")
        raise typer.Exit(1)

    from rich.table import Table
    t = Table(title="Schema KB -- load summary", border_style="green")
    t.add_column("Item"); t.add_column("Count", justify="right")
    for k, v in counts.items():
        t.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    console.print(t)
    console.print(f"\n[bold green]OK Schema KB ready at {db_path}[/bold green]")


@dict_app.command("lookup")
def dict_lookup(
    name: str = typer.Argument(..., help="Column name (canonical / db / dict form)"),
    db_path: str = typer.Option(_DEFAULT_KB_PATH, "--db"),
):
    """Resolve any column name (canonical / db / dict) to its full record."""
    kb = _open_kb(db_path)
    col = kb.lookup(name)
    if col is None:
        console.print(f"[red]X No column matching {name!r}[/red]")
        console.print("[dim]Try `ledgerhound dictionary search --contains <substr>`[/dim]")
        raise typer.Exit(1)

    aliases = kb.aliases(name)
    flags = kb.quality_flags(name)

    body = (
        f"[bold]{col.display_name}[/bold]  ([dim]{col.canonical_name}[/dim])\n"
        f"dataset:        {col.dataset or '-'}\n"
        f"db_alias:       {col.db_alias or '-'}\n"
        f"dict_name:      {col.dict_name or '-'}\n"
        f"semantic_type:  {col.semantic_type or '-'}\n"
        f"role:           {col.role or '-'}\n"
        f"status:         {col.status.value}\n"
        f"population:     {col.population.value}"
        + ("  [yellow](quality concerns)[/yellow]" if col.has_quality_concerns else "")
        + f"\nis_active:      {col.is_active}\n"
        f"comments:       {col.comments or '-'}"
    )
    console.print(Panel(body, border_style="cyan", title="Column"))

    if aliases:
        console.print(f"\n[bold]Aliases:[/bold] {', '.join(aliases)}")
    if flags:
        console.print(f"\n[bold]Quality flags ({len(flags)}):[/bold]")
        for f in flags:
            console.print(f"  [yellow]{f.severity.value}[/yellow]  {f.issue}")


@dict_app.command("search")
def dict_search(
    contains: Optional[str] = typer.Option(None, "--contains", "-c"),
    dataset:  Optional[str] = typer.Option(None, "--dataset", "-d"),
    semantic_type: Optional[str] = typer.Option(None, "--semantic-type", "-s"),
    status:   Optional[str] = typer.Option(None, "--status",
                                            help="mandatory | optional | new | deprecated"),
    limit:    int = typer.Option(50, "--limit", "-n"),
    db_path:  str = typer.Option(_DEFAULT_KB_PATH, "--db"),
):
    """Search columns by substring / type / status."""
    kb = _open_kb(db_path)
    results = kb.search(
        contains=contains, dataset=dataset, semantic_type=semantic_type, status=status,
    )
    if not results:
        console.print("[dim]No matches.[/dim]")
        return

    from rich.table import Table
    t = Table(title=f"{len(results)} matches"
                    + (f" (showing {limit})" if len(results) > limit else ""),
              border_style="cyan")
    t.add_column("canonical"); t.add_column("db_alias"); t.add_column("semantic")
    t.add_column("role"); t.add_column("population"); t.add_column("active")
    for c in results[:limit]:
        t.add_row(c.canonical_name, c.db_alias or "-",
                  c.semantic_type or "-", c.role or "-",
                  c.population.value, "Y" if c.is_active else "n")
    console.print(t)


@dict_app.command("decode")
def dict_decode(
    code: str = typer.Argument(..., help="Reference code (e.g. 41)"),
    code_type: str = typer.Option("purpose_code", "--type", "-t",
                                  help="purpose_code | mcc | initiation_mode"),
    db_path: str = typer.Option(_DEFAULT_KB_PATH, "--db"),
):
    """Translate a reference code to its description."""
    kb = _open_kb(db_path)
    rc = kb.reference_code(code, code_type)
    if rc is None:
        console.print(f"[red]X No {code_type} {code!r}[/red]")
        raise typer.Exit(1)
    console.print(Panel(
        f"[bold]{rc.code}[/bold]  ({rc.code_type})\n"
        f"description: {rc.description}\n"
        f"product:     {rc.product or '-'}\n"
        f"notes:       {rc.notes or '-'}",
        border_style="green", title="Reference code",
    ))


@dict_app.command("stats")
def dict_stats(db_path: str = typer.Option(_DEFAULT_KB_PATH, "--db")):
    """Counts of columns, aliases, quality flags, codes, relationships."""
    kb = _open_kb(db_path)
    s = kb.stats()
    from rich.table import Table
    t = Table(title="Schema KB -- stats", border_style="cyan")
    t.add_column("Item"); t.add_column("Count", justify="right")
    for k, v in s.items():
        t.add_row(k, f"{v:,}")
    console.print(t)


@dict_app.command("flags")
def dict_flags(
    name: str = typer.Argument(..., help="Column name"),
    db_path: str = typer.Option(_DEFAULT_KB_PATH, "--db"),
):
    """Show all data-quality flags for a column."""
    kb = _open_kb(db_path)
    canonical = kb._resolve_canonical(name)
    if canonical is None:
        console.print(f"[red]X No column matching {name!r}[/red]")
        raise typer.Exit(1)

    flags = kb.quality_flags(name)
    if not flags:
        console.print(f"[green]OK No documented quality issues for {canonical}[/green]")
        return

    from rich.table import Table
    t = Table(title=f"Quality flags -- {canonical}", border_style="yellow")
    t.add_column("Severity"); t.add_column("Issue"); t.add_column("Source")
    for f in flags:
        t.add_row(f.severity.value, f.issue, f.source_file or "-")
    console.print(t)


# ============================================================
# `ledgerhound memory` -- Dataset Memory bootstrap & inspection (P1.D)
# ============================================================

memory_app = typer.Typer(
    name="memory",
    help="Dataset Memory: bootstrap (HITL), inspect, list.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


_DEFAULT_MEMORY_DIR = ".ledgerhound/memory"


@memory_app.command("init")
def memory_init(
    dataset_name: str = typer.Argument(..., help="Dataset identifier"),
    data: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="Optional data file (parquet/csv/xlsx) to auto-infer schema_columns",
    ),
    from_yaml: Optional[str] = typer.Option(
        None, "--from-yaml",
        help="Load a draft YAML instead of running the interactive interview",
    ),
    role: Optional[str] = typer.Option(
        None, "--role",
        help="Pre-set role: fact | dimension | rollup | unknown "
             "(skips role prompt in interactive mode; sets role in --from-yaml)",
    ),
    base_dir: str = typer.Option(
        _DEFAULT_MEMORY_DIR, "--memory-dir",
        help="Where to write the YAML",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite an existing memory file without prompting",
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive",
        help="Skip the interview entirely; produce a stub from --data + --role",
    ),
):
    """Bootstrap a Dataset Memory file via HITL interview, YAML import, or stub.

    Three modes:

    1. **Interactive (default)**: walks through role -> sampling strata ->
       priors -> schema columns.  Use this for first-run onboarding.

    2. **`--from-yaml draft.yaml`**: load a pre-written draft (CI / repo
       check-in friendly).

    3. **`--non-interactive --data file`**: produce a minimal stub by
       inferring schema_columns from the data file. Sampling design is
       left empty; the analyst can edit the YAML later.
    """
    from ledgerhound.memory import (
        DatasetRole,
        bootstrap_from_data,
        bootstrap_from_yaml,
        interview,
        save_memory,
    )

    out_path = Path(base_dir) / f"{dataset_name}.yaml"
    if out_path.exists() and not force:
        console.print(
            f"[yellow]Memory already exists: {out_path}[/yellow]\n"
            f"[dim]Pass --force to overwrite, or "
            f"`ledgerhound memory show {dataset_name}` to inspect.[/dim]"
        )
        raise typer.Exit(1)

    # Resolve role (used by --from-yaml override and --non-interactive)
    role_enum: Optional[DatasetRole] = None
    if role:
        try:
            role_enum = DatasetRole(role)
        except ValueError:
            console.print(
                f"[red]Invalid --role {role!r}. "
                f"Use one of: fact, dimension, rollup, unknown[/red]"
            )
            raise typer.Exit(1)

    # -- Mode dispatch ------------------------------------------------------
    try:
        if from_yaml:
            overrides: dict = {"dataset_name": dataset_name}
            if role_enum:
                overrides["role"] = role_enum.value
            mem = bootstrap_from_yaml(from_yaml, overrides=overrides)
            console.print(
                f"[green]OK[/green] Loaded draft from "
                f"[cyan]{from_yaml}[/cyan]"
            )
        elif non_interactive:
            if not data:
                console.print(
                    "[red]--non-interactive requires --data <file> "
                    "to infer schema columns.[/red]"
                )
                raise typer.Exit(1)
            mem = bootstrap_from_data(
                dataset_name,
                data,
                role=role_enum or DatasetRole.UNKNOWN,
            )
            console.print(
                f"[green]OK[/green] Inferred {len(mem.schema_columns)} "
                f"schema columns from [cyan]{data}[/cyan]"
            )
        else:
            mem = interview(dataset_name, data_path=data)
            if role_enum:
                mem.role = role_enum
    except FileNotFoundError as e:
        console.print(f"[red]X {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]X {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interview cancelled -- no file written.[/yellow]")
        raise typer.Exit(130)

    saved_to = save_memory(mem, base_dir=Path(base_dir))
    console.print(
        Panel(
            f"[green]Memory saved:[/green] [cyan]{saved_to}[/cyan]\n"
            f"role={mem.role.value}  "
            f"columns={len(mem.schema_columns)}  "
            f"strata={len(mem.sampling_design.strata) if mem.sampling_design else 0}  "
            f"priors={len(mem.sampling_design.priors_encoded) if mem.sampling_design else 0}",
            title="Ledgerhound Memory",
            border_style="green",
        )
    )


@memory_app.command("show")
def memory_show(
    dataset_name: str = typer.Argument(..., help="Dataset identifier"),
    base_dir: str = typer.Option(
        _DEFAULT_MEMORY_DIR, "--memory-dir",
        help="Directory containing memory YAMLs",
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Dump raw YAML instead of a formatted view",
    ),
):
    """Pretty-print a Dataset Memory file."""
    from ledgerhound.memory import load_memory

    mem = load_memory(dataset_name, base_dir=Path(base_dir))
    if mem is None:
        console.print(
            f"[yellow]No memory found for {dataset_name!r} in "
            f"{base_dir}/.[/yellow]\n"
            f"[dim]Bootstrap one with: "
            f"ledgerhound memory init {dataset_name}[/dim]"
        )
        raise typer.Exit(1)

    if raw:
        import yaml as _yaml
        console.print(_yaml.safe_dump(mem.to_dict(), sort_keys=False))
        return

    from rich.table import Table

    # Header
    console.print(Panel(
        f"[bold cyan]{mem.dataset_name}[/bold cyan]   "
        f"[dim]role={mem.role.value}[/dim]\n"
        f"{mem.description or '[dim](no description)[/dim]'}\n\n"
        f"[dim]created  {mem.created_at}\n"
        f"updated  {mem.updated_at}[/dim]",
        title="Dataset Memory",
        border_style="cyan",
    ))

    # Schema columns
    if mem.schema_columns:
        cols_preview = ", ".join(mem.schema_columns[:8])
        more = (f"  ...and {len(mem.schema_columns) - 8} more"
                if len(mem.schema_columns) > 8 else "")
        console.print(f"[bold]Schema columns ({len(mem.schema_columns)}):[/bold] "
                      f"{cols_preview}{more}")

    # Sampling design
    if mem.sampling_design:
        sd = mem.sampling_design
        if sd.strata:
            t = Table(title="Sampling strata", border_style="green")
            t.add_column("Stratum")
            t.add_column("Method")
            t.add_column("n_sample", justify="right")
            t.add_column("p_inclusion", justify="right")
            t.add_column("Filters")
            for name, s in sd.strata.items():
                t.add_row(
                    name, s.method.value,
                    f"{s.n_sample:,}" if s.n_sample is not None else "-",
                    f"{s.p_inclusion:.2e}" if s.p_inclusion is not None else "-",
                    ", ".join(s.filters) if s.filters else "-",
                )
            console.print(t)
        if sd.priors_encoded:
            console.print(f"[bold]Priors encoded:[/bold] "
                          f"{', '.join(sd.priors_encoded)}")
        if sd.stale_if:
            console.print(f"[bold]Stale-if:[/bold] {', '.join(sd.stale_if)}")
    else:
        console.print("[dim](no sampling design recorded)[/dim]")

    # Relationships
    if mem.relationships:
        t = Table(title="Relationships", border_style="blue")
        t.add_column("From"); t.add_column("To"); t.add_column("Kind")
        for r in mem.relationships:
            t.add_row(
                f"{r.from_dataset}.{r.from_column}",
                f"{r.to_dataset}.{r.to_column}",
                r.kind,
            )
        console.print(t)

    # Findings log path
    if mem.findings_log_path:
        console.print(f"[dim]Findings log: {mem.findings_log_path}[/dim]")


@memory_app.command("list")
def memory_list(
    base_dir: str = typer.Option(
        _DEFAULT_MEMORY_DIR, "--memory-dir",
        help="Directory containing memory YAMLs",
    ),
):
    """List every dataset that has a memory YAML on disk."""
    from ledgerhound.memory import list_datasets, load_memory

    names = list_datasets(base_dir)
    if not names:
        console.print(
            f"[yellow]No memory files found in {base_dir}.[/yellow]\n"
            f"[dim]Bootstrap one with: ledgerhound memory init <dataset>[/dim]"
        )
        return

    from rich.table import Table
    t = Table(title=f"Dataset Memories ({base_dir})", border_style="cyan")
    t.add_column("Dataset")
    t.add_column("Role")
    t.add_column("Columns", justify="right")
    t.add_column("Strata", justify="right")
    t.add_column("Priors", justify="right")
    t.add_column("Updated")
    for name in names:
        mem = load_memory(name, base_dir=Path(base_dir))
        if mem is None:
            t.add_row(name, "[red]ERR[/red]", "-", "-", "-", "-")
            continue
        n_strata = (len(mem.sampling_design.strata)
                    if mem.sampling_design else 0)
        n_priors = (len(mem.sampling_design.priors_encoded)
                    if mem.sampling_design else 0)
        t.add_row(
            name,
            mem.role.value,
            f"{len(mem.schema_columns)}",
            f"{n_strata}",
            f"{n_priors}",
            mem.updated_at[:19].replace("T", " "),
        )
    console.print(t)


# ============================================================
# `ledgerhound workspace` -- workspace initialisation & status
# ============================================================

workspace_app = typer.Typer(
    name="workspace",
    help="Initialise and inspect a Ledgerhound analysis workspace.",
    no_args_is_help=True,
)
app.add_typer(workspace_app, name="workspace")


# -- project layer (Phase 1 agentic redesign) --------------------------------
project_app = typer.Typer(
    name="project",
    help="Projects (.ledgerhound/): new, list, and inspect -- Ledgerhound's CLAUDE.md-style project layer.",
    no_args_is_help=True,
)
app.add_typer(project_app, name="project")


@project_app.command("new")
def project_new(
    name: str = typer.Argument(..., help="Project name (folder created under home)"),
    home: str = typer.Option(".", "--home", help="Parent folder to create the project in"),
    save_format: str = typer.Option("md", "--save-format", help="Default output format: md|ipynb|pdf|html"),
):
    """Create a new project: home/<name>/.ledgerhound/ with project.json + LEDGERHOUND.md."""
    from ledgerhound.identity import current_user
    from ledgerhound.project import ProjectStore
    try:
        owner = current_user()
    except Exception:
        owner = ""
    p = ProjectStore(home).new(name, owner=owner, save_format=save_format)
    typer.echo(f"created project {p.config().name!r} at {p.root}")
    typer.echo(f"  memory:   {p.memory_path}")
    typer.echo(f"  sessions: {p.sessions_root}")


@project_app.command("list")
def project_list(
    home: str = typer.Option(".", "--home", help="Parent folder to scan for projects"),
):
    """List projects under home (folders containing .ledgerhound/)."""
    from ledgerhound.project import ProjectStore
    projects = ProjectStore(home).list()
    if not projects:
        typer.echo("no projects found (create one with: ledgerhound project new <name>)")
        return
    for p in projects:
        cfg = p.config()
        typer.echo(f"  {cfg.name:<24} {p.root}  [save={cfg.save_format}]")


def find_project_repo(project_root: Path, launch_dir: Path):
    """The git repo the shell's git tools act on, or None.

    L-01: the project's own repo, or the launch folder's repo ONLY when the
    project lives inside it. Launched from some other checkout (the Ledgerhound
    source tree, a colleague's repo), "push my work" used to commit and push
    that unrelated repository."""
    from ledgerhound.integrations.git import GitClient
    proj = Path(project_root).resolve()
    for candidate in (Path(project_root), Path(launch_dir)):
        try:
            proj.relative_to(candidate.resolve())
        except ValueError:
            continue
        c = GitClient(candidate)
        if c.available():
            return c
    return None


def _auto_detect_use_kernel() -> tuple[bool, str, bool]:
    """Decide kernel-vs-sandbox when the user didn't pass --use-kernel.

    Prefers the real IPython kernel (the only execution path with a cuDF
    branch -- ``ledgerhound/auto/sandbox.py`` is pandas-only, no GPU code at
    all) whenever a GPU is actually present AND the kernel deps are
    installed. Best-effort: any detection failure means "off", never crashes
    shell startup over a GPU probe -- the sandbox is always a safe fallback.

    Returns (use_kernel, reason, gpu_stranded) -- ``gpu_stranded`` is True
    when a GPU IS present but this session is about to fall back to the
    CPU-only sandbox anyway (deps missing, or detection itself failed). That
    case used to be indistinguishable from "no GPU, working as intended" in
    a dim one-line log -- V3-050 makes the caller surface it loudly instead.
    """
    try:
        from ledgerhound.auto.kernel import kernels_available
        from ledgerhound.backends.compute_router import ComputeRouter
        n_gpus = ComputeRouter().detect().gpu_count
        if not kernels_available():
            return False, "jupyter_client/ipykernel not installed", n_gpus > 0
        if n_gpus > 0:
            return True, f"{n_gpus} GPU(s) detected", False
        return False, "no GPU detected", False
    except Exception as e:  # noqa: BLE001 -- detection is advisory, never fatal
        return (False, f"detection failed ({type(e).__name__}), defaulting to sandbox",
                False)


@app.command("shell")
def shell(
    root: str = typer.Option("", "--project", "-p",
                             help="Project root (folder for .ledgerhound/); created if absent. "
                                  "Default: ../Agentic_eda_results/<username>/library"),
    name: str = typer.Option("", "--name", help="Project name when creating"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config",
                             help="LLM config yaml (else the project's /model, or configs/deepseek.yaml)"),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Pin the model id to use (e.g. 'Kimi-k2.5-thinking'), skipping "
             "discovery/selection entirely. Omit to have Ledgerhound list what "
             "the configured endpoint actually serves and pick the best one "
             "(interactively if you're at a terminal, silently otherwise)."),
    out: Optional[str] = typer.Option(None, "--out",
                             help="Root for dated run folders (default: <project>/run_outputs, "
                                  "so reports always live inside the project)"),
    save_format: Optional[str] = typer.Option(None, "--save-format",
                             help="Report format: md|html|ipynb|pdf (default: project's saved "
                                  "setting, or md if never set)"),
    autonomous: bool = typer.Option(False, "--autonomous",
                             help="Dev/eval flag: auto-answer non-risk-tier questions "
                                  "('proceed with your best judgment') instead of "
                                  "surfacing them, so a run measures only genuine "
                                  "risk-tier pauses. Off by default."),
    methodology: str = typer.Option(
        "", "--methodology", "-m",
        help=("Analysis methodology to follow (e.g. topdown_7phase, "
              "sample_first_7phase, bottom_up_7phase). Omit to be asked "
              "once on first data load; required for unattended runs that "
              "should not sit on a prompt.")),
    mode: str = typer.Option(
        "", "--mode",
        help=("A SEPARATE axis from --methodology -- how far the pipeline "
              "goes, not the reasoning order. '' (default) = full "
              "sequence. eda_only = stop before feature construction. "
              "fe_only = skip exploration, start at feature construction. "
              "eda_fe_only = stop before modelling authorisation.")),
    use_kernel: Optional[bool] = typer.Option(None, "--use-kernel/--no-use-kernel",
                             help="Run analysis in a real persistent IPython kernel "
                                  "instead of the restricted in-process sandbox. "
                                  "REQUIRED for GPU/cuDF acceleration -- the sandbox "
                                  "is pandas-only. Default: auto -- on when a GPU is "
                                  "detected and jupyter_client+ipykernel are installed "
                                  "(pip install -e '.[test]'), off otherwise. Pass "
                                  "explicitly to override either direction; falls back "
                                  "to the sandbox automatically if the kernel deps are "
                                  "missing or the kernel fails to start."),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps",
        help=("Hard ceiling on tool calls in a single turn (default 400). "
              "Raise this for a deliberately deep, unattended exploration "
              "run where depth matters more than GPU/token cost -- the "
              "default is a runaway-loop safety net, not a pace-setter.")),
    step_call_budget: Optional[int] = typer.Option(
        None, "--step-call-budget",
        help="Soft nudge after this many tool calls on one plan step without "
             "touching the plan (default 60). Raise alongside --max-steps."),
    turn_budget_seconds: Optional[int] = typer.Option(
        None, "--turn-budget-seconds",
        help="Wall-clock budget per turn in seconds (default 14400 = 4h). "
             "Raise for a run expected to take longer than 4 hours."),
):
    """Open the LLM-driven analyst shell -- a Claude-Code-style conversation for
    data science. The model drives; just talk to it."""
    from ledgerhound.llm.client import LLMAuthError, LLMClient
    from ledgerhound.llm.credentials import clear_key, ensure_api_key, resolve_method
    from ledgerhound.project import Project
    from ledgerhound.shell.conversation import ConversationalAnalyst

    # Mandatory login, once per machine -- skipped entirely on the Kubeflow
    # production path (identity already declared via $LEDGERHOUND_USER) and
    # skipped after the first successful login here (identity.py stores it
    # and never asks again). Before anything else touches disk or the LLM.
    from ledgerhound.identity import ensure_login
    ensure_login(ask=lambda p: console.input(f"[bold cyan]{p}[/bold cyan]"),
                      say=lambda s: console.print(f"[dim]{s}[/dim]"))

    # Everything lands under ../Agentic_eda_results/<username>/ by default
    # (run_outputs, data_cache, library) -- resolved AFTER login so the
    # folder is the logged-in user's. See ledgerhound.paths.
    from ledgerhound import paths as _paths
    if not root:
        root = str(_paths.library_dir())
    for _d in (_paths.run_outputs_dir(), _paths.data_cache_dir(), _paths.library_dir()):
        try:
            _d.mkdir(parents=True, exist_ok=True)
        except OSError as _e:
            console.print(f"[yellow]cannot create {_d}: {_e} -- set paths.results_root "
                          "in main_config.yaml to a writable folder.[/yellow]")
    console.print(f"[dim]results: {_paths.user_dir()}  (run_outputs, data_cache, library)[/dim]")

    p = Project(root)
    if p.exists():
        # Read once up front so a damaged project.json is repaired, and SAID,
        # here -- not discovered as a crash in the middle of startup.
        p.config()
        if p.recovered_from:
            from rich.markup import escape as _esc0
            console.print(
                "[yellow]" + _esc0(
                    f"project settings file was damaged ({p.recovered_reason}) "
                    f"-- moved it to {p.recovered_from} and started with "
                    "defaults. Only project.json was replaced: LEDGERHOUND.md, "
                    "sessions and dataset cards are untouched. Re-apply "
                    "/model or the save format if you had changed them.")
                + "[/yellow]")

    # Resolve + validate + init the LLM BEFORE creating anything on disk, so a
    # bad config / missing key / aborted startup never litters an empty project
    # skeleton (AUD: shell created <project>/.ledgerhound before the LLM was
    # confirmed, leaving stray empty `mytest` folders in whatever cwd it was
    # launched from).
    # Default config file follows main_config.yaml's llm.method (self|
    # gateway|api, default "api") -- e.g. configs/gateway.yaml. Those three
    # files resolve their actual base_url/model/key from main_config.yaml
    # at load time (see ledgerhound.llm.client.LLMConfig.from_master_config);
    # nothing provider-specific is hardcoded here.
    from ledgerhound.config import resolved_path, search_candidates
    from ledgerhound.llm.client import _load_master_config
    # V3-064: main_config.yaml resolution used to be silent -- a config
    # filled in correctly but sitting somewhere the search order doesn't
    # check (wrong launch cwd, a non-editable install) failed with no
    # visible signal beyond "the value wasn't there." Announce it plainly.
    _mc_path = resolved_path()
    if _mc_path:
        console.print(f"[dim]main_config.yaml: {_mc_path}[/dim]")
    else:
        console.print(
            "[yellow]main_config.yaml not found -- checked: "
            f"{', '.join(str(c) for c in search_candidates())}. "
            "Using env vars / built-in defaults only. Copy "
            "main_config.example.yaml to main_config.yaml (at one of "
            "those paths, or set $LEDGERHOUND_MAIN_CONFIG) to configure the "
            "LLM/judge, knowledge dirs, GPU tuning, etc.[/yellow]")
    from ledgerhound.llm.client import normalise_method, resolve_config_ref
    _method = normalise_method(str((_load_master_config().get("llm") or {})
                                   .get("method") or "api"))
    _default_cfg_ref = f"configs/{_method}.yaml"
    if llm_config:
        cfg_ref = llm_config
    elif p.exists():
        cfg_ref = p.config().model_config_ref or _default_cfg_ref
        # A project can remember an LLM config file that no longer exists
        # (renamed or removed in an upgrade). Rather than refuse to start,
        # use the one main_config.yaml's llm.method names, and remember it.
        if (cfg_ref != _default_cfg_ref
                and not Path(resolve_config_ref(cfg_ref)).exists()
                and Path(resolve_config_ref(_default_cfg_ref)).exists()):
            console.print(f"[yellow]this project pointed at {cfg_ref}, which does "
                          f"not exist -- using {_default_cfg_ref} (llm.method: "
                          f"{_method}) and saving that.[/yellow]")
            try:
                _cfg = p.config()
                _cfg.model_config_ref = _default_cfg_ref
                p.save_config(_cfg)
            except Exception:  # noqa: BLE001 -- the fallback still applies
                pass
            cfg_ref = _default_cfg_ref
    else:
        cfg_ref = _default_cfg_ref
    _asked_ref = cfg_ref
    cfg_ref = resolve_config_ref(cfg_ref)
    if not Path(cfg_ref).exists():
        import ledgerhound as _pkg
        _install = Path(_pkg.__file__).resolve().parents[1]
        console.print(f"[red]LLM config not found: {_asked_ref}[/red]\n"
                      f"Looked in {Path.cwd()} and in the Ledgerhound checkout "
                      f"({_install}). Pass --llm-config <yaml> or set one in "
                      "the project.")
        raise typer.Exit(1)
    # Ask for an API key only if genuinely nothing is set anywhere (main_
    # config.yaml, a prior stored login, or a legacy config's own
    # api_key_env) -- never re-asks once one is on file. See
    # ledgerhound.llm.credentials module docstring.
    from ledgerhound.llm.credentials import ensure_configured, ensure_section_configured
    ensure_configured(cfg_ref, ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]", password=True),
                      say=lambda s: console.print(f"[dim]{s}[/dim]"))
    # Same ask-once treatment for the JUDGE (I-03): a judge.method that
    # needs a key used to fall through to a silent self-judge when the key
    # was blank -- nothing ever asked for it.
    ensure_section_configured("judge",
                              ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]", password=True),
                              say=lambda s: console.print(f"[dim]{s}[/dim]"))
    try:
        llm = LLMClient.from_config(cfg_ref)
    except Exception as e:
        console.print(f"[red]Could not init LLM from {cfg_ref}: {e}[/red]")
        raise typer.Exit(1)

    # Model selection: list what the configured endpoint actually serves
    # and pick the best one, so a gateway (e.g. the org's gateway) fronting
    # several models doesn't silently run whatever main_config.yaml's
    # model field happened to say (or nothing, if it was left blank).
    # --model skips discovery entirely; autonomous/non-interactive runs
    # auto-pick the best one without a prompt (never blocks a run).
    from ledgerhound.llm.catalog import context_window_for
    if model:
        llm.config.model = model
        console.print(f"[dim]model: {model} (pinned via --model)[/dim]")
    else:
        from ledgerhound.llm.catalog import choose_model
        chosen = choose_model(
            llm, autonomous=autonomous,
            ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]"),
            say=lambda s: console.print(f"[dim]{s}[/dim]"))
        if chosen:
            llm.config.model = chosen


    def _load_tokenizer(model_id: str):
        """Exact token counting for this model (ledgerhound.llm.tokenizer).
        Never blocks: on any failure it prints WHY, the fix, and what the
        session uses instead, then carries on with calibrated estimates."""
        from rich.markup import escape as _esc
        from ledgerhound.llm.tokenizer import resolve_tokenizer
        counter, status = resolve_tokenizer(model_id)
        if status.ok:
            console.print(f"[dim]{_esc(status.render())}[/dim]")
        else:
            console.print(f"[yellow]{_esc(status.render())}[/yellow]")
        return counter, status

    _tok_counter, _tok_status = _load_tokenizer(llm.config.model)

    def _apply_context_window(client, *, say, hint=None) -> None:
        """Size compaction to the model in use (ledgerhound.llm.catalog.
        apply_context_window): main_config override, then the gateway's
        reported limit, then the model's own config (from the tokenizer
        manifest) or the table -- capped until the gateway confirms."""
        from ledgerhound.llm.catalog import apply_context_window
        say(apply_context_window(client, hint=hint))

    _apply_context_window(llm, say=lambda s: console.print(f"[dim]{s}[/dim]"),
                          hint=(_tok_status.details or {}).get("context_window"))

    # LLM is good -- now it's safe to create/touch the project on disk.
    if not p.exists():
        from ledgerhound.identity import current_user
        try:
            owner = current_user()
        except Exception:
            owner = ""
        p = Project.init(root, name=name, owner=owner,
                         save_format=save_format or "md")
    elif save_format:
        cfg = p.config()
        cfg.save_format = save_format
        p.save_config(cfg)

    # Reports live INSIDE the project by default (<project>/run_outputs), so
    # they show up alongside .ledgerhound/ where a user actually looks -- a bare
    # relative "run_outputs" used to resolve against the launch cwd instead.
    # V3-066: paths.run_outputs_dir in main_config.yaml forces this
    # somewhere else entirely (e.g. a shared drive) -- --out still wins if
    # explicitly passed on the command line.
    if out:
        out_root = Path(out)
    else:
        # paths.run_outputs_dir if set, else Agentic_eda_results/<user>/run_outputs
        out_root = _paths.run_outputs_dir()

    from rich.markdown import Markdown
    from rich.markup import escape as _rescape
    from rich.panel import Panel as _Panel

    def _status(s: str) -> None:
        # dim tool/thinking activity; plan checklists keep their shape.
        # `s` is model/tool-generated text and can contain `[...]` (e.g.
        # `df[col]`, or a `[/dim]`-looking sequence) -- escape it so Rich
        # renders it verbatim instead of mangling it or raising a markup
        # error (which on a long session would crash the shell mid-turn).
        if s.startswith("plan:"):
            console.print(f"[dim]{_rescape(s)}[/dim]")
            return
        s = s.replace("\n", " ")
        console.print(f"[dim]  {_rescape(s[:160])}[/dim]")

    # HITL gate: a REAL blocking terminal prompt (writes ask, reads don't) --
    # used for both the original confirm()/choose() binary gate (git/jira/
    # config) AND V3-016's high-tier decision packages (options are
    # ["proceed","cancel"], not ["yes","no"]). The interrupt/empty-input
    # fallback must therefore snap to the LAST option (options[-1]), not a
    # hardcoded "no" -- "no" isn't a member of ["proceed","cancel"], so a
    # hardcoded fallback would get silently re-snapped to options[0]
    # ("proceed") by HITLGate.high(), inverting the safe-decline-on-
    # interrupt intent for exactly the actions where that matters most.
    from ledgerhound.analyst.hitl import HITLGate
    from ledgerhound.cli.prompts import confirm_prompt

    gate = HITLGate(prompt=lambda q, o: confirm_prompt(console, _rescape, q, o))
    # tool_mode: "native" (function-calling gateways like DeepSeek), "prompt"
    # (gateways WITHOUT function-calling, e.g. some self-hosted Kimi -- drive
    # tools via a text protocol), or "auto" (both). Override with
    # tool_mode in main_config.yaml (or LEDGERHOUND_TOOL_MODE); default auto
    # so a non-function-calling model still works instead of role-playing
    # tool calls in prose.
    from ledgerhound.config import get_str
    tool_mode = get_str("tool_mode", env="LEDGERHOUND_TOOL_MODE", default="auto").strip().lower()

    # use_kernel=None means "not specified on the command line" -- auto-detect
    # rather than defaulting to the pandas-only sandbox. The exec sandbox has
    # no cuDF path at all (V4 GPU audit); the kernel is the only way this CLI
    # actually reaches a GPU. A user with an H100 who never passes
    # --use-kernel was silently getting CPU-only analysis every time.
    if use_kernel is None:
        use_kernel, _detect_reason, _gpu_stranded = _auto_detect_use_kernel()
        console.print(f"[dim]kernel mode: {'on' if use_kernel else 'off'} "
                      f"(auto -- {_detect_reason})[/dim]")
        if _gpu_stranded:
            # V3-050: this used to be the exact same dim log line as "no GPU
            # present" -- a user with real hardware had no way to tell they
            # were about to get CPU-only analysis until they noticed it was
            # slow. Loud and impossible to miss on purpose.
            console.print(
                "[bold red]WARNING:[/bold red] a GPU was detected but this "
                "session is falling back to the CPU-only sandbox "
                f"({_detect_reason}). No GPU is reachable this way "
                "regardless of hardware -- install the kernel dependencies "
                "(pip install jupyter_client ipykernel) or pass --use-kernel "
                "once they're installed.")
    from ledgerhound.shell.conversation import (
        MAX_STEPS as _DEFAULT_MAX_STEPS,
        STEP_CALL_BUDGET as _DEFAULT_STEP_CALL_BUDGET,
        TURN_WALLCLOCK_BUDGET_S as _DEFAULT_TURN_BUDGET_S,
    )
    analyst = ConversationalAnalyst(
        llm, project=p, out_root=out_root,
        renderer=_status, gate=gate, tool_mode=tool_mode,
        autonomous=autonomous, use_kernel=use_kernel,
        methodology=methodology, methodology_mode=mode,
        max_steps=max_steps if max_steps is not None else _DEFAULT_MAX_STEPS,
        step_call_budget=step_call_budget if step_call_budget is not None
            else _DEFAULT_STEP_CALL_BUDGET,
        turn_budget_seconds=turn_budget_seconds if turn_budget_seconds is not None
            else _DEFAULT_TURN_BUDGET_S)
    if getattr(analyst, "_compressor", None) is not None:
        analyst._compressor.set_counter(_tok_counter)
    analyst._events.append("tokenizer_status", ok=_tok_status.ok,
                           code=_tok_status.code, model=_tok_status.model,
                           source=_tok_status.source,
                           detail=_tok_status.message[:1000])
    if max_steps or step_call_budget or turn_budget_seconds:
        console.print(f"[dim]depth budget: max_steps={analyst.max_steps} "
                      f"step_call_budget={analyst.step_call_budget} "
                      f"turn_budget_seconds={analyst.turn_budget_seconds}[/dim]")
    if autonomous:
        console.print("[yellow]--autonomous: low-risk questions will be "
                      "auto-answered, not asked to you.[/yellow]")
    console.print(f"[dim]project: {Path(p.root).resolve()}[/dim]")
    console.print(f"[dim]run folder: {analyst.run_dir.resolve()}   "
                  f"tool-mode: {analyst.tool_mode}[/dim]")

    # unified project config + workflow tool families
    from ledgerhound.integrations.git import GitClient, build_git_tools
    from ledgerhound.integrations.jira import JiraClient, build_jira_tools
    from ledgerhound.project.settings import ProjectSettings
    from ledgerhound.reporting.session_digest import SessionDigest
    from ledgerhound.tools.config_tools import build_config_tools
    from ledgerhound.tools.digest_tools import build_digest_tools
    # propose_plan/update_plan are registered by default on the analyst
    # itself (ConversationalAnalyst._default_registry, V3-005) against
    # `analyst.plan` -- forced continuation in send() reads that same
    # PlanState, so it must NOT be re-registered here with a fresh one
    # (that would silently disconnect the checklist from the continuation
    # check).

    settings = ProjectSettings.load(p)
    settings.load_secrets_env()
    analyst.registry.register_many(build_config_tools(settings))

    git_client = find_project_repo(Path(p.root), Path.cwd())

    digest = SessionDigest(
        findings_path=analyst._findings_path,
        action_log=analyst.registry.action_log,
        git_today_fn=(lambda: git_client.today_log().text())
        if git_client else None)
    analyst.registry.register_many(build_digest_tools(digest))

    # Integrations are OPT-IN sliders: a family's tools are only exposed to the
    # model when that integration is actually switched on. git = a repo is
    # present; jira = base_url + email are set. Nothing here is required -- with
    # everything off the analyst still runs data science fully.
    enabled: set[str] = set()

    if git_client is not None:
        analyst.registry.register_many(build_git_tools(
            git_client,
            remote=settings.get("git", "remote", "origin"),
            default_branch=settings.get("git", "default_branch", ""),
            digest_fn=digest.to_commit_message))
        enabled.add("git")

    def _sync_jira(announce: bool = False) -> None:
        """Register jira_* tools once Jira becomes configured. Called at start
        and after every turn so pasting creds mid-session switches it on live
        (no restart), matching the slider mental model."""
        if "jira" in enabled:
            return
        jc = JiraClient(
            base_url=settings.get("jira", "base_url"),
            email=settings.get("jira", "email"),
            token_env=settings.get("jira", "token_env", "LEDGERHOUND_JIRA_TOKEN"),
            api_version=settings.get("jira", "api_version", "3"),
            auth=settings.get("jira", "auth", "basic"),
            verify_ssl=str(settings.get("jira", "verify_ssl", "true")).strip().lower()
            not in ("false", "0", "no", "off"))
        if jc.configured():
            analyst.registry.register_many(build_jira_tools(
                jc, project_key=settings.get("jira", "project_key"),
                digest_fn=digest.to_markdown))
            enabled.add("jira")
            if announce:
                console.print("[dim]  jira integration switched on[/dim]")

    _sync_jira()

    analyst.messages[0]["content"] += (
        "\n\nWorkflow tools always available: propose_plan/update_plan (use them "
        "for any multi-step work so the user sees progress), "
        "configure_integration/show_config, session_digest (today's work "
        "summary). Optional integrations are OPT-IN and only appear as tools "
        "when switched on: git_* (present when the project is a git repo) and "
        "jira_* (present once Jira is configured). If the user asks for git or "
        "Jira and you do NOT have those tools, it means the integration is off: "
        "ask for the details (Jira: base URL, login email, API token; git: it "
        "just needs to be a repo) and store them with configure_integration -- "
        "never ask the user to edit files by hand. After you configure Jira it "
        "switches on for the next turn.")

    # slash commands -- thin; the model stays the brain
    from ledgerhound.shell.commands import SlashRegistry, SlashResult

    slash = SlashRegistry()

    def _cmd_model(args: str) -> SlashResult:
        args = args.strip()
        if not args:
            # No args: re-list what the CURRENT endpoint serves and pick
            # live -- applies immediately, no restart (unlike a config
            # switch below, which changes provider/base_url too).
            from ledgerhound.llm.catalog import choose_model
            chosen = choose_model(
                analyst.llm, autonomous=False, interactive=True,
                ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]"),
                say=lambda s: console.print(f"[dim]{s}[/dim]"))
            if chosen:
                analyst.llm.config.model = chosen
                console.print(f"[dim]model set to {chosen} (applied now)[/dim]")
                _resize_context(chosen)
            else:
                console.print("[dim]this endpoint doesn't support model "
                              "listing (/models) -- currently using "
                              f"{analyst.llm.config.model}. Pass a model id "
                              "directly: /model <id>, or switch providers "
                              "with /model configs/x.yaml[/dim]")
            return SlashResult(handled=True)
        if args.endswith((".yaml", ".yml")) or "/" in args or "\\" in args:
            # A config PATH -- switches provider/base_url, needs a restart.
            cfg = p.config()
            cfg.model_config_ref = args
            p.save_config(cfg)
            console.print(f"[dim]model config set to {args} "
                          f"(restart the shell to apply)[/dim]")
        else:
            # A literal model id on the CURRENT endpoint -- applies now.
            analyst.llm.config.model = args
            console.print(f"[dim]model set to {args} (applied now)[/dim]")
            _resize_context(args)
        return SlashResult(handled=True)

    def _resize_context(model_id: str) -> None:
        """I-12: a mid-session model switch must move the compaction
        budget with it -- switching from a 256k model to a 128k one while
        keeping the 256k budget overflows on the next long turn."""
        # A different model usually means a different tokenizer -- load it
        # first, since its manifest entry also carries the model's window.
        counter, tstatus = _load_tokenizer(model_id)
        _apply_context_window(analyst.llm,
                              say=lambda s: console.print(f"[dim]{s}[/dim]"),
                              hint=(tstatus.details or {}).get("context_window"))
        comp = getattr(analyst, "_compressor", None)
        if comp is not None:
            comp.context_window = int(analyst.llm.config.context_window)
            comp.set_counter(counter)

    def _cmd_model_judge(args: str) -> SlashResult:
        args = args.strip()
        if args.endswith((".yaml", ".yml")) or "/" in args or "\\" in args:
            settings.set_value("judge", "config_ref", args)
            settings.set_value("judge", "model", "")  # stale override, if any
            console.print(f"[dim]judge config set to {args} "
                          "(used on the next eval/judge run)[/dim]")
            return SlashResult(handled=True)
        if args:
            settings.set_value("judge", "model", args)
            console.print(f"[dim]judge model set to {args} "
                          "(used on the next eval/judge run)[/dim]")
            return SlashResult(handled=True)
        # No args: resolve whatever the judge currently is, list what that
        # endpoint actually serves, and let the analyst pick -- same
        # discovery machinery as /model, aimed at the judge's endpoint
        # instead of the actor's.
        from ledgerhound.eval.judge_config import resolve_judge
        from ledgerhound.llm.catalog import choose_model
        try:
            actor_key = analyst.llm.config.resolve_api_key()
        except Exception:  # noqa: BLE001
            actor_key = ""
        spec = resolve_judge(p, actor_base_url=analyst.llm.config.base_url,
                             actor_model=analyst.llm.config.model,
                             actor_key=actor_key,
                             actor_verify_ssl=analyst.llm.config.verify_ssl,
                             actor_models_url=analyst.llm.config.models_url)
        console.print(f"[dim]current judge: {spec.model} @ {spec.base_url} "
                      f"({'self-judge' if spec.self_judge else spec.source})[/dim]")
        if spec.fallback_reason:
            console.print(f"[yellow]a judge IS configured but could not be "
                          f"used -- {spec.fallback_reason}. Falling back to "
                          "self-judge until that is fixed.[/yellow]")
        # spec.build() carries verify_ssl/models_url (I-02) -- constructing
        # a bare LLMClient here dropped both and made every gateway judge
        # fail its TLS handshake.
        judge_llm = spec.build()
        chosen = choose_model(
            judge_llm, autonomous=False, interactive=True,
            ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]"),
            say=lambda s: console.print(f"[dim]{s}[/dim]"))
        if chosen:
            settings.set_value("judge", "model", chosen)
            console.print(f"[dim]judge model set to {chosen}[/dim]")
        else:
            console.print(
                "[dim]this judge endpoint doesn't support model listing "
                "(/models) -- pass one directly: /model-judge <id>[/dim]")
        return SlashResult(handled=True)

    def _cmd_adversarial(args: str) -> SlashResult:
        from ledgerhound.adversarial.engine import PAYEE_ROUNDS, PAYER_ROUNDS
        from ledgerhound.adversarial.run import parse_adversarial_args, run_from_session
        try:
            parsed = parse_adversarial_args(args)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return SlashResult(handled=True)
        if parsed["off"]:
            console.print("[dim]adversarial mode only runs when you call it -- "
                          "nothing to turn off; just don't call /adversarial "
                          "again.[/dim]")
            return SlashResult(handled=True)
        cases, seed = parsed["cases"], parsed["seed"]

        console.print(
            f"[yellow]Starting adversarial deep-research over up to {cases} "
            f"fraud case(s) -- payer side runs {PAYER_ROUNDS} rounds, payee "
            f"side {PAYEE_ROUNDS}, per case, before a judge verdict. This is "
            "a long, unattended, expensive run with full tool access on "
            "both agents. Ctrl+C to abort.[/yellow]")
        try:
            # I-16: `s` carries model/tool text (df[...] args, claims with
            # [brackets]) -- escape it exactly as the main renderer does,
            # or a stray closing-tag pattern raises MarkupError mid-case.
            results, report_path = run_from_session(
                analyst, cases=cases, seed=seed,
                say=lambda s: console.print(f"[dim]{_rescape(str(s))}[/dim]"))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return SlashResult(handled=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]adversarial run interrupted -- every "
                          "case finished before the interrupt is already in "
                          "run_outputs/<run>/adversarial/report.json.[/yellow]")
            return SlashResult(handled=True)
        n_flipped = sum(1 for r in results if r.flipped)
        console.print(f"[bold]{len(results)} case(s) reviewed, {n_flipped} "
                      f"flagged as possibly mislabelled.[/bold] "
                      f"Report: {report_path}")
        return SlashResult(handled=True)

    def _cmd_config(args: str) -> SlashResult:
        # markup=False: the summary's "[jira]" / "[git]" section headers are
        # plain text -- Rich used to swallow them as style tags.
        console.print(settings.masked_summary(), markup=False)
        console.print(str(settings.config_path), style="dim", markup=False)
        try:
            from ledgerhound.io.minio import shared_settings
            m = shared_settings()
            console.print(
                f"[object store] endpoint={m['endpoint'] or '(not set)'} "
                f"bucket={m['bucket']} access_key="
                f"{'set' if m['access_key'] else 'NOT set'} secret_key="
                f"{'set (' + m['secret_source'] + ')' if m['secret_key'] else 'NOT set'}"
                "  -- /minio to check the connection", markup=False)
        except Exception:  # noqa: BLE001
            pass
        return SlashResult(handled=True)

    def _load_cmd(kind: str):
        """/local --path and /s3 --path -- see ledgerhound.shell.data_cmd."""
        from rich.markup import escape as _esc_l

        def handler(args: str) -> SlashResult:
            from ledgerhound.io.data_source import DataSourceError
            from ledgerhound.shell.data_cmd import USAGE, model_note, parse_load_args
            try:
                path, as_one, name = parse_load_args(args)
            except ValueError as e:
                console.print(f"[yellow]{_esc_l(str(e))}  usage: {USAGE[kind]}[/yellow]")
                return SlashResult(handled=True)
            try:
                out = analyst.load_from_command(path, kind=kind, as_one=as_one, name=name)
            except DataSourceError as e:
                console.print(f"[yellow]{_esc_l(str(e))}[/yellow]")
                return SlashResult(handled=True)
            if out.startswith(("could not", "found ")) or "is a folder with multiple" in out:
                console.print(f"[yellow]{_esc_l(out)}[/yellow]")
                return SlashResult(handled=True)
            first = out.splitlines()[:3]
            console.print("[dim]" + _esc_l("\n".join(first)) + "[/dim]")
            return SlashResult(handled=False, inject=model_note(kind, path, out))
        return handler

    def _cmd_paths(args: str) -> SlashResult:
        console.print(analyst.run_paths().describe(), markup=False)
        if getattr(analyst, "_dataset_id", ""):
            console.print(f"dataset name : {analyst._dataset_id}", markup=False)
        kb = analyst.team_kb()
        if kb is not None:
            console.print(kb.status(), markup=False)
        return SlashResult(handled=True)

    def _cmd_minio(args: str) -> SlashResult:
        """/minio [--key [SECRET]] [--access-key KEY] [--clear] -- see
        ledgerhound.shell.minio_cmd. Secrets are checked before saving and
        never echoed; a new key re-loads the tokenizer immediately."""
        from rich.markup import escape as _esc_m
        from ledgerhound.shell.minio_cmd import run_minio_command

        def _ask_secret(prompt: str) -> str:
            try:
                import getpass
                return getpass.getpass(prompt)
            except Exception:  # noqa: BLE001 -- no tty: fall back to visible input
                return console.input(prompt)

        def _reload() -> None:
            comp = getattr(analyst, "_compressor", None)
            counter, _ = _load_tokenizer(analyst.llm.config.model)
            if comp is not None:
                comp.set_counter(counter)

        run_minio_command(
            args,
            say=lambda m: console.print(f"[dim]{_esc_m(m)}[/dim]"),
            warn=lambda m: console.print(f"[yellow]{_esc_m(m)}[/yellow]"),
            ask_secret=_ask_secret, on_changed=_reload)
        return SlashResult(handled=True)

    def _cmd_help(args: str) -> SlashResult:
        console.print(slash.help_text())
        return SlashResult(handled=True)

    def _cmd_library(args: str) -> SlashResult:
        # V3-054: CLI-only by design (per the standing instruction to leave
        # webui alone) -- dataset count, a 2-line description per book, and
        # total story/chapter/page counts, read straight off the
        # DatasetCardStore. No new storage layer; this IS the card index.
        if p is None:
            console.print("[dim]no project active -- the library requires one.[/dim]")
            return SlashResult(handled=True)
        try:
            from ledgerhound.knowledge.dataset_card import DatasetCardStore
            store = DatasetCardStore(p)
            cards = []
            if store.dir.exists():
                import json as _json
                for f in sorted(store.dir.glob("*.json")):
                    try:
                        cards.append(_json.loads(f.read_text(encoding="utf-8")))
                    except Exception:
                        continue
            if not cards:
                console.print("[dim]library is empty -- no dataset has been "
                              "loaded in this project yet.[/dim]")
                return SlashResult(handled=True)
            console.print(f"[bold]{len(cards)} book(s) in the library[/bold]")
            for c in cards:
                from pathlib import Path as _P
                name = _P(c.get("path", "?")).name
                stories = c.get("stories", [])
                n_chapters = sum(len(s.get("chapters", {})) for s in stories)
                n_pages = sum(
                    sum(ch.get("page_count", 0) for ch in s.get("chapters", {}).values())
                    for s in stories)
                digest = c.get("digest") or {}
                ov = digest.get("overview", {})
                desc = (f"{ov.get('rows', '?')} rows x {ov.get('columns', '?')} cols"
                       if ov else "not yet profiled")
                console.print(f"  [cyan]{name}[/cyan] ({c.get('key', '')[:10]}...)")
                console.print(f"    {desc}")
                console.print(f"    {len(stories)} stor{'y' if len(stories) == 1 else 'ies'}, "
                             f"{n_chapters} chapter(s), {n_pages} page(s)")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]library lookup failed: {e}[/red]")
        return SlashResult(handled=True)

    def _cmd_book(args: str) -> SlashResult:
        card = getattr(analyst, "_card", None)
        if card is None:
            console.print("[dim]no book open -- load a dataset first "
                          "(load_data), or say /library to see what exists.[/dim]")
            return SlashResult(handled=True)
        from pathlib import Path as _P
        console.print(f"[bold]book: {_P(card.path).name}[/bold]  (key {card.key[:10]}...)")
        console.print(card.library_summary())
        return SlashResult(handled=True)

    def _cmd_story(args: str) -> SlashResult:
        card = getattr(analyst, "_card", None)
        data_sha = getattr(analyst, "_data_sha256", "")
        if card is None or not data_sha:
            console.print("[dim]no story active -- load a dataset first.[/dim]")
            return SlashResult(handled=True)
        story = card.find_story(data_sha)
        if story is None:
            console.print("[dim]this exact data version has no story yet "
                          "(nothing recorded against it).[/dim]")
            return SlashResult(handled=True)
        console.print(f"[bold]{story['story_id']}[/bold]  "
                      f"(data {data_sha[:10]}..., first seen {story.get('first_seen', '?')})")
        chapters = story.get("chapters", {})
        if not chapters:
            console.print("  no chapters opened yet")
        for name, ch in chapters.items():
            console.print(f"  - {name}: {ch.get('page_count', 0)} page(s), "
                          f"status={ch.get('status', '?')}")
        return SlashResult(handled=True)

    def _cmd_chapter(args: str) -> SlashResult:
        from ledgerhound.knowledge.chapter_taxonomy import taxonomy_block
        if args:
            # Inject as a natural-language turn so the model actually calls
            # open_chapter (which does the real dedup check) -- this
            # command is a convenience wrapper, not a bypass of it.
            return SlashResult(handled=False,
                               inject=f"Open chapter '{args.strip()}'.")
        console.print(taxonomy_block())
        return SlashResult(handled=True)

    slash.register("help", "show commands", _cmd_help)
    slash.register("model", "/model lists+picks a model on this endpoint; "
                   "/model <id> sets one directly; /model configs/x.yaml "
                   "switches provider (restart to apply)", _cmd_model)
    slash.register("model-judge", "same as /model, but for the eval/"
                   "LLM-as-judge model (independent of the actor)",
                   _cmd_model_judge)
    slash.register("adversarial", "/adversarial --cases 25 [--seed N] -- "
                   "deep-research debate (analyst vs. adversary on the "
                   "judge model) over N random fraud cases, judged at "
                   "the end. Requires set_goal(label_column, entity_key) "
                   "first.", _cmd_adversarial)
    slash.register("config", "show integration config (secrets masked)", _cmd_config)
    slash.register("local", "/local --path <file|folder> [--as-one] -- load local data",
                   _load_cmd("local"))
    slash.register("s3", "/s3 --path s3://bucket/key [--as-one] -- load data from the "
                   "object store into RAM; outputs upload back", _load_cmd("s3"))
    slash.register("paths", "where this run's data came from and where outputs go",
                   _cmd_paths)
    slash.register("minio","/minio shows the object store + checks it; "
                   "/minio --key prompts for the secret (checked, never shown); "
                   "/minio --access-key <key>; /minio --clear", _cmd_minio)
    slash.register("library", "list books (datasets) and their story/chapter counts", _cmd_library)
    slash.register("book", "show the currently loaded dataset's book", _cmd_book)
    slash.register("story", "show the current data version's story + chapters", _cmd_story)
    slash.register("chapter", "list the chapter taxonomy, or /chapter <name> to open one", _cmd_chapter)
    slash.register_inject("jira", "pending tasks / post update",
                          "Show my pending Jira issues.")
    slash.register_inject("git", "repo status / push work",
                          "Show git status of this project.")
    slash.register_inject("plan", "ask for an explicit plan first",
                          "Before doing anything else, propose a plan for:")
    slash.register_inject("digest", "summarize today's work",
                          "Give me the session digest of today's work.")

    model_name = analyst.llm.config.model or Path(cfg_ref).stem
    console.print(_Panel(
        f"[bold]Ledgerhound[/bold] -- agentic data-science analyst\n"
        f"[dim]model[/dim] {model_name}   [dim]project[/dim] {p.config().name}   "
        f"[dim]runs[/dim] {analyst.run_dir}\n"
        f"[dim]integrations[/dim] {', '.join(sorted(enabled)) or 'none on -- data science ready; /config to switch on git/jira'}\n"
        f"[dim]Just talk to me -- describe your data and goal. "
        f"/help for commands, 'exit' to leave.[/dim]",
        border_style="cyan", title="[bold cyan]data science, agentic[/bold cyan]"))

    try:
        _shell_loop(console, analyst, slash, cfg_ref, _sync_jira, Markdown,
                    LLMAuthError, resolve_method, clear_key, ensure_api_key)
    finally:
        # I-05: every exit path (exit, Ctrl+D, a crash) releases the
        # kernel -- otherwise an ipykernel holding a CUDA context outlives
        # the shell on the Kubeflow box.
        try:
            analyst.close()
        except Exception:  # noqa: BLE001
            pass


def _shell_loop(console, analyst, slash, cfg_ref, _sync_jira, Markdown,
                LLMAuthError, resolve_method, clear_key, ensure_api_key) -> None:
    """The REPL proper -- split out of shell() so the cleanup above wraps
    it in one place."""
    while True:
        try:
            user = _read_user_turn(
                console, "\n[bold green]you[/bold green] [dim]>[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not user:
            continue
        # I-06: a slash handler raising used to take the whole session
        # with it (the dispatch sat outside any try). A broken command is
        # one line of red, never a lost session.
        try:
            res = slash.dispatch(user)
        except KeyboardInterrupt:
            console.print("\n[yellow]command interrupted[/yellow]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]command failed: {type(e).__name__}: {e}[/red]")
            continue
        if res is not None:
            if res.quit:
                console.print("[dim]bye[/dim]")
                break
            if res.notice:
                from rich.markup import escape as _esc_n
                console.print(f"[yellow]{_esc_n(res.notice)}[/yellow]")
            if res.handled:
                continue
            user = res.inject or user
        try:
            console.print("[dim]  thinking...[/dim]")
            reply = analyst.send(user)
        except KeyboardInterrupt:
            # I-05: Ctrl+C stops THIS turn, not the session. Hours of
            # loaded data, kernel state and compaction history stay put;
            # the history is repaired so the next request is still valid.
            analyst.abort_turn("interrupted by the user (Ctrl+C)")
            console.print("\n[yellow]turn interrupted -- the session is still "
                          "open; say what to do next, or 'exit'.[/yellow]")
            continue
        except LLMAuthError as e:
            # Perfectly identify this as a KEY problem, not a generic
            # error -- clear whatever's stored (it's proven wrong) and
            # re-prompt once, live, instead of failing every turn from
            # here on with the same bad key.
            console.print(f"[red]API key rejected ({e}).[/red]")
            method = resolve_method(cfg_ref)
            if method:
                clear_key("llm", method)
                new_key = ensure_api_key(
                    "llm", method,
                    ask=lambda q: console.input(f"[bold cyan]{q}[/bold cyan]", password=True),
                    say=lambda s: console.print(f"[dim]{s}[/dim]"))
                if new_key:
                    analyst.llm.set_api_key(new_key)
                    console.print("[dim]new key set -- try your message "
                                 "again.[/dim]")
            else:
                console.print("[dim]this config sets its own api_key_env "
                              "directly -- fix the key in that env var, "
                              "not here.[/dim]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error: {e}[/red]")
            continue
        console.print()
        if not (reply or "").strip():
            reply = ("[dim](the model returned nothing this turn -- see "
                     f"{analyst.run_dir}/transcript.jsonl for the raw response)[/dim]")
            console.print(reply)
        else:
            try:
                console.print(Markdown(reply))
            except Exception:
                # markup=False: never re-parse model text as Rich markup
                console.print(reply, markup=False)
        _sync_jira(announce=True)   # pick up a Jira config done this turn


@project_app.command("info")
def project_info(
    root: str = typer.Argument(".", help="Project root (folder containing .ledgerhound/)"),
):
    """Show a project's config and its LEDGERHOUND.md memory."""
    from ledgerhound.project import Project
    p = Project(root)
    if not p.exists():
        typer.echo(f"not a project: {root} (no .ledgerhound/project.json)")
        raise typer.Exit(1)
    cfg = p.config()
    typer.echo(f"name:         {cfg.name}")
    typer.echo(f"owner:        {cfg.owner or '(none)'}")
    typer.echo(f"save_format:  {cfg.save_format}")
    typer.echo(f"model_config: {cfg.model_config_ref or '(default)'}")
    typer.echo(f"created_at:   {cfg.created_at}")
    typer.echo("--- LEDGERHOUND.md ---")
    typer.echo(p.memory().rstrip() or "(empty)")



def _default_out(value: str, default_name: str) -> str:
    """A command's output file left at its default name goes into
    Agentic_eda_results/<username>/run_outputs/, not the launch folder. An
    explicit -o/--out is used exactly as given."""
    if str(value or "").strip() in ("", default_name):
        from ledgerhound.paths import run_outputs_dir
        d = run_outputs_dir()
        d.mkdir(parents=True, exist_ok=True)
        return str(d / default_name)
    return value


def _ws_path(value) -> Path:
    """--workspace for the scripted commands. Left at its default it goes to
    Agentic_eda_results/<username>/library (ledgerhound.paths) -- the same place
    the shell uses -- never ./ledgerhound_workspace in the launch folder."""
    v = str(value or "").strip()
    if v in ("", "ledgerhound_workspace"):
        from ledgerhound.paths import library_dir
        return library_dir()
    return Path(v)

@workspace_app.command("init")
def workspace_init(
    root: str = typer.Option("ledgerhound_workspace", "--workspace", "-w",
                             help="Root directory for the workspace"),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Re-initialise even if workspace exists"),
):
    """Create the Ledgerhound Phase-2 workspace directory tree.

    Creates: reports/, samples/, memory/, drift/, logs/, and
    typology_evolution.json stub so Phase-2 tools have a home.
    """
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(root))
    existed = ws.root.exists()
    if existed and not force:
        console.print(f"[yellow]Workspace already exists at {ws.root}[/yellow]")
        console.print("[dim]Pass --force to re-initialise.[/dim]")
        raise typer.Exit(0)

    ws.init()
    console.print(Panel(
        f"[bold green]Workspace initialised[/bold green]\n"
        f"  root:         {ws.root}\n"
        f"  reports:      {ws.reports_dir}\n"
        f"  samples:      {ws.samples_dir}\n"
        f"  memory:       {ws.memory_dir}\n"
        f"  drift events: {ws.drift_events_path}\n"
        f"  typology evo: {ws.typology_evolution_path}",
        border_style="green",
    ))


@workspace_app.command("status")
def workspace_status(
    root: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show workspace artefact sizes and key paths."""
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(root))
    if not ws.root.exists():
        console.print(f"[red]Workspace not found: {ws.root}[/red]")
        console.print("[dim]Run `ledgerhound workspace init` first.[/dim]")
        raise typer.Exit(1)

    from rich.table import Table

    paths = [
        ("Root", ws.root),
        ("Reports", ws.reports_dir),
        ("Samples", ws.samples_dir),
        ("Memory", ws.memory_dir),
        ("Drift events", ws.drift_events_path),
        ("Signal history", ws.feature_signal_history_path),
        ("Typology evolution", ws.typology_evolution_path),
    ]

    t = Table(title=f"Workspace -- {ws.root}", border_style="cyan")
    t.add_column("Artefact")
    t.add_column("Path")
    t.add_column("Exists")
    t.add_column("Size")
    for label, p in paths:
        exists = p.exists()
        size = ""
        if exists and p.is_file():
            sz = p.stat().st_size
            size = f"{sz:,} B" if sz < 1024 else f"{sz // 1024:,} KB"
        elif exists and p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            size = f"{n} files"
        t.add_row(label, str(p), "[green]OK[/green]" if exists else "[dim]--[/dim]", size)
    console.print(t)


# ============================================================
# `ledgerhound playbook` -- FraudEDAPlaybook runner
# ============================================================

playbook_app = typer.Typer(
    name="playbook",
    help="Run or inspect the deterministic Fraud EDA Playbook.",
    no_args_is_help=True,
)
app.add_typer(playbook_app, name="playbook")


@playbook_app.command("run")
def playbook_run(
    fraud_data:  str = typer.Option(..., "--fraud",    "-f", help="Fraud parquet / CSV"),
    nonfr_data:  str = typer.Option(..., "--nonfr",    "-n", help="Non-fraud parquet / CSV"),
    workspace:   str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    period:      str = typer.Option("current", "--period", "-p",
                                    help="Time-period label (e.g. 2024-Q2)"),
    dataset_name:str = typer.Option("fraud_dataset", "--dataset", "-d"),
    features:   Optional[str] = typer.Option(
        None, "--features",
        help="Comma-separated numeric feature columns. Omit to use all numerics.",
    ),
    summary_data: Optional[str] = typer.Option(None, "--summary",   help="Summary table parquet/CSV"),
    raw_txn_data: Optional[str] = typer.Option(None, "--raw-txn",   help="Raw transaction parquet/CSV"),
    n_clusters:   Optional[int] = typer.Option(None, "--n-clusters", help="Fixed cluster count (auto if omitted)"),
    skip:         Optional[str] = typer.Option(
        None, "--skip",
        help="Comma-separated step names to skip "
             "(consistency, typology, typology_evolution, hard_negatives, "
             "psi_certification, signal_health, memory_deposit)",
    ),
    save_report:  bool = typer.Option(True, "--save-report/--no-save-report"),
):
    """Run the full Phase-2 Fraud EDA Playbook.

    Chains data-quality validation -> typology clustering -> typology
    evolution tracking -> hard negative mining -> PSI certification ->
    signal health -> institutional memory deposit.

    No LLM required.  All steps write artefacts under the workspace.
    """
    _check_file(fraud_data, "Fraud data")
    _check_file(nonfr_data, "Non-fraud data")

    import pandas as pd

    from ledgerhound.analysis.fraud_playbook import FraudEDAPlaybook
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    ws.init()

    def _load(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        return pd.read_csv(p)

    console.print("[dim]Loading fraud data...[/dim]")
    fraud_df = _load(fraud_data)
    nonfr_df = _load(nonfr_data)
    summary_df = _load(summary_data) if summary_data else None
    raw_txn_df = _load(raw_txn_data) if raw_txn_data else None

    # Feature selection
    feat_list: list[str]
    if features:
        feat_list = [f.strip() for f in features.split(",")]
    else:
        feat_list = [
            c for c in fraud_df.columns
            if pd.api.types.is_numeric_dtype(fraud_df[c])
        ]

    skip_set = set(s.strip() for s in skip.split(",")) if skip else set()

    console.print(Panel(
        f"[bold]Fraud EDA Playbook[/bold]\n"
        f"  fraud:   {fraud_data}  ({len(fraud_df):,} rows)\n"
        f"  nonfr:   {nonfr_data}  ({len(nonfr_df):,} rows)\n"
        f"  period:  {period}\n"
        f"  features: {len(feat_list)}\n"
        f"  workspace: {ws.root}",
        border_style="cyan",
    ))

    pb = FraudEDAPlaybook(
        workspace=ws,
        dataset_name=dataset_name,
        session_id=f"cli_{period}",
        n_clusters=n_clusters,
        skip_steps=skip_set,
    )
    report = pb.run(
        fraud_df=fraud_df,
        nonfr_df=nonfr_df,
        features=feat_list,
        period=period,
        summary_df=summary_df,
        raw_txn_df=raw_txn_df,
    )
    report.display()

    if save_report:
        rpath = ws.reports_dir / "playbook" / f"run_{period}.json"
        console.print(f"[dim]Report: {rpath}[/dim]")


# ============================================================
# `ledgerhound signal` -- feature signal decay tracker
# ============================================================

signal_app = typer.Typer(
    name="signal",
    help="Measure feature discriminative power and detect decay over time.",
    no_args_is_help=True,
)
app.add_typer(signal_app, name="signal")


@signal_app.command("check")
def signal_check(
    fraud_data: str = typer.Option(..., "--fraud", "-f"),
    nonfr_data: str = typer.Option(..., "--nonfr", "-n"),
    period:     str = typer.Option("current", "--period", "-p"),
    workspace:  str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    dataset_name: str = typer.Option("default", "--dataset", "-d"),
    features:   Optional[str] = typer.Option(None, "--features",
                                              help="Comma-separated feature list"),
    top_n:      int = typer.Option(15, "--top-n", help="Rows to show in table"),
    no_save:    bool = typer.Option(False, "--no-save",
                                    help="Skip persisting history to workspace"),
):
    """Measure signal strength for all numeric features and detect decay.

    Compares the current period's KS statistic, AUC, and mutual information
    against the stored baseline.  Prints a Rich grade table and lists any
    decay events.
    """
    _check_file(fraud_data, "Fraud data")
    _check_file(nonfr_data, "Non-fraud data")

    import pandas as pd

    from ledgerhound.analysis.signal_decay import SignalDecayTracker, display_signal_result
    from ledgerhound.workspace.layout import WorkspaceLayout

    def _load(path: str) -> pd.DataFrame:
        p = Path(path)
        return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)

    fraud_df = _load(fraud_data)
    nonfr_df = _load(nonfr_data)

    feat_list = (
        [f.strip() for f in features.split(",")]
        if features else
        [c for c in fraud_df.columns if pd.api.types.is_numeric_dtype(fraud_df[c])]
    )

    ws = WorkspaceLayout(root=_ws_path(workspace))
    ws.init()

    tracker = SignalDecayTracker()
    result = tracker.track(
        fraud_df=fraud_df,
        nonfr_df=nonfr_df,
        features=feat_list,
        period=period,
        workspace=ws if not no_save else None,
        session_id=f"cli_{period}",
    )
    display_signal_result(result, top_n=top_n)

    if result.decay_events:
        console.print(f"\n[bold red]{len(result.decay_events)} decay event(s):[/bold red]")
        for ev in result.decay_events:
            console.print(
                f"  [{('red' if ev.severity == 'FAIL' else 'yellow')}]"
                f"{ev.severity}[/] {ev.feature}: "
                f"{ev.primary_metric} {ev.pct_change:+.1f}% "
                f"vs baseline '{ev.baseline_period}'"
            )


@signal_app.command("history")
def signal_history(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    feature:   Optional[str] = typer.Option(None, "--feature", "-f",
                                             help="Filter to a single feature"),
    top_n:     int = typer.Option(10, "--top-n", "-n"),
):
    """Show stored signal history from the workspace.

    Each period snapshot is loaded from feature_signal_history.json and
    displayed as a trend table.
    """
    from rich.table import Table

    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    hist_path = ws.feature_signal_history_path
    if not hist_path.exists():
        console.print(f"[yellow]No signal history at {hist_path}[/yellow]")
        console.print("[dim]Run `ledgerhound signal check` at least once first.[/dim]")
        raise typer.Exit(0)

    import json
    history = json.loads(hist_path.read_text(encoding="utf-8"))

    if feature:
        if feature not in history:
            console.print(f"[red]Feature '{feature}' not in history.[/red]")
            console.print(f"[dim]Available: {list(history.keys())[:20]}[/dim]")
            raise typer.Exit(1)
        items = {feature: history[feature]}
    else:
        # Show features with most snapshots first
        items = dict(
            sorted(history.items(), key=lambda kv: -len(kv[1]))[:top_n]
        )

    t = Table(title="Feature Signal History", border_style="cyan",
              show_header=True, header_style="bold dim")
    t.add_column("Feature")
    t.add_column("Periods", justify="right")
    t.add_column("Latest KS", justify="right")
    t.add_column("Latest AUC", justify="right")
    t.add_column("Grade (latest)")
    t.add_column("Trend")

    grade_colors = {"A": "green", "B": "green", "C": "yellow",
                    "D": "red", "F": "red", "?": "dim"}

    for feat, snaps in items.items():
        if not snaps:
            continue
        latest = snaps[-1]
        n = len(snaps)
        ks  = f"{latest.get('ks_statistic', 0):.3f}"
        auc = f"{latest.get('auc', 0):.3f}"
        grade = latest.get("grade", "?")
        gc = grade_colors.get(grade, "white")

        # Simple trend arrow from first to last KS
        trend = "--"
        if n >= 2:
            first_ks = snaps[0].get("ks_statistic", 0) or 0
            last_ks  = latest.get("ks_statistic", 0) or 0
            diff = last_ks - first_ks
            if diff > 0.02:
                trend = "[green] improving[/green]"
            elif diff < -0.02:
                trend = "[red] decaying[/red]"
            else:
                trend = "[dim]-> stable[/dim]"

        t.add_row(feat, str(n), ks, auc, f"[{gc}]{grade}[/{gc}]", trend)

    console.print(t)
    console.print(f"[dim]History file: {hist_path}[/dim]")


# ============================================================
# `ledgerhound typology` -- typology clustering & evolution
# ============================================================

typology_app = typer.Typer(
    name="typology",
    help="Cluster fraud typologies and track how they evolve over time.",
    no_args_is_help=True,
)
app.add_typer(typology_app, name="typology")


@typology_app.command("run")
def typology_run(
    fraud_data:  str = typer.Option(..., "--fraud", "-f"),
    period:      str = typer.Option("current", "--period", "-p"),
    workspace:   str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    features:    Optional[str] = typer.Option(None, "--features"),
    n_clusters:  Optional[int] = typer.Option(None, "--n-clusters"),
    method:      str = typer.Option("kmeans", "--method",
                                    help="kmeans | dbscan | gmix"),
    save_report: bool = typer.Option(True, "--save-report/--no-save-report"),
):
    """Cluster confirmed fraud cases and track evolution against stored history.

    Prints both the static typology table and the evolution trajectory
    (growing / declining / emerging clusters).
    """
    _check_file(fraud_data, "Fraud data")

    import pandas as pd

    from ledgerhound.analysis.typology import ClusteringMethod, TypologyAnalyzer
    from ledgerhound.analysis.typology_evolution import TypologyEvolutionTracker
    from ledgerhound.workspace.layout import WorkspaceLayout

    def _load(path: str) -> pd.DataFrame:
        p = Path(path)
        return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)

    fraud_df = _load(fraud_data)

    feat_list = (
        [f.strip() for f in features.split(",")]
        if features else
        [c for c in fraud_df.columns if pd.api.types.is_numeric_dtype(fraud_df[c])]
    )

    ws = WorkspaceLayout(root=_ws_path(workspace))
    ws.init()

    # Static clustering
    try:
        m = ClusteringMethod(method)
    except ValueError:
        console.print(f"[red]Unknown method '{method}'. Use: kmeans, dbscan, gmix[/red]")
        raise typer.Exit(1)

    analyzer = TypologyAnalyzer(method=m, n_clusters=n_clusters)
    rpt = analyzer.analyze(fraud_df=fraud_df, features=feat_list)

    if save_report:
        p = ws.reports_dir / "typology" / "typology_report.json"
        rpt.save(p)

    # Print static clusters
    from rich.table import Table
    t = Table(title=f"Fraud Typologies (method={method})", border_style="cyan")
    t.add_column("Cluster"); t.add_column("Label"); t.add_column("Cases", justify="right")
    t.add_column("% Fraud", justify="right"); t.add_column("Top Feature")
    t.add_column("Growing")
    for typ in rpt.typologies:
        t.add_row(
            str(typ.cluster_id), typ.label, f"{typ.n_cases:,}",
            f"{typ.pct_of_fraud:.1%}",
            typ.top_features[0] if typ.top_features else "--",
            "[red]YES[/red]" if typ.is_growing else "[dim]no[/dim]",
        )
    console.print(t)
    console.print(
        f"[dim]Silhouette={rpt.silhouette_score:.3f}  "
        f"Davies-Bouldin={rpt.davies_bouldin_score:.3f}[/dim]"
    )

    # Evolution tracking
    console.print("\n[bold]Running evolution tracking...[/bold]")
    tracker = TypologyEvolutionTracker(n_clusters=n_clusters)
    evo = tracker.track(
        fraud_df=fraud_df,
        features=feat_list,
        period=period,
        workspace=ws,
    )
    evo.display()


@typology_app.command("history")
def typology_history(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    cluster:   Optional[str] = typer.Option(None, "--cluster", "-c",
                                             help="Show detail for one cluster_key"),
):
    """Show typology evolution history stored in the workspace."""
    import json

    from rich.table import Table

    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    evo_path = ws.typology_evolution_path
    if not evo_path.exists():
        console.print(f"[yellow]No evolution history at {evo_path}[/yellow]")
        console.print("[dim]Run `ledgerhound typology run` first.[/dim]")
        raise typer.Exit(0)

    history = json.loads(evo_path.read_text(encoding="utf-8"))
    periods  = history.get("periods", [])
    clusters = history.get("clusters", {})

    if cluster:
        if cluster not in clusters:
            console.print(f"[red]Cluster '{cluster}' not found.[/red]")
            console.print(f"[dim]Known clusters: {list(clusters.keys())}[/dim]")
            raise typer.Exit(1)
        snaps = clusters[cluster]
        t = Table(title=f"Cluster '{cluster}' history", border_style="cyan")
        t.add_column("Period"); t.add_column("Cases", justify="right")
        t.add_column("% Fraud", justify="right"); t.add_column("Label")
        t.add_column("QoQ Growth", justify="right")
        prev_n = None
        for s in snaps:
            n = s["n_cases"]
            growth = f"{(n - prev_n) / prev_n:+.1%}" if prev_n else "--"
            t.add_row(s["period"], f"{n:,}", f"{s['pct_of_fraud']:.1%}",
                      s["label"], growth)
            prev_n = n
        console.print(t)
        return

    # Summary table
    t = Table(title=f"Typology Evolution History -- {len(periods)} period(s)",
              border_style="cyan")
    t.add_column("Cluster"); t.add_column("Periods", justify="right")
    t.add_column("First", justify="right"); t.add_column("Latest", justify="right")
    t.add_column("Trend")
    for key, snaps in sorted(clusters.items()):
        if not snaps:
            continue
        n_periods = len(snaps)
        first_n   = snaps[0]["n_cases"]
        last_n    = snaps[-1]["n_cases"]
        if n_periods >= 2:
            rates = []
            for i in range(1, n_periods):
                pn = snaps[i - 1]["n_cases"]
                cn = snaps[i]["n_cases"]
                if pn:
                    rates.append((cn - pn) / pn)
            avg = sum(rates) / len(rates) if rates else 0.0
            if avg > 0.10:
                trend = "[red] GROWING[/red]"
            elif avg < -0.10:
                trend = "[cyan] DECLINING[/cyan]"
            else:
                trend = "[green]-> STABLE[/green]"
        else:
            trend = "[dim]-- (1 period)[/dim]"
        t.add_row(key, str(n_periods), f"{first_n:,}", f"{last_n:,}", trend)
    console.print(t)
    console.print(f"[dim]Periods: {', '.join(periods)}[/dim]")
    console.print(f"[dim]Source: {evo_path}[/dim]")


# -- Rank 10: cross-dataset typology signatures ------------------------------

def _signature_library_path(workspace: str) -> Path:
    return _ws_path(workspace) / "data_memory" / "typology_signatures.jsonl"


@typology_app.command("signature")
def typology_signature_cmd(
    data: str = typer.Option(..., "--data", "-d", help="Data file OR folder"),
    name: str = typer.Option(..., "--name", help="Typology/scam-type name"),
    features: Optional[str] = typer.Option(
        None, "--features", help="Comma-separated; omit to auto-detect numeric"),
    subset_col: Optional[str] = typer.Option(
        None, "--subset-col", help="Restrict to a slice, e.g. is_fraud"),
    subset_value: str = typer.Option("1", "--subset-value"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Build + store a typology signature (per-feature profile of a scam type)."""
    from ledgerhound import identity
    from ledgerhound.agents.fraud_nonfraud import auto_numeric_features
    from ledgerhound.analysis.typology_signature import SignatureLibrary, build_signature
    data = _resolve_data_path(data, "Typology data")
    df = _read_data(data)
    feats = ([f.strip() for f in features.split(",") if f.strip()] if features
             else auto_numeric_features(df, df))
    sv: object = subset_value
    if subset_col and subset_col in df.columns:
        try:
            sv = type(df[subset_col].dropna().iloc[0])(subset_value)
        except Exception:
            sv = subset_value
    sig = build_signature(df, feats, name=name, dataset=Path(data).stem,
                          owner=identity.current_user(),
                          subset_col=subset_col, subset_value=sv)
    SignatureLibrary(_signature_library_path(workspace)).deposit(sig)
    console.print(f"[green]Signature '{name}'[/green] built over {len(sig.profile)} "
                  f"feature(s), n={sig.n} -> {_signature_library_path(workspace)}")


@typology_app.command("signatures")
def typology_signatures(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """List stored typology signatures."""
    from rich.table import Table

    from ledgerhound.analysis.typology_signature import SignatureLibrary
    sigs = SignatureLibrary(_signature_library_path(workspace)).all()
    if not sigs:
        console.print("[yellow]No signatures yet (use `typology signature`).[/yellow]")
        return
    t = Table(title="Typology signatures", border_style="cyan")
    for c in ("Name", "Dataset", "n", "Features", "Owner"):
        t.add_column(c)
    for s in sigs:
        t.add_row(s.name, s.dataset, str(s.n), str(len(s.profile)), s.owner)
    console.print(t)


@typology_app.command("compare")
def typology_compare(
    names: str = typer.Option(
        None, "--names", help="Comma-separated stored signature names "
                              "(default: all stored)"),
    f_threshold: float = typer.Option(2.0, "--f-threshold"),
    report: Optional[str] = typer.Option(None, "--report", "-r"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Compare stored signatures: shared vs discriminating dimensions."""
    from rich.table import Table

    from ledgerhound.analysis.typology_signature import SignatureLibrary, compare_signatures
    lib = SignatureLibrary(_signature_library_path(workspace))
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        sigs = [lib.get(n) for n in wanted if lib.get(n)]
    else:
        sigs = lib.all()
    if len(sigs) < 2:
        console.print("[red]Need >= 2 stored signatures to compare.[/red]")
        raise typer.Exit(1)
    comp = compare_signatures(sigs, f_threshold=f_threshold)
    t = Table(title=f"Typology comparison ({', '.join(comp.signatures)})",
              border_style="cyan")
    for c in ("Feature", "F-ratio", "Dimension"):
        t.add_column(c)
    for d in sorted(comp.dimensions, key=lambda x: -x.f_ratio):
        t.add_row(d.feature, f"{d.f_ratio:.3f}", d.kind)
    console.print(t)
    for name in comp.signatures:
        uniq = comp.unique_to(name)
        console.print(f"  [cyan]{name}[/cyan] unique dims: "
                      f"{', '.join(uniq) or '(none)'}")
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(comp.to_markdown(), encoding="utf-8")
        console.print(f"[green]Report written:[/green] {report}")


# ============================================================
# `ledgerhound graph` -- mule motif / centrality / ring / flow CLI
# ============================================================

graph_app = typer.Typer(
    name="graph",
    help="Build account graphs and run mule-detection EDA on transactions.",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph")


def _load_txn(path: str):
    import pandas as pd
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _build_graph_or_exit(
    txn_path: str, period: str,
    payer_col: str, payee_col: str,
    amount_col: str, timestamp_col: str,
    status_col: Optional[str], success_only: bool,
):
    _check_file(txn_path, "Transaction file")
    from ledgerhound.graph.builder import AccountGraphBuilder
    txn_df = _load_txn(txn_path)
    builder = AccountGraphBuilder(
        payer_col=payer_col,
        payee_col=payee_col,
        amount_col=amount_col,
        timestamp_col=timestamp_col,
        status_col=status_col,
        success_values=("SUCCESS",) if success_only else (),
    )
    graph = builder.build(txn_df, period=period)
    console.print(f"[dim]{graph.info()}[/dim]")
    return graph, txn_df


@graph_app.command("motifs")
def graph_motifs(
    txn:        str = typer.Argument(..., help="Transaction file (parquet/csv)"),
    period:     str = typer.Option("current", "--period", "-p"),
    workspace:  str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    payer_col:  str = typer.Option("account_id",    "--payer-col"),
    payee_col:  str = typer.Option("beneficiary_id","--payee-col"),
    amount_col: str = typer.Option("txn_amount",    "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",   "--timestamp-col"),
    status_col: Optional[str] = typer.Option("status", "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status",
                                          help="Don't restrict to status=SUCCESS"),
    patterns:   Optional[str] = typer.Option(None, "--patterns",
                                              help="Comma-separated pattern names"),
    save:       bool = typer.Option(True, "--save/--no-save"),
):
    """Run the 6-pattern mule motif catalogue on a transaction file."""
    from ledgerhound.graph.motifs import (
        CyclicRing,
        FanInCollector,
        FanOutDistributor,
        LayeringChain,
        MotifDetector,
        PassThrough,
        VelocityHub,
    )
    from ledgerhound.workspace.layout import WorkspaceLayout

    graph, _ = _build_graph_or_exit(
        txn, period, payer_col, payee_col, amount_col, timestamp_col,
        status_col, success_only=not no_filter_status,
    )

    pat_map = {
        "fan_in_collector":     FanInCollector,
        "fan_out_distributor":  FanOutDistributor,
        "layering_chain":       LayeringChain,
        "cyclic_ring":          CyclicRing,
        "velocity_hub":         VelocityHub,
        "pass_through":         PassThrough,
    }
    if patterns:
        names = [p.strip() for p in patterns.split(",")]
        bad = [n for n in names if n not in pat_map]
        if bad:
            console.print(f"[red]Unknown patterns: {bad}. "
                          f"Choose from: {sorted(pat_map.keys())}[/red]")
            raise typer.Exit(1)
        detector = MotifDetector(patterns=[pat_map[n]() for n in names])
    else:
        detector = MotifDetector()

    result = detector.detect(graph)
    result.display()

    if save:
        ws = WorkspaceLayout(root=_ws_path(workspace))
        ws.init()
        out = ws.reports_dir / "graph" / f"motifs_{period}.json"
        result.save(out)
        console.print(f"[green]OK Report:[/green] {out}")


@graph_app.command("centrality")
def graph_centrality(
    txn:       str = typer.Argument(..., help="Transaction file"),
    period:    str = typer.Option("current", "--period", "-p"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    payer_col: str = typer.Option("account_id",    "--payer-col"),
    payee_col: str = typer.Option("beneficiary_id","--payee-col"),
    amount_col: str = typer.Option("txn_amount",   "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",  "--timestamp-col"),
    status_col: Optional[str] = typer.Option("status", "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status"),
    measures:  str = typer.Option("pagerank,k_core", "--measures",
                                   help="Comma-separated measure names"),
    betweenness_k: Optional[int] = typer.Option(None, "--betweenness-k",
                                                 help="k-sample size for approx betweenness"),
    top_n:     int = typer.Option(10, "--top-n", "-n"),
    save:      bool = typer.Option(True, "--save/--no-save"),
):
    """Compute centrality measures and surface pass-through mule candidates."""
    from ledgerhound.graph.centrality import CentralityComputer
    from ledgerhound.workspace.layout import WorkspaceLayout

    graph, _ = _build_graph_or_exit(
        txn, period, payer_col, payee_col, amount_col, timestamp_col,
        status_col, success_only=not no_filter_status,
    )

    measure_list = tuple(m.strip() for m in measures.split(","))
    comp = CentralityComputer(measures=measure_list, betweenness_k=betweenness_k)
    result = comp.compute(graph)
    result.display(top_n=top_n)

    if save:
        ws = WorkspaceLayout(root=_ws_path(workspace))
        ws.init()
        out = ws.reports_dir / "graph" / f"centrality_{period}.json"
        result.save(out)
        console.print(f"[green]OK Report:[/green] {out}")


@graph_app.command("rings")
def graph_rings(
    txn:       str = typer.Argument(..., help="Transaction file"),
    period:    str = typer.Option("current", "--period", "-p"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    payer_col: str = typer.Option("account_id",    "--payer-col"),
    payee_col: str = typer.Option("beneficiary_id","--payee-col"),
    amount_col: str = typer.Option("txn_amount",   "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",  "--timestamp-col"),
    status_col: Optional[str] = typer.Option("status", "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status"),
    min_size:  int = typer.Option(3, "--min-size"),
    max_size:  int = typer.Option(50, "--max-size"),
    min_density: float = typer.Option(0.10, "--min-density"),
    resolution: float = typer.Option(1.0, "--resolution"),
    top_n:     int = typer.Option(15, "--top-n", "-n"),
    save:      bool = typer.Option(True, "--save/--no-save"),
):
    """Find candidate mule rings via Louvain community detection."""
    from ledgerhound.graph.community import CommunityFinder
    from ledgerhound.workspace.layout import WorkspaceLayout

    graph, _ = _build_graph_or_exit(
        txn, period, payer_col, payee_col, amount_col, timestamp_col,
        status_col, success_only=not no_filter_status,
    )

    finder = CommunityFinder(
        min_size=min_size,
        max_size=max_size,
        min_density=min_density,
        resolution=resolution,
    )
    result = finder.find(graph)
    result.display(top_n=top_n)

    if save:
        ws = WorkspaceLayout(root=_ws_path(workspace))
        ws.init()
        out = ws.reports_dir / "graph" / f"mule_rings_{period}.json"
        result.save(out)
        console.print(f"[green]OK Report:[/green] {out}")


@graph_app.command("trace")
def graph_trace(
    txn:        str = typer.Argument(..., help="Transaction file"),
    source:     Optional[str] = typer.Option(None, "--from", "-s",
                                              help="Source account (downstream/path)"),
    sink:       Optional[str] = typer.Option(None, "--to", "-t",
                                              help="Sink account (upstream/path)"),
    period:     str = typer.Option("current", "--period", "-p"),
    workspace:  str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    payer_col:  str = typer.Option("account_id",    "--payer-col"),
    payee_col:  str = typer.Option("beneficiary_id","--payee-col"),
    amount_col: str = typer.Option("txn_amount",    "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",   "--timestamp-col"),
    status_col: Optional[str] = typer.Option("status", "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status"),
    max_hops:   int = typer.Option(6, "--max-hops"),
    max_paths:  int = typer.Option(15, "--max-paths"),
    no_time_order: bool = typer.Option(False, "--no-time-order",
                                       help="Don't enforce ts > parent_ts"),
    save:       bool = typer.Option(True, "--save/--no-save"),
):
    """Trace money flow downstream from --from, upstream from --to,
    or k-shortest paths from --from to --to."""
    from ledgerhound.graph.flow import MoneyFlowTracer
    from ledgerhound.workspace.layout import WorkspaceLayout

    if not source and not sink:
        console.print("[red]Provide at least --from or --to.[/red]")
        raise typer.Exit(1)

    graph, _ = _build_graph_or_exit(
        txn, period, payer_col, payee_col, amount_col, timestamp_col,
        status_col, success_only=not no_filter_status,
    )

    tracer = MoneyFlowTracer(
        max_hops=max_hops,
        max_paths=max_paths,
        enforce_time_order=not no_time_order,
    )

    if source and sink:
        result = tracer.shortest_paths(graph, source, sink)
    elif source:
        result = tracer.trace_downstream(graph, source, max_hops=max_hops)
    else:
        result = tracer.trace_upstream(graph, sink, max_hops=max_hops)

    result.display()

    if save:
        ws = WorkspaceLayout(root=_ws_path(workspace))
        ws.init()
        tag = source or sink
        out = ws.reports_dir / "graph" / f"flow_{result.direction}_{tag}_{period}.json"
        result.save(out)
        console.print(f"[green]OK Report:[/green] {out}")


@graph_app.command("fingerprint")
def graph_fingerprint(
    txn:           str = typer.Argument(..., help="Transaction file"),
    period:        str = typer.Option("current", "--period", "-p"),
    workspace:     str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    account:       Optional[str] = typer.Option(None, "--account", "-a",
                                                  help="Single-account focus mode"),
    account_col:   str = typer.Option("account_id",    "--account-col"),
    beneficiary_col: str = typer.Option("beneficiary_id","--beneficiary-col"),
    amount_col:    str = typer.Option("txn_amount",    "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",      "--timestamp-col"),
    device_col:    Optional[str] = typer.Option("device_id", "--device-col"),
    status_col:    Optional[str] = typer.Option("status",    "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status"),
    min_txns:      int = typer.Option(5, "--min-txns"),
    top_n:         int = typer.Option(15, "--top-n", "-n"),
    save:          bool = typer.Option(True, "--save/--no-save"),
):
    """Compute behavioral mule fingerprints. Pass --account for one account, omit for all."""
    _check_file(txn, "Transaction file")
    from ledgerhound.graph.fingerprint import BehavioralFingerprinter
    from ledgerhound.workspace.layout import WorkspaceLayout

    txn_df = _load_txn(txn)
    fp_er = BehavioralFingerprinter(
        account_col=account_col,
        beneficiary_col=beneficiary_col,
        amount_col=amount_col,
        timestamp_col=timestamp_col,
        device_col=device_col,
        status_col=status_col,
        success_only=not no_filter_status,
        min_txns=min_txns,
    )

    if account:
        fp = fp_er.fingerprint_account(txn_df, account)
        if fp is None:
            console.print(f"[red]Account '{account}' has < {min_txns} txns or doesn't exist.[/red]")
            raise typer.Exit(1)
        sev_color = {"ALERT": "red", "WARNING": "yellow", "INFO": "dim"}.get(fp.severity, "white")
        console.print(Panel(
            f"[bold]{fp.account_id}[/bold]   "
            f"[{sev_color}]{fp.severity}[/{sev_color}]   "
            f"likelihood=[bold]{fp.mule_likelihood:.3f}[/bold]\n\n"
            f"[dim]txns:[/dim] {fp.n_txns}\n"
            f"[dim]night_ratio:[/dim] {fp.night_ratio:.0%}    "
            f"[dim]weekend:[/dim] {fp.weekend_ratio:.0%}\n"
            f"[dim]median gap:[/dim] {fp.median_gap_minutes:.1f} min   "
            f"[dim]gap CV:[/dim] {fp.inter_arrival_cv:.2f}\n"
            f"[dim]burst (5min):[/dim] {fp.burst_max_5min}\n"
            f"[dim]round amounts:[/dim] {fp.round_amount_ratio:.0%}   "
            f"[dim]amt entropy:[/dim] {fp.amount_entropy:.2f}\n"
            f"[dim]unique benef:[/dim] {fp.unique_beneficiaries}   "
            f"[dim]unique devices:[/dim] {fp.unique_devices}\n"
            f"[dim]benef concentration:[/dim] {fp.beneficiary_concentration:.0%}",
            title="Behavioral Fingerprint",
            border_style=sev_color,
        ))
        return

    result = fp_er.fingerprint_all(txn_df, period=period)
    result.display(top_n=top_n)

    if save:
        ws = WorkspaceLayout(root=_ws_path(workspace))
        ws.init()
        out = ws.reports_dir / "graph" / f"fingerprints_{period}.json"
        result.save(out)
        console.print(f"[green]OK Report:[/green] {out}")


@graph_app.command("summary")
def graph_summary(
    txn:       str = typer.Argument(..., help="Transaction file"),
    period:    str = typer.Option("current", "--period", "-p"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    payer_col: str = typer.Option("account_id",    "--payer-col"),
    payee_col: str = typer.Option("beneficiary_id","--payee-col"),
    amount_col: str = typer.Option("txn_amount",   "--amount-col"),
    timestamp_col: str = typer.Option("txn_date",  "--timestamp-col"),
    status_col: Optional[str] = typer.Option("status", "--status-col"),
    no_filter_status: bool = typer.Option(False, "--no-filter-status"),
    skip_motifs:  bool = typer.Option(False, "--skip-motifs"),
    skip_centrality: bool = typer.Option(False, "--skip-centrality"),
    skip_rings:   bool = typer.Option(False, "--skip-rings"),
    skip_fingerprint: bool = typer.Option(False, "--skip-fingerprint"),
):
    """Run the full graph mule investigation chain end-to-end."""
    from ledgerhound.graph.centrality import CentralityComputer
    from ledgerhound.graph.community import CommunityFinder
    from ledgerhound.graph.fingerprint import BehavioralFingerprinter
    from ledgerhound.graph.motifs import MotifDetector
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    ws.init()

    graph, txn_df = _build_graph_or_exit(
        txn, period, payer_col, payee_col, amount_col, timestamp_col,
        status_col, success_only=not no_filter_status,
    )

    console.print(Panel(
        f"[bold]Graph mule investigation chain -- {period}[/bold]\n"
        f"  graph: {graph.info()}\n"
        f"  workspace: {ws.root}",
        border_style="cyan",
    ))

    if not skip_motifs:
        from rich.rule import Rule
        console.print(Rule("[bold]1. Motif catalogue[/bold]"))
        m = MotifDetector().detect(graph)
        m.display()
        m.save(ws.reports_dir / "graph" / f"motifs_{period}.json")

    if not skip_centrality:
        from rich.rule import Rule
        console.print(Rule("[bold]2. Centrality[/bold]"))
        c = CentralityComputer(measures=("pagerank", "k_core")).compute(graph)
        c.display(top_n=10)
        c.save(ws.reports_dir / "graph" / f"centrality_{period}.json")

    if not skip_rings:
        from rich.rule import Rule
        console.print(Rule("[bold]3. Mule rings (Louvain)[/bold]"))
        r = CommunityFinder().find(graph)
        r.display(top_n=10)
        r.save(ws.reports_dir / "graph" / f"mule_rings_{period}.json")

    if not skip_fingerprint:
        from rich.rule import Rule
        console.print(Rule("[bold]4. Behavioral fingerprints[/bold]"))
        fp = BehavioralFingerprinter().fingerprint_all(txn_df, period=period)
        fp.display(top_n=10)
        fp.save(ws.reports_dir / "graph" / f"fingerprints_{period}.json")

    console.print(f"\n[bold green]OK Graph summary complete[/bold green] -- {ws.reports_dir / 'graph'}")


# ============================================================
# `ledgerhound knowledge` -- browse & manage institutional memory
# ============================================================

knowledge_app = typer.Typer(
    name="knowledge",
    help="Browse, search, and export the DatasetKnowledge institutional memory store.",
    no_args_is_help=True,
)
app.add_typer(knowledge_app, name="knowledge")

_K_TYPE_COLORS = {
    "quality_flag":    "red",
    "drift_event":     "yellow",
    "signal_decay":    "red",
    "signal_emerging": "green",
    "anomaly":         "yellow",
    "sampling_note":   "cyan",
    "pipeline_issue":  "red",
    "analyst_note":    "dim",
    "typology":        "blue",
    "boundary":        "magenta",
}


def _open_knowledge(workspace: str, dataset: str):
    """Load DatasetKnowledge for a dataset from the given workspace."""
    from ledgerhound.memory.knowledge import DatasetKnowledge
    from ledgerhound.workspace.layout import WorkspaceLayout
    ws = WorkspaceLayout(root=_ws_path(workspace))
    if not ws.memory_datasets_dir.exists():
        console.print(f"[red]No knowledge store found at {ws.memory_datasets_dir}[/red]")
        console.print("[dim]Run `ledgerhound workspace init` and then some analysis first.[/dim]")
        raise typer.Exit(1)
    return DatasetKnowledge.load(ws, dataset)


@knowledge_app.command("datasets")
def knowledge_datasets(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """List all datasets that have recorded knowledge in the workspace."""
    from rich.table import Table

    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    if not ws.memory_datasets_dir.exists():
        console.print(f"[yellow]No knowledge directory at {ws.memory_datasets_dir}[/yellow]")
        raise typer.Exit(0)

    jsonl_files = sorted(ws.memory_datasets_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print("[yellow]No dataset knowledge files found.[/yellow]")
        console.print("[dim]Run analysis steps (playbook, signal check, etc.) to populate.[/dim]")
        raise typer.Exit(0)

    t = Table(title=f"Dataset Knowledge -- {ws.root}", border_style="cyan")
    t.add_column("Dataset")
    t.add_column("Entries", justify="right")
    t.add_column("Active", justify="right")
    t.add_column("Types")
    t.add_column("Latest entry")

    import json as _json
    for jf in jsonl_files:
        name = jf.stem
        lines = [l for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries = []
        for l in lines:
            try:
                entries.append(_json.loads(l))
            except Exception:
                pass
        active = [e for e in entries if e.get("still_valid", True)]
        types  = sorted({e.get("entry_type", "?") for e in active})
        types_str = ", ".join(types[:4]) + ("..." if len(types) > 4 else "")
        latest = max((e.get("timestamp", "") for e in entries), default="--")[:19].replace("T", " ")
        t.add_row(name, str(len(entries)), str(len(active)), types_str, latest)
    console.print(t)


@knowledge_app.command("list")
def knowledge_list(
    dataset:    str = typer.Argument(..., help="Dataset name"),
    workspace:  str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    entry_type: Optional[str] = typer.Option(None, "--type", "-t",
                                              help="Filter by entry type"),
    feature:    Optional[str] = typer.Option(None, "--feature", "-f",
                                              help="Filter by feature name"),
    period:     Optional[str] = typer.Option(None, "--period", "-p",
                                              help="Filter by period label"),
    all_entries: bool = typer.Option(False, "--all",  "-a",
                                     help="Include superseded entries"),
    limit:      int = typer.Option(50, "--limit", "-n"),
):
    """List knowledge entries for a dataset.

    By default shows only active (not superseded) entries, newest first.
    """
    kb = _open_knowledge(workspace, dataset)
    entries = kb.search(
        feature=feature,
        entry_type=entry_type,
        valid_only=not all_entries,
    )
    if period:
        entries = [e for e in entries if e.period == period]

    entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    if not entries:
        console.print(f"[yellow]No entries found for '{dataset}'[/yellow]"
                      + (" (with filters applied)" if any([feature, entry_type, period]) else "") + ".")
        raise typer.Exit(0)

    from rich.table import Table
    t = Table(
        title=f"Knowledge -- {dataset} ({len(entries)} entries)",
        border_style="cyan", show_header=True, header_style="bold dim",
    )
    t.add_column("ID");       t.add_column("Type")
    t.add_column("Feature");  t.add_column("Period")
    t.add_column("Conf", justify="right")
    t.add_column("Claim");    t.add_column("Date")
    t.add_column("Valid")

    for e in entries:
        tc = _K_TYPE_COLORS.get(e.entry_type, "white")
        valid_str = "[green]OK[/green]" if e.still_valid else "[dim]X[/dim]"
        t.add_row(
            e.entry_id,
            f"[{tc}]{e.entry_type}[/{tc}]",
            e.feature or "--",
            e.period or "--",
            f"{e.confidence:.0%}",
            (e.claim[:60] + "..." if len(e.claim) > 60 else e.claim),
            e.timestamp[:10],
            valid_str,
        )
    console.print(t)
    console.print(f"[dim]Total entries in store: {len(kb)}[/dim]")


@knowledge_app.command("show")
def knowledge_show(
    dataset:   str = typer.Argument(..., help="Dataset name"),
    entry_id:  str = typer.Argument(..., help="Entry ID (e.g. k_1a2b3c4d)"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show the full detail of one knowledge entry."""
    kb = _open_knowledge(workspace, dataset)
    matches = [e for e in kb._entries if e.entry_id == entry_id]
    if not matches:
        console.print(f"[red]Entry '{entry_id}' not found in dataset '{dataset}'.[/red]")
        raise typer.Exit(1)
    e = matches[0]
    tc = _K_TYPE_COLORS.get(e.entry_type, "white")
    valid_label = "[green]ACTIVE[/green]" if e.still_valid else "[dim]SUPERSEDED[/dim]"
    console.print(Panel(
        f"[bold]{e.entry_id}[/bold]  [{tc}]{e.entry_type}[/{tc}]  {valid_label}\n\n"
        f"[bold]Claim:[/bold]    {e.claim}\n\n"
        f"[dim]Dataset:[/dim]   {dataset}\n"
        f"[dim]Feature:[/dim]   {e.feature or '--'}\n"
        f"[dim]Period:[/dim]    {e.period or '--'}\n"
        f"[dim]Confidence:[/dim]{e.confidence:.0%}\n"
        f"[dim]Session:[/dim]   {e.session_id or '--'}\n"
        f"[dim]Timestamp:[/dim] {e.timestamp}\n"
        + (f"\n[bold]Detail:[/bold]\n{e.detail}" if e.detail else "")
        + (f"\n\n[dim]Evidence ref: {e.evidence_ref}[/dim]" if e.evidence_ref else "")
        + (f"\n[dim]Tags: {', '.join(e.tags)}[/dim]" if e.tags else ""),
        title="Knowledge Entry",
        border_style=tc,
    ))


@knowledge_app.command("search")
def knowledge_search(
    dataset:   str = typer.Argument(..., help="Dataset name"),
    query:     str = typer.Argument(..., help="Text to search for in claims and details"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    all_entries: bool = typer.Option(False, "--all", help="Include superseded entries"),
    limit:     int = typer.Option(20, "--limit", "-n"),
):
    """Full-text search across knowledge claims and detail fields."""
    kb = _open_knowledge(workspace, dataset)
    q = query.lower()
    results = [
        e for e in kb._entries
        if (not e.still_valid and not all_entries) is False
        and (
            q in e.claim.lower()
            or q in (e.detail or "").lower()
            or q in (e.feature or "").lower()
            or q in e.entry_type
            or any(q in tag.lower() for tag in (e.tags or []))
        )
    ]
    results = sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    if not results:
        console.print(f"[yellow]No entries matching '{query}' in '{dataset}'.[/yellow]")
        raise typer.Exit(0)

    from rich.table import Table
    t = Table(title=f"Search '{query}' -- {len(results)} match(es)", border_style="cyan")
    t.add_column("ID"); t.add_column("Type"); t.add_column("Feature")
    t.add_column("Claim"); t.add_column("Date")

    for e in results:
        tc = _K_TYPE_COLORS.get(e.entry_type, "white")
        t.add_row(
            e.entry_id,
            f"[{tc}]{e.entry_type}[/{tc}]",
            e.feature or "--",
            (e.claim[:70] + "..." if len(e.claim) > 70 else e.claim),
            e.timestamp[:10],
        )
    console.print(t)


@knowledge_app.command("invalidate")
def knowledge_invalidate(
    dataset:   str = typer.Argument(..., help="Dataset name"),
    entry_id:  str = typer.Argument(..., help="Entry ID to mark as superseded"),
    reason:    str = typer.Option("", "--reason", "-r", help="Why this entry is being superseded"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Mark a knowledge entry as superseded (no longer valid).

    The entry is kept in the JSONL store with `still_valid=false` so the
    history is preserved.  It will no longer appear in agent context or
    `list` output by default.
    """
    kb = _open_knowledge(workspace, dataset)
    matches = [e for e in kb._entries if e.entry_id == entry_id]
    if not matches:
        console.print(f"[red]Entry '{entry_id}' not found in dataset '{dataset}'.[/red]")
        raise typer.Exit(1)
    e = matches[0]
    if not e.still_valid:
        console.print(f"[yellow]Entry '{entry_id}' is already superseded.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Entry to supersede:[/bold] {e.claim[:80]}")
    confirmed = typer.confirm("Mark as superseded?")
    if not confirmed:
        console.print("[dim]Cancelled.[/dim]")
        raise typer.Exit(0)

    kb.invalidate(entry_id, reason=reason)
    console.print(f"[green]OK Entry '{entry_id}' marked as superseded.[/green]")


@knowledge_app.command("export")
def knowledge_export(
    dataset:   str = typer.Argument(..., help="Dataset name"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    fmt:       str = typer.Option("md", "--format", "-f",
                                  help="Output format: md | json | csv"),
    output:    Optional[str] = typer.Option(None, "--output", "-o",
                                            help="Output path. Defaults to stdout."),
    all_entries: bool = typer.Option(False, "--all",
                                     help="Include superseded entries in export"),
):
    """Export all knowledge entries for a dataset.

    Formats:
      md   -- human-readable Markdown (default)
      json -- machine-readable JSON array
      csv  -- spreadsheet-friendly CSV
    """
    import csv
    import io as _io

    kb = _open_knowledge(workspace, dataset)
    entries = kb._entries if all_entries else [e for e in kb._entries if e.still_valid]

    if fmt == "md":
        content = kb.summary()
    elif fmt == "json":
        import json as _json
        content = _json.dumps([e.to_dict() for e in entries], indent=2, default=str)
    elif fmt == "csv":
        buf = _io.StringIO()
        w = csv.DictWriter(buf, fieldnames=[
            "entry_id", "timestamp", "session_id", "entry_type",
            "feature", "claim", "confidence", "period", "tags", "still_valid",
        ])
        w.writeheader()
        for e in entries:
            row = e.to_dict()
            row["tags"] = "|".join(row.get("tags") or [])
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
        content = buf.getvalue()
    else:
        console.print(f"[red]Unknown format '{fmt}'. Use: md, json, csv[/red]")
        raise typer.Exit(1)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]OK Exported {len(entries)} entries -> {output}[/green]")
    else:
        console.print(content)


# ============================================================
# `ledgerhound scorecard` -- composite EDA quality scorecard
# ============================================================


@app.command("scorecard")
def scorecard_cmd(
    workspace:    str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    dataset:      str = typer.Option("default", "--dataset", "-d",
                                      help="Dataset name (matches knowledge file stem)"),
    period:       Optional[str] = typer.Option(None, "--period", "-p",
                                                help="Override period; defaults to latest playbook run"),
    output:       Optional[str] = typer.Option(None, "--output", "-o",
                                                help="Save JSON to this path; defaults to <workspace>/reports/scorecard.json"),
    no_display:   bool = typer.Option(False, "--no-display", help="Skip Rich rendering"),
):
    """Build the Ledgerhound EDA Scorecard for a dataset's most recent run.

    Reconstructs all inputs from the workspace's persisted artifacts:
      - playbook reports         -> completeness + data quality
      - findings JSONL           -> discipline (sample-frame validation)
      - coverage telemetry       -> completeness
      - graph reports            -> corroboration
      - DatasetKnowledge entries -> process

    Outputs a 5-component scorecard with an overall 0..1 score, per-component
    sub-metrics, and remediation suggestions for any component below 0.85.
    """
    from ledgerhound.scorecard import Scorecard
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    if not ws.root.exists():
        console.print(f"[red]Workspace not found: {ws.root}[/red]")
        console.print("[dim]Run `ledgerhound workspace init` and at least one playbook first.[/dim]")
        raise typer.Exit(1)

    sc = Scorecard.from_workspace(ws, dataset_name=dataset, period=period)

    if not no_display:
        sc.display()

    out_path = Path(output) if output else (ws.reports_dir / "scorecard.json")
    sc.save(out_path)
    console.print(f"[green]OK Scorecard saved:[/green] {out_path}")


# ============================================================
# `ledgerhound monitor` -- scheduled drift monitor (C1)
# ============================================================

monitor_app = typer.Typer(
    name="monitor",
    help="Schedule signal-decay + typology-evolution monitoring with webhook alerts.",
    no_args_is_help=True,
)
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("init")
def monitor_init(
    dataset:       str = typer.Option(..., "--dataset", "-d"),
    fraud:         Optional[str] = typer.Option(None, "--fraud", "-f",
                                                 help="fraud parquet/csv (omit if using --trino-fraud-sql)"),
    nonfr:         Optional[str] = typer.Option(None, "--nonfr", "-n",
                                                 help="non-fraud parquet/csv (omit if using --trino-nonfr-sql)"),
    workspace:     str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    raw_txn:       Optional[str] = typer.Option(None, "--raw-txn",
                                                 help="optional raw_txn for typology evolution"),
    features:      Optional[str] = typer.Option(None, "--features",
                                                 help="comma-separated feature list"),
    granularity:   str = typer.Option("quarterly", "--granularity",
                                       help="quarterly | monthly | weekly | daily"),
    interval_seconds: int = typer.Option(3600, "--interval",
                                          help="daemon interval in seconds (ignored if --days is set)"),
    days:          Optional[str] = typer.Option(None, "--days",
                                                 help="comma-separated days to run, e.g. mon,thu "
                                                      "(switches schedule to calendar mode)"),
    time_of_day:   str = typer.Option("06:00", "--time", help="HH:MM 24h, used with --days"),
    tz:            str = typer.Option("UTC", "--timezone", help="IANA timezone, used with --days"),
    trino_fraud_sql: Optional[str] = typer.Option(None, "--trino-fraud-sql",
                                                   help="pull fraud slice from Trino instead of a file"),
    trino_nonfr_sql: Optional[str] = typer.Option(None, "--trino-nonfr-sql",
                                                   help="pull non-fraud slice from Trino instead of a file"),
    trino_raw_txn_sql: Optional[str] = typer.Option(None, "--trino-raw-txn-sql"),
    trino_max_rows: int = typer.Option(2_000_000, "--trino-max-rows",
                                        help="hard row cap applied to every Trino pull"),
    auto_investigate: bool = typer.Option(False, "--auto-investigate",
                                           help="auto-investigate the worst alert each cycle with a "
                                                "real LLM session (see docs/v3.0/INFOSEC.md)"),
    webhook_url:   Optional[str] = typer.Option(None, "--webhook",
                                                 help="POST URL for alerts"),
    force:         bool = typer.Option(False, "--force", "-x"),
):
    """Create a monitor config at <workspace>/monitor/config.yaml.

    Data source is either files (--fraud/--nonfr) or Trino
    (--trino-fraud-sql/--trino-nonfr-sql) -- pick one. Trino-sourced SQL
    must be approved before it will run: `ledgerhound monitor approve-sql`.
    """
    from ledgerhound.monitor import AlertThresholds, DriftMonitorConfig, ScheduleConfig, TrinoSourceConfig, WebhookConfig
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    ws.init()
    cfg_path = DriftMonitorConfig.config_path(ws)
    if cfg_path.exists() and not force:
        console.print(f"[yellow]Config already exists at {cfg_path}[/yellow]")
        console.print("[dim]Pass --force to overwrite.[/dim]")
        raise typer.Exit(0)

    feat_list = (
        [f.strip() for f in features.split(",")] if features else
        ["sim_change_count", "device_change_count", "night_txn_rate",
         "failed_txn_rate", "new_benef_rate", "txn_velocity_7d",
         "avg_txn_amount", "unique_beneficiaries"]
    )

    if days:
        schedule = ScheduleConfig(
            mode="calendar",
            days=[d.strip() for d in days.split(",")],
            time_of_day=time_of_day,
            timezone=tz,
        )
    else:
        schedule = ScheduleConfig(mode="interval", interval_seconds=interval_seconds)

    trino_cfg = TrinoSourceConfig(
        enabled=bool(trino_fraud_sql or trino_nonfr_sql),
        fraud_sql=trino_fraud_sql or "",
        nonfr_sql=trino_nonfr_sql or "",
        raw_txn_sql=trino_raw_txn_sql or "",
        max_rows=trino_max_rows,
    )

    cfg = DriftMonitorConfig(
        dataset_name=dataset,
        fraud_data=fraud or "",
        nonfr_data=nonfr or "",
        raw_txn_data=raw_txn,
        features=feat_list,
        period_granularity=granularity,
        schedule_interval_seconds=interval_seconds,
        schedule=schedule,
        trino=trino_cfg,
        auto_investigate=auto_investigate,
        alerts=AlertThresholds(),
        webhooks=([WebhookConfig(url=webhook_url)] if webhook_url else []),
    )
    issues = cfg.validate()
    if issues:
        console.print("[yellow]Config validation issues:[/yellow]")
        for i in issues:
            console.print(f"  [yellow]![/yellow] {i}")
    saved = cfg.save(ws)

    if schedule.mode == "calendar":
        sched_line = f"  schedule:    {', '.join(schedule.days) or 'every day'} @ {schedule.time_of_day} {schedule.timezone}"
    else:
        sched_line = f"  schedule:    every {schedule.interval_seconds}s"

    console.print(Panel(
        f"[bold green]Monitor config created[/bold green]\n"
        f"  path:        {saved}\n"
        f"  dataset:     {cfg.dataset_name}\n"
        f"  source:      {'Trino (needs approve-sql)' if trino_cfg.enabled else 'files'}\n"
        f"  fraud:       {cfg.fraud_data or cfg.trino.fraud_sql}\n"
        f"  nonfr:       {cfg.nonfr_data or cfg.trino.nonfr_sql}\n"
        f"  features:    {len(cfg.features)} columns\n"
        f"  granularity: {cfg.period_granularity}\n"
        f"{sched_line}\n"
        f"  auto-investigate: {cfg.auto_investigate}\n"
        f"  webhooks:    {len(cfg.webhooks)}",
        border_style="green",
    ))
    if trino_cfg.enabled:
        console.print(
            "[yellow]Trino SQL is not yet approved -- run "
            "`ledgerhound monitor approve-sql --workspace " + workspace + "` "
            "before the first cycle.[/yellow]"
        )


@monitor_app.command("approve-sql")
def monitor_approve_sql(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    approved_by: str = typer.Option(..., "--by", help="your name/id, recorded in the config"),
):
    """Review and sign off the monitor's Trino SQL so unattended cycles can run it.

    Prints the exact SQL that will be pinned -- read it before approving.
    Re-run this any time the SQL changes; an unapproved or edited query
    blocks the monitor from running (ledgerhound.monitor.trino_source).
    """
    from ledgerhound.monitor import DriftMonitorConfig
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    try:
        cfg = DriftMonitorConfig.load(ws)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not cfg.trino.enabled:
        console.print("[yellow]This monitor is not Trino-sourced -- nothing to approve.[/yellow]")
        raise typer.Exit(0)

    from rich.syntax import Syntax
    for label, sql in (("fraud_sql", cfg.trino.fraud_sql),
                       ("nonfr_sql", cfg.trino.nonfr_sql),
                       ("raw_txn_sql", cfg.trino.raw_txn_sql)):
        if sql:
            console.print(f"[bold]{label}:[/bold]")
            console.print(Syntax(sql.strip(), "sql", theme="monokai", line_numbers=True))
    console.print(f"[dim]row cap: {cfg.trino.max_rows:,}[/dim]")

    if not typer.confirm(f"Approve this SQL as {approved_by!r} for unattended runs?"):
        console.print("[yellow]Not approved.[/yellow]")
        raise typer.Exit(1)

    cfg.trino.approve(approved_by=approved_by)
    cfg.save(ws)
    console.print(f"[green]Approved by {approved_by} at {cfg.trino.approved_at}[/green]")


@monitor_app.command("run")
def monitor_run(
    workspace:     str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    once:          bool = typer.Option(False, "--once", help="single cycle then exit"),
    interval:      Optional[int] = typer.Option(None, "--interval",
                                                 help="override config interval (seconds)"),
    period:        Optional[str] = typer.Option(None, "--period",
                                                 help="override derived period"),
    max_cycles:    Optional[int] = typer.Option(None, "--max-cycles",
                                                 help="stop after N cycles"),
    no_webhooks:   bool = typer.Option(False, "--no-webhooks",
                                        help="run detectors but skip webhook emission"),
    llm_config:    Optional[str] = typer.Option(None, "--llm-config",
                                                 help="required when the config has "
                                                      "auto_investigate=true"),
):
    """Run the monitor -- either once (cron mode) or as a daemon loop."""
    from ledgerhound.monitor import DriftMonitor, DriftMonitorConfig
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    if not ws.root.exists():
        console.print(f"[red]Workspace not found: {ws.root}[/red]")
        raise typer.Exit(1)

    try:
        cfg = DriftMonitorConfig.load(ws)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    llm = None
    if cfg.auto_investigate:
        if not llm_config:
            console.print("[red]This monitor has auto_investigate=true -- pass "
                          "--llm-config <yaml>[/red]")
            raise typer.Exit(1)
        from ledgerhound.llm.client import LLMClient
        llm = LLMClient.from_config(llm_config)
        if cfg.trino.enabled:
            console.print(
                "[yellow]auto_investigate + trino: the Trino-sourced slice "
                f"will be sent to {llm_config} on any WARNING/FAIL alert. "
                "Confirm this points at your internal/regulated LLM gateway, "
                "not a public API -- see docs/v3.0/ZONING.md.[/yellow]"
            )

    monitor = DriftMonitor(cfg, ws, llm=llm)

    if once:
        result = monitor.run_cycle(period=period, emit_webhooks=not no_webhooks)
        _display_run_result(result)
        if result.status == "ERROR":
            raise typer.Exit(2)
    else:
        if interval is not None:
            sched_desc = f"every {interval}s (--interval override)"
        elif cfg.schedule.mode == "calendar":
            days_desc = ", ".join(cfg.schedule.days) or "every day"
            sched_desc = f"{days_desc} @ {cfg.schedule.time_of_day} {cfg.schedule.timezone}"
        else:
            sched_desc = f"every {cfg.schedule.interval_seconds}s"
        console.print(Panel(
            f"[bold]Monitor daemon[/bold] -- {cfg.dataset_name}\n"
            f"  schedule: {sched_desc}\n"
            f"  max_cycles: {max_cycles or 'unlimited'}\n"
            f"  [dim]press Ctrl+C to stop[/dim]",
            border_style="cyan",
        ))
        n = monitor.run_daemon(interval_seconds=interval, max_cycles=max_cycles)
        console.print(f"[green]OK Completed {n} cycle(s)[/green]")


def _display_run_result(result) -> None:
    """Render a single MonitorRunResult."""
    sev_colours = {"FAIL": "red", "WARNING": "yellow", "INFO": "cyan", "NONE": "green"}
    c = sev_colours.get(result.overall_alert, "white")
    console.print(Panel(
        f"[bold]Cycle:[/bold] {result.cycle_id}\n"
        f"  dataset/period:  {result.dataset_name} / {result.period}\n"
        f"  status:          {result.status}\n"
        f"  overall_alert:   [{c}]{result.overall_alert}[/{c}]\n"
        f"  decay events:    {result.n_signal_decay_events}\n"
        f"  evolution events: {result.n_evolution_events}\n"
        f"  alerts emitted:  {result.n_alerts_emitted}\n"
        f"  webhooks:        {result.n_webhooks_succeeded}/{result.n_webhooks_attempted} OK\n"
        f"  manifest:        {result.manifest_path or '--'}\n"
        f"  investigation:   {result.investigation['n_findings_recorded'] if result.investigation else '--'}"
        f"{' finding(s) at ' + result.investigation['run_dir'] if result.investigation else ''}\n"
        f"  elapsed:         {result.elapsed_ms:.0f}ms",
        border_style=c,
    ))
    if result.alerts:
        from rich.table import Table
        t = Table("Severity", "Type", "Feature/Cluster", "Claim", show_header=True,
                  header_style="bold dim")
        for a in result.alerts[:10]:
            sc = sev_colours.get(a.severity, "white")
            target = a.feature or a.cluster_key or "--"
            t.add_row(
                f"[{sc}]{a.severity}[/{sc}]",
                a.event_type,
                target,
                (a.claim[:70] + "..." if len(a.claim) > 70 else a.claim),
            )
        console.print(t)
    if result.error:
        console.print(f"[red]error: {result.error[:300]}[/red]")


@monitor_app.command("status")
def monitor_status(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show the active config + latest cycle summary."""
    from ledgerhound.monitor import DriftMonitor, DriftMonitorConfig
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    try:
        cfg = DriftMonitorConfig.load(ws)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if cfg.schedule.mode == "calendar":
        sched_line = (f"  schedule: {', '.join(cfg.schedule.days) or 'every day'} "
                      f"@ {cfg.schedule.time_of_day} {cfg.schedule.timezone}")
    else:
        sched_line = f"  schedule: every {cfg.schedule.interval_seconds}s"

    console.print(Panel(
        f"[bold]Config[/bold] -- {cfg.dataset_name}\n"
        f"  source: {'Trino' if cfg.trino.enabled else 'files'}\n"
        f"  fraud: {cfg.fraud_data or cfg.trino.fraud_sql}\n"
        f"  nonfr: {cfg.nonfr_data or cfg.trino.nonfr_sql}\n"
        f"  features: {len(cfg.features)}\n"
        f"  granularity: {cfg.period_granularity}\n"
        f"{sched_line}\n"
        f"  auto-investigate: {cfg.auto_investigate}\n"
        f"  webhooks: {len(cfg.webhooks)} configured",
        border_style="cyan",
    ))

    monitor = DriftMonitor(cfg, ws)
    latest = monitor.latest()
    if latest is None:
        console.print("[dim]No cycles run yet. Use `ledgerhound monitor run --once`.[/dim]")
        return
    _display_run_result(latest)


@monitor_app.command("history")
def monitor_history(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    limit:     int = typer.Option(20, "--limit", "-n"),
):
    """Show recent cycles from <workspace>/monitor/runs.jsonl."""
    from rich.table import Table

    from ledgerhound.monitor import DriftMonitor, DriftMonitorConfig
    from ledgerhound.workspace.layout import WorkspaceLayout

    ws = WorkspaceLayout(root=_ws_path(workspace))
    try:
        cfg = DriftMonitorConfig.load(ws)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    monitor = DriftMonitor(cfg, ws)
    rows = monitor.history(limit=limit)
    if not rows:
        console.print("[dim]No cycles found.[/dim]")
        return

    sev_colours = {"FAIL": "red", "WARNING": "yellow", "INFO": "cyan", "NONE": "green"}
    t = Table("Cycle", "Period", "Status", "Alert", "Decay", "Evo",
              "Webhooks", "Elapsed", title="Monitor history (newest first)",
              show_header=True, header_style="bold dim")
    for r in rows:
        c = sev_colours.get(r.overall_alert, "white")
        t.add_row(
            r.cycle_id[-12:],
            r.period,
            "[red]ERROR[/red]" if r.status == "ERROR" else "OK",
            f"[{c}]{r.overall_alert}[/{c}]",
            str(r.n_signal_decay_events),
            str(r.n_evolution_events),
            f"{r.n_webhooks_succeeded}/{r.n_webhooks_attempted}",
            f"{r.elapsed_ms:.0f}ms",
        )
    console.print(t)


# ============================================================================
# M7 -- CLI surfaces for compute doctor + longitudinal memory
# ============================================================================


@app.command("compute-doctor")
def compute_doctor(
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w",
        help="If given, write compute_profile.json inside this workspace. "
             "Otherwise it goes to ~/.ledgerhound/compute_profile.json.",
    ),
    n_rows: int = typer.Option(
        1_000_000, "--n-rows",
        help="Row count for the dataframe groupby benchmark.",
    ),
    n_vec: int = typer.Option(
        100_000, "--n-vec",
        help="Vector count for the recall benchmark.",
    ),
    vector_dim: int = typer.Option(
        384, "--vector-dim",
        help="Embedding dimension used in the vector benchmark.",
    ),
    save: bool = typer.Option(
        True, "--save/--no-save",
        help="Persist the resulting compute_profile.json. --no-save just prints.",
    ),
):
    """Benchmark this machine and pick preferred compute backends.

    Times pandas vs cuDF for a 1M-row groupby, and numpy vs faiss vs cuvs
    for a 100K-vector recall. Writes the winner to compute_profile.json so
    the ComputeRouter can pick backends without re-deciding per run.

    Run once per machine, or whenever you install/uninstall cuDF / faiss /
    cuVS / sentence-transformers.
    """
    from ledgerhound.doctor import run_doctor
    try:
        report = run_doctor(
            workspace_root=workspace,
            save=save,
            n_rows=n_rows,
            n_vec=n_vec,
            vector_dim=vector_dim,
        )
    except Exception as exc:
        console.print(f"[red]doctor failed:[/red] {exc}")
        raise typer.Exit(1)
    report.display()
    if save:
        from ledgerhound.doctor import default_profile_path
        path = default_profile_path(workspace)
        console.print(f"\n[dim]profile saved: {path}[/dim]")


# -- Longitudinal memory subcommands (recall + stats) ----------------------
#
# These extend the existing `memory_app` (which already has init/show/list
# for the YAML-based Dataset Memory). The new commands target the vector-
# backed LongitudinalMemory introduced in Phase 4.


_DEFAULT_LONGITUDINAL_ROOT = ".ledgerhound/longitudinal"


def _open_longitudinal(root: str, *, centroid_dim: int = 8, fingerprint_dim: int = 14):
    """Best-effort open: returns (memory, embedder) or exits with a message."""
    from ledgerhound.memory import Embedder, LongitudinalMemory
    root_path = Path(root)
    if not root_path.exists():
        console.print(
            f"[yellow]No longitudinal memory at {root_path}[/yellow]\n"
            f"[dim]It will be created on the next FindingsLog write, or by "
            f"calling LongitudinalMemory(root='{root}') from Python.[/dim]"
        )
        raise typer.Exit(1)
    emb = Embedder(device="auto")
    mem = LongitudinalMemory(
        root=root_path,
        embedder=emb,
        centroid_dim=centroid_dim,
        fingerprint_dim=fingerprint_dim,
    )
    return mem, emb


@memory_app.command("stats")
def memory_stats(
    root: str = typer.Option(
        _DEFAULT_LONGITUDINAL_ROOT, "--root", "-r",
        help="Path to the longitudinal memory directory.",
    ),
):
    """Show counts, dims, and backends for each namespace of the longitudinal
    memory (text / centroid / fingerprint)."""
    mem, emb = _open_longitudinal(root)
    info = mem.stats()
    emb_info = emb.info()

    from rich.table import Table
    console.print(Panel(
        f"[bold]Longitudinal Memory[/bold]\n"
        f"root: [cyan]{info['root']}[/cyan]\n"
        f"embedder: {emb_info.backend} . {emb_info.model_name} . "
        f"device={emb_info.device} . dim={emb_info.text_dim}",
        border_style="cyan",
    ))
    t = Table("namespace", "count", "dim", "backend", title="Namespaces")
    for ns, row in info["namespaces"].items():
        t.add_row(ns, str(row["count"]), str(row["dim"]), row["backend"])
    console.print(t)


@memory_app.command("recall")
def memory_recall(
    query: str = typer.Argument(..., help="Free-text query (semantic recall over text namespace)."),
    k: int = typer.Option(5, "-k", "--top-k", help="Number of nearest records to return."),
    namespace: str = typer.Option(
        "text", "--namespace", "-n",
        help="text | centroid | fingerprint. Centroid/fingerprint take a "
             "numeric query -- use Python for those for now.",
    ),
    window: Optional[str] = typer.Option(
        None, "--window",
        help="Time window: last_30d | last_3m | last_1y | "
             "since:2024-01-01 | between:2024-01-01:2024-06-30",
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Restrict to one dataset name.",
    ),
    source: Optional[str] = typer.Option(
        None, "--source",
        help="Restrict to one source: finding | typology_centroid | "
             "account_fingerprint | knowledge_entry | free_text",
    ),
    include_superseded: bool = typer.Option(
        False, "--include-superseded",
        help="Show records that have been superseded by newer ones.",
    ),
    root: str = typer.Option(
        _DEFAULT_LONGITUDINAL_ROOT, "--root", "-r",
        help="Path to the longitudinal memory directory.",
    ),
):
    """Semantic recall over the longitudinal memory text namespace.

    Examples:
      ledgerhound memory recall "any motif velocity findings"
      ledgerhound memory recall "PSI drift" --window last_3m --dataset upi_q2
      ledgerhound memory recall "fraud rings" -k 10 --include-superseded
    """
    mem, _ = _open_longitudinal(root)
    try:
        results = mem.recall(
            query, namespace=namespace, k=k,
            window=window, dataset=dataset, source=source,
            include_superseded=include_superseded,
        )
    except ValueError as e:
        console.print(f"[red]X {e}[/red]")
        raise typer.Exit(1)

    if not results:
        console.print(f"[yellow]No matches for {query!r}[/yellow]")
        return

    from rich.table import Table
    t = Table("score", "id", "source", "dataset", "period", "as_of", "text",
              title=f"Recall: {query[:60]}")
    for r in results:
        rec = r.record
        text_preview = (rec.text or "")[:80] + ("..." if rec.text and len(rec.text) > 80 else "")
        if not text_preview and rec.extra:
            text_preview = f"[{', '.join(f'{k}={v}' for k, v in list(rec.extra.items())[:2])}]"
        t.add_row(
            f"{r.score:.3f}",
            rec.id[:18],
            rec.source,
            rec.dataset or "-",
            rec.period or "-",
            (rec.as_of or "")[:10],
            text_preview,
        )
    console.print(t)


# ============================================================================
# B5 -- Reproducibility manifest CLI
# ============================================================================


@app.command("manifest")
def manifest_cmd(
    workspace: str = typer.Argument(..., help="Workspace root to snapshot."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Where to write MANIFEST.json. Defaults to <workspace>/MANIFEST.json.",
    ),
    extra: list[str] = typer.Option(
        [], "--include", "-i",
        help="Additional dir names (relative to workspace) to hash. "
             "Repeatable. Defaults: reports, datasets, samples, findings, knowledge.",
    ),
    skip: list[str] = typer.Option(
        [], "--skip",
        help="Directory names anywhere in the tree to skip (additive to defaults).",
    ),
    seed: list[str] = typer.Option(
        [], "--seed",
        help='Random seeds to record, as "name=value". Repeatable. '
             'Example: --seed numpy=42 --seed sklearn=0',
    ),
    command: Optional[str] = typer.Option(
        None, "--command",
        help="Command line that produced this workspace (recorded verbatim).",
    ),
    no_display: bool = typer.Option(
        False, "--no-display", help="Skip the Rich summary table.",
    ),
):
    """Build a reproducibility manifest for a workspace.

    Walks the workspace, hashes every artifact under reports/ datasets/
    samples/ findings/ knowledge/ (plus any --include dirs), captures the
    git commit and environment, and writes MANIFEST.json.

    Pair with op_log.jsonl for full audit: ops + file hashes + env = replayable run.
    """
    from ledgerhound.reproducibility import build_manifest

    # Parse --seed name=value pairs
    seeds: dict[str, Any] = {}
    for s in seed:
        if "=" in s:
            k, v = s.split("=", 1)
            # Try int then float then str
            try:
                seeds[k] = int(v)
            except ValueError:
                try:
                    seeds[k] = float(v)
                except ValueError:
                    seeds[k] = v
        else:
            console.print(f"[yellow]Ignoring malformed --seed {s!r} (expected name=value)[/yellow]")

    try:
        m = build_manifest(
            workspace_root=workspace,
            seeds=seeds,
            command=command,
            extra_dirs=extra,
            skip_dirs=skip,
        )
    except Exception as exc:
        console.print(f"[red]manifest build failed:[/red] {exc}")
        raise typer.Exit(1)

    out_path = Path(output) if output else _ws_path(workspace) / "MANIFEST.json"
    m.save(out_path)
    if not no_display:
        m.display()
    console.print(f"[green]OK[/green] manifest saved: [cyan]{out_path}[/cyan]")


@app.command("run-views")
def run_views_cmd(
    run_dir: str = typer.Argument(..., help="A shell session's run folder (contains events.jsonl)."),
):
    """Regenerate session_code.py and findings.jsonl from a run's events.jsonl.

    V3-012: both files are projections of the hash-chained event stream --
    this proves it by rebuilding them from events.jsonl alone and confirms
    they exist. Run `ledgerhound replay` first if you also want to confirm the
    recorded code actually reproduces its outputs.
    """
    from ledgerhound.runtime.events import EventStream
    from ledgerhound.runtime.views import regenerate_all

    p = Path(run_dir)
    events_path = p / "events.jsonl"
    if not events_path.exists():
        console.print(f"[red]no events.jsonl found under {p}[/red]")
        raise typer.Exit(1)

    vr = EventStream(events_path).verify()
    if not vr.ok:
        console.print(f"[red]event chain broken at seq {vr.first_bad_seq}:[/red] {vr.reason}")
        raise typer.Exit(1)

    out = regenerate_all(p)
    console.print(f"[green]OK[/green] regenerated from {vr.n_events} events:")
    for name, path in out.items():
        console.print(f"  {name}: [cyan]{path}[/cyan]")


@app.command("replay")
def replay_cmd(
    run_dir: str = typer.Argument(..., help="A shell session's run folder (contains events.jsonl)."),
):
    """Re-execute a recorded session's code against the same data and report PASS/DIFF.

    V3-013: verifies the event chain is untampered, checks the input data
    file's hash against what was recorded at load_data time, then replays
    each successful exec-mode code_exec cell in a fresh sandbox and compares
    its output to what was recorded. Kernel-mode cells are reported SKIPPED
    (replaying them needs the kernel -- see ledgerhound.runtime.replay).
    """
    from ledgerhound.runtime.replay import replay

    report = replay(run_dir)
    console.print(report.render())
    if not report.passed:
        raise typer.Exit(1)


# =========================================================================== #
#  ledgerhound start  -- interactive experiment wizard                            #
# =========================================================================== #

@app.command()
def start(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w",
                                   help="Root workspace directory"),
):
    """Start a new experiment or resume a previous session.

    Guides you through dataset selection, goal setting, report format,
    auto-profiling, and launches the self-healing analysis loop.
    """
    from ledgerhound.cli.start_wizard import run_wizard
    run_wizard(workspace=workspace)


# =========================================================================== #
#  P0 backbone -- version / doctor / upgrade                                    #
# =========================================================================== #


def _installed_version() -> str:
    """Resolve the installed ledgerhound version, dev-tree friendly."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("ledgerhound")
        except PackageNotFoundError:
            return "0.0.0+dev"
    except Exception:
        return "unknown"


@app.command()
def version():
    """Print the installed Ledgerhound version + build stamp."""
    from ledgerhound import build_info
    b = build_info()
    stamp = b.get("build", "dev")
    if b.get("commit"):
        stamp = f"{b.get('commit')} ({b.get('built_at', '?')})"
    console.print(f"ledgerhound [bold cyan]{_installed_version()}[/bold cyan] "
                  f"build [bold]{stamp}[/bold]")
    if stamp == "dev":
        console.print("[dim]No build stamp -- this is an unstamped checkout. "
                      "Built zips carry a git hash so stale deploys are "
                      "obvious.[/dim]")


@app.command()
def doctor(
    minio_config: Optional[str] = typer.Option(
        None, "--minio-config",
        help="MinIO config to probe (default: integrations.minio_config "
             "in main_config.yaml, else configs/minio.yaml)"),
):
    """Diagnose the environment: version, identity, secrets, MinIO connectivity."""
    from rich.table import Table

    from ledgerhound.config import get_str, resolved_path, search_candidates
    minio_config = (minio_config
                    or get_str("integrations", "minio_config")
                    or "configs/minio.yaml")

    console.print(Panel(f"[bold]Ledgerhound Doctor[/bold]  "
                        f"v{_installed_version()}", border_style="cyan"))

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Check")
    tbl.add_column("Status")
    tbl.add_column("Detail")

    from rich.markup import escape as _esc_d

    def row(check: str, ok: bool | None, detail: str = ""):
        mark = "[green]OK[/green]" if ok else ("[yellow]--[/yellow]" if ok is None
                                               else "[red]MISSING[/red]")
        # details carry [codes] and paths -- never let Rich eat them as markup
        tbl.add_row(_esc_d(check), mark, _esc_d(detail))

    # V3-066: this command still read os.environ directly, which is
    # exactly backwards now that main_config.yaml is the primary source --
    # a user with everything correctly filled in there saw "MISSING" here
    # and had no way to tell the diagnostic was checking the wrong place.
    mc_path = resolved_path()
    row("main_config.yaml", mc_path is not None,
        str(mc_path) if mc_path else
        "not found -- checked: " + ", ".join(str(c) for c in search_candidates()))

    # Identity (Tier 1): the first-run work login counts too -- it
    # used to be invisible here, so a logged-in user read as "unset".
    from ledgerhound.identity import identity_info
    who = identity_info()
    logged_in = not who["source"].startswith("OS login")
    row("user", logged_in,
        (f"{who['user']}" + (f" <{who['email']}>" if who.get("email") else "")
         + f"  -- from {who['source']}") if logged_in
        else f"not logged in yet (using OS name '{who['user']}') -- "
             "`ledgerhound shell` asks for your work email once")

    # Where everything is written (ledgerhound.paths).
    try:
        from ledgerhound import paths as _p
        lay = _p.layout()
        row("results folder", True,
            f"{lay['results']}  (run_outputs, data_cache, library)")
    except Exception as e:  # noqa: BLE001
        row("results folder", False, f"{type(e).__name__}: {e}"[:160])

    # Secrets present (presence only; never print values). Checks
    # main_config.yaml first, the named env var as fallback -- same
    # precedence as everywhere else (ledgerhound.config).
    # L-09: a key saved by the shell's ask-once prompt lives in
    # ~/.ledgerhound/credentials.json -- it counts as set.
    from ledgerhound.llm.credentials import load_key
    for method, env_name in (("gateway", "LEDGERHOUND_LLM_GATEWAY_API_KEY"),
                             ("api", "LEDGERHOUND_LLM_API_API_KEY")):
        if get_str("llm", method, "api_key", env=env_name):
            src = "main_config.yaml / " + env_name
        elif load_key("llm", method):
            src = "saved by the shell prompt"
        else:
            src = ""
        row(f"LLM gateway key ({method})", bool(src),
            f"set ({src})" if src else "unset -- the shell asks once when needed")

    # The team object store (main_config.yaml `minio:` + /minio).
    try:
        from ledgerhound.io.minio import shared_settings
        from ledgerhound.shell.minio_cmd import check_store
        ms = shared_settings()
        row("object store endpoint (minio:)", bool(ms["endpoint"]),
            f"{ms['endpoint']}  bucket {ms['bucket']}" if ms["endpoint"]
            else "unset -- minio.endpoint in main_config.yaml")
        row("object store access key", bool(ms["access_key"]),
            f"set ({ms['access_source']})" if ms["access_key"]
            else "unset -- minio.access_key or /minio --access-key")
        row("object store secret key", bool(ms["secret_key"]),
            f"set ({ms['secret_source']})" if ms["secret_key"]
            else "unset -- run /minio --key in the shell")
        if ms["endpoint"] and ms["access_key"] and ms["secret_key"]:
            ok, code, message, fix = check_store(ms)
            row("object store connection", ok,
                message if ok else f"[{code}] {message} {fix}".strip())
    except Exception as e:  # noqa: BLE001 -- a diagnostic must not crash
        row("object store", False, f"{type(e).__name__}: {e}"[:160])

    # Tokenizer for the configured model (exact token counting).
    try:
        from ledgerhound.llm.client import _load_master_config, normalise_method
        from ledgerhound.llm.tokenizer import resolve_tokenizer
        llm_sect = _load_master_config().get("llm") or {}
        method = normalise_method(str(llm_sect.get("method") or "api"))
        model_id = str(((llm_sect.get(method) or {}).get("model")) or "")
        if model_id:
            _c, ts = resolve_tokenizer(model_id)
            row(f"tokenizer ({model_id})", ts.ok,
                f"from {ts.source}" if ts.ok else f"[{ts.code}] {ts.message} Fix: {ts.fix}")
        else:
            row("tokenizer", None, "no llm model set in main_config.yaml")
    except Exception as e:  # noqa: BLE001
        row("tokenizer", False, f"{type(e).__name__}: {e}"[:160])

    # Legacy datamem store (configs/minio.yaml, one bucket per stream)
    if minio_config and Path(minio_config).exists():
        try:
            from ledgerhound.io import MinIOStore
            store = MinIOStore.from_yaml(minio_config)
            if not store.config.endpoint:
                row("datamem store (configs/minio.yaml)", None, "endpoint not filled in config")
            else:
                h = store.health_check()
                if h["ok"]:
                    missing = [s for s, b in h["buckets"].items() if not b["exists"]]
                    detail = (f"{store.config.endpoint} -- all buckets present"
                              if not missing
                              else f"{store.config.endpoint} -- missing buckets: "
                                   f"{', '.join(missing)}")
                    row("datamem store (configs/minio.yaml)", not missing if missing else True, detail)
                else:
                    row("datamem store (configs/minio.yaml)", False, h["error"] or "unreachable")
        except Exception as e:
            row("datamem store (configs/minio.yaml)", False, str(e)[:120])
    else:
        row("datamem store (configs/minio.yaml)", None, f"no config at {minio_config}")

    console.print(tbl)


@app.command()
def upgrade(
    version_spec: Optional[str] = typer.Option(
        None, "--version", help="Pin a version, e.g. 0.2.0"),
    index_url: Optional[str] = typer.Option(
        None, "--index-url",
        help="Wheel index (defaults to $LEDGERHOUND_INDEX_URL)"),
):
    """Upgrade ledgerhound from the MinIO-hosted wheel index."""
    import subprocess
    import sys

    from ledgerhound.config import get_str
    idx = index_url or get_str("integrations", "index_url", env="LEDGERHOUND_INDEX_URL") or None
    target = f"ledgerhound=={version_spec}" if version_spec else "ledgerhound"
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", target]
    if idx:
        cmd += ["--index-url", idx]
    else:
        console.print("[yellow]No --index-url / $LEDGERHOUND_INDEX_URL set; "
                      "using the default PyPI index.[/yellow]")

    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    rc = subprocess.call(cmd)
    if rc == 0:
        console.print(f"[green]OK upgraded -- now v{_installed_version()}[/green]")
    else:
        console.print(f"[red]upgrade failed (pip exit {rc})[/red]")
        raise typer.Exit(rc)


# =========================================================================== #
#  ledgerhound auto  -- self-healing agentic execution loop                      #
# =========================================================================== #

auto_app = typer.Typer(name="auto", help="Self-healing autonomous analysis loop.", no_args_is_help=True)
app.add_typer(auto_app, name="auto")


@auto_app.command("run")
def auto_run(
    goal: str = typer.Argument(..., help="Natural-language goal, e.g. 'analyse fraud drift Jan->Mar'"),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Data file (parquet/csv)"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config", help="LLM YAML config"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM base URL"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    max_steps: int = typer.Option(20, "--max-steps"),
    max_retries: int = typer.Option(3, "--max-retries", help="Fix attempts per failing step"),
    timeout: int = typer.Option(120, "--timeout", help="Seconds per code execution"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Run a self-healing agentic analysis.

    Example:
        ledgerhound auto run "analyse fraud drift and produce a 5-page exec report" \\
          --data data/fraud.parquet --llm-config configs/kimi.yaml
    """
    if not llm_config and not model:
        console.print("[red]Provide --llm-config or --model[/red]")
        raise typer.Exit(1)
    if llm_config:
        _check_file(llm_config, "LLM config")
    if data:
        _check_file(data, "Data file")

    from ledgerhound.auto.runner import AutoRunner, RunConfig
    cfg = RunConfig(
        goal=goal,
        llm_config=llm_config,
        model=model,
        api_key=api_key,
        data=data,
        workspace=workspace,
        max_steps=max_steps,
        max_retries=max_retries,
        timeout=timeout,
        verbose=not quiet,
    )
    runner = AutoRunner(cfg)
    run_dir = runner.run()
    console.print(f"\n[bold green]OK Run complete:[/bold green] {run_dir}")


@auto_app.command("review-patches")
def auto_review_patches(
    diff: Optional[str] = typer.Option(None, "--diff", help="Show unified diff for a specific patch ID"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """List pending patch candidates or show a specific diff."""
    from ledgerhound.auto.patch import PatchStore
    store = PatchStore(_ws_path(workspace) / "patch_candidates")
    candidates = store.list()

    if not candidates:
        console.print("[dim]No patch candidates found.[/dim]")
        return

    if diff:
        c = store.get(diff)
        if not c:
            console.print(f"[red]Patch {diff!r} not found.[/red]")
            raise typer.Exit(1)
        console.print(Panel(c.diff_text(), title=f"Diff -- {c.id}", border_style="yellow"))
        console.print(f"\n[bold]Reason:[/bold] {c.reason}")
        console.print(f"[bold]Status:[/bold] {c.status}")
        return

    from rich.table import Table
    t = Table(title="Patch Candidates", border_style="yellow")
    t.add_column("ID", style="cyan", no_wrap=True)
    t.add_column("Target file")
    t.add_column("Status")
    t.add_column("Reason")
    for c in candidates:
        t.add_row(c.id, c.target_file, c.status, c.reason[:60])
    console.print(t)
    console.print("\n[dim]Apply:  ledgerhound auto apply-patch <id>[/dim]")
    console.print("[dim]Reject: ledgerhound auto reject-patch <id>[/dim]")
    console.print("[dim]Diff:   ledgerhound auto review-patches --diff <id>[/dim]")


@auto_app.command("apply-patch")
def auto_apply_patch(
    patch_id: str = typer.Argument(...),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Apply a patch candidate to its target source file."""
    from ledgerhound.auto.patch import PatchStore
    store = PatchStore(_ws_path(workspace) / "patch_candidates")
    c = store.get(patch_id)
    if not c:
        console.print(f"[red]Patch {patch_id!r} not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel(c.diff_text(), title=f"Diff -- {c.id}", border_style="yellow"))
    console.print(f"[bold]Target:[/bold] {c.target_file}")
    console.print(f"[bold]Reason:[/bold] {c.reason}")

    if not yes:
        confirm = typer.confirm("Apply this patch?", default=False)
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    store.apply(patch_id)
    console.print(f"[green]OK Patch applied to {c.target_file}[/green]")


@auto_app.command("reject-patch")
def auto_reject_patch(
    patch_id: str = typer.Argument(...),
    reason: str = typer.Option("", "--reason", "-r"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Reject and discard a patch candidate."""
    from ledgerhound.auto.patch import PatchStore
    store = PatchStore(_ws_path(workspace) / "patch_candidates")
    ok = store.reject(patch_id, reason)
    if ok:
        console.print(f"[green]OK Patch {patch_id} rejected.[/green]")
    else:
        console.print(f"[red]Patch {patch_id!r} not found.[/red]")
        raise typer.Exit(1)


@auto_app.command("list-runs")
def auto_list_runs(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    n: int = typer.Option(10, "--n", help="Number of recent runs to show"),
):
    """List recent auto runs and their status."""
    runs_dir = _ws_path(workspace) / "auto_runs"
    if not runs_dir.exists():
        console.print("[dim]No runs found.[/dim]")
        return

    runs = sorted(runs_dir.iterdir(), reverse=True)[:n]
    from rich.table import Table
    t = Table(title="Recent Auto Runs", border_style="cyan")
    t.add_column("Run ID", style="cyan", no_wrap=True)
    t.add_column("Steps")
    t.add_column("Succeeded")
    t.add_column("Report")
    for run in runs:
        if not run.is_dir():
            continue
        trace_file = run / "trace.jsonl"
        steps = succeeded = 0
        if trace_file.exists():
            entries = [json.loads(l) for l in trace_file.read_text().splitlines() if l.strip()]
            step_entries = [e for e in entries if e.get("event") != "plan"]
            steps = len(step_entries)
            succeeded = sum(1 for e in step_entries if e.get("success"))
        report = "OK" if (run / "report.html").exists() else "--"
        t.add_row(run.name, str(steps), str(succeeded), report)
    console.print(t)


# -- Golden evaluation -------------------------------------------------------- #

eval_app = typer.Typer(name="eval",
                       help="Golden-dataset evaluation (groundedness via recomputation).",
                       no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("validate")
def eval_validate(
    golden: str = typer.Argument(..., help="Golden YAML/CSV file or directory"),
    data_root: str = typer.Option(".", "--data-root", help="Root dir for dataset paths"),
):
    """Validate golden tasks: schema + every fact computation must execute.

    Run this after your team adds tasks, BEFORE a full eval.
    """
    from ledgerhound.eval import load_golden
    from ledgerhound.eval.runner import _load_df
    tasks = load_golden(golden)
    if not tasks:
        console.print("[red]No golden tasks found.[/red]")
        raise typer.Exit(1)

    from rich.table import Table
    t = Table(title="Golden Validation", border_style="cyan")
    t.add_column("Task", style="cyan"); t.add_column("Facts")
    t.add_column("Computations"); t.add_column("Status")
    bad = 0
    for task in tasks:
        df = _load_df(task, data_root)
        ok_facts, errs = 0, []
        for fact in task.facts:
            if df is None:
                errs.append(f"{fact.id}: no data to recompute")
                continue
            try:
                ns = {"df": df}
                import pandas as pd, numpy as np  # noqa
                ns.update({"pd": pd, "np": np})
                exec(fact.computation, ns)
                if "result" not in ns:
                    errs.append(f"{fact.id}: no 'result' set")
                else:
                    ok_facts += 1
            except Exception as ex:
                errs.append(f"{fact.id}: {ex}")
        status = "[green]OK[/green]" if not errs else f"[red]{len(errs)} error(s)[/red]"
        if errs:
            bad += 1
        t.add_row(task.id, str(len(task.facts)),
                  f"{ok_facts}/{len(task.facts)}", status)
        for e in errs:
            console.print(f"  [red]- {task.id}: {e}[/red]")
    console.print(t)
    if bad:
        console.print(f"[red]{bad} task(s) have problems -- fix before running eval.[/red]")
        raise typer.Exit(1)
    console.print("[green]All golden tasks valid.[/green]")


@eval_app.command("score-run")
def eval_score_run(
    golden: str = typer.Argument(..., help="Golden YAML for the single task"),
    run_dir: str = typer.Argument(..., help="Completed run dir (contains trace.jsonl)"),
    data_root: str = typer.Option(".", "--data-root"),
):
    """Score an already-completed run against a golden task (no re-running)."""
    from ledgerhound.eval import load_golden, score_run
    from ledgerhound.eval.runner import _load_df
    tasks = load_golden(golden)
    if not tasks:
        console.print("[red]No golden task found.[/red]"); raise typer.Exit(1)
    task = tasks[0]
    trace_file = Path(run_dir) / "trace.jsonl"
    if not trace_file.exists():
        console.print(f"[red]No trace.jsonl in {run_dir}[/red]"); raise typer.Exit(1)
    trace = [json.loads(l) for l in trace_file.read_text(encoding="utf-8").splitlines()
             if l.strip() and json.loads(l).get("event") != "plan"]
    report_html = Path(run_dir) / "report.html"
    report_text = report_html.read_text(encoding="utf-8") if report_html.exists() else ""
    df = _load_df(task, data_root)
    sc = score_run(task, trace, df=df, report_text=report_text, plan=[])
    console.print(Panel(sc.summary_line(), title=f"Scorecard -- {task.id}",
                        border_style="green"))
    console.print_json(json.dumps(sc.to_dict()))


@eval_app.command("trend")
def eval_trend(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w",
                                  help="Workspace to scan for scorecard.json"),
):
    """Trend per-run scorecards over time (is Ledgerhound improving?)."""
    from ledgerhound.eval.trend import build_trend, collect_scorecards, render_console
    cards = collect_scorecards(workspace)
    if not cards:
        console.print(f"[yellow]No scorecards found under {workspace}. "
                      f"Run `ledgerhound auto` first.[/yellow]")
        raise typer.Exit(0)
    trend = build_trend(cards)
    console.print(Panel(render_console(trend), title="Scorecard Trend",
                        border_style="cyan"))


@eval_app.command("run")
def eval_run(
    golden: str = typer.Argument(..., help="Golden YAML/CSV file or directory"),
    data_root: str = typer.Option(".", "--data-root", help="Root dir for dataset paths"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config", help="LLM YAML config"),
    out: Optional[str] = typer.Option(None, "--out", help="Write dashboard JSON here"),
):
    """Run Ledgerhound on every golden task and emit the evaluation dashboard."""
    if llm_config:
        _check_file(llm_config, "LLM config")
    from ledgerhound.eval.runner import evaluate_golden
    evaluate_golden(golden, data_root=data_root, llm_config=llm_config, out_path=out)


# -- Long-term Data Memory inspection ---------------------------------------- #

skills_app = typer.Typer(
    name="skills",
    help="Self-evolving procedural skills (distil / recall / reinforce).",
    no_args_is_help=True)
app.add_typer(skills_app, name="skills")


def _skill_memory(workspace: str):
    from ledgerhound.memory.skill_memory import SkillMemory
    return SkillMemory(_ws_path(workspace) / "data_memory" / "skills.jsonl")


@skills_app.command("list")
def skills_list(workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w")):
    """List learned skills, ranked by effective confidence."""
    from rich.table import Table
    sm = _skill_memory(workspace)
    skills = sorted(sm.all(), key=lambda s: -s.effective_confidence())
    if not skills:
        console.print("[yellow]No skills learned yet.[/yellow]"); return
    t = Table(title="Skills", border_style="cyan")
    for c in ("Name", "Conf", "Uses", "OK", "Dataset", "Trigger"):
        t.add_column(c)
    for s in skills:
        t.add_row(s.name, f"{s.effective_confidence():.2f}", str(s.uses),
                  str(s.successes), s.dataset, s.trigger[:40])
    console.print(t)
    console.print(f"[dim]{sm.stats()}[/dim]")


@skills_app.command("recall")
def skills_recall(
    query: str = typer.Argument(..., help="Task/objective to find skills for"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Recall skills applicable to a task."""
    block = _skill_memory(workspace).block(query, dataset=dataset)
    console.print(block or "[yellow]No applicable skills.[/yellow]")


@skills_app.command("reinforce")
def skills_reinforce(
    name: str = typer.Argument(...),
    success: bool = typer.Option(..., "--success/--fail",
                                 help="Did the skill work this time?"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Record an application outcome (self-evolution)."""
    s = _skill_memory(workspace).reinforce(name, success)
    if s is None:
        console.print(f"[red]No skill '{name}'.[/red]"); raise typer.Exit(1)
    console.print(f"[green]{name}[/green]: uses={s.uses} ok={s.successes} "
                  f"conf={s.effective_confidence():.2f}")


@skills_app.command("distil")
def skills_distil(
    dataset: str = typer.Argument(..., help="Dataset whose lessons to distil"),
    min_support: int = typer.Option(2, "--min-support"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Distil recurring lessons into reusable skills."""
    from ledgerhound.memory.lesson_memory import LessonMemory
    lm = LessonMemory(_ws_path(workspace) / "data_memory" / "lessons.jsonl")
    lessons = lm.lessons_for(dataset)
    learned = _skill_memory(workspace).distil(lessons, min_support=min_support)
    console.print(f"[green]Distilled {len(learned)} skill(s)[/green] from "
                  f"{len(lessons)} lesson(s).")
    for s in learned:
        console.print(f"  - {s.name} ({len(s.steps)} steps)")


datamem_app = typer.Typer(name="datamem",
                          help="Inspect long-term Data Memory (facts/baselines/drift/lessons).",
                          no_args_is_help=True)
app.add_typer(datamem_app, name="datamem")


def _mem_root(workspace: str) -> Path:
    return _ws_path(workspace) / "data_memory"


@datamem_app.command("facts")
def datamem_facts(
    dataset: str = typer.Argument(..., help="Dataset key (parquet stem)"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show learned numeric baselines for a dataset."""
    from ledgerhound.memory.fact_memory import FactMemory
    fm = FactMemory(_mem_root(workspace) / "facts.json")
    recs = fm.facts_for(dataset)
    if not recs:
        console.print(f"[yellow]No facts for '{dataset}' yet.[/yellow]"); return
    from rich.table import Table
    t = Table(title=f"Data Memory -- {dataset}", border_style="cyan")
    for c in ("Fact", "Baseline", "Stdev", "n", "Conf", "Status"):
        t.add_column(c)
    for r in sorted(recs, key=lambda x: -x.n):
        t.add_row(r.name, f"{r.baseline:.4g}", f"{r.stdev:.4g}", str(r.n),
                  f"{r.confidence:.0%}", r.status)
    console.print(t)


@datamem_app.command("drift")
def datamem_drift(
    dataset: str = typer.Argument(...),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show facts that have drifted from their historical baseline."""
    from ledgerhound.memory.fact_memory import FactMemory
    fm = FactMemory(_mem_root(workspace) / "facts.json")
    drift = fm.anomalies(dataset)
    if not drift:
        console.print(f"[green]No drift for '{dataset}'.[/green]"); return
    for r in drift:
        console.print(f"  [yellow]{r.name}[/yellow]: baseline {r.baseline:.4g} "
                      f"+/- {r.stdev:.4g} over {r.n} runs (DRIFTING)")


@datamem_app.command("lessons")
def datamem_lessons(
    dataset: str = typer.Argument(...),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show procedural lessons learned for a dataset."""
    from ledgerhound.memory.lesson_memory import LessonMemory
    lm = LessonMemory(_mem_root(workspace) / "lessons.jsonl")
    lessons = lm.lessons_for(dataset)
    if not lessons:
        console.print(f"[yellow]No lessons for '{dataset}' yet.[/yellow]"); return
    for l in lessons:
        console.print(f"  - {l.text}  [dim](seen {len(set(l.run_ids))}x, "
                      f"conf {l.confidence:.0%}, {l.source})[/dim]")


@datamem_app.command("recall")
def datamem_recall(
    query: str = typer.Argument(..., help="What to recall from past findings"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    k: int = typer.Option(8, "--k"),
):
    """Semantic recall of past findings (institutional memory)."""
    from ledgerhound.memory.longitudinal import LongitudinalMemory, MemorySource
    vm = LongitudinalMemory(_mem_root(workspace) / "vectors")
    hits = vm.recall(query, namespace="text", k=k, source=MemorySource.FINDING)
    if not hits:
        console.print("[yellow]No matching past findings.[/yellow]"); return
    for h in hits:
        when = (h.record.as_of or "")[:10]
        console.print(f"  [{h.score:.2f}] ({when}) {h.record.text}")


@datamem_app.command("reverify")
def datamem_reverify(
    dataset: str = typer.Argument(...),
    data: str = typer.Option(..., "--data", "-d", help="Current data file (parquet/csv)"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Re-run stored fact computations against current data; demote stale facts."""
    _check_file(data, "Data file")
    import pandas as pd
    p = Path(data)
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
    from ledgerhound.memory.fact_memory import FactMemory
    fm = FactMemory(_mem_root(workspace) / "facts.json")
    results = fm.reverify(dataset, df)
    fm.save()
    if not results:
        console.print(f"[yellow]No verifiable facts for '{dataset}'.[/yellow]"); return
    ok = sum(1 for r in results if r.get("reproduced") is True)
    bad = [r for r in results if r.get("reproduced") is False]
    console.print(f"Re-verified {len(results)} facts: [green]{ok} reproduced[/green]"
                  f"{', [red]'+str(len(bad))+' contradicted[/red]' if bad else ''}")
    for r in bad:
        console.print(f"  [red]{r['name']}: was {r.get('baseline'):.4g}, "
                      f"now {r.get('recomputed'):.4g}[/red]")


@datamem_app.command("stats")
def datamem_stats(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Overall Data Memory stats."""
    from ledgerhound.memory.fact_memory import FactMemory
    from ledgerhound.memory.lesson_memory import LessonMemory
    fm = FactMemory(_mem_root(workspace) / "facts.json")
    lm = LessonMemory(_mem_root(workspace) / "lessons.jsonl")
    console.print(Panel(
        f"Facts:   {fm.stats()}\nLessons: {lm.stats()}",
        title="Data Memory", border_style="cyan"))


def _minio_config_path(config: str | None) -> str:
    from ledgerhound.config import get_str
    path = config or get_str("integrations", "minio_config", env="LEDGERHOUND_MINIO_CONFIG")
    if not path or not Path(path).exists():
        raise typer.BadParameter(
            "No MinIO config. Pass --config or set integrations.minio_config "
            "in main_config.yaml to a configs/minio.yaml path."
        )
    return path


@datamem_app.command("push")
def datamem_push(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    config: str = typer.Option(None, "--config", "-c", help="configs/minio.yaml"),
    owner: str = typer.Option(None, "--owner", help="Override the memory owner"),
):
    """Mirror this owner's private tier + the shared tier to MinIO (durability)."""
    from ledgerhound.io import MinIOStore
    from ledgerhound.memory.librarian import Librarian
    store = MinIOStore.from_yaml(_minio_config_path(config))
    lib = Librarian(_mem_root(workspace), owner=owner)
    keys = lib.push_to_minio(store)
    console.print(f"[green]Pushed {len(keys)} object(s)[/green] for owner "
                  f"'{lib.owner}' + shared tier.")


@datamem_app.command("pull")
def datamem_pull(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
    config: str = typer.Option(None, "--config", "-c", help="configs/minio.yaml"),
    owner: str = typer.Option(None, "--owner", help="Override the memory owner"),
):
    """Restore this owner's private tier + the shared tier from MinIO."""
    from ledgerhound.io import MinIOStore
    from ledgerhound.memory.librarian import Librarian
    store = MinIOStore.from_yaml(_minio_config_path(config))
    lib = Librarian(_mem_root(workspace), owner=owner)
    paths = lib.pull_from_minio(store)
    console.print(f"[green]Restored {len(paths)} file(s)[/green] for owner "
                  f"'{lib.owner}' + shared tier into {_mem_root(workspace)}.")


agent_app = typer.Typer(
    name="agent",
    help="Specialist agents (agent_def/v1): list, show, and run them.",
    no_args_is_help=True,
)
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list():
    """List available agents (built-in + user ~/.ledgerhound/agents)."""
    from rich.table import Table

    from ledgerhound.agents.registry import AgentRegistry
    reg = AgentRegistry().load()
    t = Table(title="Agents", border_style="cyan")
    for c in ("Name", "Version", "Contract", "Scope", "Source", "Tools"):
        t.add_column(c)
    for spec in sorted(reg.list(), key=lambda s: s.name):
        t.add_row(spec.name, spec.version, spec.output_contract,
                  spec.memory_scope, reg.source_of(spec.name) or "?",
                  str(len(spec.tools)))
    console.print(t)


@agent_app.command("show")
def agent_show(name: str = typer.Argument(..., help="Agent name")):
    """Show one agent's full definition."""
    from ledgerhound.agents.registry import AgentRegistry
    spec = AgentRegistry().load().get(name)
    if spec is None:
        console.print(f"[red]Unknown agent '{name}'.[/red]")
        raise typer.Exit(1)
    console.print(Panel(
        f"[bold]{spec.name}[/bold] v{spec.version}  "
        f"[dim]({spec.output_contract}, {spec.memory_scope})[/dim]\n\n"
        f"{spec.description}\n\n[bold]Role:[/bold]\n{spec.role_prompt}\n\n"
        f"[bold]Tools:[/bold] {', '.join(spec.tools)}",
        border_style="cyan"))


@agent_app.command("run")
def agent_run(
    name: str = typer.Argument(..., help="Agent name (see `agent list`)"),
    data: str = typer.Option(..., "--data", "-d", help="Data file OR folder"),
    ask: str = typer.Option(None, "--ask", help="Question/objective for the agent"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Run a named specialist agent over a dataset (spec drives the agent)."""
    from ledgerhound.agents.factory import load_agent
    data = _resolve_data_path(data, f"{name} data")
    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)
    from ledgerhound.core import Ledgerhound
    eng = Ledgerhound(data=data)
    try:
        agent = load_agent(name, llm=llm, backend=eng.backend,
                           knowledge=eng.knowledge, findings_log=log)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    agent.set_digest(eng.profile())
    question = ask or (f"Perform your role ({name}) on this dataset and record "
                       "your findings.")
    console.print(Panel(f"[bold]Agent: {name}[/bold]\n{question}",
                        border_style="cyan"))
    agent.ask(question, interactive=True)


@agent_app.command("create")
def agent_create(
    name: str = typer.Argument(..., help="New agent name (slug)"),
    role: str = typer.Option(..., "--role", help="The agent's role_prompt"),
    tools: str = typer.Option("execute_python,record_finding,list_findings",
                              "--tools", help="Comma-separated tool whitelist"),
    contract: str = typer.Option("findings/v1", "--contract"),
    scope: str = typer.Option("private", "--scope", help="private | shared"),
    description: str = typer.Option("", "--description"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Create a user-defined agent (saved to ~/.ledgerhound/agents)."""
    from ledgerhound.agents.registry import AgentRegistry
    from ledgerhound.agents.spec import AgentSpec, AgentValidationError
    spec = AgentSpec(
        name=name, role_prompt=role, description=description,
        tools=[t.strip() for t in tools.split(",") if t.strip()],
        output_contract=contract, memory_scope=scope)
    try:
        reg = AgentRegistry().load()
        dest = reg.create(spec, overwrite=overwrite)
    except (AgentValidationError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Created agent '{name}'[/green] -> {dest}")


@agent_app.command("delete")
def agent_delete(name: str = typer.Argument(..., help="User agent to delete")):
    """Delete a user-defined agent (built-ins cannot be deleted)."""
    from ledgerhound.agents.registry import AgentRegistry
    try:
        AgentRegistry().load().delete(name)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Deleted user agent '{name}'.[/green]")


@agent_app.command("push")
def agent_push(
    config: str = typer.Option(None, "--config", "-c", help="configs/minio.yaml"),
    owner: str = typer.Option(None, "--owner", help="Override owner namespace"),
    name: str = typer.Option(None, "--name", help="Push only this agent"),
):
    """Publish your user agents to the shared MinIO library."""
    from ledgerhound.agents.library import AgentLibrary
    from ledgerhound.identity import current_user
    from ledgerhound.io import MinIOStore
    store = MinIOStore.from_yaml(_minio_config_path(config))
    lib = AgentLibrary(store)
    keys = lib.push(owner or current_user(),
                    names=[name] if name else None)
    console.print(f"[green]Pushed {len(keys)} agent(s)[/green] to the library.")


@agent_app.command("pull")
def agent_pull(
    config: str = typer.Option(None, "--config", "-c", help="configs/minio.yaml"),
    owner: str = typer.Option(None, "--owner", help="Pull only this owner's agents"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Fetch shared agents from the MinIO library into your user tier."""
    from ledgerhound.agents.library import AgentLibrary
    from ledgerhound.io import MinIOStore
    store = MinIOStore.from_yaml(_minio_config_path(config))
    lib = AgentLibrary(store)
    paths = lib.pull(owner, overwrite=overwrite)
    console.print(f"[green]Pulled {len(paths)} agent(s)[/green] into "
                  f"{lib.user_dir}.")


# -- Orchestrator (rank 12): playbooks-as-code ------------------------------

study_app = typer.Typer(
    name="study",
    help="Run a multi-agent study (playbook-as-code DAG) -> one report.",
    no_args_is_help=True,
)
app.add_typer(study_app, name="study")


@study_app.command("plan")
def study_plan(
    playbook: str = typer.Option(..., "--playbook", "-p", help="study/v1 YAML"),
):
    """Validate a playbook and show its execution DAG (dependency levels)."""
    from ledgerhound.agents.orchestrator import StudyPlan
    _check_file(playbook, "Playbook")
    try:
        plan = StudyPlan.from_yaml(playbook).validate()
    except ValueError as exc:
        console.print(f"[red]Invalid playbook: {exc}[/red]"); raise typer.Exit(1)
    console.print(Panel(f"[bold]{plan.name}[/bold] ({len(plan.tasks)} tasks)",
                        border_style="cyan"))
    for i, level in enumerate(plan.levels(), 1):
        tasks = ", ".join(f"{t.id}[{t.agent}]" for t in level)
        console.print(f"  level {i} (parallel-safe): {tasks}")


@study_app.command("run")
def study_run(
    playbook: str = typer.Option(..., "--playbook", "-p", help="study/v1 YAML"),
    data: str = typer.Option(..., "--data", "-d", help="Default data file/folder"),
    report: str = typer.Option("study_report.md", "--report", "-r"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Execute a playbook: run each agent task in DAG order -> study report."""
    from ledgerhound.agents.orchestrator import Orchestrator, StudyPlan, default_agent_factory
    _check_file(playbook, "Playbook")
    data = _resolve_data_path(data, "Study data")
    try:
        plan = StudyPlan.from_yaml(playbook).validate()
    except ValueError as exc:
        console.print(f"[red]Invalid playbook: {exc}[/red]"); raise typer.Exit(1)
    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)
    factory = default_agent_factory(llm, findings_log=log, default_data=data)
    console.print(Panel(f"[bold]Study: {plan.name}[/bold] "
                        f"({len(plan.tasks)} tasks)", border_style="cyan"))
    result = Orchestrator(factory, interactive=True).run(plan)
    result.save(report)
    console.print(f"\n[bold green]Study complete.[/bold green] "
                  f"{result.total_findings} findings, {result.failed} failed, "
                  f"{result.elapsed_ms/1000:.1f}s -> {report}")


# -- Feature engineering (rank 9): evaluate + recipe library ----------------

feature_app = typer.Typer(
    name="feature",
    help="Evaluate candidate features (IV/KS/AUC) and manage the recipe library.",
    no_args_is_help=True,
)
app.add_typer(feature_app, name="feature")


def _recipe_library_path(workspace: str) -> Path:
    return _ws_path(workspace) / "data_memory" / "recipes.jsonl"


@feature_app.command("eval")
def feature_eval_cmd(
    data: str = typer.Option(..., "--data", "-d", help="Data file OR folder"),
    target: str = typer.Option("is_fraud", "--target", help="Binary label column"),
    features: Optional[str] = typer.Option(
        None, "--features", help="Comma-separated; omit to auto-detect numeric"),
    bins: int = typer.Option(10, "--bins", help="IV WOE bin count"),
    corr: float = typer.Option(0.9, "--corr", help="Correlation-pruning threshold"),
    deposit: bool = typer.Option(
        False, "--deposit", help="Deposit medium+ features as recipes"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Evaluate candidate features against the fraud label; rank + prune."""
    from rich.table import Table

    from ledgerhound.agents.fraud_nonfraud import auto_numeric_features
    from ledgerhound.analysis.feature_eval import evaluate_features
    data = _resolve_data_path(data, "Feature data")
    df = _read_data(data)
    if target not in df.columns:
        console.print(f"[red]Target '{target}' not in data.[/red]")
        raise typer.Exit(1)
    if features:
        feats = [f.strip() for f in features.split(",") if f.strip()]
    else:
        feats = [c for c in auto_numeric_features(df, df) if c != target]
        console.print(f"[green]Auto-detected {len(feats)} feature(s).[/green]")
    report = evaluate_features(df, feats, target, n_bins=bins, corr_threshold=corr)

    t = Table(title=f"Feature evaluation vs '{target}'", border_style="cyan")
    for c in ("Feature", "IV", "Band", "KS", "AUC", "Strength", "Miss%"):
        t.add_column(c)
    for e in report.evaluations:
        t.add_row(e.feature, f"{e.iv:.4f}", e.iv_band, f"{e.ks:.3f}",
                  f"{e.auc:.3f}", f"{e.strength:.3f}", f"{e.missing_rate:.1%}")
    console.print(t)
    if report.pruned:
        console.print(f"[yellow]Pruned (correlated): {', '.join(report.pruned)}[/yellow]")
    if report.skipped:
        console.print(f"[dim]Skipped: {', '.join(report.skipped)}[/dim]")

    if deposit:
        from ledgerhound import identity
        from ledgerhound.analysis.feature_recipe import FeatureRecipe, RecipeLibrary
        lib = RecipeLibrary(_recipe_library_path(workspace))
        ds = Path(data).stem
        kept = 0
        for e in report.evaluations:
            if e.iv_band in ("useless", "weak", "suspicious"):
                continue
            lib.deposit(FeatureRecipe(
                name=e.feature, kind="raw", inputs=[e.feature],
                iv=e.iv, ks=e.ks, auc=e.auc, iv_band=e.iv_band,
                dataset=ds, owner=identity.current_user()))
            kept += 1
        console.print(f"[green]Deposited {kept} recipe(s)[/green] -> "
                      f"{_recipe_library_path(workspace)}")


@feature_app.command("recipes")
def feature_recipes(
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """List stored feature recipes, strongest first."""
    from rich.table import Table

    from ledgerhound.analysis.feature_recipe import RecipeLibrary
    lib = RecipeLibrary(_recipe_library_path(workspace))
    best = lib.best(dataset=dataset, limit=50, min_band="useless")
    if not best:
        console.print("[yellow]No recipes deposited yet.[/yellow]")
        return
    t = Table(title="Feature recipes", border_style="cyan")
    for c in ("Name", "Kind", "IV", "Band", "Dataset", "Owner", "Reviewed"):
        t.add_column(c)
    for r in best:
        t.add_row(r.name, r.kind, f"{r.iv:.4f}", r.iv_band, r.dataset,
                  r.owner, "yes" if r.leakage_reviewed else "NO")
    console.print(t)
    console.print(f"[dim]library stats: {lib.stats()}[/dim]")


# -- Productization (rank 8): runs / stats / whoami -------------------------

runs_app = typer.Typer(
    name="runs",
    help="Inspect, export, and share past analysis runs (sessions).",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """List past runs, newest first."""
    from rich.table import Table

    from ledgerhound.session.store import SessionStore
    sessions = SessionStore(workspace).list()
    if not sessions:
        console.print(f"[yellow]No runs under {workspace}/sessions.[/yellow]")
        return
    t = Table(title="Runs", border_style="cyan")
    for c in ("ID", "Name", "Created", "Status", "Goal", "Datasets", "Steps"):
        t.add_column(c)
    for s in sessions:
        t.add_row(s.id, s.name, (s.created_at or "")[:19], s.status,
                  str(s.goal), str(len(s.datasets)),
                  str(s.last_completed_step))
    console.print(t)


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(..., help="Run/session id"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Show one run in detail."""
    from ledgerhound.session.store import SessionStore
    s = SessionStore(workspace).get(run_id)
    if s is None:
        console.print(f"[red]No run '{run_id}'.[/red]")
        raise typer.Exit(1)
    ds = "\n".join(f"  - {d.name} ({d.rows or '?'} rows) {d.path}"
                   for d in s.datasets) or "  (none)"
    console.print(Panel(
        f"[bold]{s.name}[/bold]  [dim]{s.id}[/dim]\n"
        f"created: {s.created_at}\nstatus: {s.status}\n"
        f"goal: {s.goal} {s.goal_detail}\n"
        f"steps completed: {s.last_completed_step}\n"
        f"last run id: {s.last_run_id}\n\ndatasets:\n{ds}",
        title="Run", border_style="cyan"))


@runs_app.command("export")
def runs_export(
    run_id: str = typer.Argument(...),
    out: str = typer.Option(..., "--out", "-o", help="Output JSON path"),
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """Export a run's metadata as a portable JSON (for sharing/audit)."""
    from dataclasses import asdict

    from ledgerhound.session.store import SessionStore
    s = SessionStore(workspace).get(run_id)
    if s is None:
        console.print(f"[red]No run '{run_id}'.[/red]")
        raise typer.Exit(1)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(s), indent=2, default=str), encoding="utf-8")
    console.print(f"[green]Exported run '{run_id}'[/green] -> {p}")


@app.command("stats")
def stats(
    workspace: str = typer.Option("ledgerhound_workspace", "--workspace", "-w"),
):
    """One-glance summary of this install: runs, agents, and memory."""
    lines: list[str] = []

    # runs
    try:
        from ledgerhound.session.store import SessionStore
        sessions = SessionStore(workspace).list()
        by_status: dict[str, int] = {}
        for s in sessions:
            by_status[s.status] = by_status.get(s.status, 0) + 1
        status_str = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        lines.append(f"Runs:    {len(sessions)}" + (f"  ({status_str})" if status_str else ""))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Runs:    (unavailable: {exc})")

    # agents
    try:
        from ledgerhound.agents.registry import AgentRegistry
        reg = AgentRegistry().load()
        builtins = [n for n in reg.names() if reg.source_of(n) == "builtin"]
        users = [n for n in reg.names() if reg.source_of(n) == "user"]
        lines.append(f"Agents:  {len(reg.names())}  "
                     f"({len(builtins)} built-in, {len(users)} user)")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Agents:  (unavailable: {exc})")

    # memory moat
    try:
        from ledgerhound.memory.fact_memory import FactMemory
        from ledgerhound.memory.lesson_memory import LessonMemory
        root = _ws_path(workspace) / "data_memory"
        fm = FactMemory(root / "facts.json")
        lm = LessonMemory(root / "lessons.jsonl")
        lines.append(f"Memory:  facts={fm.stats()}  lessons={lm.stats()}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Memory:  (unavailable: {exc})")

    console.print(Panel("\n".join(lines), title="Ledgerhound stats",
                        border_style="cyan"))


@app.command("whoami")
def whoami():
    """Show the current identity (owner) used to namespace runs and memory."""
    from ledgerhound import identity
    owner = identity.current_user()
    declared = identity.is_declared()
    console.print(Panel(
        f"owner: [bold]{owner}[/bold]\n"
        f"source: {'LEDGERHOUND_USER (declared)' if declared else 'OS/anonymous fallback'}\n"
        f"private-tier prefix: {identity.owner_prefix('<stream>')}\n\n"
        "[dim]Tier 1 identity trusts LEDGERHOUND_USER. Tier 2 (IdP-verified "
        "identity) is delegated to the deployment's identity provider -- "
        "Ledgerhound never stores credentials.[/dim]",
        title="whoami", border_style="cyan"))


cycle_app = typer.Typer(
    name="cycle",
    help="Fraud x non-fraud end-to-end EDA cycle (design -> fetch -> compare).",
    no_args_is_help=True,
)
app.add_typer(cycle_app, name="cycle")


_DATA_SUFFIXES = (".csv", ".parquet", ".pq")


def _preflight_gate(files: list, label: str, *, interactive: bool,
                    expect_label: bool = False) -> None:
    """Run the cheap preflight scan and surface issues BEFORE any full load.

    Errors (unreadable/empty files) always abort. Warnings are advisory: in
    interactive mode the analyst is asked to proceed; in auto mode we print
    and continue (warnings are recoverable -- concat coerces dtypes etc.).
    """
    from ledgerhound.data import preflight, render_report
    rep = preflight(files, label, expect_label=expect_label)
    if not rep.issues:
        return
    console.print(render_report(rep))
    if rep.has_errors:
        console.print(f"[red]{label}: preflight found "
                      f"{len(rep.errors)} blocking issue(s); aborting.[/red]")
        raise typer.Exit(1)
    if rep.has_warnings and interactive:
        if not typer.confirm(f"{label}: preflight raised "
                             f"{len(rep.warnings)} warning(s). Proceed anyway?",
                             default=True):
            console.print(f"[yellow]{label}: re-select your files.[/yellow]")
            raise typer.Exit(1)


def _concat_data_files(files: list, label: str, *, interactive: bool = False,
                       expect_label: bool = False) -> str:
    """Read every file, concatenate, write a temp parquet, return its path.

    Robust to HETEROGENEOUS files: shards can differ in columns (e.g. a
    53-col fraud parquet vs a 52-col history CSV) and in dtype for the same
    column (``trndate`` datetime in parquet but a string in CSV). Without
    care, ``pd.concat`` produces mixed-type object columns that pyarrow
    cannot write to parquet (ArrowTypeError). We therefore concatenate on
    the COMMON columns and coerce any column whose dtype is inconsistent
    across files to a nullable string, so the write always succeeds.
    """
    import pandas as pd
    _preflight_gate(files, label, interactive=interactive,
                    expect_label=expect_label)
    frames = [pd.read_parquet(f) if f.suffix.lower() in (".parquet", ".pq")
              else pd.read_csv(f) for f in files]

    # Common columns only (a shard missing a column shouldn't inject NaN
    # columns across the whole concat).
    common = set(frames[0].columns)
    for fr in frames[1:]:
        common &= set(fr.columns)
    ordered = [c for c in frames[0].columns if c in common]
    dropped = {c for fr in frames for c in fr.columns} - set(ordered)
    if dropped:
        console.print(f"[yellow]{label}: files have mismatched schemas; using "
                      f"the {len(ordered)} common columns "
                      f"(dropped {len(dropped)}: {', '.join(sorted(dropped)[:6])}"
                      f"{'...' if len(dropped) > 6 else ''}).[/yellow]")

    # Coerce columns with inconsistent dtypes across files to string so the
    # concatenated column is uniform and parquet-writable.
    for col in ordered:
        dtypes = {str(fr[col].dtype) for fr in frames}
        if len(dtypes) > 1:
            for fr in frames:
                fr[col] = fr[col].astype("string")

    df = pd.concat([fr[ordered] for fr in frames], ignore_index=True)
    tmp = Path(".ledgerhound/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / f"{label.replace(' ', '_').lower()}_concat.parquet"
    try:
        df.to_parquet(dest, index=False)
    except Exception as e:  # noqa: BLE001
        # Last resort: any remaining un-writable object columns -> string.
        console.print(f"[yellow]{label}: parquet write hit {type(e).__name__}; "
                      "coercing remaining object columns to string.[/yellow]")
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("string")
        df.to_parquet(dest, index=False)
    console.print(f"[green]{label}: concatenated {len(files)} files -> "
                  f"{dest.name} ({len(df):,} rows, {len(ordered)} cols)[/green]")
    return str(dest)


def _resolve_data_path(path: str, label: str, *, allow_concat: bool = True,
                       interactive: bool = True,
                       selected_out: Optional[list] = None) -> str:
    """Resolve a file OR a folder-of-CSVs to a single data path.

    - A file path is returned as-is.
    - A folder is scanned for *.csv / *.parquet. One match -> auto-selected;
      many -> an interactive numbered menu. You can pick:
        * a single index          e.g. ``2``
        * a comma-separated list  e.g. ``1,3,4``   (those files, concatenated)
        * ``all``                 (every file, concatenated)
      none -> a clear error (for non-fraud, a hint that step 3 must run first).

    With ``interactive=False`` (auto mode / non-TTY jobs) a multi-file folder
    is auto-concatenated instead of blocking on a prompt.

    If ``selected_out`` is provided, the chosen file NAMES are appended to it
    so the caller can ask the analyst what each dataset is.
    """
    def _record(chosen: list[Path]) -> None:
        if selected_out is not None:
            selected_out.extend(f.name for f in chosen)

    p = Path(path)
    if p.is_file():
        _record([p])
        return str(p)
    if not p.exists():
        console.print(f"[red]{label}: path not found: {path}[/red]")
        raise typer.Exit(1)

    files = sorted(f for f in p.iterdir()
                   if f.is_file() and f.suffix.lower() in _DATA_SUFFIXES)
    if not files:
        console.print(f"[red]{label}: no CSV/parquet files in folder '{path}'.[/red]")
        if "non" in label.lower():
            console.print("[dim]Non-fraud data is produced by step 3 - run the "
                          "exported SQL in Trino and drop the result CSV into "
                          "this folder first.[/dim]")
        raise typer.Exit(1)
    if len(files) == 1:
        console.print(f"[green]{label}: using {files[0].name}[/green]")
        _record(files)
        return str(files[0])

    if not interactive:
        # Unattended: don't block on a menu -- concatenate all shards.
        if allow_concat:
            console.print(f"[green]{label}: {len(files)} files -> "
                          "concatenating all (auto mode)[/green]")
            _record(files)
            return _concat_data_files(files, label, interactive=False)
        console.print(f"[green]{label}: using {files[0].name} (auto mode)[/green]")
        _record(files[:1])
        return str(files[0])

    console.print(f"\n[bold]{label}: {len(files)} files in '{path}'[/bold]")
    for i, f in enumerate(files, 1):
        console.print(f"  [{i}] {f.name}")
    hint = ("a number, a comma-separated list like 1,3"
            + (", or 'all'" if allow_concat else ""))
    sel = typer.prompt(f"Which file(s) for {label}? ({hint})").strip().lower()

    if allow_concat and sel == "all":
        _record(files)
        return _concat_data_files(files, label, interactive=True)

    # Parse one-or-more comma-separated 1-based indices.
    try:
        idxs = [int(tok) - 1 for tok in sel.replace(" ", "").split(",") if tok]
        if not idxs or any(not 0 <= i < len(files) for i in idxs):
            raise ValueError
    except ValueError:
        console.print("[red]Invalid selection. Enter a number (2), a list "
                      "(1,3), or 'all'.[/red]")
        raise typer.Exit(1)

    chosen = [files[i] for i in dict.fromkeys(idxs)]  # dedup, keep order
    _record(chosen)
    if len(chosen) == 1:
        console.print(f"[green]{label}: using {chosen[0].name}[/green]")
        return str(chosen[0])
    if not allow_concat:
        console.print("[yellow]Multiple files not supported here; using the "
                      f"first: {chosen[0].name}[/yellow]")
        return str(chosen[0])
    console.print(f"[green]{label}: using {len(chosen)} files: "
                  f"{', '.join(f.name for f in chosen)}[/green]")
    return _concat_data_files(chosen, label, interactive=True)


_CYCLE_FINDINGS_DEFAULT = ".ledgerhound/findings/cycle.jsonl"


def _cycle_findings_path_for(data_dir: str) -> str:
    """A findings-log path namespaced to this data-dir, so cycles over
    different datasets don't cross-contaminate strata/findings."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(Path(data_dir).resolve().name)
                  or "cycle").strip("_") or "cycle"
    return f".ledgerhound/findings/cycle_{slug}.jsonl"


def _read_data(path: str):
    """Read a CSV/parquet data file into a DataFrame."""
    import pandas as pd
    p = Path(path)
    return (pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq")
            else pd.read_csv(p))


def _cycle_findings(findings_path: Optional[str]):
    """Shared FindingsLog so every cycle step compounds (same path by default)."""
    from ledgerhound.findings import FindingsLog
    p = Path(findings_path or _CYCLE_FINDINGS_DEFAULT)
    log = FindingsLog(p)
    console.print(f"[green]OK Findings log:[/green] {p}")
    return log


def _build_cycle_llm(model, model_name, api_key, llm_config):
    """Build + health-check the LLM (key from env per project rules)."""
    if not model and not llm_config:
        console.print("[red]Provide --model or --llm-config[/red]")
        raise typer.Exit(1)
    from ledgerhound.llm.client import LLMClient
    llm = (LLMClient.from_config(llm_config) if llm_config
           else LLMClient(base_url=model, model=model_name or "default",
                          api_key=api_key or "EMPTY"))
    h = llm.health_check()
    if h.get("status") != "healthy":
        console.print(f"[red]LLM not available: {h.get('error', h)}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]OK LLM: {h.get('models', ['?'])[0]}[/green]")
    return llm


def _build_eda_agent(llm, data_path, findings_log, output_dir="outputs/cycle"):
    """A digest-backed EDAAgent over ``data_path`` sharing ``findings_log``."""
    from ledgerhound.agents.eda_agent import EDAAgent
    from ledgerhound.core import Ledgerhound
    eng = Ledgerhound(data=data_path)
    agent = EDAAgent(llm=llm, backend=eng.backend, knowledge=eng.knowledge,
                     findings_log=findings_log, output_dir=output_dir)
    agent.set_digest(eng.profile())
    return agent


def _run_questions(agent, questions, label, *, interactive: bool = True):
    """Ask each question; findings land in the shared log. Returns count ok."""
    ok = 0
    for q in questions:
        console.print(f"\n  [bold]> {q}[/bold]")
        try:
            agent.ask(q, interactive=interactive)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]x {str(exc)[:80]}[/red]")
    console.print(f"[green]{label}: {ok}/{len(questions)} questions ok[/green]")
    return ok


def _load_predicates(path: Optional[str]) -> dict:
    """Load a {stratum_name: sql_predicate} map from a JSON or YAML file."""
    if not path:
        return {}
    _check_file(path, "Predicates file")
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


@cycle_app.command("plan")
def cycle_plan(
    fraud_data: str = typer.Option(
        ..., "--fraud-data",
        help="Fraud data: a CSV/parquet file, OR a folder (you'll be prompted "
             "to pick which file)"),
    source_table: str = typer.Option(
        ..., "--source-table",
        help="Fully-qualified Trino table to draw non-fraud rows from"),
    out: str = typer.Option(
        "nonfraud_plan.md", "--out", "-o", help="Output Markdown plan path"),
    predicates: Optional[str] = typer.Option(
        None, "--predicates",
        help="JSON/YAML map {stratum_name: sql_predicate}"),
    nonfraud_filter: str = typer.Option(
        "is_fraud = 0", "--nonfraud-filter",
        help="SQL boolean that selects NON-fraud rows"),
    strategy: str = typer.Option(
        "representative_reservoir", "--strategy",
        help="representative_reservoir (default; correct for descriptive EDA) "
             "| matched_controls | stratified_mirror. Matched equalises the "
             "matched columns, so they are excluded from the step-6 comparison."),
    ratio: float = typer.Option(1.0, "--ratio",
                                help="non-fraud:fraud row ratio (matched/mirror)"),
    budget: int = typer.Option(50000, "--budget",
                               help="total non-fraud rows for representative_reservoir"),
    stratify_on: Optional[str] = typer.Option(
        None, "--stratify-on",
        help="nuisance/time column to stratify the representative draw on "
             "(e.g. txn_month) - keeps the draw balanced on this axis without "
             "matching on signal features"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Steps 1-3: fraud EDA -> non-fraud sampling design -> export runnable SQL.

    Writes a Markdown plan (strategy narrative + Trino SQL) for a human to run.
    The non-fraud strata are derived from the fraud EDA findings themselves.
    """
    from ledgerhound.agents.eda_agent import EDAAgent
    from ledgerhound.agents.fraud_nonfraud import FraudNonFraudCycle
    from ledgerhound.core import Ledgerhound
    from ledgerhound.findings import FindingsLog
    from ledgerhound.llm.client import LLMClient
    from ledgerhound.sampling.design import NonFraudSamplingDesigner, SamplingStrategy
    from ledgerhound.trino.sampling_sql import SamplingSQLConfig

    fraud_data = _resolve_data_path(fraud_data, "Fraud data")
    if not model and not llm_config:
        console.print("[red]Provide --model or --llm-config[/red]")
        raise typer.Exit(1)
    try:
        strat = SamplingStrategy(strategy)
    except ValueError:
        console.print(f"[red]Unknown --strategy '{strategy}'.[/red]")
        raise typer.Exit(1)

    llm = (LLMClient.from_config(llm_config) if llm_config
           else LLMClient(base_url=model, model=model_name or "default",
                          api_key=api_key or "EMPTY"))
    h = llm.health_check()
    if h.get("status") != "healthy":
        console.print(f"[red]LLM not available: {h.get('error', h)}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]OK LLM: {h.get('models', ['?'])[0]}[/green]")

    findings_p = Path(findings_path
                      or f".ledgerhound/findings/{Path(fraud_data).stem}_cycle.jsonl")
    findings_log = FindingsLog(findings_p)
    console.print(f"[green]OK Findings log:[/green] {findings_p}")

    engine = Ledgerhound(data=fraud_data)
    digest = engine.profile()
    fraud_agent = EDAAgent(llm=llm, backend=engine.backend,
                           knowledge=engine.knowledge, findings_log=findings_log)
    fraud_agent.set_digest(digest)

    designer = NonFraudSamplingDesigner(
        strat, ratio=ratio, total_budget=budget,
        stratum_predicates=_load_predicates(predicates),
        stratify_on=stratify_on)
    cycle = FraudNonFraudCycle(
        fraud_agent, findings_log,
        designer=designer,
        sql_config=SamplingSQLConfig(source_table=source_table,
                                     nonfraud_filter=nonfraud_filter),
        interactive=True)

    console.print(Panel(f"[bold]EDA cycle - plan[/bold]\n"
                        f"  fraud:  [cyan]{fraud_data}[/cyan]\n"
                        f"  source: [cyan]{source_table}[/cyan]\n"
                        f"  strategy: [cyan]{strategy}[/cyan]",
                        border_style="cyan"))
    out_path = cycle.run_until_fetch(out)
    d = cycle.design
    console.print(f"\n[bold green]Plan written:[/bold green] {out_path}")
    console.print(f"  {d.summary_line()}")
    if d.warnings:
        for w in d.warnings:
            console.print(f"  [yellow]! {w}[/yellow]")
    if cycle.plan and cycle.plan.requires_human:
        console.print("[yellow]! Some strata need predicates - fill the TODOs "
                      "in the SQL before running (pass --predicates).[/yellow]")


@cycle_app.command("compare")
def cycle_compare(
    fraud_data: str = typer.Option(
        ..., "--fraud-data",
        help="Fraud data: a CSV/parquet file, OR a folder (you'll be prompted)"),
    nonfraud_data: str = typer.Option(
        ..., "--nonfraud-data",
        help="Non-fraud data: a CSV/parquet file, OR a folder (you'll be prompted)"),
    features: Optional[str] = typer.Option(
        None, "--features",
        help="Comma-separated numeric columns. Omit to auto-detect the shared "
             "numeric columns (minus ids/labels/helpers)."),
    derive: Optional[str] = typer.Option(
        None, "--derive",
        help="JSON/YAML list of SAFE exploratory derivations (date_diff/ratio/"
             "log/hour) applied to both sides before comparison. Tagged as "
             "EDA-derived (NOT leakage-validated) for the FE team."),
    breakdown_by: Optional[str] = typer.Option(
        None, "--breakdown-by",
        help="Segment column to break the comparison down by (e.g. "
             "purpose_code) - shows whether a signal is consistent across "
             "segments"),
    buckets: int = typer.Option(10, "--buckets", help="PSI bucket count"),
    report: Optional[str] = typer.Option(None, "--report", "-r",
                                         help="Markdown report path"),
):
    """Step 6: rank features by fraud-vs-non-fraud PSI divergence (no LLM)."""
    import pandas as pd

    from ledgerhound.agents.fraud_nonfraud import auto_numeric_features, compare_by_segment, compare_fraud_nonfraud
    from ledgerhound.analysis.eda_derive import apply_derivations, load_derivations

    fraud_data = _resolve_data_path(fraud_data, "Fraud data")
    nonfraud_data = _resolve_data_path(nonfraud_data, "Non-fraud data")

    def _read(p):
        path = Path(p)
        return (pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq")
                else pd.read_csv(path))

    fraud_df = _read(fraud_data)
    nonfraud_df = _read(nonfraud_data)

    # Optional EDA-time derivations (applied to BOTH sides).
    derived_names: list[str] = []
    if derive:
        derivations = load_derivations(derive)
        fraud_df, df_added, w1 = apply_derivations(fraud_df, derivations)
        nonfraud_df, nf_added, w2 = apply_derivations(nonfraud_df, derivations)
        derived_names = [n for n in df_added if n in nf_added]
        for w in set(w1 + w2):
            console.print(f"[yellow]derive: {w}[/yellow]")
        if derived_names:
            console.print(f"[green]Derived (exploratory, NOT leakage-validated): "
                          f"{', '.join(derived_names)}[/green]")

    if features:
        feats = [f.strip() for f in features.split(",") if f.strip()]
    else:
        feats = auto_numeric_features(fraud_df, nonfraud_df)
        console.print(f"[green]Auto-detected {len(feats)} numeric feature(s): "
                      f"{', '.join(feats) or '(none)'}[/green]")
    # Ensure derived columns are compared even if auto-detect missed them.
    for n in derived_names:
        if n not in feats:
            feats.append(n)

    comp = compare_fraud_nonfraud(fraud_df, nonfraud_df, feats,
                                  n_buckets=buckets, derived=derived_names)

    from rich.table import Table
    t = Table(title="Fraud vs non-fraud divergence (PSI)", border_style="cyan")
    for c in ("Feature", "PSI", "Status", "n_fraud", "n_nonfraud"):
        t.add_column(c)
    for d in comp.divergences:
        t.add_row(d.feature, f"{d.psi:.4f}", d.status,
                  str(d.n_fraud), str(d.n_nonfraud))
    console.print(t)
    if comp.skipped:
        console.print(f"[yellow]Skipped (missing/constant): "
                      f"{', '.join(comp.skipped)}[/yellow]")

    seg = None
    if breakdown_by:
        seg = compare_by_segment(fraud_df, nonfraud_df, feats, breakdown_by,
                                 n_buckets=buckets, derived=derived_names)
        if not seg.by_segment:
            console.print(f"[yellow]Breakdown by '{breakdown_by}': no segment "
                          f"had enough rows on both sides.[/yellow]")
        else:
            console.print(f"\n[bold]Breakdown by {breakdown_by}[/bold]")
            for sv, scomp in seg.by_segment.items():
                tops = ", ".join(f"{d.feature}={d.psi:.2f}" for d in scomp.top(3))
                console.print(f"  [cyan]{sv}[/cyan]: {tops or '-'}")

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        md = comp.to_markdown()
        if seg is not None:
            md += "\n\n" + seg.to_markdown()
        Path(report).write_text(md, encoding="utf-8")
        console.print(f"[green]Report written:[/green] {report}")


@cycle_app.command("nonfraud")
def cycle_nonfraud(
    nonfraud_data: str = typer.Option(
        ..., "--nonfraud-data",
        help="Non-fraud data: file OR folder (you'll be prompted)"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Step 4: run EDA on the fetched non-fraud sample (shares the findings log)."""
    from ledgerhound.agents.fraud_nonfraud import DEFAULT_QUESTIONS
    nonfraud_data = _resolve_data_path(nonfraud_data, "Non-fraud data")
    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)
    agent = _build_eda_agent(llm, nonfraud_data, log)
    console.print(Panel("[bold]Step 4 - Non-fraud EDA[/bold]", border_style="cyan"))
    _run_questions(agent, DEFAULT_QUESTIONS["nonfraud"], "Non-fraud EDA")


@cycle_app.command("joint")
def cycle_joint(
    fraud_data: str = typer.Option(..., "--fraud-data",
                                   help="Fraud data: file OR folder"),
    nonfraud_data: str = typer.Option(..., "--nonfraud-data",
                                      help="Non-fraud data: file OR folder"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Step 5: joint EDA over a stacked fraud + non-fraud view (_group tag)."""
    from ledgerhound.agents.fraud_nonfraud import DEFAULT_QUESTIONS, build_joint_dataframe
    fraud_data = _resolve_data_path(fraud_data, "Fraud data")
    nonfraud_data = _resolve_data_path(nonfraud_data, "Non-fraud data")
    joint = build_joint_dataframe(_read_data(fraud_data), _read_data(nonfraud_data))
    tmp = Path(".ledgerhound/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    jp = tmp / "joint_view.parquet"
    joint.to_parquet(jp, index=False)
    console.print(f"[green]Joint view:[/green] {jp} ({len(joint):,} rows, "
                  f"_group = fraud/nonfraud)")
    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)
    agent = _build_eda_agent(llm, str(jp), log)
    console.print(Panel("[bold]Step 5 - Joint EDA[/bold]", border_style="cyan"))
    _run_questions(agent, DEFAULT_QUESTIONS["joint"], "Joint EDA")


@cycle_app.command("start")
def cycle_start(
    data_dir: str = typer.Option(
        "../data", "--data-dir",
        help="Folder containing fraud/ and nonfraud/ subfolders"),
    source_table: str = typer.Option(..., "--source-table"),
    out: str = typer.Option("nonfraud_plan.md", "--out", "-o"),
    predicates: Optional[str] = typer.Option(None, "--predicates"),
    strategy: str = typer.Option("representative_reservoir", "--strategy"),
    budget: int = typer.Option(50000, "--budget"),
    stratify_on: Optional[str] = typer.Option(None, "--stratify-on"),
    nonfraud_filter: str = typer.Option("is_fraud = 0", "--nonfraud-filter"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Conversational steps 1-3: ask context, profile fraud, export the SQL plan.

    NOTE: superseded by `ledgerhound cycle run --mode hitl`, which drives all six
    steps with checkpoints and a resumable saved state. This verb is kept for
    the step-by-step workflow but will be deprecated.
    """
    out = _default_out(out, "nonfraud_plan.md")
    console.print("[dim]Tip: `ledgerhound cycle run --mode hitl` now drives the "
                  "whole cycle with checkpoints + --resume.[/dim]")
    from ledgerhound.agents.fraud_nonfraud import FraudNonFraudCycle
    from ledgerhound.findings.models import Evidence, Finding
    from ledgerhound.sampling.design import NonFraudSamplingDesigner, SamplingStrategy
    from ledgerhound.trino.sampling_sql import SamplingSQLConfig

    topic = typer.prompt("What are we working on today?")
    fraud_folder = Path(data_dir) / "fraud"
    nonfraud_folder = Path(data_dir) / "nonfraud"
    nf_n = (len(list(nonfraud_folder.glob("*"))) if nonfraud_folder.exists() else 0)
    console.print(f"\n[bold]Data under {data_dir}[/bold]")
    console.print(f"  fraud/    -> {'exists' if fraud_folder.exists() else 'MISSING'}")
    console.print(f"  nonfraud/ -> {nf_n} file(s)"
                  + ("  (empty - filled by step 3)" if nf_n == 0 else ""))

    fraud_data = _resolve_data_path(str(fraud_folder), "Fraud data")
    desc = typer.prompt("Briefly describe this fraud dataset "
                        "(what is a row? which column is the fraud label?)")

    try:
        strat = SamplingStrategy(strategy)
    except ValueError:
        console.print(f"[red]Unknown --strategy '{strategy}'[/red]")
        raise typer.Exit(1)
    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)
    # Record the human context as findings so later steps see it.
    log.append(Finding(claim=f"Cycle objective: {topic}",
                       evidence=Evidence(summary=desc, code="<analyst note>"),
                       dataset="fraud", confidence=0.5, tags=["cycle_context"]))

    agent = _build_eda_agent(llm, fraud_data, log)
    cycle = FraudNonFraudCycle(
        agent, log,
        designer=NonFraudSamplingDesigner(
            strat, total_budget=budget, stratify_on=stratify_on,
            stratum_predicates=_load_predicates(predicates)),
        sql_config=SamplingSQLConfig(source_table=source_table,
                                     nonfraud_filter=nonfraud_filter),
        interactive=True)
    out_path = cycle.run_until_fetch(out)
    console.print(f"\n[bold green]Plan written:[/bold green] {out_path}")
    console.print(f"  {cycle.design.summary_line()}")
    console.print(f"\n[dim]Next: run the SQL in {out_path}, drop the result CSV "
                  f"into {nonfraud_folder}, then `ledgerhound cycle resume`.[/dim]")


@cycle_app.command("resume")
def cycle_resume(
    data_dir: str = typer.Option("../data", "--data-dir"),
    features: Optional[str] = typer.Option(None, "--features"),
    breakdown_by: Optional[str] = typer.Option(None, "--breakdown-by"),
    report: str = typer.Option("cycle_report.md", "--report", "-r"),
    buckets: int = typer.Option(10, "--buckets"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
):
    """Conversational steps 4-6 after the non-fraud data has been fetched.

    NOTE: `ledgerhound cycle run --resume` supersedes this -- it continues the
    SAME saved cycle (honouring its mode + config) instead of a separate
    steps-4-6 flow. Kept for the step-by-step workflow.
    """
    console.print("[dim]Tip: `ledgerhound cycle run --resume` continues a saved "
                  "cycle at the right step with its original config.[/dim]")
    from ledgerhound.agents.fraud_nonfraud import (
        DEFAULT_QUESTIONS,
        auto_numeric_features,
        build_joint_dataframe,
        compare_by_segment,
        compare_fraud_nonfraud,
    )

    fraud_data = _resolve_data_path(str(Path(data_dir) / "fraud"), "Fraud data")
    nonfraud_data = _resolve_data_path(str(Path(data_dir) / "nonfraud"),
                                       "Non-fraud data")
    typer.prompt("Briefly describe the non-fraud sample (same schema as fraud?)",
                 default="", show_default=False)

    llm = _build_cycle_llm(model, model_name, api_key, llm_config)
    log = _cycle_findings(findings_path)

    # Step 4
    console.print(Panel("[bold]Step 4 - Non-fraud EDA[/bold]", border_style="cyan"))
    _run_questions(_build_eda_agent(llm, nonfraud_data, log),
                   DEFAULT_QUESTIONS["nonfraud"], "Non-fraud EDA")

    # Step 5 (joint view)
    fraud_df = _read_data(fraud_data)
    nonfraud_df = _read_data(nonfraud_data)
    joint = build_joint_dataframe(fraud_df, nonfraud_df)
    tmp = Path(".ledgerhound/tmp"); tmp.mkdir(parents=True, exist_ok=True)
    jp = tmp / "joint_view.parquet"; joint.to_parquet(jp, index=False)
    console.print(Panel("[bold]Step 5 - Joint EDA[/bold]", border_style="cyan"))
    _run_questions(_build_eda_agent(llm, str(jp), log),
                   DEFAULT_QUESTIONS["joint"], "Joint EDA")

    # Step 6
    console.print(Panel("[bold]Step 6 - Comparison[/bold]", border_style="cyan"))
    feats = ([f.strip() for f in features.split(",") if f.strip()] if features
             else auto_numeric_features(fraud_df, nonfraud_df))
    comp = compare_fraud_nonfraud(fraud_df, nonfraud_df, feats, n_buckets=buckets)
    md = comp.to_markdown()
    for d in comp.top(10):
        console.print(f"  {d.feature:<28} PSI={d.psi:.4f}  ({d.status})")
    if breakdown_by:
        seg = compare_by_segment(fraud_df, nonfraud_df, feats, breakdown_by,
                                 n_buckets=buckets)
        md += "\n\n" + seg.to_markdown()
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(md, encoding="utf-8")
    console.print(f"\n[bold green]Cycle report written:[/bold green] {report}")


# ---------------------------------------------------------------------------
#  ledgerhound cycle run  --  mode-driven, checkpointed, resumable orchestration
# ---------------------------------------------------------------------------


def _describe_datasets(selected_files: list, dataset_tag: str, log, *,
                       interactive: bool) -> None:
    """Ask the analyst what each selected file is, and record it as findings
    so the agent starts from real context (what a row is, which column is the
    label, the time period) instead of guessing. Skipped in auto mode."""
    if not interactive or not selected_files:
        return
    from ledgerhound.findings.models import Evidence, Finding
    console.print(f"\n[bold]Before I analyse the {dataset_tag} data, tell me "
                  f"what each file is[/bold] (press Enter to skip a file):")
    recorded = 0
    for name in selected_files:
        try:
            desc = typer.prompt(
                f"  '{name}' -- what is a row? which column is the label? "
                "what period does it cover?", default="", show_default=False
            ).strip()
        except Exception:
            desc = ""
        if desc:
            log.append(Finding(
                claim=f"Analyst context for {dataset_tag} file '{name}': {desc}",
                evidence=Evidence(summary=desc, code="<analyst note>"),
                dataset=dataset_tag, confidence=0.5,
                tags=["cycle_context", "dataset_description"]))
            recorded += 1
    if recorded:
        console.print(f"[green]Recorded context for {recorded} file(s) -- the "
                      "agent will use it.[/green]")


def _ask_more_questions(default_qs: list, subject: str) -> list:
    """For the 'another EDA pass' choice: let the analyst steer the next pass
    with their own question instead of blindly replaying the same list
    (which just produces duplicate findings)."""
    try:
        custom = typer.prompt(
            f"What should the next pass on the {subject} explore? "
            "(blank = repeat the standard questions)",
            default="", show_default=False).strip()
    except Exception:
        custom = ""
    return [custom] if custom else default_qs


def _checkpoint_reader(data_dir: str):
    """A rich-rendering reader for HITL checkpoints (returns the chosen key)."""
    def reader(cp) -> str:
        console.print(f"\n[bold]{cp.prompt}[/bold]")
        keyed = list(cp.choices)
        for i, ch in enumerate(keyed, 1):
            label = ch.label
            if ch.key == "export":
                label = f"{label} {Path(data_dir) / 'nonfraud'}/"
            mark = "  [dim](default)[/dim]" if ch.is_default else ""
            gate = "  [yellow](approval required)[/yellow]" if ch.gated else ""
            console.print(f"  [cyan]{i}[/cyan]/[cyan]{ch.key}[/cyan]  {label}{mark}{gate}")
        raw = typer.prompt("Choose", default=cp.default_key()).strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(keyed):
            return keyed[int(raw) - 1].key
        return raw
    return reader


@cycle_app.command("run")
def cycle_run(
    data_dir: str = typer.Option(
        "../data", "--data-dir",
        help="Folder containing fraud/ and nonfraud/ subfolders"),
    mode: str = typer.Option(
        "hitl", "--mode",
        help="plan | hitl | auto  (default hitl: pause at every step)"),
    source_table: Optional[str] = typer.Option(None, "--source-table"),
    resume: bool = typer.Option(
        False, "--resume", help="Continue from the saved cycle_state.json"),
    predicates: Optional[str] = typer.Option(None, "--predicates"),
    strategy: str = typer.Option("representative_reservoir", "--strategy"),
    budget: int = typer.Option(50000, "--budget"),
    stratify_on: Optional[str] = typer.Option(None, "--stratify-on"),
    nonfraud_filter: str = typer.Option("is_fraud = 0", "--nonfraud-filter"),
    features: Optional[str] = typer.Option(None, "--features"),
    report: str = typer.Option("cycle_report.md", "--report", "-r"),
    out: str = typer.Option("nonfraud_plan.md", "--out", "-o"),
    llm_config: Optional[str] = typer.Option(None, "--llm-config"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    model_name: Optional[str] = typer.Option(None, "--model-name"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    findings_path: Optional[str] = typer.Option(None, "--findings"),
    state_root: str = typer.Option(".ledgerhound", "--state-root"),
):
    """Drive the whole 6-step cycle with HITL checkpoints between steps.

    Modes (like a CLI permission mode, chosen once at the start):

      plan  -- run the read-only steps (fraud EDA, sampling design, SQL),
               present the plan, then STOP. Executes/fetches nothing.
      hitl  -- (default) pause after every step and ask what to do next;
               at the fetch boundary ask export vs run-now vs skip.
      auto  -- follow the canonical sequence without asking routing
               questions. Production Trino execution is STILL gated
               (AUD-098): auto defaults the fetch step to 'export'.

    Every checkpoint is a save-point (.ledgerhound/cycle_state.json), so you
    can stop after any step and resume later with --resume.
    """
    out = _default_out(out, "nonfraud_plan.md")
    report = _default_out(report, "cycle_report.md")
    from ledgerhound.agents.cycle_mode import (
        CHECKPOINTS,
        CycleMode,
        CycleState,
        plan_mode_should_stop,
    )
    from ledgerhound.agents.fraud_nonfraud import (
        DEFAULT_QUESTIONS,
        auto_numeric_features,
        build_joint_dataframe,
        compare_fraud_nonfraud,
    )
    from ledgerhound.sampling.design import NonFraudSamplingDesigner, SamplingStrategy
    from ledgerhound.trino.sampling_sql import SamplingSQLConfig, build_sql, export_markdown

    try:
        cmode = CycleMode.parse(mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    # --- state: resume or start fresh --------------------------------------
    if resume:
        state = CycleState.load(state_root)
        if state is None:
            console.print("[red]No saved cycle to --resume "
                          f"({CycleState.path_for(state_root)} not found).[/red]")
            raise typer.Exit(1)
        cmode = CycleMode.parse(state.mode)  # honour the run's original mode
        # Restore the ORIGINAL run config so resume continues the same cycle
        # (not the CLI defaults). CLI flags are ignored on resume except
        # --resume itself; start fresh to change config.
        data_dir = state.data_dir or data_dir
        pr = state.params or {}
        source_table = pr.get("source_table", source_table)
        strategy = pr.get("strategy", strategy)
        budget = pr.get("budget", budget)
        stratify_on = pr.get("stratify_on", stratify_on)
        nonfraud_filter = pr.get("nonfraud_filter", nonfraud_filter)
        predicates = pr.get("predicates", predicates)
        features = pr.get("features", features)
        out = pr.get("out", out)
        report = pr.get("report", report)
        findings_path = state.findings_path or findings_path
        console.print(f"[green]Resuming[/green] at step "
                      f"[bold]{state.current_step}[/bold] "
                      f"(mode={state.mode}, data-dir={data_dir})")
    else:
        CycleState.clear(state_root)
        # Namespace the findings log per data-dir so a new dataset's cycle
        # can't read a previous dataset's strata (unless --findings is given).
        eff_findings = findings_path or _cycle_findings_path_for(data_dir)
        state = CycleState(
            mode=cmode.value, data_dir=data_dir, findings_path=eff_findings,
            params={
                "source_table": source_table, "strategy": strategy,
                "budget": budget, "stratify_on": stratify_on,
                "nonfraud_filter": nonfraud_filter, "predicates": predicates,
                "features": features, "out": out, "report": report,
            })
        findings_path = eff_findings
        console.print(Panel(f"[bold]EDA cycle -- mode: {cmode.value}[/bold]",
                            border_style="cyan"))

    # Auto mode must be non-interactive end to end (no agent ask_user
    # prompts, no file-picker menus) so it can run unattended.
    step_interactive = cmode != CycleMode.AUTO
    reader = _checkpoint_reader(data_dir)
    log = _cycle_findings(findings_path or state.findings_path or None)

    def _need_llm():
        return _build_cycle_llm(model, model_name, api_key, llm_config)

    def _checkpoint(step: str) -> str:
        cp = CHECKPOINTS.get(step)
        if cp is None:
            return "proceed"
        choice = cp.resolve(cmode, reader if cmode == CycleMode.HITL else None)
        state.record(step, choice)
        state.save(state_root)
        return choice

    _fraud_files: list = []
    fraud_data = _resolve_data_path(str(Path(data_dir) / "fraud"), "Fraud data",
                                    interactive=step_interactive,
                                    selected_out=_fraud_files)

    # ===================================================================
    # Step 1: Fraud EDA  (repeatable via the 'more_eda' choice)
    # ===================================================================
    if state.current_step == "fraud_eda":
        # Ask what the selected file(s) are BEFORE analysing, and record it
        # as context the agent uses (only on a fresh start, interactive modes).
        _describe_datasets(_fraud_files, "fraud", log, interactive=step_interactive)
        llm = _need_llm()
        questions = DEFAULT_QUESTIONS.get("fraud", [])
        while True:
            console.print(Panel("[bold]Step 1 - Fraud EDA[/bold]", border_style="cyan"))
            _run_questions(_build_eda_agent(llm, fraud_data, log),
                           questions, "Fraud EDA", interactive=step_interactive)
            choice = _checkpoint("fraud_eda")
            if choice == "more_eda":
                questions = _ask_more_questions(
                    DEFAULT_QUESTIONS.get("fraud", []), "fraud data")
                continue
            # step 1 is done either way -- advance the cursor BEFORE stopping
            # so --resume does not re-run (and re-charge) the whole EDA pass.
            state.advance_to("nonfraud_design")
            state.save(state_root)
            if choice == "stop":
                _cycle_stop(state, state_root)
                return
            break

    # ===================================================================
    # Step 2+3: sampling design -> SQL plan  (read-only, no gate)
    # ===================================================================
    if state.current_step in ("nonfraud_design", "sql_plan"):
        if not source_table:
            console.print("[red]--source-table is required to design the "
                          "non-fraud sampling SQL.[/red]")
            raise typer.Exit(1)
        try:
            strat = SamplingStrategy(strategy)
        except ValueError:
            console.print(f"[red]Unknown --strategy '{strategy}'[/red]")
            raise typer.Exit(1)
        designer = NonFraudSamplingDesigner(
            strat, total_budget=budget, stratify_on=stratify_on,
            stratum_predicates=_load_predicates(predicates))
        design = designer.design(log, target_dataset="nonfraud",
                                 parent_sample="fraud")
        console.print(Panel("[bold]Step 2/3 - Non-fraud sampling design + SQL[/bold]",
                            border_style="cyan"))
        console.print(f"  {design.summary_line()}")
        for w in design.warnings:
            console.print(f"  [yellow]! {w}[/yellow]")
        plan = build_sql(design, SamplingSQLConfig(
            source_table=source_table, nonfraud_filter=nonfraud_filter))
        out_path = export_markdown(plan, out)
        console.print(f"  [green]SQL plan exported:[/green] {out_path}")
        state.record("sql_plan", "exported")
        state.advance_to("fetch")
        state.save(state_root)

        if cmode == CycleMode.PLAN and plan_mode_should_stop("sql_plan"):
            console.print(Panel(
                "[bold]plan mode complete[/bold] -- strategy + SQL are in "
                f"{out_path}. Nothing was executed or fetched.",
                border_style="green"))
            return

    # ===================================================================
    # Step 3.5: fetch decision (the "who runs the query" gate)
    # ===================================================================
    if state.current_step == "fetch":
        nf_dir = Path(data_dir) / "nonfraud"
        nf_present = nf_dir.exists() and any(
            f.suffix.lower() in _DATA_SUFFIXES for f in nf_dir.iterdir()
            if f.is_file())
        # If the non-fraud data is already there (e.g. a --resume after the
        # human ran the SQL), proceed to step 4 instead of asking again --
        # this is what breaks the auto-mode fetch loop.
        if nf_present:
            console.print("[green]Non-fraud data present -- continuing to "
                          "step 4.[/green]")
            choice = "skip"
            state.record("fetch", "skip (data present)")
            state.save(state_root)
        else:
            choice = _checkpoint("fetch")
        if choice == "stop":
            _cycle_stop(state, state_root)
            return
        if choice == "export":
            console.print(Panel(
                f"[bold]Run the exported SQL ({out}) in Trino[/bold], then drop "
                f"the result CSV into [cyan]{nf_dir}/[/cyan] and re-run with "
                "[bold]--resume[/bold] to continue at step 4.",
                border_style="yellow"))
            return
        if choice == "execute":
            # AUD-098: production execution is hard-gated in TrinoClient, which
            # the CLI does not yet construct. Be honest and fall back to export.
            console.print(Panel(
                "[yellow]Run-now needs a configured TrinoClient (not wired into "
                "the CLI yet); the query is NOT run here.[/yellow] Use the "
                f"exported SQL in {out} (it still passes the same approval "
                "gate when executed through the Python API), drop the CSV in "
                f"{Path(data_dir) / 'nonfraud'}/, then --resume.",
                border_style="yellow"))
            return
        # skip -> non-fraud already present; fall through to step 4
        state.advance_to("nonfraud_eda")
        state.save(state_root)

    # ===================================================================
    # Step 4: Non-fraud EDA  (repeatable)
    # ===================================================================
    nonfraud_data = None
    if state.current_step == "nonfraud_eda":
        _nf_files: list = []
        nonfraud_data = _resolve_data_path(
            str(Path(data_dir) / "nonfraud"), "Non-fraud data",
            interactive=step_interactive, selected_out=_nf_files)
        _describe_datasets(_nf_files, "nonfraud", log, interactive=step_interactive)
        llm = _need_llm()
        questions = DEFAULT_QUESTIONS.get("nonfraud", [])
        while True:
            console.print(Panel("[bold]Step 4 - Non-fraud EDA[/bold]", border_style="cyan"))
            _run_questions(_build_eda_agent(llm, nonfraud_data, log),
                           questions, "Non-fraud EDA", interactive=step_interactive)
            choice = _checkpoint("nonfraud_eda")
            if choice == "more_eda":
                questions = _ask_more_questions(
                    DEFAULT_QUESTIONS.get("nonfraud", []), "non-fraud data")
                continue
            state.advance_to("joint_eda")
            state.save(state_root)
            if choice == "stop":
                _cycle_stop(state, state_root)
                return
            break

    # ===================================================================
    # Step 5: Joint EDA
    # ===================================================================
    if state.current_step == "joint_eda":
        if nonfraud_data is None:
            nonfraud_data = _resolve_data_path(
                str(Path(data_dir) / "nonfraud"), "Non-fraud data",
                interactive=step_interactive)
        fraud_df = _read_data(fraud_data)
        nonfraud_df = _read_data(nonfraud_data)
        joint = build_joint_dataframe(fraud_df, nonfraud_df)
        tmp = Path(".ledgerhound/tmp"); tmp.mkdir(parents=True, exist_ok=True)
        jp = tmp / "joint_view.parquet"; joint.to_parquet(jp, index=False)
        llm = _need_llm()
        questions = DEFAULT_QUESTIONS.get("joint", [])
        while True:
            console.print(Panel("[bold]Step 5 - Joint EDA[/bold]", border_style="cyan"))
            _run_questions(_build_eda_agent(llm, str(jp), log),
                           questions, "Joint EDA", interactive=step_interactive)
            choice = _checkpoint("joint_eda")
            if choice == "more_eda":
                questions = _ask_more_questions(
                    DEFAULT_QUESTIONS.get("joint", []), "joint view")
                continue
            state.advance_to("compare")
            state.save(state_root)
            if choice == "stop":
                _cycle_stop(state, state_root)
                return
            break

    # ===================================================================
    # Step 6: Comparison (no LLM) -> report -> done
    # ===================================================================
    if state.current_step == "compare":
        fraud_df = _read_data(fraud_data)
        nonfraud_data = nonfraud_data or _resolve_data_path(
            str(Path(data_dir) / "nonfraud"), "Non-fraud data",
            interactive=step_interactive)
        nonfraud_df = _read_data(nonfraud_data)
        feats = ([f.strip() for f in features.split(",") if f.strip()]
                 if features else auto_numeric_features(fraud_df, nonfraud_df))
        comp = compare_fraud_nonfraud(fraud_df, nonfraud_df, feats)
        console.print(Panel("[bold]Step 6 - Fraud vs non-fraud comparison[/bold]",
                            border_style="cyan"))
        for d in comp.top(10):
            console.print(f"  {d.feature:<28} PSI={d.psi:.4f}  ({d.status})")
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(comp.to_markdown(), encoding="utf-8")
        state.record("compare", "done")
        state.save(state_root)
        console.print(f"\n[bold green]Cycle complete.[/bold green] Report: {report}")
        CycleState.clear(state_root)


def _cycle_stop(state, state_root) -> None:
    """Persist and print resume instructions when a human stops the cycle."""
    state.save(state_root)
    console.print(Panel(
        f"[bold]Paused.[/bold] Completed work is saved; on resume the cycle "
        f"continues at '[cyan]{state.current_step}[/cyan]' (nothing already "
        f"done will re-run). State: {state.path_for(state_root)}. Continue "
        "with [bold]ledgerhound cycle run --resume[/bold].",
        border_style="yellow"))


sessions_app = typer.Typer(
    name="sessions",
    help=("Cross-deployment session-quality observability. Each Ledgerhound "
         "install writes its own local sessions.jsonl (ledgerhound.runtime."
         "session_log) -- these commands pull copies you've collected "
         "into a staging folder and report on them, entirely offline "
         "(no server -- works inside a plain Kubeflow/Jupyter kernel)."),
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("collect")
def sessions_collect(
    staging_dir: str = typer.Option(
        ..., "--staging-dir", "-s",
        help="Folder where you've copied one or more *.jsonl session logs "
             "(however they got there -- kubectl cp, MinIO, etc.)."),
    archive_dir: str = typer.Option(
        ..., "--archive-dir", "-a",
        help="Your permanent local store -- new lines get appended here; "
             "already-collected lines are never re-read or duplicated."),
):
    """Pull only what's NEW since the last collect -- safe to re-run any
    time someone's staged file has grown further."""
    from ledgerhound.runtime.session_collector import collect
    result = collect(Path(staging_dir), Path(archive_dir))
    if not result:
        console.print("[dim]nothing new since the last collect.[/dim]")
        return
    for name, n in result.items():
        console.print(f"[green]{name}[/green]: +{n} new line(s)")


@sessions_app.command("report")
def sessions_report(
    path: str = typer.Argument(
        ..., help="Your archive folder (from `sessions collect`), or a "
                  "single .jsonl file."),
    plot: bool = typer.Option(True, "--plot/--no-plot",
                              help="Render inline charts if matplotlib is "
                                   "available (safe to disable in plain "
                                   "terminals)."),
):
    """Summarise collected sessions: counts, error/refusal rates, and
    quality flags -- across every user you've pulled a log from."""
    from ledgerhound.runtime.session_report import report
    report(path, plot=plot)


@eval_app.command("blade")
def eval_blade(
    datasets: str = typer.Option(..., "--datasets", "-d",
                                 help="Path to BLADE's blade_bench/datasets directory "
                                      "(git clone https://github.com/behavioral-data/BLADE)"),
    llm_config: str = typer.Option("configs/deepseek.yaml", "--llm-config",
                                   help="LLM config to evaluate"),
    arms: str = typer.Option("bare,ledgerhound", "--arms",
                             help="Comma-separated: bare | ledgerhound. Running both "
                                  "isolates whether our discipline actually improves "
                                  "analysis judgement, rather than assuming it does."),
    limit: Optional[int] = typer.Option(None, "--limit",
                                        help="Score only the first N items (smoke run)"),
    out: Optional[str] = typer.Option(None, "--out", help="Write results JSON here"),
):
    """V5-007: score against BLADE, an EXTERNAL benchmark of analysis decisions.

    Every other number in `ledgerhound eval` is scored against our own goldens.
    BLADE (arXiv 2408.09667) grades the decisions inside an open-ended
    analysis -- which is our product claim, measured by someone else.
    """
    import json as _json

    from ledgerhound.eval.blade import chance_rate, load_items, render, run_arm
    from ledgerhound.llm.client import LLMClient

    items = load_items(datasets, limit=limit)
    if not items:
        console.print(f"[red]No select_neg items found under {datasets}[/red]\n"
                      "Expected <datasets>/<name>/mcq_dataset.json files.")
        raise typer.Exit(1)

    llm = LLMClient.from_config(llm_config)
    console.print(f"[dim]model: {llm.config.model} | items: {len(items)} | "
                  f"chance: {chance_rate(items):.1%}[/dim]")

    results = []
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        with console.status(f"[cyan]running arm '{arm}'..."):
            results.append(run_arm(llm, items, arm))
        r = results[-1]
        console.print(f"  {arm}: {r.accuracy:.1%} ({r.correct}/{r.n})")

    console.print()
    console.print(render(results, items), markup=False, highlight=False)

    if out:
        Path(out).write_text(_json.dumps(
            {"model": llm.config.model, "n_items": len(items),
             "chance": chance_rate(items),
             "arms": [r.to_dict() for r in results]}, indent=2),
            encoding="utf-8")
        console.print(f"\n[dim]saved: {out}[/dim]")


@eval_app.command("null-data")
def eval_null_data(
    seeds: int = typer.Option(30, "--seeds", help="Datasets to average over"),
    rows: int = typer.Option(4000, "--rows"),
    features: int = typer.Option(60, "--features"),
    planted: int = typer.Option(6, "--planted",
                                help="Real effects in the signal arm"),
    effect: float = typer.Option(0.55, "--effect", help="Planted effect size"),
    alpha: float = typer.Option(0.05, "--alpha"),
    out: Optional[str] = typer.Option(None, "--out", help="Write results JSON here"),
):
    """V5-009: false-discovery rate under a ground truth we construct.

    Null data has no signal by construction, so every established finding is
    false and nobody has to trust our answer key. Planted data is the half
    that stops the first being won by refusing everything.

    Unlike an LLM benchmark this is deterministic and runs in seconds, so it
    can fail a build.
    """
    import json as _json

    from ledgerhound.eval.null_data import render, run_eval

    res = run_eval(n_seeds=seeds, n_rows=rows, n_features=features,
                   n_planted=planted, effect=effect, alpha=alpha)
    console.print(render(res), markup=False, highlight=False)

    ungated = res.mean(gated=False, kind="null", attr="n_established")
    gated = res.mean(gated=True, kind="null", attr="n_established")
    console.print()
    if gated <= ungated:
        console.print(f"[green]Gating cut false findings on null data "
                      f"{ungated:.2f} -> {gated:.2f} per run.[/green]")
    else:
        console.print("[red]Gating did NOT reduce false findings. "
                      "The mechanism is not doing its job.[/red]")

    if out:
        Path(out).write_text(_json.dumps(res.to_dict(), indent=2),
                             encoding="utf-8")
        console.print(f"[dim]saved: {out}[/dim]")


if __name__ == "__main__":
    app()
