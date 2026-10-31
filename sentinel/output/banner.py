"""
sentinel.output.banner
=======================

Beautiful terminal banners and summary displays for demos.
Uses Rich panels, tables, and colors for maximum visual impact.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich import box

console = Console()

LOGO = """
[bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]###[/bold blue][bold magenta]+[/bold magenta]   [bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]########[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]###[/bold blue][bold magenta]+[/bold magenta]   [bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta]
[bold blue]##[/bold blue][bold magenta]+====+[/bold magenta][bold blue]##[/bold blue][bold magenta]+====+[/bold magenta][bold blue]####[/bold blue][bold magenta]+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]|+==[/bold magenta][bold blue]##[/bold blue][bold magenta]+==+[/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]####[/bold blue][bold magenta]+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]+====+[/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta]
[bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]#####[/bold blue][bold magenta]+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta] [bold blue]##[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta] [bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]#####[/bold blue][bold magenta]+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]|[/bold magenta]
[bold magenta]+====[/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]+==+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]|+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]|+[/bold magenta][bold blue]##[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]+==+[/bold magenta]  [bold blue]##[/bold blue][bold magenta]|[/bold magenta]
[bold blue]#######[/bold blue][bold magenta]|[/bold magenta][bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]##[/bold blue][bold magenta]| +[/bold magenta][bold blue]####[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|   [/bold magenta][bold blue]##[/bold blue][bold magenta]|[/bold magenta][bold blue]##[/bold blue][bold magenta]| +[/bold magenta][bold blue]####[/bold blue][bold magenta]|[/bold magenta][bold blue]#######[/bold blue][bold magenta]+[/bold magenta][bold blue]#######[/bold blue][bold magenta]+[/bold magenta]
[bold magenta]+======++======++=+  +===+   +=+   +=++=+  +===++======++======+[/bold magenta]
"""

LOGO_SMALL = "[bold blue] SENTINEL[/bold blue] [dim]-- Agentic EDA Engine[/dim]"


def print_banner():
    """Print the full Sentinel banner."""
    console.print(LOGO)
    console.print("[dim]  Agentic EDA Engine for Fraud Modeling -- v0.1[/dim]\n")


def print_profile_summary(digest: dict):
    """Print a beautiful profile summary with stats cards."""
    ov = digest.get("overview", {})
    dq = digest.get("data_quality", {})
    cp = digest.get("column_priority", {})

    # Score color
    score = dq.get("score", 0)
    sc = "[bold green]" if score >= 80 else "[bold yellow]" if score >= 50 else "[bold red]"

    # Stats table
    table = Table(box=box.ROUNDED, border_style="blue", title="Dataset Profile",
                  title_style="bold blue", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Rows", f"[bold]{ov.get('rows', 0):,}[/bold]")
    table.add_row("Columns", f"[bold]{ov.get('columns', 0)}[/bold]")
    table.add_row("Tables", str(ov.get("tables", 1)))
    table.add_row("Target", ov.get("target", "none"))
    table.add_row("Quality", f"{sc}{score}/100[/]")
    table.add_row("Critical Issues", f"[red]{len(dq.get('critical', []))}[/red]")
    table.add_row("Warnings", f"[yellow]{len(dq.get('warnings', []))}[/yellow]")

    # Priority table
    ptable = Table(box=box.ROUNDED, border_style="magenta", title="Column Priority",
                   title_style="bold magenta", show_header=False, padding=(0, 2))
    ptable.add_column("Tier", style="bold")
    ptable.add_column("Columns")

    high = cp.get("high", [])
    med = cp.get("medium", [])
    drop = cp.get("drop_candidates", [])

    ptable.add_row("[green]High Signal[/green]", ", ".join(high[:8]) + (f" (+{len(high)-8})" if len(high) > 8 else ""))
    ptable.add_row("[yellow]Medium[/yellow]", ", ".join(med[:6]) + (f" (+{len(med)-6})" if len(med) > 6 else ""))
    if drop:
        ptable.add_row("[red]Drop[/red]", ", ".join(drop[:6]))

    console.print(Columns([table, ptable], padding=2))

    # Critical issues
    crits = dq.get("critical", [])
    if crits:
        console.print()
        for c in crits[:5]:
            console.print(f"  [red]![/red] [bold]{c.get('column', '?')}[/bold]: {c.get('issue', '')}")

    # Broken joins
    joins = digest.get("relationships", {}).get("join_health", [])
    broken = [j for j in joins if j.get("status") == "broken"]
    if broken:
        console.print()
        for j in broken:
            console.print(f"  [red][/red] Broken join: {j['from_col']} -> {j['to_table']}.{j['to_col']}")


def print_eda_start(model_name: str, playbook: str | None = None):
    """Print the EDA session start banner."""
    console.print(f"\n{LOGO_SMALL}")
    console.print(f"  [dim]LLM:[/dim] [bold]{model_name}[/bold]")
    if playbook:
        console.print(f"  [dim]Playbook:[/dim] [bold]{playbook}[/bold]")
    console.print()


def print_eda_complete(succeeded: int, total: int, failed: int,
                       elapsed_ms: float, charts: int, report_path: str = ""):
    """Print the EDA completion summary."""
    console.print()
    panel_text = (
        f"[bold green]OK EDA Complete[/bold green]\n\n"
        f"  Analyses:  [bold]{succeeded}[/bold]/{total} succeeded"
        f"{'  [red](' + str(failed) + ' failed)[/red]' if failed else ''}\n"
        f"  Runtime:   [bold]{elapsed_ms/1000:.1f}s[/bold]\n"
        f"  Charts:    [bold]{charts}[/bold] generated"
    )
    if report_path:
        panel_text += f"\n  Report:    [bold blue]{report_path}[/bold blue]"
    console.print(Panel(panel_text, border_style="green", box=box.DOUBLE))


def print_compute_decision(profile):
    """Dramatic display of compute routing -- makes GPU detection feel important."""
    from rich.table import Table
    from rich import box as bx
    import time as _t

    console.print()
    console.print("[bold cyan] Hardware Detection[/bold cyan]")
    _t.sleep(0.3)

    # GPU scan
    if profile.gpu_count == 0:
        console.print("  [dim]Scanning for GPUs...[/dim]")
        _t.sleep(0.2)
        console.print("  [yellow]! No CUDA GPUs detected[/yellow]")
    else:
        console.print("  [dim]Scanning for GPUs...[/dim]")
        _t.sleep(0.2)
        for i, (name, mem) in enumerate(zip(profile.gpu_names, profile.gpu_memory_gb)):
            console.print(f"  [green]OK GPU {i}:[/green] [bold]{name}[/bold] -- {mem:.0f} GB")
            _t.sleep(0.15)

    # Data size
    if profile.data_size_gb > 0:
        _t.sleep(0.2)
        console.print(f"  [dim]Data size:[/dim] [bold]{profile.data_size_gb:.1f} GB[/bold]")

    # Decision
    _t.sleep(0.3)
    console.print()

    decision_table = Table(box=bx.HEAVY_EDGE, border_style="cyan",
                           title=" Compute Strategy", title_style="bold cyan",
                           show_header=False, padding=(0, 2))
    decision_table.add_column("", style="bold")
    decision_table.add_column("")

    # Honesty: the current cuDF path reads on GPU then converts to pandas for
    # compute, so the compute engine is pandas either way. Don't advertise
    # full "CUDF" acceleration that isn't happening.
    if profile.backend_choice == "cudf":
        engine_label = "[bold green]cuDF read -> pandas compute[/]"
    else:
        engine_label = "[bold yellow]PANDAS[/]"
    decision_table.add_row("Data Engine", engine_label)
    llm_dev = profile.llm_device or "API (no GPU reserved)"
    decision_table.add_row("LLM Device", f"[bold]{llm_dev}[/bold]")
    decision_table.add_row("Data Device", f"[bold]{profile.data_device}[/bold]")
    free = getattr(profile, "compute_gpus", None)
    if free:
        decision_table.add_row("Compute GPUs", f"[bold]{free}[/bold]")
    decision_table.add_row("Reason", f"[dim]{profile.reason}[/dim]")

    console.print(decision_table)
    console.print()


def print_chart_live(chart_path: str, title: str = "", chart_type: str = ""):
    """Show a prominent chart notification in the terminal."""
    from pathlib import Path as P
    p = P(chart_path)
    size_kb = p.stat().st_size / 1024 if p.exists() else 0

    icon_map = {
        "histogram": "", "bar": "", "line": "",
        "scatter": "", "heatmap": "", "box": "",
    }
    icon = icon_map.get(chart_type, "")

    panel_text = f"{icon} [bold]{title or chart_type or 'Chart'}[/bold]\n"
    panel_text += f"   [dim]Saved: {chart_path} ({size_kb:.0f} KB)[/dim]"

    console.print(Panel(panel_text, border_style="green", box=box.ROUNDED,
                        title="[bold green]Chart Generated[/bold green]"))


def print_decision(icon: str, label: str, detail: str, style: str = "cyan"):
    """Show a single decision point dramatically."""
    console.print(f"  [{style}]{icon} {label}:[/{style}] {detail}")


def print_agent_phase(phase_num: int, total: int, name: str):
    """Beautiful phase separator for playbook runs."""
    bar = "-" * 60
    console.print(f"\n[bold cyan]{bar}[/bold cyan]")
    console.print(f"[bold cyan]  Phase {phase_num}/{total}: {name}[/bold cyan]")
    console.print(f"[bold cyan]{bar}[/bold cyan]\n")
