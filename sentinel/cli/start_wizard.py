"""
sentinel.cli.start_wizard
=========================

Interactive experiment wizard launched by ``sentinel start``.

Flow
----
1.  Welcome banner
2.  Choose: new experiment (with data) / new experiment (no data yet) / resume
3.  (New-data)    Collect dataset path(s) -- file or folder
3b. (No-data)     Describe what data you are expecting
4.  User describes the dataset; LLM reads profile and asks clarifying questions
5.  What are we doing? (goal selection)
6.  Report format
7.  Experiment name
8.  Creates workspace folder tree + writes session.json
9.  Auto-runs sentinel profile (non-LLM) for each dataset
10. Launches sentinel auto with goal + all context wired up
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich import box

from sentinel.session.models import Session
from sentinel.session.store import SessionStore

console = Console()

_DATA_EXTS = {".parquet", ".csv", ".tsv", ".pq"}


# ===========================================================================
# Entry point
# ===========================================================================

def run_wizard(workspace: str = "sentinel_workspace") -> None:
    _print_welcome()
    store = SessionStore(workspace)
    existing = store.list()

    choice = _ask_start_choice(existing)

    if choice == "continue":
        session = _pick_session(existing)
        _resume_session(session)
    else:
        require_data = (choice == "new_data")
        session = _new_session_wizard(workspace, require_data=require_data)
        _launch_session(session)


# ===========================================================================
# Welcome
# ===========================================================================

def _print_welcome() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]Sentinel AI[/bold cyan]  --  Agentic EDA & Fraud Modelling Engine\n\n"
        "[dim]Self-healing analysis loop | Sandboxed execution | Auditable findings\n"
        "H100 / cuDF ready  |  Kimi / vLLM / OpenAI compatible[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


# ===========================================================================
# Start choice
# ===========================================================================

def _ask_start_choice(existing: list[Session]) -> str:
    console.print("[bold]Where do we start?[/bold]\n")
    console.print("  [cyan][1][/cyan]  New experiment -- use an existing dataset or folder")
    console.print("  [cyan][2][/cyan]  New experiment -- no data yet, describe from scratch")
    if existing:
        console.print(
            f"  [cyan][3][/cyan]  Continue a previous session  "
            f"[dim]({len(existing)} available)[/dim]"
        )
    console.print()

    valid = ["1", "2", "3"] if existing else ["1", "2"]
    while True:
        raw = Prompt.ask("Choose", choices=valid, default="1")
        if raw == "3" and not existing:
            console.print("[yellow]No previous sessions found.[/yellow]")
            continue
        break

    return {"1": "new_data", "2": "new_scratch", "3": "continue"}[raw]


# ===========================================================================
# New session wizard
# ===========================================================================

def _new_session_wizard(workspace: str, require_data: bool = True) -> Session:
    # Step 1: LLM config -- needed early so enrichment can fire
    console.print(Rule("[bold]Step 1 -- LLM Config[/bold]"))
    llm_config = _ask_llm_config()
    llm = _load_llm(llm_config) if llm_config else None

    # Step 2: Dataset(s)
    console.print(Rule("[bold]Step 2 -- Dataset[/bold]"))
    datasets: list[str] = []
    if require_data:
        datasets = _collect_datasets()
    else:
        console.print(
            "[dim]No data file yet -- you can add datasets later.\n"
            "The AI will plan using your description only.[/dim]\n"
        )

    # Step 3: Describe + LLM enrichment
    console.print(Rule("[bold]Step 3 -- Dataset Context[/bold]"))
    if datasets:
        console.print(
            "[dim]Describe what this data is -- timeline, source, what it contains.\n"
            "The AI will read the actual column profile and ask follow-up questions.[/dim]\n"
        )
    else:
        console.print(
            "[dim]Describe the data you are expecting -- source, schema, timeline.\n"
            "The AI will ask clarifying questions based on your description.[/dim]\n"
        )
    description = Prompt.ask("Your description")

    digest_summaries: list[str] = []
    if datasets:
        console.print("\n[dim]Profiling data for context...[/dim]")
        digest_summaries = _quick_profile_all(datasets)

    enrichment, clarifications = _llm_enrich_dataset(
        description, digest_summaries, datasets, llm
    )

    # Step 4: Goal
    console.print(Rule("[bold]Step 4 -- Goal[/bold]"))
    goal, goal_detail = _ask_goal()

    # Step 5: Report format
    console.print(Rule("[bold]Step 5 -- Report Format[/bold]"))
    report_format = _ask_report_format()

    # Step 6: Experiment name
    console.print(Rule("[bold]Step 6 -- Experiment Name[/bold]"))
    from datetime import date
    stem = Path(datasets[0]).stem.replace("_", "-")[:20] if datasets else goal
    default_name = f"{stem}-{goal}-{date.today().isoformat()}"
    name = Prompt.ask("Experiment name", default=default_name)

    # Create session
    session = Session.new(name, workspace)
    session.description = description
    session.llm_enrichment = enrichment
    session.goal = goal
    session.goal_detail = goal_detail
    session.report_format = report_format
    session.llm_config = llm_config

    for ds_path in datasets:
        p = Path(ds_path)
        rows, cols = _quick_shape(p)
        entry = session.add_dataset(str(p), rows=rows, cols=cols)
        entry.clarifications = clarifications
        row_str = f"{rows:,} rows x {cols} cols" if rows else ""
        console.print(f"  [green]OK[/green] {p.name}  [dim]{row_str}[/dim]")

    # Log wizard conversation to session message history
    session.append_message("user", description, tag="wizard")
    if enrichment:
        session.append_message("assistant", enrichment, tag="wizard")
    for q, a in clarifications.items():
        session.append_message("assistant", q, tag="wizard")
        session.append_message("user", a, tag="decision")

    session.save()

    console.print()
    console.print(Panel(
        f"[bold]Experiment:[/bold] {session.name}\n"
        f"[bold]Goal:[/bold]       {session.goal_label()}\n"
        f"[bold]Datasets:[/bold]   {len(session.datasets)} file(s)\n"
        f"[bold]Reports:[/bold]    {session.report_format}\n"
        f"[bold]Workspace:[/bold]  {session.workspace_dir}",
        title="[bold cyan]Session Created[/bold cyan]",
        border_style="green",
    ))

    return session


# ===========================================================================
# Dataset collection
# ===========================================================================

def _collect_datasets() -> list[str]:
    console.print(
        "[dim]Enter a path to a single file (.csv / .parquet) or a folder.\n"
        "If a folder is given, all data files inside will be listed.[/dim]\n"
    )
    while True:
        raw = Prompt.ask("Dataset path").strip()
        p = Path(raw)
        if not p.exists():
            console.print(f"[red]Path not found: {raw}[/red]")
            continue
        if p.is_file():
            if p.suffix.lower() not in _DATA_EXTS:
                console.print(
                    f"[yellow]Unrecognised extension {p.suffix} -- "
                    f"accepted: {', '.join(_DATA_EXTS)}[/yellow]"
                )
                if not Confirm.ask("Use it anyway?", default=False):
                    continue
            return [str(p)]
        # Folder
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in _DATA_EXTS)
        if not files:
            console.print(f"[yellow]No data files found in {p}[/yellow]")
            continue
        _show_file_table(files)
        choice = Prompt.ask(
            "Use [bold]all[/bold] files, comma-separated numbers, or [bold]back[/bold]",
            default="all",
        )
        if choice.lower() == "back":
            continue
        if choice.lower() == "all":
            return [str(f) for f in files]
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [str(files[i]) for i in indices if 0 <= i < len(files)]
            if selected:
                return selected
        except (ValueError, IndexError):
            console.print("[red]Invalid selection -- try again[/red]")


def _show_file_table(files: list[Path]) -> None:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    t.add_column("#", style="dim", width=4)
    t.add_column("File")
    t.add_column("Size", justify="right")
    for i, f in enumerate(files, 1):
        try:
            size = f"{f.stat().st_size / 1_048_576:.1f} MB"
        except OSError:
            size = "?"
        t.add_row(str(i), f.name, size)
    console.print(t)


def _quick_shape(path: Path) -> tuple[int | None, int | None]:
    """Return (rows, cols) without loading the full frame."""
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            return meta.num_rows, meta.num_columns
        elif path.suffix.lower() in {".csv", ".tsv"}:
            import pandas as pd
            header_df = pd.read_csv(path, nrows=0)
            with open(path, "rb") as f:
                nlines = sum(1 for _ in f) - 1  # subtract header
            return nlines, len(header_df.columns)
    except Exception:
        pass
    return None, None


# ===========================================================================
# LLM helpers
# ===========================================================================

def _load_llm(llm_config: str):
    try:
        from sentinel.llm.client import LLMClient
        return LLMClient.from_config(llm_config)
    except Exception as e:
        console.print(
            f"[yellow]Could not load LLM ({e}) -- enrichment will be skipped[/yellow]"
        )
        return None


def _quick_profile_all(dataset_paths: list[str]) -> list[str]:
    """Return compact column/stat summaries for each dataset (no LLM)."""
    summaries = []
    for path in dataset_paths:
        p = Path(path)
        try:
            import pandas as pd
            if p.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, nrows=50_000)

            numeric = df.select_dtypes("number").columns.tolist()
            cat     = df.select_dtypes("object").columns.tolist()
            missing = {
                c: round(df[c].isna().mean() * 100, 1)
                for c in df.columns if df[c].isna().any()
            }

            lines = [
                f"File: {p.name}",
                f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols",
                f"Numeric columns ({len(numeric)}): {numeric[:15]}",
                f"Categorical columns ({len(cat)}): {cat[:15]}",
            ]
            if missing:
                top = sorted(missing.items(), key=lambda x: -x[1])[:5]
                lines.append(
                    "Missing values: " + ", ".join(f"{c} ({v}%)" for c, v in top)
                )

            date_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in ("date", "dt", "time", "ts", "asdt"))
            ]
            if date_cols:
                col = date_cols[0]
                try:
                    dates = pd.to_datetime(df[col], errors="coerce").dropna()
                    lines.append(
                        f"Date column '{col}': {dates.min().date()} to {dates.max().date()}"
                    )
                except Exception:
                    pass

            summaries.append("\n".join(lines))
            console.print(f"  [green]OK[/green] {p.name} profiled")
        except Exception as exc:
            summaries.append(f"File: {p.name} -- could not profile: {exc}")
            console.print(f"  [yellow]  {p.name} -- {exc}[/yellow]")
    return summaries


_ENRICH_PROMPT = """\
You are a senior data scientist reviewing a new dataset before an analysis session.

The user described their dataset as:
"{description}"

Here is what we found by profiling the actual data:
{profile}

Your job:
1. Write a 3-5 sentence factual summary of what you actually see in the data.
   Use exact column names, the real date range, actual row counts, and flag
   anything unusual (bimodal distributions, suspiciously high missing rates, etc).

2. Ask exactly 2-3 targeted clarifying questions whose answers will materially
   change how you analyse this data.
   Good: "Is 'riskscore' a pre-computed model score or a raw rule output?"
   Bad:  vague, generic, or already answered by the description.

Format your response as:
SUMMARY:
<your factual summary>

QUESTIONS:
1. <question>
2. <question>
3. <question (optional)>
"""


def _llm_enrich_dataset(
    description: str,
    digest_summaries: list[str],
    dataset_paths: list[str],
    llm,
) -> tuple[str, dict]:
    """Returns (enrichment_text, {question: answer}) or ('', {}) if no LLM."""
    if not llm:
        console.print("[dim]No LLM configured -- skipping dataset enrichment.[/dim]")
        return "", {}

    profile_text = "\n\n".join(digest_summaries) or "No data file loaded yet."
    prompt = _ENRICH_PROMPT.format(
        description=description,
        profile=profile_text[:3000],
    )

    console.print("\n[dim]AI is reviewing your data...[/dim]")
    try:
        resp = llm.generate([{"role": "user", "content": prompt}])
        enrichment = resp.content.strip()
    except Exception as exc:
        console.print(f"[yellow]Enrichment failed: {exc}[/yellow]")
        return "", {}

    # Parse SUMMARY + QUESTIONS
    summary = ""
    questions: list[str] = []
    section = None
    for line in enrichment.splitlines():
        if line.startswith("SUMMARY:"):
            section = "summary"
            continue
        elif line.startswith("QUESTIONS:"):
            section = "questions"
            continue
        if section == "summary":
            summary += line + "\n"
        elif section == "questions":
            stripped = line.lstrip("0123456789. ").strip()
            if stripped:
                questions.append(stripped)

    summary = summary.strip()
    questions = questions[:3]

    console.print()
    console.print(Panel(
        summary or enrichment[:500],
        title="[bold cyan]What I see in your data[/bold cyan]",
        border_style="cyan",
    ))

    clarifications: dict = {}
    if questions:
        console.print("\n[bold]A few clarifying questions:[/bold]\n")
        for q in questions:
            console.print(f"  [cyan]?[/cyan] {q}")
            answer = Prompt.ask("  Your answer")
            clarifications[q] = answer
            console.print()

    full_enrichment = summary
    if clarifications:
        full_enrichment += "\n\nClarifications:\n" + "\n".join(
            f"Q: {q}\nA: {a}" for q, a in clarifications.items()
        )

    return full_enrichment, clarifications


# ===========================================================================
# Goal / format / LLM config prompts
# ===========================================================================

def _ask_goal() -> tuple[str, str]:
    console.print("[dim]What are we doing with this data?[/dim]\n")
    options = [
        ("1", "eda",             "Exploratory Data Analysis -- distributions, drift, correlations, outliers"),
        ("2", "fraud_modelling", "Fraud Modelling -- PSI sampling, hard negatives, feature signal, scorecard"),
        ("3", "mule_graph",      "Mule Detection -- graph build, community detection, motif analysis"),
        ("4", "sampling",        "Data Sampling / PSI Certification only"),
        ("5", "custom",          "Custom -- I'll describe it"),
    ]
    for num, _, label in options:
        console.print(f"  [cyan][{num}][/cyan]  {label}")
    console.print()
    raw = Prompt.ask("Choose", choices=[o[0] for o in options], default="1")
    _, goal, _ = options[int(raw) - 1]
    detail = ""
    if goal == "custom":
        detail = Prompt.ask("Describe what you want to do")
    return goal, detail


def _ask_report_format() -> str:
    console.print("[dim]What format should the final report be in?[/dim]\n")
    console.print("  [cyan][1][/cyan]  HTML  -- shareable, self-contained")
    console.print("  [cyan][2][/cyan]  Jupyter Notebook  -- re-runnable cells")
    console.print("  [cyan][3][/cyan]  Both  [dim](recommended)[/dim]")
    console.print("  [cyan][4][/cyan]  Markdown  -- plain text, good for git")
    console.print()
    raw = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="3")
    return {"1": "html", "2": "jupyter", "3": "both", "4": "markdown"}[raw]


def _ask_llm_config() -> str:
    console.print(
        "[dim]Path to your LLM config YAML (e.g. configs/gateway.yaml).\n"
        "Press Enter to skip -- you can set this later.[/dim]\n"
    )
    while True:
        raw = Prompt.ask("LLM config path", default="").strip()
        if not raw:
            return ""
        p = Path(raw)
        if p.exists():
            return str(p)
        console.print(f"[yellow]File not found: {raw}[/yellow]")
        if not Confirm.ask("Use it anyway (path may not exist yet)?", default=False):
            continue
        return raw


# ===========================================================================
# Continue session
# ===========================================================================

def _pick_session(sessions: list[Session]) -> Session:
    console.print(Rule("[bold]Previous Sessions[/bold]"))
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    t.add_column("#", style="dim", width=4)
    t.add_column("Name")
    t.add_column("Goal")
    t.add_column("Datasets", justify="right")
    t.add_column("Status")
    t.add_column("Created")
    for i, s in enumerate(sessions, 1):
        t.add_row(
            str(i),
            s.name[:40],
            s.goal_label()[:30],
            str(len(s.datasets)),
            s.status,
            s.created_at[:10],
        )
    console.print(t)
    console.print()
    valid = [str(i) for i in range(1, len(sessions) + 1)]
    raw = Prompt.ask("Select session number", choices=valid)
    return sessions[int(raw) - 1]


def _resume_session(session: Session) -> None:
    console.print()
    console.print(Panel(
        f"[bold]Resuming:[/bold]  {session.name}\n"
        f"[bold]Goal:[/bold]      {session.goal_label()}\n"
        f"[bold]Status:[/bold]    {session.status}\n"
        f"[bold]Datasets:[/bold]  {', '.join(d.name for d in session.datasets)}\n"
        f"[bold]Progress:[/bold]  step {session.last_completed_step} complete\n"
        f"[bold]Workspace:[/bold] {session.workspace_dir}",
        title="[bold cyan]Session Resumed[/bold cyan]",
        border_style="cyan",
    ))
    if not session.llm_config:
        console.print()
        session.llm_config = _ask_llm_config()
        session.save()
    _launch_session(session)


# ===========================================================================
# Profile + auto launch
# ===========================================================================

def _launch_session(session: Session) -> None:
    console.print()
    console.print(Rule("[bold cyan]Auto-profiling datasets (non-LLM)[/bold cyan]"))
    _auto_profile(session)

    goal_str = _build_goal_string(session)
    console.print()
    console.print(Panel(
        f"[bold]Goal handed to AI:[/bold]\n{goal_str}",
        border_style="cyan",
    ))

    if not session.llm_config:
        console.print(
            "\n[yellow]No LLM config set. Run manually when ready:[/yellow]\n"
            f"  sentinel auto run \"{goal_str}\" \\\n"
            f"    --data {session.datasets[0].path if session.datasets else '<data>'} \\\n"
            f"    --llm-config <your-config.yaml> \\\n"
            f"    --workspace {session.workspace_dir}"
        )
        return

    if not Confirm.ask("\nStart analysis now?", default=True):
        _print_manual_command(session, goal_str)
        return

    session.status = "running"
    session.save()

    from sentinel.auto.runner import AutoRunner, RunConfig

    cfg = RunConfig(
        goal=goal_str,
        llm_config=session.llm_config or None,
        data=session.datasets[0].path if session.datasets else None,
        workspace=session.workspace_dir,
        max_steps=20,
        max_retries=3,
        timeout=120,
        session=session,
    )
    try:
        runner = AutoRunner(cfg)
        run_dir = runner.run()
        session.last_run_id = run_dir.name
        session.status = "done"
        session.save()
        console.print(f"\n[bold green]OK Session complete:[/bold green] {run_dir}")
    except Exception as e:
        session.status = "error"
        session.save()
        console.print(f"\n[red]Run failed: {e}[/red]")
        raise


def _auto_profile(session: Session) -> None:
    """Run sentinel profile on each dataset that lacks a digest."""
    if not session.datasets:
        console.print("  [dim]No datasets to profile.[/dim]")
        return

    from sentinel.core import Sentinel
    from sentinel.session.file_index import FileIndex

    lib = FileIndex(session)

    for entry in session.datasets:
        lib.register(entry.path, "data", f"input dataset: {entry.name}")

    for entry in session.datasets:
        digest_path = session.digest_dir() / f"{entry.name}.digest.yaml"
        if digest_path.exists():
            console.print(f"  [dim]OK {entry.name} -- digest already exists[/dim]")
            entry.digest_path = str(digest_path)
            lib.register(
                digest_path, "digest", f"profile of {entry.name} dataset"
            )
            session.save()
            continue

        console.print(f"  Profiling [cyan]{entry.name}[/cyan]...", end=" ")
        try:
            engine = Sentinel(data=entry.path)
            digest = engine.profile()
            digest.save(str(digest_path))
            entry.digest_path = str(digest_path)
            if not entry.rows and hasattr(digest, "row_count"):
                entry.rows = digest.row_count
            lib.register(
                digest_path, "digest",
                f"profile of {entry.name} "
                f"({entry.rows or '?'} rows x {entry.cols or '?'} cols)",
            )
            session.save()
            console.print("[green]done[/green]")
        except Exception as exc:
            console.print(f"[yellow]skipped ({exc})[/yellow]")


def _build_goal_string(session: Session) -> str:
    names = ", ".join(d.name for d in session.datasets) or "the dataset"
    goal_map = {
        "eda": (
            f"Perform a thorough exploratory data analysis on {names}. "
            f"Cover distributions, missing values, outliers, correlations, "
            f"and any temporal drift patterns."
        ),
        "fraud_modelling": (
            f"Prepare {names} for fraud modelling: PSI-weighted sampling, "
            f"hard negative boundary identification, feature signal health, "
            f"and a sample certification scorecard."
        ),
        "mule_graph": (
            f"Build a mule detection graph from {names}: construct the "
            f"transaction network, detect communities, identify mule motifs, "
            f"and score accounts by centrality."
        ),
        "sampling": (
            f"Certify a representative sample from {names}: compute PSI vs "
            f"population, flag high-drift columns, and produce a sampling manifest."
        ),
        "custom": session.goal_detail or f"Analyse {names}.",
    }
    base = goal_map.get(session.goal, f"Analyse {names}.")
    if session.description:
        base += f" Dataset context: {session.description}."
    if session.llm_enrichment:
        base += f" Additional context from profiling: {session.llm_enrichment[:300]}."
    fmt_note = {
        "html":     "Produce an HTML report.",
        "jupyter":  "Produce a Jupyter notebook.",
        "both":     "Produce both an HTML report and a Jupyter notebook.",
        "markdown": "Produce a Markdown report.",
    }.get(session.report_format, "")
    if fmt_note:
        base += f" {fmt_note}"
    return base


def _print_manual_command(session: Session, goal_str: str) -> None:
    data_flag = f"--data {session.datasets[0].path}" if session.datasets else ""
    console.print(
        f"\n[dim]Run manually when ready:[/dim]\n"
        f"  sentinel auto run \"{goal_str}\" \\\n"
        f"    {data_flag} \\\n"
        f"    --llm-config {session.llm_config or '<config.yaml>'} \\\n"
        f"    --workspace {session.workspace_dir}"
    )
