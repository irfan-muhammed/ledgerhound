"""`ledgerhound kb ...`: manage the Knowledge Pack from the terminal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

kb_app = typer.Typer(
    name="kb",
    help="Knowledge Pack: your typologies, rules, dictionary, known findings, "
         "label policies and documents.",
    no_args_is_help=True,
)

_SCOPE = typer.Option("project", "--scope", "-s", help="project or team")


def _pack():
    from ledgerhound.kb import KnowledgePack
    return KnowledgePack()


@kb_app.command("init")
def kb_init(scope: str = _SCOPE) -> None:
    """Create the scope's folder (project: ./.ledgerhound/kb)."""
    from ledgerhound.kb import KBError
    try:
        d = _pack().init(scope)
    except KBError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{scope} knowledge pack ready at {d}")


@kb_app.command("add")
def kb_add(path: Path = typer.Argument(..., help="YAML/JSON entries, or a .md/.txt/.pdf document"),
           scope: str = _SCOPE,
           title: Optional[str] = typer.Option(None, help="document title (documents only)"),
           source: Optional[str] = typer.Option(None, help="where the document came from")) -> None:
    """Add entries from a YAML/JSON file, or one document."""
    from ledgerhound.kb import KBError
    from ledgerhound.kb.pack import DOC_EXTS
    from ledgerhound.kb.schema import describe
    pack = _pack()
    try:
        if path.suffix.lower() in DOC_EXTS:
            e = pack.add_document(path, scope=scope, title=title or "", source=source or "",
                                  added_by="cli")
            typer.echo(f"added {describe(e.data)} to the {scope} pack ({e['chars']:,} characters)")
            return
        entries = pack.read_entry_file(path)
    except (KBError, OSError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    failed = 0
    for i, raw in enumerate(entries, 1):
        try:
            e, replaced = pack.add(raw, scope=scope, added_by="cli")
            typer.echo(f"{'replaced' if replaced else 'added'} {e['type']} {e['id']}")
        except KBError as err:
            failed += 1
            typer.echo(f"entry {i}: {err}", err=True)
    if failed:
        raise typer.Exit(1)


@kb_app.command("list")
def kb_list(type: Optional[str] = typer.Option(None, "--type", "-t"),
            scope: Optional[str] = typer.Option(None, "--scope", "-s")) -> None:
    """Effective entries (most specific scope wins)."""
    from ledgerhound.kb.schema import describe
    pack = _pack()
    rows = pack.entries(type=type, scope=scope)
    for e in sorted(rows, key=lambda e: (e["type"], e["id"])):
        typer.echo(f"[{e.scope}] {e['type']}/{e['id']}: {describe(e.data)[:160]}")
    typer.echo(f"{len(rows)} entr{'y' if len(rows) == 1 else 'ies'}")
    for err in pack.load_errors:
        typer.echo(f"skipped: {err}", err=True)


@kb_app.command("validate")
def kb_validate(path: Optional[Path] = typer.Argument(None, help="a file to check before adding; "
                                                       "omit to check every scope")) -> None:
    """Check entries against the schema. Exit code 1 on any error."""
    from ledgerhound.kb import validate
    if path is not None:
        bad = 0
        for i, raw in enumerate(_pack().read_entry_file(path), 1):
            _, errs = validate(raw)
            for err in errs:
                bad += 1
                typer.echo(f"entry {i}: {err}")
        typer.echo("ok" if not bad else f"{bad} problem(s)")
        raise typer.Exit(1 if bad else 0)
    pack = _pack()
    n = len(pack.entries(effective=False))
    for err in pack.load_errors:
        typer.echo(err)
    typer.echo(f"{n} valid entr{'y' if n == 1 else 'ies'}; {len(pack.load_errors)} problem(s)")
    raise typer.Exit(1 if pack.load_errors else 0)
