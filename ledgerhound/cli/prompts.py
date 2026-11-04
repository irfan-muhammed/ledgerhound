"""
ledgerhound.cli.prompts
======================

The real, blocking terminal prompt used by ``ledgerhound shell``'s HITL gate --
extracted from a closure inside ``cli.main.shell()`` so it's independently
testable (V3-016's advisor review flagged that nothing exercised this path;
every test and live smoke used the non-interactive ``auto_yes`` default).

Serves BOTH the original ``confirm()``/``choose()`` binary gate (git/jira/
config integrations, options ``["yes", "no"]`` or similar) and V3-016's
high-tier decision packages (options ``["proceed", "cancel"]``). The
interrupt/empty-input fallback returns ``options[-1]`` -- the LAST option,
by convention always the safe/declining one -- rather than a hardcoded
string, so it generalizes correctly across both option sets instead of
silently mismatching one of them.
"""

from __future__ import annotations

from typing import Callable


def confirm_prompt(console, escape: Callable[[str], str],
                   question: str, options: list[str]) -> str:
    """Print `question`, block on real terminal input, return the answer.
    `escape` is `rich.markup.escape` -- `question`/output can embed a diff,
    commit message, or data containing `[...]`, which Rich would otherwise
    try to interpret as markup."""
    console.print(f"\n[yellow]confirm[/yellow] {escape(question)}")
    try:
        ans = console.input(f"[yellow]({'/'.join(options)})[/yellow] > ").strip()
    except (EOFError, KeyboardInterrupt):
        return options[-1] if options else "no"
    return ans or (options[-1] if options else "no")
