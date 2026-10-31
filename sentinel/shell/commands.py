"""
sentinel.shell.commands
=======================

The slash-command registry for the shell REPL. Deliberately thin: commands
either do something locally (config display, model switch) or *inject* a
natural-language turn so the model stays the brain -- ``/jira`` just says
"show my pending Jira issues" for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SlashResult:
    handled: bool                 # True -> REPL should not call the model
    inject: str = ""              # non-empty -> send this text to the model
    quit: bool = False
    notice: str = ""              # printed by the REPL when non-empty


# Flags that mean "this line carries a credential". An UNKNOWN command is
# normally forwarded to the model as text -- which for a typo like
# `/mino --key abc123` would send the secret to the LLM and write it into
# the transcript and the hash-chained events.jsonl for good.
_SECRET_FLAGS = ("--key", "--secret", "--secret-key", "--access-key",
                 "--token", "--password", "--api-key")


Handler = Callable[[str], SlashResult]


class SlashRegistry:
    def __init__(self):
        self._cmds: dict[str, tuple[str, Handler]] = {}

    def register(self, name: str, help_text: str, handler: Handler) -> None:
        self._cmds[name] = (help_text, handler)

    def register_inject(self, name: str, help_text: str, turn: str) -> None:
        """A command that simply forwards a canned natural-language turn."""
        self.register(name, help_text,
                      lambda args, t=turn: SlashResult(
                          handled=False, inject=(t + (" " + args if args else ""))))

    def help_text(self) -> str:
        lines = ["Commands:"]
        for name in sorted(self._cmds):
            lines.append(f"  /{name:<10} {self._cmds[name][0]}")
        lines.append("  (anything else -- just talk; the model drives)")
        return "\n".join(lines)

    def dispatch(self, raw: str) -> Optional[SlashResult]:
        """None -> not a slash command; otherwise the result."""
        text = raw.strip()
        if text.lower() in ("exit", "quit"):
            return SlashResult(handled=True, quit=True)
        if not text.startswith("/"):
            return None
        name, _, args = text[1:].partition(" ")
        name = name.lower()
        if name in ("exit", "quit"):
            return SlashResult(handled=True, quit=True)
        entry = self._cmds.get(name)
        if entry is None and any(f in args.split() or args.startswith(f)
                                 or f"{f}=" in args for f in _SECRET_FLAGS):
            return SlashResult(
                handled=True,
                notice=(f"unknown command /{name} -- NOT sent to the model, "
                        "because it looks like it contains a secret. Did you "
                        "mean /minio? (/help lists the commands)"))
        if entry is None:
            return SlashResult(handled=True,
                               inject="",
                               quit=False) if name == "" else SlashResult(
                handled=False,
                inject=raw)  # unknown /cmd -> let the model interpret it
        return entry[1](args.strip())
