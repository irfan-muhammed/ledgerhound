"""
ledgerhound.shell.conversation
==============================

The conversational analyst -- Claude-Code-style UX for data science. There is
NO wizard and NO hardcoded on-ramp: the LLM drives the whole conversation from
the first token via native function-calling, exactly like Claude Code drives a
coding session. The user talks naturally; the model decides when to ask, how to
resolve a data path, what analysis fits, and when it is done.

`ConversationalAnalyst` is assembled from three layers:

    ledgerhound.shell.loop        ConversationLoop -- the LLM turn loop
    ledgerhound.shell.tools       AnalystTools     -- session state + every tool
    ledgerhound.shell.guardrails  one class per guardrail, used by the tools

The tool layer works without the loop (no LLM needed); the loop reaches the
session only through the tool layer's registry and `_run_tool`.

Skills are offered as *reference the model may consult*, never force-injected --
so "EDA on fraud data" does EDA, not a railroaded fraud-vs-nonfraud procedure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ledgerhound.shell.loop import (  # noqa: F401 -- re-exported
    MAX_CONTINUATION_NUDGES,
    MAX_FIREWALL_REPLIES,
    MAX_STEPS,
    PROMPT_TOOL_PROTOCOL,
    STEP_CALL_BUDGET,
    SYSTEM_PROMPT,
    TURN_WALLCLOCK_BUDGET_S,
    ConversationLoop,
)
from ledgerhound.shell.tools import TOOL_SPECS, AnalystTools  # noqa: F401 -- re-exported
from ledgerhound.shell.tools.common import (  # noqa: F401 -- re-exported
    _failure_signature,
    _parse_tool_args,
    _read_any,
)


class ConversationalAnalyst(ConversationLoop, AnalystTools):
    """An LLM-driven analyst session. `send(text)` runs one user turn to its
    natural stopping point (a text reply), executing tool calls along the way."""

    def __init__(self, llm, project=None, out_root: str | Path = "run_outputs",
                 renderer: Callable[[str], None] = print, registry=None,
                 gate=None, tool_mode: str = "auto", autonomous: bool = False,
                 turn_budget_seconds: int = TURN_WALLCLOCK_BUDGET_S,
                 max_steps: int = MAX_STEPS, step_call_budget: int = STEP_CALL_BUDGET,
                 use_kernel: bool = False, compress_context: bool = True,
                 methodology: str = "", methodology_mode: str = ""):
        self.llm = llm
        AnalystTools.__init__(
            self, project=project, out_root=out_root, renderer=renderer,
            registry=registry, gate=gate, autonomous=autonomous,
            use_kernel=use_kernel, methodology=methodology,
            methodology_mode=methodology_mode)
        ConversationLoop.__init__(
            self, llm, tool_mode=tool_mode, turn_budget_seconds=turn_budget_seconds,
            max_steps=max_steps, step_call_budget=step_call_budget,
            compress_context=compress_context)
