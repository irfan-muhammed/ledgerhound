"""
sentinel.tools.registry
=======================

One tool definition, every surface. A ``ToolDef`` carries the OpenAI
function schema, the callable, and the HITL gate level; a ``ToolRegistry``
turns a set of them into:

  - ``openai_specs()`` -- the ``tools=`` list for native function-calling
  - ``catalog()``      -- ``{name: description}`` for the ReAct tool catalog
  - ``dispatch()``     -- gated execution that returns error strings, never raises

Writes (``hitl="confirm"``) go through the injected ``HITLGate`` before the
callable runs; a decline is returned to the model as an ordinary tool result
so the conversation continues fluidly.
"""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

from sentinel.analyst.hitl import HITLGate

VALID_HITL = ("none", "confirm")


@dataclass
class ToolDef:
    """A tool the model may call, defined once for every agent surface."""
    name: str
    description: str
    parameters: dict                      # JSON schema {"type":"object",...}
    fn: Callable[..., str]                # kwargs -> result string
    hitl: str = "none"                    # "none" | "confirm"
    confirm_prompt: Optional[Callable[[dict], str]] = None

    def __post_init__(self) -> None:
        if self.hitl not in VALID_HITL:
            raise ValueError(f"tool {self.name!r}: hitl must be one of {VALID_HITL}")

    def openai_spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}

    def confirm_question(self, args: dict) -> str:
        if self.confirm_prompt is not None:
            try:
                return self.confirm_prompt(args)
            except Exception:  # noqa: BLE001
                pass
        return f"Run {self.name} with {args}?"


class ToolRegistry:
    """Holds ToolDefs; dispatches with HITL gating and error containment."""

    def __init__(self, gate: HITLGate | None = None):
        self.gate = gate or HITLGate()
        self._tools: dict[str, ToolDef] = {}
        # one line per successful call -- feeds the session digest
        self.action_log: list[str] = []

    # -- registration ------------------------------------------------------
    def register(self, tool: ToolDef) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolDef]) -> None:
        for t in tools:
            self.register(t)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    # -- surfaces ------------------------------------------------------------
    def openai_specs(self) -> list[dict]:
        return [t.openai_spec() for t in self._tools.values()]

    def catalog(self) -> dict[str, str]:
        return {t.name: t.description for t in self._tools.values()}

    # -- gated dispatch ------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name}"
        if tool.hitl == "confirm":
            decision = self.gate.confirm(tool.confirm_question(args))
            if not decision.approved:
                return (f"user declined {name} -- do not retry it; "
                        f"ask what they want instead.")
        # "bad arguments" is only for a call that does not fit the tool's
        # signature. A TypeError raised INSIDE the tool is an internal error
        # and must say so -- it used to be reported as "bad arguments" too,
        # which sent two investigations at the model's arguments when the
        # real fault was a type check deep in the tool (2026-10-05).
        try:
            inspect.signature(tool.fn).bind(**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except (ValueError, Exception):  # noqa: BLE001 -- no introspectable signature
            pass
        try:
            result = tool.fn(**args)
        except Exception as e:  # noqa: BLE001
            return f"error running {name}: {type(e).__name__}: {e}{_where(e)}"
        self.action_log.append(f"{name}: {_summ(args)}")
        return result if isinstance(result, str) else str(result)


def _where(exc: BaseException) -> str:
    """' [at file.py:123 in func]' for the innermost frame of a tool's
    exception, so an internal error is diagnosable from the message alone."""
    try:
        fr = traceback.extract_tb(exc.__traceback__)[-1]
        return f" [at {fr.filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{fr.lineno} in {fr.name}]"
    except Exception:  # noqa: BLE001
        return ""


def _summ(args: dict) -> str:
    parts = []
    for k, v in args.items():
        sv = str(v).replace("\n", " ")
        parts.append(f"{k}={sv[:60]}" + ("..." if len(sv) > 60 else ""))
    return ", ".join(parts) or "-"
