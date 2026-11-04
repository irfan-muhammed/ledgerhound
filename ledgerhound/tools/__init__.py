"""
ledgerhound.tools
==============

The unified tool layer: one ToolDef serves both agent surfaces --
the shell's native OpenAI function-calling loop and the ReAct
catalog -- with HITL gating applied at dispatch time.
"""

from ledgerhound.tools.registry import ToolDef, ToolRegistry

__all__ = ["ToolDef", "ToolRegistry"]
