"""
sentinel.tools
==============

The unified tool layer: one ToolDef serves both agent surfaces --
the shell's native OpenAI function-calling loop and the ReAct
catalog -- with HITL gating applied at dispatch time.
"""

from sentinel.tools.registry import ToolDef, ToolRegistry

__all__ = ["ToolDef", "ToolRegistry"]
