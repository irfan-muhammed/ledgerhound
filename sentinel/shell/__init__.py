"""
sentinel.shell
==============

The interactive shell layer. The primary entry is the LLM-driven
``ConversationalAnalyst`` (Claude-Code-style: the model drives the whole
conversation via native function-calling). Launched by ``sentinel shell``.
"""

from sentinel.shell.conversation import ConversationalAnalyst

__all__ = ["ConversationalAnalyst"]
