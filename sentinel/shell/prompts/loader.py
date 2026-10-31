"""
sentinel.shell.prompts.loader
================================

Loads a prompt from a text file in this directory, falling back to a
code-embedded default when the file is missing, unreadable, or empty.

Why a fallback at all
-----------------------
A prompt file is now something a non-engineer could plausibly edit (or
break) without touching Python -- that's the entire point of externalizing
it. But this analyst is the product; it must never fail to start, or
silently run with an empty/garbled system prompt, because someone's editor
mangled encoding or a deploy step dropped a file. `load_prompt()` treats
the file as the PREFERRED source and the caller's embedded string as a
last-resort safety net, never the other way around -- so a missing or
broken file degrades to "exactly today's behaviour," not "undefined
behaviour."
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Prompt text now lives in ONE tree, sentinel/prompts/, split by audience:
#   prompts/shell/    -- the conversational analyst (system, tool protocol)
#   prompts/agentic/  -- the agent specs the EDAAgent engine runs (YAML)
#   prompts/skills/   -- the team's proven procedures
# This module keeps its old import path so nothing that imports
# `sentinel.shell.prompts.loader` breaks; it just reads from the new home.
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "shell"
# Legacy location (pre-2026-08-06), still searched so a stale checkout or a
# partially-applied deploy degrades to working rather than to an empty prompt.
_LEGACY_DIR = Path(__file__).parent


def load_prompt(name: str, fallback: str) -> str:
    """Return the contents of ``<this dir>/<name>.md``, or `fallback` if the
    file doesn't exist, can't be read, or is empty/whitespace-only.

    `name` has no extension (e.g. "system", not "system.md") so callers
    can't accidentally point outside this directory with a path.
    """
    if "/" in name or "\\" in name or ".." in name:
        logger.warning("load_prompt(%r): rejected as a path-like name, using fallback", name)
        return fallback
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        legacy = _LEGACY_DIR / f"{name}.md"
        if legacy.exists():
            logger.info("load_prompt(%r): using legacy location %s", name, legacy)
            path = legacy
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fallback
    except Exception as e:  # noqa: BLE001 -- a prompt file must never crash startup
        logger.warning("load_prompt(%r) failed (%s: %s), using the embedded fallback",
                       name, type(e).__name__, e)
        return fallback
    if not text.strip():
        logger.warning("load_prompt(%r): file is empty, using the embedded fallback", name)
        return fallback
    return text
