"""
sentinel.agents.factory
=======================

Build a live agent from a declarative ``agent_def/v1`` spec (rank 3:
spec consumption). This is the bridge that turns "agents are data" into a
running agent: it reads an :class:`~sentinel.agents.spec.AgentSpec` (or looks
one up by name in the :class:`~sentinel.agents.registry.AgentRegistry`) and
constructs an :class:`~sentinel.agents.eda_agent.EDAAgent` configured by it.

The spec owns the *role* (``role_prompt``), the declared *tool whitelist*, the
*output contract*, and the *memory scope*. The engine owns every capability /
protocol section of the system prompt. Specialist agents (Profiler, Sampling,
Comparison, ...) and user-defined agents are all built through this one path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sentinel.agents.eda_agent import EDAAgent
from sentinel.agents.spec import AgentSpec

logger = logging.getLogger(__name__)


def agent_from_spec(spec: AgentSpec, *, llm, **deps: Any) -> EDAAgent:
    """Construct an :class:`EDAAgent` driven by ``spec``.

    ``deps`` are the usual EDAAgent dependencies (``backend``, ``knowledge``,
    ``digest``, ``findings_log``, ``schema_kb``, ``phase2_tools``,
    ``output_dir``, ...). The spec supplies the role prompt and is attached to
    the agent (``agent.spec``) so downstream code can read its tool whitelist /
    output contract / memory scope.
    """
    spec.validate(allow_unknown_tools=True)
    # NOTE: spec.defaults (e.g. max_steps, temperature) are behavioral knobs
    # for the loop/runner, NOT EDAAgent constructor args -- they stay on
    # agent.spec.defaults for the caller to read, and are not spread here.
    logger.debug("Building agent %r from spec (tools=%s, contract=%s)",
                 spec.name, spec.tools, spec.output_contract)
    return EDAAgent(llm=llm, spec=spec, **deps)


def load_agent(
    name: str,
    *,
    llm,
    registry: Optional[Any] = None,
    **deps: Any,
) -> EDAAgent:
    """Look ``name`` up in the agent registry and build it.

    Falls back to constructing a fresh :class:`AgentRegistry` (built-ins +
    the user's ``~/.sentinel/agents``) when none is passed. Raises
    ``KeyError`` if the agent name is unknown.
    """
    if registry is None:
        from sentinel.agents.registry import AgentRegistry
        registry = AgentRegistry().load()
    spec = registry.get(name)
    if spec is None:
        raise KeyError(
            f"unknown agent {name!r}; known: {sorted(registry.names())}"
        )
    return agent_from_spec(spec, llm=llm, **deps)
