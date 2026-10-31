"""Agentic EDA -- ReAct loop with tool execution, plus the declarative
agent abstraction (agent_def/v1) + registry that lets agents be data, not code.
"""
from sentinel.agents.eda_agent import EDAAgent
from sentinel.agents.spec import AgentSpec, AgentValidationError
from sentinel.agents.registry import AgentRegistry

__all__ = ["EDAAgent", "AgentSpec", "AgentValidationError", "AgentRegistry"]
