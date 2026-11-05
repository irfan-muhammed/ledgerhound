"""
ledgerhound.shell.tools
=======================

The analyst's tool layer: session state plus every tool the model can call,
with HITL gating and guardrails, and no LLM. `AnalystTools` can be driven by
the conversation loop (ConversationalAnalyst) or called directly -- the
shape the MCP server needs, where the host agent is the brain.

    base         AnalystToolsBase    state, registry, gated dispatch, audit log
    data         DataToolsMixin      find/load/profile data, kernel, dataset cards
    python_exec  PythonToolsMixin    run_python: gates, execution, checks
    findings     FindingsToolsMixin  findings, decisions, chapters, recall
    reporting    ReportingToolsMixin reports, notebooks, feature catalog
    workflow     WorkflowToolsMixin  methodology, steps, phases, goal, ask_user
    roles        RolesToolsMixin     column roles: propose, confirm, enforce
    knowledge    KnowledgeToolsMixin the Knowledge Pack: kb_add, kb_search
    sql          SqlToolsMixin       read-only DuckDB over df, in the kernel
    sample       SampleToolsMixin    build_sample + the construction certificate
    specs        TOOL_SPECS          the schema of every tool
    common       helpers shared with the loop
"""

from ledgerhound.shell.tools.base import AnalystToolsBase
from ledgerhound.shell.tools.data import DataToolsMixin
from ledgerhound.shell.tools.findings import FindingsToolsMixin
from ledgerhound.shell.tools.knowledge import KnowledgeToolsMixin
from ledgerhound.shell.tools.python_exec import PythonToolsMixin
from ledgerhound.shell.tools.reporting import ReportingToolsMixin
from ledgerhound.shell.tools.roles import RolesToolsMixin
from ledgerhound.shell.tools.specs import TOOL_SPECS
from ledgerhound.shell.tools.sample import SampleToolsMixin
from ledgerhound.shell.tools.sql import SqlToolsMixin
from ledgerhound.shell.tools.workflow import WorkflowToolsMixin


class AnalystTools(DataToolsMixin, PythonToolsMixin, FindingsToolsMixin,
                   ReportingToolsMixin, WorkflowToolsMixin, RolesToolsMixin,
                   KnowledgeToolsMixin, SqlToolsMixin, SampleToolsMixin,
                   AnalystToolsBase):
    """A complete analyst session without an LLM: construct it, then call
    `_run_tool(name, args)` (gated, logged) for any tool in TOOL_SPECS."""


__all__ = ["AnalystTools", "AnalystToolsBase", "TOOL_SPECS"]
