"""
ledgerhound.agents.tool_registry
==============================

The canonical catalog of engine tools an agent may be granted.

The tool whitelist in an ``agent_def/v1`` is the **security boundary**: an
agent definition carries no executable code, only the names of tools the engine
already implements. Validating a spec's ``tools`` against this catalog is what
guarantees a user-authored agent can't reach anything the engine doesn't
sanction (HITL-on-Trino etc. still hold regardless of author).

This mirrors the tool families wired into ``EDAAgent`` today; keep them in sync
when a new tool family is added to the loop.
"""

from __future__ import annotations

# name -> one-line description. Grouped by family in comments.
TOOL_CATALOG: dict[str, str] = {
    # core execution
    "execute_python": "Run Python against the loaded DataFrame in the sandbox",
    "generate_chart": "Render a chart from code via the Visualizer",
    "ask_user": "Human-in-the-loop clarification (auto-timeout)",
    # findings
    "record_finding": "Record a structured, evidence-backed finding",
    "list_findings": "List findings recorded so far",
    # Schema KB (dict_*)
    "dict_lookup": "Look up a column's KB entry",
    "dict_search": "Search the Schema KB",
    "dict_quality_flags": "Quality flags for a column",
    "dict_decode_code": "Decode a coded value",
    "dict_aliases": "Aliases for a column",
    "dict_related": "Columns related to a given one",
    "dict_stats": "Schema KB coverage stats",
    # Phase 2 (fraud-specific)
    "psi_certify": "Certify PSI stability of a feature",
    "mine_hard_negatives": "Mine hard-negative non-fraud examples",
    "cluster_typologies": "Cluster fraud typologies",
    "validate_consistency": "Cross-dataset consistency check",
    # V3-018 built this and wired it into Phase2Tools/TOOL_NAMES/the
    # eda_agent.py system prompt, but never added it here -- so no
    # agent_def/v1 spec (comparison, feature_engineering, orchestrator, ...)
    # could ever be granted it; AgentRegistry.validate() would reject the
    # spec outright. Found via V3-032 while wiring it into
    # feature_engineering.yaml for real.
    "check_leakage": "Vet a proposed feature set for target leakage (Leakage Sentinel)",
    "tier_check": "Compute-tier classification check",
    "deposit_insight": "Deposit an insight into institutional memory",
    "measure_signal_decay": "Measure decay of a signal over time",
    "track_typology_evolution": "Track how a typology evolves",
    "detect_mule_motifs": "Detect mule transaction motifs",
    "compute_account_centrality": "Graph centrality of accounts",
    "find_mule_rings": "Find mule rings in the transaction graph",
    "trace_money_flow": "Trace money flow between accounts",
    "fingerprint_accounts": "Compute account behavioural fingerprints",
    # Phase 2 analyst primitives (fluid loop toolbox)
    "save_output": "Save the report in a server-supported format (md|ipynb|html|pdf), asking the user",
    "profile": "Deterministically profile a dataset into a digest (no LLM)",
    "query_sql": "Run SQL against a source (Trino/etc.), HITL-gated",
    "recall_memory": "Recall facts/lessons/episodes from the Librarian memory",
    "save_lesson": "Persist a procedural lesson to memory",
    "check_compute": "Report GPU/compute availability and big-data split advice",
    "spawn_subagent": "Delegate a bounded objective to a specialist agent in isolated context",
    # workflow integrations (shell tool families; writes are HITL confirm-gated)
    "configure_integration": "Save a config value/secret into the project's unified config",
    "show_config": "Show the project integration config (secrets masked)",
    "git_status": "Show working-tree status of the project repo",
    "git_log": "Show recent commits",
    "git_diff": "Show a diffstat of changes",
    "git_today": "Show today's commits by the current user",
    "git_pull": "Pull latest changes from the remote (confirm-gated)",
    "git_commit_push": "Stage, commit (auto-drafted message) and push (confirm-gated)",
    "jira_pending_tasks": "List my open Jira issues",
    "jira_post_update": "Post a work update comment/worklog to a Jira issue (confirm-gated)",
    "jira_create_issue": "Create a Jira issue (confirm-gated)",
    "jira_transition": "Move a Jira issue to a new status (confirm-gated)",
    "session_digest": "Summarise this session's work (findings + actions + commits)",
}

# Output contracts the engine knows how to consume / validate.
KNOWN_CONTRACTS: frozenset[str] = frozenset({
    "findings/v1",
    "profile/v1",
    "sampling_plan/v1",
    "discriminator_report/v1",
    "feature_recipe/v1",
    "typology_signature/v1",
    "study_report/v1",
})


def is_known_tool(name: str) -> bool:
    return name in TOOL_CATALOG


def unknown_tools(tools: list[str]) -> list[str]:
    """Return the subset of ``tools`` not present in the catalog."""
    return [t for t in tools if t not in TOOL_CATALOG]
