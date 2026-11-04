"""
ledgerhound.integrations.jira.tools
================================

Jira as a tool family for the analyst shell. ``jira_pending_tasks`` is a
free read; posting updates, creating issues, and transitions are
confirm-gated. ``jira_post_update`` auto-drafts "today's work" from the
session digest when the model gives no body.
"""

from __future__ import annotations

from typing import Callable, Optional

from ledgerhound.integrations.jira.client import JiraClient, JiraError
from ledgerhound.tools.registry import ToolDef

DigestFn = Callable[[], str]  # -> markdown work update ("" if nothing)


def _fmt_issue(i: dict) -> str:
    f = i.get("fields", {})
    status = (f.get("status") or {}).get("name", "?")
    prio = (f.get("priority") or {}).get("name", "")
    due = f.get("duedate") or ""
    extra = " ".join(x for x in (f"[{status}]", prio and f"({prio})",
                                 due and f"due {due}") if x)
    return f"- {i.get('key')}: {f.get('summary', '')} {extra}".rstrip()


def build_jira_tools(client: JiraClient,
                     project_key: str = "",
                     digest_fn: Optional[DigestFn] = None) -> list[ToolDef]:

    def _guard(fn):
        def inner(**kw) -> str:
            if not client.configured():
                return ("Jira is not configured yet. I need base_url and email "
                        "(configure_integration section=jira). Ask the user for "
                        "their Jira URL, login email, and an API token.")
            try:
                return fn(**kw)
            except JiraError as e:
                return str(e)
        return inner

    @_guard
    def jira_pending_tasks(max_results: int = 15) -> str:
        issues = client.my_open_issues(max_results)
        if not issues:
            return "no open issues assigned to you -- clean slate."
        return "Your pending Jira issues:\n" + "\n".join(map(_fmt_issue, issues))

    def _draft(update: str) -> str:
        if update.strip():
            return update.strip()
        if digest_fn is not None:
            try:
                return digest_fn().strip()
            except Exception:  # noqa: BLE001
                pass
        return ""

    @_guard
    def jira_post_update(issue_key: str, update: str = "",
                         as_worklog: bool = False,
                         time_spent: str = "1h") -> str:
        body = _draft(update)
        if not body:
            return ("nothing to post: no update text given and the session has "
                    "no findings/actions yet.")
        if as_worklog:
            client.add_worklog(issue_key, body, time_spent)
            return f"logged work ({time_spent}) on {issue_key}."
        client.add_comment(issue_key, body)
        return f"posted update comment to {issue_key}."

    @_guard
    def jira_create_issue(summary: str, description: str = "",
                          issue_type: str = "Task",
                          project: str = "") -> str:
        key = project or project_key
        if not key:
            return ("no Jira project key configured -- set it via "
                    "configure_integration (section=jira, key=project_key) "
                    "or pass `project`.")
        res = client.create_issue(key, summary, description, issue_type)
        return f"created {res.get('key', '(no key returned)')}: {summary}"

    @_guard
    def jira_transition(issue_key: str, to_status: str) -> str:
        trans = client.transitions(issue_key)
        want = to_status.strip().lower()
        match = next((t for t in trans
                      if t.get("name", "").lower() == want), None)
        if match is None:  # fuzzy: substring either way
            match = next((t for t in trans
                          if want in t.get("name", "").lower()
                          or t.get("name", "").lower() in want), None)
        if match is None:
            names = ", ".join(t.get("name", "?") for t in trans) or "(none)"
            return (f"'{to_status}' is not a valid transition for {issue_key}. "
                    f"Valid: {names}")
        client.transition(issue_key, match["id"])
        return f"moved {issue_key} -> {match.get('name')}"

    def _confirm_update(a: dict) -> str:
        body = _draft(str(a.get("update", "")))
        kind = "worklog" if a.get("as_worklog") else "comment"
        return (f"Post this {kind} to {a.get('issue_key')}?\n---\n"
                f"{body[:600] or '(empty)'}")

    return [
        ToolDef(name="jira_pending_tasks",
                description="List the user's open Jira issues (assigned to "
                            "them, not Done). Use when asked 'what's pending'.",
                parameters={"type": "object", "properties": {
                    "max_results": {"type": "integer"}}, "required": []},
                fn=jira_pending_tasks),
        ToolDef(name="jira_post_update",
                description="Post a work update to a Jira issue as a comment "
                            "or worklog. Omit `update` to auto-draft from this "
                            "session's digest ('today's work').",
                parameters={"type": "object", "properties": {
                    "issue_key": {"type": "string"},
                    "update": {"type": "string"},
                    "as_worklog": {"type": "boolean"},
                    "time_spent": {"type": "string",
                                   "description": "e.g. 30m, 2h (worklog only)"}},
                    "required": ["issue_key"]},
                fn=jira_post_update, hitl="confirm",
                confirm_prompt=_confirm_update),
        ToolDef(name="jira_create_issue",
                description="Create a Jira issue in the configured project.",
                parameters={"type": "object", "properties": {
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "issue_type": {"type": "string",
                                   "description": "Task | Bug | Story"},
                    "project": {"type": "string",
                                "description": "project key override"}},
                    "required": ["summary"]},
                fn=jira_create_issue, hitl="confirm",
                confirm_prompt=lambda a: f"Create Jira issue "
                                         f"'{str(a.get('summary'))[:80]}'?"),
        ToolDef(name="jira_transition",
                description="Move a Jira issue to a new status (e.g. In "
                            "Progress, Done).",
                parameters={"type": "object", "properties": {
                    "issue_key": {"type": "string"},
                    "to_status": {"type": "string"}},
                    "required": ["issue_key", "to_status"]},
                fn=jira_transition, hitl="confirm",
                confirm_prompt=lambda a: f"Move {a.get('issue_key')} to "
                                         f"'{a.get('to_status')}'?"),
    ]
