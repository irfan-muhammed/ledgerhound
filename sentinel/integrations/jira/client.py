"""
sentinel.integrations.jira.client
=================================

Minimal Jira REST client -- stdlib urllib only (requests optional elsewhere;
this keeps zero new hard deps). Auth is Basic email:token, the standard for
Jira Cloud API tokens; api_version "3" wraps comment bodies in ADF, "2"
sends raw text (Server/DC).

Secrets follow the house convention: the config stores the env-var NAME
(``token_env``); the value comes from the environment / secrets.env.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class JiraError(RuntimeError):
    pass


def _adf(text: str) -> dict:
    """Wrap plain text in the minimal Atlassian Document Format envelope."""
    return {"type": "doc", "version": 1, "content": [
        {"type": "paragraph",
         "content": [{"type": "text", "text": line or " "}]}
        for line in text.splitlines() or [text]
    ]}


@dataclass
class JiraClient:
    base_url: str
    email: str
    token_env: str = "SENTINEL_JIRA_TOKEN"
    api_version: str = "3"
    timeout: int = 30
    # L-08: an internal Jira (Server/DC behind the org's own CA) needs these.
    verify_ssl: bool = True
    auth: str = "basic"          # basic (email:token, Cloud) | bearer (Server/DC PAT)

    # -- plumbing --------------------------------------------------------------
    def _token(self) -> str:
        tok = os.environ.get(self.token_env, "")
        if not tok:
            raise JiraError(
                f"no Jira token: env var {self.token_env} is empty. Paste your "
                f"API token in the chat and I will store it via "
                f"configure_integration (section=jira, key=token).")
        return tok

    def configured(self) -> bool:
        return bool(self.base_url and self.email)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base_url.rstrip("/") + f"/rest/api/{self.api_version}" + path
        if str(self.auth).lower() == "bearer":
            authz = f"Bearer {self._token()}"
        else:
            authz = "Basic " + base64.b64encode(
                f"{self.email}:{self._token()}".encode()).decode()
        req = urllib.request.Request(
            url, method=method,
            headers={"Authorization": authz,
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        data = json.dumps(payload).encode() if payload is not None else None
        ctx = None
        if not self.verify_ssl:
            import ssl
            ctx = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, data=data, timeout=self.timeout,
                                        context=ctx) as r:
                body = r.read().decode("utf-8", errors="replace")
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            hint = ""
            if e.code == 401:
                hint = (" -- the token was rejected. Jira Cloud uses your "
                        "login email + an API token (jira.auth: basic); Jira "
                        "Server/Data Center personal access tokens need "
                        "jira.auth: bearer.")
            raise JiraError(f"Jira {method} {path} -> HTTP {e.code}: {detail}{hint}") from e
        except urllib.error.URLError as e:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(e.reason):
                hint = (" -- the server uses a certificate this machine does "
                        "not trust (an internal CA). Set jira.verify_ssl: false "
                        "for an internal Jira.")
            raise JiraError(f"cannot reach Jira at {self.base_url}: {e.reason}{hint}") from e

    # -- reads ----------------------------------------------------------------
    def my_open_issues(self, max_results: int = 20) -> list[dict]:
        jql = "assignee = currentUser() AND statusCategory != Done ORDER BY updated DESC"
        payload = {"jql": jql, "maxResults": max_results,
                   "fields": ["summary", "status", "priority", "duedate", "issuetype"]}
        # L-07: Jira Cloud retired POST /search in favour of /search/jql;
        # Server/Data Center only has /search. Try the right one first for
        # the API version, and the other if that endpoint does not exist.
        first, second = (("/search/jql", "/search") if str(self.api_version) == "3"
                         else ("/search", "/search/jql"))
        try:
            res = self._request("POST", first, payload)
        except JiraError as e:
            if not any(f"HTTP {c}" in str(e) for c in (404, 405, 410)):
                raise
            res = self._request("POST", second, payload)
        return res.get("issues", [])

    def transitions(self, issue_key: str) -> list[dict]:
        res = self._request("GET", f"/issue/{issue_key}/transitions")
        return res.get("transitions", [])

    # -- writes ------------------------------------------------------------------
    def add_comment(self, issue_key: str, text: str) -> dict:
        body = _adf(text) if self.api_version == "3" else text
        return self._request("POST", f"/issue/{issue_key}/comment",
                             {"body": body})

    def add_worklog(self, issue_key: str, text: str,
                    time_spent: str = "1h") -> dict:
        comment = _adf(text) if self.api_version == "3" else text
        return self._request("POST", f"/issue/{issue_key}/worklog",
                             {"comment": comment, "timeSpent": time_spent})

    def create_issue(self, project_key: str, summary: str,
                     description: str = "", issue_type: str = "Task") -> dict:
        desc = _adf(description) if self.api_version == "3" else description
        fields: dict = {"project": {"key": project_key}, "summary": summary,
                        "issuetype": {"name": issue_type}}
        if description:
            fields["description"] = desc
        return self._request("POST", "/issue", {"fields": fields})

    def transition(self, issue_key: str, transition_id: str) -> dict:
        return self._request("POST", f"/issue/{issue_key}/transitions",
                             {"transition": {"id": transition_id}})
