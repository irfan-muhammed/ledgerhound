"""
ledgerhound.reporting.session_digest
=================================

"What did I do today?" as data. A ``SessionDigest`` folds three sources into
one honest summary:

  1. findings.jsonl from the current run folder (evidence-backed claims)
  2. the ToolRegistry action log (every successful tool call this session)
  3. today's git commits by the current identity (when a repo is present)

Outputs: ``to_markdown()`` for Jira updates / reports, ``to_commit_message()``
for git_commit_push drafts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional


@dataclass
class SessionDigest:
    findings_path: Optional[Path] = None
    action_log: list[str] = field(default_factory=list)
    git_today_fn: Optional[Callable[[], str]] = None
    goal: str = ""

    # -- sources -------------------------------------------------------------
    def findings(self) -> list[dict]:
        out: list[dict] = []
        if self.findings_path and Path(self.findings_path).exists():
            for line in Path(self.findings_path).read_text(
                    encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        return out

    def commits_today(self) -> str:
        if self.git_today_fn is None:
            return ""
        try:
            t = (self.git_today_fn() or "").strip()
            return "" if t.startswith("(no commits") else t
        except Exception:  # noqa: BLE001
            return ""

    def is_empty(self) -> bool:
        return not (self.findings() or self.action_log or self.commits_today())

    # -- renderings -------------------------------------------------------------
    def to_markdown(self) -> str:
        lines = [f"## Work update -- {date.today().isoformat()}"]
        if self.goal:
            lines.append(f"**Goal:** {self.goal}")
        f = self.findings()
        if f:
            lines.append("\n**Findings**")
            for x in f[:12]:
                claim = str(x.get("claim", "")).strip()
                frame = x.get("sample_frame", "")
                if claim:
                    lines.append(f"- {claim}" + (f"  _[{frame}]_" if frame else ""))
        acts = _interesting(self.action_log)
        if acts:
            lines.append("\n**Session activity**")
            lines += [f"- {a}" for a in acts[:10]]
        commits = self.commits_today()
        if commits:
            lines.append("\n**Commits today**")
            lines += [f"- {c}" for c in commits.splitlines()[:10]]
        if len(lines) == 1 + (1 if self.goal else 0):
            lines.append("_No recorded findings or actions yet this session._")
        return "\n".join(lines)

    def to_commit_message(self) -> str:
        f = self.findings()
        subject = ""
        if self.goal:
            subject = f"Analysis: {self.goal.strip()}"
        elif f:
            subject = f"Analysis: {str(f[0].get('claim', ''))[:60]}"
        if not subject:
            acts = _interesting(self.action_log)
            subject = acts[0][:70] if acts else ""
        if not subject:
            return ""
        body = [f"- {str(x.get('claim', ''))[:100]}" for x in f[:8]
                if x.get("claim")]
        msg = subject[:72]
        if body:
            msg += "\n\n" + "\n".join(body)
        return msg


_SKIP_PREFIXES = ("list_directory", "show_config", "git_status", "git_log",
                  "git_diff", "git_today", "session_digest", "consult_skill",
                  "propose_plan", "update_plan", "jira_pending_tasks")


def _interesting(actions: list[str]) -> list[str]:
    """Drop pure look-around calls; keep the work."""
    return [a for a in actions
            if not any(a.startswith(p) for p in _SKIP_PREFIXES)]
