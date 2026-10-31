"""
sentinel.integrations.git.tools
===============================

Git as a tool family for the analyst shell. Reads (status/log/diff/today)
are free; ``git_pull`` and ``git_commit_push`` are confirm-gated. When the
model omits a commit message, one is drafted from the session digest so
"push my work" just works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from sentinel.integrations.git.client import GitClient

# L-02: what "push my work" must never publish by accident. Sentinel's own
# outputs (run folders, feature exports, adversarial case slices, spilled
# results) and anything shaped like a data file are customer data, not work
# product -- they stay local unless a human commits them deliberately.
_DATA_SUFFIXES = (".parquet", ".pq", ".csv", ".tsv", ".feather", ".arrow",
                  ".xlsx", ".xls", ".jsonl", ".pkl", ".pickle", ".h5",
                  ".hdf5", ".npy", ".npz", ".db", ".sqlite", ".zip", ".gz")
_OUTPUT_DIRS = ("run_outputs/", "sentinel_workspace/", "features/",
                "scratch/", "results/", "adversarial/", ".sentinel/")
_MAX_BYTES = 5 * 1024 * 1024


def _blocked(root: Path, paths: list) -> list:
    """(path, reason) for every staged path that looks like data/output."""
    out = []
    for rel in paths:
        low = rel.replace("\\", "/").lower()
        reason = ""
        if low.endswith(_DATA_SUFFIXES):
            reason = "data file"
        elif any(f"/{d}" in f"/{low}" for d in _OUTPUT_DIRS):
            reason = "Sentinel output folder"
        else:
            try:
                if (root / rel).stat().st_size > _MAX_BYTES:
                    reason = "larger than 5 MB"
            except OSError:
                pass
        if reason:
            out.append((rel, reason))
    return out
from sentinel.tools.registry import ToolDef

DigestFn = Callable[[], str]  # -> commit-message draft ("" if nothing to say)


def build_git_tools(client: GitClient, remote: str = "origin",
                    default_branch: str = "",
                    digest_fn: Optional[DigestFn] = None) -> list[ToolDef]:

    def git_status() -> str:
        return client.status().text()

    def git_log(n: int = 10) -> str:
        return client.log(n).text()

    def git_diff(staged: bool = False) -> str:
        return client.diff(staged).text()

    def git_today() -> str:
        r = client.today_log()
        return r.text() if r.out.strip() or not r.ok else "(no commits today yet)"

    def git_pull(branch: str = "") -> str:
        return client.pull(remote, branch or default_branch).text()

    def _draft_message(message: str) -> str:
        if message.strip():
            return message.strip()
        if digest_fn is not None:
            try:
                d = digest_fn().strip()
                if d:
                    return d
            except Exception:  # noqa: BLE001
                pass
        return "Session work via sentinel shell"

    def git_commit_push(message: str = "", branch: str = "") -> str:
        st = client.status()
        dirty = any(line and not line.startswith("##")
                    for line in st.out.splitlines())
        msg = _draft_message(message)
        steps = []
        if dirty:
            blocked = _blocked(client.root, client.pending_paths())
            if blocked:
                listing = "\n".join(f"  - {p}  ({why})" for p, why in blocked[:20])
                more = f"\n  ... and {len(blocked) - 20} more" if len(blocked) > 20 else ""
                return ("REFUSED: this would commit and push files that look like "
                        "data or Sentinel output (customer data must not leave "
                        f"this machine through git):\n{listing}{more}\n"
                        "Add them to .gitignore (e.g. run_outputs/, *.parquet, "
                        "*.csv), or commit the files you mean by hand. Nothing "
                        "was staged, committed or pushed.")
            add = client.add_all()
            if not add.ok:
                return add.text()
            com = client.commit(msg)
            if not com.ok:
                return com.text()
            steps.append(f"committed: {msg.splitlines()[0][:80]}")
        else:
            steps.append("working tree clean -- nothing new to commit")
        # L-03: push the branch that was just committed to. default_branch is
        # for pulling; pushing it from a feature branch pushed an unchanged
        # local main and reported success.
        target = branch or client.current_branch()
        push = client.push(remote, target)
        if not push.ok:
            return "\n".join(steps + [push.text()])
        steps.append(f"pushed to {remote}/{target}" if target else f"pushed to {remote}")
        return "\n".join(steps)

    def _confirm_push(args: dict) -> str:
        msg = _draft_message(str(args.get("message", "")))
        paths = client.pending_paths()
        shown = "\n".join(f"  {p}" for p in paths[:25])
        more = f"\n  ... and {len(paths) - 25} more" if len(paths) > 25 else ""
        return (f"Commit & push {len(paths)} file(s) to {remote}/"
                f"{args.get('branch') or client.current_branch()}?\n"
                f"message: {msg.splitlines()[0][:100]}\n"
                f"files:\n{shown or '  (clean tree -- push only)'}{more}")

    return [
        ToolDef(name="git_status",
                description="Show working-tree status of the project repo.",
                parameters={"type": "object", "properties": {}, "required": []},
                fn=git_status),
        ToolDef(name="git_log",
                description="Show recent commits (oneline).",
                parameters={"type": "object", "properties": {
                    "n": {"type": "integer", "description": "how many (<=50)"}},
                    "required": []},
                fn=git_log),
        ToolDef(name="git_diff",
                description="Show a diffstat of unstaged (or staged) changes.",
                parameters={"type": "object", "properties": {
                    "staged": {"type": "boolean"}}, "required": []},
                fn=git_diff),
        ToolDef(name="git_today",
                description="Show today's commits by the current user -- useful "
                            "for a work update.",
                parameters={"type": "object", "properties": {}, "required": []},
                fn=git_today),
        ToolDef(name="git_pull",
                description="Pull latest changes from the remote.",
                parameters={"type": "object", "properties": {
                    "branch": {"type": "string"}}, "required": []},
                fn=git_pull, hitl="confirm",
                confirm_prompt=lambda a: f"git pull {remote} "
                                         f"{a.get('branch') or default_branch}?"),
        ToolDef(name="git_commit_push",
                description="Stage all changes, commit (message auto-drafted "
                            "from the session's work when omitted), and push. "
                            "Use when the user says 'push my work/code'.",
                parameters={"type": "object", "properties": {
                    "message": {"type": "string",
                                "description": "Commit message; omit to auto-draft."},
                    "branch": {"type": "string"}}, "required": []},
                fn=git_commit_push, hitl="confirm",
                confirm_prompt=_confirm_push),
    ]
