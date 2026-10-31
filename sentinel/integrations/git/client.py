"""
sentinel.integrations.git.client
================================

A thin subprocess wrapper around the ``git`` CLI -- no gitpython dependency.
Every call returns a ``GitResult`` (never raises); stdout/stderr are decoded
utf-8 with replacement so Windows console encodings can't crash a session.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitResult:
    ok: bool
    out: str
    err: str = ""

    def text(self) -> str:
        body = self.out.strip()
        if not self.ok:
            return f"git error: {self.err.strip() or body or 'unknown failure'}"
        return body or "(no output)"


class GitClient:
    """Runs git commands rooted at ``repo_root``."""

    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root)

    def available(self) -> bool:
        return (self.root / ".git").exists() and self._run("--version").ok

    def _run(self, *args: str, timeout: int = 120) -> GitResult:
        try:
            p = subprocess.run(
                ["git", *args], cwd=str(self.root), capture_output=True,
                timeout=timeout, text=True, encoding="utf-8", errors="replace")
            return GitResult(p.returncode == 0, p.stdout or "", p.stderr or "")
        except FileNotFoundError:
            return GitResult(False, "", "git is not installed or not on PATH")
        except subprocess.TimeoutExpired:
            return GitResult(False, "", f"git {' '.join(args)} timed out")
        except Exception as e:  # noqa: BLE001
            return GitResult(False, "", str(e))

    # -- reads ---------------------------------------------------------------
    def status(self) -> GitResult:
        return self._run("status", "--short", "--branch")

    def log(self, n: int = 10) -> GitResult:
        return self._run("log", f"-{max(1, min(n, 50))}",
                         "--oneline", "--decorate")

    def diff(self, staged: bool = False) -> GitResult:
        args = ["diff", "--stat"] + (["--cached"] if staged else [])
        return self._run(*args)

    def current_branch(self) -> str:
        r = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return r.out.strip() if r.ok else ""

    def identity(self) -> tuple[str, str]:
        name = self._run("config", "user.name").out.strip()
        email = self._run("config", "user.email").out.strip()
        return name, email

    def today_log(self) -> GitResult:
        """Today's commits by the current identity -- feeds the session digest."""
        _, email = self.identity()
        args = ["log", "--since=midnight", "--oneline"]
        if email:
            args += [f"--author={email}"]
        return self._run(*args)

    # -- writes (call only behind a HITL confirm) ------------------------------
    def pull(self, remote: str = "origin", branch: str = "") -> GitResult:
        args = ["pull", remote] + ([branch] if branch else [])
        return self._run(*args, timeout=300)

    def add_all(self) -> GitResult:
        return self._run("add", "-A")

    def commit(self, message: str) -> GitResult:
        return self._run("commit", "-m", message)

    def pending_paths(self) -> list[str]:
        """Paths `git add -A` would stage right now (new, modified, deleted),
        without staging anything."""
        r = self._run("add", "-A", "--dry-run")
        out = []
        for line in r.out.splitlines():
            line = line.strip()
            if line.startswith(("add '", "remove '")) and line.endswith("'"):
                out.append(line.split("'", 1)[1][:-1])
        return out

    def push(self, remote: str = "origin", branch: str = "") -> GitResult:
        branch = branch or self.current_branch()
        args = ["push", remote] + ([branch] if branch else [])
        return self._run(*args, timeout=300)
