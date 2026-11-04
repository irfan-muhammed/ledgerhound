"""
ledgerhound.reproducibility.manifest
=================================

ReproducibilityManifest -- the file-state companion to the OperationLog.

  * OperationLog answers: *what happened, in what order, with what params?*
  * Manifest answers:     *what files existed, with what hashes, in what
                           environment, against what code revision?*

Together they make a run replayable. Open the manifest and you know:

  - the exact ledgerhound + Python + pandas + numpy versions
  - the git commit (and whether the tree was dirty)
  - SHA-256 hash + size + mtime for every artifact in the workspace
  - random seeds used (caller-supplied)
  - the compute profile in effect (pandas vs cuDF, numpy vs faiss)
  - the command line that produced the workspace

Output: `workspace/MANIFEST.json`. One file per run is the recommended pattern
(append `MANIFEST_<period>.json` if you don't want overwrites).

Design rules
------------
* **Best-effort throughout.** Missing git, no internet, untrusted file --
  log it and keep going. A partial manifest beats no manifest.
* **Hash on read, not on write.** We hash files that exist at manifest-
  build time. Files written after the manifest aren't captured (call
  build_manifest again if you need a fresh snapshot).
* **No external dependencies.** Pure stdlib + an optional subprocess
  call to `git` if the CLI is available.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

log = logging.getLogger(__name__)


# -- Data structures --------------------------------------------------------


@dataclass
class FileEntry:
    path: str           # relative to manifest root, forward slashes
    size: int           # bytes
    sha256: str         # hex
    modified_at: float  # unix epoch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvSnapshot:
    ledgerhound_version: Optional[str]
    python_version: str
    platform: str
    machine: str
    packages: dict[str, str]  # package -> version (pandas, numpy, sklearn, ...)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GitState:
    available: bool
    commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: Optional[bool] = None
    remote: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityManifest:
    """The full file-state + environment snapshot."""

    generated_at: str              # ISO-8601
    workspace_root: str
    env: EnvSnapshot
    git: GitState
    files: list[FileEntry] = field(default_factory=list)
    seeds: dict[str, Any] = field(default_factory=dict)
    command: Optional[str] = None
    compute_profile: Optional[dict[str, Any]] = None   # mirrored from compute_profile.json
    notes: list[str] = field(default_factory=list)

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "workspace_root": self.workspace_root,
            "env": self.env.to_dict(),
            "git": self.git.to_dict(),
            "files": [f.to_dict() for f in self.files],
            "seeds": self.seeds,
            "command": self.command,
            "compute_profile": self.compute_profile,
            "notes": self.notes,
            "n_files": len(self.files),
            "total_bytes": sum(f.size for f in self.files),
        }

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReproducibilityManifest":
        env = EnvSnapshot(**d.get("env", {})) if d.get("env") else \
              EnvSnapshot(None, sys.version.split()[0], platform.platform(),
                          platform.machine(), {})
        git = GitState(**d.get("git", {"available": False})) if d.get("git") else \
              GitState(available=False)
        files = [FileEntry(**f) for f in d.get("files", [])]
        return cls(
            generated_at=d.get("generated_at", _now_iso()),
            workspace_root=d.get("workspace_root", ""),
            env=env,
            git=git,
            files=files,
            seeds=d.get("seeds", {}),
            command=d.get("command"),
            compute_profile=d.get("compute_profile"),
            notes=list(d.get("notes", [])),
        )

    # -- Display ----------------------------------------------------------

    def display(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            console = Console()
            console.print(Panel(
                f"[bold]Reproducibility Manifest[/bold]\n"
                f"workspace: [cyan]{self.workspace_root}[/cyan]\n"
                f"generated: {self.generated_at}\n"
                f"ledgerhound:  {self.env.ledgerhound_version or '(unknown)'}\n"
                f"python:    {self.env.python_version}\n"
                f"git:       {('dirty ' if self.git.dirty else '') + (self.git.commit[:10] if self.git.commit else '(no git)')}\n"
                f"files:     {len(self.files)}  /  {_humanise_bytes(sum(f.size for f in self.files))}",
                border_style="cyan",
            ))
            # File table (head-only)
            t = Table("path", "size", "sha256", title=f"First 12 of {len(self.files)} files")
            for fe in self.files[:12]:
                t.add_row(fe.path, _humanise_bytes(fe.size), fe.sha256[:12] + "...")
            console.print(t)
            if self.notes:
                console.print("\n[dim]" + "\n".join(self.notes) + "[/dim]")
        except Exception:
            print(json.dumps(self.to_dict(), indent=2)[:2000])


# -- Builders ---------------------------------------------------------------


_DEFAULT_HASH_DIRS = ("reports", "datasets", "samples", "findings", "knowledge")
_HASH_CHUNK = 1 << 20  # 1 MiB
_DEFAULT_SKIP_SUFFIXES = (".pyc",)
_DEFAULT_SKIP_NAMES = {"__pycache__", ".pytest_cache"}


def build_manifest(
    workspace_root: Union[str, Path],
    *,
    seeds: Optional[dict[str, Any]] = None,
    command: Optional[str] = None,
    extra_dirs: Iterable[str] = (),
    skip_dirs: Iterable[str] = (),
    include_compute_profile: bool = True,
) -> ReproducibilityManifest:
    """
    Walk the workspace, hash every artifact, and capture the environment.

    Parameters
    ----------
    workspace_root : root of the workspace to hash
    seeds          : caller-supplied random seeds {numpy: 42, torch: 0, ...}
    command        : the command line that produced this workspace
    extra_dirs     : additional directory names (relative to root) to hash
                     beyond the defaults (reports/datasets/samples/findings/knowledge)
    skip_dirs      : directory names anywhere in the tree to skip
    include_compute_profile : if compute_profile.json exists at the workspace
                     root, mirror its contents into the manifest
    """
    root = Path(workspace_root).resolve()
    skip = set(_DEFAULT_SKIP_NAMES) | set(skip_dirs)
    target_dirs = list(_DEFAULT_HASH_DIRS) + list(extra_dirs)

    notes: list[str] = []
    file_entries: list[FileEntry] = []

    if not root.exists():
        notes.append(f"workspace root does not exist: {root}")
    else:
        for sub in target_dirs:
            sub_path = root / sub
            if not sub_path.exists():
                continue
            for entry in _walk_files(sub_path, skip=skip):
                try:
                    rel = entry.relative_to(root).as_posix()
                    st = entry.stat()
                    sha = _sha256_file(entry)
                    file_entries.append(FileEntry(
                        path=rel, size=st.st_size, sha256=sha,
                        modified_at=st.st_mtime,
                    ))
                except Exception as exc:
                    notes.append(f"hash failed for {entry}: {exc}")

    env = _capture_env()
    git = _capture_git(root)
    compute_profile = None
    if include_compute_profile:
        cp_path = root / "compute_profile.json"
        if cp_path.exists():
            try:
                compute_profile = json.loads(cp_path.read_text(encoding="utf-8"))
            except Exception as exc:
                notes.append(f"could not read compute_profile.json: {exc}")

    return ReproducibilityManifest(
        generated_at=_now_iso(),
        workspace_root=str(root),
        env=env,
        git=git,
        files=file_entries,
        seeds=dict(seeds or {}),
        command=command,
        compute_profile=compute_profile,
        notes=notes,
    )


def load_manifest(path: Union[str, Path]) -> Optional[ReproducibilityManifest]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return ReproducibilityManifest.from_dict(
            json.loads(p.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        log.warning("Could not load manifest %s: %s", p, exc)
        return None


# -- Internals --------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _humanise_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MiB"
    return f"{n / 1024 ** 3:.2f} GiB"


def _walk_files(root: Path, *, skip: set[str]) -> Iterable[Path]:
    """Yield files under root, skipping directories whose name is in `skip`."""
    for dirpath, dirnames, filenames in os.walk(root):
        # mutate in place so os.walk doesn't descend into skipped dirs
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if any(fn.endswith(sx) for sx in _DEFAULT_SKIP_SUFFIXES):
                continue
            yield Path(dirpath) / fn


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _capture_env() -> EnvSnapshot:
    # ledgerhound version -- read from importlib.metadata if installed,
    # else from pyproject.toml as a fallback
    ledgerhound_version: Optional[str] = None
    try:
        from importlib.metadata import version as _v
        ledgerhound_version = _v("ledgerhound")
    except Exception:
        ledgerhound_version = None

    packages: dict[str, str] = {}
    for pkg in ("pandas", "numpy", "scikit-learn", "networkx",
                "sentence-transformers", "faiss-cpu", "rich", "typer"):
        try:
            from importlib.metadata import version as _v
            packages[pkg] = _v(pkg)
        except Exception:
            pass

    return EnvSnapshot(
        ledgerhound_version=ledgerhound_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        packages=packages,
    )


def _capture_git(workspace_root: Path) -> GitState:
    """Try to read git state. Searches from workspace upward for a .git dir;
    falls back to the current working directory so that running ledgerhound
    against an external workspace still tags the ledgerhound repo's commit."""

    def _find_git(start: Path) -> Optional[Path]:
        cur = start if start.is_dir() else start.parent
        while True:
            if (cur / ".git").exists():
                return cur
            if cur.parent == cur:
                return None
            cur = cur.parent

    repo_root = _find_git(workspace_root)
    if repo_root is None:
        repo_root = _find_git(Path.cwd())
    if repo_root is None:
        return GitState(available=False, error="no .git in workspace or cwd ancestry")

    def _git(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return r.stdout.strip()
            return None
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return GitState(available=False, error="git command failed")

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    dirty = bool(status) if status is not None else None
    remote = _git("config", "--get", "remote.origin.url")

    return GitState(
        available=True,
        commit=commit,
        branch=branch,
        dirty=dirty,
        remote=remote,
    )
