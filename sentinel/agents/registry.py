"""
sentinel.agents.registry
========================

``AgentRegistry`` -- resolves agent definitions from the three storage tiers.

Load order (later shadows earlier by name):
  1. Package built-ins   sentinel/prompts/agentic/*.yaml   (read-only, ship w/ release)
  2. User local          ~/.sentinel/agents/*.yaml         (PVC; survives pip upgrade)

(Tier 3, the MinIO shared library, is fetched on demand by the agent CLI and
materialized into the user-local tier; the registry itself stays filesystem-only
so it has no network dependency.)

Hard rule: the registry never writes to or deletes from the user tier. Only an
explicit ``sentinel agent delete`` may remove a user agent.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sentinel.agents.spec import AgentSpec, AgentValidationError

log = logging.getLogger(__name__)

# Agent specs (the "agentic prompts") live in the unified prompt tree:
# sentinel/prompts/agentic/*.yaml -- see sentinel/prompts/README.md
_BUILTIN_DIR = Path(__file__).resolve().parents[1] / "prompts" / "agentic"


def _user_dir() -> Path:
    return Path.home() / ".sentinel" / "agents"


class AgentRegistry:
    """In-memory index of available agent specs, keyed by name."""

    def __init__(self, builtin_dir: Path | None = None, user_dir: Path | None = None):
        self._builtin_dir = builtin_dir or _BUILTIN_DIR
        self._user_dir = user_dir if user_dir is not None else _user_dir()
        self._specs: dict[str, AgentSpec] = {}
        self._sources: dict[str, str] = {}  # name -> "builtin" | "user"

    def load(self) -> "AgentRegistry":
        """(Re)load all specs. Built-ins first, then user agents shadow them."""
        self._specs.clear()
        self._sources.clear()
        self._load_dir(self._builtin_dir, "builtin")
        self._load_dir(self._user_dir, "user")
        return self

    def _load_dir(self, directory: Path, source: str) -> None:
        if not directory or not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                spec = AgentSpec.from_yaml(path)
            except AgentValidationError as e:
                log.warning("Skipping invalid agent %s: %s", path.name, e)
                continue
            except Exception as e:  # malformed YAML etc.
                log.warning("Skipping unreadable agent %s: %s", path.name, e)
                continue
            if spec.name in self._specs and source == "user":
                log.info("User agent %r shadows built-in", spec.name)
            self._specs[spec.name] = spec
            self._sources[spec.name] = source

    # -- access -----------------------------------------------------------

    def get(self, name: str) -> AgentSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def source_of(self, name: str) -> str | None:
        return self._sources.get(name)

    def list(self) -> list[AgentSpec]:
        return [self._specs[n] for n in self.names()]

    # -- user-tier CRUD (rank 6) --------------------------------------------
    # Writes/deletes touch ONLY the user tier; built-ins are never mutated.

    @property
    def user_dir(self) -> Path:
        return self._user_dir

    def path_of(self, name: str) -> Path | None:
        """Filesystem path of a USER-tier agent (None if not a user agent)."""
        p = self._user_dir / f"{name}.yaml"
        return p if p.exists() else None

    def create(self, spec: AgentSpec, *, overwrite: bool = False) -> Path:
        """Validate and save ``spec`` to the user tier. Returns the path.

        A user agent may shadow a built-in (that is the intended override
        mechanism), but will not silently clobber an existing USER agent of
        the same name unless ``overwrite=True``.
        """
        spec.validate(allow_unknown_tools=True)
        self._user_dir.mkdir(parents=True, exist_ok=True)
        dest = self._user_dir / f"{spec.name}.yaml"
        if dest.exists() and not overwrite:
            raise FileExistsError(
                f"user agent {spec.name!r} already exists at {dest}; "
                f"pass overwrite=True to replace it"
            )
        spec.to_yaml(dest)
        # reflect immediately in this registry instance
        self._specs[spec.name] = spec
        self._sources[spec.name] = "user"
        log.info("Created user agent %r -> %s", spec.name, dest)
        return dest

    def delete(self, name: str) -> bool:
        """Delete a USER-tier agent. Refuses to touch built-ins.

        Returns True if a file was removed. Raises ValueError if ``name`` is a
        built-in (never deletable) or KeyError if unknown in the user tier.
        """
        dest = self._user_dir / f"{name}.yaml"
        if not dest.exists():
            if self._sources.get(name) == "builtin":
                raise ValueError(
                    f"agent {name!r} is a built-in and cannot be deleted; "
                    f"shadow it with a user agent of the same name instead"
                )
            raise KeyError(f"no user agent named {name!r}")
        dest.unlink()
        if self._sources.get(name) == "user":
            self._specs.pop(name, None)
            self._sources.pop(name, None)
        log.info("Deleted user agent %r (%s)", name, dest)
        return True
