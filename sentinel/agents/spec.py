"""
sentinel.agents.spec
====================

``AgentSpec`` -- the declarative ``agent_def/v1`` an agent is made of.

An agent is *data, not code*: a role prompt + a tool whitelist + an output
contract. The engine consumes this spec; the spec carries no executable code.
This single shape enables built-in specialists, user-defined agents, and the
shared library, all upgrade-safe.

Validation is strict at load/create time (fail fast, never half-load a broken
agent into the loop). The tool whitelist is checked against the canonical
catalog (the security boundary).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sentinel.agents.tool_registry import KNOWN_CONTRACTS, unknown_tools

SCHEMA = "agent_def/v2"
_VALID_SCOPES = ("private", "shared")
_VALID_COST = ("cheap", "medium", "expensive")
_VALID_HITL = ("none", "confirm", "choose")


class AgentValidationError(ValueError):
    """Raised when an agent definition fails validation."""


@dataclass
class AgentSpec:
    """A declarative agent definition (agent_def/v1)."""

    name: str
    role_prompt: str
    version: str = "0.1.0"
    owner: str = ""                       # filled from identity if blank
    schema_version: int = 1
    description: str = ""
    tools: list[str] = field(default_factory=list)
    output_contract: str = "findings/v1"
    memory_scope: str = "private"         # private | shared
    defaults: dict[str, Any] = field(default_factory=dict)

    # -- agent_def/v2 -- routing & composition metadata (all optional; --------
    # additive, so every v1 YAML loads unchanged). The Conductor/Analyst
    # reasons over these at runtime, so adding a capability = dropping a file.
    when_to_use: str = ""                 # honest one-line the model reasons over
    preconditions: list[str] = field(default_factory=list)   # e.g. two-datasets
    produces: list[str] = field(default_factory=list)        # outputs/finding types
    cost_class: str = "medium"            # cheap | medium | expensive
    hitl: str = "none"                    # none | confirm | choose

    # -- validation -------------------------------------------------------

    def validate(self, *, allow_unknown_tools: bool = False) -> "AgentSpec":
        """Validate the spec in place; return self. Raises on the first error."""
        if not self.name or not str(self.name).strip():
            raise AgentValidationError("agent 'name' is required")
        if not _is_slug_safe(self.name):
            raise AgentValidationError(
                f"agent name {self.name!r} must be a slug: lowercase letters, "
                f"digits, '-' or '_'"
            )
        if not self.role_prompt or not self.role_prompt.strip():
            raise AgentValidationError(f"agent {self.name!r}: 'role_prompt' is required")
        if self.memory_scope not in _VALID_SCOPES:
            raise AgentValidationError(
                f"agent {self.name!r}: memory_scope must be one of {_VALID_SCOPES}, "
                f"got {self.memory_scope!r}"
            )
        if not isinstance(self.tools, list):
            raise AgentValidationError(f"agent {self.name!r}: 'tools' must be a list")
        if not allow_unknown_tools:
            bad = unknown_tools(self.tools)
            if bad:
                raise AgentValidationError(
                    f"agent {self.name!r}: unknown tools {bad} "
                    f"(not in the engine tool catalog)"
                )
        if self.cost_class not in _VALID_COST:
            raise AgentValidationError(
                f"agent {self.name!r}: cost_class must be one of {_VALID_COST}, "
                f"got {self.cost_class!r}"
            )
        if self.hitl not in _VALID_HITL:
            raise AgentValidationError(
                f"agent {self.name!r}: hitl must be one of {_VALID_HITL}, "
                f"got {self.hitl!r}"
            )
        if not isinstance(self.preconditions, list):
            raise AgentValidationError(f"agent {self.name!r}: 'preconditions' must be a list")
        if not isinstance(self.produces, list):
            raise AgentValidationError(f"agent {self.name!r}: 'produces' must be a list")
        if not self.output_contract:
            raise AgentValidationError(f"agent {self.name!r}: 'output_contract' is required")
        if self.output_contract not in KNOWN_CONTRACTS:
            # Not fatal -- a new contract may be in flight -- but surface it.
            import logging
            logging.getLogger(__name__).warning(
                "agent %s: output_contract %r is not a known contract %s",
                self.name, self.output_contract, sorted(KNOWN_CONTRACTS),
            )
        return self

    # -- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict, *, validate: bool = True,
                  allow_unknown_tools: bool = False) -> "AgentSpec":
        known = cls.__dataclass_fields__
        spec = cls(**{k: v for k, v in d.items() if k in known})
        if validate:
            spec.validate(allow_unknown_tools=allow_unknown_tools)
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path, *, validate: bool = True,
                  allow_unknown_tools: bool = False) -> "AgentSpec":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, validate=validate,
                             allow_unknown_tools=allow_unknown_tools)


def _is_slug_safe(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "-_" for c in name) and name == name.lower()
