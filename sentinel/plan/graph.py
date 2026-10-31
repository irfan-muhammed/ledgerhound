"""
sentinel.plan.graph
====================

V3-015: the plan checklist becomes a task graph. Where the old flat
``PlanState.steps`` (``sentinel.tools.plan_tools``) was a numbered list with
4 statuses, a ``PlanGraph`` node additionally carries a ``type``, an
optional ``budget``, ``depends_on`` (other node ids), and what it
``produces`` -- and findings discovered mid-session can extend the graph
(``add_node``) or retire a node that turned out not to apply
(``invalidate_node``) without losing the audit trail (a retired node is
marked ``blocked``, never deleted).

Design notes:
  - Nodes are appended, never removed, and ``depends_on`` may only
    reference IDs that already exist at add-time -- this makes the graph
    acyclic by construction for the tool-exposed path (`add_node`). A cycle
    can only be constructed via `add_dependency`, which edits an EXISTING
    node's deps after creation and is therefore the one path that needs a
    real cycle check; it is deliberately NOT exposed as an LLM tool in v1
    (same reasoning as `DatasetCard.add_quirk` in V3-011 -- a low-level
    primitive for a future human/tooling workflow, not something the model
    calls directly yet).
  - The stop condition a session's forced-continuation check reads is
    "every node is done, skipped, or blocked" -- `open_nodes()`.
  - `PlanGraph` is persisted whole (``plan.json``) on every mutation, not
    incrementally -- the graph is small (a handful to a few dozen nodes),
    so a full rewrite is simpler and cheap, unlike the append-only event
    stream (V3-012) where incremental writes matter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

VALID_TYPES = ("analysis", "verification", "ask_user", "model")
VALID_STATUSES = ("pending", "active", "done", "skipped", "blocked")
CLOSED_STATUSES = ("done", "skipped", "blocked")
_ICONS = {"pending": "[ ]", "active": "[>]", "done": "[x]",
          "skipped": "[-]", "blocked": "[!]"}


@dataclass
class PlanNode:
    id: str
    text: str
    type: str = "analysis"
    status: str = "pending"
    budget: Optional[int] = None
    depends_on: list = field(default_factory=list)
    produces: str = ""
    blocked_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanNode":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlanGraph:
    nodes: list = field(default_factory=list)   # list[PlanNode], creation order
    _next_id: int = 1
    # V3-023: L0Framing.to_dict(), or None if no study-start intake has been
    # recorded on this plan yet. Stored as a plain dict (not an L0Framing
    # instance) so PlanGraph stays JSON-native like every other field here.
    framing: Optional[dict] = None

    # -- lookup -------------------------------------------------------------

    def get(self, node_id: str) -> Optional[PlanNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def __len__(self) -> int:
        return len(self.nodes)

    # -- mutation (append-only; a node is retired via status, never removed) -

    def add_node(self, text: str, type: str = "analysis",
                depends_on: Optional[list] = None, budget: Optional[int] = None,
                produces: str = "") -> "PlanNode | str":
        """Returns the new PlanNode, or an error string (tools must not
        raise) if `type` is invalid or `depends_on` references an unknown
        id -- the latter is what actually keeps this graph acyclic."""
        if type not in VALID_TYPES:
            return f"error: type must be one of {VALID_TYPES}"
        deps = list(depends_on or [])
        for d in deps:
            if self.get(d) is None:
                return f"error: unknown dependency id {d!r}"
        node = PlanNode(id=str(self._next_id), text=text.strip()[:200], type=type,
                        depends_on=deps, budget=budget, produces=produces)
        self.nodes.append(node)
        self._next_id += 1
        return node

    def add_dependency(self, node_id: str, depends_on_id: str) -> str:
        """Add a dependency to an EXISTING node after creation -- the one
        path that can construct a cycle, so it's the one path that needs a
        real cycle check. Not tool-exposed in v1; a tested primitive for
        graph edits made by trusted (non-LLM) callers."""
        node = self.get(node_id)
        if node is None:
            return f"error: unknown node id {node_id!r}"
        dep = self.get(depends_on_id)
        if dep is None:
            return f"error: unknown dependency id {depends_on_id!r}"
        if depends_on_id in node.depends_on:
            return f"{node_id} already depends on {depends_on_id}"
        node.depends_on.append(depends_on_id)
        if self.has_cycle():
            node.depends_on.remove(depends_on_id)
            return f"error: adding {node_id} -> {depends_on_id} would create a cycle"
        return f"{node_id} now depends on {depends_on_id}"

    def set_framing(self, framing_dict: dict) -> None:
        """Record L0 framing (sentinel.framing.l0.L0Framing.to_dict()) on
        this plan. Not tool-exposed in v1 -- same "trusted caller, not an
        LLM tool" status as add_dependency; V3-023's close-out doesn't
        require an interactive capture flow, only that the data can be
        stored and read back."""
        self.framing = dict(framing_dict)

    def update_status(self, node_id: str, status: str, reason: str = "") -> str:
        if status not in VALID_STATUSES:
            return f"error: status must be one of {VALID_STATUSES}"
        node = self.get(node_id)
        if node is None:
            return f"error: unknown node id {node_id!r}"
        node.status = status
        if status == "blocked":
            node.blocked_reason = reason
        if status == "active":
            for n in self.nodes:
                if n.id != node_id and n.status == "active":
                    n.status = "pending"
        return "ok"

    def invalidate_node(self, node_id: str, reason: str) -> str:
        return self.update_status(node_id, "blocked", reason)

    # -- queries --------------------------------------------------------------

    def open_nodes(self) -> list:
        return [n for n in self.nodes if n.status not in CLOSED_STATUSES]

    def ready_nodes(self) -> list:
        """Pending nodes whose dependencies are all done -- what the model
        could reasonably work on next, dependency-aware."""
        done_ids = {n.id for n in self.nodes if n.status == "done"}
        return [n for n in self.nodes
                if n.status == "pending" and all(d in done_ids for d in n.depends_on)]

    def has_cycle(self) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n.id: WHITE for n in self.nodes}

        def visit(node_id: str) -> bool:
            color[node_id] = GRAY
            node = self.get(node_id)
            for dep in (node.depends_on if node else []):
                if color.get(dep) == GRAY:
                    return True
                if color.get(dep) == WHITE and visit(dep):
                    return True
            color[node_id] = BLACK
            return False

        return any(color[n.id] == WHITE and visit(n.id) for n in self.nodes)

    # -- rendering --------------------------------------------------------------

    def render(self) -> str:
        if not self.nodes:
            return "(no plan yet)"
        lines = []
        for i, n in enumerate(self.nodes):
            extra = []
            if n.depends_on:
                extra.append(f"deps: {','.join(n.depends_on)}")
            if n.status == "blocked" and n.blocked_reason:
                extra.append(f"blocked: {n.blocked_reason}")
            suffix = f"  ({'; '.join(extra)})" if extra else ""
            lines.append(f"{_ICONS.get(n.status, '[ ]')} {i + 1}. {n.text}{suffix}")
        return "\n".join(lines)

    # -- (de)serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes], "next_id": self._next_id,
                "framing": self.framing}

    @classmethod
    def from_dict(cls, d: dict) -> "PlanGraph":
        g = cls()
        g.nodes = [PlanNode.from_dict(nd) for nd in d.get("nodes", [])]
        g._next_id = d.get("next_id", len(g.nodes) + 1)
        g.framing = d.get("framing")   # None for graphs saved before V3-023
        return g

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlanGraph":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()
