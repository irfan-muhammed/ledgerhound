"""
ledgerhound.memory.router
=========================

V3-026 -- Librarian router v2: plan-node-scoped briefs instead of raw
recall.

Before this, every recall path (``LessonMemory.find``/``.block``,
``Librarian.recall``) was ``(dataset, free-text query)`` scoped only --
nothing accepted a plan node as an input, and nothing existed to derive a
query from one (confirmed by search: ``PlanNode.text``/``.produces`` are
plain, unstructured strings -- there is no "dataset"/"stage" field on a
node to key off of). This module is the new answer surface the roadmap
asks for: "you're at feature-selection on dataset X; relevant: ..."
instead of a raw ``Librarian.recall()`` call the caller has to phrase
itself.

Deliberately does NOT modify ``Librarian.recall()``/``RecallBundle`` --
those keep working exactly as before for callers that want raw,
un-cited, token-budget-only-at-render recall. This module is additive: it
reads the same underlying stores (``LessonMemory``, the optional vector
store) directly, because neither ``RecallBundle`` nor
``RecallBundle.to_prompt_block()`` carries per-line citations (lesson id
/ finding id) today, and the roadmap's close-out explicitly wants
citations in the router's output.

Knowledge-graph edges deferred, per the roadmap's own text -- this router
does keyword/overlap retrieval (reusing ``LessonMemory.find``'s existing
mechanism) scoped by dataset + a query derived from the node, not a graph
traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ledgerhound.memory.librarian import Librarian
    from ledgerhound.plan.graph import PlanNode


def node_query(node: Any) -> str:
    """Derive a free-text query from a plan node. Duck-typed against
    ``.text``/``.produces`` (both plain strings on PlanNode) since nothing
    more structured exists to key off of."""
    text = getattr(node, "text", "") or ""
    produces = getattr(node, "produces", "") or ""
    return f"{text} {produces}".strip()


@dataclass
class Citation:
    kind: str      # "lesson" | "finding"
    ref_id: str
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ref_id": self.ref_id, "text": self.text}


@dataclass
class Brief:
    node_id: str
    node_type: str
    dataset: str
    query: str
    token_budget: int
    text: str
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "node_type": self.node_type,
            "dataset": self.dataset, "query": self.query,
            "token_budget": self.token_budget, "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
        }


def render_brief(
    librarian: "Librarian",
    node: Any,
    dataset: str,
    token_budget: int = 400,
    max_lessons: int = 5,
    k_vectors: int = 5,
) -> Brief:
    """Build a node-scoped, cited, token-budget-capped brief.

    Priority order (highest first, matching RecallBundle's instinct >
    lessons > associative convention): a one-line "you're at ..." header
    (always included -- it's the minimum needed to say what the brief is
    about, so a pathologically tiny token_budget can't drop it entirely,
    but it DOES count against the budget for what follows), then lessons
    (shared before private, cited by Lesson.id), then associative vector
    hits if enabled (cited by MemoryRecord.id). Truncates by whole line,
    never mid-line: total length is bounded by
    ``max(token_budget*4, len(header))``, never unboundedly larger.
    """
    node_id = getattr(node, "id", "")
    node_type = getattr(node, "type", "")
    node_text = getattr(node, "text", "") or ""
    query = node_query(node)

    budget_chars = max(0, token_budget) * 4
    header = f"=== BRIEF for plan node {node_id} ({node_type}): {node_text[:120]} ==="
    lines = [header]
    used = len(header)
    citations: list[Citation] = []

    for store, scope in ((librarian.shared_lessons, "shared"),
                         (librarian.private_lessons, "private")):
        for lesson in store.find(dataset, query, max_items=max_lessons):
            line = f"  - [lesson:{lesson.id}] ({scope}) {lesson.text}"
            if used + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
            citations.append(Citation(kind="lesson", ref_id=lesson.id, text=lesson.text))

    if librarian.enable_vectors and query and used < budget_chars:
        try:
            vm = librarian.vectors()
            hits = vm.recall(query, namespace="text", k=k_vectors, dataset=dataset)
        except Exception:
            hits = []
        for hit in hits:
            rec = getattr(hit, "record", None)
            text = getattr(rec, "text", None) or str(hit)
            ref_id = getattr(rec, "id", None) or "unknown"
            line = f"  - [finding:{ref_id}] {text}"
            if used + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
            citations.append(Citation(kind="finding", ref_id=ref_id, text=text))

    if len(lines) == 1:
        lines.append("  (nothing relevant found for this node yet)")

    return Brief(
        node_id=node_id, node_type=node_type, dataset=dataset, query=query,
        token_budget=token_budget, text="\n".join(lines), citations=citations,
    )
