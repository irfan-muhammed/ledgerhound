"""
ledgerhound.knowledge.loader
=========================

Loads domain knowledge (YAML files) and retrieves relevant items
for the agent prompt based on topic tags and column names.

Knowledge types
---------------
* **Business rules** -- filters, conventions, join conditions.
* **Analyst FAQs** -- investigation patterns with why/how/good/bad.
* **Column semantics** -- what columns mean (raw UPI data).
* **Feature rationale** -- WHY each engineered feature matters for fraud.
* **Playbooks** -- ordered sequences of analysis questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class KnowledgeItem:
    """A single piece of retrieved knowledge."""
    type: str          # "faq", "pattern", "column_semantic", "feature_rationale"
    content: dict
    tags: list[str] = field(default_factory=list)
    priority: str = "medium"


class KnowledgeBase:
    """Load YAML knowledge files and retrieve relevant items by tag."""

    def __init__(self, knowledge_dir: str | Path | None = None):
        self.rules: dict = {}
        self.faqs: list[dict] = []
        self.patterns: list[dict] = []
        self.semantics: dict[str, dict] = {}       # raw UPI column meanings
        self.rationale: dict[str, dict] = {}        # feature fraud rationale
        self.playbooks: dict[str, dict] = {}

        if knowledge_dir:
            self.load(Path(knowledge_dir))

    def load(self, directory: Path):
        """Recursively load all *.yaml files from directory."""
        if not directory.exists():
            return
        for f in directory.rglob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                stem = f.stem.lower()

                if "rule" in stem:
                    self.rules.update(data)
                elif "faq" in stem:
                    self.faqs.extend(data.get("faqs", []))
                elif "pattern" in stem:
                    self.patterns.extend(data.get("patterns", []))
                elif "playbook" in stem:
                    self.playbooks[data.get("name", stem)] = data
                elif "rationale" in stem or ("feature" in stem and "playbook" not in stem):
                    # Feature fraud rationale (from the feature CSV)
                    self.rationale.update(data.get("columns", {}))
                elif "semantic" in stem:
                    self.semantics.update(data.get("columns", {}))
            except Exception:
                pass

    def load_playbook(self, path: str | Path) -> dict:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        name = data.get("name", Path(path).stem)
        self.playbooks[name] = data
        return data

    # -- retrieval ------------------------------------------

    def retrieve(self, topic_tags: list[str] | None = None,
                 columns: list[str] | None = None,
                 max_items: int = 10,
                 dataset_columns: set[str] | None = None) -> list[KnowledgeItem]:
        """Return the most relevant knowledge items for the given context.

        ``dataset_columns``, when given, is the set of columns that ACTUALLY
        exist in the dataset being profiled. Column-scoped knowledge
        (semantics / fraud rationale) whose column isn't in that set is
        dropped -- ``columns`` here is a keyword-matched CANDIDATE list
        (e.g. "payee" -> UPI column names) that may not correspond to the
        current dataset at all, and without this guard the tag-fallback
        below injects domain FAQ text for columns that don't exist,
        which an agent then reports as this dataset's real schema.
        """
        items: list[KnowledgeItem] = []
        tags = set(topic_tags or [])

        # FAQs matched by tag
        for faq in self.faqs:
            if tags & set(faq.get("tags", [])):
                items.append(KnowledgeItem("faq", faq, faq.get("tags", []),
                                            faq.get("priority", "medium")))

        # Patterns matched by tag
        for pat in self.patterns:
            if tags & set(pat.get("tags", [])):
                items.append(KnowledgeItem("pattern", pat, pat.get("tags", [])))

        for col in (columns or []):
            # Column semantics (raw UPI)
            if col in self.semantics:
                items.append(KnowledgeItem("column_semantic",
                                            {"column": col, **self.semantics[col]}, [col]))
            # Feature fraud rationale (behavioral features)
            if col in self.rationale:
                items.append(KnowledgeItem("feature_rationale",
                                            {"column": col, **self.rationale[col]},
                                            self.rationale[col].get("tags", []),
                                            "high"))  # fraud rationale is always high priority

        # Also retrieve rationale by tag match (not just column name)
        if tags and not columns:
            for col_name, r in self.rationale.items():
                rtags = set(r.get("tags", []))
                if tags & rtags:
                    items.append(KnowledgeItem("feature_rationale",
                                                {"column": col_name, **r}, list(rtags)))

        # Drop column-scoped knowledge for columns absent from the actual
        # dataset (only applies to column_semantic / feature_rationale --
        # FAQ/pattern items are topic-general, not schema claims).
        if dataset_columns is not None:
            items = [it for it in items
                     if it.type not in ("column_semantic", "feature_rationale")
                     or it.content.get("column") in dataset_columns]

        # Deduplicate by column name
        seen = set()
        unique = []
        for it in items:
            key = (it.type, it.content.get("column", it.content.get("question", id(it))))
            if key not in seen:
                seen.add(key)
                unique.append(it)

        # Sort by priority
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        unique.sort(key=lambda x: order.get(x.priority, 2))
        return unique[:max_items]

    def get_playbook(self, name: str) -> dict | None:
        return self.playbooks.get(name)

    def list_playbooks(self) -> list[str]:
        return list(self.playbooks.keys())

    def get_rules(self, section: str | None = None) -> dict:
        return self.rules.get(section, {}) if section else self.rules

    def format_for_prompt(self, items: list[KnowledgeItem]) -> str:
        """Render retrieved items as a string for the LLM prompt."""
        if not items:
            return ""
        parts = []
        for it in items:
            c = it.content
            if it.type == "faq":
                parts.append(
                    f"[Analyst FAQ - {c.get('priority','medium')}]\n"
                    f"Q: {c.get('question','')}\n"
                    f"Why: {c.get('why','')}\n"
                    f"How: {c.get('how','')}\n"
                    f"Good: {c.get('good_result','')}\n"
                    f"Bad: {c.get('bad_result','')}")
            elif it.type == "pattern":
                parts.append(
                    f"[Pattern: {c.get('name','')}]\n"
                    f"{c.get('description','')}\n"
                    f"Detection: {c.get('detection_logic','')}")
            elif it.type == "column_semantic":
                parts.append(
                    f"[Column: {c.get('column','')}]\n"
                    f"Meaning: {c.get('meaning','')}\n"
                    f"Fraud relevance: {c.get('fraud_relevance','?')}\n"
                    f"Notes: {c.get('notes','')}")
            elif it.type == "feature_rationale":
                parts.append(
                    f"[Feature: {c.get('column','')}]\n"
                    f"Description: {c.get('description','')}\n"
                    f"Fraud signal: {c.get('fraud_rationale','')}")
        return "\n\n".join(parts)
