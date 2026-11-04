"""
ledgerhound.dictionary.tool
========================

Agent-callable wrappers over the Schema KB.

These are deliberately thin: each function maps one user intent
("look up this column", "decode this code") to one KB call, and returns
a JSON-serialisable dict. The agent's ReAct loop can call them
directly; the same functions are also useful from Python notebooks.

Tool registry
-------------

>>> from ledgerhound.dictionary import SchemaKB, load_defaults, make_dictionary_tools
>>> kb = SchemaKB()
>>> load_defaults(kb)
>>> tools = make_dictionary_tools(kb)
>>> tools["dict_lookup"]("payee_mobilenumber")
{'canonical_name': 'payee_mobilenumber', 'status': 'mandatory',
 'population': 'not_populated', 'quality_flags': [...], ...}

The schema descriptions in ``DICTIONARY_TOOL_SPECS`` are OpenAI-style
JSON-schema tool definitions, suitable for passing to a ``tools=`` field
on a chat-completion request.
"""

from __future__ import annotations

from typing import Callable, Optional

from .models import RelationKind
from .store import SchemaKB


# -- Tool implementations ----------------------------------


def _column_to_dict(kb: SchemaKB, name: str) -> dict:
    col = kb.lookup(name)
    if col is None:
        return {"error": f"No column named {name!r} or any alias of it."}
    qflags = [
        {
            "issue": qf.issue,
            "severity": qf.severity.value if hasattr(qf.severity, "value") else str(qf.severity),
            "source": qf.source_file,
        }
        for qf in kb.quality_flags(name)
    ]
    aliases = kb.aliases(name)
    return {
        "canonical_name": col.canonical_name,
        "db_alias": col.db_alias,
        "dict_name": col.dict_name,
        "dataset": col.dataset,
        "semantic_type": col.semantic_type,
        "role": col.role,
        "status": col.status.value if hasattr(col.status, "value") else str(col.status),
        "population": col.population.value if hasattr(col.population, "value") else str(col.population),
        "is_active": col.is_active,
        "comments": col.comments,
        "aliases": aliases,
        "quality_flags": qflags,
        "has_quality_concerns": col.has_quality_concerns,
    }


def make_dictionary_tools(kb: SchemaKB) -> dict[str, Callable]:
    """Return a dict of tool functions bound to ``kb``.

    Keys are the tool names exposed to the LLM. Values are zero/one-arg
    callables returning JSON-serialisable dicts.
    """

    def dict_lookup(name: str) -> dict:
        """Resolve any column name (canonical / db / dict) to its full record."""
        return _column_to_dict(kb, name)

    def dict_search(
        contains: Optional[str] = None,
        dataset: Optional[str] = None,
        semantic_type: Optional[str] = None,
        status: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> dict:
        """Search columns by substring / dataset / semantic_type / status."""
        results = kb.search(
            contains=contains, dataset=dataset, semantic_type=semantic_type,
            status=status, active_only=active_only,
        )
        return {
            "n_matches": len(results),
            "shown": min(len(results), limit),
            "columns": [
                {
                    "canonical_name": c.canonical_name,
                    "db_alias": c.db_alias,
                    "semantic_type": c.semantic_type,
                    "role": c.role,
                    "population": c.population.value if hasattr(c.population, "value") else str(c.population),
                    "is_active": c.is_active,
                }
                for c in results[:limit]
            ],
        }

    def dict_aliases(name: str) -> dict:
        """List every known alternative name for a column."""
        canonical = kb._resolve_canonical(name)
        if canonical is None:
            return {"error": f"No column named {name!r}."}
        return {"canonical_name": canonical, "aliases": kb.aliases(name)}

    def dict_quality_flags(name: str) -> dict:
        """Return documented data-quality issues for a column."""
        canonical = kb._resolve_canonical(name)
        if canonical is None:
            return {"error": f"No column named {name!r}."}
        return {
            "canonical_name": canonical,
            "flags": [
                {
                    "issue": qf.issue,
                    "severity": qf.severity.value if hasattr(qf.severity, "value") else str(qf.severity),
                    "source_file": qf.source_file,
                    "source_quote": qf.source_quote,
                }
                for qf in kb.quality_flags(name)
            ],
        }

    def dict_decode_code(code: str, code_type: str = "purpose_code") -> dict:
        """Translate a reference code (e.g. purpose code 41) to its meaning."""
        rc = kb.reference_code(code, code_type)
        if rc is None:
            return {"error": f"No {code_type} {code!r}."}
        return {
            "code": rc.code,
            "code_type": rc.code_type,
            "description": rc.description,
            "product": rc.product,
            "notes": rc.notes,
        }

    def dict_related(name: str, kind: Optional[str] = None) -> dict:
        """Return columns related to ``name``.

        ``kind`` filters by relationship type:
            same_entity | join_key | derived_from | semantic_pair
        """
        canonical = kb._resolve_canonical(name)
        if canonical is None:
            return {"error": f"No column named {name!r}."}
        rk = RelationKind(kind) if kind else None
        rows = kb.related(name, kind=rk)
        return {
            "canonical_name": canonical,
            "related": [
                {
                    "name": ci.canonical_name,
                    "kind": k.value if hasattr(k, "value") else str(k),
                    "notes": notes,
                    "semantic_type": ci.semantic_type,
                }
                for (ci, k, notes) in rows
            ],
        }

    def dict_stats() -> dict:
        """Quick health summary of the KB -- useful for sanity checks."""
        return kb.stats()

    return {
        "dict_lookup":         dict_lookup,
        "dict_search":         dict_search,
        "dict_aliases":        dict_aliases,
        "dict_quality_flags":  dict_quality_flags,
        "dict_decode_code":    dict_decode_code,
        "dict_related":        dict_related,
        "dict_stats":          dict_stats,
    }


# -- OpenAI-style tool specs (for chat-completions ``tools=``) --


DICTIONARY_TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "dict_lookup",
            "description": (
                "Look up a column in the Schema KB by any of its names "
                "(canonical, DB alias, or data-dictionary form). Returns the "
                "full record including data-quality flags. Use BEFORE writing "
                "analysis code to confirm the column exists, what it means, "
                "and whether it has known quality issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name in any form."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_search",
            "description": (
                "Search columns by substring or filters. Use to discover "
                "columns matching a concept (e.g. contains='amount')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contains": {"type": "string"},
                    "dataset": {"type": "string"},
                    "semantic_type": {"type": "string"},
                    "status": {"type": "string", "enum": ["mandatory", "optional", "new", "deprecated"]},
                    "active_only": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_quality_flags",
            "description": (
                "Return all documented data-quality issues for a column. "
                "ALWAYS call this before basing a finding on a column -- if "
                "severity is HIGH or CRITICAL, caveat or skip the analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_decode_code",
            "description": (
                "Translate a reference code (purpose_code, mcc, etc.) to its "
                "human description. Use whenever you see a numeric code in "
                "the data so findings read naturally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "code_type": {"type": "string", "default": "purpose_code"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_aliases",
            "description": "List every known alternative name for a column.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_related",
            "description": (
                "Find columns related to the given one. Use to discover "
                "join keys, mirror pairs (payer_X / payee_X), or aliases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["same_entity", "join_key", "derived_from", "semantic_pair"],
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dict_stats",
            "description": "Quick stats about the Schema KB -- counts of columns, flags, codes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
