"""
ledgerhound.kb.schema
=====================

The Knowledge Pack's structured entry types (design doc, "Knowledge Pack")
and their validation. Validation is strict -- an unknown field is an error,
not ignored -- because structured entries feed guardrails directly, and a
typo'd `maturity_days` silently ignored would quietly switch a guard off.

    typology       a fraud pattern: signals, roles it needs, how to falsify it
    rule           a live detection rule; its output column is never a feature
    dictionary     what a column means; may state its role
    known_finding  something already established (restatement guard)
    label_policy   when labels from a source are mature
    document       an uploaded PDF / Markdown / text file (search only)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

TYPES = ("typology", "rule", "dictionary", "known_finding", "label_policy", "document")

_COMMON = {"id": str, "type": str, "added_by": str, "added_at": str, "scope": str,
           "tags": list}

SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "typology": {
        "required": {"name": str, "description": str},
        "optional": {"rails": list, "signals": list, "roles_needed": list,
                     "falsification": list, "references": list},
    },
    "rule": {
        "required": {"name": str, "logic": str},
        "optional": {"output_column": str, "threshold": (str, int, float),
                     "status": str, "notes": str},
    },
    "dictionary": {
        "required": {"column": str, "meaning": str},
        "optional": {"role": str, "codes": dict, "quality_issues": list, "dataset": str},
    },
    "known_finding": {
        "required": {"statement": str},
        "optional": {"dataset": str, "evidence": str, "source": str},
    },
    "label_policy": {
        "required": {"label_source": str, "maturity_days": int},
        "optional": {"notes": str, "dataset": str},
    },
    "document": {
        "required": {"title": str, "file": str},
        "optional": {"source": str, "sha256": str, "chars": int},
    },
}

# the field an id is derived from when none is given
_ID_FROM = {"typology": "name", "rule": "name", "dictionary": "column",
            "known_finding": "statement", "label_policy": "label_source",
            "document": "title"}
RULE_STATUS = ("live", "retired", "shadow")


def slug(text: str, n: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:n].strip("_") or "entry"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__


def validate(entry: Any) -> tuple[dict, list[str]]:
    """Return (normalised entry, errors). Errors name the field and the fix;
    an entry with errors must not be stored."""
    from ledgerhound.roles import ROLES
    if not isinstance(entry, dict):
        return {}, ["an entry must be a mapping (JSON object / YAML mapping)"]
    e = dict(entry)
    t = str(e.get("type", "")).strip().lower()
    if t not in SCHEMAS:
        return e, [f"type must be one of {', '.join(TYPES)} (got {e.get('type')!r})"]
    e["type"] = t
    spec = SCHEMAS[t]
    errors: list[str] = []
    allowed = {**_COMMON, **spec["required"], **spec["optional"]}
    for k in e:
        if k not in allowed:
            errors.append(f"{t}: unknown field {k!r} (allowed: {', '.join(sorted(allowed))})")
    for k in spec["required"]:
        v = e.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            errors.append(f"{t}: missing required field {k!r}")
    for k, typ in allowed.items():
        if k not in e or e[k] is None:
            continue
        v = e[k]
        if typ is int and isinstance(v, bool):
            errors.append(f"{t}: {k} must be an integer")
        elif not isinstance(v, typ):
            errors.append(f"{t}: {k} must be {_type_name(typ)}, got {type(v).__name__}")
    if errors:
        return e, errors

    for k in ("rails", "signals", "falsification", "quality_issues", "tags"):
        if k in e and not all(isinstance(x, str) and x.strip() for x in e[k]):
            errors.append(f"{t}: every item of {k} must be a non-empty string")
    for ref in e.get("references", []) or []:
        if not (isinstance(ref, str) or (isinstance(ref, dict) and "title" in ref)):
            errors.append(f"{t}: references are strings or {{title, url}} mappings")
    for r in e.get("roles_needed", []) or []:
        if r not in ROLES:
            errors.append(f"{t}: roles_needed has {r!r}, which is not a role")
    if t == "dictionary" and e.get("role") and e["role"] not in ROLES:
        errors.append(f"dictionary: role {e['role']!r} is not a role ({', '.join(ROLES)})")
    if t == "label_policy" and not 0 <= e["maturity_days"] <= 1000:
        errors.append("label_policy: maturity_days must be between 0 and 1000")
    if t == "rule" and e.get("status") and e["status"] not in RULE_STATUS:
        errors.append(f"rule: status must be one of {', '.join(RULE_STATUS)}")
    if t == "label_policy":
        e["label_source"] = e["label_source"].strip().lower()
    e["id"] = slug(e.get("id") or e[_ID_FROM[t]])
    return e, errors


def describe(e: dict) -> str:
    """One line for listings and search hits."""
    t = e.get("type", "")
    if t == "typology":
        return f"{e['name']}: {e['description']}"
    if t == "rule":
        out = f" (output column {e['output_column']})" if e.get("output_column") else ""
        return f"rule {e['name']}{out}: {e['logic']}"
    if t == "dictionary":
        role = f" [role {e['role']}]" if e.get("role") else ""
        return f"column {e['column']}{role}: {e['meaning']}"
    if t == "known_finding":
        return f"known: {e['statement']}"
    if t == "label_policy":
        return f"labels from {e['label_source']} mature after {e['maturity_days']} days"
    if t == "document":
        return f"document {e['title']}" + (f" ({e['source']})" if e.get("source") else "")
    return str(e)
