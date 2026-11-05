"""
ledgerhound.kb.policy
=====================

Where Knowledge Pack entries become guardrail inputs (design doc table,
"Consumed by"):

    label_policy   -> maturity_for()        label-maturity guard (build_sample)
    known_finding  -> known_claims()        restatement guard (record_finding)
    rule           -> rule_outputs()        leakage guard (check_feature, roles)
    dictionary     -> role_hints()          role inference (load)

An entry with a `dataset` field applies only to a dataset of that name (file
name, file stem or team dataset id, case-insensitive); one without applies
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _applies(entry, names: Iterable[str]) -> bool:
    ds = str(entry.get("dataset") or "").strip().lower()
    if not ds:
        return True
    return ds in {str(n).strip().lower() for n in names if n}


def dataset_names(path: str = "", dataset_id: str = "") -> list[str]:
    out = [dataset_id]
    if path:
        p = Path(path)
        out += [p.name, p.stem]
    return [n for n in out if n]


@dataclass
class Maturity:
    days: int
    policy_id: str
    scope: str
    label_source: str
    fallback: bool

    def note(self) -> str:
        src = f"the {self.scope} label_policy '{self.policy_id}'"
        if self.fallback:
            return (f"{self.days} days, from {src} -- a fallback, because no policy "
                    f"names the label source {self.label_source!r}")
        return f"{self.days} days, from {src}"


def maturity_for(pack, label_source: str = "", names: Iterable[str] = ()) -> Maturity | None:
    """The label_policy for `label_source` (most specific scope wins), else
    the 'any' policy, flagged as a fallback; None if the pack has neither."""
    names = list(names)
    pols = [e for e in pack.entries(type="label_policy") if _applies(e, names)]
    src = (label_source or "").strip().lower()
    for e in pols:
        if src and e["label_source"] == src:
            return Maturity(e["maturity_days"], e["id"], e.scope, src, False)
    for e in pols:
        if e["label_source"] == "any":
            return Maturity(e["maturity_days"], e["id"], e.scope, src or "unknown", True)
    return None


def known_claims(pack, names: Iterable[str] = ()) -> list[tuple[str, str]]:
    """(name, statement) for the restatement guard."""
    names = list(names)
    return [(f"knowledge pack [{e.scope}]: {e['id']}", e["statement"])
            for e in pack.entries(type="known_finding") if _applies(e, names)]


def rule_outputs(pack, names: Iterable[str] = ()) -> dict[str, str]:
    """output column -> rule id, for rules not retired."""
    names = list(names)
    return {e["output_column"]: e["id"] for e in pack.entries(type="rule")
            if e.get("output_column") and e.get("status") != "retired" and _applies(e, names)}


def role_hints(pack, columns: Iterable[str], names: Iterable[str] = ()) -> dict[str, str]:
    """column -> role from dictionary entries that state one, plus every
    live rule's output column as rule_output."""
    names = list(names)
    cols = {str(c) for c in columns}
    hints = {e["column"]: e["role"] for e in pack.entries(type="dictionary")
             if e.get("role") and e["column"] in cols and _applies(e, names)}
    for col in rule_outputs(pack, names):
        if col in cols:
            hints[col] = "rule_output"
    return hints


def resolve_ref(pack, ref: str):
    """'type/id' or a bare id -> the effective entry, or None."""
    ref = (ref or "").strip()
    if "/" in ref:
        t, i = ref.split("/", 1)
        return pack.get(t.strip(), i.strip())
    hits = [e for e in pack.entries() if e["id"] == ref]
    return hits[0] if len(hits) == 1 else None
