"""
sentinel.eval.golden
====================

Load golden evaluation tasks (per GOLDEN_DATASET_SPEC.md) from YAML or CSV.

A golden task pairs a natural-language goal + dataset with KNOWN-CORRECT
answers expressed as recomputable `facts`, expected findings, trajectory
expectations, and hallucination traps.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenFact:
    id: str
    description: str
    computation: str          # runnable Python that sets `result` (df in scope)
    expected_value: object
    tolerance: float = 0.0
    unit: str = ""


@dataclass
class GoldenFinding:
    text: str
    must_include_terms: list[str] = field(default_factory=list)
    linked_fact: str = ""
    importance: str = "medium"


@dataclass
class GoldenDataset:
    name: str
    path: str
    rows: int | None = None
    cols: int | None = None
    description: str = ""


@dataclass
class GoldenTask:
    id: str
    goal: str
    category: str = "eda"
    difficulty: str = "medium"
    datasets: list[GoldenDataset] = field(default_factory=list)
    facts: list[GoldenFact] = field(default_factory=list)
    expected_findings: list[GoldenFinding] = field(default_factory=list)
    expected_capabilities: list[dict] = field(default_factory=list)
    must_not_claim: list[dict] = field(default_factory=list)
    expected_deliverables: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenTask":
        return cls(
            id=d["id"],
            goal=d.get("goal", ""),
            category=d.get("category", "eda"),
            difficulty=d.get("difficulty", "medium"),
            datasets=[GoldenDataset(**{k: v for k, v in ds.items()
                                       if k in GoldenDataset.__dataclass_fields__})
                      for ds in d.get("datasets", [])],
            facts=[GoldenFact(**{k: v for k, v in f.items()
                                 if k in GoldenFact.__dataclass_fields__})
                   for f in d.get("facts", [])],
            expected_findings=[GoldenFinding(**{k: v for k, v in f.items()
                                                if k in GoldenFinding.__dataclass_fields__})
                               for f in d.get("expected_findings", [])],
            expected_capabilities=d.get("expected_capabilities", []),
            must_not_claim=d.get("must_not_claim", []),
            expected_deliverables=d.get("expected_deliverables", {}),
            metadata=d.get("metadata", {}),
        )


def load_golden(path: str | Path) -> list[GoldenTask]:
    """Load golden tasks from a YAML file/dir or a facts-only CSV."""
    p = Path(path)
    if p.is_dir():
        tasks: list[GoldenTask] = []
        for f in sorted(p.rglob("*.yaml")) + sorted(p.rglob("*.yml")):
            tasks.extend(_load_yaml(f))
        for f in sorted(p.rglob("*.csv")):
            tasks.extend(_load_csv(f))
        return tasks
    if p.suffix in (".yaml", ".yml"):
        return _load_yaml(p)
    if p.suffix == ".csv":
        return _load_csv(p)
    raise ValueError(f"Unsupported golden format: {p}")


def _load_yaml(path: Path) -> list[GoldenTask]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return []
    items = data if isinstance(data, list) else [data]
    return [GoldenTask.from_dict(d) for d in items]


def _load_csv(path: Path) -> list[GoldenTask]:
    """Facts-only CSV: columns task_id,fact_id,description,computation,
    expected_value,tolerance,unit. Tasks are grouped by task_id."""
    by_task: dict[str, GoldenTask] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["task_id"].strip()
            t = by_task.setdefault(tid, GoldenTask(id=tid, goal=""))
            ev_raw = row.get("expected_value", "").strip()
            try:
                ev: object = float(ev_raw) if "." in ev_raw else int(ev_raw)
            except ValueError:
                ev = ev_raw
            try:
                tol = float(row.get("tolerance", "0") or 0)
            except ValueError:
                tol = 0.0
            t.facts.append(GoldenFact(
                id=row["fact_id"].strip(),
                description=row.get("description", "").strip(),
                computation=row["computation"],
                expected_value=ev,
                tolerance=tol,
                unit=row.get("unit", "").strip(),
            ))
    return list(by_task.values())
