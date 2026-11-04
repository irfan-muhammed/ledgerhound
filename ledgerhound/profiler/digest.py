"""
ledgerhound.profiler.digest
========================

The Digest is the single output artifact of the profiler -- a YAML file
containing the complete statistical X-ray of the dataset.

Phase B: now includes join_health section showing which FK->PK
relationships are healthy, broken, or unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ledgerhound.profiler.detectors.quality import QualityReport
from ledgerhound.profiler.detectors.statistics import StatsReport
from ledgerhound.profiler.detectors.priority_scorer import PriorityRanking
from ledgerhound.schema.models import DatasetSchema


@dataclass
class Digest:
    """Complete statistical digest of a dataset."""

    meta: dict = field(default_factory=dict)
    overview: dict = field(default_factory=dict)
    data_quality: dict = field(default_factory=dict)
    column_priority: dict = field(default_factory=dict)
    column_profiles: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)
    dependencies: dict = field(default_factory=dict)

    _FIELDS = ("meta", "overview", "data_quality", "column_priority",
               "column_profiles", "relationships", "dependencies")

    def save(self, path: str | Path):
        """Write digest to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f,
                      default_flow_style=False, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def load(cls, path: str | Path) -> Digest:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**{k: data.get(k, {}) for k in cls._FIELDS})

    def summary(self) -> str:
        """Human-readable one-screen summary for CLI output."""
        o = self.overview
        dq = self.data_quality
        cp = self.column_priority
        lines = [
            f"Dataset: {o.get('source', 'unknown')}",
            f"Rows: {o.get('rows', 0):,}",
            f"Columns: {o.get('columns', 0)}",
        ]
        if o.get("target") and o["target"] != "none":
            lines.append(f"Target: {o['target']}")
        lines.append(f"Quality score: {dq.get('score', '?')}/100")
        if dq.get("critical"):
            lines.append(f"Critical issues: {len(dq['critical'])}")
        if cp.get("drop_candidates"):
            lines.append(f"Drop candidates: {', '.join(cp['drop_candidates'])}")
        if cp.get("high"):
            lines.append(f"High-signal columns: {', '.join(cp['high'][:8])}")

        # Broken joins
        joins = self.relationships.get("join_health", [])
        broken = [j for j in joins if j.get("status") == "broken"]
        if broken:
            lines.append(f"Broken joins: {len(broken)}")
            for j in broken:
                lines.append(f"  ! {j['from_col']} -> {j['to_table']}.{j['to_col']}: {j['reason']}")

        return "\n".join(lines)


class DigestFormatter:
    """Assemble all profiler outputs into a ``Digest``."""

    def format(self, *, schema: DatasetSchema, quality: QualityReport,
               stats: StatsReport, priority: PriorityRanking,
               source: str = "", target_col: str | None = None) -> Digest:
        d = Digest()

        d.meta = {"generated": datetime.now().isoformat(timespec="seconds"),
                  "engine": "ledgerhound-0.1", "source": source}

        d.overview = {
            "source": source, "tables": len(schema.tables),
            "primary_table": schema.primary_table,
            "rows": quality.total_rows, "columns": quality.total_columns,
            "target": target_col or "none",
        }

        d.data_quality = {
            "score": quality.score,
            "empty_columns": quality.empty_columns,
            "critical": [{"column": i.column, "issue": i.issue, "action": i.action}
                         for i in quality.critical],
            "warnings": [{"column": i.column, "issue": i.issue, "action": i.action}
                         for i in quality.warnings],
            "info": [{"column": i.column, "issue": i.issue} for i in quality.info],
        }

        d.column_priority = {
            "high": priority.high, "medium": priority.medium,
            "low": priority.low, "identifiers": priority.identifiers,
            "hashed_pii": priority.hashed_pii,
            "drop_candidates": priority.drop_candidates,
        }

        primary = schema.primary
        for cname, cs in stats.columns.items():
            entry: dict[str, Any] = {"semantic_type": cs.semantic_type, "stats": cs.profile}
            if primary and cname in primary.columns:
                sc = primary.columns[cname]
                if sc.description: entry["description"] = sc.description
                if sc.transform:   entry["transform"] = sc.transform
                if sc.display_unit: entry["display_unit"] = sc.display_unit
                if sc.tags:        entry["tags"] = sc.tags
            if cs.by_target: entry["by_target"] = cs.by_target
            if cs.ks_test:   entry["ks_test"] = cs.ks_test
            d.column_profiles[cname] = entry

        # Relationships + join health
        d.relationships = {
            "defined": [
                {"from": f"{r.from_table}.{r.from_column}",
                 "to": f"{r.to_table}.{r.to_column}"}
                for r in schema.relationships
            ],
            "join_health": [
                {"from_col": j.from_col, "to_table": j.to_table,
                 "to_col": j.to_col, "status": j.status,
                 "reason": j.reason}
                for j in quality.join_health
            ],
        }

        d.dependencies = {
            "available": [
                {"table": t.name, "alias": a, "columns": len(t.columns)}
                for a, t in schema.tables.items() if a != schema.primary_table
            ],
            "missing": [],
        }
        return d
