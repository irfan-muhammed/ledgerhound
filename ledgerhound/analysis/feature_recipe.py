"""
ledgerhound.analysis.feature_recipe
================================

The feature recipe (``feature_recipe/v1``) + a persistent recipe library
(rank 9). A recipe is a *reproducible, evaluated* feature definition -- the
durable output of the Feature-Engineering agent, deposited so future runs and
other datasets can reuse a winning feature instead of rediscovering it.

A recipe carries its build definition (transform kind + inputs + params), its
measured strength (IV / KS / AUC from feature_eval), a leakage caveat, and
provenance (dataset + owner + when). The library is an append-only JSONL keyed
by recipe name; re-depositing a name updates it in place (idempotent).

This is deliberately separate from FactMemory's numeric baselines: recipes are
*feature definitions*, not run statistics -- though a recipe's metrics can be
corroborated across runs the same way facts are.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = "feature_recipe/v1"

# Same safe, exploratory transform vocabulary as analysis.eda_derive, plus
# 'raw' for a feature that is already a column. Production-grade transforms
# beyond these are the FE team's to implement with leakage/point-in-time rigor.
_KINDS = {"raw", "date_diff", "ratio", "log", "hour"}


@dataclass
class FeatureRecipe:
    """A reproducible, evaluated feature definition."""

    name: str
    kind: str = "raw"
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    # measured strength (from feature_eval)
    iv: float = 0.0
    ks: float = 0.0
    auc: float = 0.5
    iv_band: str = "useless"
    # provenance + safety
    dataset: str = ""
    owner: str = ""
    leakage_reviewed: bool = False
    leakage_note: str = (
        "EDA-derived strength only; NOT leakage / point-in-time validated. "
        "FE team must re-implement with as-of-transaction correctness."
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = SCHEMA

    def validate(self) -> "FeatureRecipe":
        if not self.name.strip():
            raise ValueError("recipe 'name' is required")
        if self.kind not in _KINDS:
            raise ValueError(f"recipe {self.name!r}: unknown kind {self.kind!r} "
                             f"(allowed: {sorted(_KINDS)})")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureRecipe":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class RecipeLibrary:
    """Append-only, name-keyed store of feature recipes (JSONL on disk)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._recipes: dict[str, FeatureRecipe] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = FeatureRecipe.from_dict(json.loads(line))
                self._recipes[r.name] = r  # last write wins
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping bad recipe line: %s", exc)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for r in self._recipes.values():
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    # -- API ----------------------------------------------------------------

    def deposit(self, recipe: FeatureRecipe) -> FeatureRecipe:
        """Add or update a recipe (idempotent by name). Persists immediately."""
        recipe.validate()
        self._recipes[recipe.name] = recipe
        self._flush()
        return recipe

    def get(self, name: str) -> Optional[FeatureRecipe]:
        return self._recipes.get(name)

    def all(self) -> list[FeatureRecipe]:
        return list(self._recipes.values())

    def best(self, *, dataset: Optional[str] = None,
             limit: int = 20, min_band: str = "weak") -> list[FeatureRecipe]:
        """Recipes ranked by IV, optionally filtered to a dataset + min band."""
        order = ["useless", "weak", "medium", "strong", "suspicious"]
        floor = order.index(min_band) if min_band in order else 0
        out = [r for r in self._recipes.values()
               if (dataset is None or r.dataset == dataset)
               and order.index(r.iv_band) >= floor
               and r.iv_band != "suspicious"]  # exclude likely-leaky
        out.sort(key=lambda r: -abs(r.iv))
        return out[:limit]

    def stats(self) -> dict[str, int]:
        bands: dict[str, int] = {}
        for r in self._recipes.values():
            bands[r.iv_band] = bands.get(r.iv_band, 0) + 1
        return {"recipes": len(self._recipes), **bands}
