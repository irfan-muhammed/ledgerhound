"""
sentinel.profiler.detectors.priority_scorer
============================================

Ranks columns by their likely value for fraud modeling.
High-signal columns are loaded first into the LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.schema.models import ColumnSchema, ColumnRole, SemanticType, PIIStatus
from sentinel.profiler.detectors.statistics import ColumnStats


@dataclass
class PriorityRanking:
    """Columns bucketed by importance."""
    high: list[str] = field(default_factory=list)
    medium: list[str] = field(default_factory=list)
    low: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    hashed_pii: list[str] = field(default_factory=list)
    drop_candidates: list[str] = field(default_factory=list)


class PriorityScorer:
    """Score and rank columns for context-window prioritisation."""

    _ROLE_SCORES = {
        ColumnRole.TARGET: 10, ColumnRole.AMOUNT: 8, ColumnRole.STATUS: 8,
        ColumnRole.TIMESTAMP: 7, ColumnRole.RESPONSE_CODE: 6,
        ColumnRole.ENTITY_ID: 5, ColumnRole.FEATURE: 5, ColumnRole.METADATA: 2,
    }

    def score(self, columns: dict[str, ColumnSchema],
              stats: dict[str, ColumnStats],
              empty_columns: list[str] | None = None) -> PriorityRanking:
        ranking = PriorityRanking()
        scored: list[tuple[str, float]] = []
        _empty = set(empty_columns or [])

        for cname, scol in columns.items():
            # Dead columns detected by quality detector -> drop
            if cname in _empty:
                ranking.drop_candidates.append(cname); continue
            # Route special categories first
            if scol.pii_status == PIIStatus.HASHED:
                ranking.hashed_pii.append(cname); continue
            if scol.semantic_type == SemanticType.IDENTIFIER or scol.role == ColumnRole.PRIMARY_KEY:
                ranking.identifiers.append(cname); continue

            cs = stats.get(cname)
            if cs and cs.profile.get("unique", cs.profile.get("unique_count", 999)) <= 1:
                ranking.drop_candidates.append(cname); continue

            scored.append((cname, self._compute(scol, cs)))

        scored.sort(key=lambda x: x[1], reverse=True)
        for cname, sc in scored:
            (ranking.high if sc >= 7 else ranking.medium if sc >= 4 else ranking.low).append(cname)

        return ranking

    def _compute(self, scol: ColumnSchema, cs: ColumnStats | None) -> float:
        s = float(self._ROLE_SCORES.get(scol.role, 3))
        if scol.semantic_type == SemanticType.NUMERIC:
            s += 1
        if scol.semantic_type == SemanticType.BINARY:
            s += 1
        if cs and cs.ks_test:
            ks = cs.ks_test.get("statistic", 0)
            s += 3 if ks > 0.3 else 2 if ks > 0.1 else 1 if ks > 0.05 else 0
        if cs:
            null_pct = cs.profile.get("null_pct", 0)
            if null_pct > 50: s -= 2
            elif null_pct > 20: s -= 1
        return s
