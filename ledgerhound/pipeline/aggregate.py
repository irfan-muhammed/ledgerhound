"""
ledgerhound.pipeline.aggregate
============================

**Aggregation engine** (V4-013) -- turn fetched entity history into
account-grain and transaction-grain feature tables.

Two execution paths, and the choice is not a tuning knob
---------------------------------------------------------
``PUSHDOWN`` renders the aggregation as Trino SQL so 8 crore rows are never
materialised locally. ``CHUNKED`` streams frames and combines partial
results, for the case where the analyst already has the data on disk.

There is deliberately no GPU path: with no GPU available the architecture is
chunked/Trino-side by elimination (see ``docs/v4.0/next_steps_v4.md`` V4-003),
not a variant of a local in-memory groupby.

The correctness trap this module exists to close
-------------------------------------------------
**Not every aggregation is chunk-composable.** ``sum``/``count``/``min``/
``max`` combine across chunks by construction, and ``mean`` does if you carry
(sum, count) and divide at the end. ``median``, ``nunique``, ``quantile``,
``stddev`` and ``mode`` do *not* -- they need the whole group at once, or an
approximation with error bounds nobody asked for.

A naive chunked implementation computes those per chunk and averages them,
which is **silently wrong** and looks perfectly plausible in a report. So the
composable set is explicit here, and requesting anything outside it in
``CHUNKED`` mode raises rather than approximates. Those aggregations are
exactly what ``PUSHDOWN`` is for -- Trino computes them over the full group.

Point-in-time correctness
-------------------------
``as_of`` is a hard cutoff, not a hint: every path filters to
``timestamp < as_of`` before aggregating, half-open so adjacent periods can
never double-count a boundary transaction. Aggregating an entity's history
without it silently folds post-decision behaviour into a feature that claims
to describe the pre-decision moment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ledgerhound.pipeline.history_fetch import _quote_ident, sql_literal

log = logging.getLogger(__name__)

SCHEMA = "aggregation/v1"

# Exactly composable across chunks. Anything not here must go PUSHDOWN.
CHUNK_COMPOSABLE = frozenset({"sum", "count", "min", "max", "mean"})

# Known-but-refused in CHUNKED mode: each needs the whole group at once.
# Listed explicitly so the error can name the reason instead of just "unknown".
NEEDS_FULL_GROUP = frozenset({"median", "nunique", "quantile", "stddev", "mode"})

SUPPORTED = CHUNK_COMPOSABLE | NEEDS_FULL_GROUP

MODE_PUSHDOWN = "pushdown"
MODE_CHUNKED = "chunked"

_SQL_FUNC = {
    "sum": "sum", "count": "count", "min": "min", "max": "max",
    "mean": "avg", "median": "approx_percentile", "nunique": "count",
    "quantile": "approx_percentile", "stddev": "stddev", "mode": "mode",
}


class NotChunkComposable(ValueError):
    """A requested aggregation cannot be computed correctly from chunks."""


@dataclass(frozen=True)
class AggSpec:
    """One output feature: ``func`` of ``source`` per group, named ``name``."""

    name: str
    source: str
    func: str
    quantile: Optional[float] = None   # required for func="quantile"

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.name.strip():
            problems.append("AggSpec.name must not be empty")
        if not self.source.strip():
            problems.append(f"AggSpec[{self.name}].source must not be empty")
        if self.func not in SUPPORTED:
            problems.append(
                f"AggSpec[{self.name}].func {self.func!r} is not supported; "
                f"expected one of {sorted(SUPPORTED)}"
            )
        if self.func == "quantile" and self.quantile is None:
            problems.append(f"AggSpec[{self.name}] func='quantile' requires quantile=")
        if self.quantile is not None and not (0.0 < self.quantile < 1.0):
            problems.append(
                f"AggSpec[{self.name}].quantile must be in (0,1); got {self.quantile}"
            )
        return problems

    @property
    def is_chunk_composable(self) -> bool:
        return self.func in CHUNK_COMPOSABLE


@dataclass
class AggregationPlan:
    """Group-by keys + the features to compute, with a point-in-time cutoff."""

    group_keys: list[str]
    specs: list[AggSpec]
    timestamp_column: str = ""
    as_of: Any = None

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.group_keys:
            problems.append(
                "group_keys is empty -- an aggregation with no grain would "
                "collapse every entity into one row"
            )
        if not self.specs:
            problems.append("no AggSpecs -- nothing would be computed")
        seen: set[str] = set()
        for s in self.specs:
            problems.extend(s.validate())
            if s.name in seen:
                problems.append(f"duplicate output feature name {s.name!r}")
            seen.add(s.name)
            if s.name in self.group_keys:
                problems.append(
                    f"output feature {s.name!r} collides with a group key"
                )
        if (self.as_of is not None) != bool(self.timestamp_column.strip()):
            problems.append(
                "as_of and timestamp_column must be set together -- a "
                "point-in-time cutoff without a timestamp column silently "
                "does nothing"
            )
        return problems

    @property
    def non_composable(self) -> list[AggSpec]:
        return [s for s in self.specs if not s.is_chunk_composable]

    def require_chunk_composable(self) -> None:
        """Raise unless every spec can be computed correctly from chunks."""
        bad = self.non_composable
        if not bad:
            return
        detail = ", ".join(f"{s.name}({s.func})" for s in bad)
        raise NotChunkComposable(
            f"these aggregations need the whole group at once and cannot be "
            f"combined from chunks: {detail}. Computing them per chunk and "
            f"merging would be silently wrong. Use mode='{MODE_PUSHDOWN}' so "
            f"Trino computes them over the full group, or drop them."
        )


# ---------------------------------------------------------------------------
# PUSHDOWN -- render as Trino SQL
# ---------------------------------------------------------------------------


def _agg_sql(spec: AggSpec) -> str:
    col = _quote_ident(spec.source)
    fn = _SQL_FUNC[spec.func]
    if spec.func == "nunique":
        inner = f"count(DISTINCT {col})"
    elif spec.func == "median":
        inner = f"approx_percentile({col}, 0.5)"
    elif spec.func == "quantile":
        inner = f"approx_percentile({col}, {spec.quantile!r})"
    else:
        inner = f"{fn}({col})"
    return f"{inner} AS {_quote_ident(spec.name)}"


def build_aggregation_sql(plan: AggregationPlan, source_table: str) -> str:
    """Render the plan as a single Trino GROUP BY statement.

    Used for the crore-scale path: the aggregation happens in the warehouse
    and only the (much smaller) grouped result crosses the wire.
    """
    problems = plan.validate()
    if problems:
        raise ValueError("invalid AggregationPlan: " + "; ".join(problems))
    if not source_table.strip():
        raise ValueError("source_table must not be empty")

    keys = ", ".join(_quote_ident(k) for k in plan.group_keys)
    aggs = ",\n       ".join(_agg_sql(s) for s in plan.specs)
    sql = f"SELECT {keys},\n       {aggs}\nFROM {source_table}"
    if plan.as_of is not None:
        ts = _quote_ident(plan.timestamp_column)
        sql += f"\nWHERE {ts} < {sql_literal(plan.as_of)}"
    sql += f"\nGROUP BY {keys}"
    return sql


# ---------------------------------------------------------------------------
# CHUNKED -- stream frames, combine partials
# ---------------------------------------------------------------------------

# Per-spec partial state -> the pandas agg needed to produce it.
_PARTIAL_AGGS = {
    "sum": ("sum",), "count": ("count",), "min": ("min",), "max": ("max",),
    "mean": ("sum", "count"),
}

# Integer-valued partials, restored to an integer dtype at finalize. Combining
# partials can introduce NaN (a group present in one chunk and absent from
# another), which promotes to float -- so without this a count column's dtype
# would depend on how the input happened to be chunked.
_INTEGRAL_PARTIALS = frozenset({"count"})


def _combine_partial(acc, part, kind: str):
    """Merge two partials of the same kind, aligning on group keys.

    sum/count add with a 0 fill so a group missing from one side is not
    dropped; min/max concatenate and reduce across the axis, which skips NaN
    and therefore ignores the missing side rather than treating it as zero.
    """
    import pandas as pd

    if kind in ("sum", "count"):
        return acc.add(part, fill_value=0)
    both = pd.concat([acc, part], axis=1)
    return both.min(axis=1) if kind == "min" else both.max(axis=1)


def _apply_as_of(chunk, plan: AggregationPlan):
    if plan.as_of is None:
        return chunk
    ts = plan.timestamp_column
    if ts not in chunk.columns:
        raise KeyError(
            f"timestamp_column {ts!r} not in chunk; cannot enforce the "
            f"point-in-time cutoff, and proceeding would leak future data"
        )
    import pandas as pd
    col = pd.to_datetime(chunk[ts], errors="coerce")
    return chunk[col < pd.Timestamp(plan.as_of)]


def aggregate_chunked(chunks: Iterable, plan: AggregationPlan):
    """Aggregate an iterable of frames into one grouped result.

    Raises :class:`NotChunkComposable` if any requested aggregation cannot be
    computed correctly this way -- see the module docstring.
    """
    import pandas as pd

    problems = plan.validate()
    if problems:
        raise ValueError("invalid AggregationPlan: " + "; ".join(problems))
    plan.require_chunk_composable()

    # (source, partial_kind) -> accumulated frame indexed by group keys
    partials: dict[tuple[str, str], Any] = {}
    needed: set[tuple[str, str]] = set()
    for s in plan.specs:
        for kind in _PARTIAL_AGGS[s.func]:
            needed.add((s.source, kind))

    n_chunks = 0
    for chunk in chunks:
        chunk = _apply_as_of(chunk, plan)
        if len(chunk) == 0:
            continue
        n_chunks += 1
        missing = [k for k in plan.group_keys if k not in chunk.columns]
        if missing:
            raise KeyError(f"group_keys {missing} not in chunk")

        for source, kind in sorted(needed):
            if source not in chunk.columns:
                raise KeyError(f"source column {source!r} not in chunk")
            part = chunk.groupby(plan.group_keys, dropna=False)[source].agg(kind)
            key = (source, kind)
            if key not in partials:
                partials[key] = part
            else:
                partials[key] = _combine_partial(partials[key], part, kind)

    if n_chunks == 0:
        # Empty input is a real answer (e.g. every row filtered by as_of),
        # not an error -- but it must be an empty frame with the right shape
        # rather than None, so downstream code does not special-case it.
        return pd.DataFrame(
            columns=list(plan.group_keys) + [s.name for s in plan.specs]
        )

    out = {}
    for s in plan.specs:
        if s.func == "mean":
            total = partials[(s.source, "sum")]
            n = partials[(s.source, "count")]
            out[s.name] = total / n.replace(0, pd.NA)
        else:
            kind = _PARTIAL_AGGS[s.func][0]
            series = partials[(s.source, kind)]
            if kind in _INTEGRAL_PARTIALS and series.notna().all():
                series = series.astype("int64")
            out[s.name] = series

    result = pd.DataFrame(out)
    result.index.names = plan.group_keys
    return result.reset_index()


def plan_for_grain(
    group_keys: list[str],
    specs: list[AggSpec],
    *,
    timestamp_column: str = "",
    as_of=None,
) -> AggregationPlan:
    """Build and eagerly validate a plan, so errors surface at definition
    time rather than mid-way through a crore-scale run."""
    plan = AggregationPlan(
        group_keys=list(group_keys), specs=list(specs),
        timestamp_column=timestamp_column, as_of=as_of,
    )
    problems = plan.validate()
    if problems:
        raise ValueError("invalid AggregationPlan: " + "; ".join(problems))
    return plan
