"""
ledgerhound.pipeline.history_fetch
================================

**Fraud-side history fetch** (V4-011) -- the missing front half of the model
story: take raw fraud transactions, pull out the entities that committed the
fraud, and fetch each one's full N-month transaction history from Trino.

Why this is not just another stratum draw
------------------------------------------
``ledgerhound.trino.sampling_sql`` draws random **rows** per stratum
(``WHERE <stratum> ... ORDER BY rand() LIMIT n``). It structurally cannot
express *"take these payees and give me everything they did"* -- which is
the entity-grain fetch the model story is built on, and the reason the fraud
row count expands (50k reports -> ~550k history rows) rather than contracting.

That expansion is also why this is **cluster sampling**: rows within one
entity are strongly correlated, so a certificate computed at row level with
i.i.d. thresholds runs at the wrong effective sample size. ``ModelStory``
already exposes ``is_cluster_sampled``; this module is where that becomes
physically true.

Labels are deliberately NOT applied here
----------------------------------------
Fetching is keyed on entity identity alone. Which of the returned history
rows count as fraud is a *labelling* decision that varies by typology
(``ModelStory.label_rules``) and happens downstream. Keeping the two apart
means a label-rule change never requires re-running an 8-crore fetch.

SQL safety
----------
Entity keys come from data, not from a developer, so every literal is
escaped and type-checked before it reaches a query string. There is no
string-formatting path from a raw dataframe value into SQL that skips
:func:`sql_literal`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Sequence

from ledgerhound.pipeline.model_story import ModelStory

log = logging.getLogger(__name__)

SCHEMA = "history_fetch/v1"

# Trino's IN-list has a practical parser limit; batch beyond this.
DEFAULT_MAX_KEYS_PER_QUERY = 10_000


class UnsafeSQLValue(ValueError):
    """A value could not be safely rendered as a SQL literal."""


def sql_literal(value: Any) -> str:
    """Render ``value`` as a safe Trino literal.

    Strings are single-quoted with embedded quotes doubled (the SQL standard
    escape). NULs and other C0 control characters are refused outright rather
    than escaped -- a control character in a payee identifier means the data
    is wrong, and silently passing it through would be worse than failing.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise UnsafeSQLValue(f"non-finite float cannot be a SQL literal: {value!r}")
        return repr(value)
    if isinstance(value, (datetime, date)):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'" if isinstance(value, datetime) \
            else f"DATE '{value.isoformat()}'"

    text = str(value)
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise UnsafeSQLValue(
            "value contains a control character and will not be rendered as "
            f"SQL: {text!r}"
        )
    return "'" + text.replace("'", "''") + "'"


def _quote_ident(name: str) -> str:
    """Quote a column/table identifier for Trino (double quotes, doubled)."""
    text = str(name).strip()
    if not text:
        raise UnsafeSQLValue("empty SQL identifier")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise UnsafeSQLValue(f"control character in SQL identifier: {text!r}")
    return '"' + text.replace('"', '""') + '"'


@dataclass
class HistoryFetchConfig:
    """Physical binding for the entity-history pull.

    Mirrors ``SamplingSQLConfig``'s split of concerns: the *design* says which
    entities and how far back, this says where the rows physically live.

    Parameters
    ----------
    source_table
        Fully-qualified history table (e.g. ``hive.analytics.transactions``).
    timestamp_column
        Column carrying the transaction time, used for the lookback window.
    select_columns
        Projection. ``"*"`` by default.
    max_keys_per_query
        Entity keys per generated statement; larger sets are split into
        batches so a 50k-entity fetch does not produce one unparseable query.
    """

    source_table: str
    timestamp_column: str
    select_columns: str = "*"
    max_keys_per_query: int = DEFAULT_MAX_KEYS_PER_QUERY

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.source_table.strip():
            problems.append("source_table must not be empty")
        if not self.timestamp_column.strip():
            problems.append("timestamp_column must not be empty")
        if self.max_keys_per_query <= 0:
            problems.append(
                f"max_keys_per_query must be > 0; got {self.max_keys_per_query}"
            )
        return problems


@dataclass
class EntityKeySet:
    """The distinct fraud entities whose history we intend to fetch.

    ``columns`` is ordered and ``keys`` are tuples in that same order, so a
    composite key (payee + IFSC) stays unambiguous.
    """

    columns: list[str]
    keys: list[tuple] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def is_composite(self) -> bool:
        return len(self.columns) > 1

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns),
                "n_keys": len(self.keys),
                "is_composite": self.is_composite}


def extract_entity_keys(df, story: ModelStory) -> EntityKeySet:
    """Pull the distinct entity keys out of a raw fraud transaction frame.

    Rows with a null in any key column are dropped -- a partial composite key
    cannot identify an entity, and quietly fetching on the non-null half
    would silently widen the pull.
    """
    missing = [c for c in story.fetch_keys if c not in df.columns]
    if missing:
        raise KeyError(
            f"fetch_keys {missing} not present in the fraud frame; "
            f"available columns: {list(df.columns)[:20]}"
        )
    if not story.fetch_keys:
        raise ValueError("ModelStory.fetch_keys is empty -- nothing to fetch on")

    sub = df[list(story.fetch_keys)].dropna(how="any")
    n_dropped = len(df) - len(sub)
    if n_dropped:
        log.warning(
            "history_fetch: dropped %d/%d fraud rows with a null in %s "
            "(a partial composite key cannot identify an entity)",
            n_dropped, len(df), story.fetch_keys,
        )

    seen: set[tuple] = set()
    keys: list[tuple] = []
    for row in sub.itertuples(index=False, name=None):
        if row not in seen:
            seen.add(row)
            keys.append(row)

    return EntityKeySet(columns=list(story.fetch_keys), keys=keys)


def _window_predicate(config: HistoryFetchConfig, start, end) -> str:
    ts = _quote_ident(config.timestamp_column)
    return f"{ts} >= {sql_literal(start)} AND {ts} < {sql_literal(end)}"


def _key_predicate(batch: Sequence[tuple], columns: Sequence[str]) -> str:
    """``col IN (...)`` for a single key, row-constructor IN for composites."""
    if len(columns) == 1:
        values = ", ".join(sql_literal(k[0]) for k in batch)
        return f"{_quote_ident(columns[0])} IN ({values})"
    cols = ", ".join(_quote_ident(c) for c in columns)
    tuples = ", ".join(
        "(" + ", ".join(sql_literal(v) for v in key) + ")" for key in batch
    )
    return f"({cols}) IN (VALUES {tuples})"


@dataclass
class HistoryFetchPlan:
    """Runnable SQL for the entity-history pull, plus what it is fetching."""

    story_name: str
    config: HistoryFetchConfig
    key_set: EntityKeySet
    window_start: Any
    window_end: Any
    statements: list[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def n_batches(self) -> int:
        return len(self.statements)

    @property
    def combined_sql(self) -> str:
        """All batches UNION ALL-ed. Use only when it fits one statement."""
        return "\nUNION ALL\n".join(self.statements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "story_name": self.story_name,
            "generated_at": self.generated_at,
            "source_table": self.config.source_table,
            "window_start": str(self.window_start),
            "window_end": str(self.window_end),
            "key_set": self.key_set.to_dict(),
            "n_batches": self.n_batches,
            "statements": list(self.statements),
        }


def build_history_sql(
    key_set: EntityKeySet,
    config: HistoryFetchConfig,
    *,
    window_start,
    window_end,
    story_name: str = "",
) -> HistoryFetchPlan:
    """Render the entity-history fetch as one or more runnable statements.

    The window is half-open ``[window_start, window_end)`` so adjacent
    periods cannot double-count a boundary transaction.
    """
    problems = config.validate()
    if problems:
        raise ValueError("invalid HistoryFetchConfig: " + "; ".join(problems))
    if not key_set.keys:
        raise ValueError(
            "no entity keys to fetch -- an unbounded history pull is never "
            "what was intended"
        )
    if window_start >= window_end:
        raise ValueError(
            f"window_start {window_start!r} must be strictly before "
            f"window_end {window_end!r}"
        )

    window = _window_predicate(config, window_start, window_end)
    statements: list[str] = []
    for i in range(0, len(key_set.keys), config.max_keys_per_query):
        batch = key_set.keys[i: i + config.max_keys_per_query]
        statements.append(
            f"SELECT {config.select_columns}\n"
            f"FROM {config.source_table}\n"
            f"WHERE {_key_predicate(batch, key_set.columns)}\n"
            f"  AND {window}"
        )

    return HistoryFetchPlan(
        story_name=story_name,
        config=config,
        key_set=key_set,
        window_start=window_start,
        window_end=window_end,
        statements=statements,
    )


def plan_from_story(
    df,
    story: ModelStory,
    config: HistoryFetchConfig,
    *,
    window_end,
    window_start=None,
) -> HistoryFetchPlan:
    """Convenience: raw fraud frame + story -> runnable fetch plan.

    ``window_start`` defaults to ``story.lookback_months`` before
    ``window_end``, computed on calendar months rather than a fixed 30-day
    approximation.
    """
    key_set = extract_entity_keys(df, story)
    if window_start is None:
        months = story.lookback_months
        y, m = window_end.year, window_end.month - months
        while m <= 0:
            m += 12
            y -= 1
        try:
            window_start = window_end.replace(year=y, month=m)
        except ValueError:
            # e.g. 31 Mar - 1 month; clamp to the last valid day of the month.
            day = 28
            while True:
                try:
                    window_start = window_end.replace(year=y, month=m, day=day + 1)
                    day += 1
                except ValueError:
                    window_start = window_end.replace(year=y, month=m, day=day)
                    break
    return build_history_sql(
        key_set, config,
        window_start=window_start, window_end=window_end,
        story_name=story.name,
    )
