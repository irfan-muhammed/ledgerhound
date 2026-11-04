"""
ledgerhound.trino.sampling_sql
===========================

Turn a :class:`~ledgerhound.sampling.design.NonFraudSamplingDesign` (step 2 of
the EDA cycle) into the **step-3 fetch artifact**, with two interchangeable
sinks:

    * EXPORT  -> a Markdown file: the sampling strategy narrative + the
                 runnable Trino SQL in a fenced block.  A human takes it,
                 runs it in their own Trino console, drops the result back.
                 No DB access required from Ledgerhound.

    * EXECUTE -> the same strategy + SQL, routed through the existing
                 HITL gate on ``TrinoClient.submit_async`` (the
                 "execute button"): the user sees the SQL panel + row
                 estimate and approves before anything runs.

Both sinks consume the *same* :class:`SamplingSQLPlan`, so the exported
Markdown is always a faithful record of exactly what the execute path would
run.

Physical-table details (which table to draw from, how non-fraud is flagged)
are deliberately NOT carried on the design object -- the design is about
*which strata to mirror*, not *where the rows live*.  They are supplied here
via :class:`SamplingSQLConfig`.

This module's SQL-building and Markdown rendering are pure string ops with
no Trino dependency, so they import and test without a live cluster; only
:func:`execute` touches the client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ledgerhound.sampling.design import NonFraudSamplingDesign, SamplingStrategy

if TYPE_CHECKING:  # avoid importing the client (and its deps) at module load
    from ledgerhound.trino.client import TrinoClient
    from ledgerhound.trino.jobs import TrinoJob

logger = logging.getLogger(__name__)

# Placeholder emitted when a stratum has no SQL predicate yet, so the SQL is
# still readable but obviously not runnable until a human fills it in.
_TODO_PREDICATE = "/* TODO: supply predicate for this stratum */ TRUE"


@dataclass
class SamplingSQLConfig:
    """Physical binding for the non-fraud pull.

    Parameters
    ----------
    source_table
        Fully-qualified table to draw rows from (e.g.
        ``hive.analytics.transactions``).
    nonfraud_filter
        Boolean SQL that selects NON-fraud rows (the complement of the fraud
        label).  Default assumes an ``is_fraud`` flag.
    random_fn
        SQL expression used to randomise selection within a stratum.  Trino's
        ``rand()`` is the default; override for a seeded/reproducible draw.
    select_columns
        Columns to project.  ``"*"`` by default.
    """

    source_table: str
    nonfraud_filter: str = "is_fraud = 0"
    random_fn: str = "rand()"
    select_columns: str = "*"


@dataclass
class StratumSQL:
    """The SQL branch for one stratum."""

    name: str
    sql: str
    target_count: int
    predicate_known: bool


@dataclass
class SamplingSQLPlan:
    """Everything step 3 needs: narrative + runnable SQL, from one design."""

    design: NonFraudSamplingDesign
    config: SamplingSQLConfig
    branches: list[StratumSQL] = field(default_factory=list)
    combined_sql: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def requires_human(self) -> bool:
        """True if any stratum still needs a predicate before it can run."""
        return any(not b.predicate_known for b in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source_table": self.config.source_table,
            "requires_human": self.requires_human,
            "combined_sql": self.combined_sql,
            "design": self.design.to_dict(),
            "branches": [
                {
                    "name": b.name,
                    "target_count": b.target_count,
                    "predicate_known": b.predicate_known,
                    "sql": b.sql,
                }
                for b in self.branches
            ],
        }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _branch_sql(
    config: SamplingSQLConfig,
    *,
    predicate: Optional[str],
    target_count: int,
    stratum_name: str,
) -> str:
    """One stratum's SELECT: non-fraud rows matching the stratum, random n."""
    cond = predicate.strip() if (predicate and predicate.strip()) else _TODO_PREDICATE
    where = f"{config.nonfraud_filter} AND ({cond})"
    return (
        f"SELECT {config.select_columns}, '{stratum_name}' AS _stratum\n"
        f"FROM {config.source_table}\n"
        f"WHERE {where}\n"
        f"ORDER BY {config.random_fn}\n"
        f"LIMIT {target_count}"
    )


def _stratified_reservoir_sql(
    config: SamplingSQLConfig, *, stratify_on: str, budget: int
) -> str:
    """Representative draw stratified PROPORTIONALLY on a nuisance axis.

    Per bucket of ``stratify_on`` we keep a count proportional to the bucket's
    share of the non-fraud population (row_number within the bucket, capped at
    budget * bucket_share). This guarantees the draw is not skewed on that axis
    while leaving the signal features free to vary -- so step 6 can still see
    their differences. Helper columns are prefixed ``_`` (drop after fetch).
    """
    return (
        f"SELECT {config.select_columns}, '_iid_reservoir' AS _stratum\n"
        f"FROM (\n"
        f"  SELECT *,\n"
        f"         row_number() OVER (PARTITION BY {stratify_on} "
        f"ORDER BY {config.random_fn}) AS _rn,\n"
        f"         CAST({budget} * (count(*) OVER (PARTITION BY {stratify_on}))\n"
        f"              / (count(*) OVER ()) AS BIGINT) AS _cap\n"
        f"  FROM {config.source_table}\n"
        f"  WHERE {config.nonfraud_filter}\n"
        f") s\n"
        f"WHERE _rn <= GREATEST(_cap, 1)"
    )


def build_sql(
    design: NonFraudSamplingDesign, config: SamplingSQLConfig
) -> SamplingSQLPlan:
    """Render a :class:`NonFraudSamplingDesign` into a runnable SQL plan.

    Each stratum becomes a non-fraud SELECT bounded by the stratum predicate
    and capped at the stratum's target count; branches are ``UNION ALL``-ed
    into one query so the execute path is a single HITL approval and one
    result file.  A stratum tags its rows with a ``_stratum`` literal so the
    downstream non-fraud EDA can recover which stratum each row came from.

    A representative reservoir with ``design.stratify_on`` set is rendered as a
    single window-function query that samples proportionally per nuisance
    bucket instead of a flat random LIMIT.
    """
    if design.stratify_on and len(design.strata) == 1:
        spec = design.strata[0]
        sql = _stratified_reservoir_sql(
            config, stratify_on=design.stratify_on, budget=spec.target_count)
        branch = StratumSQL(name=spec.name, sql=sql,
                            target_count=spec.target_count, predicate_known=True)
        return SamplingSQLPlan(design=design, config=config,
                               branches=[branch], combined_sql=sql)

    branches: list[StratumSQL] = []
    for spec in design.strata:
        sql = _branch_sql(
            config,
            predicate=spec.predicate,
            target_count=spec.target_count,
            stratum_name=spec.name,
        )
        branches.append(
            StratumSQL(
                name=spec.name,
                sql=sql,
                target_count=spec.target_count,
                predicate_known=spec.predicate_known,
            )
        )

    combined = "\nUNION ALL\n".join(b.sql for b in branches)
    return SamplingSQLPlan(
        design=design, config=config, branches=branches, combined_sql=combined
    )


# ---------------------------------------------------------------------------
# Sink 1: export to Markdown
# ---------------------------------------------------------------------------


def to_markdown(plan: SamplingSQLPlan) -> str:
    """Render the strategy narrative + runnable SQL as Markdown text."""
    d = plan.design
    lines: list[str] = []
    lines.append(f"# Non-fraud sampling plan -- `{d.target_dataset}`")
    lines.append("")
    lines.append(f"*Generated:* {plan.generated_at}  ")
    lines.append(f"*Derived from fraud sample:* `{d.parent_sample}`  ")
    lines.append(f"*Strategy:* **{d.strategy.value}**  ")
    lines.append(f"*Sample frame:* `{d.sample_frame.value}`  ")
    lines.append(f"*Source table:* `{plan.config.source_table}`  ")
    lines.append(f"*Total target rows:* {d.total_target}")
    lines.append("")

    if d.base_rate_note:
        lines.append(f"> {d.base_rate_note}")
        lines.append("")

    lines.append("## Strategy")
    lines.append("")
    lines.append(_strategy_blurb(d.strategy))
    lines.append("")

    lines.append("## Strata")
    lines.append("")
    lines.append("| Stratum | Target n | Fraud n | Confidence | Predicate | From findings |")
    lines.append("|---|---|---|---|---|---|")
    for spec in d.strata:
        pred = spec.predicate if spec.predicate_known else "**(needs predicate)**"
        srcs = ", ".join(spec.source_finding_ids) or "-"
        fraud_n = "-" if spec.fraud_count is None else str(spec.fraud_count)
        lines.append(
            f"| `{spec.name}` | {spec.target_count} | {fraud_n} | "
            f"{spec.confidence:.2f} | {pred} | {srcs} |"
        )
    lines.append("")

    if d.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in d.warnings:
            lines.append(f"- {w}")
        lines.append("")

    if plan.requires_human:
        lines.append(
            "> **This SQL is not yet runnable.** One or more strata are "
            "missing a predicate (shown as a `TODO` in the SQL below). Fill "
            "those in before running."
        )
        lines.append("")

    lines.append("## SQL")
    lines.append("")
    lines.append("```sql")
    lines.append(plan.combined_sql)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _strategy_blurb(strategy: SamplingStrategy) -> str:
    if strategy == SamplingStrategy.MATCHED_CONTROLS:
        return (
            "Matched controls: one (or `ratio`) non-fraud row per fraud row "
            "*inside the same stratum*, so the fraud-vs-non-fraud comparison "
            "is not confounded by amount / age / purpose / time differences."
        )
    if strategy == SamplingStrategy.STRATIFIED_MIRROR:
        return (
            "Stratified mirror: non-fraud drawn so each stratum's proportion "
            "mirrors the fraud sample (documented weights), a middle ground "
            "between matched controls and a representative reservoir."
        )
    return (
        "Representative reservoir: an i.i.d. non-fraud draw reflecting true "
        "base rates, enabling population-level prevalence claims. Record "
        "fraud prevalence separately so those claims stay calibratable."
    )


def export_markdown(plan: SamplingSQLPlan, path: str | Path) -> Path:
    """Write the Markdown plan to ``path``; returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_markdown(plan), encoding="utf-8")
    logger.info("Wrote non-fraud sampling plan -> %s", p)
    return p


# ---------------------------------------------------------------------------
# Sink 2: execute via the HITL-gated client
# ---------------------------------------------------------------------------


def execute(
    plan: SamplingSQLPlan,
    client: "TrinoClient",
    *,
    result_filename: str = "nonfraud_sample.parquet",
) -> "TrinoJob":
    """Submit the plan through the HITL gate (the "execute button").

    Refuses to submit a plan that still has unfilled predicates -- those must
    be resolved (via the exported Markdown, by a human) first, so we never
    run a query with a ``TODO TRUE`` that would silently pull the wrong rows.
    """
    if plan.requires_human:
        missing = [b.name for b in plan.branches if not b.predicate_known]
        raise ValueError(
            "Cannot execute: strata are missing predicates "
            f"{missing}. Export to Markdown, fill the predicates, then rerun."
        )
    context = (
        f"Non-fraud sample for '{plan.design.target_dataset}' "
        f"(strategy={plan.design.strategy.value}, "
        f"frame={plan.design.sample_frame.value})"
    )
    return client.submit_async(
        plan.combined_sql, context=context, result_filename=result_filename
    )
