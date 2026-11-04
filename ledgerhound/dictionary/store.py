"""
ledgerhound.dictionary.store
=========================

DuckDB-backed implementation of the Schema Knowledge Base.

Why DuckDB?
-----------
* In-process -- no server, no auth, no network. Just a file (or ``:memory:``).
* Columnar + vectorised -- sub-millisecond on hundreds of thousands of rows
  (we'll have at most ~10k schema entries, so this is overkill in the best way).
* Speaks SQL -- the agent's ``query`` tool can hit it directly if needed.
* Already a Phase-1 dependency for T3-local data access.

We avoid graph databases because schema relationships here are flat
(column <-> alias, column <-> quality flag, column <-> join-target). Anything a
graph DB does, a few SQL joins do faster on data this small.

A small embedding-based semantic search may be added later (Phase 2);
for Phase 1 we use SQL ``ILIKE`` + token matching, which is honest and
debuggable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb

from .models import (
    ColumnInfo,
    DictionaryStatus,
    PopulationStatus,
    QualityFlag,
    QualitySeverity,
    ReferenceCode,
    RelationKind,
    Relationship,
)

log = logging.getLogger(__name__)


# -- DDL ----------------------------------------------------
#
# Kept as a list of statements (not a single string split on `;`) so that
# any future statement containing literal semicolons in identifiers,
# comments, or string defaults won't silently corrupt the schema setup.

_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS columns (
        canonical_name VARCHAR PRIMARY KEY,
        db_alias       VARCHAR,
        dict_name      VARCHAR,
        dataset        VARCHAR,
        semantic_type  VARCHAR,
        role           VARCHAR,
        status         VARCHAR,
        population     VARCHAR,
        description    VARCHAR,
        comments       VARCHAR,
        is_active      BOOLEAN DEFAULT TRUE,
        tags           VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS aliases (
        canonical_name VARCHAR,
        alias          VARCHAR,
        source         VARCHAR,
        PRIMARY KEY (canonical_name, alias)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quality_flags (
        canonical_name VARCHAR,
        issue          VARCHAR,
        severity       VARCHAR,
        source_quote   VARCHAR,
        source_file    VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reference_codes (
        code        VARCHAR,
        code_type   VARCHAR,
        description VARCHAR,
        product     VARCHAR,
        notes       VARCHAR,
        PRIMARY KEY (code, code_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS relationships (
        col_a VARCHAR,
        col_b VARCHAR,
        kind  VARCHAR,
        notes VARCHAR
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_columns_dataset  ON columns(dataset)",
    "CREATE INDEX IF NOT EXISTS idx_columns_semantic ON columns(semantic_type)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_alias    ON aliases(alias)",
    "CREATE INDEX IF NOT EXISTS idx_qflags_canonical ON quality_flags(canonical_name)",
    "CREATE INDEX IF NOT EXISTS idx_refcodes_type    ON reference_codes(code_type)",
    "CREATE INDEX IF NOT EXISTS idx_relations_a      ON relationships(col_a)",
    "CREATE INDEX IF NOT EXISTS idx_relations_b      ON relationships(col_b)",
]


# -- Store --------------------------------------------------


class SchemaKB:
    """DuckDB-backed Schema Knowledge Base.

    Open with ``SchemaKB()`` for in-memory (good for tests, fast),
    or ``SchemaKB("path/to/kb.duckdb")`` to persist across runs.

    All methods are synchronous and idempotent -- calling ``add_column``
    twice with the same canonical name is an upsert, not an error.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path)
        self._ensure_schema()

    # -- lifecycle ------------------------------------------

    def _ensure_schema(self) -> None:
        for stmt in _DDL:
            self.con.execute(stmt)

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- upsert API -----------------------------------------

    def add_column(self, col: ColumnInfo) -> None:
        """Insert or update a column entry (upsert by canonical_name)."""
        self.con.execute(
            """
            INSERT OR REPLACE INTO columns
              (canonical_name, db_alias, dict_name, dataset, semantic_type, role,
               status, population, description, comments, is_active, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                col.canonical_name,
                col.db_alias,
                col.dict_name,
                col.dataset,
                col.semantic_type,
                col.role,
                col.status.value if isinstance(col.status, DictionaryStatus) else col.status,
                col.population.value if isinstance(col.population, PopulationStatus) else col.population,
                col.description,
                col.comments,
                col.is_active,
                ",".join(col.tags) if col.tags else "",
            ],
        )
        # Self-alias: canonical_name -> canonical_name (uniform lookup)
        self._add_alias_raw(col.canonical_name, col.canonical_name, "self")
        if col.db_alias and col.db_alias != col.canonical_name:
            self._add_alias_raw(col.canonical_name, col.db_alias, "db")
        if col.dict_name and col.dict_name != col.canonical_name:
            self._add_alias_raw(col.canonical_name, col.dict_name, "dictionary")

    def add_alias(self, canonical_name: str, alias: str, source: str = "manual") -> None:
        """Register an alternative name for a column."""
        self._add_alias_raw(canonical_name, alias, source)

    def _add_alias_raw(self, canonical: str, alias: str, source: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO aliases (canonical_name, alias, source) VALUES (?, ?, ?)",
            [canonical, alias, source],
        )

    def add_quality_flag(self, qf: QualityFlag) -> None:
        """Attach a documented data-quality issue to a column."""
        self.con.execute(
            """
            INSERT INTO quality_flags
              (canonical_name, issue, severity, source_quote, source_file)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                qf.canonical_name,
                qf.issue,
                qf.severity.value if isinstance(qf.severity, QualitySeverity) else qf.severity,
                qf.source_quote,
                qf.source_file,
            ],
        )

    def add_reference_code(self, rc: ReferenceCode) -> None:
        """Insert or update a reference-code lookup entry."""
        self.con.execute(
            """
            INSERT OR REPLACE INTO reference_codes
              (code, code_type, description, product, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            [rc.code, rc.code_type, rc.description, rc.product, rc.notes],
        )

    def add_relationship(self, rel: Relationship) -> None:
        """Add an inter-column relationship (alias / join-key / derived / pair)."""
        self.con.execute(
            "INSERT INTO relationships (col_a, col_b, kind, notes) VALUES (?, ?, ?, ?)",
            [
                rel.col_a,
                rel.col_b,
                rel.kind.value if isinstance(rel.kind, RelationKind) else rel.kind,
                rel.notes,
            ],
        )

    # -- query API ------------------------------------------

    def lookup(self, name: str) -> Optional[ColumnInfo]:
        """Find a column by canonical name OR any alias.

        Returns None if no match. This is the **primary entry point**
        for the agent -- pass any name shape (db, dict, canonical) and
        get the unified record.
        """
        canonical = self._resolve_canonical(name)
        if canonical is None:
            return None
        row = self.con.execute(
            "SELECT * FROM columns WHERE canonical_name = ?", [canonical]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_column(row)

    def _resolve_canonical(self, name: str) -> Optional[str]:
        """Map any alias back to its canonical name."""
        # Try exact canonical first (cheapest path)
        row = self.con.execute(
            "SELECT canonical_name FROM columns WHERE canonical_name = ?", [name]
        ).fetchone()
        if row:
            return row[0]
        # Then alias table
        row = self.con.execute(
            "SELECT canonical_name FROM aliases WHERE alias = ? LIMIT 1", [name]
        ).fetchone()
        return row[0] if row else None

    def aliases(self, name: str) -> list[str]:
        """Return all known aliases for a column."""
        canonical = self._resolve_canonical(name)
        if canonical is None:
            return []
        rows = self.con.execute(
            "SELECT alias FROM aliases WHERE canonical_name = ? AND alias != ?",
            [canonical, canonical],
        ).fetchall()
        return [r[0] for r in rows]

    def quality_flags(self, name: str) -> list[QualityFlag]:
        """Return data-quality issues for a column (empty list if clean)."""
        canonical = self._resolve_canonical(name)
        if canonical is None:
            return []
        rows = self.con.execute(
            "SELECT canonical_name, issue, severity, source_quote, source_file "
            "FROM quality_flags WHERE canonical_name = ?",
            [canonical],
        ).fetchall()
        return [
            QualityFlag(
                canonical_name=r[0],
                issue=r[1],
                severity=QualitySeverity(r[2]) if r[2] else QualitySeverity.MEDIUM,
                source_quote=r[3] or "",
                source_file=r[4] or "",
            )
            for r in rows
        ]

    def decode(self, code: str, code_type: str = "purpose_code") -> Optional[str]:
        """Translate a reference code (e.g. ``41``) to its description."""
        row = self.con.execute(
            "SELECT description FROM reference_codes WHERE code = ? AND code_type = ?",
            [str(code), code_type],
        ).fetchone()
        return row[0] if row else None

    def reference_code(self, code: str, code_type: str = "purpose_code") -> Optional[ReferenceCode]:
        """Full reference-code record (description + product)."""
        row = self.con.execute(
            "SELECT code, code_type, description, product, notes "
            "FROM reference_codes WHERE code = ? AND code_type = ?",
            [str(code), code_type],
        ).fetchone()
        if row is None:
            return None
        return ReferenceCode(
            code=row[0], code_type=row[1], description=row[2],
            product=row[3], notes=row[4] or "",
        )

    def search(
        self,
        contains: Optional[str] = None,
        dataset: Optional[str] = None,
        semantic_type: Optional[str] = None,
        status: Optional[DictionaryStatus | str] = None,
        active_only: bool = True,
    ) -> list[ColumnInfo]:
        """Filter columns by simple predicates.

        All filters are AND-ed. ``contains`` does case-insensitive substring
        match across canonical_name, db_alias, dict_name, description, comments.
        """
        clauses, params = [], []
        if active_only:
            clauses.append("is_active = TRUE")
        if dataset:
            clauses.append("dataset = ?")
            params.append(dataset)
        if semantic_type:
            clauses.append("semantic_type = ?")
            params.append(semantic_type)
        if status:
            s = status.value if isinstance(status, DictionaryStatus) else status
            clauses.append("status = ?")
            params.append(s)
        if contains:
            clauses.append(
                "(LOWER(canonical_name) LIKE ? OR LOWER(COALESCE(db_alias,'')) LIKE ? "
                "OR LOWER(COALESCE(dict_name,'')) LIKE ? OR LOWER(COALESCE(description,'')) LIKE ? "
                "OR LOWER(COALESCE(comments,'')) LIKE ?)"
            )
            needle = f"%{contains.lower()}%"
            params.extend([needle] * 5)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.con.execute(f"SELECT * FROM columns{where}", params).fetchall()
        return [self._row_to_column(r) for r in rows]

    def related(
        self, name: str, kind: Optional[RelationKind | str] = None
    ) -> list[tuple[ColumnInfo, RelationKind, str]]:
        """Return columns related to ``name`` plus the relation kind & notes.

        Looks at both directions (col_a -> col_b and col_b -> col_a).
        """
        canonical = self._resolve_canonical(name)
        if canonical is None:
            return []
        params: list = [canonical, canonical]
        kind_clause = ""
        if kind:
            k = kind.value if isinstance(kind, RelationKind) else kind
            kind_clause = " AND kind = ?"
            params.append(k)

        rows = self.con.execute(
            f"""
            SELECT
              CASE WHEN col_a = ? THEN col_b ELSE col_a END AS other,
              kind,
              notes
            FROM relationships
            WHERE (col_a = ? OR col_b = ?){kind_clause}
            """,
            [canonical] + params,
        ).fetchall()

        out = []
        for other_name, k, notes in rows:
            ci = self.lookup(other_name)
            if ci is not None:
                out.append((ci, RelationKind(k), notes or ""))
        return out

    def all_columns(
        self, dataset: Optional[str] = None, active_only: bool = True
    ) -> list[ColumnInfo]:
        """Dump every column (filtered)."""
        return self.search(dataset=dataset, active_only=active_only)

    def stats(self) -> dict:
        """Quick summary -- useful for ``ledgerhound dictionary status``."""
        def n(table: str) -> int:
            return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        return {
            "columns": n("columns"),
            "active_columns": self.con.execute(
                "SELECT COUNT(*) FROM columns WHERE is_active = TRUE"
            ).fetchone()[0],
            "aliases": n("aliases"),
            "quality_flags": n("quality_flags"),
            "reference_codes": n("reference_codes"),
            "relationships": n("relationships"),
        }

    # -- helpers --------------------------------------------

    @staticmethod
    def _row_to_column(row: tuple) -> ColumnInfo:
        (
            canonical_name, db_alias, dict_name, dataset, semantic_type, role,
            status, population, description, comments, is_active, tags,
        ) = row
        return ColumnInfo(
            canonical_name=canonical_name,
            db_alias=db_alias,
            dict_name=dict_name,
            dataset=dataset,
            semantic_type=semantic_type,
            role=role,
            status=DictionaryStatus(status) if status else DictionaryStatus.UNKNOWN,
            population=PopulationStatus(population) if population else PopulationStatus.UNKNOWN,
            description=description or "",
            comments=comments or "",
            is_active=bool(is_active),
            tags=[t for t in (tags or "").split(",") if t],
        )
