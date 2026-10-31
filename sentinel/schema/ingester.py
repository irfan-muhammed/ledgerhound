"""
sentinel.schema.ingester
========================

Parses schema definitions from multiple formats into DatasetSchema.

Supported formats:
* Python dict / JSON (nested tables format)
* JSON flat list (schema2.json: [{name, dtype, description}, ...])
* YAML
* SQL DDL
"""

from __future__ import annotations

import json, re
from pathlib import Path
from typing import Any

import yaml

from sentinel.schema.models import (
    ColumnRole, ColumnSchema, DatasetSchema, PIIStatus,
    Relationship, SemanticType, TableSchema, TargetConfig,
)


class SchemaIngester:
    """Route to the right parser based on format."""

    def ingest(self, source: str | dict | list | Path, fmt: str = "auto") -> DatasetSchema:
        # Raw dict
        if isinstance(source, dict):
            return _DictParser().parse(source)
        # Raw list (flat column list like schema2.json)
        if isinstance(source, list):
            return _FlatListParser().parse(source)

        source = Path(source)
        if fmt == "auto":
            fmt = self._detect(source)

        raw = self._load(source, fmt)

        # Auto-detect flat list vs nested dict
        if isinstance(raw, list):
            return _FlatListParser().parse(raw)

        parsers = {"json": _DictParser, "yaml": _DictParser, "ddl": _DDLParser}
        parser_cls = parsers.get(fmt)
        if not parser_cls:
            raise ValueError(f"Unknown schema format '{fmt}'. Choices: {list(parsers)}")
        return parser_cls().parse(raw)

    def _detect(self, path: Path) -> str:
        mapping = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".sql": "ddl"}
        fmt = mapping.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"Cannot auto-detect format for {path}. Pass fmt= explicitly.")
        return fmt

    def _load(self, path: Path, fmt: str) -> Any:
        text = path.read_text(encoding="utf-8")
        if fmt in ("json",):
            return json.loads(text)
        if fmt == "yaml":
            return yaml.safe_load(text)
        return text


# -- Flat list parser (schema2.json style) ------------------

class _FlatListParser:
    """Parse [{name, dtype, description}, ...] into a single-table DatasetSchema."""

    _DTYPE_MAP = {
        "numeric": "FLOAT", "categorical": "VARCHAR", "datetime": "TIMESTAMP",
        "int": "INTEGER", "float": "FLOAT", "str": "VARCHAR", "object": "VARCHAR",
    }

    def parse(self, columns: list[dict]) -> DatasetSchema:
        cols: dict[str, ColumnSchema] = {}
        target_col = None

        for cdef in columns:
            name = cdef.get("name", "")
            raw_dtype = cdef.get("dtype", "VARCHAR").lower()
            mapped_dtype = self._DTYPE_MAP.get(raw_dtype, raw_dtype.upper())
            desc = cdef.get("description", "")

            cs = ColumnSchema(
                name=name, dtype=mapped_dtype, description=desc,
                sample_values=cdef.get("sample", []), provided=True,
            )
            cs.semantic_type = self._infer_semantic(name, raw_dtype)
            cs.role = self._infer_role(name, desc, cs.semantic_type)
            cs.pii_status = PIIStatus.NONE
            cols[name] = cs

            if cs.role == ColumnRole.TARGET:
                target_col = name

        table = TableSchema(
            name="primary", alias="primary",
            description="Behavioral feature dataset",
            columns=cols,
            row_grain="one row per account",
        )

        target = TargetConfig()
        if target_col:
            target = TargetConfig(mode="explicit", column=target_col)

        return DatasetSchema(
            tables={"primary": table},
            primary_table="primary",
            target=target,
        )

    @staticmethod
    def _infer_semantic(name: str, dtype: str) -> SemanticType:
        if dtype in ("datetime",):
            return SemanticType.TEMPORAL
        if dtype in ("numeric", "int", "float"):
            return SemanticType.NUMERIC
        if dtype in ("categorical",):
            if "date" in name or "time" in name or "asdt" in name:
                return SemanticType.TEMPORAL
            return SemanticType.CATEGORICAL
        return SemanticType.UNKNOWN

    @staticmethod
    def _infer_role(name: str, desc: str, sem: SemanticType) -> ColumnRole:
        n = name.lower()
        d = desc.lower()
        if any(kw in n for kw in ("frd_flag", "fraud", "target", "label", "is_fraud")):
            return ColumnRole.TARGET
        if any(kw in n for kw in ("amt", "amount", "spend", "value")):
            return ColumnRole.AMOUNT
        if sem == SemanticType.TEMPORAL:
            return ColumnRole.TIMESTAMP
        if n in ("bankifsc", "account_id", "user_id"):
            return ColumnRole.ENTITY_ID
        return ColumnRole.FEATURE


# -- Nested dict parser -------------------------------------

class _DictParser:
    """Parse upi_schema_dict-style dicts."""

    def parse(self, data: dict) -> DatasetSchema:
        tables: dict[str, TableSchema] = {}
        first_alias: str | None = None

        for alias, tdef in data.get("tables", {}).items():
            first_alias = first_alias or alias
            columns = {}
            for cname, cdef in tdef.get("columns", {}).items():
                cs = ColumnSchema(
                    name=cname, dtype=cdef.get("type", "VARCHAR"),
                    description=cdef.get("description", ""),
                    sample_values=cdef.get("sample", []), provided=True,
                )
                cs.semantic_type = self._infer_semantic(cs)
                cs.pii_status = self._detect_pii(cs)
                cs.role = self._infer_role(cs)
                columns[cname] = cs
            tables[alias] = TableSchema(
                name=tdef.get("name", alias), alias=alias,
                description=tdef.get("description", ""), columns=columns)

        rels = [Relationship(
            from_table=r.get("from_table",""), from_column=r.get("from_column",""),
            to_table=r.get("to_table",""), to_column=r.get("to_column",""))
            for r in data.get("relationships", [])]

        return DatasetSchema(tables=tables, relationships=rels,
                             primary_table=first_alias or "")

    def _infer_semantic(self, col):
        dtype, name = col.dtype.upper(), col.name.lower()
        samples = col.sample_values
        if self._looks_like_hash(samples): return SemanticType.HASH
        if dtype in ("TIMESTAMP","DATE","DATETIME") or "date" in name or "time" in name:
            return SemanticType.TEMPORAL
        if dtype in ("BIGINT","INT","INTEGER","FLOAT","DOUBLE","DECIMAL","NUMERIC"):
            return SemanticType.NUMERIC
        if samples:
            ur = len(set(str(s) for s in samples)) / max(len(samples), 1)
            if ur >= 0.9 and dtype == "VARCHAR": return SemanticType.IDENTIFIER
        if dtype == "VARCHAR": return SemanticType.CATEGORICAL
        return SemanticType.UNKNOWN

    @staticmethod
    def _looks_like_hash(samples):
        valid = [s for s in samples if isinstance(s, str) and s]
        if len(valid) < 2: return False
        lengths = {len(s) for s in valid}
        if len(lengths) > 1: return False
        if next(iter(lengths)) not in (32,40,64,128): return False
        return all(set(s) <= set("0123456789abcdefABCDEF") for s in valid)

    @staticmethod
    def _detect_pii(col):
        if col.semantic_type == SemanticType.HASH: return PIIStatus.HASHED
        if any(kw in col.name.lower() for kw in ("accno","accname","mobile","phone","email")):
            return PIIStatus.RAW
        return PIIStatus.NONE

    @staticmethod
    def _infer_role(col):
        n, d = col.name.lower(), col.description.lower()
        if n in ("txnid","txn_id","id") or "transaction id" in d: return ColumnRole.PRIMARY_KEY
        if "amount" in n or "amount" in d: return ColumnRole.AMOUNT
        if col.semantic_type == SemanticType.TEMPORAL: return ColumnRole.TIMESTAMP
        if n.endswith("respcode"): return ColumnRole.RESPONSE_CODE
        if "status" in n: return ColumnRole.STATUS
        if any(kw in n for kw in ("fraud","target","label","is_fraud","frd_flag")): return ColumnRole.TARGET
        if any(kw in n for kw in ("vpa","vaddr","mobile")): return ColumnRole.ENTITY_ID
        return ColumnRole.FEATURE


class _DDLParser:
    _TABLE_RE = re.compile(r"CREATE\s+TABLE\s+([\w.]+)\s*\((.*?)\)", re.S | re.I)
    _COL_RE = re.compile(r"(\w+)\s+([\w()]+(?:\(\d+\))?)\s*(?:--(.*))?[,)]")

    def parse(self, ddl_text: str) -> DatasetSchema:
        tables = {}
        for tname, cols_text in self._TABLE_RE.findall(ddl_text):
            columns = {}
            for m in self._COL_RE.finditer(cols_text):
                cname = m.group(1).strip()
                columns[cname] = ColumnSchema(
                    name=cname, dtype=m.group(2).strip().upper(),
                    description=(m.group(3) or "").strip().rstrip(","), provided=True)
            alias = tname.split(".")[-1] if "." in tname else tname
            tables[alias] = TableSchema(name=tname, alias=alias, columns=columns)
        return DatasetSchema(tables=tables, primary_table=(list(tables)[0] if tables else ""))
