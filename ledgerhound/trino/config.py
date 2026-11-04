"""
ledgerhound.trino.config
=====================

TrinoConfig -- connection parameters for the KubeFlow Trino cluster.

Credentials are NEVER hardcoded. Load order:
  1. Explicit constructor arguments
  2. YAML config file (path passed explicitly -- never auto-discovered)
  3. Environment variables (LEDGERHOUND_TRINO_*)

The `use_trino` flag is the master toggle. When False, all TrinoClient
calls are intercepted and return empty results. Use for local-only dev,
CI, or when KubeFlow is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Environment variable names
_ENV_HOST     = "LEDGERHOUND_TRINO_HOST"
_ENV_PORT     = "LEDGERHOUND_TRINO_PORT"
_ENV_USER     = "LEDGERHOUND_TRINO_USER"
_ENV_PASSWORD = "LEDGERHOUND_TRINO_PASSWORD"
_ENV_CATALOG  = "LEDGERHOUND_TRINO_CATALOG"
_ENV_SCHEMA   = "LEDGERHOUND_TRINO_SCHEMA"
_ENV_SCHEME   = "LEDGERHOUND_TRINO_HTTP_SCHEME"
_ENV_USE      = "LEDGERHOUND_TRINO_USE"


@dataclass
class TrinoConfig:
    host: str = ""
    port: int = 8080
    user: str = ""
    password: str = ""
    http_scheme: str = "https"
    catalog: str = ""
    schema: str = ""
    use_trino: bool = True
    # Optional: request timeout for aggregation (stats-only) queries
    query_timeout_seconds: int = 3600
    # Rows-per-second calibration for runtime estimation
    # Tune this against your cluster's observed throughput
    rows_per_second_estimate: int = 5_000_000

    # -- Factories ---------------------------------------------------------

    @classmethod
    def from_env(cls) -> "TrinoConfig":
        """Load Trino settings: main_config.yaml's `trino:` block wins,
        falling back to LEDGERHOUND_TRINO_* environment variables per field
        (the name `from_env` is kept for callers, but env is now the
        fallback, not the primary source)."""
        from ledgerhound.config import get_bool, get_int, get_str
        return cls(
            host=get_str("trino", "host", env=_ENV_HOST),
            port=get_int("trino", "port", env=_ENV_PORT, default=8080),
            user=get_str("trino", "user", env=_ENV_USER),
            password=get_str("trino", "password", env=_ENV_PASSWORD),
            http_scheme=get_str("trino", "http_scheme", env=_ENV_SCHEME,
                               default="https"),
            catalog=get_str("trino", "catalog", env=_ENV_CATALOG),
            schema=get_str("trino", "schema", env=_ENV_SCHEMA),
            use_trino=get_bool("trino", "use", env=_ENV_USE, default=True),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrinoConfig":
        """Load from a YAML config file. The file must NOT be committed to git."""
        import yaml  # optional dep -- only needed when using yaml config

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg = raw.get("trino", raw)  # support both top-level and nested 'trino' key
        return cls(
            host=cfg.get("host", ""),
            port=int(cfg.get("port", 8080)),
            user=cfg.get("user", ""),
            password=cfg.get("password", ""),
            http_scheme=cfg.get("http_scheme", "https"),
            catalog=cfg.get("catalog", ""),
            schema=cfg.get("schema", ""),
            use_trino=cfg.get("use_trino", True),
            query_timeout_seconds=int(cfg.get("query_timeout_seconds", 3600)),
            rows_per_second_estimate=int(cfg.get("rows_per_second_estimate", 5_000_000)),
        )

    @classmethod
    def local_only(cls) -> "TrinoConfig":
        """Convenience: local-only mode, no Trino connection needed."""
        return cls(use_trino=False)

    # -- Connection kwargs for trino-python-client --------------------------

    def to_connection_kwargs(self) -> dict:
        """Return kwargs suitable for trino.dbapi.connect()."""
        kwargs: dict = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "http_scheme": self.http_scheme,
        }
        if self.catalog:
            kwargs["catalog"] = self.catalog
        if self.schema:
            kwargs["schema"] = self.schema
        if self.password:
            from trino.auth import BasicAuthentication
            kwargs["auth"] = BasicAuthentication(self.user, self.password)
        return kwargs

    # -- Validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of config problems. Empty list = config is valid."""
        if not self.use_trino:
            return []
        problems = []
        if not self.host:
            problems.append(f"host is empty (set {_ENV_HOST})")
        if not self.user:
            problems.append(f"user is empty (set {_ENV_USER})")
        if not self.catalog:
            problems.append(f"catalog is empty (set {_ENV_CATALOG})")
        return problems

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def __repr__(self) -> str:
        # Never include password in repr
        return (
            f"TrinoConfig(host={self.host!r}, port={self.port}, "
            f"user={self.user!r}, catalog={self.catalog!r}, "
            f"schema={self.schema!r}, use_trino={self.use_trino})"
        )
