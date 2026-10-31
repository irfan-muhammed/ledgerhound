"""
sentinel.io.minio
=================

MinIOStore -- the object-storage I/O layer for Sentinel.

Backs four durable streams (see configs/minio.yaml):
  wheels     versioned wheels + PyPI-style simple/ index
  agents     user agent library
  data       eval / run / feedback JSONL (date-partitioned, append-only)
  artifacts  run outputs (charts, reports) for `runs share`

Design notes
------------
* **Credentials never live in the repo.** MinIOConfig resolves access/secret
  keys from env vars named in the YAML (mirrors LLMConfig.api_key_env).
* **Append-only JSONL = one object per record** under a date partition, e.g.
  ``eval/dt=2026-06-17/<id>.jsonl``. Object storage is immutable, so we never
  read-modify-write a shared file -- every pod writes its own object and readers
  concatenate. Safe for many notebooks writing concurrently.
* **Lazy SDK import.** The ``minio`` package only imports on first real use, so
  ``import sentinel.io`` works in environments (CI, smoke tests) without it.
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

log = logging.getLogger(__name__)


# -- Config -------------------------------------------------------------------


@dataclass
class MinIOConfig:
    """Connection + bucket settings for MinIO.

    Credentials resolve from the env vars named in ``access_key_env`` /
    ``secret_key_env`` -- never store literal keys here or in YAML.
    """

    endpoint: str = ""                       # host:port, no scheme
    secure: bool = True                      # True => https
    verify_ssl: bool = True                  # False only for self-signed internal
    region: str = "us-east-1"
    access_key_env: str = "MINIO_ACCESS_KEY"
    secret_key_env: str = "MINIO_SECRET_KEY"
    buckets: dict[str, str] = field(default_factory=lambda: {
        "wheels": "sentinel-wheels",
        "agents": "sentinel-agents",
        "data": "sentinel-data",
        "artifacts": "sentinel-artifacts",
    })

    def resolve_access_key(self) -> str:
        val = os.environ.get(self.access_key_env)
        if not val:
            raise ValueError(
                f"access_key_env={self.access_key_env!r} but env var is empty. "
                f"Set it before constructing MinIOStore."
            )
        return val

    def resolve_secret_key(self) -> str:
        val = os.environ.get(self.secret_key_env)
        if not val:
            raise ValueError(
                f"secret_key_env={self.secret_key_env!r} but env var is empty. "
                f"Set it before constructing MinIOStore."
            )
        return val

    def bucket(self, name: str) -> str:
        """Map a logical stream ('data') to its configured bucket name."""
        if name not in self.buckets:
            raise KeyError(f"Unknown bucket stream {name!r}; "
                           f"configured: {sorted(self.buckets)}")
        return self.buckets[name]

    @classmethod
    def from_dict(cls, d: dict) -> "MinIOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MinIOConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("minio", data))


# -- Store ------------------------------------------------------------------


class MinIOStore:
    """Thin wrapper over the MinIO SDK with Sentinel's access patterns.

    >>> store = MinIOStore.from_yaml("configs/minio.yaml")
    >>> store.append_jsonl("data", "eval", {"run_id": "r1", "auc": 0.81})
    >>> url = store.presigned_get_url("artifacts", "runs/r1/report.html")
    """

    def __init__(self, config: MinIOConfig):
        self.config = config
        self._client = None  # lazy

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MinIOStore":
        return cls(MinIOConfig.from_yaml(path))

    # -- client --------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if not self.config.endpoint:
                raise ValueError(
                    "MinIO endpoint is empty -- fill `endpoint` in configs/minio.yaml."
                )
            try:
                from minio import Minio  # heavy / optional dep
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The 'minio' package is required for object storage. "
                    "Install with: pip install minio"
                ) from e

            http_client = None
            if self.config.secure and not self.config.verify_ssl:
                # Self-signed internal cert (mirrors LLM gateway `verify_ssl: false`).
                import urllib3
                urllib3.disable_warnings()
                http_client = urllib3.PoolManager(cert_reqs="CERT_NONE")

            self._client = Minio(
                self.config.endpoint,
                access_key=self.config.resolve_access_key(),
                secret_key=self.config.resolve_secret_key(),
                secure=self.config.secure,
                region=self.config.region,
                http_client=http_client,
            )
        return self._client

    # -- buckets -----------------------------------------------------------

    def ensure_bucket(self, stream: str) -> str:
        """Create the bucket for a logical stream if it doesn't exist."""
        bucket = self.config.bucket(stream)
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket, location=self.config.region)
            log.info("Created MinIO bucket %s", bucket)
        return bucket

    # -- put / get -----------------------------------------------------------

    def put_bytes(self, stream: str, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> str:
        bucket = self.config.bucket(stream)
        self.client.put_object(
            bucket, key, io.BytesIO(data), length=len(data),
            content_type=content_type,
        )
        return f"{bucket}/{key}"

    def put_file(self, stream: str, key: str, path: str | Path,
                 content_type: str = "application/octet-stream") -> str:
        bucket = self.config.bucket(stream)
        self.client.fput_object(bucket, key, str(path), content_type=content_type)
        return f"{bucket}/{key}"

    def get_bytes(self, stream: str, key: str) -> bytes:
        bucket = self.config.bucket(stream)
        resp = self.client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    # -- append-only JSONL (one object per record, date-partitioned) ---------

    def append_jsonl(self, stream: str, dataset: str, record: dict,
                     as_of: datetime | None = None) -> str:
        """Append one JSONL record as its own immutable object.

        Layout: ``<dataset>/dt=YYYY-MM-DD/<uuid>.jsonl`` in the stream's bucket.
        Returns the object key. Concurrency-safe: no shared file is mutated.
        """
        as_of = as_of or datetime.now(timezone.utc)
        day = as_of.strftime("%Y-%m-%d")
        key = f"{dataset}/dt={day}/{uuid.uuid4().hex}.jsonl"
        line = (json.dumps(record, default=str) + "\n").encode("utf-8")
        self.put_bytes(stream, key, line, content_type="application/x-ndjson")
        return key

    def read_jsonl(self, stream: str, prefix: str) -> Iterator[dict]:
        """Concatenate all JSONL objects under a prefix, yielding records."""
        bucket = self.config.bucket(stream)
        for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True):
            blob = self.get_bytes(stream, obj.object_name)
            for line in blob.decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)

    # -- listing + presigned URLs --------------------------------------------

    def list_keys(self, stream: str, prefix: str = "", recursive: bool = True) -> list[str]:
        bucket = self.config.bucket(stream)
        return [o.object_name for o in
                self.client.list_objects(bucket, prefix=prefix, recursive=recursive)]

    def presigned_get_url(self, stream: str, key: str, expires_seconds: int = 3600) -> str:
        """Time-limited download link -- used by `sentinel runs share`."""
        from datetime import timedelta
        bucket = self.config.bucket(stream)
        return self.client.presigned_get_object(
            bucket, key, expires=timedelta(seconds=expires_seconds))

    # -- health ------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Probe connectivity + report which configured buckets exist.

        Never raises on a missing bucket -- returns a status dict for `doctor`.
        """
        out: dict[str, Any] = {"endpoint": self.config.endpoint, "ok": False,
                               "buckets": {}, "error": ""}
        try:
            for stream, bucket in self.config.buckets.items():
                out["buckets"][stream] = {
                    "name": bucket,
                    "exists": bool(self.client.bucket_exists(bucket)),
                }
            out["ok"] = True
        except Exception as e:  # pragma: no cover - network dependent
            out["error"] = str(e)[:200]
        return out


# -- the team object store, from main_config.yaml's `minio:` section ----------
#
# One place that answers "which store, which bucket, which credentials" for
# everything that talks to the team's S3-compatible store (Ceph RGW at a payments organization):
# tokenizers today, the shared knowledge base next. The older MinIOConfig
# above (configs/minio.yaml, one bucket per stream, env-only credentials) is
# unchanged for `sentinel datamem push/pull`.
#
# Credentials, first hit wins -- the secret never has to be in a file under
# git, and the shell's `/minio --key` is the no-terminal-tricks way to set it:
#   1. SENTINEL_MINIO_SECRET_KEY / MINIO_SECRET_KEY (env; pod secrets)
#   2. stored by `/minio --key` in ~/.sentinel/credentials.json (0600)
#   3. minio.secret_key in main_config.yaml (discouraged, but honoured)
# The access key resolves the same way (MINIO_ACCESS_KEY, `/minio
# --access-key`, minio.access_key).

_SECRET_SLOT = ("minio", "secret_key")
_ACCESS_SLOT = ("minio", "access_key")
DEFAULT_BUCKET = "sentinel-data"


def _first(*pairs) -> tuple[str, str]:
    for value, source in pairs:
        if value and str(value).strip():
            return str(value).strip(), source
    return "", ""


def shared_settings() -> dict:
    """Resolved connection settings for the team store, with WHERE each
    credential came from (never the value) for status messages."""
    from sentinel.config import get, get_bool, get_str
    from sentinel.llm.credentials import load_key
    access, access_src = _first(
        (os.environ.get("SENTINEL_MINIO_ACCESS_KEY"), "env SENTINEL_MINIO_ACCESS_KEY"),
        (os.environ.get("MINIO_ACCESS_KEY"), "env MINIO_ACCESS_KEY"),
        (get("minio", "access_key"), "main_config.yaml"),
        (load_key(*_ACCESS_SLOT), "saved with /minio --access-key"))
    secret, secret_src = _first(
        (os.environ.get("SENTINEL_MINIO_SECRET_KEY"), "env SENTINEL_MINIO_SECRET_KEY"),
        (os.environ.get("MINIO_SECRET_KEY"), "env MINIO_SECRET_KEY"),
        (get("minio", "secret_key"), "main_config.yaml"),
        (load_key(*_SECRET_SLOT), "saved with /minio --key"))
    return {
        "endpoint": get_str("minio", "endpoint", env="SENTINEL_MINIO_ENDPOINT").strip(),
        "bucket": get_str("minio", "bucket", default=DEFAULT_BUCKET).strip() or DEFAULT_BUCKET,
        "access_key": access, "access_source": access_src,
        "secret_key": secret, "secret_source": secret_src,
        "verify_ssl": get_bool("minio", "verify_ssl", default=False),
    }


def save_secret(value: str) -> None:
    from sentinel.llm.credentials import store_key
    store_key(*_SECRET_SLOT, value.strip())


def save_access_key(value: str) -> None:
    from sentinel.llm.credentials import store_key
    store_key(*_ACCESS_SLOT, value.strip())


def clear_saved_credentials() -> None:
    from sentinel.llm.credentials import clear_key
    clear_key(*_SECRET_SLOT)
    clear_key(*_ACCESS_SLOT)
