"""
ledgerhound.io.data_source
=========================

Where the data comes from, and where this run's outputs go.

Two sources:

* **local** -- a path on this machine (today's behaviour, unchanged).
* **s3** -- an object (or a prefix of Parquet parts) in the team object
  store, `s3://<bucket>/<key>`. It is downloaded ONCE, in parallel, into a
  cache folder -- by default the RAM disk (/dev/shm) on a Linux pod, which
  is what a pod with a small PVC and lots of RAM wants -- and from there
  the shell and the kernel read it exactly like a local file. Reading the
  object store cell-by-cell would re-pay network latency on every pass;
  one bulk copy to RAM pays it once.

Outputs follow the source (`RunPaths`): with local data, features stay in
the run folder as before; with s3 data they are written to the cache
(RAM, not the small PVC) and uploaded after every run_python step to
`s3://<bucket>/<outputs_prefix>/<user>/<run>/features/`.

Credentials are the shared ones (`minio:` in main_config.yaml plus the
secret from `/minio --key`, see ledgerhound.io.minio.shared_settings).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

DATA_EXTS = (".parquet", ".pq", ".csv", ".tsv", ".json", ".feather")
_PARQUET = (".parquet", ".pq")
_SIDE = ".ledgerhound-source.json"          # cache marker: what was downloaded
_S3_RE = re.compile(r"^s3://([^/]+)/?(.*)$", re.IGNORECASE)


class DataSourceError(Exception):
    """A load that cannot proceed. `message` says what, `fix` what to do."""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.message = message
        self.fix = fix

    def __str__(self) -> str:
        return f"{self.message}\n  fix: {self.fix}" if self.fix else self.message


# -- parsing ------------------------------------------------------------------

@dataclass
class DataSource:
    kind: str                 # "local" | "s3"
    uri: str                  # as the user should see it
    bucket: str = ""
    key: str = ""

    @property
    def is_s3(self) -> bool:
        return self.kind == "s3"


def is_s3(text: str) -> bool:
    return bool(_S3_RE.match((text or "").strip()))


def parse(text: str, *, kind: str | None = None, default_bucket: str = "") -> DataSource:
    """`s3://bucket/key` -> s3; anything else -> local. `kind="s3"` (the
    /s3 command) also accepts `bucket/key` or just `key` in the default
    bucket; `kind="local"` refuses an s3:// URL so the two commands never
    silently swap meaning."""
    t = (text or "").strip().strip('"').strip("'")
    if not t:
        raise DataSourceError("no path given.", "e.g. /s3 --path s3://sentinel-data/data/july/  or  /local --path /data/july.parquet")
    m = _S3_RE.match(t)
    if kind == "local" and m:
        raise DataSourceError(f"{t} is an object-store path.", f"use /s3 --path {t}")
    if m:
        return DataSource("s3", f"s3://{m.group(1)}/{m.group(2)}", m.group(1), m.group(2))
    if kind == "s3":
        bucket = default_bucket
        key = t.lstrip("/")
        first, _, rest = key.partition("/")
        if rest and first == default_bucket:
            key = rest
        if not bucket:
            raise DataSourceError("no bucket: the path has no s3://<bucket>/ and minio.bucket is not set.",
                                  "write the full s3://<bucket>/<key>, or set minio.bucket in main_config.yaml.")
        return DataSource("s3", f"s3://{bucket}/{key}", bucket, key)
    return DataSource("local", t)


def parts_form_one_dataset(rel_names: list[str]) -> bool:
    """Do these files (paths relative to one folder/prefix) form ONE dataset
    split into Parquet parts -- what Spark, Trino exports and
    `to_parquet(partition_cols=...)` write? Conservative on purpose: a
    folder of july.parquet / aug.parquet is several datasets, and merging
    them silently would analyse the wrong thing. Yes only for hive
    `col=value` folders, a `_SUCCESS` marker, or all-`part-*` names -- and
    only when every data file is Parquet. Anything else needs `--as-one`."""
    data = [n for n in rel_names if Path(n).suffix.lower() in DATA_EXTS]
    if len(data) < 2 or any(Path(n).suffix.lower() not in _PARQUET for n in data):
        return False
    if any("=" in part for n in data for part in Path(n).parts[:-1]):
        return True
    if any(Path(n).name == "_SUCCESS" for n in rel_names):
        return True
    return all(Path(n).name.lower().startswith(("part-", "part_")) for n in data)


# -- cache location -----------------------------------------------------------

def cache_root() -> Path:
    """Where /s3 data is copied: see ledgerhound.paths.data_cache_dir."""
    # paths.data_cache_dir / $LEDGERHOUND_DATA_CACHE_DIR if set, else
    # Agentic_eda_results/<user>/data_cache. For RAM speed on a pod with a
    # small PVC, set paths.data_cache_dir: /dev/shm/sentinel-data.
    from ledgerhound.paths import data_cache_dir
    return data_cache_dir()


def is_ram_backed(path: Path) -> bool:
    return str(path).startswith("/dev/shm")


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# -- object store client --------------------------------------------------------

def _settings(settings: dict | None) -> dict:
    if settings is not None:
        return settings
    from ledgerhound.io.minio import shared_settings
    return shared_settings()


def _check_settings(s: dict) -> None:
    if not s.get("endpoint"):
        raise DataSourceError("the object store is not configured (minio.endpoint is empty).",
                              "set minio.endpoint in main_config.yaml.")
    if not s.get("access_key"):
        raise DataSourceError("the object store access key is not set.",
                              "set minio.access_key in main_config.yaml, or run /minio --access-key <key>.")
    if not s.get("secret_key"):
        raise DataSourceError("the object store secret key is not set.",
                              "run /minio --key in the shell.")


class _StoreErr(Exception):
    """A client error with the S3 error `code` normalised, whichever
    library raised it."""

    def __init__(self, code: str, text: str):
        super().__init__(text)
        self.code = code


@dataclass
class _Obj:
    object_name: str
    size: int
    etag: str


class _Boto3Store:
    """boto3 -- the team's standard S3 client. Path-style addressing
    (what Ceph RGW and MinIO serve) and multipart parallel transfers."""

    name = "boto3"

    def __init__(self, s: dict, workers: int):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        secure = not s["endpoint"].startswith("http://")
        ep = s["endpoint"].rstrip("/")
        if not re.match(r"^https?://", ep):
            ep = ("https://" if secure else "http://") + ep
        if not s.get("verify_ssl"):
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:  # noqa: BLE001
                pass
        self._c = boto3.session.Session().client(
            "s3", endpoint_url=ep, aws_access_key_id=s["access_key"],
            aws_secret_access_key=s["secret_key"], region_name="us-east-1",
            # the org's Ceph RGW is HTTPS with a self-signed cert: `verify_ssl:
            # false` (the default) means `verify=False`, matching `curl -k`.
            # `use_ssl` is passed EXPLICITLY, not left to boto3 to infer from
            # endpoint_url's scheme -- a confirmed-working connection script
            # against this exact endpoint set both `use_ssl=True` and
            # `verify=False` explicitly, and relying on inference was less
            # reliable in practice. False only for a plain http:// endpoint
            # (a local test server), never for the real store.
            use_ssl=secure, verify=bool(s.get("verify_ssl")),
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                          max_pool_connections=max(10, workers * 4),
                          connect_timeout=10, read_timeout=120,
                          retries={"max_attempts": 3, "mode": "standard"}))
        self._tc = TransferConfig(max_concurrency=4, multipart_chunksize=64 * 1024 * 1024)

    @staticmethod
    def _wrap(e: Exception) -> Exception:
        resp = getattr(e, "response", None)
        if isinstance(resp, dict):
            code = str((resp.get("Error") or {}).get("Code") or "")
            code = {"404": "NoSuchKey", "403": "AccessDenied", "NotFound": "NoSuchKey"}.get(code, code)
            return _StoreErr(code, str(e))
        return e

    def stat(self, bucket: str, key: str) -> _Obj:
        try:
            h = self._c.head_object(Bucket=bucket, Key=key)
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e
        return _Obj(key, int(h["ContentLength"]), str(h.get("ETag", "")).strip('"'))

    def list(self, bucket: str, prefix: str) -> list[_Obj]:
        out = []
        try:
            for page in self._c.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=prefix):
                for o in page.get("Contents", []) or []:
                    if not o["Key"].endswith("/"):
                        out.append(_Obj(o["Key"], int(o["Size"]),
                                        str(o.get("ETag", "")).strip('"')))
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e
        return out

    def download(self, bucket: str, key: str, dest: str) -> None:
        try:
            self._c.download_file(bucket, key, dest, Config=self._tc)
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e

    def upload(self, src: str, bucket: str, key: str) -> None:
        try:
            self._c.upload_file(src, bucket, key, Config=self._tc)
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e

    def put_bytes(self, bucket: str, key: str, data: bytes) -> None:
        try:
            self._c.put_object(Bucket=bucket, Key=key, Body=data)
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e

    def get_bytes(self, bucket: str, key: str) -> bytes:
        try:
            return self._c.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e

    def delete(self, bucket: str, key: str) -> None:
        try:
            self._c.delete_object(Bucket=bucket, Key=key)
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e) from e


class _MinioStore:
    """Fallback when boto3 is not installed: the `minio` package (a core
    dependency, also used for tokenizers). Same S3 protocol."""

    name = "minio"

    def __init__(self, s: dict, workers: int):
        from minio import Minio
        import urllib3
        http = urllib3.PoolManager(
            maxsize=max(10, workers * 2),
            timeout=urllib3.Timeout(connect=10.0, read=120.0),
            retries=urllib3.Retry(total=3, backoff_factor=0.5),
            cert_reqs="CERT_REQUIRED" if s.get("verify_ssl") else "CERT_NONE")
        endpoint = re.sub(r"^https?://", "", s["endpoint"]).rstrip("/")
        self._c = Minio(endpoint, access_key=s["access_key"], secret_key=s["secret_key"],
                        secure=not s["endpoint"].startswith("http://"), http_client=http)

    def stat(self, bucket: str, key: str) -> _Obj:
        st = self._c.stat_object(bucket, key)
        return _Obj(key, int(st.size), str(st.etag or "").strip('"'))

    def list(self, bucket: str, prefix: str) -> list[_Obj]:
        return [_Obj(o.object_name, int(o.size), str(o.etag or "").strip('"'))
                for o in self._c.list_objects(bucket, prefix=prefix, recursive=True)
                if not getattr(o, "is_dir", False)]

    def download(self, bucket: str, key: str, dest: str) -> None:
        self._c.fget_object(bucket, key, dest)

    def upload(self, src: str, bucket: str, key: str) -> None:
        self._c.fput_object(bucket, key, src)

    def put_bytes(self, bucket: str, key: str, data: bytes) -> None:
        import io
        self._c.put_object(bucket, key, io.BytesIO(data), length=len(data))

    def get_bytes(self, bucket: str, key: str) -> bytes:
        r = self._c.get_object(bucket, key)
        try:
            return r.read()
        finally:
            r.close()
            r.release_conn()

    def delete(self, bucket: str, key: str) -> None:
        self._c.remove_object(bucket, key)


def make_client(s: dict, *, workers: int = 8):
    """boto3 when installed (`pip install ".[s3]"`), else the minio package.
    `minio.client: boto3|minio` in main_config.yaml forces one."""
    from ledgerhound.config import get_str
    want = get_str("minio", "client", env="LEDGERHOUND_S3_CLIENT").strip().lower()
    order = [want] if want in ("boto3", "minio") else ["boto3", "minio"]
    missing = []
    for lib in order:
        try:
            return (_Boto3Store if lib == "boto3" else _MinioStore)(s, workers)
        except ImportError:
            missing.append(lib)
    raise DataSourceError(f"no S3 client library installed (tried {', '.join(missing)}).",
                          'pip install ".[s3]" (boto3), or pip install minio.')


def _classify(exc: Exception, s: dict, what: str) -> DataSourceError:
    code = getattr(exc, "code", "") or ""
    if code in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch",
                "InvalidToken", "ExpiredToken"):
        return DataSourceError(f"the object store rejected the credentials ({code}) reading {what}.",
                               "check the secret with /minio (re-set it with /minio --key).")
    if code == "NoSuchBucket":
        return DataSourceError(f"bucket '{s.get('bucket')}' does not exist or is not visible to this key.",
                               "check the bucket name in the path / minio.bucket.")
    if code == "NoSuchKey":
        return DataSourceError(f"{what} does not exist.", "check the path; /s3 --path accepts a file or a folder prefix.")
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if any(k in low for k in ("connection refused", "max retries", "timed out",
                              "name or service not known", "getaddrinfo",
                              "failed to establish", "no route",
                              "could not connect to the endpoint",
                              "connect timeout", "connectionclosederror")):
        return DataSourceError(f"could not reach the object store at {s.get('endpoint')} ({text[:160]}).",
                               "check you are on the org's network and minio.endpoint is right; /minio tests the connection.")
    return DataSourceError(f"object store error reading {what}: {text[:300]}", "/minio tests the connection.")


# -- download -------------------------------------------------------------------

@dataclass
class FetchResult:
    source: DataSource
    local_path: Path                  # a file, or a folder of Parquet parts
    files: int
    total_bytes: int
    downloaded_bytes: int
    reused_bytes: int
    seconds: float
    content_id: str                   # sha256 over (key, etag, size) -- no re-hash of GBs
    cache_dir: Path
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        how = (f"downloaded {_human(self.downloaded_bytes)} in {self.seconds:.0f}s"
               if self.downloaded_bytes else "already cached")
        if self.downloaded_bytes and self.reused_bytes:
            how += f" (+{_human(self.reused_bytes)} reused from cache)"
        where = "RAM disk" if is_ram_backed(self.local_path) else "local cache"
        kind = f"{self.files} Parquet parts, read as ONE dataset" if self.files > 1 else "1 file"
        return (f"{self.source.uri} -> {self.local_path} ({kind}, {_human(self.total_bytes)}; "
                f"{how}; {where})")


def _list_objects(client, s: dict, src: DataSource) -> tuple[list, bool]:
    """(objects, is_prefix). A key that names one object is a file; else it
    is treated as a folder prefix."""
    key = src.key
    if key and not key.endswith("/"):
        try:
            return [client.stat(src.bucket, key)], False
        except Exception as e:  # noqa: BLE001
            if (getattr(e, "code", "") or "") not in ("NoSuchKey", "NoSuchObject", ""):
                raise _classify(e, s, src.uri) from e
            if not getattr(e, "code", ""):
                # No S3 code at all: a network/TLS failure, not "missing".
                raise _classify(e, s, src.uri) from e
    prefix = key.rstrip("/") + "/" if key else ""
    try:
        objs = client.list(src.bucket, prefix)
    except Exception as e:  # noqa: BLE001
        raise _classify(e, s, src.uri) from e
    return objs, True


def fetch(src: DataSource, *, settings: dict | None = None, cache: Path | None = None,
          workers: int = 8, as_one: bool = False,
          progress: Callable[[int, int, int, int], None] | None = None) -> FetchResult:
    """Download `src` into the cache (skipping what is already there and
    unchanged) and return where it landed."""
    if not src.is_s3:
        raise DataSourceError(f"{src.uri} is not an object-store path.")
    s = dict(_settings(settings))
    s["bucket"] = src.bucket
    _check_settings(s)
    client = make_client(s, workers=workers)
    objs, is_prefix = _list_objects(client, s, src)
    data = [o for o in objs if Path(o.object_name).suffix.lower() in DATA_EXTS]
    if not data:
        seen = ", ".join(Path(o.object_name).name for o in objs[:8]) or "nothing"
        raise DataSourceError(f"no data files at {src.uri} (found: {seen}).",
                              f"point at a {'/'.join(e.lstrip('.') for e in DATA_EXTS)} file, or a folder of Parquet parts.")
    if is_prefix and len(data) > 1:
        prefix = src.key.rstrip("/") + "/" if src.key else ""
        rel = [o.object_name[len(prefix):] for o in objs]
        all_parquet = all(Path(o.object_name).suffix.lower() in _PARQUET for o in data)
        listing = "\n".join(f"    s3://{src.bucket}/{o.object_name}" for o in data[:20])
        more = f"\n    ... +{len(data) - 20} more" if len(data) > 20 else ""
        if not all_parquet:
            raise DataSourceError(
                f"{src.uri} holds {len(data)} data files and not all are Parquet, so they "
                f"cannot be read as one dataset:\n{listing}{more}",
                "point /s3 --path at one of these files.")
        if not (as_one or parts_form_one_dataset(rel)):
            raise DataSourceError(
                f"{src.uri} holds {len(data)} separate Parquet files -- they do not look "
                f"like parts of one dataset (no col=value folders, no _SUCCESS, not "
                f"part-* files):\n{listing}{more}",
                "point at one file, or if they ARE one dataset split into files, "
                f"run: /s3 --path {src.uri} --as-one")

    root = Path(cache) if cache else cache_root()
    base = root / src.bucket
    if is_prefix:
        prefix = src.key.rstrip("/") + "/" if src.key else ""
        dest_dir = base / (src.key.rstrip("/") or "_root")
        targets = [(o, dest_dir / o.object_name[len(prefix):]) for o in data]
        local_path = dest_dir if len(data) > 1 else targets[0][1]
        marker = dest_dir.with_name(dest_dir.name + _SIDE)   # beside, never inside, the dataset
    else:
        o = data[0]
        dest_dir = base / Path(o.object_name).parent
        targets = [(o, base / o.object_name)]
        local_path = targets[0][1]
        marker = local_path.with_name(local_path.name + _SIDE)

    try:
        cached = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    except Exception:  # noqa: BLE001
        cached = {}
    todo = [(o, p) for o, p in targets
            if not (p.exists() and p.stat().st_size == o.size
                    and cached.get(o.object_name) == (o.etag or "").strip('"'))]
    total = sum(o.size for o, _ in targets)
    need = sum(o.size for o, _ in todo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    if need and need * 1.05 > free:
        raise DataSourceError(
            f"not enough space in {root} for {src.uri}: needs {_human(need)}, "
            f"{_human(free)} free.",
            "free space there (old datasets live under it and can be deleted), or set "
            "paths.data_cache_dir in main_config.yaml to a bigger folder.")

    t0 = time.perf_counter()
    done_b, done_n = 0, 0

    def _one(o, p: Path) -> int:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        try:
            client.download(src.bucket, o.object_name, str(tmp))
            tmp.replace(p)
        except Exception as e:  # noqa: BLE001
            try:
                tmp.unlink()
            except OSError:
                pass
            raise _classify(e, s, f"s3://{src.bucket}/{o.object_name}") from e
        return o.size

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as ex:
            futs = [ex.submit(_one, o, p) for o, p in todo]
            for f in as_completed(futs):
                done_b += f.result()
                done_n += 1
                if progress:
                    progress(done_b, need, done_n, len(todo))

    notes = []
    if is_prefix and len(data) > 1:
        # Files deleted from the store since the last download must not stay
        # in the dataset folder and silently join the data.
        wanted = {p.resolve() for _, p in targets}
        for f in list(dest_dir.rglob("*")):
            if f.is_file() and f.name != _SIDE and f.resolve() not in wanted:
                try:
                    f.unlink()
                    notes.append(f"removed stale cached file {f.name}")
                except OSError:
                    pass
    marker.write_text(json.dumps({o.object_name: (o.etag or "").strip('"') for o, _ in targets},
                                 indent=0), encoding="utf-8")
    import hashlib
    h = hashlib.sha256()
    for o, _ in sorted(targets, key=lambda t: t[0].object_name):
        h.update(f"{o.object_name}\0{(o.etag or '').strip(chr(34))}\0{o.size}\n".encode())
    return FetchResult(src, local_path, len(data), total, need, total - need,
                       time.perf_counter() - t0, h.hexdigest(), root, notes)


# -- where outputs go -----------------------------------------------------------

@dataclass
class RunPaths:
    """The per-run path record (run_dir/paths.json). Rewritten on every
    load, so it always says where THIS run's data came from and where its
    outputs are going."""
    source: str = "local"             # local | s3
    data: str = ""                    # what the user pointed at
    local_copy: str = ""              # s3 only: the cached copy the kernel reads
    run_dir: str = ""
    scratch_dir: str = ""
    features_dir: str = ""
    features_upload: str = ""         # s3 only: s3://.../features/
    updated_at: str = ""

    def save(self, run_dir: Path) -> Path:
        from datetime import datetime, timezone
        self.updated_at = datetime.now(timezone.utc).isoformat()
        p = Path(run_dir) / "paths.json"
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    def describe(self) -> str:
        lines = [f"data source  : {self.source}  {self.data}"]
        if self.local_copy:
            lines.append(f"local copy   : {self.local_copy}")
        lines.append(f"run folder   : {self.run_dir}")
        lines.append(f"features     : {self.features_dir}")
        if self.features_upload:
            lines.append(f"  uploaded to: {self.features_upload}")
        lines.append(f"scratch      : {self.scratch_dir}")
        return "\n".join(lines)


def outputs_prefix(run_name: str, *, settings: dict | None = None) -> tuple[str, str]:
    """(bucket, key-prefix) for this run's uploaded outputs:
    <minio.outputs_prefix>/<user>/<run>/ in the shared bucket."""
    from ledgerhound.config import get_str
    from ledgerhound.identity import current_user
    s = _settings(settings)
    top = get_str("minio", "outputs_prefix", default="outputs").strip("/") or "outputs"
    return s["bucket"], f"{top}/{current_user()}/{run_name}/"


class OutputUploader:
    """Mirrors a local folder to the object store, uploading only files that
    are new or changed since the last sync. Never raises: failures come back
    as text so a network blip never breaks an analysis step."""

    def __init__(self, local_dir: Path, bucket: str, prefix: str, *,
                 settings: dict | None = None, workers: int = 8):
        self.local_dir = Path(local_dir)
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self._settings = settings
        self._workers = workers
        self._state: dict[str, tuple[int, int]] = {}
        self._client = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def _changed(self) -> list[tuple[Path, str]]:
        out = []
        if not self.local_dir.is_dir():
            return out
        for f in self.local_dir.rglob("*"):
            if not f.is_file() or f.name.endswith((".part", ".tmp")):
                continue
            st = f.stat()
            rel = f.relative_to(self.local_dir).as_posix()
            sig = (st.st_size, st.st_mtime_ns)
            if self._state.get(rel) != sig:
                out.append((f, rel))
        return out

    def sync(self) -> tuple[list[str], list[str]]:
        """(uploaded 'rel (size)' lines, error lines)."""
        todo = self._changed()
        if not todo:
            return [], []
        s = dict(_settings(self._settings))
        s["bucket"] = self.bucket
        try:
            _check_settings(s)
            if self._client is None:
                self._client = make_client(s, workers=self._workers)
        except DataSourceError as e:
            return [], [f"upload skipped: {e.message} Fix: {e.fix}"]
        done, errs = [], []

        def _put(f: Path, rel: str):
            st = f.stat()
            self._client.upload(str(f), self.bucket, self.prefix + rel)
            return rel, (st.st_size, st.st_mtime_ns)

        with ThreadPoolExecutor(max_workers=max(1, min(self._workers, len(todo)))) as ex:
            futs = {ex.submit(_put, f, rel): rel for f, rel in todo}
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    rel, sig = fut.result()
                    self._state[rel] = sig
                    done.append(f"{rel} ({_human(sig[0])})")
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{rel}: {_classify(e, s, self.uri + rel).message}")
        return sorted(done), errs
