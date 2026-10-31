"""
sentinel.llm.tokenizer
========================

Exact token counting for the model the shell is talking to, from the
tokenizer files kept in the team's object store (or a mounted folder).

Why
---
Compaction decides when the history is too big by ESTIMATING tokens from
characters. Measured on UPI-style rows (ids, VPAs, amounts), real density
is 1.48 chars/token for Gemma 4, 1.54 for Qwen 3.6, 2.1-2.35 for GLM, Kimi
and DeepSeek -- the old 3.6 assumption undercounted by more than 2x.

Where the tokenizer comes from, in order
----------------------------------------
1. ``tokenizers.local_dir`` in main_config.yaml -- a mounted folder (e.g. a
   PVC) laid out like ``Desktop/LLM-EDA/tokenizers`` (manifest.json + one
   folder per model).
2. The team object store (main_config.yaml ``minio:``; secret via the
   shell's ``/minio --key``), under ``tokenizers.prefix``,
   uploaded by ``tokenizers/upload_to_minio.py``: ``<prefix>/manifest.json``
   plus one de-duplicated ``<prefix>/blobs/<hash>/`` folder per tokenizer.
   Files are downloaded once into ``~/.sentinel/tokenizers`` and checked
   against the manifest's sha256 before use.
3. The local cache alone, when the store is unreachable but a previous
   session already downloaded what this model needs.

Failure is never fatal. Every way this can fail returns a
``TokenizerStatus`` with its own cause, the exact fix, and what the session
uses instead -- the shell prints it once at startup. Counting then falls
back to calibration from the gateway's reported ``usage.prompt_tokens``
(``sentinel.shell.compressor``), which is exact for everything the gateway
has already seen.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Stable cause codes -- tests and the event log key off these, the prose can
# change freely.
OK = "ok"
NOT_CONFIGURED = "not_configured"
STORE_UNREACHABLE = "store_unreachable"
AUTH_FAILED = "auth_failed"
MANIFEST_MISSING = "manifest_missing"
MODEL_NOT_FOUND = "model_not_found"
DOWNLOAD_FAILED = "download_failed"
CHECKSUM_MISMATCH = "checksum_mismatch"
LIBRARY_MISSING = "library_missing"
LOAD_FAILED = "load_failed"
MISMATCH = "count_mismatch"   # raised later, by the compressor's cross-check

_QUANT_SUFFIX = re.compile(
    r"[-_.](?:fp8|fp16|bf16|int4|int8|w4a16|w8a8|awq|gptq(?:-int4)?|gguf|"
    r"q4_0|q4_k_m|q8_0|dynamic)$", re.IGNORECASE)


@dataclass
class TokenizerStatus:
    ok: bool
    code: str
    model: str
    message: str = ""
    fix: str = ""
    source: str = ""                 # local_dir | object_store | cache
    matched: str = ""                # manifest model the name resolved to
    details: dict = field(default_factory=dict)

    def render(self, fallback: str = "") -> str:
        """One block for the shell: what happened, how to fix it, and what
        the session uses in the meantime."""
        if self.ok:
            return (f"tokenizer: exact token counts for {self.model} "
                    f"(using the {self.matched} tokenizer, from {self.source})")
        lines = [f"tokenizer NOT available for model '{self.model}' "
                 f"[{self.code}]: {self.message}"]
        if self.fix:
            lines.append(f"  fix: {self.fix}")
        lines.append("  meanwhile: " + (fallback or
                     "token counts are estimated until the gateway's first "
                     "reply, then calibrated from the usage it reports -- "
                     "the session is not affected beyond that."))
        return "\n".join(lines)


class TokenCounter:
    """Counts tokens for one model. Per-text results are memoised because
    the compressor re-counts the (mostly unchanged) history before every
    request."""

    def __init__(self, encode: Callable[[str], int], *, name: str, kind: str):
        self._encode = encode
        self.name = name
        self.kind = kind
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "replace"),
                              digest_size=12).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = int(self._encode(text))
        if len(self._cache) > 20_000:
            self._cache.clear()
        self._cache[key] = n
        return n


# -- configuration -----------------------------------------------------------

def _settings() -> dict:
    """tokenizers: overrides, else the shared `minio:` section
    (sentinel.io.minio.shared_settings) -- so the secret saved with the
    shell's `/minio --key` is found here too, not only env vars."""
    from sentinel.config import get, get_bool, get_str
    from sentinel.io.minio import shared_settings
    m = shared_settings()
    tok_verify = get("tokenizers", "verify_ssl")
    return {
        "local_dir": get_str("tokenizers", "local_dir",
                             env="SENTINEL_TOKENIZERS_DIR"),
        "endpoint": get_str("tokenizers", "endpoint",
                            env="SENTINEL_TOKENIZERS_ENDPOINT") or m["endpoint"],
        "bucket": get_str("tokenizers", "bucket") or m["bucket"],
        "prefix": get_str("tokenizers", "prefix", default="tokenizers").strip("/"),
        "access_key": get_str("tokenizers", "access_key",
                              env="SENTINEL_TOKENIZERS_ACCESS_KEY") or m["access_key"],
        "secret_key": (os.environ.get("SENTINEL_TOKENIZERS_SECRET_KEY")
                       or str(get("tokenizers", "secret_key") or "")
                       or m["secret_key"]),
        "verify_ssl": (get_bool("tokenizers", "verify_ssl")
                       if tok_verify is not None else m["verify_ssl"]),
        "cache_dir": get_str("tokenizers", "cache_dir") or _default_tokenizer_cache(),
    }


def _default_tokenizer_cache() -> str:
    """Agentic_eda_results/<user>/data_cache/tokenizers (sentinel.paths)."""
    try:
        from sentinel.paths import data_cache_dir
        return str(data_cache_dir() / "tokenizers")
    except Exception:  # noqa: BLE001
        return str(Path.home() / ".sentinel" / "tokenizers")


def normalise_model_name(name: str) -> str:
    """'Qwen/Qwen3.6-35B-A3B-FP8' and 'qwen-3.6-35B-A3B' -> 'qwen3635ba3b'."""
    s = (name or "").strip().split("/")[-1]
    prev = None
    while prev != s:                       # strip stacked quant suffixes
        prev, s = s, _QUANT_SUFFIX.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_model(model: str, manifest: dict) -> tuple[Optional[str], list]:
    """(manifest key, suggestions). Aliases first, then model ids."""
    want = normalise_model_name(model)
    models = manifest.get("models") or {}
    for alias, target in (manifest.get("aliases") or {}).items():
        if normalise_model_name(alias) == want and target in models:
            return target, []
    for key, entry in models.items():
        if normalise_model_name(entry.get("model_id") or key) == want:
            return key, []
    # Aliases first so a real model id with the same normalised form wins.
    names = list((manifest.get("aliases") or {}).keys())
    names += [e.get("model_id") or k for k, e in models.items()]
    by_norm = {normalise_model_name(n): n for n in names}
    close = difflib.get_close_matches(want, list(by_norm), n=3, cutoff=0.5)
    return None, [by_norm[c] for c in close]


# -- object store ------------------------------------------------------------

class _StoreError(Exception):
    def __init__(self, code: str, message: str, fix: str):
        super().__init__(message)
        self.code, self.message, self.fix = code, message, fix


def _classify_store_error(exc: Exception, s: dict, what: str) -> _StoreError:
    endpoint = s["endpoint"]
    code = getattr(exc, "code", "") or ""
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ValueError) and "bucket" in str(exc).lower():
        return _StoreError(
            NOT_CONFIGURED,
            f"minio.bucket '{s['bucket']}' is not a valid bucket name "
            f"({exc}).",
            "set minio.bucket in main_config.yaml to the real bucket "
            "(e.g. sentinel-data).")
    if code in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch",
                "InvalidToken", "ExpiredToken"):
        return _StoreError(
            AUTH_FAILED,
            f"the object store at {endpoint} rejected the credentials ({code}) "
            f"while reading {what}.",
            "set the right secret with /minio --key in the shell (it checks "
            "the key before saving), or export MINIO_SECRET_KEY; check "
            "minio.access_key in main_config.yaml.")
    if code in ("NoSuchBucket",):
        return _StoreError(
            STORE_UNREACHABLE,
            f"bucket '{s['bucket']}' does not exist at {endpoint} or is not "
            "visible to this access key.",
            "check minio.bucket in main_config.yaml.")
    if code in ("NoSuchKey",):
        return _StoreError(
            MANIFEST_MISSING if what.endswith("manifest.json") else DOWNLOAD_FAILED,
            f"{what} is not in s3://{s['bucket']}/ -- the tokenizers were "
            "probably never uploaded.",
            "run `python upload_to_minio.py --go` from the tokenizers folder "
            "(see its README).")
    if "ssl" in text.lower() or "certificate" in text.lower():
        return _StoreError(
            STORE_UNREACHABLE,
            f"TLS handshake with {endpoint} failed ({text[:160]}).",
            "the endpoint uses an internal certificate -- set "
            "minio.verify_ssl: false in main_config.yaml.")
    low = text.lower()
    if any(k in low for k in ("getaddrinfo", "name or service not known",
                              "nameresolution", "nodename nor servname",
                              "temporary failure in name resolution")):
        cause = "the host name does not resolve (DNS) from this machine"
        fix = ("this pod is probably outside the org's network, or the endpoint "
               "name is mistyped -- check minio.endpoint in main_config.yaml.")
    elif any(k in low for k in ("connection refused", "actively refused",
                                "failed to establish a new connection")):
        cause = "the connection was refused (nothing is listening on that port)"
        fix = ("check the port in minio.endpoint, and that the store is "
               "up (curl -k https://" + endpoint + ").")
    elif "timed out" in low or "timeout" in low:
        cause = "the connection timed out"
        fix = ("a firewall may be blocking this pod from the store, or it is "
               "overloaded -- retry, then check network policy.")
    else:
        cause = text[:160]
        fix = (f"check the endpoint and that this pod can reach it "
               f"(curl -k https://{endpoint}).")
    return _StoreError(
        STORE_UNREACHABLE,
        f"could not reach the object store at {endpoint}: {cause}.", fix)


def _client(s: dict):
    try:
        from minio import Minio
    except ImportError as e:
        raise _StoreError(LIBRARY_MISSING, "the `minio` package is not installed.",
                          "pip install minio (it is a core dependency -- "
                          "reinstall sentinel).") from e
    import urllib3
    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=5.0, read=30.0),
        retries=urllib3.Retry(total=1, backoff_factor=0.5),
        cert_reqs="CERT_REQUIRED" if s["verify_ssl"] else "CERT_NONE")
    endpoint = re.sub(r"^https?://", "", s["endpoint"]).rstrip("/")
    return Minio(endpoint, access_key=s["access_key"], secret_key=s["secret_key"],
                 secure=not s["endpoint"].startswith("http://"), http_client=http)


def _fetch(client, s: dict, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        client.fget_object(s["bucket"], key, str(tmp))
    except _StoreError:
        raise
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise _classify_store_error(e, s, key) from e
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- loading -----------------------------------------------------------------

_NEEDED = {"hf-json": ["tokenizer.json"],
           "tiktoken": ["tiktoken.model", "tokenization_kimi.py"]}
_LIBRARY = {"hf-json": ("tokenizers", "tokenizers"),
            "tiktoken": ("tiktoken", "tiktoken")}


def _kimi_pattern(py_file: Path) -> str:
    """Read Kimi's split regex from its tokenization file WITHOUT executing
    it: the class assigns ``pat_str = "|".join([r"...", ...])``."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "pat_str" for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                return "|".join(ast.literal_eval(value.args[0]))
            return ast.literal_eval(value)
    raise ValueError(f"no pat_str assignment in {py_file.name}")


def _load(kind: str, folder: Path, name: str) -> TokenCounter:
    if kind == "hf-json":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        return TokenCounter(
            lambda t: len(tok.encode(t, add_special_tokens=False).ids),
            name=name, kind=kind)
    if kind == "tiktoken":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        enc = tiktoken.Encoding(
            name=name, pat_str=_kimi_pattern(folder / "tokenization_kimi.py"),
            mergeable_ranks=load_tiktoken_bpe(str(folder / "tiktoken.model")),
            special_tokens={})
        return TokenCounter(lambda t: len(enc.encode(t, disallowed_special=())),
                            name=name, kind=kind)
    raise ValueError(f"unsupported tokenizer kind {kind!r}")


# -- entry point -------------------------------------------------------------

def resolve_tokenizer(model: str, *, settings: Optional[dict] = None
                      ) -> tuple[Optional[TokenCounter], TokenizerStatus]:
    """Find, fetch, verify and load the tokenizer for `model`. Never raises."""
    try:
        return _resolve(model, settings or _settings())
    except Exception as e:  # noqa: BLE001 -- the contract is "never raises"
        return None, TokenizerStatus(
            False, LOAD_FAILED, model,
            f"unexpected error while preparing the tokenizer "
            f"({type(e).__name__}: {e}).",
            "report this with the message above; the session continues.")


def _resolve(model: str, s: dict) -> tuple[Optional[TokenCounter], TokenizerStatus]:
    def fail(code, message, fix, **kw):
        return None, TokenizerStatus(False, code, model, message, fix, **kw)

    if not (model or "").strip():
        return fail(MODEL_NOT_FOUND, "no model name is set, so there is nothing "
                    "to look up.", "select a model (/model) or set llm.<method>.model.")

    cache = Path(s["cache_dir"])
    local = Path(s["local_dir"]) if s["local_dir"] else None
    manifest, source, store_note, client = None, "", "", None

    # 1) mounted folder
    if local is not None:
        mpath = local / "manifest.json"
        if not mpath.exists():
            return fail(MANIFEST_MISSING,
                        f"tokenizers.local_dir is set to {local} but it has no "
                        "manifest.json.",
                        "point local_dir at the folder that contains manifest.json "
                        "(the tokenizers folder itself, not its parent).")
        manifest, source = json.loads(mpath.read_text(encoding="utf-8")), "local_dir"

    # 2) object store (manifest refreshed every start: it is small)
    elif s["endpoint"]:
        if not s["access_key"] or not s["secret_key"]:
            missing = "secret key" if s["access_key"] else "access key"
            cached = cache / "manifest.json"
            if not cached.exists():
                return fail(AUTH_FAILED,
                            f"the object store is configured ({s['endpoint']}) "
                            f"but its {missing} is not set, and nothing is cached "
                            "from an earlier session.",
                            "run /minio --key in the shell (or export "
                            "MINIO_SECRET_KEY), and set minio.access_key in "
                            "main_config.yaml or with /minio --access-key.")
            store_note = f"{missing} not set -- using the local cache"
        else:
            try:
                client = _client(s)
                _fetch(client, s, f"{s['prefix']}/manifest.json", cache / "manifest.json")
                source = "object_store"
            except _StoreError as e:
                if not (cache / "manifest.json").exists():
                    return fail(e.code, e.message, e.fix)
                store_note = f"{e.message} -- using the local cache"
                client = None
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        source = source or "cache"

    # 3) cache only
    elif (cache / "manifest.json").exists():
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        source = "cache"
    else:
        return fail(NOT_CONFIGURED,
                    "no tokenizer source is configured.",
                    "set minio.endpoint (and access_key) in main_config.yaml "
                    "and the secret with /minio --key in the shell, or "
                    "tokenizers.local_dir for a mounted folder.")

    key, suggestions = match_model(model, manifest)
    if key is None:
        hint = f" Closest names in the manifest: {', '.join(suggestions)}." \
            if suggestions else ""
        return fail(MODEL_NOT_FOUND,
                    f"'{model}' is not in the tokenizer manifest ({source})."
                    + hint,
                    "if it is one of those, add it under \"aliases\" in "
                    "manifest.json; otherwise download its tokenizer into the "
                    "tokenizers folder and re-upload (tokenizers/README.md).",
                    source=source)
    entry = manifest["models"][key]
    kind = entry.get("kind", "")
    needed = _NEEDED.get(kind)
    if needed is None:
        return fail(LOAD_FAILED, f"tokenizer kind '{kind}' for {key} is not "
                    "supported by this version of Sentinel.",
                    "only tokenizer.json and Kimi tiktoken tokenizers are "
                    "supported; upgrade Sentinel.", source=source, matched=key)

    module, pip_name = _LIBRARY[kind]
    try:
        __import__(module)
    except ImportError:
        return fail(LIBRARY_MISSING,
                    f"the '{module}' Python package is not installed, so the "
                    f"{key} tokenizer cannot be read.",
                    f"pip install {pip_name}  (through the organization wheel mirror: "
                    "export PIP_CONFIG_FILE=configs/pip_index.conf first), "
                    "then restart the shell.", source=source, matched=key)

    # Where the files live
    if local is not None:
        folder = local / entry["dir"]
        missing = [f for f in needed if not (folder / f).exists()]
        if missing:
            return fail(DOWNLOAD_FAILED,
                        f"{', '.join(missing)} missing from {folder}.",
                        "re-copy that model's folder into local_dir.",
                        source=source, matched=key)
    else:
        blob = entry.get("blob") or ""
        # The LOCAL cache folder must key off the same identity the bucket
        # actually groups files by -- the blob path's own name -- not off
        # tokenizer_sha256 alone. Two models can share a vocabulary (same
        # tokenizer_sha256) but ship different wrapper code (e.g. Kimi-K3
        # vs Kimi-K2.7-Code's tokenization_kimi.py): keying the cache dir
        # on tokenizer_sha256 made both models write into the SAME local
        # folder, so whichever loaded second silently reused the first
        # one's wrapper file. Fixed 2026-09-21 alongside the matching
        # upload-side bug (tokenizers/upload_to_minio.py's blob_key).
        digest = Path(blob).name if blob else \
            (entry.get("tokenizer_sha256") or "")[:12] or normalise_model_name(key)
        folder = cache / "blobs" / digest
        hashes = entry.get("files") or {}
        todo = [f for f in needed
                if not (folder / f).exists()
                or (hashes.get(f) and _sha256(folder / f) != hashes[f])]
        if todo:
            if client is None:
                why = store_note or "the object store is not reachable"
                return fail(DOWNLOAD_FAILED,
                            f"{', '.join(todo)} for {key} is not cached yet and "
                            f"cannot be downloaded ({why}).",
                            "fix the object-store access (see the note), or set "
                            "tokenizers.local_dir to a mounted copy.",
                            source=source, matched=key)
            if not blob:
                return fail(MANIFEST_MISSING,
                            f"the manifest entry for {key} has no 'blob' path, "
                            "so the file cannot be located in the bucket.",
                            "re-upload with tokenizers/upload_to_minio.py, which "
                            "writes blob paths into the manifest.",
                            source=source, matched=key)
            for f in todo:
                try:
                    _fetch(client, s, f"{blob}/{f}", folder / f)
                except _StoreError as e:
                    return fail(e.code, e.message, e.fix, source=source, matched=key)
                if hashes.get(f) and _sha256(folder / f) != hashes[f]:
                    (folder / f).unlink(missing_ok=True)
                    return fail(CHECKSUM_MISMATCH,
                                f"{f} for {key} downloaded but its sha256 does "
                                "not match the manifest -- the file was "
                                "discarded.",
                                "re-upload the tokenizers (the bucket copy is "
                                "stale or corrupt), then restart.",
                                source=source, matched=key)
            source = "object_store"

    try:
        counter = _load(kind, folder, entry.get("model_id") or key)
        counter.count("warm-up 123,456.78 ACC0001@ybl")
    except Exception as e:  # noqa: BLE001
        if local is None:
            shutil.rmtree(folder, ignore_errors=True)   # re-download next start
        return fail(LOAD_FAILED,
                    f"the {key} tokenizer files are present but could not be "
                    f"loaded ({type(e).__name__}: {str(e)[:200]}).",
                    ("the cached copy was deleted and will be re-downloaded on "
                     "the next start; if this repeats, re-upload the tokenizer.")
                    if local is None else
                    f"check the files in {folder}; they may be truncated.",
                    source=source, matched=key)

    status = TokenizerStatus(True, OK, model, source=source,
                             matched=entry.get("model_id") or key,
                             details={"context_window": entry.get("context_window")})
    if store_note:
        status.source = f"{source} ({store_note})"
    return counter, status
