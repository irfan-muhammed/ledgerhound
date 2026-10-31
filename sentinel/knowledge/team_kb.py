"""
sentinel.knowledge.team_kb
============================

The team knowledge base in the object store: dataset cards (with the
answers to data questions), books, and recorded findings -- shared by
everyone, synced automatically, never manually.

Never losing data is the design constraint, so:

* **Append-only.** Every save writes a NEW object
  (`kb/v1/cards/<dataset>/<utc>-<user>-<rand>.json`); nothing is ever
  overwritten. Readers list all revisions and merge them with the same
  union-merge the local card store uses (`merge_cards`). Two people saving
  at the same moment both land; no conditional-write support is needed
  from the store.
* **Outbox first.** A write goes to `~/.sentinel/kb_outbox/` on disk
  before any network call, and is deleted from there only after the store
  accepted it. A crash, a network outage or an unreachable store loses
  nothing: the outbox is retried on the next save and at every start.
* **Compaction never deletes unseen data.** When a card has many
  revisions, a merged revision is written first, and then only the
  revisions that went INTO that merge are removed.

Layout (bucket = minio.bucket, prefix = minio.kb_prefix, default kb/v1):

    kb/v1/cards/<id>/<rev>.json      card revisions (merged on read)
    kb/v1/books/<id>/<book-id>.json  one object per saved book
    kb/v1/findings/<id>/<rev>.json   one object per recorded finding

`<id>` is a filesystem-safe hash of the dataset's shared name
(sentinel.knowledge.team_kb.dataset_id_for).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

COMPACT_AT = 25            # revisions of one card before a merged one replaces them


# -- dataset identity ------------------------------------------------------------

def dataset_id_for(*, name: str = "", s3_uri: str = "", content_sha256: str = "") -> str:
    """The name a dataset has for EVERY analyst.

    An explicit name wins (`trino:<catalog.schema.table>` for a Trino
    export, or any team name given with --name); then the object-store
    location (`s3://bucket/key`, trailing slash dropped); then the file's
    content hash -- a local file has no shared name, but identical bytes
    are the same data on any machine."""
    if name and name.strip():
        n = name.strip()
        return n if ":" in n else f"name:{n}"
    if s3_uri:
        return s3_uri.rstrip("/")
    if content_sha256:
        return f"sha256:{content_sha256}"
    return ""


def _slot(dataset_id: str) -> str:
    return hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:24]


def _rev_name(user: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (user or "anon"))[:40]
    return f"{stamp}-{safe}-{uuid.uuid4().hex[:8]}.json"


# -- publish filter: no customer identifiers leave the machine -----------------
# A card's profile keeps each column's most frequent values -- for a payer
# VPA, mobile or account column those are REAL customer identifiers (the
# busiest accounts, i.e. exactly the sensitive ones). The local card keeps
# them; everything shared passes through scrub_for_sharing first.
import re as _re

_ID_COLUMN_HINTS = ("vpa", "upi_id", "acct", "account", "accno", "mobile", "phone",
                    "msisdn", "email", "mail", "pan", "aadhaar", "aadhar", "card",
                    "crn", "cust", "customer", "device", "imei", "ip_addr", "ipaddr",
                    "name", "address", "remitter", "beneficiary", "payer", "payee")
_VALUE_KEYS = ("top_values", "mode", "min", "max", "sample", "samples", "examples",
               "values", "value_counts", "first", "last")
_PII_PATTERNS = [
    _re.compile(r"[A-Za-z0-9._\-]{2,}@[A-Za-z][A-Za-z0-9.\-]+"),          # VPA / e-mail
    _re.compile(r"(?<![\d])(?:\+?91[\s-]?)?[6-9]\d{9}(?![\d])"),           # Indian mobile
    _re.compile(r"(?<![\d])\d{11,19}(?![\d])"),                            # account / card / Aadhaar
    _re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                                 # PAN
]
REDACTED = "[redacted]"


def _is_pii_text(s: str) -> bool:
    return any(p.search(s) for p in _PII_PATTERNS)


def _redact_text(s: str) -> str:
    for p in _PII_PATTERNS:
        s = p.sub(REDACTED, s)
    return s


def _scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if _is_pii_text(ks):
                continue                      # a value used as a key (top_values)
            out[k] = _scrub(v)
        return out
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str):
        return _redact_text(obj)
    return obj


def _scrub_profiles(digest: dict) -> dict:
    """Identifier-like columns lose their example values entirely (this also
    covers hashed ids and UUIDs no pattern would recognise)."""
    profiles = (digest or {}).get("column_profiles")
    if not isinstance(profiles, dict):
        return digest
    for col, prof in profiles.items():
        if not isinstance(prof, dict):
            continue
        low = str(col).lower()
        stats = prof.get("stats") if isinstance(prof.get("stats"), dict) else {}
        ident = (any(h in low for h in _ID_COLUMN_HINTS)
                 or prof.get("semantic_type") in ("identifier", "id", "hashed_pii", "pii")
                 or (stats.get("unique") and stats.get("count")
                     and stats["unique"] > 50 and stats["unique"] / max(stats["count"], 1) > 0.02
                     and prof.get("semantic_type") in ("categorical", "text", "string")))
        if ident:
            for key in _VALUE_KEYS:
                stats.pop(key, None)
                prof.pop(key, None)
            prof["values_withheld"] = "identifier-like column: values are not shared"
    return digest


def scrub_for_sharing(body: dict) -> dict:
    """A deep copy of `body` that is safe for the shared knowledge base:
    identifier-like columns' example values removed, and any VPA, e-mail,
    mobile number, account/card/Aadhaar-length number or PAN anywhere in
    the text redacted. Label classes and business codes (U30, P2M, 5411)
    are kept -- they are what the team needs to understand the data."""
    import copy
    b = copy.deepcopy(body)
    for field_name in ("digest", "reference_distributions"):
        if isinstance(b.get(field_name), dict):
            if field_name == "digest":
                b[field_name] = _scrub_profiles(b[field_name])
            else:
                b[field_name] = {c: ({"values_withheld": "identifier-like column"}
                                     if any(h in str(c).lower() for h in _ID_COLUMN_HINTS) else v)
                                 for c, v in b[field_name].items()}
    for local_only in ("path", "key"):          # this machine's file path
        b.pop(local_only, None)
    return _scrub(b)


def enabled() -> bool:
    """On by default. `minio.team_kb: false` (or SENTINEL_TEAM_KB=0) turns
    the shared knowledge base off -- e.g. for a sandbox that must not
    publish, and for the test suite."""
    import os
    env = os.environ.get("SENTINEL_TEAM_KB", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    from sentinel.config import get_bool
    return get_bool("minio", "team_kb", default=True)


# -- the store -------------------------------------------------------------------

class TeamKB:
    def __init__(self, *, settings: dict | None = None, user: str = "",
                 prefix: str = "", outbox: Path | None = None, client=None,
                 background: bool = True):
        from sentinel.config import get_str
        from sentinel.io.minio import shared_settings
        self.settings = dict(settings if settings is not None else shared_settings())
        self.bucket = self.settings.get("bucket") or "sentinel-data"
        self.prefix = (prefix or get_str("minio", "kb_prefix", default="kb/v1")).strip("/") or "kb/v1"
        if not user:
            from sentinel.identity import current_user
            user = current_user()
        self.user = user
        import os
        self.outbox = (Path(outbox) if outbox else
                       Path(os.environ.get("SENTINEL_KB_OUTBOX") or
                            Path.home() / ".sentinel" / "kb_outbox"))
        self._client = client
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self.last_error = ""
        self._background = background

    # -- availability ---------------------------------------------------------
    def configured(self) -> bool:
        s = self.settings
        return bool(s.get("endpoint") and s.get("access_key") and s.get("secret_key"))

    def _c(self):
        if self._client is None:
            from sentinel.io.data_source import make_client
            self._client = make_client(self.settings, workers=4)
        return self._client

    # -- writing (outbox first, then upload) -----------------------------------
    def _enqueue(self, key: str, body: dict) -> None:
        self.outbox.mkdir(parents=True, exist_ok=True)
        item = self.outbox / f"{time.time_ns()}-{uuid.uuid4().hex[:6]}.json"
        tmp = item.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "body": body}, default=str), encoding="utf-8")
        tmp.replace(item)
        self._kick()

    def _kick(self) -> None:
        if not self.configured() or not self._background:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._drain_loop, daemon=True,
                                            name="sentinel-team-kb")
            self._worker.start()
        self._wake.set()

    def _drain_loop(self) -> None:
        idle = 0
        while idle < 3:
            sent = self.flush()
            if not self.pending():
                idle += 1
            else:
                idle = 0 if sent else idle + 1
            self._wake.wait(timeout=2.0 if self.pending() else 5.0)
            self._wake.clear()

    def flush(self) -> int:
        """Upload what is waiting in the outbox. Returns how many went up.
        Never raises; the error is kept in `last_error`."""
        if not self.configured() or not self.outbox.is_dir():
            return 0
        n = 0
        with self._lock:
            for item in sorted(self.outbox.glob("*.json")):
                try:
                    rec = json.loads(item.read_text(encoding="utf-8"))
                    self._c().put_bytes(self.bucket, rec["key"],
                                        json.dumps(rec["body"], default=str).encode("utf-8"))
                except Exception as e:  # noqa: BLE001 -- stays in the outbox
                    self.last_error = f"{type(e).__name__}: {e}"[:300]
                    log.info("team kb upload deferred: %s", self.last_error)
                    break
                try:
                    item.unlink()
                except OSError:
                    pass
                n += 1
            if n:
                self.last_error = ""
        return n

    def pending(self) -> int:
        return len(list(self.outbox.glob("*.json"))) if self.outbox.is_dir() else 0

    def close(self, timeout: float = 10.0) -> None:
        """Best-effort final upload at exit; whatever remains stays queued."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass

    # -- cards -----------------------------------------------------------------
    def publish_card(self, card) -> None:
        if not getattr(card, "dataset_id", ""):
            return
        body = scrub_for_sharing(card.to_dict())
        body["_published_by"] = self.user
        self._enqueue(f"{self.prefix}/cards/{_slot(card.dataset_id)}/{_rev_name(self.user)}", body)

    def fetch_card(self, dataset_id: str):
        """Every revision of this dataset's card, merged; None if the team has
        none (or the store cannot be reached -- the caller keeps its local card)."""
        if not dataset_id or not self.configured():
            return None
        from sentinel.knowledge.dataset_card import DatasetCard, merge_cards
        base = f"{self.prefix}/cards/{_slot(dataset_id)}/"
        try:
            objs = sorted(self._c().list(self.bucket, base), key=lambda o: o.object_name)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            return None
        merged, used = None, []
        for o in objs:
            try:
                d = json.loads(self._c().get_bytes(self.bucket, o.object_name))
                d.pop("_published_by", None)
                card = DatasetCard.from_dict(d)
            except Exception:  # noqa: BLE001 -- an unreadable revision is skipped, not deleted
                continue
            merged = card if merged is None else merge_cards(merged, card)
            used.append(o.object_name)
        if merged is not None and len(used) >= COMPACT_AT:
            self._compact(base, merged, used)
        return merged

    def _compact(self, base: str, merged, used: list[str]) -> None:
        """Write the merged revision FIRST, then remove only the revisions
        that went into it. Anything written meanwhile is untouched."""
        try:
            body = merged.to_dict()
            body["_published_by"] = f"{self.user} (compacted {len(used)} revisions)"
            self._c().put_bytes(self.bucket, base + _rev_name(self.user),
                                json.dumps(body, default=str).encode("utf-8"))
        except Exception:  # noqa: BLE001 -- no merged copy written: delete nothing
            return
        for name in used:
            try:
                self._c().delete(self.bucket, name)
            except Exception:  # noqa: BLE001 -- no delete right: harmless, just not compacted
                return

    # -- books -----------------------------------------------------------------
    def publish_book(self, dataset_id: str, book_dict: dict) -> None:
        if not dataset_id:
            return
        bid = str(book_dict.get("id") or uuid.uuid4().hex)
        body = dict(scrub_for_sharing(book_dict), dataset_id=dataset_id, _published_by=self.user)
        self._enqueue(f"{self.prefix}/books/{_slot(dataset_id)}/{bid}.json", body)

    def fetch_books(self, dataset_id: str) -> list[dict]:
        return self._fetch_all(f"{self.prefix}/books/{_slot(dataset_id)}/") if dataset_id else []

    # -- findings --------------------------------------------------------------
    def publish_finding(self, dataset_id: str, finding: dict) -> None:
        if not dataset_id:
            return
        body = dict(scrub_for_sharing(finding), dataset_id=dataset_id, _published_by=self.user)
        self._enqueue(f"{self.prefix}/findings/{_slot(dataset_id)}/{_rev_name(self.user)}", body)

    def fetch_findings(self, dataset_id: str) -> list[dict]:
        return self._fetch_all(f"{self.prefix}/findings/{_slot(dataset_id)}/") if dataset_id else []

    def _fetch_all(self, base: str) -> list[dict]:
        if not self.configured():
            return []
        try:
            objs = sorted(self._c().list(self.bucket, base), key=lambda o: o.object_name)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            return []
        out = []
        for o in objs:
            try:
                out.append(json.loads(self._c().get_bytes(self.bucket, o.object_name)))
            except Exception:  # noqa: BLE001
                continue
        return out

    def status(self) -> str:
        if not self.configured():
            return ("team knowledge: OFF -- the object store is not fully configured "
                    "(/minio shows what is missing); cards and findings stay local and "
                    f"queue in {self.outbox}")
        p = self.pending()
        s = f"team knowledge: s3://{self.bucket}/{self.prefix}/ -- automatic"
        if p:
            s += f"; {p} update(s) waiting to upload"
        if self.last_error:
            s += f" (last error: {self.last_error})"
        return s
