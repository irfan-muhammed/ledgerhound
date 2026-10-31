"""
sentinel.memory.longitudinal
============================

LongitudinalMemory -- the layer that gives Sentinel "how the data was before
vs how it is now" recall.

What it stores
--------------
Findings, typology centroids, account fingerprints, knowledge entries -- each
embedded once on entry, indexed in a vector store, and tagged with:

  * `source`      -- "finding" | "typology_centroid" | "account_fingerprint" |
                    "knowledge_entry" | "free_text"
  * `as_of`       -- ISO date for "when this observation was made"
  * `dataset`     -- dataset name this came from
  * `period`      -- analytical period label ("2024-Q2") if applicable
  * `confidence`  -- float in [0,1], optional
  * `superseded_by` -- id of a newer record that replaces this one, optional
  * `tags`        -- caller-supplied list

Three separate namespaces
-------------------------
Text, centroid, fingerprint have different dimensionality. Each gets its own
vector store on disk:

    memory_root/
      text/        # findings + knowledge + free-text
      centroid/    # typology centroids
      fingerprint/ # account fingerprints

`recall(query, namespace="text"|"centroid"|"fingerprint")` searches one.
For most user-facing queries, namespace="text" is correct.

Temporal windowing
------------------
`recall(..., window="last_6_months" | "last_quarter" | "all" | ISO_date_range)`
applies a filter at search time. Implemented via a `filter_fn` passed to the
backend. Backends with native filtering will get a richer impl later; for now,
numpy and faiss both honour filter_fn via the over-fetch-then-filter strategy.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

from sentinel.memory.embeddings import Embedder
from sentinel.memory.vector_store import (
    VectorRecord,
    VectorStore,
    create_store,
    load_store,
    save_store,
)

log = logging.getLogger(__name__)


# -- Source taxonomy --------------------------------------------------------


class MemorySource(str):
    FINDING = "finding"
    TYPOLOGY_CENTROID = "typology_centroid"
    ACCOUNT_FINGERPRINT = "account_fingerprint"
    KNOWLEDGE_ENTRY = "knowledge_entry"
    FREE_TEXT = "free_text"


VALID_SOURCES = frozenset({
    MemorySource.FINDING,
    MemorySource.TYPOLOGY_CENTROID,
    MemorySource.ACCOUNT_FINGERPRINT,
    MemorySource.KNOWLEDGE_ENTRY,
    MemorySource.FREE_TEXT,
})


# -- Memory record metadata -------------------------------------------------


@dataclass
class MemoryRecord:
    """The metadata payload attached to every stored vector."""
    id: str
    source: str
    text: Optional[str] = None
    as_of: str = ""                    # ISO-8601
    dataset: Optional[str] = None
    period: Optional[str] = None
    confidence: Optional[float] = None
    superseded_by: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        known = {"id", "source", "text", "as_of", "dataset", "period",
                 "confidence", "superseded_by", "tags", "extra"}
        return cls(**{k: v for k, v in d.items() if k in known})


# -- Recall result ----------------------------------------------------------


@dataclass
class Recall:
    record: MemoryRecord
    score: float
    namespace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": self.score,
            "namespace": self.namespace,
        }


# -- Longitudinal memory ----------------------------------------------------


NAMESPACES = ("text", "centroid", "fingerprint")


class LongitudinalMemory:
    """
    The institutional memory layer. Holds three vector stores (one per
    namespace) plus a sidecar metadata index for fast filtered recall.

    Parameters
    ----------
    root : Path
        Directory under which the three namespaces persist.
    embedder : Embedder
        Used for text records and (optionally) re-normalisation of numeric
        inputs.
    backend : "auto" | "numpy" | "faiss" | "cuvs"
        Passed through to create_store(). Default "auto" picks numpy.
        Override per machine via sentinel doctor's compute_profile.
    centroid_dim : int | None
        Dimensionality of typology centroids. Required if you'll store any.
    fingerprint_dim : int | None
        Dimensionality of account fingerprints (Sentinel uses 14). Required
        if you'll store any.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        embedder: Optional[Embedder] = None,
        backend: str = "auto",
        centroid_dim: Optional[int] = None,
        fingerprint_dim: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self.backend = backend
        self._dims = {
            "text": self.embedder.text_dim,
            "centroid": centroid_dim,
            "fingerprint": fingerprint_dim,
        }
        self._stores: dict[str, Optional[VectorStore]] = {n: None for n in NAMESPACES}
        self._records: dict[str, dict[str, MemoryRecord]] = {n: {} for n in NAMESPACES}
        # Lazy BM25 sparse index over the text namespace (rebuilt when stale).
        self._bm25 = None
        self._bm25_n = -1
        self._load_existing()

    # -- Existing-state restore -------------------------------------------

    def _ns_path(self, namespace: str) -> Path:
        return self.root / namespace

    def _load_existing(self) -> None:
        for ns in NAMESPACES:
            p = self._ns_path(ns)
            if (p / "meta.json").exists():
                try:
                    store = load_store(p, backend=self.backend)
                    self._stores[ns] = store
                    self._dims[ns] = store.dim
                    rec_lines = (p / "memory_records.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines() if (p / "memory_records.jsonl").exists() else []
                    for line in rec_lines:
                        line = line.strip()
                        if not line:
                            continue
                        rec = MemoryRecord.from_dict(json.loads(line))
                        self._records[ns][rec.id] = rec
                except Exception as exc:
                    log.warning("Could not restore %s namespace: %s", ns, exc)

    # -- Adding records ---------------------------------------------------

    def remember(
        self,
        *,
        source: str,
        text: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        as_of: Optional[str] = None,
        dataset: Optional[str] = None,
        period: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[list[str]] = None,
        extra: Optional[dict[str, Any]] = None,
        record_id: Optional[str] = None,
        supersedes: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Add a memory record. Exactly one of `text` or `vector` must be set
        (text for text-source records; vector for centroid/fingerprint).
        """
        if source not in VALID_SOURCES:
            raise ValueError(f"Unknown source {source!r}")
        if (text is None) == (vector is None):
            raise ValueError("Provide exactly one of `text` or `vector`")

        ns = _namespace_for_source(source)
        # Embed
        if text is not None:
            emb = self.embedder.embed_text(text)
        else:
            emb = self.embedder.embed_vector(vector)  # type: ignore[arg-type]
        emb = np.asarray(emb, dtype=np.float32).ravel()

        # Lazy-init store the first time we see a namespace
        if self._stores[ns] is None:
            dim = self._dims[ns] if self._dims[ns] else emb.shape[0]
            self._stores[ns] = create_store(self.backend, dim)
            self._dims[ns] = dim

        if emb.shape[0] != self._dims[ns]:
            raise ValueError(
                f"Embedding dim {emb.shape[0]} != namespace '{ns}' dim {self._dims[ns]}"
            )

        rid = record_id or _new_id(source)
        rec = MemoryRecord(
            id=rid,
            source=source,
            text=text,
            as_of=as_of or _now_iso(),
            dataset=dataset,
            period=period,
            confidence=confidence,
            superseded_by=None,
            tags=list(tags or []),
            extra=dict(extra or {}),
        )

        self._stores[ns].add([VectorRecord(id=rid, vector=emb, metadata=rec.to_dict())])
        self._records[ns][rid] = rec

        if supersedes is not None:
            self.supersede(supersedes, rid)

        return rec

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark `old_id` as superseded by `new_id`. Returns True on success."""
        for ns in NAMESPACES:
            if old_id in self._records[ns]:
                self._records[ns][old_id].superseded_by = new_id
                # Reflect in the vector store metadata too (for filter_fn).
                store = self._stores[ns]
                if store is not None:
                    # NumpyBruteForce / FaissCpu both expose `_records`.
                    for vr in getattr(store, "_records", []):
                        if vr.id == old_id:
                            vr.metadata["superseded_by"] = new_id
                            break
                return True
        return False

    # -- Recall -----------------------------------------------------------

    def recall(
        self,
        query: Union[str, np.ndarray],
        *,
        namespace: str = "text",
        k: int = 10,
        window: Optional[str] = None,
        source: Optional[str] = None,
        dataset: Optional[str] = None,
        include_superseded: bool = False,
        hybrid: bool = True,
    ) -> list[Recall]:
        """
        Search a namespace for the k nearest records.

        Parameters
        ----------
        query    : str (for text namespace) or np.ndarray (for centroid/fingerprint)
        window   : "last_Nd" / "last_Nm" / "last_Ny" / "all" / "since:ISO" /
                   "between:ISOa:ISOb"; None = "all"
        source   : restrict to one MemorySource value
        dataset  : restrict to one dataset name
        include_superseded : if False (default), drop records that have been
                   superseded by a newer one.
        hybrid   : for the text namespace with a string query, fuse the dense
                   (bge-large) ranking with a BM25 sparse ranking via
                   reciprocal-rank fusion. Ignored for numeric namespaces or
                   vector queries. Score on the returned Recall is then the
                   fused RRF score (not cosine).
        """
        if namespace not in NAMESPACES:
            raise ValueError(f"namespace must be one of {NAMESPACES}, got {namespace!r}")
        store = self._stores[namespace]
        if store is None or len(store) == 0:
            return []

        time_filter = _build_window_filter(window)

        def _meta_ok(meta: dict) -> bool:
            if not include_superseded and meta.get("superseded_by"):
                return False
            if source is not None and meta.get("source") != source:
                return False
            if dataset is not None and meta.get("dataset") != dataset:
                return False
            if time_filter is not None and not time_filter(meta.get("as_of", "")):
                return False
            return True

        use_hybrid = (
            hybrid and namespace == "text" and isinstance(query, str)
            and query.strip()
        )
        if use_hybrid:
            return self._recall_hybrid(query, store, k, _meta_ok, namespace)

        if isinstance(query, str):
            # Asymmetric retrieval: queries get the instruction prefix (BGE);
            # stored passages were embedded without it.
            qvec = self.embedder.embed_query(query)
        else:
            qvec = self.embedder.embed_vector(query)

        raw = store.search(
            np.asarray(qvec, dtype=np.float32).ravel(), k=k,
            filter_fn=lambda vr: _meta_ok(vr.metadata),
        )
        out: list[Recall] = []
        for vr, score in raw:
            rec = self._records[namespace].get(vr.id)
            if rec is None:
                # Recover from store metadata if local cache is missing.
                rec = MemoryRecord.from_dict(vr.metadata)
            out.append(Recall(record=rec, score=score, namespace=namespace))
        return out

    # -- Hybrid (dense + BM25) recall --------------------------------------

    def _text_bm25(self):
        """Lazily (re)build the BM25 index over the current text corpus."""
        from sentinel.memory.bm25 import BM25
        recs = self._records["text"]
        if self._bm25 is None or self._bm25_n != len(recs):
            self._bm25 = BM25((rid, r.text or "") for rid, r in recs.items())
            self._bm25_n = len(recs)
        return self._bm25

    def _recall_hybrid(self, query, store, k, meta_ok, namespace) -> list[Recall]:
        """Fuse dense (bge-large) + BM25 sparse rankings via RRF."""
        from sentinel.memory.bm25 import reciprocal_rank_fusion

        # Over-fetch each channel so fusion has room to reorder.
        fetch = max(k * 4, 20)

        qvec = self.embedder.embed_query(query)
        dense = store.search(
            np.asarray(qvec, dtype=np.float32).ravel(), k=fetch,
            filter_fn=lambda vr: meta_ok(vr.metadata),
        )
        dense_ids = [vr.id for vr, _ in dense]

        bm25 = self._text_bm25()
        sparse_ids = [
            rid for rid, _ in bm25.search(query, k=fetch)
            if rid in self._records["text"]
            and meta_ok(self._records["text"][rid].to_dict())
        ]

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]

        # Map ids back to records (prefer the local cache; fall back to the
        # dense hit's stored metadata for any id the cache is missing).
        dense_meta = {vr.id: vr.metadata for vr, _ in dense}
        out: list[Recall] = []
        for rid, fscore in ranked:
            rec = self._records["text"].get(rid)
            if rec is None and rid in dense_meta:
                rec = MemoryRecord.from_dict(dense_meta[rid])
            if rec is None:
                continue
            out.append(Recall(record=rec, score=float(fscore), namespace=namespace))
        return out

    # -- Bulk helpers for typology and fingerprint tools ------------------

    def remember_clusters(
        self,
        clusters: Iterable[dict],
        *,
        dataset: Optional[str] = None,
        period: Optional[str] = None,
        centroid_key: str = "centroid",
        id_key: str = "cluster_id",
    ) -> list[MemoryRecord]:
        """Embed and store each cluster centroid as a TYPOLOGY_CENTROID
        memory. `clusters` is an iterable of dicts with a numeric centroid
        vector under `centroid_key` and an identifier under `id_key`."""
        out: list[MemoryRecord] = []
        for c in clusters:
            vec = c.get(centroid_key)
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).ravel()
            extras = {k: v for k, v in c.items() if k != centroid_key}
            rec = self.remember(
                source=MemorySource.TYPOLOGY_CENTROID,
                vector=arr,
                dataset=dataset,
                period=period,
                extra=extras,
                record_id=f"typo_{dataset}_{period}_{c.get(id_key, len(out))}" if dataset and period else None,
            )
            out.append(rec)
        return out

    def remember_fingerprints(
        self,
        fingerprints: Iterable[dict],
        *,
        dataset: Optional[str] = None,
        period: Optional[str] = None,
        vector_key: str = "fingerprint",
        id_key: str = "account_id",
    ) -> list[MemoryRecord]:
        """Embed and store account fingerprint vectors as
        ACCOUNT_FINGERPRINT memories."""
        out: list[MemoryRecord] = []
        for fp in fingerprints:
            vec = fp.get(vector_key)
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).ravel()
            account_id = fp.get(id_key)
            extras = {k: v for k, v in fp.items() if k != vector_key}
            rec = self.remember(
                source=MemorySource.ACCOUNT_FINGERPRINT,
                vector=arr,
                dataset=dataset,
                period=period,
                extra=extras,
                record_id=f"acct_{account_id}" if account_id else None,
            )
            out.append(rec)
        return out

    # -- Persistence ------------------------------------------------------

    def save(self) -> None:
        """Flush every namespace to disk."""
        for ns in NAMESPACES:
            store = self._stores[ns]
            if store is None:
                continue
            p = self._ns_path(ns)
            p.mkdir(parents=True, exist_ok=True)
            save_store(store, p)
            with (p / "memory_records.jsonl").open("w", encoding="utf-8") as f:
                for rec in self._records[ns].values():
                    f.write(json.dumps(rec.to_dict(), default=str) + "\n")

    # -- Stats / introspection --------------------------------------------

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"root": str(self.root), "namespaces": {}}
        for ns in NAMESPACES:
            store = self._stores[ns]
            count = 0 if store is None else len(store)
            backend = "(empty)" if store is None else store.name
            out["namespaces"][ns] = {
                "count": count,
                "dim": self._dims[ns],
                "backend": backend,
            }
        return out


# -- Helpers ----------------------------------------------------------------


def _namespace_for_source(source: str) -> str:
    if source in (MemorySource.FINDING, MemorySource.KNOWLEDGE_ENTRY, MemorySource.FREE_TEXT):
        return "text"
    if source == MemorySource.TYPOLOGY_CENTROID:
        return "centroid"
    if source == MemorySource.ACCOUNT_FINGERPRINT:
        return "fingerprint"
    raise ValueError(f"No namespace for source {source!r}")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix[:4]}_{uuid.uuid4().hex[:10]}"


def _build_window_filter(window: Optional[str]) -> Optional[Callable[[str], bool]]:
    """Compile a window spec into a function (as_of: str) -> bool."""
    if window is None or window == "all":
        return None
    now = _dt.datetime.now(_dt.timezone.utc)

    def _parse_iso(s: str) -> Optional[_dt.datetime]:
        if not s:
            return None
        try:
            dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt
        except Exception:
            return None

    if window.startswith("since:"):
        cutoff = _parse_iso(window[6:])
        if cutoff is None:
            return None
        return lambda s: (_parse_iso(s) is not None and _parse_iso(s) >= cutoff)  # type: ignore[operator]

    if window.startswith("between:"):
        try:
            a_str, b_str = window[8:].split(":", 1)
            a = _parse_iso(a_str)
            b = _parse_iso(b_str)
        except Exception:
            return None
        if a is None or b is None:
            return None
        return lambda s: (_parse_iso(s) is not None and a <= _parse_iso(s) <= b)  # type: ignore[operator]

    # last_Nd / last_Nm / last_Ny
    if window.startswith("last_"):
        rest = window[5:]
        try:
            num = int(rest[:-1])
            unit = rest[-1]
        except Exception:
            return None
        if unit == "d":
            cutoff = now - _dt.timedelta(days=num)
        elif unit == "m":
            cutoff = now - _dt.timedelta(days=num * 30)
        elif unit == "y":
            cutoff = now - _dt.timedelta(days=num * 365)
        else:
            return None
        return lambda s: (_parse_iso(s) is not None and _parse_iso(s) >= cutoff)  # type: ignore[operator]

    return None
