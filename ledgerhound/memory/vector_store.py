"""
ledgerhound.memory.vector_store
============================

VectorStore -- backend-agnostic interface for adding and searching dense
float32 vectors with metadata.

Three backends, chosen by `ledgerhound doctor` based on corpus size and
recall latency budget:

  * **NumpyBruteForce** -- exact cosine similarity. Best up to ~50K vectors.
    Always available. No external deps.
  * **FaissCPU** -- HNSW or IVF-Flat. Sweet spot 50K - 1M vectors. Requires
    `faiss-cpu`. Falls back to NumpyBruteForce if not installed.
  * **CuVS** -- NVIDIA CUDA Vector Search (CAGRA / IVF-PQ). For 1M+ vectors
    with sub-millisecond recall. Stub here; M5 (`ledgerhound doctor`) decides
    whether to install / use this backend per machine.

All backends share the same `VectorStore` interface so the upstream code
never references a specific backend.

Persistence layout
------------------
A store persists as a directory:

    store_root/
      meta.json          # backend, dim, count, created_at
      vectors.npy        # float32 (n, dim)
      records.jsonl      # one record-meta per line, aligned to vector row

This means a vector store can be loaded by any backend (the data is
backend-agnostic) -- only the index structure differs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Union

import numpy as np

log = logging.getLogger(__name__)


# -- Record -----------------------------------------------------------------


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray                # 1D float32
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialise only the meta (not the vector -- that's in the .npy)."""
        return {"id": self.id, "metadata": self.metadata}

    @classmethod
    def from_meta(cls, d: dict, vector: np.ndarray) -> "VectorRecord":
        return cls(id=d["id"], vector=vector, metadata=d.get("metadata", {}))


# -- Backend probe ----------------------------------------------------------


def _have_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def _have_cuvs() -> bool:
    try:
        import cuvs  # type: ignore[import-not-found] # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> list[str]:
    names = ["numpy"]
    if _have_faiss():
        names.append("faiss")
    if _have_cuvs():
        names.append("cuvs")
    return names


# -- Interface --------------------------------------------------------------


class VectorStore(Protocol):
    """Minimum contract every backend implements."""

    name: str
    dim: int

    def add(self, records: Iterable[VectorRecord]) -> int: ...
    def search(self, query: np.ndarray, *, k: int = 10,
               filter_fn: Optional[Any] = None) -> list[tuple[VectorRecord, float]]: ...
    def delete(self, ids: list[str]) -> int: ...
    def __len__(self) -> int: ...
    def save(self, path: Union[str, Path]) -> None: ...


# -- NumpyBruteForce backend ------------------------------------------------


class NumpyBruteForceStore:
    """
    Exact cosine similarity via dense matmul. No external deps.
    O(n * d) per query -- fine for tens of thousands of vectors.
    Vectors are assumed L2-normalised (caller's responsibility); cosine is
    then just a dot product.
    """

    name = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._vectors: list[np.ndarray] = []
        self._records: list[VectorRecord] = []
        self._id_index: dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None  # cached stack

    def __len__(self) -> int:
        return len(self._records)

    def _invalidate_cache(self) -> None:
        self._matrix = None

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self._vectors:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            else:
                self._matrix = np.vstack(self._vectors).astype(np.float32, copy=False)
        return self._matrix

    def add(self, records: Iterable[VectorRecord]) -> int:
        n = 0
        for r in records:
            v = np.asarray(r.vector, dtype=np.float32).ravel()
            if v.shape[0] != self.dim:
                raise ValueError(f"vector dim {v.shape[0]} != store dim {self.dim} for id={r.id}")
            if r.id in self._id_index:
                # Overwrite in place.
                idx = self._id_index[r.id]
                self._vectors[idx] = v
                self._records[idx] = VectorRecord(id=r.id, vector=v, metadata=r.metadata)
            else:
                self._id_index[r.id] = len(self._vectors)
                self._vectors.append(v)
                self._records.append(VectorRecord(id=r.id, vector=v, metadata=r.metadata))
            n += 1
        self._invalidate_cache()
        return n

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 10,
        filter_fn: Optional[Any] = None,
    ) -> list[tuple[VectorRecord, float]]:
        if not self._vectors:
            return []
        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != store dim {self.dim}")
        M = self._ensure_matrix()
        sims = M @ q  # cosine if both sides are L2-normalised
        order = np.argsort(-sims)
        results: list[tuple[VectorRecord, float]] = []
        for idx in order:
            rec = self._records[int(idx)]
            if filter_fn is not None:
                try:
                    if not filter_fn(rec):
                        continue
                except Exception:
                    continue
            results.append((rec, float(sims[idx])))
            if len(results) >= k:
                break
        return results

    def delete(self, ids: list[str]) -> int:
        removed = 0
        for rid in ids:
            if rid in self._id_index:
                idx = self._id_index.pop(rid)
                # Mark with ledgerhound for rebuild.
                self._vectors[idx] = None  # type: ignore[assignment]
                self._records[idx] = None  # type: ignore[assignment]
                removed += 1
        if removed:
            self._vectors = [v for v in self._vectors if v is not None]
            self._records = [r for r in self._records if r is not None]
            self._id_index = {r.id: i for i, r in enumerate(self._records)}
            self._invalidate_cache()
        return removed

    def save(self, path: Union[str, Path]) -> None:
        save_store(self, Path(path))


# -- FaissCPU backend --------------------------------------------------------


class FaissCpuStore:
    """
    Faiss IndexFlatIP (exact inner product). When vectors are L2-normalised,
    IP == cosine. Suitable up to ~1M vectors on CPU; for >1M use IVF or HNSW
    (left as a future optimisation -- ledgerhound doctor decides per machine).
    """

    name = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss
        self.dim = int(dim)
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dim)
        self._records: list[VectorRecord] = []
        self._id_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add(self, records: Iterable[VectorRecord]) -> int:
        batch: list[np.ndarray] = []
        new_records: list[VectorRecord] = []
        for r in records:
            v = np.asarray(r.vector, dtype=np.float32).ravel()
            if v.shape[0] != self.dim:
                raise ValueError(f"vector dim {v.shape[0]} != store dim {self.dim}")
            if r.id in self._id_index:
                # Overwrite isn't trivial in IndexFlatIP -- rebuild on overwrite.
                self.delete([r.id])
            batch.append(v)
            new_records.append(VectorRecord(id=r.id, vector=v, metadata=r.metadata))
        if not batch:
            return 0
        mat = np.vstack(batch).astype(np.float32, copy=False)
        self._index.add(mat)
        for rec in new_records:
            self._id_index[rec.id] = len(self._records)
            self._records.append(rec)
        return len(batch)

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 10,
        filter_fn: Optional[Any] = None,
    ) -> list[tuple[VectorRecord, float]]:
        if not self._records:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        # Over-fetch when filtering so we can drop matches without short-counting.
        fetch_k = k * 4 if filter_fn is not None else k
        fetch_k = min(fetch_k, len(self._records))
        dists, idxs = self._index.search(q, fetch_k)
        out: list[tuple[VectorRecord, float]] = []
        for i, d in zip(idxs[0], dists[0]):
            if i < 0:
                continue
            rec = self._records[int(i)]
            if filter_fn is not None:
                try:
                    if not filter_fn(rec):
                        continue
                except Exception:
                    continue
            out.append((rec, float(d)))
            if len(out) >= k:
                break
        return out

    def delete(self, ids: list[str]) -> int:
        keep_records: list[VectorRecord] = []
        removed = 0
        for r in self._records:
            if r.id in ids:
                removed += 1
            else:
                keep_records.append(r)
        if not removed:
            return 0
        self._records = keep_records
        self._id_index = {r.id: i for i, r in enumerate(self._records)}
        # Rebuild index.
        self._index = self._faiss.IndexFlatIP(self.dim)
        if self._records:
            mat = np.vstack([r.vector for r in self._records]).astype(np.float32, copy=False)
            self._index.add(mat)
        return removed

    def save(self, path: Union[str, Path]) -> None:
        save_store(self, Path(path))


# -- CuVS backend (stub for now) ---------------------------------------------


class CuVSStore:
    """
    CUDA Vector Search via cuVS. CAGRA index for large corpora.

    This is a stub. Full impl is gated on `ledgerhound doctor` deciding the
    machine has cuVS installed and the corpus is large enough to benefit
    (>1M vectors typically). Until then, factory will not return this backend.
    """

    name = "cuvs"

    def __init__(self, dim: int) -> None:
        raise NotImplementedError(
            "CuVS backend not yet implemented. "
            "ledgerhound doctor will activate it when corpus size warrants."
        )


# -- Factory + persistence ---------------------------------------------------


def create_store(
    backend: str,
    dim: int,
) -> VectorStore:
    """Construct a store by backend name. Falls back to numpy if requested
    backend is unavailable. Returns the store (cast to the Protocol)."""
    b = (backend or "auto").lower()
    if b == "auto":
        # Conservative default: numpy unless someone explicitly opts up.
        # ledgerhound doctor's job to escalate.
        return NumpyBruteForceStore(dim)  # type: ignore[return-value]
    if b == "numpy":
        return NumpyBruteForceStore(dim)  # type: ignore[return-value]
    if b == "faiss":
        if _have_faiss():
            return FaissCpuStore(dim)  # type: ignore[return-value]
        log.warning("faiss-cpu not installed -- falling back to numpy backend")
        return NumpyBruteForceStore(dim)  # type: ignore[return-value]
    if b == "cuvs":
        if _have_cuvs():
            return CuVSStore(dim)  # type: ignore[return-value]
        log.warning("cuVS not installed -- falling back to faiss/numpy backend")
        return create_store("faiss", dim)
    raise ValueError(f"Unknown backend: {backend!r}")


# -- On-disk save / load ----------------------------------------------------


def save_store(store: VectorStore, path: Path) -> None:
    """Persist a store directory: vectors.npy + records.jsonl + meta.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    records: list[VectorRecord]
    if isinstance(store, NumpyBruteForceStore):
        records = store._records
    elif isinstance(store, FaissCpuStore):
        records = store._records
    else:
        # Fall back to private attr name convention.
        records = getattr(store, "_records", [])  # type: ignore[arg-type]

    if records:
        mat = np.vstack([r.vector for r in records]).astype(np.float32, copy=False)
    else:
        mat = np.zeros((0, store.dim), dtype=np.float32)

    np.save(path / "vectors.npy", mat)
    with (path / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_meta_dict(), default=str) + "\n")
    meta = {
        "backend": store.name,
        "dim": store.dim,
        "count": len(records),
        "saved_at": time.time(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_store(path: Union[str, Path], *, backend: Optional[str] = None) -> VectorStore:
    """Load a store from disk. `backend` overrides the saved backend; useful
    if the corpus grew and you want to swap numpy->faiss without re-embedding."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    dim = int(meta["dim"])
    target = backend or meta.get("backend", "numpy")
    store = create_store(target, dim)
    mat = np.load(path / "vectors.npy")
    rec_lines = (path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    records: list[VectorRecord] = []
    for i, line in enumerate(rec_lines):
        if not line.strip():
            continue
        d = json.loads(line)
        records.append(VectorRecord.from_meta(d, mat[i]))
    store.add(records)
    return store
