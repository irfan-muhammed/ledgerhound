"""
ledgerhound.memory.embeddings
==========================

Embedder -- turns text, centroids, or fingerprint vectors into the dense
float32 vectors used by the vector store.

Three input types
-----------------
* **Text** -- findings, knowledge entries, analyst notes. Goes through a
  sentence-transformer model when available, falls back to a deterministic
  hashing TF-IDF embedder so tests + CI never break.
* **Centroid** -- typology centroid (already a vector). Passed through with
  optional L2 normalisation so it lives in the same space as text embeddings.
  Different dim than text -- kept in a separate namespace by the vector store.
* **Fingerprint** -- 14-dim account fingerprint. Same treatment as centroids.

Device handling
---------------
* `device="auto"` -- uses GPU (torch.cuda) if torch and CUDA are visible,
  else CPU. Sentence-transformers handles this internally; we just pass it.
* `device="cpu"` / `device="gpu"` -- explicit.

Heavy deps are lazy: torch + sentence-transformers only import on first call.
A machine with no transformer install still gets a working (degraded) Embedder.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np

log = logging.getLogger(__name__)

# Locked on-prem default (AUD-137): bge-large-en-v1.5 on GPU. English fraud-EDA
# text; 1024-dim. BGE retrieval uses an instruction prefix on QUERIES only
# (passages get no prefix). Prefixes apply only to the sentence-transformers
# backend; the hash-fallback ignores them.
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_TEXT_DIM = 1024           # bge-large-en-v1.5 output dim
FALLBACK_HASH_DIM = 256
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_PASSAGE_PREFIX = ""


# -- Backend probes ---------------------------------------------------------


def _have_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _have_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(pref: str) -> str:
    """Map 'auto' / 'cpu' / 'gpu' to a concrete device tag."""
    p = (pref or "auto").lower().strip()
    if p == "auto":
        return "cuda" if _have_torch_cuda() else "cpu"
    if p in ("gpu", "cuda"):
        return "cuda" if _have_torch_cuda() else "cpu"
    return "cpu"


# -- Embedder ---------------------------------------------------------------


@dataclass
class EmbedderInfo:
    backend: str          # "sentence-transformers" | "hash-fallback"
    model_name: str
    device: str
    text_dim: int


class Embedder:
    """
    Unified text + vector embedder.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model id. Ignored on hash-fallback backend.
    device : "auto" | "cpu" | "gpu"
        Where to run the transformer. "auto" picks GPU if available.
    normalise : bool
        L2-normalise every output vector. Required for cosine search.
        Default True.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        *,
        normalise: bool = True,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        passage_prefix: str = DEFAULT_PASSAGE_PREFIX,
    ) -> None:
        self.model_name = model_name
        self.normalise = normalise
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.device = resolve_device(device)
        self._st_model = None
        self._backend = "sentence-transformers" if _have_sentence_transformers() else "hash-fallback"
        if self._backend == "hash-fallback":
            warnings.warn(
                "sentence-transformers not installed -- using hash-fallback "
                "embedder (deterministic but semantically weak). "
                "Install with: pip install sentence-transformers",
                RuntimeWarning,
                stacklevel=2,
            )
            self._text_dim = FALLBACK_HASH_DIM
        else:
            self._text_dim = DEFAULT_TEXT_DIM  # corrected on first real call

    # -- Introspection ----------------------------------------------------

    @property
    def text_dim(self) -> int:
        return self._text_dim

    @property
    def backend(self) -> str:
        return self._backend

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            backend=self._backend,
            model_name=self.model_name,
            device=self.device,
            text_dim=self._text_dim,
        )

    # -- Public API -------------------------------------------------------

    def embed_text(self, text: Union[str, list[str]], *,
                   is_query: bool = False) -> np.ndarray:
        """
        Embed one string or a list of strings.

        Parameters
        ----------
        is_query:
            When True, prepend ``query_prefix`` (BGE-style asymmetric retrieval:
            queries get an instruction prefix, stored passages do not). Applies
            only to the sentence-transformers backend; ignored on hash-fallback.

        Returns
        -------
        np.ndarray
            1D (text_dim,) for single string, 2D (n, text_dim) for list.
            Always float32. L2-normalised if `normalise=True`.
        """
        single = isinstance(text, str)
        batch = [text] if single else list(text)
        if not batch:
            return np.zeros((0, self._text_dim), dtype=np.float32)

        if self._backend == "sentence-transformers":
            prefix = self.query_prefix if is_query else self.passage_prefix
            if prefix:
                batch = [prefix + t for t in batch]
            vecs = self._st_encode(batch)
        else:
            vecs = self._hash_encode(batch)

        if self.normalise:
            vecs = _l2_normalise(vecs)
        return vecs[0] if single else vecs

    def embed_query(self, text: Union[str, list[str]]) -> np.ndarray:
        """Embed search queries (applies the query instruction prefix)."""
        return self.embed_text(text, is_query=True)

    def embed_vector(self, vec: Union[np.ndarray, list[float]]) -> np.ndarray:
        """
        Pass-through embedder for already-numeric inputs (centroids,
        fingerprints). Accepts 1D or 2D arrays; returns float32, optionally
        L2-normalised.
        """
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 1:
            out = arr.copy()
            if self.normalise:
                out = _l2_normalise(out[None, :])[0]
            return out
        if arr.ndim == 2:
            out = arr.astype(np.float32, copy=True)
            if self.normalise:
                out = _l2_normalise(out)
            return out
        raise ValueError(f"embed_vector expects 1D or 2D array, got {arr.ndim}D")

    # -- Internals --------------------------------------------------------

    def _st_encode(self, batch: list[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # heavy
            log.info("Loading sentence-transformer %s on %s", self.model_name, self.device)
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            # Detect the real model dim from the loaded module.
            try:
                self._text_dim = int(self._st_model.get_sentence_embedding_dimension())
            except Exception:
                pass
        vecs = self._st_model.encode(
            batch,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # we handle this ourselves for consistency
        )
        return np.asarray(vecs, dtype=np.float32)

    def _hash_encode(self, batch: list[str]) -> np.ndarray:
        """
        Deterministic hashing embedder -- bag-of-bytes onto a fixed-width
        vector via blake2b. Not semantic -- exists so the pipeline runs in
        environments without sentence-transformers (CI, smoke tests).
        """
        out = np.zeros((len(batch), FALLBACK_HASH_DIM), dtype=np.float32)
        for i, s in enumerate(batch):
            tokens = (s or "").lower().split()
            for tok in tokens:
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
                idx = int.from_bytes(h, "little") % FALLBACK_HASH_DIM
                out[i, idx] += 1.0
        return out


# -- Helpers ----------------------------------------------------------------


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """Normalise each row to unit L2 norm. Safe on zero-norm rows."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience: cosine similarity between two 1D vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
