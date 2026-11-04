"""
ledgerhound.memory.bm25
=====================

A dependency-free BM25 lexical retriever -- the sparse half of the moat's
hybrid recall (the dense half is bge-large vectors, AUD-137/AUD-150).

Why sparse at all
-----------------
Dense embeddings capture meaning but miss exact tokens: account ids, column
names like ``payee_device_id``, typology codes, rupee thresholds. BM25 nails
those literal matches. Fusing the two (reciprocal-rank fusion) gives recall
that is robust to both paraphrase (dense) and exact terminology (sparse) --
the standard hybrid-retrieval result.

Pure Python on purpose: no ``rank_bm25`` / ``elasticsearch`` dependency, so it
runs in CI and on-prem with nothing installed, mirroring the hash-fallback
embedder. Corpora here are thousands of short findings, not millions of docs,
so a NumPy scan per query is ample.

BM25 Okapi with the usual defaults (k1=1.5, b=0.75).
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens. Keeps underscores so column names
    like ``payee_device_id`` survive as a single token."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25:
    """In-memory BM25 Okapi index over a fixed corpus of documents.

    Built once from a list of (doc_id, text) pairs; ``search`` returns ranked
    ``(doc_id, score)`` tuples. Cheap to rebuild, so callers reconstruct it
    when the underlying corpus changes rather than mutating it in place.
    """

    def __init__(
        self,
        documents: Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        for doc_id, text in documents:
            self.doc_ids.append(doc_id)
            self._tokens.append(tokenize(text))

        self.n = len(self._tokens)
        self.doc_len = np.array([len(t) for t in self._tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0

        # Document frequency per term, and per-doc term frequencies.
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        for toks in self._tokens:
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        # Smoothed IDF (BM25 form); clamp at 0 so common terms never go
        # negative and subtract from the score.
        self._idf: dict[str, float] = {}
        for term, df in self._df.items():
            self._idf[term] = max(0.0, math.log((self.n - df + 0.5) / (df + 0.5) + 1.0))

    def __len__(self) -> int:
        return self.n

    def scores(self, query: str) -> np.ndarray:
        """Raw BM25 score for every document, aligned to ``self.doc_ids``."""
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        q_terms = set(tokenize(query))
        if not q_terms:
            return out
        for term in q_terms:
            idf = self._idf.get(term)
            if not idf:
                continue
            for i in range(self.n):
                f = self._tf[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                out[i] += idf * (f * (self.k1 + 1.0)) / denom
        return out

    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        """Top-k ``(doc_id, score)`` by BM25, dropping zero-score docs."""
        if not self.n:
            return []
        sims = self.scores(query)
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for idx in order:
            s = float(sims[idx])
            if s <= 0.0:
                break
            out.append((self.doc_ids[int(idx)], s))
            if len(out) >= k:
                break
        return out


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[str]],
    *,
    k0: int = 60,
) -> dict[str, float]:
    """Reciprocal-rank fusion over several ranked id lists.

    RRF score for an id = sum over lists of 1/(k0 + rank), rank 0-based.
    k0=60 is the value from the original RRF paper; it damps the influence of
    any single list's top rank so dense and sparse contribute comparably.
    Returns {id: fused_score}; caller sorts descending.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k0 + rank)
    return fused
