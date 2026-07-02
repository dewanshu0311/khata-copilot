"""
Hybrid search over ledger entries — dense (SBERT + FAISS) fused with sparse (BM25).

Pipeline role (Phase 4): a list of LedgerEntryRecords in -> ranked LedgerSearchHits
out. No LLM here; this is pure retrieval that the Insights Agent sits on top of.

Why hybrid (reused from my semantic-search-rag-flipkart repo): a small khata is
searched two very different ways. Customer NAMES need exact keyword matching —
that is BM25's job. Natural-language QUESTIONS ("who hasn't paid in a while?")
need semantic matching — that is the SBERT/FAISS job. We fuse both scores:

    final = alpha * dense_norm + (1 - alpha) * sparse_norm

with alpha biased toward keyword (see HYBRID_ALPHA) because names dominate.

Resilience contract: this must survive with NO network and NO heavy deps. If
sentence-transformers / faiss can't load (offline, model not downloaded), the
index silently drops to BM25-only (dense_enabled = False) instead of crashing —
so mock-mode demos and the test suite always run.
"""
from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np

from core.config import EMBEDDING_MODEL, HYBRID_ALPHA, SEARCH_TOP_K
from core.schemas import LedgerEntryRecord, LedgerSearchHit


# ── Text building + tokenizing ───────────────────────────────────────────────
def build_entry_text(record: LedgerEntryRecord) -> str:
    """The searchable string for one entry: name + amount + status + date (+ raw).

    The amount is rendered plain (no ₹/commas) so a query like "1200" tokenizes to
    a clean "1200" that BM25 can match; the pretty ₹-formatted form lives only in
    the human-facing citation (see LedgerSearchHit.citation).
    """
    amount = f"{record.amount:g}"
    parts = [record.customer_name, f"₹{amount}", record.status, record.raw_date or ""]
    if record.raw_text:
        parts.append(record.raw_text)
    return " ".join(p for p in parts if p).strip()


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens. `\\w+` keeps Devanagari (Hindi names) and digits."""
    return [t for t in re.findall(r"\w+", (text or "").lower()) if t]


# ── Dense embedding wrapper (SBERT all-MiniLM-L6-v2) ─────────────────────────
class _Embedder:
    """Lazy SBERT wrapper (ported from flipkart/src/embedding_model.py).

    Loaded on first use so importing this module never drags in torch, and so the
    keyword-only fallback works even when sentence-transformers is absent.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode + L2-normalize, so FAISS inner product == cosine similarity."""
        import faiss
        model = self._load()
        vecs = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        return vecs


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; all-equal (incl. all-zero) collapses to zeros."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ── The index ────────────────────────────────────────────────────────────────
class LedgerSearchIndex:
    """Builds a hybrid (dense + sparse) index over ledger entries and searches it."""

    def __init__(
        self,
        records: Sequence[LedgerEntryRecord],
        *,
        model_name: str = EMBEDDING_MODEL,
        alpha: float = HYBRID_ALPHA,
    ) -> None:
        self.records: List[LedgerEntryRecord] = list(records)
        self.alpha = alpha
        self.texts = [build_entry_text(r) for r in self.records]
        self.dense_enabled = False
        self._embedder: _Embedder | None = None
        self._faiss = None
        self._bm25 = None
        self._build_sparse()
        self._build_dense(model_name)

    # -- build --------------------------------------------------------------
    def _build_sparse(self) -> None:
        """BM25 over tokenized entry texts. Absent rank_bm25 -> sparse scores are 0."""
        if not self.texts:
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [tokenize(t) for t in self.texts]
            if any(tokenized):  # BM25Okapi needs a non-empty corpus
                self._bm25 = BM25Okapi(tokenized)
        except Exception:
            self._bm25 = None

    def _build_dense(self, model_name: str) -> None:
        """SBERT embeddings in a flat FAISS index. Any failure -> keyword-only."""
        if not self.texts:
            return
        try:
            embedder = _Embedder(model_name)
            vecs = embedder.encode(self.texts)  # may download the model / raise offline
            import faiss
            index = faiss.IndexFlatIP(vecs.shape[1])  # exact search; no training needed
            index.add(vecs)
            self._embedder, self._faiss, self.dense_enabled = embedder, index, True
        except Exception:
            self.dense_enabled = False  # offline / missing deps -> BM25 carries the search

    # -- score --------------------------------------------------------------
    def _sparse_scores(self, query: str, n: int) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(n, dtype="float32")
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype="float32")

    def _dense_scores(self, query: str, n: int) -> np.ndarray:
        if not self.dense_enabled:
            return np.zeros(n, dtype="float32")
        try:
            qvec = self._embedder.encode([query])
            sims, idxs = self._faiss.search(qvec, n)
            scores = np.zeros(n, dtype="float32")
            for sim, idx in zip(sims[0], idxs[0]):
                if 0 <= idx < n:
                    scores[idx] = sim
            return scores
        except Exception:
            return np.zeros(n, dtype="float32")

    # -- search -------------------------------------------------------------
    def search(self, query: str, k: int = SEARCH_TOP_K) -> List[LedgerSearchHit]:
        """Return up to k entries ranked by fused hybrid score, most relevant first."""
        n = len(self.records)
        if n == 0:
            return []

        dense = _minmax(self._dense_scores(query, n))
        sparse = _minmax(self._sparse_scores(query, n))
        fused = self.alpha * dense + (1.0 - self.alpha) * sparse

        order = np.argsort(fused)[::-1][:k]
        # If we have real signal, drop zero-score tail rather than pad the LLM's
        # context with irrelevant customers; if everything is zero, keep top-k.
        if fused[order[0]] > 0:
            order = [i for i in order if fused[i] > 0]

        return [
            LedgerSearchHit(
                record=self.records[i],
                score=round(float(fused[i]), 4),
                dense_score=round(float(dense[i]), 4),
                sparse_score=round(float(sparse[i]), 4),
            )
            for i in order
        ]


# ── Extractive fallback (no LLM) ─────────────────────────────────────────────
def extractive_answer(question: str, hits: Sequence[LedgerSearchHit], limit: int = 3) -> str:
    """Build a grounded answer straight from the top hits when the LLM is down.

    Reused idea from my Rag-Assistant-masterclass extractive_fallback_answer: when
    generation is unavailable, surface the retrieved facts verbatim (exact numbers,
    real names, entry citations) rather than failing. Honest, if less fluent.
    """
    if not hits:
        return "I couldn't find any matching entries in your ledger for that question."
    lines = []
    for hit in hits[:limit]:
        r = hit.record
        verb = "has paid" if r.status == "paid" else "still owes"
        when = f" ({r.raw_date})" if r.raw_date else ""
        lines.append(f"{r.customer_name} {verb} ₹{r.amount:,.0f}{when} [Entry #{r.id}].")
    return "Based on your ledger: " + " ".join(lines)
