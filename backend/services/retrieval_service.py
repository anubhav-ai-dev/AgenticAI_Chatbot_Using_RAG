"""
Hybrid retrieval: dense (FAISS cosine) ∥ lexical (FTS5 BM25) → Reciprocal Rank
Fusion → cross-encoder rerank → relevance gating. Degrades gracefully:
no embedder → lexical only; no reranker → fused order gated by cosine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from backend.core.config import Settings
from backend.services.embedding_service import EmbeddingService
from backend.services.index_store import ChunkRecord, HybridIndexStore
from backend.services.rerank_service import RerankService
from backend.utils.cache import TTLCache
from backend.utils.ranking import reciprocal_rank_fusion
from backend.utils.text import normalize_query, query_terms, term_coverage

logger = logging.getLogger(__name__)


@dataclass
class RetrievedItem:
    chunk_id: int
    source: str
    page: Optional[int]
    text: str
    relevance: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    items: list[RetrievedItem] = field(default_factory=list)
    used_rag: bool = False
    mode: str = "none"             # hybrid | dense | lexical | none
    reranked: bool = False
    cached: bool = False
    error: Optional[str] = None
    timings: dict[str, float] = field(default_factory=dict)


class RetrievalService:
    def __init__(self, settings: Settings, store: HybridIndexStore,
                 embedder: EmbeddingService, reranker: RerankService) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._results: TTLCache = TTLCache(maxsize=256, ttl_s=settings.retrieval_cache_ttl_s)
        self._qvecs: TTLCache = TTLCache(maxsize=512, ttl_s=3600)

    # ── helpers ──────────────────────────────────────────────────────────────

    def has_documents(self, user_id: str) -> bool:
        return self._store.has_documents(user_id)

    def version(self, user_id: str) -> int:
        return self._store.version(user_id)

    async def embed_query(self, text: str) -> Optional[np.ndarray]:
        """Cached query embedding; None when no embedding backend is available."""
        key = normalize_query(text)
        hit = self._qvecs.get(key)
        if hit is not None:
            return hit
        if not await self._embedder.warmup():
            return None
        try:
            vec = await self._embedder.aembed_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query embedding failed: %s", exc)
            return None
        self._qvecs.set(key, vec)
        return vec

    def stats(self) -> dict[str, Any]:
        return {"results_cache": self._results.stats(), "query_vec_cache": self._qvecs.stats()}

    # ── search ───────────────────────────────────────────────────────────────

    async def search(self, user_id: str, query: str, top_k: Optional[int] = None,
                     min_relevance: Optional[float] = None) -> RetrievalResult:
        t0 = time.perf_counter()
        top_k = top_k or self._s.rag_top_k
        if not query.strip() or not self._store.has_documents(user_id):
            return RetrievalResult(timings={"total_ms": 0.0})

        key = (user_id, self._store.version(user_id), normalize_query(query), top_k, min_relevance)
        cached = self._results.get(key)
        if cached is not None:
            return RetrievalResult(**{**cached.__dict__, "cached": True,
                                      "timings": {"total_ms": (time.perf_counter() - t0) * 1000}})
        try:
            result = await asyncio.wait_for(
                self._search(user_id, query, top_k, min_relevance), self._s.retrieval_timeout_s)
        except asyncio.TimeoutError:
            return RetrievalResult(error="timeout",
                                   timings={"total_ms": (time.perf_counter() - t0) * 1000})
        except Exception as exc:  # noqa: BLE001 — never break the chat on retrieval errors
            logger.exception("Retrieval failed")
            return RetrievalResult(error=f"{type(exc).__name__}",
                                   timings={"total_ms": (time.perf_counter() - t0) * 1000})
        result.timings["total_ms"] = (time.perf_counter() - t0) * 1000
        self._results.set(key, result)
        return result

    async def _search(self, user_id: str, query: str, top_k: int,
                      min_relevance: Optional[float]) -> RetrievalResult:
        s = self._s
        timings: dict[str, float] = {}

        t = time.perf_counter()
        qvec = await self.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        lexical_task = asyncio.to_thread(self._store.lexical_search, user_id, query, s.rag_candidate_k)
        if qvec is not None:
            dense, lexical = await asyncio.gather(
                asyncio.to_thread(self._store.dense_search, user_id, qvec, s.rag_candidate_k),
                lexical_task)
        else:
            dense, lexical = [], await lexical_task
        timings["search_ms"] = (time.perf_counter() - t) * 1000

        if not dense and not lexical:
            return RetrievalResult(mode="hybrid" if qvec is not None else "lexical", timings=timings)

        fused = reciprocal_rank_fusion(
            [([i for i, _ in dense], s.rag_dense_weight),
             ([i for i, _ in lexical], s.rag_lexical_weight)], s.rrf_k)
        mode = "hybrid" if dense and lexical else ("dense" if dense else "lexical")
        candidate_ids = [cid for cid, _ in fused[: s.rerank_candidates]]
        records = await asyncio.to_thread(self._store.get_chunks, candidate_ids)
        ordered = [records[i] for i in candidate_ids if i in records]
        ordered = self._dedupe(ordered)
        if not ordered:
            return RetrievalResult(mode=mode, timings=timings)

        # Dense cosine is a robust signal for vague paraphrases where cross-encoders saturate
        # to ~0; the cross-encoder is sharper on explicit queries. Relevance = max of both.
        floor, ceil = s.rag_cosine_floor, max(s.rag_cosine_ceil, s.rag_cosine_floor + 1e-6)
        cos_scores: dict[int, float] = {}
        if qvec is not None:
            cos = await asyncio.to_thread(self._store.cosines, user_id, qvec, [r.id for r in ordered])
            cos_scores = {i: min(1.0, max(0.0, (c - floor) / (ceil - floor))) for i, c in cos.items()}

        rerank_scores: dict[int, float] = {}
        reranked = False
        if self._reranker.enabled:
            t = time.perf_counter()
            try:
                vals = await self._reranker.arerank(
                    query, [r.text[: s.rerank_max_chars] for r in ordered])
                rerank_scores = {r.id: v for r, v in zip(ordered, vals)}
                reranked = True
            except Exception as exc:  # noqa: BLE001 — fall back to dense/fused order
                logger.warning("Rerank skipped (%s: %s)", type(exc).__name__, exc)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        # Lexical evidence: chunks BM25 matched, scored by how many query terms they cover.
        terms = query_terms(query)
        lex_ids = {cid for cid, _ in lexical}
        lex_scores = {
            r.id: 0.3 + 0.7 * cov
            for r in ordered
            if r.id in lex_ids and (cov := term_coverage(terms, r.text)) > 0
        }

        gate = s.rag_min_relevance if min_relevance is None else min_relevance
        if qvec is None and not reranked:              # lexical-only: bm25-derived pseudo score
            lex = dict(lexical)
            best = max(lex.values()) if lex else 1.0
            scores = {r.id: 0.5 + 0.5 * (lex.get(r.id, 0.0) / best if best > 0 else 0.0)
                      for r in ordered}
            ordered = [r for r in ordered if lex.get(r.id, 0.0) >= 0.25 * best]
        else:
            scores = {}
            for r in ordered:
                rr, cs = rerank_scores.get(r.id), cos_scores.get(r.id)
                # no vector/rerank signal (vector-less chunk, reranker off): neutral pass-through
                base = max(gate, 0.5) if rr is None and cs is None else max(rr or 0.0, cs or 0.0)
                scores[r.id] = max(base, lex_scores.get(r.id, 0.0))
            ordered.sort(key=lambda r: scores[r.id], reverse=True)

        items = [
            RetrievedItem(r.id, r.source, r.page, r.text, float(scores[r.id]), r.meta)
            for r in ordered if scores[r.id] >= gate
        ][:top_k]
        return RetrievalResult(items=items, used_rag=bool(items), mode=mode,
                               reranked=reranked, timings=timings)

    @staticmethod
    def _dedupe(records: list[ChunkRecord]) -> list[ChunkRecord]:
        seen: set[str] = set()
        out: list[ChunkRecord] = []
        for r in records:
            key = " ".join(r.text[:160].lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
