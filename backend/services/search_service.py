"""
Web search policy: TTL caching, a hard overall deadline, status mapping and
multi-query fan-out for research mode. HTTP lives in `backend.client.search_client`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from backend.client.search_client import SerperClient, TavilyClient
from backend.core.config import Settings
from backend.models.api import SearchResult
from backend.utils.cache import TTLCache
from backend.utils.text import normalize_query

logger = logging.getLogger(__name__)


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    status: str = "ok"                 # ok | disabled | error | timeout
    elapsed_ms: float = 0.0
    cached: bool = False


class WebSearchService:
    def __init__(self, settings: Settings, client: TavilyClient,
                 fallback: Optional[SerperClient] = None) -> None:
        self._s = settings
        self._client = client
        self._fallback = fallback
        self._cache: TTLCache = TTLCache(maxsize=256, ttl_s=settings.search_cache_ttl_s)

    @property
    def available(self) -> bool:
        return self._client.configured or bool(self._fallback and self._fallback.configured)

    async def search(self, query: str, max_results: Optional[int] = None,
                     depth: Optional[str] = None) -> SearchOutcome:
        t0 = time.perf_counter()
        if not self.available:
            return SearchOutcome(status="disabled")
        n = max_results or self._s.tavily_max_results
        depth = depth or self._s.tavily_search_depth
        key = (normalize_query(query), n, depth)
        hit = self._cache.get(key)
        if hit is not None:
            return SearchOutcome(hit, "ok", (time.perf_counter() - t0) * 1000, cached=True)

        status = "error"
        if self._client.configured:
            try:
                raw = await asyncio.wait_for(self._client.search(query, n, depth),
                                             self._s.search_timeout_s)
                results = _map_results(raw)
                self._cache.set(key, results)
                return SearchOutcome(results, "ok", (time.perf_counter() - t0) * 1000)
            except (asyncio.TimeoutError, httpx.TimeoutException):
                status = "timeout"
            except Exception as exc:  # noqa: BLE001 — search is optional context
                logger.warning("Tavily error: %s", type(exc).__name__)
                status = "error"

        if self._fallback is not None and self._fallback.configured:
            try:
                raw = await asyncio.wait_for(self._fallback.search(query, n),
                                             self._s.search_timeout_s)
                results = _map_results(raw)
                self._cache.set(key, results)
                logger.info("web search used Serper fallback")
                return SearchOutcome(results, "ok", (time.perf_counter() - t0) * 1000)
            except (asyncio.TimeoutError, httpx.TimeoutException):
                if not self._client.configured:
                    status = "timeout"
            except Exception as exc:  # noqa: BLE001 — search is optional context
                logger.warning("Serper error: %s", type(exc).__name__)
                if not self._client.configured:
                    status = "error"

        return SearchOutcome(status=status, elapsed_ms=(time.perf_counter() - t0) * 1000)

    async def multi_search(self, queries: list[str], max_results_each: int = 3,
                           depth: Optional[str] = None) -> SearchOutcome:
        """Run sub-queries in parallel, dedupe by URL, return best-scored overall."""
        t0 = time.perf_counter()
        queries = list(dict.fromkeys(q for q in queries if q.strip()))[:3]
        if not queries:
            return SearchOutcome(status="ok")
        outcomes = await asyncio.gather(
            *(self.search(q, max_results_each, depth) for q in queries))
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for o in outcomes:
            for r in o.results:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    merged.append(r)
        merged.sort(key=lambda r: r.score, reverse=True)
        if merged:
            status = "ok"
        else:
            status = next((o.status for o in outcomes if o.status != "ok"), "ok")
        return SearchOutcome(merged[: self._s.tavily_max_results], status,
                             (time.perf_counter() - t0) * 1000)

    @staticmethod
    def generate_sub_queries(query: str) -> list[str]:
        """Cheap decomposition for research mode (no LLM call)."""
        q = query.strip()
        queries = [q]
        year = datetime.now(timezone.utc).year
        lowered = q.lower()
        if not any(w in lowered for w in (str(year), str(year - 1), "latest", "recent", "current")):
            queries.append(f"{q} {year}")
        if lowered.startswith(("what is", "what are", "explain", "how does", "how do")):
            queries.append(f"{q} examples overview")
        return queries[:3]


def _map_results(raw: list) -> list[SearchResult]:
    return [
        SearchResult(title=r.get("title", "") or "", url=r.get("url", "") or "",
                     snippet=r.get("content", "") or "", score=float(r.get("score") or 0.0))
        for r in raw
    ]
