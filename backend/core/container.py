"""
Composition root: builds every service once, wires their dependencies, and
exposes FastAPI `Depends` getters.

Services receive collaborators via constructor injection; nothing else in the
code base constructs services or reads the environment. Tests/MCP can pass
`overrides={"embedder": FakeEmbedder(), ...}` to swap any component.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx
from fastapi import Depends, Request

from backend.client.http import build_http_client
from backend.client.search_client import TavilyClient
from backend.client.web_client import WebPageClient
from backend.core.config import Settings
from backend.core.database import Database
from backend.services.chat_service import ChatService, QueryRouter
from backend.services.document_service import DocumentService
from backend.services.embedding_service import EmbeddingService
from backend.services.index_store import HybridIndexStore
from backend.services.llm_service import LLMService
from backend.services.parser import DocumentParser
from backend.services.prompt_builder import PromptBuilder
from backend.services.rerank_service import RerankService
from backend.services.retrieval_service import RetrievalService
from backend.services.search_service import WebSearchService
from backend.services.session_service import SessionService
from backend.utils.cache import SemanticCache
from backend.utils.rate_limiter import QuotaGuard, RateLimiter

logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(self, settings: Settings, overrides: Optional[dict[str, Any]] = None) -> None:
        o = overrides or {}
        self.settings = settings

        self.http: httpx.AsyncClient = o.get("http") or build_http_client(settings)
        self.db: Database = o.get("db") or Database(settings.db_url)

        self.embedder: EmbeddingService = o.get("embedder") or EmbeddingService(settings)
        self.reranker: RerankService = o.get("reranker") or RerankService(settings)
        self.store: HybridIndexStore = o.get("store") or HybridIndexStore(settings.index_dir)
        self.retrieval: RetrievalService = o.get("retrieval") or RetrievalService(
            settings, self.store, self.embedder, self.reranker)
        self.parser: DocumentParser = o.get("parser") or DocumentParser(
            WebPageClient(settings.allow_private_urls, settings.url_fetch_timeout_s))
        self.documents: DocumentService = o.get("documents") or DocumentService(
            settings, self.db, self.parser, self.embedder, self.store)

        self.quota: QuotaGuard = o.get("quota") or QuotaGuard()
        self.llm: LLMService = o.get("llm") or LLMService(settings, self.quota, self.http)
        self.search: WebSearchService = o.get("search") or WebSearchService(
            settings, TavilyClient(settings.tavily_api_key, settings.tavily_base_url,
                                   self.http, settings.search_timeout_s))
        self.prompts: PromptBuilder = o.get("prompts") or PromptBuilder(settings)
        self.sessions: SessionService = o.get("sessions") or SessionService(settings, self.db)
        self.semantic_cache: SemanticCache = o.get("semantic_cache") or SemanticCache(
            settings.semantic_cache_size, settings.semantic_cache_ttl_s,
            settings.semantic_cache_threshold)
        self.rate_limiter: RateLimiter = o.get("rate_limiter") or RateLimiter(
            settings.rate_limit_rpm_per_user)
        self.chat: ChatService = o.get("chat") or ChatService(
            settings, self.llm, self.retrieval, self.search, self.prompts,
            self.sessions, self.semantic_cache, QueryRouter())

        self._background: list[asyncio.Task] = []

    @classmethod
    def build(cls, settings: Settings, overrides: Optional[dict[str, Any]] = None) -> "ServiceContainer":
        return cls(settings, overrides)

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def startup(self, warm: bool = True) -> None:
        self.db.init()
        for warning in self.settings.validate_runtime():
            logger.warning("Config: %s", warning)
        logger.info("Effective config: %s", self.settings.masked())
        if warm:
            self._background.append(asyncio.create_task(self._warm(), name="warmup"))

    async def _warm(self) -> None:
        """Non-blocking warm-up: model downloads/loads, discovery, pending reindex."""
        async def safe(label: str, coro: Any) -> None:
            try:
                await coro
            except Exception:  # noqa: BLE001 — warm-up must never take the app down
                logger.exception("Warm-up step failed: %s", label)

        await asyncio.gather(
            safe("embeddings", self.embedder.warmup()),
            safe("reranker", self.reranker.warmup()),
            safe("llm-discovery", self.llm.discover()),
        )
        await safe("reindex", self.documents.reindex_if_needed())
        logger.info("Warm-up complete")

    async def shutdown(self) -> None:
        for task in self._background:
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        await self.http.aclose()
        self.embedder.close()
        self.reranker.close()
        self.store.close()
        self.db.dispose()

    # ── health ───────────────────────────────────────────────────────────────

    def health(self) -> tuple[str, dict[str, str], dict[str, Any]]:
        services: dict[str, str] = {}
        details: dict[str, Any] = {}

        try:
            self.db.ping()
            services["database"] = "connected"
        except Exception as exc:  # noqa: BLE001
            services["database"] = f"error: {type(exc).__name__}"

        providers = self.settings.provider_configs()
        for name, cfg in providers.items():
            services[name] = "configured" if cfg.configured else "missing_key"
        services["tavily"] = "configured" if self.search.available else "missing_key"

        emb = self.embedder.status()
        rer = self.reranker.status()
        services["embeddings"] = ("ready" if emb["ready"] else
                                  ("error" if emb["error"] else "loading"))
        services["reranker"] = ("disabled" if not rer["enabled"] and not rer["error"] else
                                ("ready" if rer["ready"] else ("error" if rer["error"] else "loading")))
        details.update(embeddings=emb, reranker=rer, llm=self.llm.status(),
                       index=self.store.stats(), caches={
                           **self.retrieval.stats(), "semantic": self.semantic_cache.stats()})

        if services["database"] != "connected":
            status = "unavailable"
        elif not self.llm.configured or services["embeddings"] == "error":
            status = "degraded"
        else:
            status = "healthy"
        return status, services, details


# ── FastAPI dependency getters ────────────────────────────────────────────────

def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_chat_service(c: ServiceContainer = Depends(get_container)) -> ChatService:
    return c.chat


def get_session_service(c: ServiceContainer = Depends(get_container)) -> SessionService:
    return c.sessions


def get_document_service(c: ServiceContainer = Depends(get_container)) -> DocumentService:
    return c.documents


def get_retrieval_service(c: ServiceContainer = Depends(get_container)) -> RetrievalService:
    return c.retrieval


def get_search_service(c: ServiceContainer = Depends(get_container)) -> WebSearchService:
    return c.search


def get_llm_service(c: ServiceContainer = Depends(get_container)) -> LLMService:
    return c.llm
