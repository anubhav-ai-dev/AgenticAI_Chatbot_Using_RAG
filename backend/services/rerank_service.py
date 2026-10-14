"""
Rerank service: lazy backend lifecycle, time-boxed async calls and status.

Backends (local ONNX cross-encoder | Cohere) live in `backend.client.rerank_client`;
scores are relevance probabilities in [0, 1] aligned with the input order.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

from backend.client.rerank_client import RerankClient, build_rerank_client
from backend.core.config import Settings

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self, settings: Settings,
                 client_factory: Optional[Callable[[Settings], RerankClient]] = None) -> None:
        self._settings = settings
        self._factory = client_factory or build_rerank_client
        self._backend: Optional[RerankClient] = None
        self._error: Optional[str] = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        s = self._settings
        return s.rerank_enabled and s.rerank_provider != "none" and self._error is None

    @property
    def ready(self) -> bool:
        return self._loaded and self._error is None

    @property
    def provider(self) -> str:
        return self._settings.rerank_provider

    def _load(self) -> None:
        if self._loaded or self._error is not None:
            return
        with self._lock:
            if self._loaded or self._error is not None:
                return
            s = self._settings
            try:
                self._backend = self._factory(s)
                self._loaded = True
                logger.info("Reranker ready: %s:%s", s.rerank_provider, s.rerank_model_id)
            except Exception as exc:  # noqa: BLE001 — retrieval falls back to fused order
                self._error = f"{type(exc).__name__}: {exc}"
                logger.warning("Reranker unavailable (%s) — using fused order", self._error)

    async def warmup(self) -> bool:
        if self._settings.rerank_provider == "none" or not self._settings.rerank_enabled:
            return False
        await asyncio.to_thread(self._load)
        return self.ready

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Blocking. Raises RuntimeError when unavailable."""
        if not docs:
            return []
        self._load()
        if not self.ready or self._backend is None:
            raise RuntimeError(self._error or "reranker disabled")
        return self._backend.rerank(query, docs)

    async def arerank(self, query: str, docs: list[str], timeout_s: Optional[float] = None) -> list[float]:
        timeout = timeout_s if timeout_s is not None else self._settings.rerank_timeout_s
        return await asyncio.wait_for(asyncio.to_thread(self.rerank, query, docs), timeout)

    def close(self) -> None:
        closer = getattr(self._backend, "close", None)
        if callable(closer):
            closer()

    def status(self) -> dict[str, Any]:
        s = self._settings
        return {"provider": s.rerank_provider, "model": s.rerank_model_id,
                "enabled": self.enabled, "ready": self.ready, "error": self._error}
