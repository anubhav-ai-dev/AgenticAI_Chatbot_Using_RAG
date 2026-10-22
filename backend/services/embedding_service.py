"""
Embedding service: lazy backend lifecycle, failure memo, async wrappers and status.

The backends themselves (local ONNX | Google | Cohere) live in
`backend.client.embedding_client`; every vector is L2-normalised float32.
Blocking methods are meant to run in worker threads (`aembed_*` wrappers do so).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from backend.client.embedding_client import EmbeddingClient, build_embedding_client
from backend.core.config import Settings
from backend.core.exceptions import EmbeddingUnavailable

logger = logging.getLogger(__name__)

RETRY_AFTER_S = 60.0     # a failed model load is retried after this long, not remembered forever


class EmbeddingService:
    def __init__(self, settings: Settings,
                 client_factory: Optional[Callable[[Settings], EmbeddingClient]] = None) -> None:
        self._settings = settings
        self._factory = client_factory or build_embedding_client
        self._backend: Optional[EmbeddingClient] = None
        self._error: Optional[str] = None
        self._failed_at = 0.0
        self._lock = threading.Lock()
        self._dim: Optional[int] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return f"{self._settings.embed_provider}:{self._settings.embed_model_id}"

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def ready(self) -> bool:
        return self._backend is not None

    def _recently_failed(self) -> bool:
        return self._error is not None and time.monotonic() - self._failed_at < RETRY_AFTER_S

    def _load(self) -> Optional[EmbeddingClient]:
        """Instantiate the backend; failures are remembered for RETRY_AFTER_S, never raised."""
        if self._backend is not None or self._recently_failed():
            return self._backend
        with self._lock:
            if self._backend is not None or self._recently_failed():
                return self._backend
            try:
                backend = self._factory(self._settings)
                self._dim = int(backend.embed_query("warmup").shape[-1])
                self._backend = backend
                self._error = None
                logger.info("Embeddings ready: %s (dim=%s)", self.model_id, self._dim)
            except Exception as exc:  # noqa: BLE001 — degrade to lexical-only retrieval
                self._error = f"{type(exc).__name__}: {exc}"
                self._failed_at = time.monotonic()
                logger.warning("Embeddings unavailable (%s) — lexical-only retrieval", self._error)
            return self._backend

    async def warmup(self) -> bool:
        return (await asyncio.to_thread(self._load)) is not None

    def close(self) -> None:
        closer = getattr(self._backend, "close", None)
        if callable(closer):
            closer()

    # ── embedding API ────────────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        backend = self._load()
        if backend is None:
            raise EmbeddingUnavailable(self._error or "embedding backend not available")
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        return backend.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        backend = self._load()
        if backend is None:
            raise EmbeddingUnavailable(self._error or "embedding backend not available")
        return backend.embed_query(text)

    async def aembed_query(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed_documents, texts)

    def status(self) -> dict[str, Any]:
        return {"provider": self._settings.embed_provider, "model": self._settings.embed_model_id,
                "ready": self.ready, "dim": self._dim, "error": self._error}
