"""
Embedding backends: local ONNX (fastembed), Google Gemini, Cohere.

All clients are synchronous (the service runs them in worker threads) and return
L2-normalised float32 matrices so inner product equals cosine.
"""

from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np

from backend.core.config import Settings
from backend.core.exceptions import EmbeddingUnavailable
from backend.utils.numeric import l2_normalize


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbeddingClient:
    """fastembed (ONNX Runtime) — free, offline after the first model download."""

    def __init__(self, model_id: str, cache_dir: str, batch_size: int) -> None:
        from fastembed import TextEmbedding  # lazy: heavy import

        self._model = TextEmbedding(model_name=model_id, cache_dir=cache_dir)
        self._batch = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # similar lengths per batch → less padding; restore input order afterwards
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = np.array(list(self._model.embed([texts[i] for i in order], batch_size=self._batch)))
        restored = np.empty_like(vectors)
        restored[order] = vectors
        return l2_normalize(restored)

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(np.array(list(self._model.query_embed(text))))[0]


class _HttpEmbeddingClient:
    def __init__(self, timeout_s: float = 30.0) -> None:
        self._http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._http.close()


class GoogleEmbeddingClient(_HttpEmbeddingClient):
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model_id: str) -> None:
        super().__init__()
        self._key = api_key
        self._model = model_id.removeprefix("models/")

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            body = {"requests": [
                {"model": f"models/{self._model}", "content": {"parts": [{"text": t}]},
                 "taskType": task}
                for t in texts[i:i + 100]
            ]}
            r = self._http.post(f"{self.BASE}/models/{self._model}:batchEmbedContents",
                                headers={"x-goog-api-key": self._key}, json=body)
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return l2_normalize(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class CohereEmbeddingClient(_HttpEmbeddingClient):
    URL = "https://api.cohere.com/v2/embed"

    def __init__(self, api_key: str, model_id: str) -> None:
        super().__init__()
        self._key = api_key
        self._model = model_id

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):                     # Cohere caps a request at 96 texts
            r = self._http.post(
                self.URL, headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self._model, "texts": texts[i:i + 96],
                      "input_type": input_type, "embedding_types": ["float"]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"]["float"])
        return l2_normalize(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "search_query")[0]


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Instantiate the configured backend; raises EmbeddingUnavailable if misconfigured."""
    model = settings.embed_model_id
    if settings.embed_provider == "local":
        return LocalEmbeddingClient(model, str(settings.models_dir), settings.embed_batch_size)
    if settings.embed_provider == "google":
        if not settings.google_api_key:
            raise EmbeddingUnavailable("GOOGLE_API_KEY not set")
        return GoogleEmbeddingClient(settings.google_api_key, model)
    if not settings.cohere_api_key:
        raise EmbeddingUnavailable("COHERE_API_KEY not set")
    return CohereEmbeddingClient(settings.cohere_api_key, model)
