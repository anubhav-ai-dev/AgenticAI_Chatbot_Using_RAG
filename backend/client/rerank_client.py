"""
Cross-encoder rerank backends (local ONNX | Cohere). `rerank()` returns one relevance
score in [0, 1] per input document, aligned with input order.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from backend.core.config import Settings
from backend.utils.numeric import sigmoid


class RerankClient(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class LocalRerankClient:
    def __init__(self, model_id: str, cache_dir: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: heavy import

        self._model = TextCrossEncoder(model_name=model_id, cache_dir=cache_dir)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [sigmoid(float(x)) for x in self._model.rerank(query, docs)]


class CohereRerankClient:
    URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: str, model_id: str, timeout_s: float) -> None:
        self._key = api_key
        self._model = model_id
        self._http = httpx.Client(timeout=timeout_s)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        r = self._http.post(
            self.URL, headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "query": query, "documents": docs, "top_n": len(docs)})
        r.raise_for_status()
        scores = [0.0] * len(docs)
        for item in r.json().get("results", []):
            scores[item["index"]] = float(item["relevance_score"])
        return scores

    def close(self) -> None:
        self._http.close()


def build_rerank_client(settings: Settings) -> RerankClient:
    """Instantiate the configured backend; raises RuntimeError if misconfigured."""
    if settings.rerank_provider == "local":
        return LocalRerankClient(settings.rerank_model_id, str(settings.models_dir))
    if settings.rerank_provider == "cohere":
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY not set")
        return CohereRerankClient(settings.cohere_api_key, settings.rerank_model_id,
                                  settings.rerank_timeout_s)
    raise RuntimeError("reranking disabled")
