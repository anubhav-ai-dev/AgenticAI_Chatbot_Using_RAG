"""
Transport adapter for OpenAI-compatible chat endpoints (Groq, Google Gemini/Gemma).

Owns the SDK: opening streams (with response headers), listing model ids, chunk
accessors and mapping SDK/network exceptions to normalised `LLMError` codes.
Fallback, breaker and quota policy live in `backend.services.llm_service`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx
import openai
from openai import AsyncOpenAI

from backend.core.config import Settings
from backend.core.exceptions import LLMError
from backend.utils.timeutil import parse_duration


@dataclass
class OpenedStream:
    """A started completion: response headers plus the SDK stream and its iterator."""

    headers: Mapping[str, str]
    upstream: Any
    iterator: Any


class ProviderClient:
    """One provider endpoint behind a pooled SDK client."""

    def __init__(self, name: str, sdk: Any) -> None:
        self.name = name
        self._sdk = sdk

    async def open_stream(self, *, model: str, messages: list[dict[str, str]], max_tokens: int,
                          temperature: float, extra_body: Optional[dict[str, Any]],
                          timeout_s: float) -> OpenedStream:
        kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        raw = await asyncio.wait_for(
            self._sdk.chat.completions.with_raw_response.create(**kwargs), timeout_s)
        upstream = raw.parse()
        return OpenedStream(raw.headers, upstream, upstream.__aiter__())

    async def list_model_ids(self, timeout_s: float = 8.0) -> set[str]:
        page = await asyncio.wait_for(self._sdk.models.list(), timeout_s)
        return {m.id.removeprefix("models/") for m in page.data}


def build_provider_clients(settings: Settings, http: httpx.AsyncClient) -> dict[str, ProviderClient]:
    """One client per provider whose API key is configured."""
    return {
        name: ProviderClient(name, AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                                               http_client=http, max_retries=0))
        for name, cfg in settings.provider_configs().items() if cfg.configured
    }


def wrap_clients(clients: Mapping[str, Any]) -> dict[str, ProviderClient]:
    """Accept SDK-shaped clients (e.g. test doubles) alongside ready-made ProviderClients."""
    return {n: c if isinstance(c, ProviderClient) else ProviderClient(n, c) for n, c in clients.items()}


# ── stream chunk accessors ────────────────────────────────────────────────────

def chunk_text(chunk: Any) -> Optional[str]:
    """Visible content delta of a chunk (reasoning deltas are deliberately ignored)."""
    if not getattr(chunk, "choices", None):
        return None
    return chunk.choices[0].delta.content or None


def chunk_usage(chunk: Any) -> Optional[dict[str, int]]:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


async def close_stream(upstream: Any) -> None:
    """Best-effort close of an SDK stream (idempotent)."""
    closer = getattr(upstream, "close", None)
    if closer is None:
        return
    try:
        res = closer()
        if asyncio.iscoroutine(res):
            await res
    except Exception:  # noqa: BLE001
        pass


# ── error normalisation ───────────────────────────────────────────────────────

def classify_exception(exc: BaseException, model: str) -> LLMError:
    """Map SDK/network exceptions to a normalised LLMError."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        headers = getattr(exc.response, "headers", {}) or {}
        retry_after = parse_duration(headers.get("retry-after")) or \
            parse_duration(headers.get("x-ratelimit-reset-tokens")) or \
            parse_duration(headers.get("x-ratelimit-reset-requests"))
        if status == 429:
            return LLMError("rate_limited", model=model, status=status, retry_after=retry_after)
        if status == 404:
            return LLMError("model_not_found", model=model, status=status, retryable=False)
        if status == 401:
            return LLMError("auth", model=model, status=status, retryable=False)
        if status == 403:
            return LLMError("forbidden", model=model, status=status, retryable=False)
        if status in (400, 413, 422):
            return LLMError("bad_request", model=model, status=status, retryable=False)
        return LLMError("unavailable", model=model, status=status, retry_after=retry_after)
    if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError, httpx.TimeoutException)):
        return LLMError("timeout", model=model)
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return LLMError("network", model=model)
    return LLMError("unknown", model=model)
