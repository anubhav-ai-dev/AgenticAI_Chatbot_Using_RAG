"""Shared pooled HTTP client factory."""

from __future__ import annotations

import httpx

from backend.core.config import Settings


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    """One keep-alive pool reused by every outbound API call (LLM providers, Tavily)."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=settings.llm_connect_timeout_s),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
