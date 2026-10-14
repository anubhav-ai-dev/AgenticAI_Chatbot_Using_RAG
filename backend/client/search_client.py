"""Tavily web-search HTTP client (async, pooled, one retry on transient failures)."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class TavilyClient:
    def __init__(self, api_key: Optional[str], base_url: str, http: httpx.AsyncClient,
                 timeout_s: float = 6.0) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._http = http
        self._timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self._key)

    async def search(self, query: str, max_results: int, depth: str) -> list[dict[str, Any]]:
        """Raw result dicts (title, url, content, score). Raises httpx errors on failure."""
        body = {"query": query, "search_depth": depth, "max_results": max_results,
                "include_answer": False, "include_raw_content": False}
        last: Optional[Exception] = None
        for attempt in range(2):
            try:
                resp = await self._http.post(
                    f"{self._base}/search", headers={"Authorization": f"Bearer {self._key}"},
                    json=body, timeout=self._timeout_s)
                if resp.status_code >= 500 and attempt == 0:
                    last = httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
                    continue                                   # one retry on server errors
                resp.raise_for_status()
                return list(resp.json().get("results", []))
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as exc:                # stale keep-alive / connection reset
                last = exc
                if attempt == 1:
                    raise
        raise last or RuntimeError("search failed")
