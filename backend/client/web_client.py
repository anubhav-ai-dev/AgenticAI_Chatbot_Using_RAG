"""Safe web-page fetcher for URL ingestion (SSRF-guarded, redirect-aware, size-capped)."""

from __future__ import annotations

from urllib.parse import urljoin

import httpx

from backend.core.exceptions import FetchError
from backend.utils.net import UnsafeURLError, assert_public_url

MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
_USER_AGENT = "Mozilla/5.0 (compatible; AIAssistantPro/4.0)"


class WebPageClient:
    def __init__(self, allow_private_urls: bool = False, timeout_s: float = 15.0) -> None:
        self.allow_private_urls = allow_private_urls
        self.timeout_s = timeout_s

    def fetch_html(self, url: str) -> str:
        """Blocking. Every redirect hop is re-validated against the SSRF guard."""
        current = url
        with httpx.Client(timeout=self.timeout_s, follow_redirects=False,
                          headers={"User-Agent": _USER_AGENT}) as client:
            for _ in range(MAX_REDIRECTS + 1):
                try:
                    assert_public_url(current, self.allow_private_urls)
                except UnsafeURLError as exc:
                    raise FetchError(str(exc)) from exc
                try:
                    with client.stream("GET", current) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                raise FetchError("Redirect without a Location header.")
                            current = urljoin(current, location)
                            continue
                        if resp.status_code >= 400:
                            raise FetchError(f"The URL returned HTTP {resp.status_code}.")
                        ctype = resp.headers.get("content-type", "").lower()
                        if ctype and not any(t in ctype for t in ("html", "text", "xml", "json")):
                            raise FetchError(f"Unsupported content type '{ctype.split(';')[0]}'.")
                        body = bytearray()
                        for part in resp.iter_bytes():
                            body.extend(part)
                            if len(body) > MAX_BYTES:
                                raise FetchError("The page is too large to ingest.")
                        return body.decode(resp.encoding or "utf-8", errors="replace")
                except httpx.HTTPError as exc:
                    raise FetchError(f"Could not fetch the URL ({type(exc).__name__}).") from exc
        raise FetchError("Too many redirects.")
