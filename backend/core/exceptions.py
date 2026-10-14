"""Shared exception types.

Every user-facing failure derives from `AppError` and carries a message that is safe to
show to clients (never upstream provider text).
"""

from __future__ import annotations

from typing import Optional


class AppError(Exception):
    """Base class for expected, user-presentable failures."""


# ── documents ────────────────────────────────────────────────────────────────

class DocumentParseError(AppError, ValueError):
    """Unreadable / unsupported / unsafe document input."""


class FetchError(DocumentParseError):
    """A web page could not be fetched (bad URL, blocked address, HTTP error, too large)."""


class IngestError(AppError):
    """Ingestion failure with an HTTP status hint."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ── sessions / retrieval ─────────────────────────────────────────────────────

class SessionNotFound(AppError, LookupError):
    """Session or message id does not exist."""


class EmbeddingUnavailable(AppError, RuntimeError):
    """No embedding backend can serve the request."""


# ── LLM ──────────────────────────────────────────────────────────────────────

_LLM_MESSAGES = {
    "rate_limited": "The model is rate limited right now.",
    "unavailable": "The model service is temporarily unavailable.",
    "timeout": "The model took too long to respond.",
    "network": "Could not reach the model service.",
    "model_not_found": "This model is not available for the configured API key.",
    "forbidden": "This model is not permitted for the configured API key.",
    "auth": "The model provider rejected the API key.",
    "bad_request": "The model rejected the request.",
    "empty_response": "The model returned an empty response.",
    "no_provider": "No LLM provider is configured (set GROQ_API_KEY or GOOGLE_API_KEY).",
    "all_models_failed": "All configured models failed. Please try again shortly.",
    "unknown": "The model request failed.",
}


class LLMError(AppError):
    """Normalised LLM failure: `code` is machine-readable, `message` is user-safe."""

    def __init__(self, code: str, *, model: Optional[str] = None, status: Optional[int] = None,
                 retry_after: Optional[float] = None, retryable: bool = True,
                 attempts: Optional[list[dict[str, str]]] = None) -> None:
        self.code = code
        self.model = model
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable
        self.attempts = attempts or []
        self.message = _LLM_MESSAGES.get(code, _LLM_MESSAGES["unknown"])
        super().__init__(f"{code}: {self.message}")
