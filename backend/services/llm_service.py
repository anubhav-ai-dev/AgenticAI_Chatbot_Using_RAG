"""
LLM policy layer for free-tier chat models (Groq + Google Gemini/Gemma).

* Model registry comes from config; discovery marks IDs the key cannot use.
* `open_stream()` fails over across the configured chain *before the first token*:
  timeouts, 429/5xx, 404 (model gone) and empty replies move to the next model.
* Circuit breaker + quota guard keep known-bad / exhausted models out of the path.
* Transport (SDK, headers, error mapping) lives in `backend.client.llm_client`;
  upstream error text is never surfaced — callers get normalised `LLMError` codes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import httpx

from backend.client.llm_client import (
    ProviderClient,
    build_provider_clients,
    chunk_text,
    chunk_usage,
    classify_exception,
    close_stream,
    wrap_clients,
)
from backend.core.config import ModelSpec, Settings
from backend.core.exceptions import LLMError
from backend.utils.rate_limiter import QuotaGuard
from backend.utils.streaming import ThinkFilter
from backend.utils.text import estimate_tokens

logger = logging.getLogger(__name__)


# ── streaming wrapper ────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text: str
    model: str
    fell_back_from: list[str] = field(default_factory=list)
    ttft_ms: float = 0.0
    usage: Optional[dict[str, int]] = None


class LLMStream:
    """Async-iterable of visible text deltas from an already-started completion."""

    def __init__(self, *, model: str, fell_back_from: list[str], ttft_ms: float,
                 upstream: Any, iterator: Any, first_text: str, think: ThinkFilter,
                 usage: Optional[dict[str, int]], deadline: float,
                 on_finish: Callable[["LLMStream", bool], None]) -> None:
        self.model = model
        self.fell_back_from = fell_back_from
        self.ttft_ms = ttft_ms
        self.usage = usage
        self._upstream = upstream
        self._it = iterator
        self._first = first_text
        self._think = think
        self._deadline = deadline
        self._on_finish = on_finish

    async def __aiter__(self) -> AsyncIterator[str]:
        ok = False
        try:
            if self._first:
                yield self._first
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMError("timeout", model=self.model)
                try:
                    chunk = await asyncio.wait_for(self._it.__anext__(), remaining)
                except StopAsyncIteration:
                    break
                usage = chunk_usage(chunk)
                if usage is not None:
                    self.usage = usage
                piece = chunk_text(chunk)
                if piece:
                    visible = self._think.feed(piece)
                    if visible:
                        yield visible
            tail = self._think.flush()
            if tail:
                yield tail
            ok = True
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalise mid-stream failures
            raise classify_exception(exc, self.model) from exc
        finally:
            await self.aclose()
            self._on_finish(self, ok)

    async def aclose(self) -> None:
        await close_stream(self._upstream)


# ── circuit breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """Per-model cool-downs. Soft trips are overridable as a last resort; hard are not."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._until: dict[str, tuple[float, str, bool]] = {}
        self._failures: dict[str, int] = {}

    def trip(self, model: str, cooldown_s: float, reason: str, hard: bool = False) -> None:
        self._until[model] = (self._clock() + cooldown_s, reason, hard)

    def state(self, model: str) -> Optional[tuple[str, bool]]:
        entry = self._until.get(model)
        if entry is None:
            return None
        until, reason, hard = entry
        if until <= self._clock():
            del self._until[model]
            return None
        return reason, hard

    def failure(self, model: str) -> int:
        self._failures[model] = self._failures.get(model, 0) + 1
        return self._failures[model]

    def success(self, model: str) -> None:
        self._failures.pop(model, None)
        entry = self._until.get(model)
        if entry and not entry[2]:
            del self._until[model]

    def reset(self) -> None:
        self._until.clear()
        self._failures.clear()


# ── service ──────────────────────────────────────────────────────────────────

class LLMService:
    def __init__(self, settings: Settings, quota: QuotaGuard, http: httpx.AsyncClient,
                 clients: Optional[dict[str, Any]] = None) -> None:
        self._s = settings
        self._quota = quota
        self._specs: dict[str, ModelSpec] = {m.id: m for m in settings.model_specs()}
        self._providers = settings.provider_configs()
        self._breaker = CircuitBreaker()
        self._discovered: dict[str, set[str]] = {}
        self._discovered_at: float = 0.0
        self._clients: dict[str, ProviderClient] = (
            wrap_clients(clients) if clients is not None else build_provider_clients(settings, http))

    # ── registry ─────────────────────────────────────────────────────────────

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def configured(self) -> bool:
        return bool(self._clients)

    def spec(self, model_id: str) -> Optional[ModelSpec]:
        return self._specs.get(model_id)

    def has_model(self, model_id: str) -> bool:
        spec = self._specs.get(model_id)
        return spec is not None and spec.enabled

    def availability(self, model_id: str) -> tuple[bool, Optional[str]]:
        spec = self._specs.get(model_id)
        if spec is None or not spec.enabled:
            return False, "not in registry"
        if spec.provider not in self._clients:
            return False, f"{spec.provider} API key not set"
        known = self._discovered.get(spec.provider)
        if known and model_id not in known:
            return False, "not offered by the provider for this key"
        state = self._breaker.state(model_id)
        if state and state[1]:
            return False, state[0]
        return True, None

    def public_models(self) -> list[dict[str, Any]]:
        out = []
        for spec in self._specs.values():
            if not spec.enabled:
                continue
            ok, reason = self.availability(spec.id)
            out.append({"id": spec.id, "label": spec.label, "provider": spec.provider,
                        "description": spec.description, "available": ok, "reason": reason,
                        "default": spec.id == self._s.llm_default_model})
        return out

    async def discover(self) -> dict[str, int]:
        """Ask each configured provider which model IDs the key can see."""
        if not self._s.model_discovery:
            return {}

        async def one(name: str, client: ProviderClient) -> tuple[str, Optional[set[str]]]:
            try:
                return name, await client.list_model_ids(8.0)
            except Exception as exc:  # noqa: BLE001 — unknown ⇒ assume available
                logger.warning("Model discovery failed for %s (%s)", name, type(exc).__name__)
                return name, None

        results = await asyncio.gather(*(one(n, c) for n, c in self._clients.items()))
        counts: dict[str, int] = {}
        for name, ids in results:
            if ids:
                self._discovered[name] = ids
                counts[name] = len(ids)
        self._discovered_at = time.time()
        return counts

    # ── chain selection ──────────────────────────────────────────────────────

    def chain(self, requested: Optional[str] = None, fast: bool = False) -> list[ModelSpec]:
        order: list[str] = []
        first = self._s.llm_fast_model if fast else (requested or self._s.llm_default_model)
        for mid in [first, *self._s.fallback_chain]:
            if mid in self._specs and mid not in order:
                order.append(mid)
        return [self._specs[m] for m in order if self._specs[m].enabled]

    # ── streaming ────────────────────────────────────────────────────────────

    async def open_stream(self, messages: list[dict[str, str]], *, model: Optional[str] = None,
                          fast: bool = False, max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          extra: Optional[dict[str, Any]] = None) -> LLMStream:
        if not self._clients:
            raise LLMError("no_provider", retryable=False)
        max_tokens = max_tokens or self._s.llm_max_tokens
        est = sum(estimate_tokens(m.get("content", "")) + 4 for m in messages) + max_tokens
        chain = self.chain(model, fast)

        attempts: list[dict[str, str]] = []
        failed: list[str] = []
        last: Optional[LLMError] = None

        # Pass 1 honours soft cool-downs and the quota guard. Pass 2 (only reached when
        # pass 1 attempted nothing) ignores them: a possibly-throttled model beats no answer.
        for honor_soft in (True, False):
            for spec in chain:
                if spec.id in failed:
                    continue
                ok, reason = self.availability(spec.id)
                if not ok:
                    if honor_soft:
                        attempts.append({"model": spec.id, "skipped": reason or "unavailable"})
                    continue
                if honor_soft:
                    soft = self._breaker.state(spec.id)
                    if soft:
                        attempts.append({"model": spec.id, "skipped": soft[0]})
                        continue
                    if not self._quota.allows(spec.id, est, spec.rpm, spec.tpm):
                        attempts.append({"model": spec.id, "skipped": "quota guard"})
                        continue
                try:
                    return await self._open_one(spec, messages, max_tokens, temperature, extra, failed)
                except LLMError as err:
                    last = err
                    failed.append(spec.id)
                    attempts.append({"model": spec.id, "error": err.code})
                    self._apply_breaker(spec, err)
                    logger.warning("Model %s failed before first token: %s", spec.id, err.code)
            if last is not None:
                break                                      # something was attempted and failed

        raise LLMError("all_models_failed", attempts=attempts,
                       retry_after=last.retry_after if last else None)

    async def _open_one(self, spec: ModelSpec, messages: list[dict[str, str]], max_tokens: int,
                        temperature: Optional[float], extra: Optional[dict[str, Any]],
                        failed: list[str]) -> LLMStream:
        client = self._clients[spec.provider]
        params = {**spec.extra, **(extra or {})}
        temp = temperature if temperature is not None else params.pop("temperature", None)
        params.pop("temperature", None)
        if temp is None:
            temp = self._s.llm_temperature

        t0 = time.monotonic()
        first_deadline = t0 + self._s.llm_first_token_timeout_s
        total_deadline = t0 + self._s.llm_total_timeout_s
        upstream: Any = None
        try:
            opened = await client.open_stream(
                model=spec.id, messages=self._prepare_messages(spec, messages),
                max_tokens=max_tokens, temperature=temp, extra_body=params or None,
                timeout_s=self._s.llm_first_token_timeout_s)
            upstream, iterator = opened.upstream, opened.iterator
            self._quota.observe_headers(spec.id, opened.headers)
            think = ThinkFilter()
            first = ""
            usage: Optional[dict[str, int]] = None
            exhausted = False
            while not first:
                remaining = first_deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMError("timeout", model=spec.id)
                try:
                    chunk = await asyncio.wait_for(iterator.__anext__(), remaining)
                except StopAsyncIteration:
                    exhausted = True
                    break
                usage = chunk_usage(chunk) or usage
                piece = chunk_text(chunk)
                if piece:
                    first += think.feed(piece)
            if exhausted:
                first += think.flush()
            if not first.strip():
                raise LLMError("empty_response", model=spec.id)
        except BaseException as exc:
            if upstream is not None:
                await close_stream(upstream)
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            raise classify_exception(exc, spec.id) from exc

        ttft_ms = (time.monotonic() - t0) * 1000
        return LLMStream(
            model=spec.id, fell_back_from=list(failed), ttft_ms=ttft_ms, upstream=upstream,
            iterator=iterator, first_text=first.lstrip(),
            think=think, usage=usage, deadline=total_deadline, on_finish=self._on_stream_finish)

    def _on_stream_finish(self, stream: LLMStream, ok: bool) -> None:
        tokens = (stream.usage or {}).get("total_tokens", 0)
        if tokens:
            self._quota.record(stream.model, tokens)
        if ok:
            self._breaker.success(stream.model)

    def _apply_breaker(self, spec: ModelSpec, err: LLMError) -> None:
        s = self._s
        if err.code == "rate_limited":
            cooldown = min(max(err.retry_after or s.breaker_cooldown_s, 5.0), 300.0)
            self._breaker.trip(spec.id, cooldown, f"rate limited ({int(cooldown)}s)")
        elif err.code in ("model_not_found", "forbidden"):
            self._breaker.trip(spec.id, s.model_unavailable_cooldown_s,
                               "model not available for this API key", hard=True)
        elif err.code == "auth":
            for other in self._specs.values():
                if other.provider == spec.provider:
                    self._breaker.trip(other.id, s.model_unavailable_cooldown_s,
                                       "provider rejected the API key", hard=True)
        elif err.code == "bad_request":
            return
        elif self._breaker.failure(spec.id) >= 2:
            self._breaker.trip(spec.id, s.breaker_cooldown_s, f"recent failures ({err.code})")

    @staticmethod
    def _prepare_messages(spec: ModelSpec, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if spec.supports_system:
            return messages
        system = [m["content"] for m in messages if m["role"] == "system"]
        rest = [dict(m) for m in messages if m["role"] != "system"]
        if system:
            preface = "Follow these instructions:\n" + "\n\n".join(system) + "\n\n---\n"
            for m in rest:
                if m["role"] == "user":
                    m["content"] = preface + m["content"]
                    break
            else:
                rest.insert(0, {"role": "user", "content": preface})
        return rest

    # ── convenience ──────────────────────────────────────────────────────────

    async def complete(self, messages: list[dict[str, str]], *, model: Optional[str] = None,
                       fast: bool = False, max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> LLMResult:
        stream = await self.open_stream(messages, model=model, fast=fast,
                                        max_tokens=max_tokens, temperature=temperature)
        parts: list[str] = []
        async for piece in stream:
            parts.append(piece)
        return LLMResult("".join(parts), stream.model, stream.fell_back_from,
                         stream.ttft_ms, stream.usage)

    def status(self) -> dict[str, Any]:
        return {
            "providers": {n: {"configured": n in self._clients,
                              "discovered_models": len(self._discovered.get(n, ()))}
                          for n in self._providers},
            "models": {s.id: {"available": self.availability(s.id)[0],
                              "tripped": (self._breaker.state(s.id) or (None,))[0]}
                       for s in self._specs.values()},
            "quota": self._quota.snapshot(),
        }
