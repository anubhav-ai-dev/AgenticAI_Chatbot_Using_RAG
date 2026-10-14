"""
Request throttling (per user) and proactive model-quota tracking (per model).

Groq's free tier enforces separate tokens-per-minute buckets per model, so the
quota guard lets the LLM service skip a model *before* it would 429.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Mapping, Optional, Tuple

from backend.utils.timeutil import parse_duration


class RateLimiter:
    """Token bucket per key. `allow()` returns (allowed, retry_after_seconds)."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 10_000) -> None:
        self.capacity = float(max(1, rpm))
        self.refill_per_s = self.capacity / 60.0
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: Dict[str, Tuple[float, float]] = {}   # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> Tuple[bool, float]:
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                allowed, retry = True, 0.0
            else:
                self._buckets[key] = (tokens, now)
                allowed, retry = False, (1.0 - tokens) / self.refill_per_s
            if len(self._buckets) > self._max_keys:
                self._prune(now)
            return allowed, retry

    def _prune(self, now: float) -> None:
        full_after = 60.0
        stale = [k for k, (_, ts) in self._buckets.items() if now - ts > full_after]
        for k in stale:
            del self._buckets[k]


class QuotaGuard:
    """Rolling 60-second request/token accounting per model + server-reported limits."""

    WINDOW_S = 60.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: Dict[str, Deque[Tuple[float, int]]] = {}
        self._reported: Dict[str, Tuple[int, float]] = {}   # model -> (remaining_tokens, reset_at)
        self._lock = threading.Lock()

    def _trim(self, model: str, now: float) -> Deque[Tuple[float, int]]:
        q = self._events.setdefault(model, deque())
        while q and now - q[0][0] > self.WINDOW_S:
            q.popleft()
        return q

    def allows(self, model: str, est_tokens: int, rpm: Optional[int], tpm: Optional[int]) -> bool:
        now = self._clock()
        with self._lock:
            q = self._trim(model, now)
            if rpm is not None and len(q) + 1 > rpm:
                return False
            if tpm is not None and sum(t for _, t in q) + est_tokens > tpm:
                return False
            reported = self._reported.get(model)
            if reported and now < reported[1] and est_tokens > reported[0]:
                return False
            return True

    def record(self, model: str, tokens: int) -> None:
        now = self._clock()
        with self._lock:
            self._trim(model, now).append((now, max(0, tokens)))

    def observe_headers(self, model: str, headers: Mapping[str, str]) -> None:
        """Learn remaining tokens from x-ratelimit-* response headers (Groq)."""
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if remaining is None or reset is None:
            return
        try:
            rem = int(float(remaining))
        except ValueError:
            return
        with self._lock:
            self._reported[model] = (rem, self._clock() + reset)

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        now = self._clock()
        with self._lock:
            out: Dict[str, Dict[str, int]] = {}
            for model in list(self._events):
                q = self._trim(model, now)
                out[model] = {"requests_60s": len(q), "tokens_60s": sum(t for _, t in q)}
            return out
