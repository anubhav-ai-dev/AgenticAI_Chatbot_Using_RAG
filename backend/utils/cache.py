"""
In-process caches: a generic TTL/LRU cache and a vector-similarity answer cache.
Both are thread-safe (they are touched from the event loop and worker threads).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Hashable, Optional, TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Bounded LRU cache whose entries expire after `ttl_s` seconds."""

    def __init__(self, maxsize: int = 512, ttl_s: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.maxsize = max(1, maxsize)
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: "OrderedDict[K, tuple[float, V]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, value = item
            if expires <= self._clock():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: K, value: V, ttl_s: Optional[float] = None) -> None:
        ttl = self.ttl_s if ttl_s is None else ttl_s
        with self._lock:
            self._data[key] = (self._clock() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def delete(self, key: K) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            now = self._clock()
            for key in [k for k, (exp, _) in self._data.items() if exp <= now]:
                del self._data[key]
            return len(self._data)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self), "hits": self.hits, "misses": self.misses,
            "hit_ratio": round(self.hits / total, 3) if total else 0.0,
        }


@dataclass
class CachedAnswer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    route: str = "direct"


@dataclass
class _Entry:
    vec: np.ndarray
    answer: CachedAnswer
    expires: float


class SemanticCache:
    """
    Returns a stored answer when a new query embedding is within `threshold`
    cosine similarity of a previous one *in the same scope*.

    Scope is an opaque hashable tuple (user, mode, document-index version, …) so
    answers never leak across users or survive a document change.
    """

    def __init__(self, maxsize: int = 300, ttl_s: float = 3600.0, threshold: float = 0.95,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.maxsize = max(1, maxsize)
        self.ttl_s = ttl_s
        self.threshold = threshold
        self._clock = clock
        self._scopes: "OrderedDict[Hashable, list[_Entry]]" = OrderedDict()
        self._lock = threading.Lock()
        self._count = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def lookup(self, scope: Hashable, vec: np.ndarray) -> Optional[CachedAnswer]:
        q = self._unit(vec)
        now = self._clock()
        with self._lock:
            entries = self._scopes.get(scope)
            if not entries:
                self.misses += 1
                return None
            live = [e for e in entries if e.expires > now]
            self._count -= len(entries) - len(live)
            if not live:
                del self._scopes[scope]
                self.misses += 1
                return None
            self._scopes[scope] = live
            sims = np.stack([e.vec for e in live]) @ q
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                self._scopes.move_to_end(scope)
                self.hits += 1
                return live[best].answer
            self.misses += 1
            return None

    def store(self, scope: Hashable, vec: np.ndarray, answer: CachedAnswer) -> None:
        entry = _Entry(self._unit(vec), answer, self._clock() + self.ttl_s)
        with self._lock:
            self._scopes.setdefault(scope, []).append(entry)
            self._scopes.move_to_end(scope)
            self._count += 1
            while self._count > self.maxsize and self._scopes:
                _, evicted = self._scopes.popitem(last=False)   # drop least-recent scope
                self._count -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._scopes.clear()
            self._count = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": self._count, "hits": self.hits, "misses": self.misses,
            "hit_ratio": round(self.hits / total, 3) if total else 0.0,
        }
