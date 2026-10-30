"""
Shared fixtures. Default runs are fully offline: embeddings, reranking, LLM and
web search are replaced with deterministic fakes injected through the
ServiceContainer's `overrides`.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, Optional

import httpx
import numpy as np
import openai
import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.core.container import ServiceContainer
from backend.main import create_app
from backend.services.llm_service import LLMService
from backend.utils.rate_limiter import QuotaGuard

# ── fake embedder / reranker ─────────────────────────────────────────────────

SYNONYMS = {
    "refund": "money_back", "reimbursement": "money_back", "money": "money_back",
    "back": "money_back", "leave": "time_off", "vacation": "time_off", "holiday": "time_off",
    "parental": "time_off", "encryption": "crypto", "encrypted": "crypto", "aes": "crypto",
    "secure": "crypto", "security": "crypto",
}
DIM = 2048
STOP = {"how", "to", "the", "a", "an", "of", "is", "are", "for", "in", "and", "or", "what", "can", "i", "my", "do", "you"}


def _tokens(text: str) -> list[str]:
    return [SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class FakeEmbedder:
    """Hashing bag-of-words embedder with a small synonym table (deterministic)."""

    def __init__(self, model_id: str = "fake:bow-v1", fail: bool = False) -> None:
        self.model_id = model_id
        self.fail = fail
        self.dim = DIM
        self.calls = 0

    @property
    def ready(self) -> bool:
        return not self.fail

    @property
    def error(self) -> Optional[str]:
        return "boom" if self.fail else None

    async def warmup(self) -> bool:
        return not self.fail

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.fail:
            raise RuntimeError("embedder down")
        self.calls += 1
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self.fail:
            raise RuntimeError("embedder down")
        return self._vec(text)

    async def aembed_query(self, text: str) -> np.ndarray:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def status(self) -> dict:
        return {"provider": "fake", "model": self.model_id, "ready": self.ready,
                "dim": DIM, "error": self.error}

    def close(self) -> None:
        pass


class FakeReranker:
    """Scores by query-token overlap (0..1)."""

    def __init__(self, enabled: bool = True, fail: bool = False) -> None:
        self._enabled = enabled
        self.fail = fail
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def ready(self) -> bool:
        return self._enabled

    async def warmup(self) -> bool:
        return self._enabled

    async def arerank(self, query: str, docs: list[str], timeout_s: Optional[float] = None) -> list[float]:
        if self.fail:
            raise RuntimeError("rerank down")
        self.calls += 1
        q = set(_tokens(query))
        return [len(q & set(_tokens(d))) / max(1, len(q)) for d in docs]

    def status(self) -> dict:
        return {"provider": "fake", "model": "fake", "enabled": self._enabled,
                "ready": self._enabled, "error": None}

    def close(self) -> None:
        pass


# ── fake OpenAI-compatible client ────────────────────────────────────────────

def make_api_error(status: int, headers: Optional[dict[str, str]] = None) -> openai.APIStatusError:
    req = httpx.Request("POST", "http://fake/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {}, json={"error": "x"})
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError,
           403: openai.PermissionDeniedError, 404: openai.NotFoundError,
           429: openai.RateLimitError}.get(status)
    if cls is None:
        cls = openai.InternalServerError if status >= 500 else openai.APIStatusError
    return cls("fake error", response=resp, body=None)


def _chunk(text: Optional[str] = None, usage: Optional[dict] = None, reasoning: Optional[str] = None):
    delta = SimpleNamespace(content=text, reasoning=reasoning)
    choices = [SimpleNamespace(delta=delta)] if (text is not None or reasoning is not None) else []
    u = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(choices=choices, usage=u)


class _FakeStream:
    def __init__(self, items: list[Any], delay: float = 0.0, fail_after: Optional[int] = None) -> None:
        self._items = items
        self._i = 0
        self._delay = delay
        self._fail_after = fail_after
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio

        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail_after is not None and self._i >= self._fail_after:
            raise make_api_error(503)
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item

    async def close(self) -> None:
        self.closed = True


class FakeProviderClient:
    """
    Minimal AsyncOpenAI look-alike. `script[model]` is a list of behaviours consumed
    one per call (the last one repeats): "ok", ("ok", text), ("raise", status),
    ("hang",), ("empty",), ("think", text), ("midfail", text), ("chunks", [deltas...]).
    """

    def __init__(self, script: Optional[dict[str, list]] = None,
                 model_ids: Optional[list[str]] = None, headers: Optional[dict] = None) -> None:
        self.script = script or {}
        self.model_ids = model_ids
        self.calls: list[dict[str, Any]] = []
        self.headers = headers or {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self._create)))
        self.models = SimpleNamespace(list=self._list)

    async def _list(self):
        if self.model_ids is None:
            raise openai.APIConnectionError(request=httpx.Request("GET", "http://fake"))
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.model_ids])

    def next_behaviour(self, model: str):
        steps = self.script.get(model, ["ok"])
        n = sum(1 for c in self.calls if c["model"] == model) - 1
        return steps[min(n, len(steps) - 1)]

    async def _create(self, **kwargs):
        import asyncio

        self.calls.append(kwargs)
        model = kwargs["model"]
        step = self.next_behaviour(model)
        if isinstance(step, str):
            step = (step,)
        kind = step[0]
        text = step[1] if len(step) > 1 else f"answer from {model}"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if kind == "raise":
            raise make_api_error(step[1], step[2] if len(step) > 2 else None)
        if kind == "hang":
            await asyncio.sleep(3600)
        if kind == "empty":
            items = [_chunk(None, usage)]
        elif kind == "think":
            items = [_chunk("<think>secret"), _chunk(" reasoning</think>"), _chunk(text), _chunk(None, usage)]
        elif kind == "chunks":              # ("chunks", [exact text deltas...])
            items = [_chunk(c) for c in step[1]] + [_chunk(None, usage)]
        elif kind == "reasoning":
            items = [_chunk(None, reasoning="hmm"), _chunk(text), _chunk(None, usage)]
        else:
            words = text.split(" ")
            items = [_chunk(w + (" " if i < len(words) - 1 else "")) for i, w in enumerate(words)]
            items.append(_chunk(None, usage))
        stream = _FakeStream(items, fail_after=2 if kind == "midfail" else None)
        return SimpleNamespace(headers=self.headers, parse=lambda: stream)


# ── settings / container fixtures ────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY", "COHERE_API_KEY",
                "LLM_MODELS", "LLM_DEFAULT_MODEL", "LLM_FALLBACK_CHAIN", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return Settings(
        _env_file=None, data_dir=str(tmp_path / "data"), groq_api_key="test-groq",
        google_api_key="test-google", tavily_api_key="test-tavily", model_discovery=False,
        llm_first_token_timeout_s=1.0, rerank_timeout_s=1.0, retrieval_timeout_s=3.0,
        allowed_origins="http://localhost:8000", rate_limit_rpm_per_user=1000,
        rag_cosine_floor=0.05, rag_cosine_ceil=0.4,      # scale of the fake bag-of-words embedder
    )


def tavily_transport(results: Optional[list[dict]] = None, status: int = 200,
                     counter: Optional[list] = None,
                     serper_results: Optional[list[dict]] = None) -> httpx.MockTransport:
    results = results if results is not None else [
        {"title": "Example News", "url": "https://example.com/news", "content": "Latest news snippet.", "score": 0.9},
        {"title": "Another Source", "url": "https://example.org/a", "content": "Another snippet.", "score": 0.5},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if counter is not None:
            counter.append(json.loads(request.content))
        if "serper.dev" in url:
            if serper_results is None:
                return httpx.Response(status if status != 200 else 500, json={"error": "nope"})
            return httpx.Response(200, json={"organic": serper_results})
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(200, json={"results": results})

    return httpx.MockTransport(handler)


@pytest.fixture
def fake_llm_client() -> FakeProviderClient:
    return FakeProviderClient()


@pytest.fixture
def build_container(settings):
    """Factory: build_container(llm_client=..., embedder=..., ...) → ServiceContainer."""
    made: list[ServiceContainer] = []

    def _build(*, llm_client: Optional[FakeProviderClient] = None, google_client: Optional[Any] = None,
               embedder: Optional[FakeEmbedder] = None, reranker: Optional[FakeReranker] = None,
               tavily: Optional[httpx.MockTransport] = None, settings_obj: Optional[Settings] = None):
        s = settings_obj or settings
        http = httpx.AsyncClient(transport=tavily or tavily_transport())
        quota = QuotaGuard()
        clients = {"groq": llm_client or FakeProviderClient()}
        if google_client is not None:
            clients["google"] = google_client
        llm = LLMService(s, quota, http, clients=clients)
        c = ServiceContainer.build(s, overrides={
            "http": http, "quota": quota, "llm": llm,
            "embedder": embedder or FakeEmbedder(), "reranker": reranker or FakeReranker(),
        })
        c.db.init()                      # idempotent; tests that skip lifespan still need tables
        made.append(c)
        return c

    yield _build
    for c in made:
        c.store.close()
        c.db.dispose()


@pytest.fixture
def client_factory(build_container):
    """Factory returning a TestClient bound to a freshly built app (lifespan active)."""
    clients: list[TestClient] = []

    def _make(**kwargs) -> tuple[TestClient, ServiceContainer]:
        container = build_container(**kwargs)
        app = create_app(container.settings, container=container)
        tc = TestClient(app)
        tc.__enter__()
        clients.append(tc)
        return tc, container

    yield _make
    for tc in clients:
        tc.__exit__(None, None, None)


def parse_sse(text: str) -> list[dict]:
    """Parse an SSE body into [{'event': name, 'data': obj}, ...]."""
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if data is not None:
            events.append({"event": name, "data": data})
    return events
