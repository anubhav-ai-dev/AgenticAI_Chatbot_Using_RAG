"""LLMService: failover before first token, circuit breaker, quota guard, output hygiene."""

from __future__ import annotations

import httpx
import pytest

from backend.client.llm_client import classify_exception
from backend.core.exceptions import LLMError
from backend.services.llm_service import LLMService
from backend.utils.rate_limiter import QuotaGuard
from tests.conftest import FakeProviderClient, make_api_error

MSGS = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]


def make_llm(settings, groq=None, google=None, **overrides):
    s = settings.model_copy(update=overrides) if overrides else settings
    clients = {"groq": groq or FakeProviderClient()}
    if google is not None:
        clients["google"] = google
    return LLMService(s, QuotaGuard(), httpx.AsyncClient(), clients=clients)


async def collect(stream):
    return "".join([p async for p in stream])


async def test_happy_path_streams_and_reports(settings):
    llm = make_llm(settings)
    stream = await llm.open_stream(MSGS, model="openai/gpt-oss-120b")
    text = await collect(stream)
    assert text == "answer from openai/gpt-oss-120b"
    assert stream.model == "openai/gpt-oss-120b" and stream.fell_back_from == []
    assert stream.usage["total_tokens"] == 15 and stream.ttft_ms >= 0


async def test_model_params_come_from_registry(settings):
    groq = FakeProviderClient()
    llm = make_llm(settings, groq)
    await collect(await llm.open_stream(MSGS, model="openai/gpt-oss-120b"))
    call = groq.calls[0]
    assert call["stream"] is True and call["extra_body"] == {"reasoning_effort": "low"}
    assert call["stream_options"] == {"include_usage": True}
    assert call["temperature"] == settings.llm_temperature and call["max_tokens"] == settings.llm_max_tokens


async def test_failover_on_rate_limit_and_5xx_and_timeout(settings):
    groq = FakeProviderClient({
        "openai/gpt-oss-120b": [("raise", 429, {"retry-after": "7"})],
        "openai/gpt-oss-20b": [("raise", 503)],
        "qwen/qwen3.8-27b": ["ok"],
    })
    llm = make_llm(settings, groq)
    stream = await llm.open_stream(MSGS, model="openai/gpt-oss-120b")
    assert await collect(stream) == "answer from qwen/qwen3.8-27b"
    assert stream.fell_back_from == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    # 429 trips the breaker for ~retry-after; next request skips it without calling upstream
    before = len(groq.calls)
    stream2 = await llm.open_stream(MSGS, model="openai/gpt-oss-120b")
    await collect(stream2)
    assert groq.calls[before]["model"] != "openai/gpt-oss-120b"        # tripped model not re-tried


async def test_hang_hits_first_token_timeout_then_falls_back(settings):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("hang",)]})
    llm = make_llm(settings, groq, llm_first_token_timeout_s=0.2)
    stream = await llm.open_stream(MSGS)
    assert (await collect(stream)).startswith("answer from openai/gpt-oss-20b")
    assert stream.fell_back_from == ["openai/gpt-oss-120b"]


async def test_empty_response_triggers_fallback(settings):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("empty",)]})
    stream = await make_llm(settings, groq).open_stream(MSGS)
    assert (await collect(stream)).startswith("answer from openai/gpt-oss-20b")


async def test_404_hard_disables_model_and_hides_it(settings):
    google = FakeProviderClient({"gemini-2.5-flash-lite": [("raise", 404)]})
    llm = make_llm(settings, FakeProviderClient({"openai/gpt-oss-120b": [("raise", 503)],
                                                 "openai/gpt-oss-20b": [("raise", 503)],
                                                 "qwen/qwen3.8-27b": [("raise", 503)]}), google,
                   llm_fallback_chain="gemini-2.5-flash-lite,gemini-3.5-flash-lite")
    stream = await llm.open_stream(MSGS, model="gemini-2.5-flash-lite")
    assert (await collect(stream)) == "answer from gemini-3.5-flash-lite"
    ok, reason = llm.availability("gemini-2.5-flash-lite")
    assert not ok and "not available" in reason
    info = {m["id"]: m for m in llm.public_models()}
    assert info["gemini-2.5-flash-lite"]["available"] is False
    assert info["gemini-3.5-flash-lite"]["available"] is True
    assert sum(1 for c in google.calls if c["model"] == "gemini-2.5-flash-lite") == 1   # never retried


async def test_all_models_failing_raises_normalised_error(settings):
    script = {m: [("raise", 500)] for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")}
    google = FakeProviderClient({m: [("raise", 500)] for m in ("gemini-3.5-flash-lite", "gemini-3.7-flash")})
    llm = make_llm(settings, FakeProviderClient(script), google)
    with pytest.raises(LLMError) as ei:
        await llm.open_stream(MSGS)
    err = ei.value
    assert err.code == "all_models_failed"
    assert "500" not in err.message and "fake error" not in err.message        # no upstream text leaks
    assert len(err.attempts) >= 5


async def test_no_provider_configured(settings):
    llm = LLMService(settings, QuotaGuard(), httpx.AsyncClient(), clients={})
    with pytest.raises(LLMError) as ei:
        await llm.open_stream(MSGS)
    assert ei.value.code == "no_provider" and not llm.configured


async def test_mid_stream_failure_is_normalised(settings):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("midfail", "one two three four five six")]})
    llm = make_llm(settings, groq)
    stream = await llm.open_stream(MSGS, model="openai/gpt-oss-120b")
    got = []
    with pytest.raises(LLMError) as ei:
        async for piece in stream:
            got.append(piece)
    assert ei.value.code == "unavailable" and "".join(got)       # partial text was delivered


async def test_think_blocks_and_reasoning_deltas_never_reach_user(settings):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("think", "final answer")],
                               "openai/gpt-oss-20b": [("reasoning", "clean answer")]})
    llm = make_llm(settings, groq)
    assert (await collect(await llm.open_stream(MSGS, model="openai/gpt-oss-120b"))).strip() == "final answer"
    assert (await collect(await llm.open_stream(MSGS, model="openai/gpt-oss-20b"))).strip() == "clean answer"


async def test_gemma_gets_system_prompt_merged(settings):
    google = FakeProviderClient()
    llm = make_llm(settings, None, google, llm_fallback_chain="gemma-4-31b-it")
    specs = {s.id: s for s in settings.model_specs()}
    assert specs["gemma-4-31b-it"].supports_system is True            # default (Gemma 4 supports it)
    llm._specs["gemma-4-31b-it"] = specs["gemma-4-31b-it"].model_copy(update={"supports_system": False})
    await collect(await llm.open_stream(MSGS, model="gemma-4-31b-it"))
    sent = google.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user"] and "be terse" in sent[0]["content"] and "hi" in sent[0]["content"]


async def test_quota_guard_skips_exhausted_model_proactively(settings):
    groq = FakeProviderClient()
    llm = make_llm(settings, groq)
    llm._quota.record("openai/gpt-oss-120b", 8000)                  # tpm budget fully used
    stream = await llm.open_stream(MSGS, model="openai/gpt-oss-120b")
    await collect(stream)
    assert groq.calls[0]["model"] == "openai/gpt-oss-20b" and stream.fell_back_from == []


async def test_quota_guard_last_resort_when_everything_is_throttled(settings):
    groq = FakeProviderClient()
    llm = make_llm(settings, groq, llm_fallback_chain="openai/gpt-oss-120b")
    llm._quota.record("openai/gpt-oss-120b", 8000)
    assert (await collect(await llm.open_stream(MSGS, model="openai/gpt-oss-120b"))).startswith("answer from")


async def test_headers_feed_quota_guard(settings):
    groq = FakeProviderClient(headers={"x-ratelimit-remaining-tokens": "5", "x-ratelimit-reset-tokens": "30s"})
    llm = make_llm(settings, groq)
    await collect(await llm.open_stream(MSGS, model="openai/gpt-oss-120b"))
    assert not llm._quota.allows("openai/gpt-oss-120b", 500, None, None)


async def test_discovery_marks_unavailable_ids_and_survives_errors(settings):
    groq = FakeProviderClient(model_ids=["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    google = FakeProviderClient(model_ids=None)                   # discovery raises → assume available
    llm = make_llm(settings, groq, google, model_discovery=True)
    counts = await llm.discover()
    assert counts == {"groq": 2}
    assert llm.availability("qwen/qwen3.8-27b")[0] is False
    assert llm.availability("openai/gpt-oss-120b")[0] is True
    assert llm.availability("gemini-3.7-flash")[0] is True


async def test_unknown_model_falls_back_to_default_chain(settings):
    llm = make_llm(settings)
    assert [s.id for s in llm.chain("nope/none")][0] == "openai/gpt-oss-120b"
    assert llm.has_model("nope/none") is False


async def test_fast_model_chain_and_complete(settings):
    groq = FakeProviderClient()
    llm = make_llm(settings, groq)
    res = await llm.complete(MSGS, fast=True, max_tokens=50, temperature=0.0)
    assert res.model == "openai/gpt-oss-20b" and groq.calls[0]["temperature"] == 0.0


def test_classify_exception_codes():
    assert classify_exception(make_api_error(429, {"retry-after": "3"}), "m").retry_after == 3.0
    assert classify_exception(make_api_error(401), "m").code == "auth"
    assert classify_exception(make_api_error(403), "m").code == "forbidden"
    assert classify_exception(make_api_error(404), "m").code == "model_not_found"
    assert classify_exception(make_api_error(400), "m").code == "bad_request"
    assert classify_exception(make_api_error(502), "m").code == "unavailable"
    assert classify_exception(httpx.ReadTimeout("x"), "m").code == "timeout"
    assert classify_exception(ValueError("x"), "m").code == "unknown"
