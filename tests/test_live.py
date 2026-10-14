"""
Live smoke/quality/latency tests against real providers and the real local models.
Run:  pytest -m live -s        (needs .env keys; downloads models on first run)
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import numpy as np
import pytest
import pytest_asyncio

from backend.core.config import BASE_DIR, Settings
from backend.core.container import ServiceContainer
from backend.models.api import SessionCreate
from backend.services.chat_service import ChatCommand

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="module")]

CORPUS = {
    "refunds.txt": "Refund policy. Enterprise customers may request a full refund within 30 days of purchase. "
                   "After 30 days refunds are prorated over the remaining subscription term. Starter plan "
                   "customers are only eligible for a refund within 14 days of purchase.",
    "support.txt": "Support hours. Our support team is available Monday through Friday from 9am to 6pm Pacific "
                   "Time. Premium customers can reach a 24/7 emergency hotline at extension 4411 for outages.",
    "security.txt": "Security overview. All customer data is encrypted at rest with AES-256 and in transit with "
                    "TLS 1.3. We complete SOC 2 Type II audits annually and run quarterly third-party "
                    "penetration tests.",
    "leave.txt": "Parental leave. Employees receive 16 weeks of paid parental leave which can be taken in one "
                 "block or split across the first year after birth or adoption.",
    "expenses.txt": "Expense reports. Submit expenses within 30 days through the Concur portal. Meals while "
                    "travelling are reimbursed up to 75 USD per day and hotels up to 250 USD per night.",
}
# (query, expected source or None for 'must not retrieve')
QUERIES = [
    ("what is the enterprise refund window", "refunds.txt"),             # exact terms
    ("can I get my money back after a month?", "refunds.txt"),           # vague paraphrase
    ("who do I call at night if the system is down", "support.txt"),
    ("is customer data encrypted", "security.txt"),
    ("what encryption standard do you use", "security.txt"),
    ("how much time off do new parents get", "leave.txt"),
    ("how much can I claim for a hotel on a business trip", "expenses.txt"),
    ("AES-256", "security.txt"),                                         # rare token
    ("extension 4411", "support.txt"),
    ("how to bake sourdough bread", None),
    ("who won the football world cup in 2018", None),
    ("what is the capital of australia", None),
]


@pytest.fixture(scope="module")
def live_settings(tmp_path_factory) -> Settings:
    data = tmp_path_factory.mktemp("live")
    (data / "models").symlink_to(BASE_DIR / "data" / "models")           # reuse downloaded ONNX models
    s = Settings(data_dir=str(data))
    if not (s.groq_api_key or s.google_api_key):
        pytest.skip("no LLM provider keys in .env")
    return s


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_container(live_settings):
    c = ServiceContainer.build(live_settings)
    await c.startup(warm=False)
    await asyncio.gather(c.embedder.warmup(), c.reranker.warmup(), c.llm.discover())
    yield c
    await c.shutdown()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def corpus_loaded(live_container):
    for name, text in CORPUS.items():
        await live_container.documents.ingest_file("live", name, text.encode())
    return live_container


async def test_every_registered_model_streams_or_reports_why(live_container):
    """Matrix of the configured free models. Groq must work; Google entries are informational."""
    c = live_container
    rows, ok_by_provider = [], {}
    for spec in c.settings.model_specs():
        available, reason = c.llm.availability(spec.id)
        if not available:
            rows.append((spec.id, "unavailable", reason, None))
            continue
        try:
            res = await asyncio.wait_for(
                c.llm.complete([{"role": "system", "content": "Be terse."},
                                {"role": "user", "content": "Reply with exactly: PONG"}],
                               model=spec.id, max_tokens=64), 40)
            actual = res.model
            status = "ok" if actual == spec.id else f"fell back to {actual}"
            if actual == spec.id:
                ok_by_provider[spec.provider] = ok_by_provider.get(spec.provider, 0) + 1
            rows.append((spec.id, status, f"ttft={res.ttft_ms:.0f}ms", res.text.strip()[:30]))
        except Exception as exc:  # noqa: BLE001
            rows.append((spec.id, "error", type(exc).__name__, None))
        c.llm.breaker.reset()                       # keep each probe independent
    print("\n  MODEL MATRIX")
    for r in rows:
        print("   ", r)
    assert ok_by_provider.get("groq", 0) >= 1


async def test_invalid_model_id_falls_back_without_error(live_settings):
    specs = live_settings.model_specs()
    bogus = {"id": "openai/definitely-not-a-model", "label": "Bogus", "provider": "groq"}
    s = live_settings.model_copy(update={
        "llm_models": json.dumps([bogus] + [m.model_dump() for m in specs]),
        "llm_default_model": bogus["id"],
        "llm_fallback_chain": ",".join([bogus["id"]] + live_settings.fallback_chain),
        "model_discovery": False,
    })
    c = ServiceContainer.build(s)
    try:
        res = await c.llm.complete([{"role": "user", "content": "Say hi in two words"}], max_tokens=64)
        assert res.text.strip() and res.model != bogus["id"] and bogus["id"] in res.fell_back_from
        assert c.llm.availability(bogus["id"])[0] is False         # hard-disabled after the 404
    finally:
        await c.shutdown()


async def test_retrieval_quality_hybrid_beats_single_signals(corpus_loaded):
    c = corpus_loaded
    hits = {"hybrid": 0, "dense": 0, "lexical": 0}
    negatives_ok = 0
    n_pos = sum(1 for _, e in QUERIES if e)
    detail = []
    for q, expected in QUERIES:
        qv = await c.embedder.aembed_query(q)
        dense = c.store.dense_search("live", qv, 3)
        lex = c.store.lexical_search("live", q, 3)
        res = await c.retrieval.search("live", q, top_k=3)
        src = lambda ids: c.store.get_chunks([ids[0]])[ids[0]].source if ids else None  # noqa: E731
        d1 = src([i for i, _ in dense])
        l1 = src([i for i, _ in lex])
        h1 = res.items[0].source if res.items else None
        if expected:
            hits["hybrid"] += h1 == expected
            hits["dense"] += d1 == expected
            hits["lexical"] += l1 == expected
        else:
            negatives_ok += not res.used_rag
        detail.append((q[:46], expected, h1, round(res.items[0].relevance, 2) if res.items else None))
    print(f"\n  RETRIEVAL hits@1 of {n_pos}: {hits}   negatives rejected: {negatives_ok}/{len(QUERIES) - n_pos}")
    for row in detail:
        print("   ", row)
    assert hits["hybrid"] >= max(hits["dense"], hits["lexical"])
    assert hits["hybrid"] >= n_pos - 1
    assert negatives_ok == len(QUERIES) - n_pos


async def test_end_to_end_chat_latency_report(corpus_loaded):
    c = corpus_loaded
    session = c.sessions.create_session(SessionCreate(user_id="live"))
    report = {}

    async def run(label, message, **kw):
        t0 = time.perf_counter()
        first = None
        text, done = [], {}
        async for ev in c.chat.stream(ChatCommand(session_id=session.id, message=message, **kw)):
            if ev["event"] == "token":
                first = first or (time.perf_counter() - t0) * 1000
                text.append(ev["data"]["text"])
            elif ev["event"] == "done":
                done = ev["data"]
            elif ev["event"] == "error":
                pytest.fail(f"{label}: {ev['data']}")
        report[label] = {"first_token_ms": round(first or -1), "total_ms": round((time.perf_counter() - t0) * 1000),
                         "route": done.get("route"), "model": done.get("model"), "cached": done.get("cached")}
        return "".join(text), done

    # use a fresh user so all documents belong to "live" session user
    await run("direct", "Explain what a hash map is in two sentences", mode="general", web_search=False)
    answer, _ = await run("rag", "What is the enterprise refund window?", mode="general", web_search=False)
    assert "30" in answer
    await run("rag_cached", "What is the enterprise refund window?", mode="general", web_search=False)
    await run("web", "What is the latest news about Python releases today?", mode="general", web_search=True)
    print("\n  LATENCY REPORT")
    for k, v in report.items():
        print(f"    {k:11s} {v}")
    assert report["rag_cached"]["cached"] is True and report["rag_cached"]["first_token_ms"] < 500


async def test_retrieval_latency_distribution(live_container):
    """p50/p95 of retrieval over a synthetic 1k-chunk corpus with real embedder + reranker."""
    c = live_container
    rng = np.random.default_rng(7)
    vocab = [f"word{i}" for i in range(4000)]
    texts = [" ".join(rng.choice(vocab, 60)) + f" marker{i}" for i in range(1000)]
    t0 = time.perf_counter()
    vecs = c.embedder.embed_documents(texts)
    embed_s = time.perf_counter() - t0
    from backend.services.index_store import ChunkIn

    c.store.add_chunks("bench", "bench-doc", "bench.txt", [ChunkIn(t, page=i) for i, t in enumerate(texts)],
                       vecs, c.embedder.model_id)
    lat = []
    for i in range(40):
        q = f"{' '.join(rng.choice(vocab, 6))} marker{rng.integers(0, 1000)}"
        t = time.perf_counter()
        await c.retrieval.search("bench", q + f" #{i}")
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    p50, p95 = statistics.median(lat), lat[int(len(lat) * 0.95) - 1]
    print(f"\n  RETRIEVAL on 1000 chunks: embed corpus {embed_s:.1f}s | p50={p50:.0f}ms p95={p95:.0f}ms")
    assert p95 < 600


async def test_tavily_basic_vs_advanced_latency(live_container):
    c = live_container
    if not c.search.available:
        pytest.skip("no TAVILY_API_KEY")
    out = {}
    for depth in ("basic", "advanced"):
        o = await c.search.search(f"python 3.13 release notes {depth}", depth=depth)
        out[depth] = (o.status, round(o.elapsed_ms))
        assert o.status == "ok" and o.results
    print("\n  TAVILY (status, ms):", out)
