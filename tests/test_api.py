"""HTTP API + chat pipeline tests (offline). Exercises every endpoint and the SSE contract."""

from __future__ import annotations

import io

import pytest

from backend.models.api import SessionCreate
from backend.services.chat_service import ChatCommand
from tests.conftest import FakeProviderClient, parse_sse, tavily_transport

DOC = (
    "Refund policy: Enterprise customers may request a full refund within 30 days of purchase.\n\n"
    "Support hours: Monday through Friday from 9am to 6pm Pacific Time.\n\n"
    + "Padding sentence about office life. " * 30
)


def new_session(tc, **kw):
    body = {"user_id": "u1", "title": "New Chat", "mode": "general", **kw}
    r = tc.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def chat(tc, sid, message, **kw):
    payload = {"session_id": sid, "message": message, "user_id": "u1", "mode": "general",
               "web_search": True, **kw}
    r = tc.post("/chat/stream", json=payload)
    return r, parse_sse(r.text) if r.status_code == 200 else []


def names(events):
    return [e["event"] for e in events]


def text_of(events):
    return "".join(e["data"]["text"] for e in events if e["event"] == "token")


# ── system ───────────────────────────────────────────────────────────────────

def test_ping_health_models_and_spa(client_factory):
    tc, _ = client_factory()
    assert tc.get("/ping").json() == {"pong": True}
    h = tc.get("/health").json()
    assert h["status"] == "healthy" and h["services"]["database"] == "connected"
    assert h["services"]["groq"] == "configured" and h["services"]["embeddings"] == "ready"
    m = tc.get("/models").json()
    ids = {x["id"] for x in m["models"]}
    assert m["default"] == "openai/gpt-oss-120b" and "gemini-3.7-flash" in ids
    assert all({"id", "label", "provider", "available"} <= set(x) for x in m["models"])
    assert "AI Assistant Pro" in tc.get("/").text
    assert tc.get("/js/app.js").status_code == 200 and tc.get("/css/main.css").status_code == 200


def test_health_degraded_without_llm_keys(settings, build_container):
    from fastapi.testclient import TestClient
    from backend.main import create_app
    from backend.services.llm_service import LLMService
    from backend.utils.rate_limiter import QuotaGuard
    import httpx

    s = settings.model_copy(update={"groq_api_key": None, "google_api_key": None})
    llm = LLMService(s, QuotaGuard(), httpx.AsyncClient(), clients={})
    c = build_container(settings_obj=s)
    c.llm = llm
    c.chat._llm = llm
    with TestClient(create_app(s, container=c)) as tc:
        assert tc.get("/health").json()["status"] == "degraded"
        sid = new_session(tc)["id"]
        r = tc.post("/chat/stream", json={"session_id": sid, "message": "hi"})
        assert r.status_code == 503 and "No LLM provider" in r.json()["detail"]
        assert tc.post("/chat", json={"session_id": sid, "message": "hi"}).status_code == 503


# ── sessions & messages ──────────────────────────────────────────────────────

def test_session_crud_and_cascade(client_factory):
    tc, _ = client_factory()
    s = new_session(tc, model="openai/gpt-oss-20b", mode="code")
    assert s["model"] == "openai/gpt-oss-20b" and s["mode"] == "code" and s["messages"] == []
    default = new_session(tc)
    assert default["model"] == "openai/gpt-oss-120b"                  # server default, not client-hardcoded
    assert [x["id"] for x in tc.get("/sessions", params={"user_id": "u1"}).json()][0] in {s["id"], default["id"]}
    assert tc.get("/sessions", params={"user_id": "nobody"}).json() == []

    patched = tc.patch(f"/sessions/{s['id']}", json={"title": "Renamed", "mode": "research"}).json()
    assert patched["title"] == "Renamed" and patched["mode"] == "research"
    assert tc.patch("/sessions/missing", json={"title": "x"}).status_code == 404

    m = tc.post(f"/sessions/{s['id']}/messages", json={"role": "user", "content": "hello world there friend now ok yes"})
    assert m.status_code == 201
    assert tc.post(f"/sessions/{s['id']}/messages", json={"role": "hacker", "content": "x"}).status_code == 422
    assert tc.post(f"/sessions/{s['id']}/messages", json={"role": "user", "content": ""}).status_code == 422
    assert tc.post("/sessions/missing/messages", json={"role": "user", "content": "x"}).status_code == 404

    full = tc.get(f"/sessions/{s['id']}").json()
    assert len(full["messages"]) == 1
    assert tc.delete(f"/sessions/{s['id']}").status_code == 204
    assert tc.get(f"/sessions/{s['id']}").status_code == 404
    assert tc.delete(f"/sessions/{s['id']}").status_code == 404


def test_session_title_auto_generated_and_ordering(client_factory):
    tc, _ = client_factory()
    a = new_session(tc)
    new_session(tc)                                                    # a newer, idle session
    tc.post(f"/sessions/{a['id']}/messages", json={"role": "user", "content": "one two three four five six seven eight"})
    sessions = tc.get("/sessions", params={"user_id": "u1"}).json()
    assert sessions[0]["id"] == a["id"]                                # activity bumps updated_at
    assert sessions[0]["title"] == "one two three four five six..."


# ── documents ────────────────────────────────────────────────────────────────

def test_upload_list_retrieve_delete_flow(client_factory):
    tc, c = client_factory()
    up = tc.post("/upload", params={"user_id": "u1"}, files={"file": ("policy.txt", io.BytesIO(DOC.encode()))})
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["filename"] == "policy.txt" and body["chunk_count"] >= 2 and body["indexing"] == "hybrid"
    docs = tc.get("/documents", params={"user_id": "u1"}).json()
    assert [d["id"] for d in docs] == [body["document_id"]] and docs[0]["chunk_count"] == body["chunk_count"]

    r = tc.post("/retrieve", json={"query": "enterprise refund 30 days", "user_id": "u1", "top_k": 3,
                                   "threshold": 0.2}).json()
    assert r["used_rag"] and r["chunks"][0]["source"] == "policy.txt" and "Refund" in r["chunks"][0]["snippet"]
    assert set(r) >= {"chunks", "used_rag", "query", "mode", "reranked", "elapsed_ms"}

    assert tc.delete(f"/documents/{body['document_id']}", params={"user_id": "someone-else"}).status_code == 404
    assert tc.delete(f"/documents/{body['document_id']}", params={"user_id": "u1"}).status_code == 204
    assert tc.get("/documents", params={"user_id": "u1"}).json() == []
    assert not tc.post("/retrieve", json={"query": "refund", "user_id": "u1"}).json()["used_rag"]


@pytest.mark.parametrize("name,payload,status", [
    ("empty.txt", b"", 400),
    ("evil.exe", b"MZ....", 415),
    ("blank.txt", b"   \n\n  ", 422),
])
def test_upload_rejections(client_factory, name, payload, status):
    tc, _ = client_factory()
    r = tc.post("/upload", params={"user_id": "u1"}, files={"file": (name, io.BytesIO(payload))})
    assert r.status_code == status and "detail" in r.json()


def test_upload_requires_file_or_url_and_blocks_ssrf(client_factory):
    tc, _ = client_factory()
    assert tc.post("/upload", params={"user_id": "u1"}).status_code == 400
    for url in ("http://127.0.0.1:8000/health", "http://169.254.169.254/", "file:///etc/passwd"):
        r = tc.post("/upload", params={"user_id": "u1", "url": url})
        assert r.status_code == 422, (url, r.text)


def test_upload_oversize_returns_413(client_factory, settings):
    tc, _ = client_factory(settings_obj=settings.model_copy(update={"max_upload_mb": 1}))
    big = b"word " * (400 * 1024)
    r = tc.post("/upload", params={"user_id": "u1"}, files={"file": ("big.txt", io.BytesIO(big))})
    assert r.status_code == 413


def test_upload_csv_and_markdown(client_factory):
    tc, _ = client_factory()
    csv_data = "city,population\n" + "\n".join(f"city{i},{i * 1000}" for i in range(80))
    assert tc.post("/upload", params={"user_id": "u1"},
                   files={"file": ("cities.csv", io.BytesIO(csv_data.encode()))}).status_code == 200
    md = "# Title\n\nSome markdown content about gardening tools. " * 20
    r = tc.post("/upload", params={"user_id": "u1"}, files={"file": ("notes.md", io.BytesIO(md.encode()))})
    assert r.status_code == 200 and r.json()["file_type"] == "md"
    types = {d["file_type"] for d in tc.get("/documents", params={"user_id": "u1"}).json()}
    assert types == {"csv", "md"}


def test_user_id_validation(client_factory):
    tc, _ = client_factory()
    assert tc.get("/sessions", params={"user_id": ""}).status_code == 422
    assert tc.get("/documents", params={"user_id": "x" * 500}).status_code == 422


# ── web search ───────────────────────────────────────────────────────────────

def test_search_endpoint_ok_disabled_and_error(client_factory, settings):
    tc, _ = client_factory()
    r = tc.post("/search", json={"query": "news", "max_results": 2}).json()
    assert r["status"] == "ok" and r["results"][0]["url"] == "https://example.com/news"

    tc2, _ = client_factory(settings_obj=settings.model_copy(update={"tavily_api_key": None}))
    assert tc2.post("/search", json={"query": "news"}).json() == {"results": [], "query": "news", "status": "disabled"}

    tc3, _ = client_factory(tavily=tavily_transport(status=500))
    assert tc3.post("/search", json={"query": "news"}).json()["status"] == "error"
    assert tc.post("/search", json={"query": ""}).status_code == 422


def test_search_falls_back_to_serper_when_tavily_fails(client_factory, settings):
    organic = [{"title": "Serper Hit", "link": "https://example.com/serper", "snippet": "Fallback snippet."}]
    tc, _ = client_factory(
        settings_obj=settings.model_copy(update={"serper_api_key": "test-serper"}),
        tavily=tavily_transport(status=500, serper_results=organic),
    )
    r = tc.post("/search", json={"query": "news", "max_results": 2}).json()
    assert r["status"] == "ok" and r["results"][0]["url"] == "https://example.com/serper"


def test_search_uses_serper_when_tavily_is_unconfigured(client_factory, settings):
    organic = [{"title": "Serper Only", "link": "https://example.com/s", "snippet": "s"}]
    tc, _ = client_factory(
        settings_obj=settings.model_copy(update={"tavily_api_key": None, "serper_api_key": "test-serper"}),
        tavily=tavily_transport(status=500, serper_results=organic),
    )
    r = tc.post("/search", json={"query": "news"}).json()
    assert r["status"] == "ok" and r["results"][0]["title"] == "Serper Only"


# ── chat pipeline ────────────────────────────────────────────────────────────

def test_chat_direct_route_streams_and_persists(client_factory):
    tc, c = client_factory()
    sid = new_session(tc)["id"]
    r, ev = chat(tc, sid, "Explain recursion simply please")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert names(ev)[0] == "meta" and names(ev)[-1] == "done"
    assert ev[0]["data"]["route"] == "direct"
    assert text_of(ev).strip() == "answer from openai/gpt-oss-120b"
    done = ev[-1]["data"]
    assert done["persisted"] and done["model"] == "openai/gpt-oss-120b" and not done["cached"]
    assert {"ttft_ms", "total_ms"} <= set(done["timings"])
    msgs = tc.get(f"/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]            # exactly one user turn (no duplicate)
    assert msgs[1]["meta"]["route"] == "direct" and msgs[1]["id"] == done["message_id"]


def test_chat_rag_route_uses_documents_and_returns_citations(client_factory):
    groq = FakeProviderClient()
    tc, _ = client_factory(llm_client=groq)
    tc.post("/upload", params={"user_id": "u1"}, files={"file": ("policy.txt", io.BytesIO(DOC.encode()))})
    sid = new_session(tc)["id"]
    r, ev = chat(tc, sid, "What is the enterprise refund window within 30 days?", web_search=False)
    assert ev[0]["data"]["route"] == "rag"
    cits = next(e["data"]["items"] for e in ev if e["event"] == "citations")
    assert cits[0]["type"] == "doc" and cits[0]["source"] == "policy.txt" and cits[0]["id"] == 1
    system = groq.calls[0]["messages"][0]["content"]
    assert "Refund policy" in system and "policy.txt" in system and "[1]" in system
    stages = [(e["data"]["stage"], e["data"]["state"]) for e in ev if e["event"] == "status"]
    assert ("retrieving", "start") in stages and ("retrieving", "done") in stages
    saved = tc.get(f"/sessions/{sid}").json()["messages"][-1]
    assert saved["citations"][0]["source"] == "policy.txt"                # citations persisted


def test_chat_web_route_and_graceful_search_failure(client_factory, settings):
    groq = FakeProviderClient()
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc)["id"]
    r, ev = chat(tc, sid, "What is the latest news today?")
    assert ev[0]["data"]["route"] == "web"
    cits = next(e["data"]["items"] for e in ev if e["event"] == "citations")
    assert cits[0]["type"] == "web" and cits[0]["url"] == "https://example.com/news"
    assert "Example News" in groq.calls[0]["messages"][0]["content"]

    groq2 = FakeProviderClient()
    tc2, _ = client_factory(llm_client=groq2, tavily=tavily_transport(status=500))
    sid2 = new_session(tc2)["id"]
    _, ev2 = chat(tc2, sid2, "latest news today")
    assert names(ev2)[-1] == "done" and "citations" not in names(ev2)    # answered anyway
    status = next(e["data"] for e in ev2 if e["event"] == "status" and e["data"].get("stage") == "searching"
                  and e["data"]["state"] == "done")
    assert status["error"] == "error"
    assert "web search was unavailable" in groq2.calls[0]["messages"][0]["content"].lower()

    groq3 = FakeProviderClient()
    organic = [{"title": "Serper Hit", "link": "https://example.com/serper", "snippet": "Fallback snippet."}]
    tc3, _ = client_factory(
        llm_client=groq3,
        settings_obj=settings.model_copy(update={"serper_api_key": "test-serper"}),
        tavily=tavily_transport(status=500, serper_results=organic),
    )
    sid3 = new_session(tc3)["id"]
    _, ev3 = chat(tc3, sid3, "latest news today")
    cits3 = next(e["data"]["items"] for e in ev3 if e["event"] == "citations")
    assert cits3[0]["url"] == "https://example.com/serper"
    assert "Serper Hit" in groq3.calls[0]["messages"][0]["content"]


def test_chat_research_mode_always_searches_multi_query(client_factory):
    seen: list = []
    tc, _ = client_factory(tavily=tavily_transport(counter=seen))
    sid = new_session(tc, mode="research")["id"]
    _, ev = chat(tc, sid, "What is quantum error correction?", mode="research", web_search=False)
    assert ev[0]["data"]["route"] == "web" and len(seen) >= 2
    assert {s["search_depth"] for s in seen} == {"advanced"}


def test_chat_document_mode_without_documents_tells_the_model(client_factory):
    groq = FakeProviderClient()
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc, mode="document")["id"]
    chat(tc, sid, "Summarize my contract", mode="document")
    assert "has not uploaded any documents" in groq.calls[0]["messages"][0]["content"]


def test_chat_history_is_used_without_duplicating_current_turn(client_factory):
    groq = FakeProviderClient()
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc)["id"]
    chat(tc, sid, "My favourite colour is teal and I like it very much")
    chat(tc, sid, "Please remember that and tell me what colour I said")
    sent = groq.calls[-1]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]
    assert sent[-1]["content"] == "Please remember that and tell me what colour I said"
    assert sum(1 for m in sent if m["content"] == sent[-1]["content"]) == 1


def test_semantic_cache_hit_skips_llm(client_factory):
    groq = FakeProviderClient()
    tc, _ = client_factory(llm_client=groq)
    s1, s2 = new_session(tc)["id"], new_session(tc)["id"]
    _, first = chat(tc, s1, "Explain how photosynthesis works in plants")
    assert len(groq.calls) == 1 and not first[-1]["data"]["cached"]
    _, second = chat(tc, s2, "Explain how photosynthesis works in plants")
    assert len(groq.calls) == 1                                           # served from cache
    assert second[-1]["data"]["cached"] is True and text_of(second) == text_of(first)
    assert tc.get(f"/sessions/{s2}").json()["messages"][-1]["meta"]["cached"] is True
    # different user must never see another user's cached answer
    r = tc.post("/chat/stream", json={"session_id": new_session(tc, user_id="u2")["id"], "user_id": "u2",
                                       "message": "Explain how photosynthesis works in plants"})
    assert len(groq.calls) == 2 and parse_sse(r.text)[-1]["data"]["cached"] is False


def test_regenerate_replaces_last_answer(client_factory):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("ok", "first answer"), ("ok", "second answer")]})
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc)["id"]
    chat(tc, sid, "Tell me something interesting about volcanoes")
    r, ev = chat(tc, sid, "", regenerate=True)
    assert text_of(ev).strip() == "second answer"
    msgs = tc.get(f"/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["content"].strip() == "second answer"
    r2, ev2 = chat(tc, new_session(tc)["id"], "", regenerate=True)        # nothing to regenerate
    assert ev2[-1]["event"] == "error" and ev2[-1]["data"]["code"] == "nothing_to_regenerate"


def test_failover_is_reported_to_client(client_factory):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("raise", 429)]})
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc)["id"]
    _, ev = chat(tc, sid, "Explain the water cycle in brief")
    model_ev = next(e["data"] for e in ev if e["event"] == "model")
    assert model_ev["model"] == "openai/gpt-oss-20b" and model_ev["fell_back_from"] == ["openai/gpt-oss-120b"]
    assert ev[-1]["data"]["fell_back_from"] == ["openai/gpt-oss-120b"]


def test_all_models_down_yields_error_event_not_500(client_factory):
    every = {m: [("raise", 503)] for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")}
    tc, _ = client_factory(llm_client=FakeProviderClient(every),
                           google_client=FakeProviderClient({m: [("raise", 503)] for m in
                                                             ("gemini-3.5-flash-lite", "gemini-3.7-flash")}))
    sid = new_session(tc)["id"]
    r, ev = chat(tc, sid, "Anything at all please")
    assert r.status_code == 200 and ev[-1]["event"] == "error"
    assert ev[-1]["data"]["code"] == "all_models_failed" and ev[-1]["data"]["retryable"]
    # the user turn is preserved so a retry/regenerate works
    assert [m["role"] for m in tc.get(f"/sessions/{sid}").json()["messages"]] == ["user"]


def test_mid_stream_error_keeps_partial_answer(client_factory):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("midfail", "one two three four five six seven")]})
    tc, _ = client_factory(llm_client=groq)
    sid = new_session(tc)["id"]
    _, ev = chat(tc, sid, "Count for me please from one to seven")
    assert ev[-1]["event"] == "error" and ev[-1]["data"]["partial"] is True
    msgs = tc.get(f"/sessions/{sid}").json()["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["meta"]["partial"] is True


def test_chat_validation_and_errors(client_factory):
    tc, _ = client_factory()
    assert tc.post("/chat/stream", json={"session_id": "nope", "message": "hi"}).status_code == 404
    sid = new_session(tc)["id"]
    r, ev = chat(tc, sid, "   ")
    assert ev[-1]["data"]["code"] == "empty_message"
    assert tc.post("/chat/stream", json={"message": "hi"}).status_code == 422
    assert tc.post("/chat/stream", json={"session_id": sid, "message": "x" * 20001}).status_code == 422


def test_non_streaming_chat_and_legacy_generate(client_factory):
    tc, _ = client_factory()
    sid = new_session(tc)["id"]
    r = tc.post("/chat", json={"session_id": sid, "message": "Say something nice about autumn", "user_id": "u1"})
    j = r.json()
    assert r.status_code == 200 and j["response"].strip() == "answer from openai/gpt-oss-120b"
    assert j["model"] == "openai/gpt-oss-120b" and j["message_id"]

    g = tc.post("/generate", json={"messages": [{"role": "user", "content": "hello"}], "system_prompt": "be brief"})
    assert g.status_code == 200 and g.json()["model"] == "openai/gpt-oss-120b"
    assert tc.post("/generate", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]}).status_code == 400
    assert tc.post("/generate", json={"messages": []}).status_code == 400


def test_feedback_roundtrip(client_factory):
    tc, _ = client_factory()
    sid = new_session(tc)["id"]
    _, ev = chat(tc, sid, "Give me a fun fact about octopuses")
    mid = ev[-1]["data"]["message_id"]
    assert tc.post(f"/messages/{mid}/feedback", json={"value": 1}).json()["feedback"] == 1
    assert tc.post(f"/messages/{mid}/feedback", json={"value": -1}).json()["feedback"] == -1
    assert tc.post(f"/messages/{mid}/feedback", json={"value": 0}).json()["feedback"] is None
    assert tc.post(f"/messages/{mid}/feedback", json={"value": 5}).status_code == 422
    assert tc.post("/messages/missing/feedback", json={"value": 1}).status_code == 404
    user_msg = tc.get(f"/sessions/{sid}").json()["messages"][0]["id"]
    assert tc.post(f"/messages/{user_msg}/feedback", json={"value": 1}).status_code == 404   # assistant only


def test_rate_limit_returns_429(client_factory, settings):
    tc, _ = client_factory(settings_obj=settings.model_copy(update={"rate_limit_rpm_per_user": 2}))
    sid = new_session(tc)["id"]
    codes = [tc.post("/chat/stream", json={"session_id": sid, "message": f"hello number {i} there"}).status_code
             for i in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes
    r = tc.post("/chat/stream", json={"session_id": sid, "message": "again please now"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


async def test_client_disconnect_persists_partial_answer(build_container):
    groq = FakeProviderClient({"openai/gpt-oss-120b": [("ok", "alpha beta gamma delta epsilon zeta eta theta")]})
    c = build_container(llm_client=groq)
    session = c.sessions.create_session(SessionCreate(user_id="u1"))
    gen = c.chat.stream(ChatCommand(session_id=session.id, message="Please list greek letters for me"))
    seen = 0
    async for ev in gen:
        if ev["event"] == "token":
            seen += 1
            if seen == 3:
                break
    await gen.aclose()                                        # what Starlette does on disconnect / Stop
    msgs = c.sessions.get_session(session.id).messages
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].meta["stopped"] is True and msgs[1].content.strip().startswith("alpha")


def test_migration_adds_columns_to_legacy_database(tmp_path):
    import sqlite3
    from backend.core.database import Database

    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sessions(id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
            model TEXT NOT NULL, mode TEXT NOT NULL, created_at DATETIME, updated_at DATETIME);
        CREATE TABLE messages(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, citations JSON, created_at DATETIME);
        CREATE TABLE documents(id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL,
            file_type TEXT NOT NULL, chunk_count INTEGER, indexed_at DATETIME);
        INSERT INTO sessions VALUES('s1','u','t','m','general','2026-01-01','2026-01-01');
        INSERT INTO messages VALUES('m1','s1','user','hello',NULL,'2026-01-01');
    """)
    con.commit()
    con.close()
    db = Database(f"sqlite:///{path}")
    db.init()
    db.init()                                                 # idempotent
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(messages)")}
    assert {"feedback", "meta"} <= cols
    assert sqlite3.connect(path).execute("SELECT content FROM messages").fetchone()[0] == "hello"
    db.dispose()


def test_public_config_advertises_the_real_upload_limits(client_factory):
    """GET /config is read-only and reports what /upload really enforces (the UI validates against it)."""
    from backend.services.parser import SUPPORTED_EXTENSIONS

    tc, _ = client_factory()
    body = tc.get("/config").json()
    assert set(body) == {"upload", "github_oauth"}
    assert body["github_oauth"] is False
    assert body["upload"]["extensions"] == sorted(SUPPORTED_EXTENSIONS)
    assert isinstance(body["upload"]["max_mb"], int) and body["upload"]["max_mb"] >= 1
    # additive and read-only: no other verb is accepted, and nothing secret is exposed
    assert tc.post("/config").status_code == 405
    assert "key" not in str(body).lower()


def test_public_config_limit_follows_the_setting(settings, build_container, client_factory):
    tc, c = client_factory()
    assert tc.get("/config").json()["upload"]["max_mb"] == c.settings.max_upload_mb


def test_frontend_files_must_be_revalidated_so_a_page_never_mixes_old_and_new_scripts(client_factory):
    """Regression: with no Cache-Control a browser reused stale scripts after an edit and startup failed
    (model list stuck on "Loading models…"). Every frontend response now says no-cache and supports ETag/304."""
    tc, _ = client_factory()
    for path in ("/", "/js/app.js", "/js/ui.js", "/css/main.css"):
        r = tc.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
    first = tc.get("/js/ui.js")
    etag = first.headers["etag"]
    again = tc.get("/js/ui.js", headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.headers["cache-control"] == "no-cache" and not again.content
    # API responses are untouched
    assert "cache-control" not in tc.get("/models").headers
