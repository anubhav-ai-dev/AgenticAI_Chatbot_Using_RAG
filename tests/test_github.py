"""GitHub connect, indexing, OAuth device flow, and MCP wrappers."""

from __future__ import annotations

import base64

import httpx
import pytest

from backend.services.github_service import GitHubError, _select_paths, parse_repo_url
from tests.conftest import FakeProviderClient, parse_sse

MODEL = "openai/gpt-oss-120b"
README = "# Sample app\nThis repository is a small Python service used in tests.\n" * 3
PY = "def main():\n    print('hello from the connected application')\n" * 4


def github_transport(*, token_error: str | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.tavily.com" in url:
            return httpx.Response(200, json={"results": []})
        if url.rstrip("/").endswith("/login/device/code"):
            return httpx.Response(200, json={
                "device_code": "dev-1", "user_code": "ABCD-1234",
                "verification_uri": "https://example.com/device",
                "expires_in": 900, "interval": 5,
            })
        if "oauth/access_token" in url:
            if token_error:
                return httpx.Response(200, json={"error": token_error})
            return httpx.Response(200, json={"access_token": "gho_secret"})
        if "/repos/missing/" in url:
            return httpx.Response(404, json={"message": "Not Found"})
        if "/git/trees/" in url:
            return httpx.Response(200, json={"tree": [
                {"path": "README.md", "type": "blob", "size": len(README)},
                {"path": "src/main.py", "type": "blob", "size": len(PY)},
                {"path": "node_modules/pkg/index.js", "type": "blob", "size": 40},
                {"path": "photo.png", "type": "blob", "size": 80},
            ]})
        if "/contents/README.md" in url:
            return httpx.Response(200, json={
                "encoding": "base64",
                "content": base64.b64encode(README.encode()).decode(),
            })
        if "/contents/src/main.py" in url:
            return httpx.Response(200, json={
                "encoding": "base64",
                "content": base64.b64encode(PY.encode()).decode(),
            })
        if url.rstrip("/").endswith("/repos/acme/app"):
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(404, json={"error": "nope"})

    return httpx.MockTransport(handler)


def test_parse_repo_url_and_path_selection():
    assert parse_repo_url("https://github.com/acme/app") == ("acme", "app")
    assert parse_repo_url("https://github.com/acme/app.git") == ("acme", "app")
    with pytest.raises(GitHubError, match="repository URL"):
        parse_repo_url("https://example.com/acme/app")
    paths = _select_paths([
        {"path": "README.md", "type": "blob", "size": 100},
        {"path": "src/main.py", "type": "blob", "size": 80},
        {"path": "node_modules/x.js", "type": "blob", "size": 10},
        {"path": "dist/app.js", "type": "blob", "size": 10},
        {"path": "huge.bin", "type": "blob", "size": 500_000},
        {"path": "photo.png", "type": "blob", "size": 20},
    ])
    assert paths[0] == "README.md" and "src/main.py" in paths
    assert all("node_modules" not in p and "dist/" not in p for p in paths)
    assert "photo.png" not in paths and "huge.bin" not in paths


def test_connect_indexes_repo_and_hides_the_token(client_factory):
    groq = FakeProviderClient({MODEL: [("ok", "Python service with a README and src/main.py.")]})
    tc, _ = client_factory(llm_client=groq, tavily=github_transport())
    r = tc.post("/github/connect", json={
        "url": "https://github.com/acme/app", "token": "ghp_secret_token", "user_id": "u1",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["owner"] == "acme" and body["repo"] == "app" and body["file_count"] == 2
    dumped = str(body).lower()
    assert "ghp_secret" not in dumped and "gho_secret" not in dumped
    assert "token" not in body
    repos = tc.get("/github/repos", params={"user_id": "u1"}).json()
    assert len(repos) == 1 and repos[0]["id"] == body["id"]
    docs = tc.get("/documents", params={"user_id": "u1"}).json()
    names = {d["filename"] for d in docs}
    assert "acme/app/README.md" in names and "acme/app/src/main.py" in names
    assert all(d["file_type"] == "github" for d in docs)
    assert tc.delete(f"/github/repos/{body['id']}", params={"user_id": "u1"}).status_code == 204
    assert tc.get("/github/repos", params={"user_id": "u1"}).json() == []
    leftover = tc.get("/documents", params={"user_id": "u1"}).json()
    assert leftover == []


def test_connect_rejects_unknown_and_non_github_urls(client_factory):
    tc, _ = client_factory(tavily=github_transport())
    assert tc.post("/github/connect", json={"url": "https://example.com/acme/app"}).status_code == 400
    missing = tc.post("/github/connect", json={"url": "https://github.com/missing/x", "user_id": "u1"})
    assert missing.status_code == 404


def test_oauth_device_flow(client_factory, settings, build_container):
    s = settings.model_copy(update={"github_client_id": "client-id"})
    tc, _ = client_factory(settings_obj=s, tavily=github_transport())
    start = tc.post("/github/oauth/device")
    assert start.status_code == 200, start.text
    data = start.json()
    assert data["user_code"] == "ABCD-1234" and data["device_code"] == "dev-1"
    poll = tc.post("/github/oauth/poll", json={"device_code": data["device_code"], "user_id": "u1"})
    assert poll.status_code == 200 and poll.json()["status"] == "connected"
    assert "token" not in poll.json()


def test_oauth_pending_and_unconfigured(client_factory, settings):
    tc, _ = client_factory()
    assert tc.post("/github/oauth/device").status_code == 501
    s = settings.model_copy(update={"github_client_id": "client-id"})
    pending, _ = client_factory(settings_obj=s, tavily=github_transport(token_error="authorization_pending"))
    start = pending.post("/github/oauth/device").json()
    poll = pending.post("/github/oauth/poll", json={"device_code": start["device_code"]})
    assert poll.status_code == 200 and poll.json()["status"] == "pending"


def test_code_chat_gets_the_repo_brief(client_factory):
    groq = FakeProviderClient({MODEL: [
        ("ok", "Python service. Entry point src/main.py."),
        ("ok", "Reuse main() from src/main.py."),
    ]})
    tc, _ = client_factory(llm_client=groq, tavily=github_transport())
    assert tc.post("/github/connect", json={"url": "https://github.com/acme/app", "user_id": "u1"}).status_code == 200
    sid = tc.post("/sessions", json={"user_id": "u1", "title": "Code", "mode": "code"}).json()["id"]
    r = tc.post("/chat/stream", json={
        "session_id": sid, "message": "```python\nprint(1)\n```\nHow do I extend this?",
        "user_id": "u1", "mode": "code", "web_search": False,
    })
    assert r.status_code == 200 and "done" in {e["event"] for e in parse_sse(r.text)}
    system = groq.calls[-1]["messages"][0]["content"]
    assert "Repository context" in system and "src/main.py" in system
    assert "Codebase owner" in system


def test_mcp_exposes_github_tools():
    from backend.api import mcp_server
    assert callable(mcp_server.connect_github_repo)
    assert callable(mcp_server.list_github_repos)
    assert callable(mcp_server.github_context)
    assert callable(mcp_server.search_documents)
