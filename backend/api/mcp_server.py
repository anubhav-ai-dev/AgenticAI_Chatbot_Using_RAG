"""
Optional MCP server (stdio) exposing the same retrieval, search and chat services
the web app uses, so Claude Code or any MCP client can reuse them.

    python -m backend.api.mcp_server

Register with Claude Code:
    claude mcp add rag-chatbot -- /path/to/venv/bin/python -m backend.api.mcp_server
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from backend.core.config import configure_logging, get_settings
from backend.core.container import ServiceContainer
from backend.models.api import SessionCreate
from backend.services.chat_service import ChatCommand

server = MCPServer("rag-chatbot", instructions=(
    "Search the user's indexed documents (hybrid BM25 + vector with reranking), search the web, "
    "connect a GitHub repository, or ask the full chatbot pipeline."))


class _Runtime:
    """Builds the service container lazily inside the server's event loop."""

    def __init__(self) -> None:
        self._container: Optional[ServiceContainer] = None
        self._lock: Optional[asyncio.Lock] = None

    async def get(self) -> ServiceContainer:
        if self._container is None:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._container is None:
                    c = ServiceContainer.build(get_settings())
                    await c.startup(warm=False)
                    self._container = c
        return self._container


_runtime = _Runtime()


@server.tool()
async def search_documents(query: str, user_id: str = "default", top_k: int = 5) -> list[dict[str, Any]]:
    """Hybrid search over the user's uploaded documents. Returns ranked excerpts."""
    c = await _runtime.get()
    result = await c.retrieval.search(user_id, query, top_k=max(1, min(top_k, 20)))
    return [{"source": i.source, "page": i.page, "relevance": round(i.relevance, 3), "text": i.text}
            for i in result.items]


@server.tool()
async def list_documents(user_id: str = "default") -> list[dict[str, Any]]:
    """List the user's indexed documents."""
    c = await _runtime.get()
    rows = await asyncio.to_thread(c.documents.list_documents, user_id)
    return [{"id": d.id, "filename": d.filename, "type": d.file_type, "chunks": d.chunk_count}
            for d in rows]


@server.tool()
async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web (Tavily). Returns titles, URLs and snippets."""
    c = await _runtime.get()
    outcome = await c.search.search(query, max(1, min(max_results, 10)))
    return {"status": outcome.status,
            "results": [r.model_dump() for r in outcome.results]}


@server.tool()
async def connect_github_repo(url: str, user_id: str = "default",
                              token: str = "") -> dict[str, Any]:
    """Index a GitHub repository into the user's document store and keep a repo brief."""
    c = await _runtime.get()
    row = await c.github.connect(user_id, url, token or None)
    return {"id": row.id, "owner": row.owner, "repo": row.repo, "branch": row.branch,
            "file_count": row.file_count}


@server.tool()
async def list_github_repos(user_id: str = "default") -> list[dict[str, Any]]:
    """List GitHub repositories connected for this user."""
    c = await _runtime.get()
    rows = await asyncio.to_thread(c.github.list_repos, user_id)
    return [{"id": r.id, "owner": r.owner, "repo": r.repo, "branch": r.branch,
             "file_count": r.file_count} for r in rows]


@server.tool()
async def github_context(user_id: str = "default") -> str:
    """Return the stored briefs for connected GitHub repositories."""
    c = await _runtime.get()
    return await asyncio.to_thread(c.github.combined_brief, user_id)


@server.tool()
async def ask(question: str, user_id: str = "default", mode: str = "general",
              model: Optional[str] = None) -> dict[str, Any]:
    """Ask the full chatbot pipeline (routing, RAG, web search, model fallback)."""
    c = await _runtime.get()
    session = await asyncio.to_thread(
        c.sessions.create_session, SessionCreate(user_id=user_id, title="MCP", mode=mode))
    result = await c.chat.complete(ChatCommand(
        session_id=session.id, message=question, mode=mode, model=model, web_search=True))
    if result.get("error"):
        return {"error": result["error"].get("message", "Chat failed.")}
    return {"answer": result["response"], "model": result["model"],
            "route": result["route"], "citations": result["citations"]}


def main() -> None:
    configure_logging("WARNING")        # stdout is the protocol channel; logs go to stderr
    server.run("stdio")


if __name__ == "__main__":
    main()
