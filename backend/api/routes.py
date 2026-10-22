"""
HTTP layer — thin handlers that delegate to injected services.

Chat:      POST /chat/stream (SSE) · POST /chat (JSON) · POST /generate (legacy)
Sessions:  /sessions CRUD · POST /sessions/{id}/messages · POST /messages/{id}/feedback
Documents: POST /upload · GET /documents · DELETE /documents/{id}
RAG/Web:   POST /retrieve · POST /search
System:    GET /health · GET /models · GET /config · GET /ping
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from fastapi.concurrency import run_in_threadpool

from backend.core.container import (
    ServiceContainer,
    get_chat_service,
    get_container,
    get_document_service,
    get_llm_service,
    get_retrieval_service,
    get_search_service,
    get_session_service,
)
from backend.core.exceptions import IngestError, LLMError, SessionNotFound
from backend.models.api import (
    ChatRequest,
    ChatResponse,
    DocumentSchema,
    FeedbackRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MessageCreate,
    MessageSchema,
    PublicConfig,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    SessionCreate,
    SessionSchema,
    SessionSummary,
    SessionUpdate,
    UploadLimits,
    UploadResponse,
)
from backend.services.chat_service import ChatCommand, ChatService
from backend.services.document_service import DocumentService
from backend.services.llm_service import LLMService
from backend.services.parser import SUPPORTED_EXTENSIONS
from backend.services.retrieval_service import RetrievalService
from backend.services.search_service import WebSearchService
from backend.services.session_service import SessionService
from backend.utils.sse import format_sse

logger = logging.getLogger(__name__)
router = APIRouter()

UserId = Query(default="default", min_length=1, max_length=128)


# ── helpers ───────────────────────────────────────────────────────────────────

def _throttle(c: ServiceContainer, request: Request, user_id: str) -> None:
    ip = request.client.host if request.client else "unknown"
    allowed, retry_after = c.rate_limiter.allow(f"{ip}|{user_id}")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(int(retry_after) + 1)})


def _llm_http_error(err: LLMError) -> HTTPException:
    if err.code == "no_provider":
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif err.code == "rate_limited":
        code = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        code = status.HTTP_502_BAD_GATEWAY
    headers = {"Retry-After": str(int(err.retry_after) + 1)} if err.retry_after else None
    return HTTPException(status_code=code, detail=err.message, headers=headers)


def _command(payload: ChatRequest) -> ChatCommand:
    return ChatCommand(session_id=payload.session_id, message=payload.message, mode=payload.mode,
                       model=payload.model, web_search=payload.web_search,
                       regenerate=payload.regenerate)


# ── system ────────────────────────────────────────────────────────────────────

@router.get("/ping", tags=["System"])
def ping():
    """Lightweight keepalive endpoint for free-tier uptime monitors."""
    return {"pong": True}


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(c: ServiceContainer = Depends(get_container)):
    state, services, details = c.health()
    return HealthResponse(status=state, services=services, details=details)


@router.get("/config", response_model=PublicConfig, tags=["System"])
def public_config(c: ServiceContainer = Depends(get_container)):
    """Upload limits for client-side validation (read-only; /upload enforces them again)."""
    return PublicConfig(upload=UploadLimits(max_mb=c.settings.max_upload_mb,
                                            extensions=sorted(SUPPORTED_EXTENSIONS)))


@router.get("/models", tags=["System"])
async def list_models(refresh: bool = False, llm: LLMService = Depends(get_llm_service),
                      c: ServiceContainer = Depends(get_container)):
    """Configured chat models with availability for the current API keys."""
    if refresh:
        await llm.discover()
    return {"models": llm.public_models(), "default": c.settings.llm_default_model}


# ── chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(payload: ChatRequest, request: Request,
                      chat: ChatService = Depends(get_chat_service),
                      sessions: SessionService = Depends(get_session_service),
                      c: ServiceContainer = Depends(get_container)):
    """Server-sent events: meta · status · citations · model · token · error · done."""
    _throttle(c, request, payload.user_id)
    if not c.llm.configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "No LLM provider is configured (set GROQ_API_KEY or GOOGLE_API_KEY).")
    if await run_in_threadpool(sessions.session_meta, payload.session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    async def events() -> AsyncIterator[str]:
        async for ev in chat.stream(_command(payload)):
            yield format_sse(ev)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_complete(payload: ChatRequest, request: Request,
                        chat: ChatService = Depends(get_chat_service),
                        c: ServiceContainer = Depends(get_container)):
    """Same pipeline as /chat/stream, returned as one JSON document."""
    _throttle(c, request, payload.user_id)
    if not c.llm.configured:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "No LLM provider is configured (set GROQ_API_KEY or GOOGLE_API_KEY).")
    result = await chat.complete(_command(payload))
    err = result.get("error")
    if err:
        code = {"session_not_found": 404, "empty_message": 400, "nothing_to_regenerate": 400,
                "rate_limited": 429}.get(err.get("code"), 502)
        raise HTTPException(code, err.get("message", "Chat failed."))
    return ChatResponse(message_id=result["message_id"], response=result["response"],
                        model=result["model"], route=result["route"], cached=result["cached"],
                        citations=result["citations"], timings=result["timings"])


@router.post("/generate", response_model=GenerateResponse, tags=["Chat"])
async def generate(payload: GenerateRequest, llm: LLMService = Depends(get_llm_service),
                   c: ServiceContainer = Depends(get_container)):
    """Plain completion (no RAG/search/history) with model fallback."""
    model_id = payload.model or c.settings.llm_default_model
    if not llm.has_model(model_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown model '{model_id}'.")
    messages: list[dict[str, str]] = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    for m in payload.messages:
        role = (m.role or "").lower()
        messages.append({"role": role if role in ("system", "assistant") else "user",
                         "content": m.content})
    if not any(m["role"] != "system" for m in messages):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one message is required.")
    try:
        result = await llm.complete(messages, model=model_id)
    except LLMError as err:
        raise _llm_http_error(err) from err
    return GenerateResponse(response=result.text, model=result.model)


# ── sessions & messages ───────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionSummary], tags=["Sessions"])
def list_sessions(user_id: str = UserId, sessions: SessionService = Depends(get_session_service)):
    return sessions.list_sessions(user_id)


@router.post("/sessions", response_model=SessionSchema, status_code=status.HTTP_201_CREATED,
             tags=["Sessions"])
def create_session(payload: SessionCreate, sessions: SessionService = Depends(get_session_service)):
    return sessions.create_session(payload)


@router.get("/sessions/{session_id}", response_model=SessionSchema, tags=["Sessions"])
def get_session(session_id: str, sessions: SessionService = Depends(get_session_service)):
    try:
        return sessions.get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.") from exc


@router.patch("/sessions/{session_id}", response_model=SessionSchema, tags=["Sessions"])
def update_session(session_id: str, payload: SessionUpdate,
                   sessions: SessionService = Depends(get_session_service)):
    try:
        return sessions.update_session(session_id, payload)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.") from exc


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Sessions"])
def delete_session(session_id: str, sessions: SessionService = Depends(get_session_service)):
    try:
        sessions.delete_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sessions/{session_id}/messages", response_model=MessageSchema,
             status_code=status.HTTP_201_CREATED, tags=["Messages"])
def save_message(session_id: str, payload: MessageCreate,
                 sessions: SessionService = Depends(get_session_service)):
    try:
        return sessions.add_message(session_id, payload.role, payload.content, payload.citations)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.") from exc


@router.post("/messages/{message_id}/feedback", response_model=MessageSchema, tags=["Messages"])
def message_feedback(message_id: str, payload: FeedbackRequest,
                     sessions: SessionService = Depends(get_session_service)):
    try:
        return sessions.set_feedback(message_id, payload.value)
    except SessionNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found.") from exc


# ── documents ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: Optional[UploadFile] = File(None), url: Optional[str] = Query(None),
                          user_id: str = UserId,
                          docs: DocumentService = Depends(get_document_service),
                          c: ServiceContainer = Depends(get_container)):
    """Upload and index a document, image (OCR) or web URL. See parser.SUPPORTED_EXTENSIONS."""
    if not file and not url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Provide either a file or a URL.")
    try:
        if url:
            result = await docs.ingest_url(user_id, url.strip())
        else:
            limit = c.settings.max_upload_mb * 1024 * 1024
            data = await file.read(limit + 1)           # never buffer more than the limit
            if len(data) > limit:
                raise HTTPException(413,
                                    f"File exceeds the {c.settings.max_upload_mb} MB limit.")
            result = await docs.ingest_file(user_id, file.filename or "upload", data)
    except IngestError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    return UploadResponse(document_id=result.document_id, filename=result.filename,
                          file_type=result.file_type, chunk_count=result.chunk_count,
                          user_id=user_id, indexing=result.indexing)


@router.get("/documents", response_model=List[DocumentSchema], tags=["Documents"])
def list_documents(user_id: str = UserId, docs: DocumentService = Depends(get_document_service)):
    return docs.list_documents(user_id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def delete_document(doc_id: str, user_id: str = UserId,
                          docs: DocumentService = Depends(get_document_service)):
    if not await docs.delete_document(user_id, doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── RAG retrieval & web search ────────────────────────────────────────────────

@router.post("/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def retrieve(payload: RetrieveRequest,
                   retrieval: RetrievalService = Depends(get_retrieval_service)):
    """Hybrid (BM25 + vector) retrieval with reranking."""
    result = await retrieval.search(payload.user_id, payload.query, top_k=payload.top_k,
                                    min_relevance=payload.threshold)
    chunks = [RetrievedChunk(source=i.source, page=i.page, snippet=i.text, relevance=i.relevance)
              for i in result.items]
    return RetrieveResponse(chunks=chunks, used_rag=result.used_rag, query=payload.query,
                            mode=result.mode, reranked=result.reranked,
                            elapsed_ms=round(result.timings.get("total_ms", 0.0), 1))


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def web_search(payload: SearchRequest,
                     search: WebSearchService = Depends(get_search_service)):
    outcome = await search.search(payload.query, payload.max_results)
    return SearchResponse(results=outcome.results, query=payload.query, status=outcome.status)
