"""
FastAPI application — complete REST API and static file serving.

Endpoints:
  GET  /health                   — Health check (DB, Cohere, Tavily)
  GET  /models                   — Available LLM models
  POST /upload                   — Multi-format document ingestion
  POST /retrieve                 — Hybrid RAG document retrieval
  POST /search                   — Web search (Tavily)
  GET  /sessions                 — List sessions for user
  POST /sessions                 — Create new session
  GET  /sessions/{id}            — Get session + full message history
  PATCH /sessions/{id}           — Update session metadata (title, model, mode)
  DELETE /sessions/{id}          — Delete session + messages
  POST /sessions/{id}/messages   — Append message (user or assistant)
  GET  /documents                — List indexed documents for user
  GET  /ping                     — Keepalive endpoint for free-tier uptime monitors
  GET  /                         — Serves the HTML5/CSS/JS single page application
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.config import (
    ALL_MODELS,
    ALLOWED_ORIGINS,
    BACKEND_HOST,
    BACKEND_PORT,
    COHERE_API_KEY,
    FRONTEND_DIR,
    TAVILY_API_KEY,
)
from backend.database import get_db, init_db
from backend.models.api import (
    DocumentSchema,
    HealthResponse,
    MessageCreate,
    MessageSchema,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    SessionCreate,
    SessionSchema,
    SessionSummary,
    SessionUpdate,
    UploadResponse,
)
from backend.models.db import ChatSession, Document as DBDocument, Message as DBMessage
from backend.rag.retriever import process_and_index, retrieve_documents, store_manager
from backend.services.parser import parse_file, parse_url
from backend.services.search import multi_search, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create SQLite tables
    init_db()
    print("[main] Database initialized.")
    yield


app = FastAPI(
    title="AI Assistant Pro — Agentic RAG Platform",
    description="Multipurpose AI chatbot with hybrid RAG, web search, persistent memory, and multi-model support.",
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System / Health ───────────────────────────────────────────────────────────

@app.get("/ping", tags=["System"])
def ping():
    """Lightweight keepalive endpoint for free-tier uptime monitors."""
    return {"pong": True}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Health check: verifies DB, Cohere, and Tavily status."""
    services = {
        "database": "connected",
        "cohere": "configured" if COHERE_API_KEY else "missing_key",
        "tavily": "configured" if TAVILY_API_KEY else "missing_key",
    }
    try:
        db.execute(DBMessage.__table__.select().limit(1))
    except Exception as exc:
        services["database"] = f"error: {exc}"

    return HealthResponse(status="healthy", services=services)


@app.get("/models", tags=["System"])
def get_models():
    """Return all available models grouped by provider."""
    return {"models": ALL_MODELS}


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/sessions", response_model=List[SessionSummary], tags=["Sessions"])
def list_sessions(user_id: str = "default", db: Session = Depends(get_db)):
    """List all chat sessions for a user, sorted newest first."""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return sessions


@app.post("/sessions", response_model=SessionSchema, status_code=status.HTTP_201_CREATED, tags=["Sessions"])
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    """Create a new chat session."""
    session = ChatSession(
        user_id=payload.user_id,
        title=payload.title,
        model=payload.model,
        mode=payload.mode,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/sessions/{session_id}", response_model=SessionSchema, tags=["Sessions"])
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get a session and its full message history."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.patch("/sessions/{session_id}", response_model=SessionSchema, tags=["Sessions"])
def update_session(session_id: str, payload: SessionUpdate, db: Session = Depends(get_db)):
    """Update session title, model, or mode."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if payload.title is not None:
        session.title = payload.title
    if payload.model is not None:
        session.model = payload.model
    if payload.mode is not None:
        session.mode = payload.mode

    db.commit()
    db.refresh(session)
    return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Sessions"])
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a session and all its messages."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    db.delete(session)
    db.commit()
    return None


# ── Messages ──────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/messages", response_model=MessageSchema, status_code=status.HTTP_201_CREATED, tags=["Messages"])
def save_message(session_id: str, payload: MessageCreate, db: Session = Depends(get_db)):
    """Append a user or assistant message to a session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    citations_json = [c.model_dump() for c in payload.citations] if payload.citations else None

    msg = DBMessage(
        session_id=session_id,
        role=payload.role,
        content=payload.content,
        citations=citations_json,
    )
    db.add(msg)

    # Auto-generate title from first user message if still default
    if payload.role == "user" and session.title in ("New Chat", "Untitled"):
        words = payload.content.strip().split()
        session.title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

    db.commit()
    db.refresh(msg)
    return msg


# ── Ingestion / Documents ─────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(None),
    url: str = Query(None),
    user_id: str = "default",
    db: Session = Depends(get_db),
):
    """
    Upload and index a document (PDF, DOCX, CSV, XLSX, TXT, or web URL).
    Extracts text, builds chunks, and saves embeddings into FAISS.
    """
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL.")

    pages = []
    filename = ""
    file_type = ""

    if url:
        filename = url
        file_type = "url"
        pages = parse_url(url)
    elif file:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        file_type = ext or "bin"

        # Save to temp file for parsing
        suffix = f".{ext}" if ext else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            pages = parse_file(tmp_path, filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not pages:
        raise HTTPException(status_code=422, detail=f"No readable text extracted from '{filename}'.")

    # Index into persistent FAISS
    success = process_and_index(user_id=user_id, pages=pages, filename=filename)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to build vector index.")

    # Record in SQLite
    doc_record = DBDocument(
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        chunk_count=len(pages),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    return UploadResponse(
        document_id=doc_record.id,
        filename=filename,
        file_type=file_type,
        chunk_count=len(pages),
        user_id=user_id,
    )


@app.get("/documents", response_model=List[DocumentSchema], tags=["Documents"])
def list_documents(user_id: str = "default", db: Session = Depends(get_db)):
    """List all indexed documents for a user."""
    docs = (
        db.query(DBDocument)
        .filter(DBDocument.user_id == user_id)
        .order_by(DBDocument.indexed_at.desc())
        .all()
    )
    return docs


# ── RAG Retrieval ─────────────────────────────────────────────────────────────

@app.post("/retrieve", response_model=RetrieveResponse, tags=["RAG"])
def retrieve_endpoint(payload: RetrieveRequest):
    """
    Search indexed documents for relevant excerpts.
    Uses FAISS recall + Cohere reranking.
    """
    docs = retrieve_documents(
        user_id=payload.user_id,
        query=payload.query,
        top_k=payload.top_k,
    )

    chunks = []
    for d in docs:
        relevance = float(d.metadata.get("relevance", 0.0))
        if relevance >= payload.threshold:
            chunks.append(
                RetrievedChunk(
                    source=d.metadata.get("source", "Unknown"),
                    page=d.metadata.get("page", "?"),
                    snippet=d.page_content,
                    relevance=relevance,
                )
            )

    return RetrieveResponse(
        chunks=chunks,
        used_rag=len(chunks) > 0,
        query=payload.query,
    )


# ── Web Search ────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_endpoint(payload: SearchRequest):
    """Perform real-time Tavily web search."""
    return search(query=payload.query, max_results=payload.max_results)


# ── Frontend Static Files & SPA Routing ───────────────────────────────────────

# Mount assets/css/js under their relative paths
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/", tags=["Frontend"])
def serve_spa():
    """Serve the single-page application entry point."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Frontend not found. API is running."}
    return FileResponse(str(index_path))


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
