"""
FastAPI application — all HTTP endpoints.
Run with:  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agent import (
    clear_chat_history,
    get_chat_history,
    get_response,
    get_user_documents,
    process_pdf,
)
from backend.config import ALLOWED_MODELS, BACKEND_HOST, BACKEND_PORT
from backend.models import (
    ChatHistoryRequest,
    ChatRequest,
    ChatResponse,
    ClearHistoryRequest,
    HealthResponse,
    UserDocumentsRequest,
)

app = FastAPI(
    title="AI Chatbot API — RAG & Memory",
    description="LangGraph-powered chatbot with PDF RAG, session memory, and web search.",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_credentials=True requires an explicit origin list (not "*").
# Read from env so Render/Railway can set the deployed frontend URL.
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "AI Chatbot API — RAG & Memory",
        "version": "2.0.0",
        "endpoints": {
            "POST /chat": "Send a message and get an AI response",
            "POST /upload-pdf": "Upload a PDF for document Q&A",
            "POST /chat-history": "Retrieve session chat history",
            "POST /clear-history": "Clear session chat history",
            "POST /user-documents": "List documents uploaded by a user",
            "GET  /health": "Health check",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    return HealthResponse(status="healthy", message="API is running.")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with memory, RAG, and optional web search."""
    if request.model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Allowed: {ALLOWED_MODELS}",
        )

    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "default"

    try:
        response_text = get_response(
            llm_id=request.model_name,
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider,
            user_id=user_id,
            session_id=session_id,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(response=response_text, session_id=session_id, user_id=user_id)


@app.post("/upload-pdf", tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = "default",
):
    """Upload and process a PDF file for RAG."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Write to a temp file so pdfplumber/PyPDF2 can read it from disk
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as fh:
            fh.write(content)

        success = process_pdf(user_id=user_id, pdf_path=temp_path, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not success:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "filename": file.filename,
        "user_id": user_id,
    }


@app.post("/chat-history", tags=["Memory"])
def chat_history_endpoint(request: ChatHistoryRequest):
    """Return the full message history for a session."""
    history = get_chat_history(request.session_id)
    return {"history": history, "session_id": request.session_id}


@app.post("/clear-history", tags=["Memory"])
def clear_history_endpoint(request: ClearHistoryRequest):
    """Delete the message history for a session."""
    clear_chat_history(request.session_id)
    return {"message": f"History cleared for session '{request.session_id}'."}


@app.post("/user-documents", tags=["Documents"])
def user_documents_endpoint(request: UserDocumentsRequest):
    """Return the list of documents indexed for a user."""
    documents = get_user_documents(request.user_id)
    return {"documents": documents, "user_id": request.user_id}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
