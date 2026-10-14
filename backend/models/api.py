"""
Pydantic request / response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=128)
    title:   str = Field(default="New Chat", max_length=200)
    model:   Optional[str] = None            # None → server default (config)
    mode:    str = "general"


class SessionUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[str] = None
    mode:  Optional[str] = None


class CitationSchema(BaseModel):
    id:      int
    type:    str = "doc"                      # doc | web
    source:  str
    page:    Any = None                       # int for PDFs, row range for CSV
    url:     Optional[str] = None
    snippet: str = ""


class MessageSchema(BaseModel):
    id:         str
    session_id: str
    role:       str
    content:    str
    citations:  Optional[list[CitationSchema]] = None
    feedback:   Optional[int] = None
    meta:       Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionSchema(BaseModel):
    id:         str
    user_id:    str
    title:      str
    model:      str
    mode:       str
    created_at: datetime
    updated_at: datetime
    messages:   list[MessageSchema] = []

    model_config = {"from_attributes": True}


class SessionSummary(BaseModel):
    id:         str
    user_id:    str
    title:      str
    model:      str
    mode:       str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Messages ──────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    role:      Literal["user", "assistant", "system"]
    content:   str = Field(min_length=1)
    citations: Optional[list[CitationSchema]] = None


class FeedbackRequest(BaseModel):
    value: Literal[-1, 0, 1]                  # 0 clears feedback


# ── Chat (primary pipeline) ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str = Field(default="", max_length=20000)
    user_id:    str = Field(default="default", min_length=1, max_length=128)
    mode:       str = "general"
    model:      Optional[str] = None
    web_search: bool = True
    regenerate: bool = False


class ChatResponse(BaseModel):
    message_id:  Optional[str] = None
    response:    str
    model:       Optional[str] = None
    route:       str = "direct"
    cached:      bool = False
    citations:   list[CitationSchema] = []
    timings:     dict[str, float] = {}


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id:  str
    filename:     str
    file_type:    str
    chunk_count:  int
    user_id:      str
    indexing:     str = "hybrid"              # hybrid | lexical


# ── RAG retrieve ──────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:     str = Field(min_length=1, max_length=4000)
    user_id:   str = Field(default="default", min_length=1, max_length=128)
    threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    top_k:     int   = Field(default=5,    ge=1,   le=20)


class RetrievedChunk(BaseModel):
    source:    str
    page:      Any    = None
    snippet:   str
    relevance: float  = 0.0


class RetrieveResponse(BaseModel):
    chunks:      list[RetrievedChunk]
    used_rag:    bool
    query:       str
    mode:        str = "hybrid"               # hybrid | lexical | dense
    reranked:    bool = False
    elapsed_ms:  float = 0.0


# ── Web search ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:    str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    title:   str
    url:     str
    snippet: str
    score:   float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query:   str
    status:  str = "ok"                       # ok | disabled | error | timeout


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    id:          str
    filename:    str
    file_type:   str
    chunk_count: int
    indexed_at:  datetime

    model_config = {"from_attributes": True}


# ── Models / generate ─────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id:          str
    label:       str
    provider:    str
    description: str = ""
    available:   bool = True
    reason:      Optional[str] = None         # why unavailable
    default:     bool = False


class ChatMessage(BaseModel):
    role:    str  # system | user | assistant
    content: str


class GenerateRequest(BaseModel):
    model:         Optional[str] = None
    messages:      list[ChatMessage]
    system_prompt: str = ""


class GenerateResponse(BaseModel):
    response: str
    model:    str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str                             # healthy | degraded | unavailable
    services: dict[str, str]
    details:  dict[str, Any] = {}
