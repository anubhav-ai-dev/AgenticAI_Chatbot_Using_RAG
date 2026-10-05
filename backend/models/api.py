"""
Pydantic request / response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str = "default"
    title:   str = "New Chat"
    model:   str = "claude-sonnet-4-5"
    mode:    str = "general"


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    mode:  Optional[str] = None


class CitationSchema(BaseModel):
    id:      int
    source:  str
    page:    Any      = None   # int for PDFs, row range for CSV
    snippet: str = ""


class MessageSchema(BaseModel):
    id:         str
    session_id: str
    role:       str
    content:    str
    citations:  Optional[list[CitationSchema]] = None
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
    role:      str
    content:   str
    citations: Optional[list[CitationSchema]] = None


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id:  str
    filename:     str
    file_type:    str
    chunk_count:  int
    user_id:      str


# ── RAG retrieve ──────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:     str
    user_id:   str = "default"
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


# ── Web search ────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:    str
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    title:   str
    url:     str
    snippet: str
    score:   float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query:   str


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    id:          str
    filename:    str
    file_type:   str
    chunk_count: int
    indexed_at:  datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str
    services: dict[str, str]
