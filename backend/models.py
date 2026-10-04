"""
Pydantic models for all FastAPI request and response bodies.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: list[str]
    allow_search: bool
    user_id: Optional[str] = "default"
    session_id: Optional[str] = "default"
    similarity_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str


class ChatHistoryRequest(BaseModel):
    session_id: str


class ClearHistoryRequest(BaseModel):
    session_id: str


class UserDocumentsRequest(BaseModel):
    user_id: str


class HealthResponse(BaseModel):
    status: str
    message: str
