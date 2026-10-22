"""
Central configuration — the ONLY module that reads environment variables.

Everything else receives a `Settings` instance through dependency injection
(see backend/core/container.py). Models, thresholds and limits are all env-driven;
the defaults below are the values validated against live free-tier endpoints.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/core/config.py -> project root is two levels above the package directory
BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

MODES: tuple[str, ...] = ("general", "document", "research", "code")


# ── LLM model registry ───────────────────────────────────────────────────────

class ModelSpec(BaseModel):
    """One selectable chat model. Every field can be overridden via LLM_MODELS."""

    id: str
    label: str
    provider: str                       # key into Settings.provider_configs()
    description: str = ""
    enabled: bool = True
    supports_system: bool = True        # False → system prompt merged into first user turn
    extra: dict[str, Any] = Field(default_factory=dict)  # request params (reasoning_effort, temperature…)
    rpm: Optional[int] = None           # soft requests/min used by the quota guard
    tpm: Optional[int] = None           # soft tokens/min used by the quota guard


# Free-tier models only. `extra` keeps latency low (reasoning effort) per model.
DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B", "provider": "groq",
        "description": "Best overall / complex reasoning",
        "extra": {"reasoning_effort": "low"}, "rpm": 30, "tpm": 8000,
    },
    {
        "id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B", "provider": "groq",
        "description": "Fast + cheaper — chat, agents, coding",
        "extra": {"reasoning_effort": "low"}, "rpm": 30, "tpm": 8000,
    },
    {
        "id": "qwen/qwen3.8-27b", "label": "Qwen3.8 27B", "provider": "groq",
        "description": "Reasoning + general tasks (preview)",
        "extra": {}, "rpm": 30, "tpm": 8000,
    },
    {
        "id": "gemini-3.7-flash", "label": "Gemini 3.7 Flash", "provider": "google",
        "description": "Coding, reasoning and agents",
        "extra": {"reasoning_effort": "low", "temperature": 1.0},
    },
    {
        "id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite", "provider": "google",
        "description": "Fast, lightweight chat and simple agents",
        "extra": {"temperature": 1.0},
    },
    {
        "id": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash-Lite", "provider": "google",
        "description": "Fast, lightweight chat (closed to new projects on some keys)",
        "extra": {},
    },
    {
        "id": "gemma-4-26b-a4b-it", "label": "Gemma 4 26B", "provider": "google",
        "description": "Open-weight model (experimental, slower)",
        "extra": {},
    },
    {
        "id": "gemma-4-31b-it", "label": "Gemma 4 31B", "provider": "google",
        "description": "Open-weight model (experimental)",
        "extra": {},
    },
]

DEFAULT_FALLBACK_CHAIN = (
    "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b,"
    "gemini-3.5-flash-lite,gemini-3.7-flash"
)


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


# ── Settings ─────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        case_sensitive=False,
    )

    # API keys (all optional — missing keys degrade, never crash)
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Provider endpoints (OpenAI-compatible chat/completions)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    tavily_base_url: str = "https://api.tavily.com"

    # LLM
    llm_default_model: str = "openai/gpt-oss-120b"
    llm_fallback_chain: str = DEFAULT_FALLBACK_CHAIN
    llm_fast_model: str = "openai/gpt-oss-20b"
    llm_models: str = ""                      # JSON list of ModelSpec dicts; replaces defaults
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=16, le=8192)
    llm_connect_timeout_s: float = 5.0
    llm_first_token_timeout_s: float = 12.0
    llm_total_timeout_s: float = 90.0
    model_discovery: bool = True
    breaker_cooldown_s: int = 30
    model_unavailable_cooldown_s: int = 3600

    # Embeddings / rerank
    embed_provider: Literal["local", "google", "cohere"] = "local"
    embed_model: str = ""
    embed_batch_size: int = Field(default=64, ge=1, le=256)
    rerank_provider: Literal["local", "cohere", "none"] = "local"
    rerank_model: str = ""
    rerank_enabled: bool = True
    rerank_candidates: int = Field(default=12, ge=1, le=50)
    rerank_max_chars: int = Field(default=1000, ge=100)   # cross-encoder cost grows with length
    rerank_timeout_s: float = 4.0

    # RAG
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    rag_candidate_k: int = Field(default=30, ge=1, le=200)
    rag_top_k: int = Field(default=8, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1)
    rag_dense_weight: float = 1.0
    rag_lexical_weight: float = 1.0
    rag_min_relevance: float = Field(default=0.2, ge=0.0, le=1.0)   # gate on blended relevance
    rag_cosine_floor: float = 0.5     # cosine at/below which dense similarity counts as 0 (bge-small)
    rag_cosine_ceil: float = 0.8      # cosine at/above which dense similarity counts as 1
    retrieval_timeout_s: float = 8.0
    retrieval_cache_ttl_s: int = 300

    # Web search
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    tavily_research_depth: Literal["basic", "advanced"] = "advanced"
    tavily_max_results: int = Field(default=4, ge=1, le=10)
    search_cache_ttl_s: int = 600
    search_timeout_s: float = 6.0

    # Conversation budgets (rough token estimates)
    history_max_messages: int = Field(default=10, ge=0)
    history_token_budget: int = 1200
    context_token_budget: int = 3000
    query_rewrite_enabled: bool = True

    # Semantic answer cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    semantic_cache_ttl_s: int = 3600
    semantic_cache_size: int = 300

    # Limits / safety
    rate_limit_rpm_per_user: int = Field(default=30, ge=1)
    max_upload_mb: int = Field(default=20, ge=1)
    allow_private_urls: bool = False
    url_fetch_timeout_s: float = 15.0

    # Server / storage
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:8000"
    database_url: str = ""
    data_dir: str = ""
    log_level: str = "INFO"

    # ── derived values ───────────────────────────────────────────────────────

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir).expanduser() if self.data_dir else BASE_DIR / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_dir(self) -> Path:
        path = self.data_path / "index"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir(self) -> Path:
        path = self.data_path / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_path / 'chatbot.db'}"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def fallback_chain(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_chain.split(",") if m.strip()]

    @property
    def embed_model_id(self) -> str:
        if self.embed_model:
            return self.embed_model
        return {
            "local": "BAAI/bge-small-en-v1.5",
            "google": "gemini-embedding-001",
            "cohere": "embed-english-v3.0",
        }[self.embed_provider]

    @property
    def rerank_model_id(self) -> str:
        if self.rerank_model:
            return self.rerank_model
        return {
            "local": "Xenova/ms-marco-MiniLM-L-6-v2",
            "cohere": "rerank-english-v3.0",
            "none": "",
        }[self.rerank_provider]

    def provider_configs(self) -> dict[str, ProviderConfig]:
        return {
            "groq": ProviderConfig(name="groq", base_url=self.groq_base_url, api_key=self.groq_api_key),
            "google": ProviderConfig(name="google", base_url=self.google_base_url, api_key=self.google_api_key),
        }

    def model_specs(self) -> list[ModelSpec]:
        """Registry from LLM_MODELS (JSON) or the built-in free-tier defaults."""
        raw: Any = DEFAULT_MODELS
        if self.llm_models.strip():
            try:
                raw = json.loads(self.llm_models)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM_MODELS is not valid JSON: {exc}") from exc
            if not isinstance(raw, list):
                raise ValueError("LLM_MODELS must be a JSON list of model objects")
        return [ModelSpec(**item) for item in raw]

    def validate_runtime(self) -> list[str]:
        """Return human-readable warnings; raises ValueError for unusable config."""
        warnings: list[str] = []
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        specs = {s.id for s in self.model_specs()}
        if self.llm_default_model not in specs:
            raise ValueError(f"LLM_DEFAULT_MODEL {self.llm_default_model!r} is not in the model registry")
        for model_id in self.fallback_chain:
            if model_id not in specs:
                warnings.append(f"LLM_FALLBACK_CHAIN entry {model_id!r} is not in the registry (ignored)")
        if not any(p.configured for p in self.provider_configs().values()):
            warnings.append("No LLM provider key set (GROQ_API_KEY / GOOGLE_API_KEY): chat will return 503")
        if not self.tavily_api_key:
            warnings.append("TAVILY_API_KEY not set: web search disabled")
        return warnings

    def masked(self) -> dict[str, Any]:
        """Settings snapshot safe to log (keys reduced to set/unset)."""
        data = self.model_dump()
        for key in list(data):
            if key.endswith("_api_key"):
                data[key] = "set" if data[key] else "unset"
        data.pop("llm_models", None)
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
