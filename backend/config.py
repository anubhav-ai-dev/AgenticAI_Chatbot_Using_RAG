"""
Central configuration: environment variables, constants, allowed model names.
All other modules import from here — never call os.getenv() directly elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")

# ── Server ────────────────────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# ── Allowed models ────────────────────────────────────────────────────────────
ALLOWED_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "gpt-4o-mini",
]

# ── RAG / chunking defaults ───────────────────────────────────────────────────
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 200
RETRIEVAL_K: int = 5          # final docs returned after reranking
RETRIEVAL_INITIAL_K: int = 20  # candidates fetched before reranking
DEFAULT_SIMILARITY_THRESHOLD: float = 0.5

# ── Memory ────────────────────────────────────────────────────────────────────
SESSION_HISTORY_LIMIT: int = 10  # messages kept per session

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1
TAVILY_MAX_RESULTS: int = 2

# ── Cohere model IDs ──────────────────────────────────────────────────────────
COHERE_EMBED_MODEL: str = "embed-english-v3.0"
COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
