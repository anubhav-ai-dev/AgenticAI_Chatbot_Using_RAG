"""
Central configuration — single source of truth for all env vars and constants.
Every other module imports from here; no module calls os.getenv() directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = BASE_DIR / "frontend"

# ── API keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str | None       = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None     = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY: str | None     = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY: str | None     = os.getenv("GOOGLE_API_KEY")
# Optional — users who have paid accounts can add these
OPENAI_API_KEY: str | None     = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None  = os.getenv("ANTHROPIC_API_KEY")

# ── Server ────────────────────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# ── CORS ──────────────────────────────────────────────────────────────────────
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _origins_env.split(",") if o.strip()]

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL") or f"sqlite:///{DATA_DIR / 'chatbot.db'}"

# ── Cohere ────────────────────────────────────────────────────────────────────
COHERE_EMBED_MODEL: str  = "embed-english-v3.0"
COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
COHERE_RERANK_TOP_N: int = 5

# ── FAISS / RAG ───────────────────────────────────────────────────────────────
FAISS_DIR: Path  = DATA_DIR / "faiss"
FAISS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE: int            = 800
CHUNK_OVERLAP: int         = 200
RETRIEVAL_INITIAL_K: int   = 20   # candidates before reranking
RETRIEVAL_FINAL_K: int     = 5    # docs returned after reranking
DEFAULT_THRESHOLD: float   = 0.40

# ── Tavily ────────────────────────────────────────────────────────────────────
TAVILY_MAX_RESULTS: int     = 5
TAVILY_SEARCH_DEPTH: str    = "advanced"   # "basic" | "advanced"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.1

# Models available via backend (Groq — free, fast)
GROQ_MODELS: list[dict] = [
    {"id": "llama-3.3-70b-versatile",         "label": "Llama 3.3 70B",     "provider": "groq"},
    {"id": "deepseek-r1-distill-llama-70b",   "label": "DeepSeek R1 70B",   "provider": "groq"},
    {"id": "llama3-70b-8192",                  "label": "Llama 3 70B",       "provider": "groq"},
]

# Models available via Puter.js (browser-side, free)
PUTER_MODELS: list[dict] = [
    {"id": "claude-sonnet-4-5",                        "label": "Claude Sonnet 4.5",     "provider": "puter"},
    {"id": "claude-opus-4",                            "label": "Claude Opus 4",          "provider": "puter"},
    {"id": "gpt-4o",                                   "label": "GPT-4o",                 "provider": "puter"},
    {"id": "gpt-4.1",                                  "label": "GPT-4.1",                "provider": "puter"},
    {"id": "google/gemini-2.5-pro",                    "label": "Gemini 2.5 Pro",         "provider": "puter"},
    {"id": "google/gemini-2.0-flash",                  "label": "Gemini 2.0 Flash",       "provider": "puter"},
    {"id": "deepseek-ai/DeepSeek-R1",                  "label": "DeepSeek R1",            "provider": "puter"},
    {"id": "deepseek-ai/DeepSeek-V3",                  "label": "DeepSeek V3",            "provider": "puter"},
    {"id": "meta-llama/llama-4-maverick",               "label": "Llama 4 Maverick",       "provider": "puter"},
    {"id": "x-ai/grok-2-latest",                       "label": "Grok 2",                 "provider": "puter"},
]

ALL_MODELS: list[dict] = GROQ_MODELS + PUTER_MODELS

# ── Agent modes ───────────────────────────────────────────────────────────────
AGENT_MODES: list[dict] = [
    {"id": "general",   "label": "General Chat",    "icon": "💬"},
    {"id": "document",  "label": "Document Expert", "icon": "📄"},
    {"id": "research",  "label": "Deep Research",   "icon": "🌐"},
    {"id": "code",      "label": "Code & Data",     "icon": "🧮"},
]

# ── Session memory limit ──────────────────────────────────────────────────────
SESSION_HISTORY_LIMIT: int = 20  # messages loaded into LLM context
