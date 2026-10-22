# AI Assistant Pro — low-latency Agentic RAG chatbot

A free-tier chatbot with **streaming answers**, **hybrid retrieval (BM25 + vectors + reranking)**,
**web search**, persistent chat history, and an automatic **model fallback chain** across Groq and
Google Gemini. Every model, threshold and limit is configured through environment variables.

| Capability | How it works |
|---|---|
| Chat | `POST /chat/stream` (SSE): route → retrieve ∥ search → prompt → streamed tokens → persisted |
| Models | Groq `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` · Google `gemini-3.7-flash`, `gemini-3.5-flash-lite`, `gemini-2.5-flash-lite`, `gemma-4-*` — all free tier, all env-configurable |
| Resilience | Per-model circuit breaker, quota guard, fail-over *before the first token*, hard-disable on 404 |
| Retrieval | SQLite FTS5 (BM25) ∥ FAISS cosine → Reciprocal Rank Fusion → cross-encoder rerank → relevance gate |
| Embeddings | Local ONNX (`BAAI/bge-small-en-v1.5`, `ms-marco-MiniLM-L-6`) — free, no quota; Cohere/Google switchable |
| Memory | Server-side history window (token-budgeted); follow-ups rewritten into standalone queries |
| Caching | Retrieval + web-search TTL caches, semantic answer cache (user/mode/doc-version scoped) |
| UI | Workspace dropdown (General · Documents · Research · Code), streaming chat with sources, My Documents panel, Settings, light/dark/system themes, responsive drawer/rail/desktop shell |
| MCP | Optional stdio MCP server exposing document search, web search and `ask` |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt          # or requirements.txt for runtime only
cp .env.example .env                          # add GROQ_API_KEY (and optionally GOOGLE_API_KEY, TAVILY_API_KEY)
uvicorn backend.main:app --port 8000          # UI + API on http://localhost:8000
```

`backend/main.py` is the single entry point and works from any directory:

```bash
uvicorn backend.main:app --port 8000   # project root (what the Procfile uses)
python -m backend.main                 # project root
python backend/main.py                 # anywhere
cd backend && python main.py           # from inside backend/
```

The first start downloads ~150 MB of ONNX models into `data/models/` (warm-up is non-blocking; until
they load, retrieval transparently uses BM25 only). Use **one worker** — caches and indexes are per-process.

## Architecture

```
backend/
├── main.py            entry point: app factory, lifespan, static files (the only top-level module)
├── api/               delivery   routes.py (HTTP + SSE) · mcp_server.py (stdio MCP)
├── services/          business   chat · retrieval · llm · documents · sessions · prompts · search
│   ├── chat_service.py        QueryRouter + ChatService (route → context → stream → persist)
│   ├── retrieval_service.py   dense ∥ BM25 → RRF → rerank → relevance gate
│   ├── llm_service.py         registry · discovery · breaker · quota · fail-over · streaming
│   ├── index_store.py         SQLite FTS5 + per-user FAISS (atomic, path-safe)
│   └── document_service.py · parser.py · embedding_service.py · rerank_service.py
│       search_service.py · prompt_builder.py · session_service.py
├── client/            external I/O (transport only, no policy)
│   ├── llm_client.py          pooled OpenAI-compatible SDK, headers, error normalisation
│   ├── search_client.py       Tavily        · web_client.py  SSRF-safe page fetcher
│   └── embedding_client.py · rerank_client.py · http.py
├── core/              application core
│   ├── config.py              Settings — the ONLY place environment variables are read
│   ├── container.py           ServiceContainer: composition root + FastAPI Depends getters
│   ├── database.py            SQLAlchemy engine, WAL, additive auto-migration
│   └── exceptions.py          AppError hierarchy (user-safe messages)
├── models/            api.py (pydantic schemas) · db.py (ORM)
└── utils/             pure helpers: text · ranking · numeric · timeutil · sse · net · streaming
                       cache (TTL + semantic) · rate_limiter (token bucket + quota guard)
```

Dependencies point downward only: `utils` ← `core` ← `client` ← `services` ← `api` ← `main`.
`tests/test_architecture.py` enforces the layer rules, forbids import cycles, and checks that
only `core/config.py` reads the environment or names a model id.

```
frontend (vanilla JS, SSE)  ──POST /chat/stream──►  api/routes ──► ChatService
                                                       ├─► RetrievalService ──► HybridIndexStore ◄── DocumentService
                                                       │        └─► EmbeddingService · RerankService ──► client/*
                                                       ├─► WebSearchService ──► client/search_client
                                                       ├─► PromptBuilder
                                                       ├─► LLMService ──► client/llm_client (Groq · Gemini)
                                                       └─► SessionService
```

Services are classes with constructor injection; `ServiceContainer.build(settings, overrides=…)`
is the single composition root (tests and the MCP server reuse it).

## Frontend

Plain HTML, CSS and JavaScript served by FastAPI from `frontend/` (no build step, no framework). The page loads
`marked`, `DOMPurify` and `highlight.js` from CDNs; everything else is local.

```
frontend/
├── index.html            shell, inline icon sprite (Lucide, ISC), dialogs; first-paint snippet applies theme,
│                         shell, workspace and preferences before any stylesheet paints (no flash)
├── css/                  main (tokens: light + dark) · components · reference (shell look) · workspace
│                         (workspace panels, My Documents, Settings) · animations · ai-viz · syntax
└── js/
    ├── api.js · app.js   the controller: State, every network call, streaming, uploads, sessions
    ├── ui.js · chat.js   the view facade the controller talks to, and message rendering
    ├── theme.js          light | dark | system
    ├── components/       menu · dialog · model-select · notifications · prefs · radiogroup · documents ·
    │                     workspaces · settings · motion · ai-viz   (view only: no fetch, no application state)
    └── vendor/           gsap.min.js, pinned by SHA-256 (see NOTICE-gsap.txt)
```

* **Views never fetch.** `app.js` owns state and the network; components render what they are given and report
  clicks through callbacks (`UI.onModeChange`, `UI.onMyDocuments`, `UI.onSettingsPane`, …). Tests enforce this.
* **Workspace mode** is one dropdown above *New Chat* in the sidebar; it is the only place the four modes appear.
* **Preferences** (text size, spacing, pop-up notices, Code language) live in this browser only (`pref.*`).
  Theme has its own key. Nothing is sent to the server.
* **Motion**: CSS for hover, focus and menus. GSAP is used only by `components/motion.js` for the workspace
  entrance and transition; it loads lazily and never loads under *reduced motion*. Its license
  (GreenSock "no charge") is not an OSI license: read `js/vendor/NOTICE-gsap.txt` before redistributing.
* Three.js is deliberately not used: the CSS/SVG assistant orb does the job without a 3D runtime.

### Chat request lifecycle

1. Rate-limit → persist the user turn → load a token-budgeted history window from the DB.
2. `QueryRouter` (regex heuristics, microseconds): `direct` · `rag` · `web` · `rag+web`.
   Documents are used whenever they are relevant (any mode); web search runs for time-sensitive
   queries when the toggle is on, and always in *Research* mode.
3. Follow-ups (“what about page 5?”) are rewritten to a standalone query by the fast model.
4. Semantic cache lookup for standalone, non-live queries.
5. Retrieval and web search run **in parallel**, each with its own timeout and graceful degrade.
6. Prompt is built (numbered context, citations) and streamed from the first healthy model.
7. Assistant message, citations, model and timings are persisted; `done` event closes the stream.

SSE events: `meta` · `status` · `citations` · `model` · `token` · `error` · `done`.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/chat/stream` | SSE chat `{session_id, message, mode, model?, web_search, regenerate?}` |
| POST | `/chat` | Same pipeline, one JSON response |
| POST | `/generate` | Plain completion with fallback (legacy) |
| GET | `/models[?refresh=1]` | Registry + availability for your keys |
| GET/POST/PATCH/DELETE | `/sessions[/{id}]` | Chat sessions (`user_id` query for listing) |
| POST | `/sessions/{id}/messages` | Append a message |
| POST | `/messages/{id}/feedback` | `{value: 1 \| -1 \| 0}` |
| POST | `/upload` | `file` (pdf, docx, csv, xlsx, txt, md) or `?url=` |
| GET / DELETE | `/documents`, `/documents/{id}` | List / remove (also drops index entries) |
| POST | `/retrieve` | Hybrid search `{query, top_k, threshold}` |
| POST | `/search` | Tavily web search |
| GET | `/health`, `/ping` | Component status (`healthy` · `degraded` · `unavailable`) |
| GET | `/config` | Read-only upload limits `{upload: {max_mb, extensions}}` the UI validates against (`/upload` enforces them again) |

Interactive docs: `http://localhost:8000/docs`.

## Configuration

All settings live in `.env` (see **`.env.example`** for every variable with defaults). Highlights:

* `LLM_DEFAULT_MODEL`, `LLM_FALLBACK_CHAIN`, `LLM_FAST_MODEL`, `LLM_MODELS` (JSON registry override)
* `EMBED_PROVIDER` / `RERANK_PROVIDER` — `local` (default), `google`, `cohere`, `none`
* `RAG_*`, `CHUNK_*`, `RERANK_*`, `TAVILY_*`, `HISTORY_*`, `SEMANTIC_CACHE_*`, `RATE_LIMIT_RPM_PER_USER`

Changing the embedding model re-embeds stored chunks automatically on the next start (chunk text is
kept in `data/index/rag.sqlite3`, so nothing needs re-uploading).

### Graceful degradation

| Missing / failing | Behaviour |
|---|---|
| One provider key | That provider's models are hidden; the chain uses the rest |
| No LLM keys | App boots; `/chat*` returns 503 with a clear message; `/health` = degraded |
| Tavily key / outage | Web branch skipped, reported in a `status` event |
| ONNX models unavailable | Documents indexed lexically (BM25) and re-embedded when they return |
| Reranker slow/failing | Fused (RRF) order, dense-cosine gate |
| Model 404 / rate limit / 5xx / timeout | Next model in the chain; 404 disables the model for an hour |

## Testing

```bash
pytest                       # ~330 offline tests: fakes for LLM, embeddings, Tavily, architecture + frontend guards
                             # (frontend browser scenarios use headless Chrome with a stubbed fetch and skip cleanly without it)
pytest -m live -s            # real Groq/Gemini/Tavily + real local models: matrix, quality, latency
ruff check backend tests --select F,E9,B --ignore B008,B905,B011
```

## MCP server (optional)

```bash
python -m backend.api.mcp_server                     # stdio
claude mcp add rag-chatbot -- $(pwd)/venv/bin/python -m backend.api.mcp_server
```

Tools: `search_documents`, `list_documents`, `web_search`, `ask`.

## Measured performance (8-core CPU dev machine, free-tier APIs)

| Step | Typical |
|---|---|
| Query embedding (local) | ~10–35 ms |
| Dense + BM25 search (3k chunks) | ~10 ms |
| Rerank 8 candidates (local) | ~100–130 ms |
| First token — direct / RAG | ~0.35–0.55 s / ~0.4–1.1 s |
| First token — cached answer | < 10 ms |
| Web search (Tavily basic) | ~1.6–3.7 s (dominates web-route latency) |

Numbers vary with provider load; the live suite prints a fresh report (`pytest -m live -s`).

## Notes

* `user_id` is client-supplied (no authentication). Add auth before exposing this publicly.
* Groq's free tier limits tokens per minute *per model*; the fallback chain spreads load across models.
* Reusable task recipes (add a model, switch embeddings, tune retrieval, …) are in **`skills.md`**.
