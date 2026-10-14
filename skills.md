# skills.md — reusable task recipes for this project

Short, tested runbooks. Each one lists *what to touch*, *how to verify*, and *pitfalls*.
Project root = this directory. Activate the venv first: `source venv/bin/activate`.

---

## 1. Add or change a chat model (config only)

**Touch:** `.env` only (code never hard-codes ids).

1. Find the exact id: `curl -s localhost:8000/models?refresh=1 | jq '.models[].id'` (provider catalog is
   queried with your keys) or the provider console.
2. Add it to the registry via `LLM_MODELS` (a JSON list **replaces** the built-ins, so include the ones you keep):
   ```
   LLM_MODELS=[{"id":"openai/gpt-oss-20b","label":"GPT-OSS 20B","provider":"groq",
                "extra":{"reasoning_effort":"low"},"rpm":30,"tpm":8000}]
   ```
   Fields: `provider` (`groq`|`google`), `supports_system` (false → system prompt merged into the first
   user turn), `extra` (request params, e.g. `reasoning_effort`, `temperature`), `rpm`/`tpm` (quota guard).
3. Put it in `LLM_DEFAULT_MODEL` and/or `LLM_FALLBACK_CHAIN` (comma list, fastest/most reliable first).
4. Verify: `pytest -m live -s -k matrix` prints a per-model table (ok / fell back / unavailable).

**Pitfalls:** an id that exists in the catalog can still 404 for your key ("no longer available to new
projects") — the service hard-disables it for an hour and the UI greys it out. Preview models
(e.g. `qwen/qwen3.8-27b`) can vanish; keep them late in the chain.

## 2. Add a new LLM provider

Any OpenAI-compatible endpoint works. In `backend/core/config.py`: add `<name>_api_key` and
`<name>_base_url` to `Settings`, add the provider to `provider_configs()`, and (optionally) add default
`DEFAULT_MODELS` entries. `backend/client/llm_client.py::build_provider_clients` creates one pooled
client per configured provider automatically — no other code changes.
Verify with recipe 1 step 4. Add the key to `.env.example`.

## 3. Switch embedding / rerank provider (and reindex)

1. Set `EMBED_PROVIDER` (`local|google|cohere`), optionally `EMBED_MODEL`; same for `RERANK_*`.
2. Recalibrate the cosine gate for the new model (recipe 6): `RAG_COSINE_FLOOR/CEIL`.
3. Restart. `DocumentService.reindex_if_needed()` runs in warm-up: stored chunk text is re-embedded and the
   FAISS index rebuilt (`data/index/`); no re-upload needed. Log line: `Embedding model changed … reindexing`.
4. Verify: `GET /health` → `details.index.embed_model`, `unembedded: 0`; `pytest -m live -s -k retrieval_quality`.

**Pitfall:** vectors of different dimensions are never mixed — a model switch always rebuilds per user.

## 4. Support a new document format

1. `backend/services/parser.py`: add `_parse_<fmt>(path) -> list[{page_number, text, metadata}]`,
   register the extension in `SUPPORTED_EXTENSIONS` and the `handlers` dict in `parse_file`.
   Raise `DocumentParseError("user-safe message")` for unreadable input; return `[]` for "no text".
2. Add the extension to `ALLOWED` in `frontend/js/app.js` and the `accept=` list in `frontend/index.html`.
3. Add a unit test in `tests/test_units.py` (see `test_parser_text_csv_and_errors`) and an upload test in
   `tests/test_api.py`.

## 5. Add or edit an agent mode / system prompt

1. `backend/core/config.py` → `MODES` tuple; `backend/services/prompt_builder.py` → `MODE_PROMPTS[<mode>]`.
2. Routing rules (when docs/web are used) live in `QueryRouter.decide` in `chat_service.py`.
3. UI: add a `.mode-tab[data-mode="<mode>"]` button in `frontend/index.html`.
4. Test: extend `test_router_decisions` and a `test_chat_*` case in `tests/test_api.py`.

## 6. Tune retrieval quality / thresholds

1. Put 5–10 labelled queries (incl. 3 that should return nothing) in `tests/test_live.py::QUERIES`.
2. `pytest -m live -s -k retrieval_quality` → prints hits@1 for hybrid / dense / lexical + each relevance.
3. Adjust, in order: `RAG_COSINE_FLOOR/CEIL` (dense gate), `RAG_MIN_RELEVANCE` (global gate),
   `CHUNK_SIZE/CHUNK_OVERLAP` (re-upload docs after changing), `RAG_TOP_K`, `RERANK_CANDIDATES`.
4. Relevance = `max(rerank_prob, calibrated_cosine, lexical_coverage)` — see `RetrievalService._search`.
   Cross-encoders saturate to ~0 on vague paraphrases; the cosine/lexical signals rescue those.

## 7. Latency benchmark

`pytest -m live -s -k "latency or tavily"` prints: retrieval p50/p95 on 1 000 chunks, TTFT for
direct/RAG/cached/web routes, Tavily basic vs advanced. Per-request numbers are also in every SSE
`done` event (`timings`) and persisted in `messages.meta`.

Levers (all env): `RERANK_CANDIDATES`↓, `RERANK_MAX_CHARS`↓, `RERANK_ENABLED=false`,
`TAVILY_SEARCH_DEPTH=basic`, `LLM_FALLBACK_CHAIN` order, `LLM_MAX_TOKENS`, `HISTORY_TOKEN_BUDGET`.

## 8. Diagnose a failing / slow stream

1. `curl -N -X POST localhost:8000/chat/stream -H 'content-type: application/json' \
      -d '{"session_id":"<id>","message":"hi"}'` — read `meta`, `status`, `error` events.
2. `error.code`: `all_models_failed` (see `attempts`), `rate_limited`, `empty_message`, `session_not_found`.
3. `GET /health` → `details.llm.models.<id>.tripped` shows breakers; `details.llm.quota` shows 60 s usage.
4. Nothing streams in the browser behind a proxy → disable buffering (`X-Accel-Buffering: no` is already set).
5. Slow first token: compare `status.generating.first_token.elapsed_ms` (provider) with `retrieve_ms`/`search_ms`.

## 9. Full verification (run before every release)

```bash
ruff check backend tests --select F,E9,B --ignore B008,B905,B011   # undefined names / unused imports
python -m compileall -q backend
pytest                                                              # offline suite + architecture guards
pytest -m live -s                                                   # real providers + models
for f in frontend/js/*.js; do node --check "$f"; done
```
`tests/test_architecture.py` fails the build if a layer imports upward (`utils` ← `core` ← `client` ←
`services` ← `api`), if module-level imports form a cycle, if anything but `core/config.py` reads
`os.environ` or names a model id, or if `backend/` gains a top-level module other than `main.py`.

## 10. Add an endpoint the project way

1. External I/O (a new API, SDK or file source) → an adapter in `backend/client/`; no policy there.
2. Logic → a method on the relevant service class in `backend/services/` (no FastAPI imports there).
   Pure helpers go in `backend/utils/`; shared exception types in `backend/core/exceptions.py`.
3. Wire any new collaborator in `ServiceContainer.__init__` (`backend/core/container.py`) and add a `get_*` getter.
4. Handler in `backend/api/routes.py` using `Depends(get_*)`; request/response models in `models/api.py`.
5. Tests: service-level with fakes (`tests/conftest.py`), plus an HTTP test via `client_factory`;
   `tests/test_architecture.py` will tell you if a dependency points the wrong way.

## 11. Inspect or repair the index

```bash
sqlite3 data/index/rag.sqlite3 "select user_id, count(*) from chunks group by 1;"
sqlite3 data/index/rag.sqlite3 "select * from index_meta;"        # embed_model in use
sqlite3 data/index/rag.sqlite3 "select count(*) from chunks where embedded=0;"   # pending vectors
```
To rebuild vectors for everyone: delete `data/index/faiss/` and set `embedded=0` (or just change
`EMBED_MODEL`) and restart — warm-up re-embeds from the stored chunk text.

## 12. Register the MCP server with Claude Code

```bash
claude mcp add rag-chatbot -- "$(pwd)/venv/bin/python" -m backend.api.mcp_server
```
Tools: `search_documents(query, user_id, top_k)`, `list_documents`, `web_search`, `ask`. It shares `.env`
and `data/` with the web app (stop the web app or use a different `DATA_DIR` for concurrent heavy use).
