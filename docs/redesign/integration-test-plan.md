# Integration test plan: real backend, isolated environment

Status: PLAN ONLY. Nothing in this document has been executed. No backend was started, no request was sent, no LLM or external service was contacted while writing it. Every step below needs explicit user approval (section 10) before it runs.

Baseline reviewed: working tree on `main` at `15e1d81` plus uncommitted redesign changes. File:line citations refer to that tree.

---

## 0. What the code allows (facts the plan rests on)

| Fact | Evidence |
|---|---|
| Only `backend/core/config.py` reads env vars; `.env` is read from `BASE_DIR/.env` (repo root, derived from the file location), env vars override it, blank values ignored | `config.py:1-5`, `:21`, `:106-112` |
| LLM provider endpoints are configurable without code changes: `GROQ_BASE_URL`, `GOOGLE_BASE_URL` | `config.py:121-122`, `:249-253` |
| A provider is only built if its API key is non-empty; the key can be any dummy string | `config.py:98-100`, `llm_client.py:60-66` |
| Chat transport is the `openai` SDK `AsyncOpenAI(base_url=..., http_client=shared httpx client, max_retries=0)` against `/chat/completions` (streaming) and `/models` (discovery) | `llm_client.py:60-66`, `:55-57` |
| Storage root: `DATA_DIR` (default `<repo>/data`); SQLite `DATABASE_URL` (default `<data>/chatbot.db`); index dir `<data>/index` containing `rag.sqlite3` (BM25/vectors); ONNX model cache `<data>/models` | `config.py:193-194`, `:199-219`, `index_store.py:92`, `container.py:47,51` |
| No separate uploads directory: uploads are parsed from a `tempfile.mkstemp` file in the OS temp dir and deleted immediately; only chunks are stored | `document_service.py:66-76` |
| Per-user throttle: token bucket keyed `"<client ip>|<user_id>"`, capacity = `RATE_LIMIT_RPM_PER_USER` (min 1); 429 + `Retry-After` is raised before the stream starts (`/chat/stream`, `/chat`) | `config.py:184`, `routes.py:172-179`, `:230`, `rate_limiter.py:35-60` |
| `/chat/stream` pre-stream HTTP errors: 503 when no provider key, 404 unknown session, 422 pydantic validation; 429 as above. `/generate` maps `LLMError` to 503/429/502 with Retry-After | `routes.py:231-235`, `:182-190` |
| In-stream errors are SSE `error` events (HTTP stays 200): `session_not_found`, `empty_message`, `nothing_to_regenerate`, LLM codes (`all_models_failed`, `rate_limited`, `timeout`, ...) with `retryable`, `attempts`, `retry_after`; mid-stream failure adds `partial` and `message_id` | `chat_service.py:136,147,153,257-259,288-291` |
| Event order: `meta`, optional `status` (rewriting/retrieving/searching), optional `citations`, `status generating start`, `status first_token`, `model`, `token`*, `done` (or `error`) | `chat_service.py:163-299` |
| Stop: client disconnect cancels the generator; `finally` closes the upstream stream and persists any partial text with `stopped:true, partial:true` (if some text was produced). The user turn is persisted BEFORE generation, so it survives a stop | `chat_service.py:155`, `:300-307` |
| Regenerate: `pop_last_exchange` removes the last assistant (and user) turn then re-asks; error `nothing_to_regenerate` if none | `chat_service.py:144-149` |
| Fallback happens only before the first token, across `LLM_FALLBACK_CHAIN`; the `model` event carries `fell_back_from`; persisted in message meta | `llm_service.py:234-279`, `chat_service.py:263,277-283` |
| Breaker: rate-limit trips the model for `max(Retry-After, 5)` capped at 300 s (soft); model 404/403 trips for `MODEL_UNAVAILABLE_COOLDOWN_S` (default 3600, hard, makes `/models` report `available:false`); 2 consecutive other failures trip `BREAKER_COOLDOWN_S` (30). Pass 2 ignores soft cooldowns if nothing was attempted | `llm_service.py:176-188`, `:343-359`, `config.py:136-137` |
| Timeouts: `LLM_CONNECT_TIMEOUT_S` 5, `LLM_FIRST_TOKEN_TIMEOUT_S` 12, `LLM_TOTAL_TIMEOUT_S` 90 | `config.py:132-134`, `llm_service.py:292-293` |
| Discovery (`GET /models?refresh=true` or at startup): a model not in the provider's `/models` list is `available:false`; discovery failure means "assume available" | `llm_service.py:201-220`, `:181-183`, `routes.py:213-219` |
| Upload: allowed extensions pdf, docx, csv, xlsx, txt, md, rst; unsupported -> 415, empty -> 400, parse failure/no text -> 422, over limit -> 413 (checked twice), > 5000 chunks -> 413. Response includes `indexing: "hybrid"` or `"lexical"` | `parser.py:247,260-271`, `document_service.py:59-82,111-113`, `routes.py:354-370` |
| Re-upload of same filename replaces the old document | `document_service.py:150-161` |
| URL ingestion SSRF guard: http(s) only, every resolved address rejected if private/loopback/link-local/reserved/multicast/unspecified, re-checked on each of up to 5 redirects, 5 MB cap, content-type allow-list. Bypass flag `ALLOW_PRIVATE_URLS` (default false) | `net.py:38-52`, `web_client.py:22-53`, `config.py:186`, `container.py:55` |
| CORS middleware has no `expose_headers`; `allow_origins = ALLOWED_ORIGINS` (default `http://localhost:8000`) | `main.py:72-78`, `config.py:190` |
| Frontend base URL: `window.CONFIG.BACKEND_URL`, default `""` = same origin. Nothing in `index.html` or `frontend/js/*.js` defines `window.CONFIG`; `frontend/js/config.js` is gitignored and not referenced by the page | `api.js:6-8`, `.gitignore` entry for `frontend/js/config.js`, grep of `index.html` |
| Static frontend is always served from `<repo>/frontend` (`FRONTEND_DIR`), not configurable | `config.py:22`, `main.py:81-85` |
| Existing tests: fully offline via `ServiceContainer(overrides=...)`, `FakeProviderClient`, `settings` fixture with `tmp_path` `data_dir`; `pytest -m live` (marker in `pytest.ini`) hits real endpoints | `tests/conftest.py:195-241`, `pytest.ini:3-6`, `tests/test_live.py:1-4,56-60` |
| Authorization gap: session GET/PATCH/DELETE and feedback do not check `user_id`; `user_id` is client-chosen | `routes.py:301-343`; `docs/redesign/security-issue-authz.md` |

## 1. Isolation design

### 1.1 Environment shape
Run on a disposable copy, not the developer's checkout:

1. `mkdir -p /tmp/aiap-it/{app,data,run,evidence}` (new empty dir; mode 700).
2. Copy the code under test into `/tmp/aiap-it/app`: `backend/`, `frontend/` (the uncommitted redesign must be included; a `git worktree` of `HEAD` would NOT contain it, so use a plain copy such as `rsync -a --exclude data --exclude .env --exclude venv --exclude server.log --exclude .git`). Record `git rev-parse HEAD` plus `git diff HEAD | sha256sum` as the build identifier.
3. Reuse the existing `venv` read-only (invoke `<repo>/venv/bin/python` with `cwd=/tmp/aiap-it/app`), or build a fresh venv from `requirements.txt` if the user prefers (needs package downloads, so would need approval).
4. Do NOT copy `.env`. Because `Settings` resolves `.env` as `<copy>/.env` (`config.py:21,107`), the copy has none; also launch with `env -i` plus an explicit variable list so no ambient `GROQ_API_KEY` etc. leaks in from the shell.

### 1.2 Settings that isolate state (all via environment variables)

| Variable | Value | Effect | Citation |
|---|---|---|---|
| `DATA_DIR` | `/tmp/aiap-it/data` | Moves SQLite DB, `index/rag.sqlite3` and `models/` | `config.py:199-219` |
| `DATABASE_URL` | `sqlite:////tmp/aiap-it/data/chatbot.db` (explicit, belt and braces) | DB location | `config.py:217-219` |
| `BACKEND_HOST` / `BACKEND_PORT` | `127.0.0.1` / `18000` (non-default; fake LLM on `18001`) | Bind loopback only, avoid the default `0.0.0.0:8000` | `config.py:190-191`, `main.py:100-104` |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:18000` | CORS for same-origin page | `config.py:192` |
| `GROQ_API_KEY` | `fake-key-not-real` | Enables the groq provider without a real secret | `config.py:98-100` |
| `GOOGLE_API_KEY` | unset (or a second fake key if a Google-provider fallback is tested) | | |
| `TAVILY_API_KEY`, `COHERE_API_KEY` | unset | Web search disabled (`validate_runtime` warns), no Tavily call possible; Cohere not used because `EMBED_PROVIDER`/`RERANK_PROVIDER` stay `local` | `config.py:280-281`, `:140,143` |
| `GROQ_BASE_URL` | `http://127.0.0.1:18001/openai/v1` | Points chat and `/models` discovery at the fake | `config.py:121,251` |
| `LLM_MODELS` | small JSON registry, e.g. 4 groq-provider models `fake/a`, `fake/b`, `fake/c`, `fake/d`, `rpm`/`tpm` unset | Deterministic chain; avoids `DEFAULT_MODELS` names | `config.py:129,255-265` |
| `LLM_DEFAULT_MODEL`, `LLM_FALLBACK_CHAIN`, `LLM_FAST_MODEL` | `fake/a`, `fake/a,fake/b,fake/c`, `fake/c` | Must exist in `LLM_MODELS` or startup raises | `config.py:272-277` |
| `RATE_LIMIT_RPM_PER_USER` | `1000` normally; `2` in the dedicated throttle run | 429 + Retry-After test | `config.py:184` |
| `LLM_FIRST_TOKEN_TIMEOUT_S`, `LLM_TOTAL_TIMEOUT_S`, `BREAKER_COOLDOWN_S` | 3, 20, 6 (shortened to keep the suite fast) | Timeout/cooldown cases finish in seconds | `config.py:133-136` |
| `SEMANTIC_CACHE_ENABLED` | `false` for streaming/stop/fallback cases; `true` in one dedicated cache case | A repeated standalone question otherwise replays from cache with 48-char chunks and `cached:true` and never reaches the fake | `chat_service.py:180-189,334-347` |
| `MODEL_DISCOVERY` | `true` (fake serves `/models`) | Needed for the "unavailable model" case | `config.py:135` |
| `ALLOW_PRIVATE_URLS` | `false` (default) for SSRF run; `true` only in a separate, later run for positive URL ingestion against a local test page | | `config.py:186` |
| `MAX_UPLOAD_MB` | `1` | Makes the 413 case cheap (1 MiB + 1 byte instead of 20 MiB) | `config.py:185`, `routes.py:360-364` |
| `LOG_LEVEL` | `INFO`; log to `/tmp/aiap-it/run/backend.log` | Config is logged masked (`masked()` reduces keys to set/unset), prompts are not logged | `config.py:284-291` |

Embeddings / rerank: `EMBED_PROVIDER=local` (default) uses fastembed ONNX models from `<DATA_DIR>/models` and downloads from the internet on first use if absent (`embedding_client.py:26,108`). Two options, user decides (section 10):
- (a) copy (not symlink-write) the already-downloaded `data/models/` read-only into `/tmp/aiap-it/data/models` (`tests/test_live.py:57-60` symlinks it the same way). No network. Preferred.
- (b) leave models absent and run offline: `warmup()` fails and uploads index as `indexing:"lexical"` (`document_service.py:138-148`); retrieval is then BM25 only. Valid for most cases but the "hybrid" indexing assertion (UP-12) is skipped.

### 1.3 Separate user_id namespace
Every test uses `user_id` of the form `it-<runid>-<name>` (max 128 chars, `models/api.py:16`). Sessions, documents and index rows are keyed by it (`session_service.py:40`, `document_service.py:167`). With a fresh DB this is belt and braces, and it makes cross-user authorization tests (section 3) unambiguous. Never use `default` (the value the shipped frontend uses when no user is set).

### 1.4 What is NOT isolatable (must be accepted or controlled otherwise)
1. The process-wide rate limiter key includes the client IP (`routes.py:174`), so all tests from one machine share the "ip" part; mitigated by a distinct `user_id` per test.
2. `FRONTEND_DIR` is fixed relative to the code location (`config.py:22`); isolation depends on the copy, not on a setting.
3. In-memory state (breaker, quota guard, semantic cache, rate limiter, retrieval cache) lives for the process lifetime; tests must either restart the backend between groups or use unique models/prompts. Plan: restart between groups G1..G8 (section 3).
4. `BASE_DIR/.env` is read if present. Isolation relies on running from the copy (no `.env`) and `env -i`; verify at pre-flight with the masked config log line (`container.py:87`).
5. The fastembed/HF model download (if option (a) is not taken) is an external network call that cannot be pointed at a fake.
6. Tavily base URL is configurable (`config.py:123`) but we simply leave the key unset; web-search behaviour (`web_search` toggle with no key) is covered only as "disabled" (see SE-1). A fake Tavily could be added later; out of scope.
7. SSRF guard resolves names with the OS resolver (`net.py:45`); use IP literals only to avoid any DNS traffic.
8. The unauthenticated, wildcard-capable API surface (`/sessions`, `/documents`, `/retrieve`) is exposed on the chosen port to anything that can reach it; mitigate by loopback bind and the network allow-list (section 4).

## 2. LLM strategy

### 2.1 Preferred: local scripted fake provider (no code changes needed)
The backend can be pointed at a fake WITHOUT backend code changes: `GROQ_BASE_URL` (`config.py:121`) goes straight into `AsyncOpenAI(base_url=...)` (`llm_client.py:63`). The fake must implement, on loopback only:

- `GET /openai/v1/models` -> `{"data":[{"id":"fake/a"},...]}` (discovery; a subset hides models to test `available:false`).
- `POST /openai/v1/chat/completions` with `stream:true` -> SSE `data: {...}` chunks in the OpenAI delta shape (`choices[0].delta.content`, final chunk with `usage`) and `data: [DONE]`. `stream_options`/`extra_body` fields (`reasoning_effort`) are accepted and ignored. A non-streaming branch is not needed (`LLMService.complete` uses `open_stream`, see `llm_service.py:379+`; confirm during build).
- Scenario selection WITHOUT a side channel: the fake inspects `model` and the last user message for a marker such as `[[sc:ok]]`, `[[sc:429:7]]` (429 with `Retry-After: 7`), `[[sc:503]]`, `[[sc:404]]` (model gone), `[[sc:slow:4000]]` (delay before first token), `[[sc:midfail:3]]` (error/abort after 3 chunks), `[[sc:hang]]`, `[[sc:empty]]`, `[[sc:think]]` (`<think>` block), `[[sc:long:N]]` (N chunks at a fixed cadence for Stop tests), `[[sc:headers]]` (x-ratelimit-* headers to exercise the quota guard, `rate_limiter.py:504-515`). Per-model overrides come from a small control endpoint on the fake (`POST /__script`) bound to the same loopback port; this is a request to the FAKE, not to the system under test.
- Every received request is appended to `/tmp/aiap-it/evidence/fake-requests.jsonl` (model, marker, timestamp, whether the client closed the connection early). This is the ground truth for "server stopped consuming the upstream" in the Stop tests.
- Implementation: ~150 lines Python (`aiohttp`/FastAPI+uvicorn already in the venv; verify) kept OUTSIDE the repo (`/tmp/aiap-it/run/fake_llm.py`). Not committed unless the user asks.

Limitations to state in results: the fake validates our client logic, SSE plumbing, failover and persistence. It does not validate real Groq/Gemini behaviours (real rate-limit header semantics, reasoning field, model names).

Mid-stream failure fidelity: the SDK maps a truncated/aborted connection to `APIConnectionError`/`httpx.TransportError`, which `classify_exception` turns into `network` (`llm_client.py:109-134`); to produce an SSE-level `error` object from the provider use `data: {"error":{...}}` and check how the SDK handles it. Both variants are listed in the matrix.

### 2.2 If a fake were NOT sufficient (decision for the user)
Nothing requires a backend code change for the fake. Optional code changes the user may want (each needs a separate decision and is out of scope for the QA/frontend agents, ownership: backend developer):
- Add `expose_headers=["Retry-After"]` to `CORSMiddleware` (`main.py:72-78`). Only matters when the page and API are on different origins; same-origin hosting (the default) is unaffected. Plan tests both ways (EH-5).
- Per-provider fake switch is not needed; `GROQ_BASE_URL` already exists.
- A fake Tavily would use `TAVILY_BASE_URL` (`config.py:123`); no change needed.

### 2.3 Optional tier 2: real provider, only with explicit approval
Only after tier 1 is green, and only if the user wants a real-provider smoke test.
- Requires: explicit written approval, a dedicated throwaway key (never the developer's `.env` key), key supplied through the process environment of the test run only (never written to the repo, logs, evidence, or shell history; use `read -s` or a secret file with mode 600 in `/tmp/aiap-it/run`, deleted afterwards), and a revoked/rotated key afterwards.
- Budget cap: free-tier models only (as in `DEFAULT_MODELS`, `config.py:44-85`), at most 20 requests total, `LLM_MAX_TOKENS=64`, prompts of at most 20 words ("Reply with the single word: pong"), `RATE_LIMIT_RPM_PER_USER=5`, `FALLBACK_CHAIN` limited to one model so a failure cannot fan out into extra calls. No document content, no personal data in prompts. Query rewrite (extra LLM call per follow-up, `chat_service.py:356-377`) disabled with `QUERY_REWRITE_ENABLED=false`.
- Cases: RL-1 (one streamed answer, event order, ttft), RL-2 (Stop after first tokens), RL-3 (`/models?refresh=true` availability, one list call), nothing else. 429/5xx behaviour is NOT provoked against the real service.
- Abort condition: any HTTP 401/403/429 from the provider, spend or request count reaching the cap, or any unexpected outbound host.

## 3. Test matrix

Conventions: "API" = scripted HTTP client from a Python script (stdlib `urllib`/`httpx` from the venv, against `127.0.0.1:18000`, `GET`-safe except where throwaway data is created); "UI" = headless Chrome driven via CDP (section 4). Every case runs only against the isolated environment. Grouped so the backend is restarted between groups (state, section 1.4 #3). Expected results cite the code that defines them.

### G1 SSE streaming (fake `ok`, cache off)
| ID | Case | Expected |
|---|---|---|
| ST-1 | `POST /chat/stream` simple prompt | 200, `text/event-stream`, headers `Cache-Control: no-cache`, `X-Accel-Buffering: no` (`routes.py:241-242`); events in order `meta`, `status generating start`, `status first_token`, `model`, `token`+, `done` (`chat_service.py:163-299`) |
| ST-2 | Event ordering with RAG route (after UP-8) | `meta` -> `status retrieving start` -> `status retrieving done` -> `citations` -> generating... -> `done`; `done.route` is `rag` |
| ST-3 | Chunking/incrementality | Tokens arrive over time (fake cadence 50 ms x 20): first `token` arrives well before `done` (timestamps captured client-side), no single buffered flush |
| ST-4 | `[[sc:think]]` | `<think>` text is never emitted in `token` events (`streaming.py`, `llm_service.py:84-90`) |
| ST-5 | Wire format | Each frame is `event: X\ndata: <json>\n\n`; non-ASCII preserved (`sse.py:9-12`); a payload with `\n` inside text is JSON-escaped (no frame injection) |
| ST-6 | Persistence | After `done`, `GET /sessions/{id}` shows user turn + assistant turn with `meta` (model, route, timings) and `done.persisted:true`, `message_id` set |
| ST-7 | UI render | In the real frontend: bubble grows token by token, thought-steps/status chips appear, model badge shown, no console errors, `aria-live` not per-token (QA checklist D) |
| ST-8 | Cache path (cache on) | Same standalone question twice: second is `cached:true`, tokens in 48-char slices, fake receives only 1 request (`chat_service.py:334-347`) |
| ST-9 | `POST /chat` (non-stream) | Same content aggregated; error mapping 404/400/429/502 (`routes.py:255-259`) |

### G2 Stop and cancellation (fake `[[sc:long:200]]`)
| ID | Case | Expected |
|---|---|---|
| SP-1 | API client closes the socket after ~5 tokens | Backend log shows no exception storm; fake log shows its client disconnected early within ~1 s (upstream closed by `aclose`, `chat_service.py:301`); no further requests |
| SP-2 | Persistence after stop | `GET /sessions/{id}`: user turn present; assistant partial present with `meta.stopped:true, partial:true` (`chat_service.py:302-307`) - exactly one assistant message |
| SP-3 | Stop before first token (`[[sc:slow:4000]]`, abort at 1 s) | User turn persisted, no assistant message (no `parts`), session usable; next send works; documents: this leaves a user turn without answer (relevant to M1) |
| SP-4 | UI Stop button | Click Stop mid-stream: `AbortController` aborts, button returns to Send, composer enabled/focused, partial text stays visible, no spinner stuck, no error toast |
| SP-5 | After Stop, reload page | Partial reappears from server history; "stopped" not rendered as an error |
| SP-6 | Stop twice / Stop after `done` | No error, no duplicate message |
| SP-7 | Rapid Stop->Send | New stream starts cleanly; previous stream's late events do not append to the new bubble |
| SP-8 | Client disconnect while retrieval running (RAG question) | `asyncio.CancelledError` path cancels sub-tasks (`chat_service.py:226-229`); no orphan tasks (backend stays responsive, `/health` fast) |

### G3 Regenerate
| ID | Case | Expected |
|---|---|---|
| RG-1 | `regenerate:true` after a complete exchange | Previous assistant answer replaced; transcript has no duplicate user turn (`pop_last_exchange`, `chat_service.py:144-149`) |
| RG-2 | `regenerate:true` on an empty session | `error` event `nothing_to_regenerate`, `retryable:false` |
| RG-3 | Regenerate bypasses cache | Fake receives the request even for a previously cached question (`chat_service.py:181`) |
| RG-4 | UI regenerate button | Behaves as RG-1; a message with feedback loses it (document actual behaviour) |
| RG-5 | Retry after a stream error (QA M1 flow) | Inline Retry sends `regenerate:true`; server transcript contains exactly one user turn (see DR-3) |

### G4 Model fallback and model list
| ID | Case | Expected |
|---|---|---|
| FB-1 | `fake/a` = `[[sc:503]]`, `fake/b` ok | Answer served by `fake/b`; `model` event `fell_back_from:["fake/a"]`; `done.fell_back_from` same; persisted meta has it (`chat_service.py:263,280`); UI shows fallback badge |
| FB-2 | `fake/a` 429 with `Retry-After: 7` | Falls back to `fake/b`; `fake/a` skipped for next requests during cooldown (>= 7 s, `llm_service.py:346`), `attempts` shows `skipped` once the cooldown is active; after cooldown `fake/a` is tried again (time-bounded test, use `BREAKER_COOLDOWN_S=6`) |
| FB-3 | Slow first token (`slow` > `LLM_FIRST_TOKEN_TIMEOUT_S=3`) | `timeout` -> fallback to `fake/b` |
| FB-4 | `empty` response | Fallback (`llm_service.py` docstring l.5-6) |
| FB-5 | All chain models fail | `error` event `all_models_failed`, `retryable:true`, `attempts[]`, `retry_after` set when last error was 429; UI shows error with Retry, no duplicate bubbles |
| FB-6 | Mid-stream failure after 3 chunks (`midfail`) | No fallback (post-first-token); `error` event with `partial:true`, `message_id` set; the partial is persisted with `partial:true` meta (`chat_service.py:277-291`); UI keeps partial text and shows retry |
| FB-7 | Quota guard header learning (`[[sc:headers]]`) | Subsequent big prompts skip the model with `skipped:"quota guard"` (`rate_limiter.py:486-497`) |
| FB-8 | `GET /models` | Lists exactly the `LLM_MODELS` registry with `available`, `reason`, and `default` (`routes.py:213-219`); no hard-coded names in the UI (frontend renders only what the API returns) |
| FB-9 | Model absent from fake `/models` + `?refresh=true` | That model `available:false, reason:"not offered by the provider for this key"`; UI shows it disabled/hidden per design; sending with it explicitly falls back to default (`chat_service.py:140-141`) |
| FB-10 | Fake `[[sc:404]]` for `fake/a` | Hard trip (3600 s): `/models` then reports `available:false` and other models of the same provider are tripped too (`llm_service.py:349-358`) |
| FB-11 | Unknown `model` in `ChatRequest` | Silently uses session/default model (`chat_service.py:140-141`); `/generate` with unknown model -> 400 (`routes.py:270-271`) |
| FB-12 | Fake discovery endpoint down | Startup/refresh logs warning; models stay `available:true` (`llm_service.py:209-211`) |

### G5 Upload / ingestion (user `it-<run>-up`)
Generated fixtures in `/tmp/aiap-it/run/fixtures` (tiny, synthetic, no real content): `a.txt`, `b.md`, `c.rst`, `d.csv`, `e.docx`, `f.xlsx`, `g.pdf` (text), `h.pdf` (image-only, no text), `big.txt` (1 MiB + 1 byte), `empty.txt`, `bad.pdf` (random bytes), `evil.exe`, `../traversal.txt` as a multipart filename.

| ID | Case | Expected |
|---|---|---|
| UP-1..UP-7 | Upload each of txt, md, rst, csv, docx, xlsx, pdf | 200; `chunk_count>=1`; `file_type` = extension; `GET /documents` lists it; `indexing` is `hybrid` (models present) or `lexical` |
| UP-8 | Ask a question answered only by the uploaded text (fake LLM echoes the retrieved excerpts via a `[[sc:echo-context]]` marker so grounding is observable) | `route:"rag"`, `citations` event with `source`=filename, `page`, `snippet`; UI citation chips/modal show those real values; no source appears that is not in the event |
| UP-9 | Image-only PDF | 422 `No readable text...` (`document_service.py:109-110`) |
| UP-10 | Corrupt PDF | 422, backend still healthy |
| UP-11 | Unsupported `.exe` | 415 `Unsupported file type` (`document_service.py:81`) |
| UP-12 | Indexing label truthfulness | UI shows "indexed" only after the `/upload` 200 response; with embeddings unavailable shows lexical/degraded wording or at least no claim of vector indexing (`UploadResponse.indexing`) |
| UP-13 | `big.txt` (> `MAX_UPLOAD_MB`) | 413, body not fully buffered (`routes.py:361`); UI shows the message, no stuck progress bar |
| UP-14 | Empty file | 400 |
| UP-15 | Traversal filename | `Path(filename).name` used (`document_service.py:63`); stored filename has no path components |
| UP-16 | Re-upload same filename | One document in the list (replaced), old chunks removed (`document_service.py:134-135`) |
| UP-17 | Delete document | 204; list empty; RAG question no longer cites it; deleting twice -> 404 |
| UP-18 | Missing file and URL | 400 (`routes.py:354-355`) |
| UP-19 | Drag-and-drop and file picker in the UI | Both go through `/upload`; list refreshes; errors surface as toasts with the server message |
| UP-20 | Upload while a stream is running | Both complete; no cross-talk |

### G6 URL ingestion and SSRF (all targets local or literal; no outbound internet)
Backend runs with `ALLOW_PRIVATE_URLS=false`. Because loopback is refused, the "success" path cannot be tested in this run; it is covered in the separate G6b below.

| ID | URL (query param to `POST /upload?url=`) | Expected |
|---|---|---|
| SS-1 | `http://127.0.0.1:18002/` (a local dummy server that logs hits) | 422 "private or local addresses are not allowed"; dummy server log shows ZERO hits (`net.py:49-52`) |
| SS-2 | `http://localhost/`, `http://[::1]/`, `http://0.0.0.0/` | 422, zero hits |
| SS-3 | `http://169.254.169.254/latest/meta-data/` (link-local literal; rejected before any connect) | 422 |
| SS-4 | `http://10.0.0.1/`, `http://192.168.1.1/`, `http://172.16.0.1/` | 422 (guard resolves to literal; no packet is sent because the check precedes the request) |
| SS-5 | Decimal / hex / octal IP forms (`http://2130706433/`, `http://0x7f.0.0.1/`) | 422 or resolver error 422; zero hits (verify how `getaddrinfo` handles them) |
| SS-6 | `file:///etc/passwd`, `ftp://...`, `gopher://...`, `javascript:...` | 422 "Only http(s)" (`net.py:40-41`) |
| SS-7 | Redirect from an allowed host to a private one | Cannot be produced without a public host; **not testable offline**; covered by unit tests (`tests/test_units.py`/`test_api.py`, check) - documented as residual |
| SS-8 | DNS rebinding (resolve public then private) | **Not tested**: `assert_public_url` resolves separately from httpx's own resolution (`net.py:45`, `web_client.py:33`); a TOCTOU gap exists in principle. Report as an observation, not a test. |
| SS-9 | Add-URL UI | Error text shown; no document added; `window` not navigated |
| G6b | Separate run with `ALLOW_PRIVATE_URLS=true`, local page served on `127.0.0.1:18002` (static HTML, > 100 chars of text, plus an oversize 6 MB page and a binary content-type page) | 200 + chunks; 5 MB cap -> 422 "too large"; `image/png` -> 422 content-type (`web_client.py:42-44`); HTTP 500/404 from the local page -> 422; confirms trafilatura/bs4 extraction. Run only with approval because it deliberately weakens the guard (loopback only). |

### G7 Error handling and cooldown
`RATE_LIMIT_RPM_PER_USER=2` for EH-1/EH-2 (separate backend start).

| ID | Case | Expected |
|---|---|---|
| EH-1 | 3 rapid `POST /chat/stream` same user | Third: HTTP 429, `Retry-After` integer >= 1 (`routes.py:179`), JSON detail "Too many requests..." |
| EH-2 | UI under throttle | Send fails before stream: composer text retained (draft preserved), toast with the seconds, cooldown prevents sends until expiry, live-region announces "You can send again." (view-contract 478-479) |
| EH-3 | Stream `error` event with `retry_after` (FB-5 with 429 chain) | Retry button disabled / cooldown countdown; Retry during cooldown makes NO request (verify via fake log) |
| EH-4 | No provider (separate start with `GROQ_API_KEY` unset) | HTTP 503 with the "No LLM provider" detail; UI toast; `/models` shows all `available:false` with reason "groq API key not set" (`llm_service.py:178-179`) |
| EH-5 | Retry-After CORS exposure gap | (a) same-origin UI: header readable by `fetch` -> countdown works. (b) page served from a different origin (e.g. `http://localhost:18003` static server) with `ALLOWED_ORIGINS` including it: `resp.headers.get("Retry-After")` is `null` because the CORS middleware sets no `expose_headers` (`main.py:72-78`) -> frontend falls back to no countdown (`api.js:25-26`). Expected result today: gap CONFIRMED in (b). Record as backend finding, not a frontend bug. |
| EH-6 | 404 session | `POST /chat/stream` with unknown `session_id` -> HTTP 404 (`routes.py:234-235`); `GET/PATCH/DELETE` unknown -> 404; UI recovers (clears active session, no endless loop) |
| EH-7 | 422 validation | `message` > 20000 chars, `user_id` empty/ >128, bad JSON -> HTTP 422 with array `detail`; UI shows a readable message, not "[object Object]" |
| EH-8 | Backend killed mid-stream | UI shows network error, Send re-enabled, partial text retained; backend restart -> session list recovers |
| EH-9 | Malformed/oversized SSE from fake (`data: not-json`) | Backend never forwards raw upstream text (`llm_service.py:7-8`); client gets a normalized `error` |
| EH-10 | Error text leakage | No upstream URL, key, or provider body appears in any response or toast |

### G8 Draft preservation and duplicate user turn (QA M1)
| ID | Case | Expected |
|---|---|---|
| DR-1 | Pre-stream failure (429/503/404) | Composer text restored, no bubbles added, server has no new message (the user turn is persisted only inside the stream; `routes.py:230-235` raise before) |
| DR-2 | Stream error before first token (FB-5) | Draft restored (merged with any text typed meanwhile), error bubble with Retry; **server already stored the user turn** (`chat_service.py:155`) |
| DR-3 | Re-send restored draft instead of Retry | Observe server transcript: **expected today = two identical user turns** (known, documented in view-contract 479; backlog `stage4-backlog.md:8`). Record actual state. Pass criterion for this plan = behaviour matches documentation; a FAIL of the product is raised only if the user turn count differs from the documented outcome. |
| DR-4 | Retry via inline button | Exactly one user turn (RG-5) |
| DR-5 | Type while streaming | Textarea stays enabled; typed text survives completion and Stop |
| DR-6 | Reload during draft | Draft behaviour documented (localStorage key check) |

### G9 Session CRUD (throwaway user only)
| ID | Case | Expected |
|---|---|---|
| SC-1 | Create, list, select | Appears in the sidebar; list filtered by `user_id` (`session_service.py:38-43`) |
| SC-2 | Rename (`PATCH`), empty/200+ char title | 200 / 422 (`models/api.py:23`); UI inline rename saves on Enter, cancels on Esc |
| SC-3 | Delete | 204; messages gone; subsequent GET 404 |
| SC-4 | Search sessions | Filters locally by title (client-side per audit) |
| SC-5 | Feedback up/down, then toggle | 200, persisted; unknown message -> 404 |
| SC-6 | Cascade: delete session does NOT delete documents | Documents remain (per-user, not per-session) |
| SC-7 | Model/mode change persisted on the session | `GET /sessions/{id}` reflects it |

### G10 Concurrency
| ID | Case | Expected |
|---|---|---|
| CC-1 | Double-send (two rapid Enter presses) | UI gates via `isLoading`; at most one request; if two reach the server the transcript is consistent (document actual) |
| CC-2 | Two parallel streams, same session (API) | Both complete or the second fails cleanly; history order consistent; no DB lock errors (`busy_timeout=5000`, WAL, `database.py:513-518`) |
| CC-3 | Two parallel streams, different sessions | Both succeed; tokens not interleaved across streams |
| CC-4 | Two browser tabs, same user | New session in tab A appears in tab B after refresh; stream in A unaffected by deleting a different session in B; deleting the session open in A mid-stream: record behaviour (persist step will fail gracefully? `chat_service.py:386-387`) |
| CC-5 | Upload + chat in parallel, delete document during retrieval | No 500; answer arrives |
| CC-6 | 20 concurrent short streams (fake, no cost) | All complete; memory/FD counts stable; `/health` responsive |

### G11 Authorization gap (current behaviour, documented only)
| ID | Case | Expected today |
|---|---|---|
| AZ-1 | User B reads `GET /sessions/{A's id}` | 200 with A's messages (no ownership check; `routes.py:301-306`) |
| AZ-2 | User B `PATCH` rename A's session | 200 |
| AZ-3 | User B `DELETE` A's session | 204 |
| AZ-4 | User B `POST /messages/{A's msg}/feedback` | 200 |
| AZ-5 | User B `POST /chat/stream` with A's `session_id` | Accepted; the stream uses A's `user_id` for retrieval (`chat_service.py:138`), i.e. B can query A's documents via A's session |
| AZ-6 | `GET /documents?user_id=A`, `DELETE /documents/{id}?user_id=A` | Open to anyone who supplies the user_id |
| AZ-7 | `GET /sessions?user_id=A` | Lists A's sessions |
Result classification: these are recorded as "KNOWN GAP (security-issue-authz.md), behaviour confirmed" - they do not fail the run. Only the throwaway ids from this run are used (`it-<run>-A`, `it-<run>-B`); the tests must never be pointed at a non-isolated backend (pre-flight gate checks the data dir and config echo).

## 4. Browser automation approach

- Tooling as in earlier QA: `/usr/bin/google-chrome` headless driven by `puppeteer-core` (pinned version recorded) installed in a FRESH scratch dir under `/tmp/aiap-it/tools` (never inside the repo; scripts kept in `/tmp/aiap-it/scripts`; packages treated as untrusted, `npm ci`-style with a lockfile or exact `--save-exact`; optionally `axe-core` for accessibility spot checks). Package download requires network access and therefore approval; the alternative is to reuse the scratch install from earlier QA runs if it still exists.
- Frontend loaded from the isolated backend itself: `http://127.0.0.1:18000/` (same origin, `BACKEND_URL` default empty, `api.js:6-8`). No `window.CONFIG` injection is needed. For EH-5(b), a second static server on a different loopback port serves a copy of `frontend/` with a `config.js` injecting `window.CONFIG={BACKEND_URL:"http://127.0.0.1:18000"}` (the file is gitignored for exactly this purpose; it lives only in the temp copy, and the page would need a `<script>` tag for it - modify only the TEMP copy of `index.html`).
- Network allow-list so nothing else is reachable:
  1. Chrome flags: `--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"` plus `--proxy-server="http://127.0.0.1:9"` (black-hole) and `--disable-background-networking --disable-component-update --no-first-run --user-data-dir=/tmp/aiap-it/run/chrome-profile`.
  2. CDP `Fetch.enable` / request interception: every request whose origin is not `127.0.0.1:18000` (or `:18003` for EH-5b) is failed with `BlockedByClient` and logged; any such hit is a test failure (the frontend loads CDN scripts: marked, DOMPurify, highlight.js, fonts; see `index.html` - these CDN requests WILL be blocked, which is itself a finding: the UI must degrade gracefully, or the user decides to serve vendored copies; decision D7).
  3. Backend side: bind to 127.0.0.1; optionally run the backend under `unshare -rn` / a network namespace with only loopback up (also blocks accidental Tavily/HF/LLM calls). If namespaces are unavailable, rely on dummy keys (any real call would fail with 401 and be visible in logs) and absent Tavily key.
- Drive: viewport 1366x768 and 390x844; helpers for `waitForSelector`, listening to `console`/`pageerror`, capturing `Network.eventSourceMessageReceived`-style data via a `fetch` stream tap, HAR-lite log of every request.
- Stop via the real Stop button (UI), and via `AbortController` (API script using `httpx` stream + close).
- Evidence screenshots at key states (streaming, stopped, fallback badge, error+retry, 429 cooldown, upload progress/indexed, citation modal, long unbroken string).

## 5. Procedure, gates and cleanup

### 5.1 Pre-flight checklist (all must be true before the first request)
1. User approval recorded for section 10 decisions.
2. `/tmp/aiap-it` created fresh (must not exist beforehand); copy contains no `.env`, no `data/`, no `server.log`.
3. Start command run with `env -i PATH=... HOME=/tmp/aiap-it/run` and only the variables from 1.2; **config echo check**: the backend log line "Effective config" (`container.py:87`, masked) must show `data_dir` = `/tmp/aiap-it/data`, `database_url`, `groq_base_url` = loopback, `groq_api_key: set` (dummy), `tavily_api_key: unset`, `allowed_origins` loopback. The harness aborts if any value differs.
4. Hard guard in the harness: refuse to run unless the resolved base URL host is `127.0.0.1` and `GET /health` + `GET /models` show the fake registry (`fake/*`); refuse if any model id outside `fake/*` exists (prevents pointing at the developer's server on :8000).
5. `ss -ltnp` shows the intended ports free before start (18000-18003); the dev's server on :8000 is not touched.
6. Real `data/` untouched: record `stat`/`sha256sum` of `<repo>/data/*.db` before and compare after (read-only checksum).
7. Fake LLM self-test (`curl` to the fake only).
8. Disk space check (< 200 MB needed, plus model copy ~150 MB if option (a)).

### 5.2 Safety gates and stop conditions
Abort the whole run immediately, kill all processes, keep evidence, if:
- any request leaves loopback (Chrome interception log, or firewall/netns counters), or any non-fake outbound host appears in backend logs;
- any HTTP 401/403 from the fake with a real-looking key (sign that a real key leaked in), or any `Authorization` value other than the dummy appears in the fake's request log;
- the real `data/` checksum changes;
- the harness detects a non-isolated config (5.1 #3/#4);
- backend memory > 1.5 GB, CPU pegged > 120 s, or more than 3 unexpected 5xx in a group;
- disk usage of `/tmp/aiap-it` > 1 GB.
Per-group time-boxes: 10 minutes each; global 2 hours.

### 5.3 Secrets handling
No real keys exist in tier 1 (dummy `fake-key-not-real`). `.env` is never read or copied. Logs are scanned post-run with a regex for `gsk_`, `AIza`, `tvly-`, `Bearer ` to prove nothing key-like was written. Tier 2 (if approved) follows 2.3. Nothing from `/tmp/aiap-it/run` is committed.

### 5.4 Data cleanup
After evidence is archived: stop backend/fake/static servers (`kill` by PID file, then verify with `ss -ltnp` and `pgrep -f aiap-it`), `rm -rf /tmp/aiap-it/data /tmp/aiap-it/app /tmp/aiap-it/run /tmp/aiap-it/tools` (verify path prefix before `rm`), keep only `/tmp/aiap-it/evidence` until the user has read the report (then delete on request, or move to `docs/redesign/` only if the user wants it). Check afterwards that `<repo>/data` checksum is unchanged and `git status` shows no new files from the run.

### 5.5 Evidence to capture
Per case: request/response (status, headers incl. `Retry-After`, SSE frame transcript with timestamps), fake-LLM request log, backend log excerpt (INFO), SQLite row dumps (read-only `sqlite3 -readonly`) for persistence assertions, screenshots, console/pageerror log, network interception log, config echo, versions (Python, Chrome, puppeteer-core, build identifier). A single `results.json` (case id, result, evidence paths) and a Markdown summary.

### 5.6 Pass/fail criteria
- PASS: observed behaviour equals the expected column; or equals documented known behaviour (DR-3, AZ-*, EH-5b) and is labelled as such.
- FAIL: deviation from expected; any unhandled exception in the backend log; stuck UI state (spinner, disabled composer after completion/stop); duplicate/missing persisted messages beyond the documented ones; any request outside the allow-list; XSS-like rendering of fake-provided HTML (use `[[sc:html]]` marker emitting `<img onerror>` / `<script>` text: must be sanitized).
- BLOCKED / NOT RUN: a precondition missing (e.g. embeddings models absent -> UP-12 hybrid assertion).
- Severity mapping follows QA reports (Blocker/Major/Minor/Nit); backend findings are routed to the backend owner, UI findings to the frontend owners.

### 5.7 Exit criteria
Run is complete when: all cases are PASS/KNOWN/BLOCKED with evidence; no safety gate fired; cleanup verified; report written (`docs/redesign/qa-report-NN.md` style) with counts and the open questions (CORS exposure, M1 duplicate turn, authz gap, CDN offline behaviour).

### 5.8 Estimated effort
| Item | Estimate |
|---|---|
| Fake LLM server + scenario markers | 3-4 h |
| Harness (env launcher, API client, SSE parser, checksum guards, evidence writer) | 4-6 h |
| Browser scripts (G1/G2/G4/G5/G7/G8/G10 UI parts) | 6-8 h |
| Execution + triage (single run, restarts between groups) | 3-4 h |
| Report | 1-2 h |
| Total | about 2.5-3 working days; tier 2 adds about 1 h |

## 6. Known risks and open observations to resolve while building the harness
1. `LLMService.complete` (used by query rewrite and `/generate`) must be checked against the fake: confirm it also goes through `open_stream` and that the 4 s rewrite timeout (`chat_service.py:371`) does not fire on the slow-first-token marker unintentionally.
2. The semantic cache needs the embedder (`chat_service.py:182`): with models absent it is skipped; with models present, repeated test prompts must be unique to avoid accidental `cached:true`.
3. `user_id` default is `default` in several routes (`routes.py:167`): the frontend's behaviour for user id (localStorage key) must be set to the test namespace through the browser profile (pre-seed localStorage), otherwise UI tests would run as `default`. Still safe in the fresh DB, but confirm.
4. The frontend loads third-party CDN scripts; under the network allow-list they are blocked. Either vendor copies in the temp copy (needs a one-time approved download) or accept degraded rendering and test only what works. This is decision D7.
5. Mid-stream SSE-level `error` objects from the provider are not handled identically to transport errors in the SDK; both scenarios are listed (FB-6, EH-9).
6. Windows/CRLF, proxies and HTTP/2 buffering are out of scope (uvicorn direct, HTTP/1.1).

## 7. What this plan deliberately does not do
No call to Groq, Google, Tavily, Cohere or HuggingFace in tier 1 (option (a) for models). No writes to the developer's `data/`, `.env`, or running server. No changes to the repository (the plan itself is the only file added). No load or performance benchmarking beyond CC-6. No fix attempts: findings go to the orchestrator with owners.

## 8. Mapping to the requested coverage

| Requested area | Cases |
|---|---|
| Real SSE streaming | ST-1..ST-9 |
| Stop / cancellation | SP-1..SP-8 |
| Regenerate | RG-1..RG-5 |
| Model fallback and model list | FB-1..FB-12 |
| Document upload (all types, limits, corrupt, RAG + citations) | UP-1..UP-20 |
| URL ingestion + SSRF | SS-1..SS-9, G6b |
| Errors (429/503/Retry-After, partial/error events, cooldown, 404, 422) | EH-1..EH-10, FB-5, FB-6 |
| Draft preservation / duplicate user turn | DR-1..DR-6 |
| Session CRUD / rename / delete | SC-1..SC-7 |
| Concurrency | CC-1..CC-6 |
| Authorization gap (documentation only) | AZ-1..AZ-7 |

## 9. Items that cannot be tested in this environment
Real provider behaviour (without tier 2), DNS rebinding and public-redirect SSRF (SS-7, SS-8), real Tavily search/research mode (no key; only "disabled" behaviour), Cohere providers, production deployment (Procfile, proxies, TLS), multi-process deployments (in-memory limiter/breaker per process), long-running soak.

## 10. Decisions needed from the user
1. Environment location: `/tmp/aiap-it` on this machine, another disposable host, or a container? Is a network namespace (`unshare -rn`) acceptable?
2. LLM: tier 1 fake only (recommended), or also tier 2 real key? If tier 2: whose key, which provider/models, request budget (proposed 20), and who rotates the key afterwards.
3. Embedding models: copy the existing `data/models` read-only (no network), allow a one-time HuggingFace download, or run lexical-only (UP-12 hybrid case skipped)?
4. Tooling downloads: may `puppeteer-core` (+ `axe-core`) be installed from npm into the scratch dir, or reuse the earlier QA install? May CDN assets (marked, DOMPurify, highlight.js, fonts) be fetched once and vendored into the temp copy so the UI works under the allow-list (D7)?
5. Backend code changes: allowed or not? (a) `expose_headers=["Retry-After"]` on CORS (`main.py:72-78`); (b) nothing else is required for the fake because `GROQ_BASE_URL` already exists; (c) the authorization gap and the duplicate-user-turn behaviour stay as documented unless the user opens a separate change.
6. Who runs it: the QA agent unattended, the QA agent with the user watching each group (recommended for the first run), or the user manually? Who owns the fake LLM script if it should be kept (commit under `tests/tools/` or leave in `/tmp`)?
7. Scope of G6b (running once with `ALLOW_PRIVATE_URLS=true` on loopback): approve or skip.
8. Time-box and budget: approve about 3 working days and the 2-hour execution window, or reduce scope to priority groups (G1, G2, G4, G7, G8 first).
