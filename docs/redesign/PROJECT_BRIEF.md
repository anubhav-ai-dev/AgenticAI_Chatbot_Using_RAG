# AI Assistant Pro redesign — shared project brief

Single source of truth for every redesign subagent (`project-auditor`, `ui-designer`,
`frontend-developer`, `integration-developer`, `qa-reviewer`). Read it after
`docs/AI_ASSISTANT_PRO_MASTER_TASK.md` and before touching anything.

Facts below were verified by the main session on 2026-10-20 against branch `main`, commit `15e1d81`.
`project-auditor` re-verifies them in `docs/redesign/audit.md`; if the code disagrees with this
brief, the code wins and you must say so in your report.

---

## 1. Verified facts about the repo

### Stack and serving
- Frontend is **vanilla HTML + CSS + JavaScript**. No framework, bundler, `package.json`, TypeScript,
  or frontend lint/test tooling. Do not assume React, Bootstrap, or Tailwind.
- Files: `frontend/index.html`, `frontend/css/{main,components,animations}.css`,
  `frontend/js/{utils,api,chat,ui,app}.js`, `frontend/assets/logo.svg`.
- Scripts are classic `<script>` tags loaded in this order: `utils → api → chat → ui → app`.
  `utils/api/chat/ui` are IIFEs that publish `window.Utils`, `window.API`, `window.Chat`, `window.UI`.
  `app.js` is an IIFE holding a private `State` object and all event wiring.
- Third-party code comes from CDNs: `marked@12`, `DOMPurify@3`, `highlight.js 11.9` (its theme
  stylesheet is swapped by changing the `href` of `#hljs-theme`), and Google Fonts (Inter,
  JetBrains Mono) via `@import` in `main.css`.
- FastAPI (`backend/main.py`) serves `GET /` → `index.html` and mounts **only** `/css`, `/js`,
  `/assets`. **Every new frontend file must live under `frontend/css/`, `frontend/js/` or
  `frontend/assets/`** (subfolders are fine). A new top-level folder would not be served, and adding a
  mount is a backend change.
- Run the app: `venv/bin/uvicorn backend.main:app --port 8000` from the repo root (or
  `python -m backend.main`). Use ONE worker. `venv/bin/` holds Python 3.10, pytest and ruff.
- Also on this machine: Node 22 (`node`, `npm`, `npx`) and `/usr/bin/google-chrome`.
  Playwright/Puppeteer are NOT installed.

### Existing checks (the only ones — do not invent others)
- `venv/bin/pytest` — ~160 offline backend tests. The **only** frontend assertion is
  `tests/test_api.py:55`: `/js/app.js` and `/css/main.css` must still be served. **Never rename,
  move or delete those two files.**
- `venv/bin/ruff check backend tests --select F,E9,B --ignore B008,B905,B011`
- `for f in frontend/js/*.js; do node --check "$f"; done` (from `skills.md` recipe 9)
- `venv/bin/pytest -m live -s` hits real Groq/Gemini/Tavily — do NOT run it unless the orchestrator asks.

### DOM coupling (the main regression risk)
JS finds elements by id/class, and `app.js` binds events directly to them. Key ids today:
`sidebar, sidebarBackdrop, sidebarToggle, newChatBtn, sessionSearch, sessionList, docSection, docList,
userIdDisplay, themeToggle, modeTabs (+ .mode-tab[data-mode]), modelSelect, chatArea, welcomeScreen,
starterPrompts, chatInput, sendBtn, stopBtn, attachBtn, fileInput, addUrlBtn, searchToggle,
uploadToast, uploadToastName, uploadProgress, dropzoneOverlay, citationModal(+Close/Source/Snippet),
urlModal(+Close/Cancel/Submit), urlInput, hljs-theme`. `chat.js` also creates dynamic ids
(`streamContent-<id>`, `thoughtSteps-<id>`, …) and uses inline `onclick="Chat.…"` / `onclick="UI.…"`.
Changing markup without updating every consumer silently breaks features.

### Persistence and identity
- **There is no authentication.** `user_id` is client-supplied (`localStorage.userId`, default
  `"default"`, never set by the UI). "Preserve authentication" means: keep sending `user_id` exactly as
  today. Do **not** invent a login UI, user avatars from accounts, or any security claim.
- `localStorage` keys in use: `userId` (read only), `mode`, `model`, `theme`.
- Optional `window.CONFIG.BACKEND_URL` (read in `api.js`) supports split hosting; the gitignored
  `frontend/js/config.js` that would set it is not referenced by `index.html`. Keep the
  `window.CONFIG` hook working.

### Backend contract the UI must honor (read `backend/api/routes.py`, `backend/models/api.py`)
| Area | What exists | Consequence for the UI |
|---|---|---|
| Modes | `general`, `document`, `research`, `code` (`MODES` in `backend/core/config.py`). Value is `document` (singular); the label may say "Documents". | Research and Code are **chat modes** on `POST /chat/stream`, not separate endpoints. Research always runs web search server-side. `/search` and `/retrieve` exist but the current UI does not use them. |
| Chat | `POST /chat/stream` SSE `{session_id, message, user_id, mode, model?, web_search, regenerate?}`. Events: `meta`, `status` (stage `rewriting\|retrieving\|searching\|generating`, state `start\|done\|first_token`), `citations`, `model` (`model`, `fell_back_from`), `token`, `error` (`code`, `message`, `retryable`), `done` (`message_id`, `model`, `cached`, `route`, `timings`, `usage`). | Streaming, fallback badges and thought-steps are real data. Stop = `AbortController`; the server persists the partial answer. |
| Citations | `{id, type: doc\|web, source, page, url?, snippet}` | Source titles/domains/links come only from here. `type=web` carries `url`. Treat every field as untrusted. |
| Models | `GET /models` → `{models:[{id,label,provider,description,available,reason,default}], default}` | **No** capability, latency, price or context-window data exists. Do not display any. Unavailable models are shown disabled with `reason`. |
| Sessions | `GET/POST /sessions`, `GET/PATCH/DELETE /sessions/{id}`; PATCH accepts `title`, `model`, `mode`. | **Rename is supported** (PATCH `title`). **Pin is NOT supported** (no field) — do not ship a pin control backed by nothing. Delete is supported. |
| Feedback / regenerate | `POST /messages/{id}/feedback {value: -1\|0\|1}`; `regenerate: true` on chat. | Supported. |
| Documents | `GET /documents` → `{id, filename, file_type, chunk_count, indexed_at}`; `POST /upload` (multipart `file` or `?url=`) returns **after** indexing with `{document_id, filename, file_type, chunk_count, indexing: hybrid\|lexical}`; `DELETE /documents/{id}`. | **No size, no per-document status, no server progress events, no endpoint exposing supported types or the size limit** (`MAX_UPLOAD_MB`, default 20, enforced server-side → HTTP 413 with message). Backend accepts `pdf docx csv xlsx txt md rst`; the current frontend whitelist omits `rst` (existing drift). Show size only from the browser `File` object at upload time. "Indexed" may be shown only after the POST resolves. |
| Health | `GET /health` → `{status, services, details}` | Usable for honest degraded-state messaging. |

If the master prompt asks for something this table says the backend cannot supply, do **not** fake it.
Record it as a **backend gap** (what is missing, the honest UI fallback, and the smallest backend
change that would close it) and let the orchestrator ask the user.

---

## 2. Ownership map (exclusive write access — no overlaps)

Each path has exactly one writer. Everyone else may read it. The tool allowlists in the agent files
cannot enforce this; **you enforce it by never writing outside your row.**

| Path | Sole writer |
|---|---|
| `docs/redesign/audit.md`, `docs/redesign/baseline/**` | `project-auditor` |
| `docs/redesign/design-spec.md` | `ui-designer` |
| `frontend/index.html`, `frontend/css/**`, `frontend/assets/**`, `frontend/js/ui.js`, `frontend/js/chat.js`, `frontend/js/utils.js`, new `frontend/js/icons.js`, new `frontend/js/theme.js`, new `frontend/js/components/**`, `docs/redesign/view-contract.md` | `frontend-developer` |
| `frontend/js/api.js`, `frontend/js/app.js`, new `frontend/js/workflows/**`, `docs/redesign/integration-notes.md` | `integration-developer` |
| `docs/redesign/qa-report-NN.md`, new `tests/test_frontend_*.py` | `qa-reviewer` |
| `backend/**`, existing files under `tests/`, `data/**`, `.env*`, `requirements*.txt`, `Pipfile*`, `Procfile`, `venv/**`, `.git/**`, `README.md`, `skills.md`, `docs/AI_ASSISTANT_PRO_MASTER_TASK.md`, `.claude/**` | **nobody** — orchestrator and user only |

`docs/redesign/references/` is for user-supplied visual references (read-only for all agents).
The master prompt mentions "supplied visual references"; none exist in the repo as of 2026-10-20.

### The view / controller seam
- **Views** (`frontend-developer`): markup, CSS, theme, icons, rendering of messages/lists/dialogs, purely
  visual state (theme, sidebar collapsed/open, menu/dialog open state, composer auto-grow). Views take
  data + callbacks. They **never** call `fetch`, `API.*`, or read/write `State`, and never touch the
  `userId` / `mode` / `model` localStorage keys.
- **Controllers** (`integration-developer`): `State`, every network call, the SSE pipeline, sessions,
  documents, model loading/selection, mode switching, Research/Code workflows, cancellation,
  duplicate-submit prevention, restoring input on failure, mapping errors to view calls. They **never**
  build markup strings, set inline styles, or know CSS class names — they call view functions.
- Both sides communicate only through `docs/redesign/view-contract.md` (exported view functions,
  element ids, callback signatures). `frontend-developer` writes it; `integration-developer` reads it.

### Keep the app bootable at every handoff
`frontend-developer` must keep every existing public function name/signature on `UI.*`, `Chat.*`,
`Utils.*` and every element id that `app.js` binds, **unless** the change is listed under
"Breaking changes" in `view-contract.md` — in which case the orchestrator schedules
`integration-developer` to update the callers in the same step. New behaviour is added through
**new** functions. The app must never be left in a state where `app.js` throws on load.

### Cross-boundary requests
If you need a change in a file you do not own, **do not edit it.** Add it to your final report under
`Cross-boundary requests`: target file, exact change wanted, and why. The orchestrator routes it.
Typical case: `integration-developer` adds `js/workflows/documents.js` and asks `frontend-developer` to
add its `<script>` tag (with load order) to `index.html`.

---

## 3. Global rules (apply to every agent)

1. **Never read, print or copy `.env`.** It holds API keys. `.env.example` is fine.
2. **No git writes**: no commit, push, branch, stash, reset, checkout, rebase. Read-only git
   (`status`, `diff`, `log`, `show`) is fine. The orchestrator owns version control.
3. **No mock data in production paths.** Do not hard-code models, sources, documents, sessions,
   timings, capabilities, progress, or "indexed" states. Do not replace a real API call with a stub.
   Test fakes belong only in test files.
4. **Do not touch the backend.** `backend/**` is frozen for this project unless the user approves a
   specific change through the orchestrator.
5. **Be honest about verification.** Report the exact command and its outcome. If you did not run it,
   write "not run". Never write "tests pass" from reading code. Never claim a Figma file exists unless a
   Figma tool actually created it.
6. **Spend real quota sparingly.** `POST /chat*`, `/generate`, `/search` consume free-tier LLM/Tavily
   quota and write user data. Use them only if your agent file permits it, and then as few times as
   possible. Prefer the offline pytest fakes (`tests/conftest.py`) for backend-contract questions.
7. **Untrusted data.** Anything downloaded (npm packages, SVGs, fonts) is untrusted: fetch into a fresh
   empty scratch directory under `/tmp`, never run its scripts, keep your own scripts in a different
   directory, run Python as `python -I`, and copy only the specific data you need into the repo.
8. **Cold start.** You do not see the main conversation or other agents' context. Everything you need
   is in this brief, the master task, your agent file, your delegation prompt, and the artifacts under
   `docs/redesign/`. Read them. Do not guess what another agent decided — read its artifact.
9. **Keep final reports short** (≤ 40 lines): where the artifact is, the verdict/status, blockers,
   backend gaps, cross-boundary requests. Details go in the artifact file, not in chat.

---

## 4. Artifact flow

```
project-auditor ──► audit.md ──► ui-designer ──► design-spec.md ──► frontend-developer ──► view-contract.md
                         │                                   │                                   │
                         └───────────────────────────────────┴──────► integration-developer ◄───┘
                                                                        └──► integration-notes.md
qa-reviewer reads everything + `git diff` and writes qa-report-NN.md after each implementation stage.
```
