# AI Assistant Pro: Phase 1 audit

Audited 2026-10-20 against `main` @ `15e1d81` (working tree: only docs/.claude untracked). Read-only. All
`file:line` references are to that tree. "Inferred" marks anything not directly observed.

---

## 1. Summary

- AI Assistant Pro is a FastAPI backend (hybrid RAG, Tavily web search, Groq/Gemini with model fallback) serving a
  **vanilla HTML/CSS/JS SPA** (about 3,100 LOC, no build, no package manager, no frontend tests).
- Solid: a clean SSE contract (`meta/status/citations/model/token/error/done`), real stop/partial-persist,
  regenerate, feedback, fallback badges, DOMPurify-sanitized Markdown, ids that are all consistent (no missing DOM ids).
- Baseline is green: 156 pytest passed, ruff clean, all 5 JS files pass `node --check` (section 10).
- Risk 1: the UI is welded to ids/classes and inline `onclick`; markup rewrites must be staged against the DOM contract (section 4).
- Risk 2: several master-prompt items have **no backend support**: pin, document size/status/progress, supported-type/limit
  discovery, model capabilities, authentication, user profile (section 8). These must degrade honestly.
- Risk 3: "Research" and "Code" are **chat modes**, not separate workflows. Research citations carry `url`, but the UI never
  renders it (`ui.js:242-253`, `chat.js:179-187`).
- Risk 4: input is cleared before the request and never restored on failure (`app.js:163-166`); regenerate removes the DOM
  answer before the request (`app.js:286-291`). Both violate the master prompt's "preserve input" rule today.
- Risk 5: session timestamps are naive UTC and the browser parses them as local, so "Today/Yesterday" grouping and times are
  skewed (section 9, D1).
- No visual references exist in the repo (open question Q1). No light-theme screenshot could be captured (section 10).

---

## 2. Stack, serving, build

| Item | Fact | Evidence |
|---|---|---|
| Frontend | Plain HTML + 3 CSS + 5 JS files, classic `<script>` tags, order `utils, api, chat, ui, app` | `frontend/index.html:273-277` |
| Framework / bundler / lint / types | none. There is no `package.json`, `tsconfig`, or eslint config in the repo | `ls` of repo root; brief 1 confirmed |
| Libraries (CDN, no SRI, floating majors) | marked@12, dompurify@3 (jsdelivr), highlight.js 11.9.0 (cdnjs), Google Fonts Inter + JetBrains Mono via CSS `@import` | `index.html:15,264-268`, `main.css:5` |
| Module pattern | `window.Utils/API/Chat/UI` IIFEs. `app.js` is a private IIFE with `State` | `utils.js:5`, `api.js:5`, `chat.js:6`, `ui.js:6`, `app.js:7-23` |
| Serving | `GET /` returns `index.html`. Mounts only `/css`, `/js`, `/assets` | `backend/main.py:78-90` |
| Split hosting | `window.CONFIG.BACKEND_URL` read in `api.js:6-8`. `frontend/js/config.js` is gitignored (`.gitignore:22`) and **not referenced** by `index.html`, so the hook is currently inert (D7) |
| CORS | `allowed_origins` default `http://localhost:8000` | `backend/core/config.py:192,222` |
| Build | **None.** Edit a file, refresh. Static-file test only: `tests/test_api.py:55` (`/js/app.js`, `/css/main.css` must return 200) |
| Run | `venv/bin/uvicorn backend.main:app --port 8000`. Single worker (per-process index and caches) |
| Required network | Browser needs internet for CDNs and fonts. Without marked/DOMPurify, `renderMarkdown` falls back to escaped text (`utils.js:61`). Without hljs, `highlightCode` returns silently (`utils.js:65`) |

Default theme is **dark**: `<html data-theme="dark">` is hardcoded (`index.html:2`), and `UI.initTheme` runs only at
`DOMContentLoaded` (`ui.js:10-13`, `app.js:27`). A light-theme user therefore gets a dark flash on every load (D3).

---

## 3. File inventory

| File | LOC | Responsibility | Public API | Called by |
|---|---|---|---|---|
| `frontend/index.html` | 280 | Static shell: dropzone overlay, upload toast, 2 modals, sidebar, topbar (mode tabs + model select), chat area with welcome, composer dock | n/a | served at `/` |
| `frontend/css/main.css` | 282 | Tokens (`:root` dark, `[data-theme=light]` override), reset, layout, sidebar base, one `@media (max-width:768px)` | n/a | index.html |
| `frontend/css/components.css` | 821 | Buttons, inputs, chat bubbles, citations, code blocks, dock, welcome, tooltips, modals, thought steps, badges | n/a | index.html |
| `frontend/css/animations.css` | 177 | Keyframes, `animate-*` classes, one `prefers-reduced-motion` block (`:172`) | n/a | index.html |
| `frontend/assets/logo.svg` | 24 | Logo; **not referenced anywhere** (sidebar uses an inline SVG, favicon is an emoji data URI `index.html:18`) | n/a | nobody |
| `frontend/js/utils.js` | 159 | Markdown to sanitized HTML, citation badge decoration, code-block wrapper + copy, clipboard, escape, time/date helpers, uuid | `Utils.{renderMarkdown, highlightCode, copyText, escapeHtml, formatTime, formatDateGroup, uuid}` (`:150-158`) | chat.js, ui.js |
| `frontend/js/api.js` | 160 | `fetch` wrapper, REST calls, SSE parser over `fetch` + `ReadableStream` | `API.{listSessions, createSession, getSession, updateSession, deleteSession, setFeedback, listDocuments, uploadFile, uploadUrl, deleteDocument, getModels, chatStream}` (`:146-159`) | app.js only |
| `frontend/js/chat.js` | 409 | Message DOM creation, streaming renderer object, thought steps, badges, action buttons, history rendering, welcome show/hide | `Chat.{appendUserMessage, startAssistantMessage, showThinking, removeThinking, renderHistory, clearChat, copyMessage, getCitation}` (`:399-408`) | app.js, inline `onclick` (`:68,182`) |
| `frontend/js/ui.js` | 346 | Theme, sidebar open/close, session list, doc list, model select, toasts, citation modal, URL modal, dropzone, active mode tab, user-id label | `UI.{initTheme, setTheme, toggleTheme, openSidebar, closeSidebar, toggleSidebar, renderSessionList, renderDocList, renderModelSelect, showUploadToast, hideUploadToast, showToast, showCitationModal, hideCitationModal, showUrlModal, hideUrlModal, initDropzone, setActiveMode, setUserIdDisplay}` (`:325-345`) | app.js, chat.js, inline `onclick` (`index.html:81`) |
| `frontend/js/app.js` | 444 | `State`, init, all event binding, chat pipeline, sessions, documents | none (private IIFE) | boot |

Dead code: `Chat.showThinking/removeThinking` (`chat.js:223-248`) are never called. `UI.setTheme` is called only internally.
No file mixes presentation and API logic cleanly: `app.js` queries the DOM directly (`app.js:55-146,179,287,326,421-423`) and `chat.js`
calls `UI.showCitationModal` (`:165,299`).

---

## 4. DOM contract

JS references ids via `getElementById` only (plus the `closest/querySelector` selectors below). Programmatic cross-check of
`getElementById` targets against `index.html` ids: **no JS id is missing from the HTML.** Unused-by-JS HTML id: `docSection`
(`index.html:117`, styled only). The reverse check for dynamic ids is in the last block.

### Static ids

| id | JS consumer | Purpose |
|---|---|---|
| `sidebar` | `ui.js:38,48,58` | toggles class `mobile-open` |
| `sidebarBackdrop` | `ui.js:39,49` + inline `onclick="UI.closeSidebar()"` (`index.html:81`) | style `display` toggled; inline `style` for layout |
| `sidebarToggle` | `app.js:88`, `ui.js:43,53` | `aria-expanded`. Hidden by CSS above 768px (`components.css:767-768,791`) |
| `newChatBtn` | `app.js:86` | |
| `sessionSearch` | `app.js:128` | client-side title filter |
| `sessionList` | `ui.js:66` | innerHTML target. Delegated per-item listeners bound after each render (`ui.js:115-129`) |
| `docList` | `ui.js:135` | innerHTML target |
| `userIdDisplay` | `ui.js:321` | text `user: <id>` |
| `themeToggle` | `app.js:87`, `ui.js:18` | **text content is the emoji** (`ui.js:19`) |
| `modeTabs` + `.mode-tab[data-mode]` | `app.js:69-72`, `ui.js:311-315` | tabs; `data-mode` must be one of `general, document, research, code` |
| `modelSelect` | `app.js:80,326`, `ui.js:176` | native `<select>`, optgroups per provider |
| `chatArea` | `chat.js:21,47,…`, `app.js:179,287` | `role=log aria-live=polite`; also holds the welcome node |
| `welcomeScreen` | `chat.js:376,390,395` | moved/reinserted by `_resetArea`; display toggled by inline style |
| `starterPrompts` + `.welcome__starter[data-prompt]` | `app.js:137-139` | click fills textarea and sends |
| `chatInput` | `app.js:59,142,159,423` | textarea. Auto-resize max 180px (`app.js:432-435`) |
| `sendBtn`, `stopBtn` | `app.js:55,56,421-425` | visibility toggled with inline `style.display` |
| `attachBtn`, `fileInput` | `app.js:91-95` | `accept=".pdf,.docx,.csv,.xlsx,.txt,.md"` (`index.html:204`) |
| `addUrlBtn` | `app.js:101` | |
| `searchToggle` | `app.js:113-118` | colour via inline `style.color`; `aria-pressed` |
| `uploadToast`, `uploadToastName`, `uploadProgress` | `ui.js:202-221` | progress bar is a fake 2.5s CSS animation (D6) |
| `dropzoneOverlay` | `ui.js:279` | class `active` |
| `citationModal`, `citationModalClose`, `citationSource`, `citationSnippet` | `app.js:122-125`, `ui.js:242-258` | display `flex`/`none` via inline style |
| `urlModal`, `urlModalClose`, `urlModalCancel`, `urlModalSubmit`, `urlInput` | `app.js:102-110`, `ui.js:262-274` | |
| `hljs-theme` | `ui.js:22` | `<link href>` swapped between github-dark and github CSS from cdnjs |

### Selectors and inline handlers

| Selector / handler | Where | Notes |
|---|---|---|
| `.session-item[data-session-id]`, `.delete-session-btn` | `ui.js:95,102,115,124` | generated markup |
| `.message__regen`, `.message--assistant` | `app.js:179,287` | app.js depends on chat.js class names |
| `.citation` (`<sup data-id>`), `.citation-card[data-cid]` | `chat.js:161,301,303`, `utils.js:35-36` | |
| `.message__extra-actions`, `.thought-step__{icon,label,time}` | `chat.js:307,117-119` | |
| `.code-block__copy`, `pre code` | `utils.js:66,90` | |
| `onclick="Chat.copyMessage('<uuid>')"` | `chat.js:68` | |
| `onclick="UI.showCitationModal(Chat.getCitation('<uuid>',<int>))"` | `chat.js:182` | |
| `onclick="UI.closeSidebar()"` | `index.html:81` | |

### Dynamic ids (created in `chat.js`, all consumed in `chat.js`)
`thoughtSteps-<id>` (`:61,98`), `streamContent-<id>` (`:63,85,154,200,213`), `citationCards-<id>` (`:65,177`),
`msgBadges-<id>` (`:66,130`), `msgExtra-<id>` (`:71,143`), `msgTime-<id>` (`:72,171`), `thought-<id>-<uuid>` (`:102`, unused as a lookup),
`thinkingPlaceholder` (`:226`, dead path). `renderHistory` messages have **no ids** and no copy button (`chat.js:274-292`).

Inline `style="..."` attributes in `index.html`: 12. JS `.style.` writes: ui.js 13, app.js 5, chat.js 2. These must be
replaced by class toggles during the redesign or they will override new CSS.

---

## 5. State and persistence map

### Client `State` (`app.js:11-23`, private, not exported)
| Field | Source / persistence | Notes |
|---|---|---|
| `userId` | `localStorage.userId` or `"default"`; **read-only, nothing writes it** | sent on session create, list, upload, doc list/delete, chat payload |
| `sessionId` | memory only | lost on reload: the app always opens on an empty chat, with no auto-restore of the last session |
| `mode` | `localStorage.mode` (written `app.js:74`) | on `_selectSession` it is overwritten from the session **without** writing localStorage (`app.js:321`) |
| `model`, `defaultModel`, `models` | `localStorage.model`; list from `GET /models` | reset to default/first usable if stored id is not available (`app.js:44-48`) |
| `sessions`, `documents` | memory, reloaded from server | |
| `webSearch` | memory only, default `true` (`app.js:20`), **not persisted** | |
| `isLoading`, `abort` | memory | `abort` is the `AbortController` (`app.js:205`) |

### Other client state
- `localStorage.theme` (`ui.js:11,17`), values `dark` / `light` only.
- DOM-held state: `Chat._citations` registry keyed by client uuid (`chat.js:14`); per-message raw text in closures (`chat.js:80`); sidebar open class; modal display styles.
- No `sessionStorage`, cookies, IndexedDB, service worker.

### Server persistence
SQLite (`data/chatbot.db`): `ChatSession(id, user_id, title, model, mode, created_at, updated_at)`, `Message(role, content, citations, feedback, meta, created_at)`,
`Document(id, filename, file_type, chunk_count, indexed_at)` plus FAISS/BM25 index under `data/`. Auto-title = first 6 words of the first user message (`session_service.py:108-110`).
Message `meta` stores `model, route, mode, fell_back_from, cached, timings, usage, partial|stopped` (`chat_service.py:277-283,305-307`). The UI renders only `model` and `cached` from history (`chat.js:286`).

**Documents are per `user_id`, not per chat session**: retrieval runs over all of the user's documents in any mode when any exist
(`chat_service.py:91-92` `RouteDecision(has_docs, ...)`; `:159`). Mode `document` only changes the prompt and notes (`:236-241`).

---

## 6. API and SSE inventory

### Endpoints (`backend/api/routes.py`)
| Endpoint | UI uses? | Caller | Notes |
|---|---|---|---|
| `GET /ping` | no | n/a | |
| `GET /health` | **no** | n/a | shape verified live: `{status, services{database,groq,google,tavily,embeddings,reranker}, details{...}}` |
| `GET /models` (`?refresh=1`) | yes (no refresh) | `app.js:37` | live shape: `{models:[{id,label,provider,description,available,reason,default}], default}`. Providers seen: `groq`, `google`. 8 models |
| `POST /chat/stream` | yes | `app.js:207` | payload `{session_id, message, user_id, mode, model|null, web_search, regenerate}`. `message` max 20000 (`models/api.py:91`). **`user_id` is ignored by the pipeline** (the session's owner is used, `chat_service.py:138`); it is used only for the rate-limit key (`routes.py:126`) |
| `POST /chat`, `POST /generate`, `POST /retrieve`, `POST /search` | no | n/a | not needed by redesign. `/search` returns `{results:[{title,url,snippet,score}], status}` and could back a source list, but it spends Tavily quota |
| `GET/POST /sessions`, `GET/PATCH/DELETE /sessions/{id}` | list/create/get/**PATCH (mode, model only)**/delete | `api.js:39-57` | **UI never sends `title`** (rename is supported server-side, not exposed). No `pinned` field anywhere |
| `POST /sessions/{id}/messages` | no | n/a | |
| `POST /messages/{id}/feedback` | yes | `api.js:59` | value -1/0/1 |
| `POST /upload` (multipart `file` or `?url=`) | yes | `api.js:72-82` | returns `{document_id, filename, file_type, chunk_count, user_id, indexing}` **after** indexing completes |
| `GET /documents`, `DELETE /documents/{id}` | yes | `api.js:68,84` | doc fields: `id, filename, file_type, chunk_count, indexed_at` only |

### SSE events (`chat_service.py`), and what `app.js:217-238` does with each
| Event | Payload | UI handling |
|---|---|---|
| `meta` | `route, mode, reason, requested_model` | **no handler** |
| `status` | `stage` rewriting/retrieving/searching/generating; `state` start/done/first_token; `elapsed_ms`, `detail`, `error`, `mode`, `reranked` | thought-step cards (`app.js:188-232`) |
| `citations` | `items[{id,type,source,page,url,snippet}]` (`prompt_builder.py:87-101`) | stored, rendered as badges and cards, **`url`/`type` ignored** |
| `model` | `model`, `fell_back_from[]` | stored; badge shown after the stream (`chat.js:129-139`). No live "fell back" notice during the stream |
| `token` | `text` | appended; full Markdown re-render of the whole message every 60 ms (`chat.js:123-125`) |
| `error` | `code, message, retryable, attempts?, retry_after?, partial?, message_id?` | only `message`, and `message_id` used. **`code`, `retryable`, `retry_after`, `partial` unused** |
| `done` | `message_id, model, fell_back_from, cached, route, persisted, timings, usage` | `message_id` and `cached` used |

SSE parser: splits on `\n\n`, handles `event:`/`data:` lines, swallows non-JSON blocks silently (`api.js:115-143`).
It does not handle `\r\n` framing (inferred: the server emits `\n`, so fine today).

### Error / cancel / fallback / rate-limit behaviour today
- **HTTP errors** (before the stream starts: 404 session, 503 no provider, 429 per-user throttle `routes.py:68-75`, 502): `_errorMessage` returns `detail` if a string, else `message`, else `HTTP <n>` (`api.js:10-18`). `Retry-After` is not read. FastAPI 422 (`detail` is an array) becomes plain `HTTP 422`. Shown via `renderer.showError` (`app.js:268`), in the assistant bubble, with **no retry control**.
- **Stream `error` event**: if no text yet, shows the message in the bubble (`app.js:242-244`). If partial text exists, finalizes and toasts (`:258`). `rate_limited` is not distinguished from other codes. There is no retry button, no countdown, and the partial flag is not displayed.
- **Cancel**: Stop calls `State.abort.abort()` (`app.js:277-279`). The catch block finalizes the partial answer and toasts "Generation stopped" (`:261-265`). The server persists the partial (`chat_service.py:300-307`). The stopped message gets **no actions** (no regenerate/feedback) because there is no `message_id`.
- **Fallback**: only visible as a "fallback" badge with a title listing the prior models (`chat.js:135-137`); not shown for restored history even though `meta.fell_back_from` is persisted.
- **Duplicate submit**: `State.isLoading` guard (`app.js:158`) plus `textarea.disabled` while loading (`:427`). The session-create step happens after `_setLoading(true)`, so it is covered.
- **Input restoration**: **none.** `textarea.value = ""` happens before the request (`app.js:163-166`). If session creation fails, the text is lost (`:171-175`).
- **Regenerate**: the last assistant node is removed from the DOM first (`app.js:289`), then the server pops the last exchange when the stream starts (`session_service.py:127-139`). If the request fails before the stream (429/503), the DOM answer is gone but the server still has it (inferred from the order of operations; reload restores it).
- Note: Enter submits even during IME composition (`app.js:61` has no `isComposing` check) (D9).

---

## 7. Feature matrix (master prompt vs repo)

Status key: exists / partial / missing in UI / blocked by backend / contradicts backend reality.

### Phase 4: layout and sidebar
| Requirement | Status | Evidence |
|---|---|---|
| Two regions only, no right sidebar | exists | `index.html:86-258` has only `aside.sidebar` and `main.main` |
| Soft light-lavender sidebar, not dark in light mode | missing in UI | light tokens are white/grey-blue (`main.css:70-79`); sidebar uses `--bg-surface` (`main.css:121`). Dark theme is the default |
| Brand: logo, name, tagline | partial | inline SVG + name (`index.html:92-99`); no tagline; `assets/logo.svg` unused |
| Sidebar collapse (desktop) | missing in UI | CSS `.sidebar.collapsed` exists (`main.css:132`), **no JS or control** |
| Mobile drawer | partial | bottom sheet 75dvh, not a side drawer (`main.css:262-278`); no focus trap, no Esc, no `aria-modal`; backdrop is a `div` with inline `onclick` |
| New Chat primary action | exists | `#newChatBtn` (`index.html:100`), `app.js:304-313`. It creates an empty server session immediately, so "New Chat" sessions pile up (visible in live `/sessions`: untitled "New Chat" rows) |
| Nav General/Documents/Research/Code | partial | exists as top-bar tabs (`index.html:145-158`), not in the sidebar. Value for documents is `document` |
| Conversation search | partial | client-side title-only filter (`app.js:128-132`); empty result prints "No chats yet" (`ui.js:69-72`), which is misleading |
| Date groups Today/Yesterday/7d/30d/Earlier | exists | `ui.js:82`, `utils.js:130-141`. Wrong for non-UTC users (D1) |
| Titles, timestamps | exists | `ui.js:97-99` (time only; no date) |
| Rename | missing in UI (backend supports) | `PATCH /sessions/{id}` accepts `title` (`models/api.py:22-25`, `session_service.py:73-74`); `API.updateSession` already forwards any body |
| Pin | **blocked by backend** | no field in `SessionUpdate`/`SessionSchema`/DB model |
| Delete | exists, uses `confirm()` | `app.js:335-336` |
| Empty / loading states for history | partial | empty exists; **no loading state**; errors swallowed (`app.js:354`) |
| My Documents entry, real count/status | partial | doc list with filename, type, tooltip with chunk count (`ui.js:134-171`); count is available client-side (`State.documents.length`); **status is not available** (section 8) |
| User section: existing user info | partial | only `user: <userId>` text (`index.html:126-128`). There are no accounts |
| Settings entry | missing in UI | nothing to hold; only theme exists |
| Theme preferences | partial | single toggle button in the sidebar footer (`index.html:129`) |

### Phase 5: theme
| Requirement | Status | Evidence |
|---|---|---|
| Light/Dark | exists | `data-theme` on `<html>`, tokens `main.css:10-79`, `ui.js:15-28` |
| System preference | missing in UI | no `prefers-color-scheme` anywhere (`grep`) |
| Sun/moon control in top nav | missing in UI | control is in the sidebar footer and uses emoji text |
| No flash on load | missing in UI | hardcoded `data-theme="dark"` and no early inline script (D3) |
| Persist | exists | `localStorage.theme` |
| Whole-UI theming | partial | 54 hard-coded colour literals in `components.css`; light overrides only 11 tokens (`main.css:70-79`); accents/gradients unchanged; `[data-theme]` selectors appear only once. Needs a token pass |
| Visible focus both themes | partial | `.btn:focus-visible` only (`components.css:25`); inputs `outline:none` with border colour; no global focus ring |
| Selected state not by colour alone | missing in UI | |
| hljs theme switch | exists | `ui.js:22-27` (swapping `href` triggers a CDN fetch and a flash) |

### Phase 6: animated AI identity
| Requirement | Status | Evidence |
|---|---|---|
| Animated visualization with idle/processing/responding/error | missing in UI | welcome uses a floating 🤖 emoji (`index.html:174`, `components.css:650`); `float` keyframes exist (`animations.css:78`). No state wiring |
| Reduced motion | partial | global duration clamp (`animations.css:172-177`); the `float` infinite animation is shortened, not disabled |
| Signals available to drive states | exists | `status` events (processing), first `token` (responding), `error` event, `isLoading`. All in `app.js:217-238` |

### Phase 7: welcome
| Requirement | Status | Evidence |
|---|---|---|
| Heading "What can I help you with?" + supporting text | missing in UI | current: "AI Assistant Pro" + a long description (`index.html:175-178`) |
| Four feature cards with click action | missing in UI | six pill starters (`index.html:179-186`) that **send immediately** in the current mode, without switching mode. The "Summarize my document" starter fires with no document check |
| Cards map to modes | feasible | mode switch = `State.mode` + `UI.setActiveMode` (`app.js:69-77`) |
| Capabilities strip | optional | do not make claims beyond: multiple models (`/models`), file upload, web search (Tavily, when `health.services.tavily=configured`), no privacy claim can be backed from the repo |

### Phase 8: general chat
| Requirement | Status | Evidence |
|---|---|---|
| User/assistant hierarchy | exists (heavy bubbles) | `chat.js:27-35,53-75` |
| Markdown, lists, tables | exists | marked + GFM (`utils.js:8-13`). Table CSS: not verified |
| Syntax highlighting, language label, copy code | exists | `utils.js:64-101`. Only after `finalize`, so no highlight while streaming |
| Copy response | partial | live messages only (`chat.js:68`); **history messages have no copy button** (`chat.js:287-291`) |
| Regenerate | partial | only for the newest live answer (`app.js:179`); not after reopening a session |
| Feedback | exists | `chat.js:333-373`, restored from history (`:306-311`); failures silently ignored (`:365`) |
| Streaming | exists | `app.js:205-240` |
| Stop | exists | `#stopBtn` (`index.html:242`) |
| Loading indicators | exists | thought steps with spinner (`chat.js:96-110`). `showThinking` is dead |
| Error + retry | partial | error text only (`chat.js:195-205`); **no retry** |
| Persistence | exists | server-side, loaded on selecting a session (`app.js:315-333`). No auto-restore on reload |
| Scroll behaviour | partial | always scrolls to bottom on every render (`chat.js:383-387`), even if the user scrolled up |
| Fallback visibility | partial | badge only after completion |

### Phase 9: composer
| Requirement | Status | Evidence |
|---|---|---|
| Auto-grow multiline | exists | max 180px (`app.js:432-435`) |
| Attachment control + existing upload | exists | `#attachBtn` to `#fileInput` to `_handleFileUpload` (`app.js:91-98,358-377`); single file; drop uses `files[0]` only (`ui.js:304`) |
| Web-search toggle | exists, semantically weak | server only searches on "fresh info" regex hits, or always in research mode (`chat_service.py:85-92`). The toggle has no effect in research mode, and its UI state is inline colour |
| Selected-model indicator in composer | missing in UI | model select lives in the top bar only |
| Attached-file chips | **contradicts backend reality** | uploads are per-user documents, not per-message attachments. A chip can only mean "an upload in progress" or "a document already indexed" |
| Upload progress | **blocked by backend/transport** | `fetch` gives no upload progress and the server gives no indexing events; the current bar is a fake 2.5s animation (D6). Honest option: indeterminate state |
| File validation feedback | partial | extension check + toast (`app.js:359-364`); server 413/415 message shown on failure |
| Disabled/loading states | partial | send/stop swap, textarea disabled while loading |
| Keyboard | exists | Enter sends, Shift+Enter newline (`app.js:60-66`); IME bug (D9) |
| Preserve input on failure | missing in UI | `app.js:163-166` |
| Add URL | exists (extra) | `#addUrlBtn`, modal (`index.html:60-75`). Not in the master prompt, but "do not remove working features" applies |

### Phase 10: model selection
| Requirement | Status | Evidence |
|---|---|---|
| Names, provider, selected | exists | native select, optgroups, `label — description` (`ui.js:175-198`) |
| Verified capabilities, latency, price | **blocked by backend** | `/models` has only `id,label,provider,description,available,reason,default` (verified live) |
| Search/filter | missing in UI | 8 models only; optional |
| Loading/error states | partial | placeholder "Loading models…" is never replaced on failure: `State.models=[]` then `renderModelSelect` yields an empty select (`app.js:40-49`) |
| Unavailable models | exists | `option.disabled` + `title=reason` (`ui.js:191-192`; native option titles are not accessible) |
| Fallback | exists | server-side; badge on message |
| Per-session model | exists | PATCH on change (`app.js:83`); restored on session select (`:323-328`) |

### Phase 11: documents workspace
| Requirement | Status | Evidence |
|---|---|---|
| Upload + drag-drop | exists | overlay (`index.html:26-32`), `ui.js:278-306` |
| Supported types and limits from backend | **blocked by backend** | no endpoint; backend allows `pdf docx csv xlsx txt md rst` (`parser.py:20`), frontend omits `rst` (D2). Limit is `MAX_UPLOAD_MB` default 20 (`config.py:185`), enforced in `routes.py:256-260` (413) |
| Document list | partial | sidebar rows only (name truncated to 22 chars) |
| Filenames and sizes | partial / **blocked** | filename yes; **size not stored or returned**. Only `File.size` at upload time |
| Processing/indexing status | **blocked by backend** | `POST /upload` is synchronous; no per-document status field; `indexing` mode (hybrid/lexical) is returned only in the upload response (`app.js:370`) and not persisted |
| Error and retry | partial | error toast; no retry |
| Existing doc Q&A | exists | via `mode: document` chat; citations opened in a modal |
| Delete | exists, `confirm()` | `app.js:399-408` |
| Dedicated workspace view | missing in UI | there is no document view; the mode tab only changes the chat mode |
| Drop only works through `files[0]` | partial | multi-file drops silently ignore the rest |

### Phase 12: research workspace
| Requirement | Status | Evidence |
|---|---|---|
| Research input, loading, results | partial | mode `research` through the chat pipeline; thought steps show "Searching the web…" (`app.js:151-155`) |
| Source titles | exists in data | web citation `source = title or url` (`prompt_builder.py:100`) |
| Domains | derivable | from `url` (client-side parse). **Treat as untrusted** |
| Clickable links | missing in UI | `url` is never rendered; the citation modal shows source + snippet only (`ui.js:242-253`) |
| Sources distinguishable from summary | partial | citation cards under the bubble (`chat.js:175-189`); `type` ignored |
| Dedicated results UI | **contradicts backend reality if interpreted as separate** | there is no research endpoint; `/search` is a raw Tavily wrapper that costs quota. Keep Research as a chat mode |
| Empty / error / retry | partial | see section 6 |

### Phase 13: code workspace
| Requirement | Status | Evidence |
|---|---|---|
| Markdown, highlight, language labels, copy, horizontal scroll | exists | `utils.js:64-101`; scroll CSS not verified |
| Code mode | exists | `code` prompt (`prompt_builder.py:39-42`); mode tab `🧮` |
| Execution / preview | n/a | no support in the backend; do not add |

### Phases 14-17: icons, libraries, responsive, a11y
| Requirement | Status | Evidence |
|---|---|---|
| Lucide or consistent icon set | missing in UI | emoji for mode tabs, doc types, session icons, toasts, thought steps, buttons (`index.html`, `ui.js:91,143,224`, `chat.js:134,284,347-354`, `app.js:151-154`); a mix of inline SVG in the composer. No icon module |
| Tooltips for icon-only controls | partial | CSS-only `[data-tooltip]` shows on `:hover` only (`components.css:668-688`), not on keyboard focus; some controls use `title` |
| Dialogs/confirmations | partial | custom modals for URL + citation; **native `confirm()`** for deletes (`app.js:336,400`) |
| Context menus / ellipsis menus | missing in UI | none |
| Responsive | partial | a single breakpoint at 768px; at 390px the model select is clipped and the composer placeholder is truncated (screenshot) |
| Dialog semantics, focus trap, Esc | missing in UI | no `role=dialog`, `aria-modal`, Esc handler (grep) |
| Semantic landmarks | partial | `aside`, `main`, `header`, `section role=log`, `footer role=contentinfo`; tab roles without arrow-key support (`index.html:145-158`) |
| Live region | risk | `aria-live=polite` on the whole chat area while innerHTML is re-rendered every 60 ms (inferred noisy for screen readers) |
| Reduced motion | partial | see Phase 6 |
| Authentication/authorization | **contradicts backend reality** | none exists (section 8) |

---

## 8. Backend gaps and contradictions

Each: what is missing, honest UI fallback, smallest backend change *if the user wants it* (not required unless noted).

| # | Gap / contradiction | Evidence | Honest UI fallback | Smallest backend change |
|---|---|---|---|---|
| B1 | **No authentication/authorization.** `user_id` is client-chosen (default `"default"`). `GET/PATCH/DELETE /sessions/{id}` and `POST /messages/{id}/feedback` take **no user check**: anyone with an id can read or delete a session | `routes.py:197-239`; `app.js:12` | Keep sending `user_id` exactly as now. No login UI, no avatar, no claims. Show `user: <id>` neutrally | none for the redesign. A security finding for the owner |
| B2 | **Pin not supported** | `models/api.py:22-25,50-60` | Do not ship a pin control. Offer rename + delete only | add `pinned: bool` to `ChatSession`, `SessionUpdate`, `SessionSummary`, plus an alembic-less column migration |
| B3 | **Document size and status not stored or returned** | `models/api.py:167-174` | Show filename, type, chunk count, and relative `indexed_at`. Show size only for the in-flight upload from `File.size`. Show "Indexing…" only while the POST is pending, "Indexed" only after resolve | add `size_bytes` (and optional `status`) to `Document` + `DocumentSchema` |
| B4 | **No progress for uploads/indexing** | `routes.py:244-266` (synchronous) | indeterminate spinner and elapsed time. Do not animate a percentage | use XHR for byte progress (frontend only); for indexing progress there is no backend event stream |
| B5 | **No endpoint for supported types or size limit** | `parser.py:20`, `config.py:185` | hardcode the frontend list **to match** `SUPPORTED_EXTENSIONS` (pdf, docx, csv, xlsx, txt, md, rst); label the limit as "up to the server limit (default 20 MB)" or show the server's 413 message verbatim | add `GET /config` or extend `/health` with `{upload:{max_mb, extensions}}` |
| B6 | **Model capabilities, latency, price absent** | `llm_service.py:190-199` | show label, provider, description, availability + reason only | none; do not add |
| B7 | Research and Code are chat modes; Research ignores the web toggle; docs apply in every mode | `chat_service.py:85-92` | present Research and Code as modes of one composer, and explain "web search is always used in Research" | none |
| B8 | Per-chat "attachments" do not exist (documents are per user) | `chat_service.py:159` | upload chip = transient status; documents list = persistent. Word the UI as "your documents" | a per-session document scope would be a new feature, not needed |
| B9 | Session timestamps are naive UTC with no `Z` | `utils/timeutil.py:13-15`, `models/api.py:69-70` live JSON `"2026-10-14T11:23:30.234970"` | parse as UTC client-side (append `Z`) in `Utils` | emit timezone-aware datetimes |
| B10 | Stream `error` has useful fields the UI ignores (`retryable`, `retry_after`, `partial`) and the throttle 429 carries `Retry-After` | `chat_service.py:111-112,289-290`, `routes.py:72-75` | map to retry state, partial notice, countdown | none |
| B11 | `GET /health` is not used | `routes.py:103-106` | optional degraded banner, e.g. when `services.groq` and `google` are not configured | none |
| B12 | `message_id` missing for stopped/aborted answers on the client | `app.js:261-265` vs `chat_service.py:302-307` (persisted without notifying) | no regenerate/feedback on stopped partials, or reload the session to obtain ids | emit a final `done`-like event on abort (not possible after disconnect) |

---

## 9. Defects and risks observed (verified unless marked)

| ID | Finding | Evidence |
|---|---|---|
| D1 | **Timezone skew**: server timestamps are naive UTC; `new Date("2026-10-14T11:23:30.234970")` parses as local time. Day grouping and times are wrong for non-UTC users. Live check: the session created at `11:23:30` is shown as "11:23 AM" (this machine's TZ not recorded, so the skew itself is inferred) | `utils.js:124-141`, `timeutil.py:13-15`, `baseline/api-shapes.txt`, `dark-1440.png` |
| D2 | **Type whitelist drift**: backend accepts `rst`; frontend `ALLOWED` and `accept=` omit it; the dropzone text omits "rst" and says "web URLs" | `parser.py:20`, `app.js:359`, `index.html:30,204`, `skills.md:51` |
| D3 | **Theme flash**: `data-theme="dark"` hardcoded; theme applied after load; the hljs `<link>` also defaults to github-dark | `index.html:2,15`, `ui.js:10-13` |
| D4 | **Input lost on failure** (cleared before request; session-create failure) | `app.js:163-175` |
| D5 | **Regenerate removes the answer before the request**, so a pre-stream failure leaves the UI inconsistent with the server | `app.js:286-291` |
| D6 | **Fake upload progress bar** (CSS animation, 2.5s `both`, unrelated to the request) | `ui.js:208-214` |
| D7 | `window.CONFIG.BACKEND_URL` hook is unreachable: no script tag loads `config.js`. | `api.js:6-8`, `index.html:273-277`, `.gitignore:22` |
| D8 | Native `confirm()` for delete session and delete document | `app.js:336,400` |
| D9 | Enter sends during IME composition | `app.js:61` |
| D10 | Emoji used as icons everywhere, including interactive state (theme button text, toasts, regenerate `↻`, feedback `👍👎`, close `✕`) | `ui.js:19,222-225`, `chat.js:347-354`, `index.html:129,50` |
| D11 | `Chat.showThinking/removeThinking`, `docSection` id, `assets/logo.svg`, `.sidebar.collapsed` CSS are dead | section 3, 4 |
| D12 | Model select has no failure state: stays empty, send still uses `model:null` (`app.js:213`), so the server default applies (not a break, but silent) | `app.js:35-50` |
| D13 | Search filter empty result shows "No chats yet" | `ui.js:69-72` |
| D14 | `createSession` on every "New" click creates an empty server row; abandoned empty "New Chat" sessions accumulate (live list shows one) | `app.js:304-313`, `baseline/api-shapes.txt` |
| D15 | History rendering lacks copy button, regenerate, fallback badge, and mode/route info | `chat.js:250-322` |
| D16 | `Utils.formatTime` shows time only; there is no date for older items | `ui.js:93` |
| D17 | Unknown `d.model` assumption: `s.model.split("/")` throws if a session had a null model (schema says `str`, so low risk) | `ui.js:99` |
| D18 | Chat area `aria-live` + full-message innerHTML re-render every 60 ms (inferred a11y/perf cost for long answers: O(n^2) parse work) | `chat.js:83-90,123-126` |
| D19 | Tooltips appear only on hover, not on focus | `components.css:688` |
| D20 | Always auto-scrolls to bottom during streaming, fighting a user reading earlier text | `chat.js:83-90,383-387` |

### Security observations on the rendering path
- Markdown: `DOMPurify.sanitize(marked.parse(text), {USE_PROFILES:{html:true}})` (`utils.js:55`), then `decorateCitations` re-parses the *sanitized* HTML into a `<template>` and only adds `<sup>` nodes (`:20-46`). Sound. Falls back to escaped text.
- `html` profile permits `<img>`/`<a href>` (inferred from DOMPurify defaults): a model/web-sourced image URL will be fetched by the browser (tracking/exfil by URL) and links open in the **same tab with no `rel`/`target` hook** (no `DOMPurify.addHook`). Redesign should decide link policy (target=_blank + `rel="noopener noreferrer"`) and whether to allow remote images.
- `innerHTML` sites: `chat.js:29,56,104,138,157,179,203,208,228,259,274`; `ui.js:70,95,112,139,151,183,230,250`; `utils.js:77,92,94`. Escaped via `Utils.escapeHtml`: source, title, snippet (as attribute), model, filename, file_type, toast message, user message. **Unescaped interpolations**: `citation.page` / `c.page` (`chat.js:180,284`, `ui.js:249-250`), typed `Any` server-side (`models/api.py:32`); `s.id` (`ui.js:95,102`), `c.id` (`chat.js:182`, `int`), `lang` (`utils.js:73`, `\w+`-constrained). `page` values are produced by the backend parser (int or row-range string), so the risk is low but not zero; escape it.
- Citation `url`: not rendered anywhere today. When rendered, validate `^https?://` (a `javascript:` URL in the `url` field would otherwise be dangerous) and use `textContent` / DOM APIs.
- Inline handlers: 3 (`index.html:81`, `chat.js:68,182`). They embed only client-generated uuids and ints. Moving to `addEventListener` removes any future CSP constraint.
- No CSP, no SRI on CDN scripts, floating majors for marked/DOMPurify.
- Secrets: no API keys in frontend files (grep of `frontend/`: none).

---

## 10. Baseline verification

| Command | Result |
|---|---|
| `venv/bin/pytest` | exit 0. `156 passed, 6 deselected in 16.87s` (`baseline/pytest.log`). The 6 deselected are the `live` marker; **not run** |
| `venv/bin/ruff check backend tests --select F,E9,B --ignore B008,B905,B011` | exit 0, `All checks passed!` (`baseline/ruff.log`) |
| `for f in frontend/js/*.js; do node --check "$f"; done` | all 5 exit 0 (`baseline/node-check.log`) |
| Server | nothing was on `:8000`. Started `venv/bin/uvicorn backend.main:app --port 8765 --workers 1`, got `{"pong":true}`, then stopped it (`ss -ltn` confirms nothing on 8765) |
| GET-only calls | `/models`, `/health`, `/documents` (`[]`), `/sessions?user_id=default` (4 rows of the local user's data; titles are in `baseline/api-shapes.txt`, delete the file if that is sensitive) |
| Not run | `pytest -m live`, any POST/PATCH/DELETE, `/models?refresh=1`, browser interaction (no Playwright/Puppeteer, so only the initial render was captured) |

Screenshots (headless Chrome, `--virtual-time-budget=6000`, **dark theme only**, the app default):
`baseline/dark-1440.png`, `baseline/dark-768.png`, `baseline/dark-390.png`. The light theme could not be captured: `localStorage.theme`
cannot be set from the CLI and no scripted browser is available. What the screenshots show:
- **1440**: 280px sidebar with brand, blue "+ New" pill, search box, grouped chat list (Previous 7 Days / 30 Days) with emoji mode icons, "MY DOCUMENTS" with "No documents indexed yet", footer `user: default` + moon button. Top bar: four emoji + text tabs and a full-width native model select. Centre: floating robot emoji, gradient "AI Assistant Pro", paragraph, six pill starters. Dock with attach / globe / search-toggle icons, textarea, gradient send button, hint line. The composer is visually offset relative to the content column (the left tool group sits outside the input).
- **768**: sidebar hidden, hamburger shown, tabs collapse to emoji-only, same welcome.
- **390**: model select is **clipped at the right edge** ("GPT-OSS 120B — …" cut off), the textarea placeholder wraps/clips to two lines, the hint line wraps. Tab labels hidden (icon-only, no tooltips).
No horizontal scrollbar was visible (`--hide-scrollbars` is on, so this is not conclusive).

---

## 11. Recommended implementation plan

Constraints: owners per `PROJECT_BRIEF.md` section 2; backend frozen; `/js/app.js` and `/css/main.css` must remain served
(`tests/test_api.py:55`); every new file lives under `frontend/{css,js,assets}`.

**Stage 2: design foundation** (ui-designer, then frontend-developer)
- `ui-designer` -> `docs/redesign/design-spec.md`: tokens, type scale, 8px grid, component list, states, light/dark/system, icon set choice (Lucide as a self-hosted SVG sprite or `icons.js` under `frontend/`; avoid a CDN dependency), the two-region shell, welcome copy, the honest-fallback rules from section 8.
- `frontend-developer` (owns `index.html`, `css/**`, `assets/**`, `ui.js`, `chat.js`, `utils.js`, new `icons.js`, `theme.js`, `components/**`, `view-contract.md`):
  1. tokens in `main.css` (replace the 54 literals; add light/dark/system via `data-theme`), early inline theme script in `<head>` (fixes D3);
  2. `theme.js` (light/dark/system) while keeping `UI.initTheme/setTheme/toggleTheme`;
  3. `icons.js`, replacing emoji, keeping all existing ids.
- Sequencing risk: add new functions/ids, never rename existing ones until `integration-developer` updates callers.

**Stage 3: main shell** (frontend-developer; integration-developer for the callers)
- Sidebar: brand, New Chat, nav, history groups (add the date to `Utils.formatTime`/UTC fix, D1, in `utils.js`), My Documents, user/settings/theme, desktop collapse, mobile left drawer with focus trap and Esc. Top nav: compact, theme control, model selector.
- Breaking changes (if any) listed in `view-contract.md`. `integration-developer` (owns `app.js`, `api.js`, `workflows/**`) updates event binding, and uses `UI.openSidebar/close` for focus handling. New view function for rename; new controller code to call `API.updateSession(id,{title})`.
- Do not ship pin (B2).

**Stage 4: core experience** (frontend-developer, with integration-developer for wiring)
- AI visualization as a view with `setState('idle'|'processing'|'responding'|'error')`. `integration-developer` calls it from the existing `status`/`token`/`error`/`done` handlers (`app.js:217-238`).
- Welcome with four cards. Cards call a controller callback that sets the mode and focuses the composer (do not auto-send).
- Composer: model indicator, upload chips (transient), web toggle (disabled with an explanation in Research), stop. Model selector: custom listbox needs a view with data + callbacks; the controller supplies `State.models` and shows a real error state (D12).
- Needs `frontend-developer` to add `<script>` tags for any `workflows/*.js` (cross-boundary request).

**Stage 5: integration** (integration-developer)
- Documents view: list from `GET /documents`, upload with `File.size`, indeterminate progress (B4), supported types aligned with `SUPPORTED_EXTENSIONS` (D2), delete via a view dialog (replaces `confirm()`), real counts.
- Research: render citation `url` and domain through a view that validates `https?:` and sets `rel="noopener noreferrer"`. Code: reuse the existing code-block pipeline.
- Chat: restore input on failure (D4), fix regenerate ordering (D5), map `error.code/retryable/retry_after/partial` to view calls, live fallback notice, history parity (D15), optional `/health` banner.
- Fix `localStorage` handling of `mode` on `_selectSession` (section 5).

**Stage 6: polish** (both)
- Focus ring and tooltips on focus, ARIA for dialogs/menus, reduced-motion, smart auto-scroll, throttle rendering (D18, D20), IME check (D9), 390px overflow, 768-1024 range.

**Stage 7: verification** (qa-reviewer)
- Re-run section 10 commands. Add `tests/test_frontend_*.py` (new file only) with static checks: every `getElementById` id in JS exists in `index.html`, `ALLOWED` extensions equal `SUPPORTED_EXTENSIONS`, no `confirm(`, no hardcoded model ids. Manual verification of streaming/upload/research needs a real browser (no Playwright installed). Live flows consume quota and need the orchestrator's approval.

**File to owner (exclusive)**
| File | Owner |
|---|---|
| `frontend/index.html`, `css/*`, `assets/*`, `js/ui.js`, `js/chat.js`, `js/utils.js`, new `js/icons.js`, `js/theme.js`, `js/components/**`, `docs/redesign/view-contract.md` | frontend-developer |
| `frontend/js/api.js`, `js/app.js`, `js/workflows/**`, `docs/redesign/integration-notes.md` | integration-developer |
| `docs/redesign/design-spec.md` | ui-designer |
| `tests/test_frontend_*.py`, `docs/redesign/qa-report-NN.md` | qa-reviewer |
| `backend/**`, existing `tests/**`, `README.md`, `skills.md` | nobody |

Seam note: `chat.js` (frontend-developer) currently calls `UI.showCitationModal` and registers feedback/regenerate callbacks supplied by `app.js`;
that boundary already matches the view/controller split. But `app.js` directly touches DOM (`app.js:55-146,179,287,326,421-435`),
`chat.js` holds the citation registry, and `ui.js` writes inline styles. Each is a coupling point to resolve through `view-contract.md`.

Risks: (1) ordering of renames, since DOM ids are consumed in two owners' files; (2) CDN dependence (offline/air-gapped fails); (3) icon library
without a build step means self-hosting SVG; (4) no browser automation, so most UX verification is manual; (5) free-tier quota limits live testing;
(6) the light theme has never been rendered here.

---

## 12. Open questions for the user

1. **Visual references**: the master prompt says "supplied visual references", but `docs/redesign/references/` does not exist. Are there references or a Figma file? Without them the designer works from the written palette only.
2. **Backend changes**: is a small backend change acceptable for any of: pin (B2), document size/status (B3), `/config` for types and limit (B5), timezone-aware timestamps (B9)? Default assumption: no, use the fallbacks in section 8.
3. **Icon delivery**: self-hosted Lucide SVG under `frontend/assets/` (no CDN, adds ~files) vs a CDN script. Which is acceptable?
4. **Offline/CDN policy**: keep marked/DOMPurify/hljs on CDNs (as today) or vendor them under `frontend/js/vendor/`? Vendoring is a repo-size and licence decision.
5. **Default theme**: today dark. Should the default become System?
6. **Link/image policy** for model-generated Markdown: open links in a new tab (`noopener`)? Allow remote images?
7. **Empty "New Chat" rows** (D14): may the UI defer session creation until the first message? That changes when `POST /sessions` is called, but not the contract.
8. **"My Documents" in the sidebar vs a dedicated Documents workspace**: should Documents mode show a document-management view, or only the chat with the list in the sidebar? The backend has no extra data to fill a larger page (B3).
9. **Authentication wording**: the master prompt says "preserve authentication". Confirm the intent is "keep `user_id` as is" and that no login is expected.
