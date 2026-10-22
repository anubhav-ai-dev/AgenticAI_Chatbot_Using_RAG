# Integration notes (controller layer: api.js, app.js)

State: Stage 4 backlog implemented in `frontend/js/app.js` + `frontend/js/api.js`. No `js/workflows/` modules were created (not needed; app.js stays the single composition file). Verification here is static plus the offline pytest suite; no browser run, no real `/chat*` or `/upload` call in this task (earlier stub-fetch CDP runs are recorded in view-contract.md section 14).

## Workflow map
| UI action | Controller | Endpoint | View calls |
|---|---|---|---|
| Welcome card click / Enter / Space | `#starterPrompts` handler -> `_setMode`, `_fillComposer` | PATCH `/sessions/{id}` only if mode changed and a session exists. Never `createSession`/`chatStream` | `UI.setActiveMode`, focus `#chatInput` |
| Send / Enter | `_onSend` | POST `/sessions` (first send), POST `/chat/stream` `{session_id,message,user_id,mode,model,web_search,regenerate}` | `Chat.appendUserMessage`, `startAssistantMessage`, renderer.* |
| Stop | `_onStop` -> `AbortController.abort()` | stream closed; server persists partial | `markPartial({reason:"stopped"})`, toast |
| Regenerate / inline Retry | `_regenerateLast` / `_retryAfterError` -> `_onSend({regenerate:true})` | POST `/chat/stream` | old answer removed only in `onOpen` (after HTTP 2xx) and restored if stopped pre-token |
| Feedback | `_sendFeedback` | POST `/messages/{id}/feedback` | |
| Rename | `_renameSession` (rejects so the view reverts) | PATCH `/sessions/{id}` `{title}` (1-200) | `renderSessionList(..., {onRename,onRetry})` |
| Delete chat / document | `_deleteSession` / `_deleteDocument` after `UI.confirm` | DELETE `/sessions/{id}`, `/documents/{id}?user_id=` | toast |
| Select session | `_selectSession` | GET `/sessions/{id}` | `Chat.renderHistory`, mode/model restored |
| Models | `_loadModels` | GET `/models` | `renderModelSelect(models, id, {error,onRetry})`; fallback: stored -> default -> first `available` |
| Upload file/URL/drop | `_handleFileUpload`/`_onUrlSubmit` -> `_runUpload` (one at a time) | POST `/upload` (multipart / `?url=`), then GET `/documents` | `showUploadToast` (indeterminate), `showUploadError` with Retry; "indexed" toast only after POST resolves (`indexing: hybrid|lexical`) |
| Research / Code | `_setMode` | same `/chat/stream`, `mode` value; `citations` event passed separately to `finalize` | |

SSE: `meta, status, citations, model, token, error, done` all handled. Fallback badge from `model.fell_back_from`, cached from `done.cached`, thought timing from `status.elapsed_ms`.

## Preservation checklist
- window.CONFIG.BACKEND_URL, `user_id`, localStorage `mode/model/userId`, payload keys: verified (code read; payload-key guard in tests/test_frontend_guards.py passes).
- ALLOWED extensions: equal to backend `SUPPORTED_EXTENSIONS` (pdf docx pptx odt rtf csv xlsx xls txt md rst html htm json + images). The task text listed only 7; the backend parser now accepts more, so the controller list mirrors the real backend (guard test asserts equality with `parser.SUPPORTED_EXTENSIONS` and with `#fileInput[accept]`). Verified by pytest.
- No pin, no mock data, no hard-coded size limit (server 413 text shown): verified by grep/read.
- Welcome cards never send: verified statically (`test_welcome_card_handler_never_sends`) and by earlier stub-fetch CDP run (14.1). Not re-run in a browser here.
- Input preserved on failure, regenerate keeps old answer, duplicate-submit gate (`isLoading`), AbortController cleanup (`finally`): code read; stub CDP run earlier. Not verified against a live LLM.

## Backend gaps (honest UI instead)
- Pin: unsupported; no control rendered. Smallest fix: `pinned` column + field on `SessionUpdate`.
- Document size/status/server-side progress, upload limit and allowed types not exposed: only the server 413 text is shown; progress is indeterminate. Fix: `GET /config` with `max_upload_mb`, `extensions`.
- Model capability data absent. Rewrite-and-resend of a stored-but-unanswered user turn leaves a duplicate server turn (Minor-4 in view-contract 14.1a); needs a backend replace-last-turn option.

## Known issues
- Real-backend streaming, Stop mid-stream and upload paths not exercised in a browser in this task.
- Parallel view edits briefly made two guard tests fail mid-run (`test_feature_card_try_line_is_never_hidden`, `test_welcome_card_handler_never_sends`); both pass on rerun.

## Cross-boundary requests
None outstanding. (`.rst` accept, `showUploadError`, model retry, `UI.confirm`, rename were already landed view-side.)

## Commands run
- `find frontend/js -name '*.js' -exec node --check {} \;` -> exit 0
- `venv/bin/pytest -q` -> all pass, 1 skipped (xlwt not installed)
- `venv/bin/ruff check backend tests --select F,E9,B --ignore B008,B905,B011` -> All checks passed

## QA-04 m1 follow-up
- Welcome cards now use `UI.onWelcomeCard(({mode, prompt}) => ...)` registered in `init()` (switch mode, fill composer, never sends). The old `#starterPrompts` listener and its false comment are removed.
- `UI.setComposerBusy` and `Chat.holdLastAnswer` are intentionally NOT adopted: `_setLoading` also manages the `isLoading` gate and other state, and the `oldAnswer` restore logic is tied to the stream's stop/error paths. Neither is a clean drop-in, so the current logic is kept.

## QA-05 m3 follow-up
- Upload Files welcome card (`mode === "document"`): handler still switches mode and fills the composer, then calls `_openFilePicker()` (separate helper, clicks `#fileInput`, runs inside the card's click/keypress activation). Nothing is created or sent; selecting a file goes through the existing `change` -> `_handleFileUpload`. Other cards unchanged.
- `UI.onMyDocuments` registered: `_setMode("document")` (the view default clicked the Documents tab) plus `_loadDocuments()` (GET /documents).
- Errors previously only swallowed now call `UI.notify(msg, "error")`: `_loadDocuments` and `_newChat`.
- Guard: `test_welcome_card_handler_never_sends` also asserts `_openFilePicker` exists, uses `#fileInput`, and never references createSession/chatStream/_onSend/API.chat.
- Not exercised in a browser (native picker needs a real gesture).

## Reference-fidelity pass (AI Assistant Pro Responsive Dashboard.png)

Gaps closed against the supplied design: brand mark (`#i-brand`, `assets/logo.svg`), workspace pill tabs in the
top bar (`#topTabs`, mirrors `#modeTabs`; clicks are forwarded to the sidebar button so the controller still owns
the mode), glossy orb with gradient orbit rings (`ai-viz.js` / `ai-viz.css`), lavender sidebar wash, dark navy
icon rail on tablet in both themes (`--rail-*` tokens), compact composer (`clip | globe` + search-icon switch),
single-line capability strip, neutral list rows and a scrollable tab row on mobile.

Motion lives in `animations.css` ("Reference motion pack") plus the hero keyframes in `ai-viz.css`; only
`transform` / `opacity` animate, all of it is removed under `prefers-reduced-motion` and paused while the tab is
hidden. `ui.js` only measures (sliding highlights: `--thumb-x/-s/-y/-h`) and writes `--look-x/--look-y`
(the orb's eyes follow the pointer; skipped for touch and reduced motion).

Deliberate deviations from the picture (do not "fix"):
- Capability strip says "Safe Uploads / File type and size are checked", not "Secure & Private" (guarded by
  `test_no_unverifiable_privacy_claim_in_capability_strip`).
- The composer keeps the model chip on mobile (logo only) and the "Index web URL" action (globe button) because
  the design has no other way to change models or add a URL on a phone.
- User name / e-mail rows show the client-side user id, not an account.
- `--border-ui` stays >= 3:1 (lavender tint) on the composer, search field, Web Search pill and model chips.
