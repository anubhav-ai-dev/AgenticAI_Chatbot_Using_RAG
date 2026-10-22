# View / Controller contract

Scope: frontend-only redesign, backend contracts unchanged. `.rst` is a valid upload type (backend `parser.py:20`).
Sources: `audit.md`, `design-spec.md` (esp. 3, 8.3, 12.2, 12.4), and the code as of `main` @ `15e1d81` (`file:line` cited).
"(inferred)" marks anything not directly verified in code or stated in the spec. "(new)" marks an addition that does not exist today.

Rule of thumb: the **view layer** never calls `API.*`, never reads `State`, never writes `localStorage` keys `userId/mode/model`.
The **controller layer** never builds markup and (target state) never touches the DOM except through the ids in section 1.

---

## 0. Layers and ownership

| File | Layer | Owner |
|---|---|---|
| `frontend/index.html` | view | frontend-developer |
| `frontend/css/*` (incl. new `syntax.css`, optional `layout.css`) | view | frontend-developer |
| `frontend/assets/*` | view | frontend-developer |
| `frontend/js/ui.js`, `chat.js`, `utils.js` | view | frontend-developer |
| `frontend/js/icons.js`, `theme.js`, `components/**` (new) | view | frontend-developer |
| `docs/redesign/view-contract.md` | contract | frontend-developer (this file; changes announced here first) |
| `frontend/js/api.js`, `app.js`, `workflows/**` (new, optional) | controller | integration-developer |
| `docs/redesign/integration-notes.md` | controller notes | integration-developer |

Cross-layer facts today: `app.js` touches the DOM directly (`app.js:55-146,179,287,326,421-435`); `chat.js` calls `UI.showCitationModal`
(`chat.js:165,299`) and has inline `onclick` into `Chat`/`UI` (`chat.js:68,182`, `index.html:81`). Script order (classic scripts,
`index.html:273-277`): `utils, api, chat, ui, app`. Target order (spec 5): `icons, utils, api, chat, theme, components/menu,
components/dialog, components/model-select, components/ai-viz, ui, app`, plus any `workflows/*.js` before `app.js`.
**A `<script>` tag for `workflows/*` is a view-owned edit** (`index.html`): integration-developer requests it, frontend-developer adds it.
Do not rename `app.js` or `main.css` (`tests/test_api.py:55` requires both to be served).

---

## 1. DOM ids and selectors the controller depends on (must be preserved)

Every id below is read by `app.js` (all `getElementById` calls throw on `null` when followed by `.addEventListener`, so a missing id
aborts `_bindEvents` and breaks everything after it). Element type constraints matter: the controller assumes the listed event fires.

| id / selector | app.js usage (line) | Required element / behaviour |
|---|---|---|
| `#sendBtn` | click -> `_onSend` (55); `style.display` toggle (421-425) | `<button>`; hidden while loading |
| `#stopBtn` | click -> `_onStop` (56); `style.display` toggle | `<button>`; shown only while loading |
| `#chatInput` | `.value`, `keydown`, `input`, `.disabled`, `.focus()`, `.style.height`, `.scrollHeight` (59-66,142,159,423,432) | `<textarea>`. View must not rely on controller-set inline `height`; see 7 |
| `#modeTabs` | delegated click, `e.target.closest(".mode-tab")` (69-70) | container of tabs; may move to the sidebar |
| `.mode-tab[data-mode]` | `tab.dataset.mode` (72) | one of `general`, `document`, `research`, `code`. Tab children (icon svg, label span) are fine since `closest` is used. Must stay `class="mode-tab"` + `data-mode` |
| `#modelSelect` | `change` listener, `.value` read (80-81), `.value =` write (327) | **must remain a `<select>` (or a hidden native `<select>` kept in sync)**; a custom listbox has to set `.value` and dispatch a real `change` event |
| `#newChatBtn` | click (86) | `<button>` |
| `#themeToggle` | click -> `UI.toggleTheme` (87) | `<button>` that exists at load; spec 3.4 turns it into the menu trigger |
| `#sidebarToggle` | click -> `UI.toggleSidebar` (88) | `<button>`; exists in DOM even when hidden by CSS |
| `#attachBtn` | click (91) | `<button>`; triggers `#fileInput.click()` |
| `#fileInput` | `click()`, `change`, `e.target.files[0]`, then `value=""` (92-98) (inferred for the reset; verify at `app.js:94-99`) | `<input type=file>`; `accept` is view-owned and must include `.rst` (see 8) |
| `#addUrlBtn` | click -> `UI.showUrlModal` (101) | `<button>` |
| `#urlModal` | click-on-backdrop test `e.target === #urlModal` (108-109) | the **backdrop element itself**. If it becomes `<dialog>`, `e.target===dialog` still holds only if the dialog has no padding wrapper; coordinate with `UI.showUrlModal` (spec R2) |
| `#urlModalClose`, `#urlModalCancel`, `#urlModalSubmit` | click (102-104) | `<button>`s |
| `#urlInput` | `keydown` Enter, `.value` read in `_onUrlSubmit` (105,380) | `<input>` |
| `#searchToggle` | click; sets `style.color` (inline, to be removed) and `aria-pressed` (113-117) | `<button>`. After Stage 1 the view styles by `[aria-pressed]`; the controller must stop writing `style.color` (see 3, request C9) |
| `#citationModal`, `#citationModalClose` | click; backdrop test (122-124) | same backdrop rule as `#urlModal` |
| `#sessionSearch` | `input` event, `.value` (128-129) | `<input>` |
| `#starterPrompts` + `.welcome__starter[data-prompt]` | delegated click, `closest(".welcome__starter")`, `dataset.prompt` (137-139) | **`getElementById("starterPrompts").addEventListener` is not null-guarded (`app.js:137`): the element must exist until the controller changes.** Keep an empty (hidden) element if the starters are removed |
| `#chatArea` | `querySelectorAll("#chatArea .message__regen")` (179), `"#chatArea .message--assistant"` (287) | container of messages (`role=log`) |
| `.message__regen` | selector (179): removes regen buttons of earlier answers | class on the regenerate button created by `Chat` (`chat.js` `_renderActionButtons`). Keep the class name |
| `.message--assistant` | selector (287): last one is removed on regenerate | class on every assistant message element, **live and history** (`chat.js:46-60,271`). Keep |

Ids the controller does **not** use but the **view** layer does (so frontend-developer may change them, but must also update view JS): `sidebar`,
`sidebarBackdrop`, `userIdDisplay`, `sessionList`, `docList`, `docSection`, `welcomeScreen`, `uploadToast`, `uploadToastName`, `uploadProgress`,
`dropzoneOverlay`, `citationSource`, `citationSnippet`, `hljs-theme`, and the dynamic message ids (`thoughtSteps-*`, `streamContent-*`,
`citationCards-*`, `msgBadges-*`, `msgExtra-*`, `msgTime-*`). Spec 12.2 lists most of these as "must survive"; the stricter, controller-only list is the table above.
`#uploadToast*`/`#uploadProgress` should be treated as view-internal once request C6 lands (the controller only calls `UI.showUploadToast/hideUploadToast`).

The spec's 12.2 says `#starterPrompts` can be removed "only if app.js guards null". Verified: it does **not** (`app.js:137`). It is a hard dependency until request C5.

---

## 2. Functions the view exports and the controller calls

### 2.1 Existing, must keep signature (additions only)

| Function | Args | Returns | Side effects | Caller (app.js) |
|---|---|---|---|---|
| `UI.initTheme()` | none | void | reads `localStorage.theme`, sets `data-theme` on `<html>`, swaps `#hljs-theme` href (`ui.js:10-28`) | 27 |
| `UI.setTheme(t)` | `"light"\|"dark"` today; target adds `"system"` | void | persists `theme` | not called by app.js; wrapper retained |
| `UI.toggleTheme()` | none | void | today flips dark/light; target opens the ThemeSwitcher menu (spec 3.3) | 87 (passed as the click handler **unbound**: it must not rely on `this`) |
| `UI.openSidebar / closeSidebar / toggleSidebar()` | none | void | toggles `.mobile-open`, backdrop `display`, `aria-expanded` on `#sidebarToggle` (`ui.js:37-62`) | `toggleSidebar` 88 (unbound handler) |
| `UI.renderSessionList(sessions, activeId, onSelect, onDelete)` | `sessions`: array of `SessionSummary` (`id,title,model,mode,updated_at,...`); callbacks take `sessionId` | void | writes `#sessionList`; groups by `Utils.formatDateGroup`; on item click calls `onSelect(id)` and auto-closes drawer when `innerWidth<=768` | 131, 329, 353 (and one more in `_loadSessions`) |
| `UI.renderDocList(documents, onDelete)` | docs `{id,filename,file_type,chunk_count,indexed_at}`; `onDelete(id)` | void | writes `#docList` | 414 |
| `UI.renderModelSelect(models, selectedId)` | `models` from `GET /models`; `selectedId` string or `""` | void | rebuilds `#modelSelect` optgroups per provider; `option.disabled` when `!available` | 49 |
| `UI.showUploadToast(filename)` | string (or URL) | void | shows `#uploadToast` | 366, 387 |
| `UI.hideUploadToast()` | none | void | hides it | 369,374,390,394 |
| `UI.showToast(message, type)` | string; `type` in `info\|success\|error` (default `info`) | void | self-removing toast; message is HTML-escaped by the view | many |
| `UI.showCitationModal(citation)` | `{id,type,source,page,url,snippet}` | void | opens `#citationModal` | called by `chat.js` only |
| `UI.hideCitationModal()` | none | void | | 122 (unbound handler) |
| `UI.showUrlModal()` / `UI.hideUrlModal()` | none | void | modal open (also focuses `#urlInput`, inferred from `ui.js:262-274`) | 101-104,386 (unbound handlers) |
| `UI.initDropzone(onFileDrop)` | `onFileDrop(file: File)` | void | binds `dragenter/over/leave/drop` on `document`; today passes only `files[0]` (`ui.js:304`) | 134 |
| `UI.setActiveMode(mode)` | `general\|document\|research\|code` | void | marks `.mode-tab`; target also `aria-current="page"` | 28, 75, 312, 322 |
| `UI.setUserIdDisplay(userId)` | string | void | text in `#userIdDisplay` | 29 |
| `Chat.appendUserMessage(text, userId)` | string, string | void (inferred; `chat.js:20`) | appends user bubble, hides welcome | 177 |
| `Chat.startAssistantMessage()` | none | **renderer** (below) | appends assistant node, hides welcome | 180 |
| `Chat.renderHistory(messages, handlers)` | `Message[]`; `handlers = { onFeedback(messageId, value) }` (only `onFeedback` is used today; no `onRegenerate`, no copy) | void | resets area, renders; shows welcome when empty | 320 |
| `Chat.clearChat()` | none | void | resets `#chatArea`, shows welcome | 307, 341 |
| `Chat.copyMessage(msgId)` / `Chat.getCitation(msgId, id)` | | | used by inline onclick only (`chat.js:68,182`); **not called by app.js**. Keep as public until inline handlers are replaced (Stage 1), then they may become private | none |
| `Chat.showThinking / removeThinking` | | | dead (`chat.js:223-248`); the spec replaces them with a skeleton. No controller caller; may be removed (inferred: no external caller found by grep of `app.js`) | none |

Also called but outside UI/Chat: `Utils.*` is not called by `app.js` (only `chat.js`, `ui.js`). `Utils` public names persist (spec 12.2).

### 2.2 Renderer object returned by `Chat.startAssistantMessage()` (`chat.js:92-207`)

| Method | Args | Returns | Notes |
|---|---|---|---|
| `msgId` | | string uuid | property |
| `addThought(icon, label, status="pending")` | `icon`: **emoji string today**; target: icon **name** (`pencil, file-search, globe, sparkles, circle-check, circle-alert`) **or legacy text** | step element or `null` | `app.js:191` passes `STAGE_LABELS[stage].icon` |
| `resolveThought(stepEl, {icon, label, elapsed, status="done"})` | `elapsed` is a string like `"1.2s"` | void | `app.js:197-203`; `icon` currently `"⚠️"`/`"✅"`; target `circle-alert`/`circle-check` |
| `appendToken(token)` | string | void | re-renders Markdown throttled |
| `finalize(citations=[])` | `items[]` | void | highlight + citation badges + cards |
| `setBadges({model, cached, fellBackFrom})` | | void | |
| `enableActions({messageId, feedback, onRegenerate, onFeedback})` | `onRegenerate()` ; `onFeedback(messageId, value)` returning a Promise | void | `app.js:253-257` |
| `showError(msg)` | string | void | today inline `style` + emoji; target `circle-alert` glyph + class (view-only change) |
| `getRawText()` | | string | used to decide error-vs-partial (`app.js:242`) |
| **`setVizState(state)` (new)** | `"idle"\|"processing"\|"responding"\|"error"` | void | spec 8.3; no-op if the viz is unavailable. Controller mapping in section 4 |

Compatibility rule for renamed arguments: every renderer method keeps accepting today's argument shapes. Emoji in `icon` must not throw
(spec 9.1: `Icons.html` returns unknown names escaped).

---

## 3. Additions the spec requires (new API; view implements, controller adopts)

All additions are **additive**; the old calls keep working until the controller migrates (request ids refer to spec 12.4).

| # | New API | Signature | Behaviour / return | Controller change |
|---|---|---|---|---|
| C1 | `UI.confirm(opts)` | `{ title, message, confirmLabel, cancelLabel?, destructive?: bool } -> Promise<boolean>` | Native `<dialog>` or equivalent. Resolves `true` on confirm; `false` on cancel, Esc, backdrop. Focus restored to the invoker. Only one open at a time (a second call resolves the first with `false`, inferred) | Replace `confirm()` at `app.js:336` and `:400` with `await UI.confirm({...})` |
| C2 | Rename | `UI.renderSessionList(sessions, activeId, onSelect, onDelete, { onRename, onRetry }?)` (spec 12.4.2 puts `onRename`, `onRetry` on `renderSessionList`; option-bag form is inferred to keep positional compatibility) | `onRename(sessionId, newTitle: string)`. View trims, rejects empty, enforces a max length it chooses (server limit: verify `SessionUpdate.title` in `models/api.py:22-25`; not re-verified here), returns focus. View awaits nothing: if `onRename` returns a Promise, a rejection should revert the inline field (inferred) | Implement `onRename -> API.updateSession(id, {title})`, then update `State.sessions` and re-render. `onRetry()` = reload the sessions list after a failed `GET /sessions` (the view needs a loading and an error state; see C7) |
| C3 | Visualization state | `renderer.setVizState(state)` (above) and `UI.setVizState(state)` for the welcome-screen instance (inferred name; spec 8.3 writes `AIViz.setState("...", {target})` internally and says the controller must go through `Chat`/`UI`) | Idempotent, safe before DOM exists | Call per mapping in section 4 |
| C4 | `#liveStatus` | `UI.setLiveStatus(text: string)` | Writes to a visually hidden `<div id="liveStatus" role="status" aria-live="polite">` (id and role view-owned; it is **not** a controller-referenced id). Should be coalesced by the view. Empty string clears | Call at stage changes, completion ("Response complete"), error, stop |
| C5 | Feature cards | no new function. Cards are `[data-mode]` elements in `#welcomeScreen` (class name view-owned; suggested `.feature-card`; spec 6.7) | Click must reach the controller. Today only `#starterPrompts` is bound. **Proposed contract:** the controller adds one delegated listener on `#welcomeScreen` (or a stable wrapper id the view guarantees) for `closest("[data-mode]")` -> `State.mode = mode; localStorage.mode; UI.setActiveMode(mode); API.updateSession(...)` and focuses `#chatInput`. Cards **do not auto-send**. Suggestion chips (text-only) keep `.welcome__starter[data-prompt]` and `#starterPrompts` (no change in controller) | Add the delegated handler; then `#starterPrompts` may be dropped |
| C6 | Upload retry + progress | `UI.showUploadToast(filename, { size?, onRetry?, onCancel? }?)` and `UI.showUploadError(filename, { message, onRetry })` (inferred names; the spec only names `onUploadRetry`) | View shows indeterminate progress (no percentage, remove the fake 2.5s animation: audit D6). Retry button calls `onRetry()` with no args | Keep the `File` object in a closure/State; implement `onUploadRetry`; one drop with >1 file -> `UI.showToast("Only the first file was added.", "info")` (spec 12.4.6). The view's `initDropzone` still passes a single `File` to `onFileDrop` and may pass a second argument `{ extraCount }` (inferred) |
| C7 | Models retry | `UI.renderModelSelect(models, selectedId, { error?: string, onRetry?: fn })` | View renders a real error state and a retry control calling `onRetry()` (no args). `selectedId` empty = nothing selected | `_loadModels` calls the hook on failure instead of rendering an empty select (audit D12). Keep reading `#modelSelect.value` and the `change` event |
| C8 | Icon names instead of emoji | `STAGE_LABELS[*].icon`, `finishStep` icons, `renderer.addThought/resolveThought` | Names: `pencil`, `file-search`, `globe`, `sparkles`, `circle-check`, `circle-alert` | Replace the emoji literals in `app.js:151-154,198`; strip emoji from toast strings (`app.js:371,391`). `UI.showToast` selects its own icon from `type` |
| C9 | Search toggle styling | none | View styles `#searchToggle[aria-pressed="true"]`. Controller keeps writing `aria-pressed` only | Remove `searchToggle.style.color = ...` (`app.js:115`); inline style would override the new CSS |
| C10 | `UI.setActiveMode(mode)` | unchanged | Sets `aria-current="page"`; may also set a document count | none |
| C11 | `UI.renderDocList(documents, onDelete)` | unchanged | Also feeds the Documents count badge (`documents.length`). Delete uses `UI.confirm` **inside the controller's `onDelete`** (C1), not inside the view | controller wraps |
| C12 | Smart-scroll hook | none | The view owns scroll; the controller does not scroll | none |
| C13 | Theme API | `Theme.get() / set(pref) / resolved()` (new `theme.js`) and `UI.setTheme("light"\|"dark"\|"system")` | See section 6 | none; `app.js:27` `UI.initTheme()` stays |

Not offered by the view (no backend support): `onPin`, document size/status props, model capability fields. Do not add parameters for them.

Undecided: whether the `UI.confirm`/rename/upload-retry names above are final. The spec fixes `UI.confirm`, `onRename`, `onRetry`, `onUploadRetry`,
`onRetryModels` (spec 12.4.2,6,7), `UI.setLiveStatus`, `setVizState`. The remaining signatures in this table are **inferred proposals** for
frontend-developer to confirm by editing this file before coding.

---

## 4. Events and callbacks flowing view -> controller

| Source (view) | Mechanism | Controller handler | Payload |
|---|---|---|---|
| Send / Stop buttons | DOM `click` on `#sendBtn`/`#stopBtn` | `_onSend`, `_onStop` | none |
| Enter in `#chatInput` | `keydown`; controller owns the check (`app.js:60-66`). **Controller must add `!e.isComposing`** (audit D9; not a view task) | `_onSend` | |
| Mode tab click | DOM click, delegated on `#modeTabs` | sets `State.mode`, persists `localStorage.mode`, `UI.setActiveMode`, `PATCH` | `tab.dataset.mode` |
| Feature card click (new, C5) | delegated click on `#welcomeScreen` | same as mode tab, then focus | `data-mode` |
| Model change | native `change` on `#modelSelect` | `State.model`, `localStorage.model`, `PATCH` | `e.target.value` |
| New chat | click `#newChatBtn` | `_newChat` | |
| Theme / sidebar toggle | click `#themeToggle`/`#sidebarToggle` | `UI.toggleTheme`, `UI.toggleSidebar` (view functions invoked directly) | |
| Attach / file chosen | click `#attachBtn`; `change` on `#fileInput` | `_handleFileUpload(file)` | `File` |
| Dropzone | `UI.initDropzone(cb)` | `cb(file)` | one `File` |
| URL modal | clicks on `#urlModalSubmit/#urlModalCancel/#urlModalClose`; Enter in `#urlInput`; backdrop | `_onUrlSubmit`, `UI.hideUrlModal` | `#urlInput.value` |
| Search toggle | click `#searchToggle` | flips `State.webSearch` | |
| Session search | `input` on `#sessionSearch` | filters `State.sessions` by `title` and re-renders (title-only; keep, spec 10) | |
| Session click / delete | callbacks of `renderSessionList` | `_selectSession(id)`, `_deleteSession(id)` | session id |
| Session rename (new) | `onRename` callback | `API.updateSession(id, {title})` | id, trimmed title |
| Doc delete | `onDelete` callback of `renderDocList` | `_deleteDocument(id)` | document id |
| Feedback / regenerate | `onFeedback(messageId, value)` / `onRegenerate()` from `enableActions`/`renderHistory` | `_sendFeedback`, `_regenerateLast` | `value` in `{-1,0,1}`; returns a Promise (view should show failure, inferred) |
| Model retry / sessions retry / upload retry (new) | `onRetryModels`, `onRetry`, `onUploadRetry` | reload | none |
| Confirm dialog | `UI.confirm()` Promise | awaited | boolean |

Controller -> view state mapping (spec 8.3, to be implemented by integration-developer using `renderer.setVizState` and `UI.setVizState`):
`request submitted / meta / status.state=="start" -> processing`; first `token` or `status` `first_token -> responding`;
`done -> idle`; `error` event or fetch failure -> `error`; abort (Stop) -> `idle`; new chat / mode switch -> `idle`.
Handlers today (`app.js:217-238`): `status`, `citations`, `model`, `token`, `error`, `done`. **No `meta` handler exists**; add one if the spec's
`meta -> processing` is wanted, or set `processing` immediately after `Chat.startAssistantMessage()` (the effect is the same).

Input restoration (audit D4) and regenerate ordering (D5) are controller-only fixes; the view only needs `#chatInput` to accept `.value =`
and a call to its auto-resize (still `_autoResize` in `app.js:430-435`, which writes `style.height`; view CSS must allow a `min-height`/`max-height`
and not set `!important` on `height`).

---

## 5. Controller-supplied data shapes the view relies on (verified)

- Session: `id, title, model, mode, updated_at, created_at` (`ui.js:82,93-99`). `s.model` is dereferenced with `.split("/")` (`ui.js:99`): the view must tolerate null/undefined (audit D17).
- Timestamps: naive UTC strings without `Z` (audit D1/B9). **The fix is view-side:** `Utils.formatTime/formatDateGroup` (frontend-developer) append `Z` when no zone is present. The controller must pass server strings through unmodified.
- Document: `id, filename, file_type, chunk_count, indexed_at` only. No size, no status (audit B3).
- Models: `{id,label,provider,description,available,reason,default}` (audit section 6).
- Citation: `{id,type,source,page,url,snippet}`; `page` may be int or string (escape it; audit section 9); `url` untrusted, render only if `^https?://`, with `target="_blank" rel="noopener noreferrer"` and no inline handler.
- Messages (history): `{id, role, content, citations, feedback, meta, created_at}` (`chat.js:251-320`).

---

## 6. Theme storage contract (view-owned; the controller does not read or write it)

- Key `localStorage["theme"]`, values `"light" | "dark" | "system"`. New visitors get `"system"` (spec 3.1).
- Key `localStorage["themeVersion"]`, value `"2"` once migrated. Migration (runs once, in the no-flash `<head>` snippet and again inside `theme.js` for safety, spec 3.2-3.3):
  `themeVersion !== "2"` -> `theme = (stored === "light") ? "light" : "system"`; set `themeVersion = "2"`.
  Reason: the old code persisted `dark` on every load (`ui.js:10-17`), so stored `dark` cannot be told apart from a default.
- DOM outputs: `<html data-theme="light|dark">` (resolved), `<html data-theme-pref="light|dark|system">`, `style.colorScheme`, `<meta name="theme-color">`. `index.html:2` no longer hardcodes `data-theme="dark"`.
- `Theme.get() -> pref`, `Theme.set(pref)`, `Theme.resolved() -> "light"|"dark"`; listens to `prefers-color-scheme` change (only when pref is `system`) and the `storage` event.
- Legacy API retained: `UI.initTheme()`, `UI.setTheme(x)`, `UI.toggleTheme()` as thin wrappers (spec 3.3). `UI.initTheme()` becomes idempotent and must not write `dark` unconditionally as it does today.
- `#themeToggle` stays a `<button>` present at DOM load; `app.js:87` binds `click -> UI.toggleTheme` (an unbound reference). The button becomes a menu trigger (`aria-haspopup="menu"`, `aria-expanded`). **Click handling conflict (inferred risk):** if `theme.js` also binds its own click on `#themeToggle`, the menu opens twice. Rule: one owner. Recommendation: `theme.js`/`UI.toggleTheme` own the toggle; `app.js` keeps its listener unchanged (it only calls `UI.toggleTheme`), and `UI.toggleTheme` is the **only** opener.
- `#hljs-theme` link and its `href` swapping are removed (spec 3.6); `css/syntax.css` handles both themes. The controller does not reference `#hljs-theme`.
- Other localStorage keys, controller-owned and untouched by the view: `userId`, `mode`, `model`.

---

## 7. Stage 1 changes: view-only, must not break `app.js`

Safe if the rules in the right column hold.

| Change | View-only? | What must not break |
|---|---|---|
| Tokens (replace the 54 hex literals, legacy variable aliases kept for one commit) | yes | no id or class renames; `.mode-tab`, `.message__regen`, `.message--assistant` stay |
| Theme: no-flash snippet, `theme.js`, `data-theme-pref`, System | yes | `#themeToggle` exists as `<button>`; `UI.initTheme/setTheme/toggleTheme` keep working; `localStorage.theme` key unchanged (values extended); see conflict note in 6 |
| Remove `#hljs-theme` swap, add `syntax.css` | yes | `Utils.highlightCode` stays safe if hljs missing (`utils.js:65`) |
| Icons: `icons.js`, sprite, replacing emoji in markup and view templates | yes | renderer/`UI.showToast` accept both icon names and legacy emoji (controller still sends emoji until C8); `UI.showToast(msg,type)` must keep stripping nothing: controller strings with a leading "check" emoji (`app.js:371,391`) render as plain text until C8 |
| `UI.confirm` / `components/dialog.js` | yes to add | `confirm()` stays in `app.js` until C1; nothing removed |
| Dialog markup (`<dialog>`, `role=dialog`, `aria-modal`, Esc, focus trap) for `#urlModal` and `#citationModal` | **only if** backdrop-click semantics survive | `app.js:108-109,123-124` test `e.target === #urlModal / #citationModal`. If these ids move to a `<dialog>`, the dialog element must receive the backdrop click, and `#urlModalClose/Cancel/Submit`, `#urlInput`, `#citationModalClose` must keep their ids. `UI.showUrlModal/hideUrlModal/showCitationModal/hideCitationModal` remain the only open/close paths (`display:flex` toggling replaced inside them) |
| Remove inline `style=` (12 in HTML, ~20 JS `.style.` writes) | mostly | **Cannot remove** `app.js` writes: `#sendBtn/#stopBtn` `style.display` (421-425), `#searchToggle` `style.color` (115), `#chatInput` `style.height` (432-434). Until C9 and a decision on `_setLoading`, the view must tolerate them: style `#searchToggle` so inline `color` does not make it unreadable in either theme; do not give `#sendBtn/#stopBtn` a `display` rule with higher specificity than inline (inline wins anyway); keep `#chatInput` height unconstrained except `max-height`/`min-height` |
| Remove inline `onclick` (`index.html:81`, `chat.js:68,182`) | yes | replace with `addEventListener` inside `chat.js`/`ui.js`; `Chat.copyMessage` and `Chat.getCitation` keep existing as public functions (harmless) |
| Move `#modeTabs` into the sidebar, new shell | yes if ids/classes kept | `#modeTabs` must be present at `DOMContentLoaded`, before `app.js` init (`app.js:69`); `#sidebarToggle`, `#sidebarBackdrop` keep ids; the sidebar visibility logic stays in `UI.*` |
| Skip link, landmarks, `aria-current`, `aria-pressed` styling, focus ring, tooltip on focus | yes | none |
| Welcome: replace six starter pills with feature cards | **not safe alone** | `#starterPrompts` must exist (`app.js:137` unguarded). Either keep `#starterPrompts` with text-only `.welcome__starter[data-prompt]` chips, or keep an empty hidden `#starterPrompts` until C5 |
| Custom model listbox | **not view-only** | must keep a real `<select id="modelSelect">`, synced; `app.js:327` sets `.value`; `change` must fire |
| `.rst` in `accept` and drop text (`index.html:30,204`) | yes (view) | the controller's `ALLOWED` array (`app.js:359`) also needs `rst` (a controller change). Until it lands, the file picker would allow `.rst` and the controller would reject it with "Unsupported file type". Therefore the view and controller change must ship together (C14 below) |
| Unify `renderHistory` with live messages (copy, fallback badge) | yes | still returns void; still honours `handlers.onFeedback`; add optional `handlers.onRegenerate`; `.message--assistant` class on history assistants |
| `UI.showUploadToast` fake progress removal | yes | signature `showUploadToast(filename)` compatible (new args optional) |
| Timestamp fix in `Utils` | yes | `formatDateGroup` still returns one of `Today, Yesterday, Previous 7 Days, Previous 30 Days, Earlier` (the group names are matched literally by `renderSessionList`'s order list, `ui.js:78`) |

Extra controller change listed here because it blocks a view change: **C14** - add `"rst"` to `ALLOWED` in `app.js:359` (spec 12.6 q.4, user-approved by the task constraints).

---

## 8. Upload type contract

- Backend accepts `pdf docx csv xlsx txt md rst` (`backend/services/parser.py:20`; size limit `MAX_UPLOAD_MB`, default 20, enforced server-side with 413: `config.py:185`, `routes.py:256-260`).
- Today's frontend whitelist is `pdf, docx, csv, xlsx, txt, md` in `app.js:359` and `accept=".pdf,.docx,.csv,.xlsx,.txt,.md"` in `index.html:204` (drift, audit D2).
- Target (frontend-only): `accept` = `.pdf,.docx,.csv,.xlsx,.txt,.md,.rst` (view), `ALLOWED` includes `rst` (controller, C14). Dropzone helper text and URL support wording are view copy. No frontend-side size check is required; show the server's 413 message verbatim (spec 10). A client-side 20 MB pre-check would hard-code a server default and is not recommended (inferred).

---

## 9. Compatibility checklist before the controller migrates

1. `node --check` every file in `frontend/js/**`.
2. `venv/bin/pytest -q tests/test_api.py` still serves `/js/app.js` and `/css/main.css`.
3. A scripted check (qa-reviewer, new test file only): every `getElementById("...")` string in `app.js` exists in `index.html` (table in section 1).
4. `grep` that `app.js` contains no `confirm(` after C1, and no emoji after C8.

---

## 10. Summary of ambiguities

1. **Signatures marked inferred** (C2 option bag, C6 `showUploadError`/second `showUploadToast` argument, C3 `UI.setVizState`) are proposals; the spec fixes names only. frontend-developer should confirm or edit them here before coding; integration-developer should not code against them until then.
2. **`#starterPrompts`**: the spec says it can go if `app.js` guards null; it does not (`app.js:137`). Either keep the element or land C5 first.
3. **`#themeToggle` ownership**: `app.js:87` already binds click to `UI.toggleTheme`; `theme.js` must not add a second opener (section 6).
4. **`.rst`** needs both a view change (`accept`) and a controller change (`ALLOWED`, C14); shipping only one gives a visible inconsistency.
5. **Inline style writers in `app.js`** (`#sendBtn/#stopBtn` display, `#searchToggle` color, `#chatInput` height) are controller code; the view can only tolerate them. Decide whether the controller switches to class/attribute toggles (cleaner) or the view keeps them working.
6. **Dialog backdrop semantics**: `app.js:108,123` compare `e.target` to the modal backdrop id; native `<dialog>` markup must preserve that or integration-developer must change those two listeners.
7. **`meta` event** has no handler today; the spec's `meta -> processing` mapping needs either a new handler or an immediate `processing` after `startAssistantMessage()`.
8. **Session title max length / validation** for rename was not re-verified against `models/api.py` here; check before finalizing the inline field.
9. **`renderHistory` regenerate/copy parity** (audit D15): whether `onRegenerate` is offered for restored sessions is a product decision; the contract only reserves the optional handler.
10. **Model fetch failure mode**: the controller currently swallows errors and sends `model: null` (audit D12); with C7 it also needs to decide whether sending is allowed while the list failed (server default applies today).

---

## 11. Stage 1 as implemented (frontend-developer, view-only; `app.js` / `api.js` untouched)

Verified in headless Chrome through scripted CDP calls (no `/chat*`, `/upload`, `/generate`, `/search` request was made). Items marked
"not exercised" were not run.

### 11.1 Script load order (current `index.html`)
`icons.js, utils.js, api.js, chat.js, theme.js, components/menu.js, components/dialog.js, ui.js, app.js` (marked, DOMPurify, highlight.js stay
CDN `<script>`s before them). Stylesheets: `main.css` (tokens + legacy aliases), `animations.css`, `components.css`, `syntax.css`.
Fonts are `<link>` tags in `<head>` (the CSS `@import` is gone). The no-flash theme snippet is the first script in `<head>`.

### 11.2 Ids (unchanged unless listed)
Every id in section 1 exists (checked by script against `getElementById(...)` in `app.js`). `#urlModal` and `#citationModal` are now
`<dialog class="dialog">` elements; the dialog is a transparent full-viewport layer, so `e.target === #urlModal|#citationModal` is still true for
a click outside `.modal` (verified: backdrop click closes, a drag that starts inside `.modal` and ends outside does NOT, Esc closes natively,
focus returns to the invoker). `#stopBtn` keeps an inline `style="display:none"` on purpose (app.js `_setLoading` writes `style.display = ""`
to show it). New ids: `#liveStatus` (view-owned), `.skip-link`, `.toast-region` (created lazily).
Retired: `#hljs-theme`. Changed element type only: `#uploadToast` now uses the `hidden` attribute instead of inline display.

### 11.3 New / changed view API
| API | Notes |
|---|---|
| `Theme.get() / set(pref) / resolved() / onChange(fn) / init()` (`theme.js`) | prefs `light|dark|system`; storage per section 6; `init()` idempotent |
| `UI.initTheme()` | idempotent; no longer writes `dark` |
| `UI.setTheme("light"|"dark"|"system")` | wrapper over `Theme.set` |
| `UI.toggleTheme()` | opens/closes the theme menu on `#themeToggle` (unbound-safe; `app.js:87` unchanged) |
| `UI.confirm({title, message?, confirmLabel?, cancelLabel?, destructive?}) -> Promise<boolean>` | C1; native `<dialog>`; focus starts on Cancel; Esc/backdrop/Cancel -> `false`; a second call resolves the first with `false`; falls back to `window.confirm` if `<dialog>` is unsupported. `app.js` still uses `confirm()` (C1 not applied) |
| `UI.setLiveStatus(text)` | C4; coalesced (150 ms); `""` clears. `Chat` already calls it on thought steps, completion, error, copy, so the controller need not |
| `UI.showToast(msg, type)` | region, max 3, 4 s / 8 s (error stays while hovered/focused, has Dismiss); legacy emoji in `msg` is shown as plain text; `role="alert"` for errors |
| `UI.showUploadToast(name)` / `hideUploadToast()` | same signature; indeterminate CSS bar, no fake percentage |
| `UI.setActiveMode(mode)` | sets `.active` and `aria-current="page"` (no more `aria-selected`) |
| `Icons.html(name,{size,cls}) / Icons.el(name,{size,cls}) / Icons.has(name)` | unknown names return escaped text; known legacy emoji (`✏️ 🔍 🌐 🤖 ✅ ⚠️ ❌ ℹ️`) map to icons, so `addThought/resolveThought` accept both names and emoji (C8 can land any time) |
| `Utils.safeUrl(url)`, `Utils.parseDate(v)`, `Utils.copyText(text)` | new; `copyText` now returns `Promise<boolean>`; `formatTime/formatDateGroup` treat naive timestamps as UTC (return values unchanged: Today, Yesterday, Previous 7 Days, Previous 30 Days, Earlier; invalid/missing -> "Earlier" / "") |
| `Menu.open/close/isOpen`, `Dialog.open/close/confirm` | internal components, available globally |
| `Chat.copyMessage(msgId)` | now returns `Promise<boolean>` and flashes "Copied" on the message's button; `Chat.getCitation` unchanged |

Renderer methods (`addThought/resolveThought/appendToken/setBadges/enableActions/finalize/showError/getRawText`) keep their signatures.
`resolveThought` without `icon` now picks `circle-check` / `circle-alert` from `status` instead of leaving the spinner. Not added yet: `setVizState`, `renderSessionList` options bag, `renderModelSelect` error state, `showUploadError`.

### 11.4 Markup / behaviour changes the controller may notice (no id or selector in section 1 changed)
- Session row: `div.session-item[data-session-id]` now contains two sibling buttons, `button.session-item__main` (select) and `button.delete-session-btn` (delete, shown on hover and on keyboard focus). Callbacks and their arguments are unchanged. `s.model` may be null.
- `#modeTabs` is a plain `<nav aria-label="Workspaces">`; `role=tablist/tab` and `aria-selected` were removed; the active button has `aria-current="page"`. Children are `svg` + `span.mode-tab__label`.
- `<footer role=contentinfo>` became `div.input-dock[role=group][aria-label="Message composer"]`; `<main>` lost its redundant role.
- `#chatArea` has `aria-live="off"`; announcements come from `#liveStatus`. Messages are `<article aria-label="You said|Assistant said">` (`.message`, `.message--assistant`, `.message__regen` unchanged); streaming sets `aria-busy`.
- `#searchToggle` is styled by `[aria-pressed]` (pressed shows fill, border and a check). Inline `style.color` writes by `app.js` still work (aliases `--accent-green` / `--text-muted` resolve to readable tokens in both themes) but should be removed (C9).
- `.stagger`, `.animate-float`, `.animate-bounce-in`, `.btn--green`, `.glass-card`, gradient/blur tokens: removed or neutralised. Legacy variables remain as aliases in `main.css` (see the comment there) until `app.js` stops using them.
- Markdown policy (`Utils.renderMarkdown`): DOMPurify with an http(s)-only URI regexp; forms, inputs, buttons, styles, iframes, embeds, svg, math, `style`/`id`/`name`/`class`/`role`/`download`/`ping`/`data-*` attributes are dropped (only `code.language-xxx` is kept for highlight.js; all our classes/roles, e.g. citation badges, are applied after sanitising); all links get `target="_blank" rel="noopener noreferrer"` and lose `href` unless http(s); every `<img>` is replaced by a text placeholder ("Image blocked: alt"), with an "(open)" link when the source is http(s); citation markers `[n]` are `sup.citation[role=button][tabindex=0]`. Source cards are `button.citation-card`. Citation `url` is rendered only if `http(s)`.
- Fake upload progress animation and its JS were removed.

### 11.5 Controls intentionally not rendered (no backing action)
Pin, rename (until `onRename` exists), "Retry" on errors, model capability/price chips, document size/status, upload percentage, Settings.

### 11.6 Still open for later stages
`#modeTabs` move into the sidebar, drawer focus trap / `inert`, model listbox, AI visualization, feature cards, `.rst` in `accept` (needs C14), smart scroll. Controller requests unchanged: C1 (use `UI.confirm` at `app.js:336,400`), C8 (icon names), C9 (drop `searchToggle.style.color`), C14.

---

## 12. Stage 2 as implemented (shell, welcome, composer, model selector; frontend-developer, view-only)

`app.js`, `api.js` and `backend/**` are unchanged. Verified in headless Chrome via scripted CDP (50 scripted checks, all passing; all non-GET
requests to `localhost:8000` were blocked by the harness; `API.createSession/chatStream` were stubbed for the card test). Not exercised: real
screen readers, real touch gestures, Safari/Firefox, hardware-keyboard `(pointer:fine)` (headless reports `pointer:none`, so the keyboard hint
was checked with a forced `display:block`).

### 12.1 Script load order (changed)
`icons, utils, api, chat, theme, components/menu, components/dialog, components/model-select (new), ui, app`. CSS files unchanged
(`main.css`, `animations.css`, `components.css`, `syntax.css`). The head snippet now also sets `<html data-shell>` (see 12.2).

### 12.2 Shell
`<html data-shell="drawer|rail|expanded">`: drawer < 768; rail 768-1023 (always) and >= 1024 when the user collapsed it; expanded >= 1024.
The head snippet sets it before paint; `ui.js` keeps it in sync (matchMedia). `localStorage.sidebarState = "expanded"|"rail"` is a view preference.
`.sidebar` is `position:fixed` in every shell; `.main` has `margin-left: var(--shell-offset)` (272 / 72 / 0). Overlay (drawer, or the rail
expanding on tablet) never pushes the content.

| API / element | Behaviour |
|---|---|
| `UI.openSidebar()` / `UI.closeSidebar(opts?)` / `UI.toggleSidebar()` | Same names, still safe as unbound handlers (`app.js:88` binds `UI.toggleSidebar` to `#sidebarToggle`). `< 1024`: toggles the overlay (`.mobile-open`; `role=dialog aria-modal=true aria-label="Navigation"` on `#sidebar`; `#main` inert, `aria-hidden` fallback; scrim `#sidebarBackdrop.is-open`; focus moves to the first `.mode-tab`; Tab/Shift+Tab trapped; Esc and scrim close; focus returns to `#sidebarToggle`). `>= 1024`: `toggleSidebar` collapses/expands the rail. `closeSidebar` is a no-op when closed; `{restoreFocus:false}` skips the focus return. |
| `UI.setSidebarCollapsed(bool, {focus?})`, `UI.isSidebarCollapsed()`, `UI.isSidebarOpen()` (new) | Persisted collapse (>= 1024). |
| `#sidebarToggle` | Hidden when expanded. Label "Open navigation" (< 1024) or "Expand sidebar"; `aria-expanded` mirrors overlay state; icon is `menu` in drawer, `panel-left-open` in rail. |
| `#sidebarCollapse` (new) | In-sidebar button: "Collapse sidebar" on the expanded desktop sidebar, "Close navigation" (x icon) while the overlay is open; hidden in the rail. |
| `#contextTitle` (new, view-owned) | Top-bar `<p>` with the workspace name, updated by `UI.setActiveMode`. |
| `#themeToggle` | Moved from the sidebar footer to the top bar (id, button type and `UI.toggleTheme` behaviour unchanged). |
| `#modeTabs` / `.mode-tab[data-mode]` | Now inside the sidebar (present at parse time). Labels: General, **Documents** (was "Document"; `data-mode` stays `document`), Research, Code. Active item = `.active` + `aria-current="page"`. Clicking a tab (any shell) closes the overlay; `app.js` still owns mode state. In the rail the label is visually hidden (still the accessible name) and a CSS tooltip (`data-rail-tip`) appears on hover/focus. |
| `#docCount` (new) | Count pill inside the Documents tab, fed by `UI.renderDocList(documents, ...)` (`documents.length`); hidden at 0; contains sr-only " documents". |
| `#newChatBtn` | Same id/click. Full-width "New chat" button; icon-only (label sr-only) in the rail; closes the overlay when clicked. |
| `#sessionSearch` | Same input, event and value (title filter stays in the controller). |

### 12.3 Session list
`UI.renderSessionList(sessions, activeId, onSelect, onDelete, handlers?)` (new optional 5th parameter, `handlers = { onRename? }`).
- Markup: `section.session-group[aria-labelledby]` (heading = group name from `Utils.formatDateGroup`) > `ul` > `li.session-item[data-session-id]` containing `button.session-item__main` (select) and `.session-item__actions` with `button.delete-session-btn` (and `button.rename-session-btn` only when `handlers.onRename` is a function). No nested interactive elements. Row = mode icon, title, and a time (today) or short date (older). The model name only appears in the row's `title` tooltip; `s.model` may be null.
- `onRename(sessionId, newTitle)`: not rendered unless supplied (`app.js` does not pass it, so it is absent today). Inline input, `maxlength=200` (`SessionUpdate.title` limit), Enter/blur commit when non-empty and changed, Esc cancels, a rejected promise reverts. Tested with a harness callback only.
- Empty list: "No chats yet", or `No chats match "q"` when `#sessionSearch` has text.
- Keyboard focus survives the controller's full re-render (same session id + same control is refocused).
- The row click calls `onSelect(id)` and closes the overlay if it is open.

### 12.4 Welcome and example prompts
- `#starterPrompts` (kept) now contains four feature cards: `button.welcome__starter.feature-card[data-mode][data-prompt]` (General chat, Analyze documents, Web research, Write and debug code). `app.js`'s existing handler fills and **sends** `data-prompt` (unchanged). `ui.js` registers an earlier listener on the same element that, for a card whose `data-mode` differs from the active workspace, calls `.click()` on the matching `.mode-tab`, so `app.js`'s own tab handler updates `State.mode` / `localStorage.mode` before the prompt is sent. Each card shows the exact prompt it will send ("Asks: ...") so the action is not hidden. Controller C5 is therefore **not** needed; if the controller later wants cards to switch mode without sending, it must drop the `.welcome__starter` class from the cards or change its handler.
- `#modePrompts` (new, view-owned): up to three static example prompts per workspace (`MODE_PROMPTS` in `ui.js`), rendered as `button.prompt-chip` by `UI.setActiveMode`. A click fills `#chatInput`, dispatches `input` (so `app.js` auto-resizes), focuses the composer, and never sends.
- Removed copy: the "free models" claim in the welcome text and in `<meta name="description">`. No privacy/security/speed claims.

### 12.5 Composer
Markup: `.input-dock[role=group]` > `.dock-inner` > `.composer` (box with border and `:focus-within` ring) > `textarea#chatInput` + `.composer__bar` (`.dock-tools`: `#attachBtn`, `#fileInput`, `#addUrlBtn`, `#searchToggle`; `.composer__send`: `#sendBtn`, `#stopBtn`) and `p.dock-hint`.
- All ids unchanged. `#chatInput` min-height 44px, `max-height: min(200px, 30dvh)`; `app.js` still writes `style.height` (<= 180px), verified to grow to 136px for 5 lines and reset to 44px.
- `#stopBtn` keeps its inline `display:none`; `app.js` writes `style.display` on both buttons (verified). `#searchToggle` is a pill: globe + "Web" (label hidden < 480px) + check when `aria-pressed="true"`; the controller's inline `style.color` writes still resolve to readable colours.
- Hint (Enter / Shift+Enter) only at `(min-width:1024px) and (min-height:720px) and (pointer:fine)`. The drag-and-drop sentence was removed from it.
- Height budget at 1366x768 (hint forced on): dock 155px, chat area 557px, welcome 533px, no scroll. Also fits at 1280x600.
- Send is **not** disabled when empty (the controller ignores empty sends, and a view-set `disabled` would go stale on programmatic `.value` writes).
- Chat column/composer max width is `--column-w: 860px` to stay aligned with the existing message column (spec target 768 is a Stage 3 decision).

### 12.6 Model selector (`js/components/model-select.js`)
`ModelSelect.init(selectEl)` (called by `ui.js`; idempotent), `ModelSelect.open()/close()/isOpen()`.
- `<select id="modelSelect">` stays the source of truth and stays in the DOM (visually hidden, `tabindex=-1`, `aria-hidden`). Choosing sets `select.value` and dispatches a bubbling `change` (verified once per real change, none when re-choosing the current model; `app.js` then stored `localStorage.model`).
- External sync: `select.value =` / `selectedIndex =` assignments (which fire no event, e.g. `app.js:327`) are intercepted by wrapping both setters on that element; option rebuilds are caught by a MutationObserver (an open list closes); the `change` event also re-syncs.
- UI: `button#modelTrigger[aria-haspopup=listbox][aria-expanded]` ("AI model: {label}") and a fixed popover `div.model-list[role=listbox]` (focus stays on it; `aria-activedescendant`; groups = optgroups as `role=group` with labelled captions). Keys: Enter/Space/ArrowUp/ArrowDown on the trigger, ArrowUp/Down/Home/End/PageUp/PageDown, Enter/Space select, Esc closes and returns focus, Tab closes, printable typeahead (skips unavailable options). Click outside / resize / outside scroll close it.
- Data shown: option label, description, availability. `UI.renderModelSelect(models, selected)` now also writes `data-label`, `data-description`, `data-reason` on each `<option>` (option text and `title` unchanged). Unavailable options are `aria-disabled`, not selectable, show "Unavailable - {reason}" (or "Unavailable"). Nothing else (no price, latency, capability).
- States: placeholder-only select = "Loading models..." (spinner, `aria-busy`, disabled); zero options = "No models available" (disabled; the controller swallows fetch errors today, so error and empty look the same); no match for `select.value` = "Select a model". Retry / `onRetry` (C7) is **not** implemented (needs the controller).
- Deviation: the spec's filter input for > 8 models is not built (typeahead only).

### 12.7 Breaking / behavioural changes
| Old | New | Controller impact |
|---|---|---|
| Legacy bottom sheet `max-width:768` | Left drawer < 768, rail 768-1023 | none |
| `UI.closeSidebar` hard-closed on `innerWidth <= 768` after a session click | `closeSidebar()` closes only an open overlay | none |
| Mode tab label "Document" | "Documents" | none (`data-mode` unchanged) |
| `.select-wrap__chevron` svg, `select` visible | Chevron removed; select hidden once the trigger exists | none |
| `.welcome__starter` = pill, text only | Feature card (button with icon/title/description) | `app.js:137` unchanged and working |
| `#themeToggle` in sidebar footer | In the top bar | none |
| Session row `div.session-item` | `li.session-item` (+ time instead of model/time meta) | none (callbacks unchanged) |

### 12.8 Still open
Stage 3 items (message rendering, AI visualization, workspace views), `.rst` (C14), upload retry (C6), `UI.confirm` adoption (C1), rename wiring (C2:
pass `{ onRename }` as the 5th argument of `renderSessionList` and the rename control appears), model retry (C7), C8/C9.

---

## 13. Stage 3 as implemented (messages, citations, AI visualization, smart scroll; frontend-developer, view-only)

`app.js`, `api.js`, `backend/**`, `tests/**` untouched. Verified in headless Chrome via scripted CDP against a static server (all non-GET and all
non-static requests blocked; no backend running; no chat/upload request made). Fixtures lived only in scratch scripts. Not exercised: real screen
readers, real touch, Safari/Firefox, a live stream from the backend (renderer methods were driven by fixtures), `(pointer:coarse)` (CSS only).

### 13.1 Script / style order and new ids
Scripts: `icons, utils, api, chat, theme, components/menu, components/dialog, components/model-select, components/ai-viz (new), ui, app`.
Stylesheets add `css/ai-viz.css` after `components.css`. New ids (view-owned): `#welcomeViz` (replaces `.welcome__logo`; holds the static sparkles
icon until `AIViz` mounts), `#jumpLatest` (inside new wrapper `.chat-stage` that now contains `#chatArea`). Dynamic message ids kept:
`thoughtSteps-*`, `streamContent-*`, `citationCards-*` (now the Sources block), `msgBadges-*`, `msgExtra-*`, `msgTime-*`; new `activity-*`.
`.message`, `.message--assistant`, `.message__regen` unchanged (history assistants also carry `.message--assistant`).

### 13.2 Decisions
- Column width: `--column-w` is now **768px** (spec 4.5) for messages and composer; `html[data-workspace="code"]` widens it to **880px**
  (`UI.setActiveMode` sets `document.documentElement.dataset.workspace`). Prose (`p, ul, ol, blockquote`) is capped at 68ch; tables/code use the column.
  Chat-area side padding 24px (16px below 768).
- User message: right-aligned tinted bubble, no avatar (the `userId` argument of `appendUserMessage` is accepted and ignored).
  Assistant: no bubble; 28px AIViz avatar at >= 640px, hidden below (article `aria-label` carries the identity).
- Welcome viz size: 96px (> 800px height), 64px (<= 800), hidden (< 720) to keep the Stage 2 fit at 1366x650 / 1280x600 (spec would show 64px down to 600).
- Action row of a streaming answer is `visibility:hidden` (space reserved, not focusable) instead of rendering disabled buttons; it appears on finalize / error.
- "Jump to latest": real (`#jumpLatest`, shown only when the reader is > 80px above the bottom and messages exist). Its accessible name is
  "Jump to latest message" (contains the visible text) instead of the spec's "Scroll to latest message".
- Copy now copies the answer's raw Markdown (not rendered text, which included code-block chrome).
- Web link rule: any citation whose `url` passes `Utils.safeUrl` (http/https) is rendered as a link, regardless of `type` (indexed web pages are `type:"doc"` with a url).

### 13.3 View API
| API | Behaviour |
|---|---|
| `AIViz.create({size:"sm"|"lg", state})`, `mount(container, opts)`, `setState(svgOrContainer, state)`, `getState`, `STATES` | New. State is `svg[data-state]`; unknown state ignored. 7 SVG nodes. Sets `html[data-page-hidden]` on `visibilitychange` (CSS pauses all viz animations) |
| `UI.setVizState("idle"|"processing"|"responding"|"error")` | New (C3). Drives the welcome instance; no-op if unavailable. `Chat` resets it to `idle` whenever the welcome screen is shown |
| `renderer.setVizState(state)` | New. Same states for the message avatar |
| Automatic viz in the renderer | `startAssistantMessage` -> processing; `addThought` -> processing (until first token); first `appendToken` -> responding; `finalize` -> idle; `showError` -> error; `markPartial({reason:"error"})` -> error. The controller need not call anything for the default mapping (spec 8.3) |
| `renderer.addThought / resolveThought` | Signatures unchanged. Steps render as `role=listitem` rows; the panel stays open while streaming, then collapses behind a disclosure button "Activity - N steps" (N = real step count) on `finalize` |
| `renderer.finalize(citations)` | Also renders the Sources block from `citations` only; `[n]` markers with no matching citation become plain text; empty answer shows "No response text was received." |
| `renderer.showError(msg, opts?)` | `opts.onRetry` (optional function): renders Retry + Dismiss (Dismiss removes the message). Without it, no buttons (same as before) |
| `renderer.markPartial({reason:"stopped"|"error", message?})` | New, optional. Note under the answer ("Stopped..." / "Response interrupted..." + server `message` verbatim). Controller call is Stage 4 (`AbortError` -> `stopped`; `error` event with partial text -> `error`) |
| `renderer.setBadges({model, cached, fellBackFrom})` | Unchanged; fallback shown as visible text "Fell back from ...", full list in `title` |
| `Chat.renderHistory(messages, handlers)` | Unified with live rendering; still returns void. `handlers.onRegenerate` (optional) is offered on the newest assistant message only. Reads real persisted meta: `model`, `cached`, `fell_back_from`, and `partial` / `stopped` (shown as notes). No activity steps are shown for history (none are persisted) |
| `Chat.copyMessage / getCitation / clearChat / showThinking / removeThinking` | Kept |

Sources block: `div.sources[role=group][aria-label="Sources (n)"] > ul > li.source` (number = `citation.id`, else position). Citation with safe url: `<a target=_blank rel="noopener noreferrer">`
(+ domain from the url, page when present, "Show excerpt" toggle for the snippet). Otherwise a `<button>` that calls `UI.showCitationModal(c)` with the unchanged citation object. All text via `textContent`.
No favicon fetches, no counts beyond `citations.length`. `.citation-card(s)` classes are retired. Research "no sources returned" note is not rendered (the view does not know the mode of a message).

Markdown additions (`utils.js`, after sanitising): tables are wrapped in `div.table-wrap[role=region][tabindex=0][aria-label=Table]` (scrolls inside; page never widens);
code-block label comes from the fence (`Code` when absent, never an auto-detected guess); sanitizer policy unchanged (class/role/download/ping still forbidden pre-sanitising).

Smart scroll (`chat.js`): follows new content only while within 80px of the bottom; sending a message and restoring history always go to the bottom; scrolling is `behavior:"instant"`.

### 13.3a Citation badge hit areas (coarse pointers)
`Utils.decorateCitations` marks genuinely adjacent badges (no text or space between, e.g. `[3][4]`) with `data-cite-next` / `data-cite-prev`; the coarse-pointer CSS shortens only those facing sides.
Badges separated by text or a space keep the full 44x44 zone (a space-separated pair overlaps by a few px; the later badge wins the overlap). Citation ids must be safe integers (`Number.isSafeInteger`); others are re-assigned. The dialog title uses the same label as the Sources row (source, else url host, else `Source N`).
Known limitations: truly adjacent badges keep only ~24-37px width (WCAG 2.5.8 spacing exception). QA-CIT-2 (accepted): on stacked lines the 44px-tall zones of badges on consecutive lines can overlap vertically.

### 13.4 Breaking / behavioural changes (none for `app.js`)
`.welcome__logo` -> `#welcomeViz.welcome__viz`; `.citation-card` -> `.sources`/`.source`; user message no longer has `.message__avatar`; `.thought-step` is no longer a coloured pill;
`#chatArea` is wrapped in `.chat-stage`; `renderer.showError` markup now has `.message-error__text`. All ids/selectors in section 1 are unchanged (checked by `tests/test_frontend_guards.py`).

### 13.5 Controller requests for Stage 4 (optional, additive)
Call `renderer.markPartial(...)` on Stop / partial error; pass `{onRetry}` to `showError`; pass `onRegenerate` to `renderHistory`; call `UI.setVizState` only if a welcome-level state is wanted. STEP 0: the welcome card "Asks:" line now allows 2 lines at every height.

---

## 14. Stage 4 as implemented (integration-developer; `app.js` and `api.js` only)

Verified in headless Chrome (CDP) with `window.fetch` replaced by an in-page stub (SSE bodies are stub `ReadableStream`s); static files were served by CDP
request interception, so nothing reached `localhost:8000` and no backend ran. 73 scripted checks passed (stub only). Real-backend streaming is NOT verified.

### 14.1 Controller changes
- **Welcome cards never send (S4-1).** `#starterPrompts` click: switch mode (`_setMode`, idempotent, PATCH only when the mode changed and a session exists), fill `#chatInput` with `data-prompt`, auto-resize, focus. No `createSession`/`chatStream`. Works with the earlier `ui.js` listener that clicks the `.mode-tab` (now redundant, harmless). Click, Enter and Space tested on each of the 4 cards.
- **Drafts.** The composer is cleared only after the server accepted the request (`API.chatStream(payload, handlers, signal, { onOpen })`, called after HTTP 2xx). Failures before that (session creation, 429/503/404, network) leave the text, add no bubbles, and show an error toast (message + `Retry-After`). A stream `error` before the first token restores the text (merged with anything typed since) and shows `renderer.showError(msg, { onRetry })`. The textarea is no longer disabled while a request runs (only Send is gated by `isLoading`); Stop replaces Send and focus returns to the composer.
- **Retry.** Inline Retry sends `regenerate:true` (the server already stored the user turn and `pop_last_exchange` keeps it), so no duplicate user turn. Re-sending the restored draft **unedited** (composer text === `State.failed.text`, same session) is routed through the same `{regenerate:true, message:""}` path as Retry (M1), so the server never stores the question twice and the DOM keeps exactly one user bubble; the composer is cleared only once that request is accepted. An **edited** draft is sent normally (`regenerate:false`) and replaces the failed bubbles. `retry_after` (error event) and `Retry-After` (HTTP) set a cooldown, passed to the view as `renderer.showError(msg, { onRetry, retryAfter })` (countdown Retry); Retry/Send during it make no request and toast the seconds left; `UI.setLiveStatus("You can send again.")` at expiry.
- **Regenerate.** The old answer and its buttons are removed only once the new stream is accepted; a failed regenerate leaves the previous answer (toast says so). `onRegenerate` is also passed to `Chat.renderHistory`.
- **Stream contract.** `meta` (viz `processing`), `status`, `citations`, `model`, `token`, `done` (`fell_back_from`, `cached`, `persisted:false` -> info toast), `error` (`retryable`, `retry_after`, `partial`, `message_id`). Stop -> `markPartial({reason:"stopped"})`; error with text -> `markPartial({reason:"error", message})` plus actions when `message_id` exists; a stream that ends without `done` is treated as interrupted.
- **C1** `UI.confirm` for chat and document delete. **C2** `onRename` (+ `onRetry`) as 5th argument of `renderSessionList`; `PATCH /sessions/{id} {title}` (max 200, backend `SessionUpdate`); failure rethrows so the view reverts, old title kept, toast. The session filter is also applied after each reload. **C7** `renderModelSelect(models, selected, { error, onRetry })` on failure (3rd argument ignored by the current view; `onRetry` is a real `GET /models`). **C8** icon names, no emoji in toasts. **C9** no `style.color`; `aria-pressed` set at init and on click. **C14** `rst` in `ALLOWED`.
- Uploads: one at a time; the `File`/URL is kept in a closure. On failure `UI.showUploadError(filename, { message, onRetry })` (C6, implemented in `ui.js`, see 15) shows the server's own text with Retry; the feature-detect fallback (error toast, URL dialog reopened) remains only as a safety net.
- **Stage 4 QA fixes.** M2: `API.createSession(data, signal)` (payload-neutral) receives the send's `AbortSignal`, so Stop works while `POST /sessions` is pending: draft kept, no bubbles, loading reset; if the creation completes just after abort the session id is kept (`if (!State.sessionId)`) and reused by the next send (no orphan, no second `POST /sessions`). N1: Enter/Send/session-select while streaming shows the info toast "A response is in progress. Press Stop to cancel." at most once per 4 s, draft kept. N2: Stop before the first token (non-regenerate, same session) removes the empty assistant bubble, restores the draft only if the composer is empty, sets `State.failed` (`assistantEl: null`) so M1 applies, and toasts "Generation stopped. Your message is kept; press Send to try again."; Stop with partial text is unchanged (`markPartial({reason:"stopped"})`). N4: `api.js` dispatches an SSE event only if `Object.prototype.hasOwnProperty.call(handlers, event)` and it is a function (`__proto__`/`constructor`/`toString` ignored); framing stays LF-only (comment in code; backend emits LF only).
- Verification (stub only): in-page `window.fetch` stub with `ReadableStream` SSE, static files served via CDP interception, no backend, no real POST/PATCH/DELETE; 20 scripted checks for M1/M2/N1/N2/N4 passed.
- Enter ignores IME composition. A failed `GET /models` no longer clobbers the stored model.

### 14.1a QA follow-up fixes (supersede the N2/M1 wording above where they differ)
- **Persistence proof (Minor-1).** HTTP 200 + `onOpen` only means the stream was accepted; Starlette sends headers before `chat.stream()` runs `add_message`, and `meta` is the first event yielded after persistence. `State.failed` (the unedited-resend-as-`regenerate:true` path) is therefore set only after any SSE event other than `error` arrived for that send (`persisted` flag in `_onSend`). If the stream closes/fails/is stopped with zero events, the send is treated as not stored: the optimistic user bubble and the empty assistant bubble are removed, the draft is restored (`_restoreDraft`), `State.failed = null`, and an unedited Send is a NORMAL send (`regenerate:false`, text). Zero-event failures show an error toast ("... Your message is kept.") instead of an inline error bubble, because inline Retry would regenerate.
- **Regenerate + Stop before the first token (Minor-2).** The removed previous answer node is kept (`oldAnswer`) until the stream produces text; on Stop with no text the empty bubble is removed and the old answer (with its Regenerate button) is put back in place; toast "Generation stopped. Your previous answer is restored." Errors before the first token still use the inline Retry (regenerate) as before.
- **Pre-token Stop toast (Minor-3).** The draft is restored with `_restoreDraft` (merged with anything typed since, e.g. "Q\n\nnewer"), so "Your message is kept; press Send to try again." is true; if the session changed, the toast is just "Generation stopped."
- **Nit-1.** When the inline Retry shows a `retryAfter` countdown, the controller no longer announces "You can send again."; only `chat.js` announces "Retry is available.".
- **Known limitation (Minor-4, not fixed; needs a backend change).** If a send failed after the server accepted and stored the user turn (`meta` seen) and the user then EDITS the restored draft and sends it, that is a normal send (`regenerate:false`): the earlier user turn stays on the server, so the stored history holds both turns while the DOM shows only the new one until reload. Fixing it needs a backend way to replace/delete the last unanswered user turn (e.g. a `regenerate` variant that accepts replacement text).

### 14.2 Cross-boundary requests
1. `frontend/index.html:376`: `accept=".pdf,.docx,.csv,.xlsx,.txt,.md"` -> `accept=".pdf,.docx,.csv,.xlsx,.txt,.md,.rst"` (and `.rst` in the dropzone copy). Landed (see 15).
2. `ui.js`: implement `UI.showUploadError(filename, { message, onRetry })` (C6) so upload retry is a real control; render the C7 error state and retry in `renderModelSelect`; optionally drop the `.mode-tab` click in the `#starterPrompts` listener; the welcome card "Asks:" copy is now inaccurate (cards only fill), consider "Starts with:".
3. `chat.js` (optional): `showError` option for a disabled/countdown Retry (e.g. `opts.retryAfter`); today early clicks get a toast.

## 15. Stage 4 view-side follow-up (frontend-developer; answers 14.2)

Verified in headless Chrome against a GET-only static server with `window.fetch` stubbed to reject (no non-GET request possible); nothing was sent.

- **`.rst`** added to `#fileInput[accept]` and the dropzone helper copy (`accept` now equals `app.js ALLOWED`).
- **C6 `UI.showUploadError(filename, { message, onRetry })`.** Reuses `#uploadToast`: title "Upload failed" (`#uploadToastTitle`, new id), progress bar hidden, `message` via `textContent`, `role="group"` + `aria-label="Upload failed"` (it is `role="status"` again on `showUploadToast`), announced once via `#liveStatus`, focus not moved. Buttons: Retry only when `onRetry` is a function (hides the toast, then calls `onRetry()`), Dismiss always; Esc inside the toast dismisses. 44px targets on coarse pointers. `showUploadToast`/`hideUploadToast` signatures unchanged and both reset the error state.
- **C7 `UI.renderModelSelect(models, selected, { error, onRetry })`** -> `ModelSelect.setError(msg, onRetry)` / `ModelSelect.clearError()` (new exports). Error with no options: trigger reads "Couldn't load models" (alert icon, disabled, full text in `aria-label`/`title`), distinct from "No models available". A `.model-retry` button ("Retry loading models") is rendered only when `onRetry` is given; with options already loaded the list stays usable and Retry is still shown. A render without `error` clears the state. The native `<select id="modelSelect">` is still the source of truth.
- **Welcome cards** now read "Starts with: “<prompt>”" (they only fill the composer). Card `aria-label` format (WCAG 2.5.3, starts with the exact visible title): `<Visible title>. Switches to <Workspace> and fills the message box with: <prompt>`, e.g. `General chat. Switches to General and fills the message box with: Explain quantum computing in simple terms`; titles/workspaces: General chat/General, Analyze documents/Documents, Web research/Research, Write and debug code/Code. The `ui.js` `#starterPrompts` listener that clicked the `.mode-tab` was removed: `app.js` `#starterPrompts` handler calls `_setMode(card.dataset.mode)` (updates `State.mode`, localStorage, `UI.setActiveMode`, PATCH if changed) before filling, so it is redundant.
- **`Chat` `showError(msg, { onRetry, retryAfter })`**: `retryAfter` (seconds, > 0) renders Retry disabled with a "Retry in Ns" text updated once a second (chained `setTimeout`, no `setInterval`, not live-announced per tick) and a single "Retry is available." announcement at 0. `app.js` now passes `retryAfter`. `showError` no longer calls `#liveStatus`: the `role="alert"` box is the single error announcement (previously it was announced twice); `#liveStatus` is used only for "Retry is available." at countdown end.

## 16. Stage 4 seam and view hooks (frontend-developer)

Verified: `node --check` on every js file, `pytest tests/test_frontend_*.py` (94 passed), static screenshots at 1440 and 390 (GET-only static server). Interaction of the new hooks NOT exercised in a browser.

### 16.1 Stage 4 seam: welcome cards (S4-1)  -- controller must wire
`UI.onWelcomeCard(handler)`; `handler({ mode: string, prompt: string, action: string })` (`mode` = card `data-mode`: `general|document|research|code`; `prompt` = card `data-prompt`). Pass `null` to unregister.
- Fires for mouse click and for Enter / Space (cards are `<button>`s, so both arrive as `click`). The view does nothing else: it never switches mode, fills the composer, focuses or sends.
- Controller expected behaviour: `_setMode(mode)`; put `prompt` in `#chatInput` (resize); `UI.focusComposer()` (or `.focus()`); NO `createSession` / `chatStream`.
- Implementation: capture-phase listener on the ancestor `#welcomeScreen` (ui.js). Once a handler is registered it calls `preventDefault()` + `stopPropagation()`, so the legacy bubble listener on `#starterPrompts` in `app.js` no longer runs (no double handling). With no handler registered nothing changes: the existing `app.js` listener keeps working. The controller can therefore migrate by calling `UI.onWelcomeCard(...)` in `init()` and deleting its `#starterPrompts` listener (the element must still exist while that listener is unmigrated).

### 16.2 New view functions (all additive)
| API | Behaviour |
|---|---|
| `UI.focusComposer() -> boolean` | Focuses `#chatInput`, caret at end; false when hidden/disabled |
| `UI.setComposerBusy(busy)` | Optional replacement for `_setLoading`'s DOM part: swaps `#sendBtn`/`#stopBtn` (sets both `hidden` and inline display), `aria-busy` on `.composer`, never disables `#chatInput`; if focus was on Send/Stop it moves to the composer. Stop (`#stopBtn`) stays in `.composer__send`, always reachable while busy |
| `Chat.holdLastAnswer() -> {restore(), discard()} \| null` | Regenerate hook: hides (does not remove) the newest `.message--assistant`. Call before `Chat.startAssistantMessage()`. `discard()` removes it after the new stream succeeded; `restore()` unhides it on failure/Stop. Idempotent. Note `app.js` selector `#chatArea .message--assistant` also matches the held node |

### 16.3 Backlog items already present in the view (no new work)
C1 `UI.confirm`, C2 `renderSessionList(..., { onRename })` (rename button appears only when supplied), C7 `renderModelSelect(models, sel, { error, onRetry })`, C8 icon names (and legacy emoji map), C9 `#searchToggle[aria-pressed]` styling, `.rst` in `#fileInput[accept]`.

### 16.4 Visual changes
Per-workspace card accents (tokens `--success`, new `--info` / `--info-bg`, `--warning`; icon + title still carry meaning); static capability strip `ul.capabilities` (three real features, hidden < 900px wide or < 860px tall); mobile (< 640px) cards become compact rows with the "Starts with" line clamped to one line (not hidden; guard test forbids hiding it). New `.icon--20`.
Not built: bottom navigation bar (modes stay in the drawer/rail), orbiting satellite icons around the orb, notification/search/avatar/settings controls from the reference (no backing actions), "Secure & private" claim (no evidence).

---

## 17. Reference-design pass (frontend-developer; "AI Assistant Pro Responsive Dashboard.png")

User instruction: the UI must be the SAME as the reference (light desktop, dark, tablet, mobile light/dark). This supersedes the "Not built" list in 16.4 (orbit
satellites, bell, search, avatar, settings, capability copy are now built). `app.js`, `api.js`, `backend/**`, `tests/**` untouched. Verified: `node --check` on every
js file; `venv/bin/pytest -q tests/test_frontend_*.py tests/test_api.py` (124 passed); headless Chrome screenshots at 1440x1024, 1366x768, 1024x768, 768x1024, 390x844 in light and
dark (default = no stored pref + OS light => light reference look; dark via OS dark) against a scratch static server that mocks only GET `/models`, `/sessions`, `/documents`
(no POST/PUT/DELETE possible); scripted CDP checks (19 assertions, all behaviours below marked "tested") with the same mock. Not exercised: real backend, touch, screen readers.

### 17.1 Files and load order
New: `css/reference.css` (after `ai-viz.css`), `js/components/notifications.js` (after `ai-viz.js`, before `ui.js`). Order: `icons, utils, api, chat, theme, components/menu,
components/dialog, components/model-select, components/ai-viz, components/notifications, ui, app`. Tokens added in `main.css` (both themes): `--accent-fill`, `--on-accent-fill`,
`--grad-accent(-hover)`, `--page-glow`, `--orb-*`, `--sat-*`, `--switch-off`, `--border-ui`, `--shadow-accent/-soft`, `--card-{gen,doc,res,cod}-{bg,bd,tile,fg}`.
`--sidebar-w` 256px; light sidebar `#F0F1FC`, workspace white; dark = navy surfaces.

### 17.2 Ids and markup the controller may rely on
Unchanged (all of section 1). New view-owned ids: `#recentToggle`, `#recentPanel`, `#sessionFilterBtn`, `#myDocsBtn`, `#sidebarAvatar`, `#topAvatar`, (`#topModeTabs` / `.topnav__tab` were REMOVED, see 17.3),
`#themeLight`, `#themeDark`, `#themeQuick`, `#notifBtn`, `#notifDot`, `#topSearchBtn`, `#composerModel`. Changed meaning: `#themeToggle` is now the sidebar **settings gear** (still opens the
Light/Dark/System menu via `UI.toggleTheme`; `app.js` binding unchanged). `#userIdDisplay` now holds only the id text (e.g. `default`), not `user: default`. `#docSection` starts `hidden` and is
opened by the My Documents card. `#modePrompts` stays but is `hidden` (not in the reference). `#searchToggle` is a toggle button drawn as a switch (still `aria-pressed`, still written by app.js;
role=switch was not used because `aria-pressed` is not allowed on it). The `.dock-hint` paragraph was removed from the markup.
Welcome cards: titles/copy per reference (General Chat, Analyze Documents, Web Research, Write & Debug Code); the "Starts with" line is kept in the DOM (`.feature-card__try.sr-only`) but visually hidden;
`data-mode` / `data-prompt` / aria-label format unchanged. The Analyze Documents card therefore still fills "Summarize the uploaded document" (guard tests require a non-empty prompt); it does not open the file picker.

### 17.3 Behaviour implemented in the view (works without controller changes)
- **Top pill tabs removed** (breaking, view-only): `#topModeTabs` and `.topnav__tab` no longer exist; mode switching is via `#modeTabs .mode-tab[data-mode]` (sidebar, rail, mobile drawer). `UI.setActiveMode` now marks only `.mode-tab`. `#contextTitle` is now the visible top-bar title (left) at >=640px and visually hidden on mobile (brand shown). `app.js` never referenced the removed ids.
- **Recent Conversations** heading collapses/expands `#recentPanel` (not persisted). **Filter** button (`#sessionFilterBtn`, menu: All conversations / Today / Previous 7 Days / Older) filters the
  sessions the controller last rendered; state is view-only and re-applied on every `renderSessionList` (tested).
- **Session grouping**: `renderSessionList` now groups by the real `updated_at` into **Today / Previous 7 Days / Older** (calendar days ago: 0 = Today, 1-7 = Previous 7 Days, 8+ = Older; `Utils.formatDateGroup` returns `Previous 7 Days` for days 2-7). Row time: today = `h:mm AM/PM`
  (locale), others = short date ("Oct 16"). Inputs unchanged (`sessions[].updated_at`); no fake rows.
- **My Documents card** (`#myDocsBtn`, disclosure): first click opens `#docSection` (fed by `renderDocList`) and switches to the Documents workspace; second click closes the list. Default action = click the Documents tab without closing the
  drawer; override with `UI.onMyDocuments(fn)` (see 17.5) (tested).
- **User row**: avatar initial = first letter of the userId; name = userId; second line "User ID" (no email is invented); `UI.setUserIdDisplay(userId)` feeds both avatars. The top-bar avatar is a non-interactive `role=img`.
- **Theme**: top-bar Light|Dark segmented switch (`#themeLight/#themeDark`, `aria-pressed` = resolved theme) sets an explicit preference; `#themeQuick` (< 640px) toggles; gear menu keeps System (tested).
- **Search icon** (`#topSearchBtn`): `UI.focusSessionSearch()` opens the drawer/rail overlay or expands the collapsed rail, opens Recent, focuses `#sessionSearch` (tested).
- **Bell** (`Notifications`, `components/notifications.js`): in-memory list (max 20, newest first) of what the app showed through `UI.showToast` and `UI.showUploadError` during this page session; popover `role=dialog`, "Clear all" (view action), empty state
  "No notifications"; the dot is shown only while unread items exist; opening marks read; Esc/outside click close and return focus to the bell (tested). Nothing is persisted or invented.
- **Model**: top-bar pill = provider letter tile (from the optgroup/provider name), label, provider on a second line; composer chip (`#composerModel`) mirrors the selection and opens the same listbox (`ModelSelect.bindChip`, `ModelSelect.open(anchor)`);
  choosing dispatches the real `change` on `#modelSelect` (tested: label, provider, `localStorage.model` updated by app.js). Failed load: compact "Models unavailable" pill + icon Retry (top bar) and a disabled chip.
  The top-bar pill and search icon are hidden when `.main` is < 900px wide (the chip covers the model); on mobile the chip shows only the logo + chevron.
- **Hero**: `AIViz.create/mount({ size: "hero" })` builds the orb scene (two eye pills, two tilted orbit rings with orbiting dots, document/globe/code satellites, sparkle); `UI.setVizState` still drives `idle|processing|responding|error`
  (tested). Animations are CSS transform/opacity only, paused when the tab is hidden and disabled under `prefers-reduced-motion` (the existing `.ai-viz *` rules).
- **Responsive**: >= 1024 viewport: 256px sidebar, 4 cards in a row (container >= 900px), capability strip. Tablet 768-1023: 72px rail, tabs without icons, 2x2 cards, compact strip, composer docked. <= 639px: header with hamburger,
  centred brand, theme toggle, bell, avatar; scrollable tab row; compact orb; cards as rows; strip hidden; drawer for the sidebar. No bottom navigation. The strip is also hidden when the viewport is shorter than 790px.

### 17.4 Capability strip copy (static)
Multiple AI Models / "Choose the best model for your task"; File Upload / "PDF, DOCX, TXT, images and more" (backend `SUPPORTED_EXTENSIONS` includes images); Web Search / "Get real-time information";
Secure & Private / "Your data stays yours". **The last item was reworded (QA-05 M1) to "Safe Uploads / File type and size are checked"** (backend enforces `SUPPORTED_EXTENSIONS` and the upload size limit). No privacy claim is made anywhere.

### 17.5 New view functions (all additive)
| API | Behaviour |
|---|---|
| `UI.onMyDocuments(fn \| null)` | Optional. `fn()` is called instead of the default (the view clicking the Documents tab) when the card opens. Controller wiring: none needed. |
| `UI.focusSessionSearch() -> boolean` | See 17.3. |
| `UI.notify(text, type)` / `UI.clearNotifications()` | Record a notice in the bell list without a toast (`type`: `info\|success\|error`). `UI.showToast`/`UI.showUploadError` already record. |
| `Notifications.record/open/close/isOpen/clear/count` | Component API behind the above. |
| `ModelSelect.bindChip(btn)`, `ModelSelect.open(anchorEl?)` | `bindChip` is called by `ui.js` for `#composerModel`. |
| `AIViz.create({ size: "hero" })` | New variant (a `div.ai-viz.ai-viz--hero`, same `data-state` contract). |

### 17.6 Controller wiring list
Required: none (the app boots and every new control works with the current `app.js`). Optional, marked **controller work**: (a) call `UI.notify(msg, "error")` for errors it only logs or renders inline (e.g. stream errors) if those should appear in the bell;
(b) `UI.onMyDocuments(fn)` if the Documents workspace should also trigger a document reload or the file picker; (c) the controller still owns session titles, grouping inputs are `updated_at`; (d) bind `#topModeTabs` directly instead of the forwarded click if preferred
(then remove the forward in `ui.js _wireReferenceShell`).

### 17.7 Breaking / behavioural changes
`#themeToggle` moved from the top bar to the sidebar footer (gear). `#userIdDisplay` text format. Session groups reduced to three names (`Utils.formatDateGroup` still returns the five names; the mapping is in `ui.js`). `.welcome__starter` card titles
changed ("General chat" -> "General Chat" etc.). Placeholder is "Ask anything...". `#sidebarCollapse` icon is `chevrons-left`. The `.feature-card` border is the tinted card colour, not `--border-control` (reference look; the guard test only inspects `components.css`,
which still has the 3:1 rule that `reference.css` overrides). Search/filter/composer outlines use the softer `--border-ui` (QA-05 m1: earlier value was 1.50:1 light / 1.62:1 dark, the old "about 1.9:1" was wrong). Now `--border-ui` = #8790AF light (3.16:1 on `--surface`) and #56669E dark (3.14:1 on `--surface`), guarded by `tests/test_frontend_reference.py`.

**Welcome-card border exemption (QA-05 m2).** The tinted `--card-*-bd` borders (1.28-1.38:1 light, 1.91-2.73:1 dark) are intentionally NOT held to 3:1: each card is a button whose state and purpose are carried by its visible text label (title, description, action), the fill is tinted, and focus shows the 2px `--ring` outline. Card text (`--card-*-fg` and `--text` on both gradient stops) is >= 4.5:1 in both themes (asserted in the test file).


### 17.8 QA-05 follow-ups
- **Upload Files card hook (m3).** The Analyze Documents card carries `data-action="upload"` (plus its unchanged `data-prompt`). `UI.onWelcomeCard` handlers now receive `action` (`"upload"` for that card, `""` otherwise).
  Controller contract (attribute order in markup: `data-action` precedes `data-mode`): `if (card.action === "upload") { switch to the document mode; document.getElementById("fileInput").click(); }` and still fill the composer with `prompt` without sending. The view never opens the picker itself and never sends.
  With the legacy `app.js` `#starterPrompts` listener the attribute is `dataset.action`.
- **Mode tabs (n2).** `.mode-tab` / `.topnav__tab` use `aria-pressed="true|false"` (set by `UI.setActiveMode` and in the initial markup); `aria-current="page"` is no longer set. CSS keys off `[aria-pressed="true"]`. This supersedes the `aria-current` statements above.
- **Reduced motion (n3).** The card hover lift is disabled under `prefers-reduced-motion: reduce`.
- **Short viewports (n4).** At `max-height: 790px` the welcome badge and subtitle are hidden and cards compact so all four cards sit above the docked composer at 1024x768.
