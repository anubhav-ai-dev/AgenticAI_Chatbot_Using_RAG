# QA report 05 - Reference-design pass (reference.css, notifications.js, shell restyle)

## 1. Verdict: PASS WITH ISSUES
The UI is very close to the reference in all five panels. Every new control works in headless Chrome (stubbed fetch, no live calls). qa-report-04's Major (toast covering Stop) is fixed. axe wcag2a/2aa/21aa gives 0 violations in 32 runs (8 viewports x 2 themes x welcome/chat) plus 14 runs with overlays open, and there is no horizontal overflow. No blocker. One Major (honesty): the "Secure & Private - Your data stays yours" claim. The rest are Minor/Nit: soft outlines far below 3:1, a misleading "Upload Files" CTA, a few fidelity gaps, and guard-test gaps (closed by a new test file).

## 2. Scope
Working tree vs 15e1d81, frontend only. Read in full: notifications.js, reference.css (first 80 lines plus targeted greps), ui.js session/avatar/notify paths, tokens in main.css. Docs read: PROJECT_BRIEF, integration-notes, qa-report-04, view-contract 16-17, reference PNG. Not touched in this pass by mtime: app.js, api.js, utils.js, chat.js. Backend and test files modified in the tree are the same out-of-scope set as qa-04.

## 3. Findings
- **M1 (Major) Unsupported privacy claim.** `frontend/index.html:467` "Secure & Private / Your data stays yours". Prompts and retrieved text go to third-party LLM and search providers (view-contract 17.4 admits this). Master task and brief forbid security claims the backend does not supply. No guard covers it (`test_no_unverifiable_marketing_claims` word list omits it). Fix: reword to something verifiable, or drop the item, or get the product owner's sign-off. Owner: frontend-developer (copy); decision: orchestrator/user. The new xfail test `test_no_unverifiable_privacy_claim_in_capability_strip` tracks it.
- **m1 (Minor) Soft outlines fail 3:1.** `--border-ui` is #CDD2E8 on #FFFFFF = **1.50:1** (light) and #2E3A70 on #10173A = **1.62:1** (dark). The contract says "about 1.9:1", which is wrong. It applies to the session search field, composer, Web Search pill, model chip/pill, filter and divider (reference.css:89,122,140,215,488,504,523,533,566). Recommended compliant values: light **#8A93B2 = 3.04:1**, dark **#56669E = 3.14:1 on surface** (3.41 on app-bg). Note this is clearly darker than the reference's hairline. A middle option is to apply 3:1 only to the text-entry boundaries (search field, composer) and keep the pills soft, since they carry visible text. The composer already switches to `--ring` on focus. Owner: frontend-developer (`main.css:138,230`).
- **m2 (Minor) Welcome-card borders and a bypassed guard.** Tinted `--card-*-bd` are 1.28-1.38:1 light and 1.91-2.73:1 dark. `test_feature_card_border_uses_control_token` still passes only because it reads components.css, and reference.css overrides it (reference.css:448). Cards are text buttons with tinted fill, so a documented exemption is defensible. Either darken the borders or have the designer record the exemption. Owner: frontend-developer / ui-designer. Tracked by the xfail `test_welcome_card_borders_meet_3_to_1_in_both_themes`.
- **m3 (Minor) "Upload Files" CTA does not upload.** `index.html:428`. The Analyze Documents card only switches to the Documents mode and fills "Summarize the uploaded document" (12/12 activations confirm there is no send and no file chooser). Label promises an action that does not exist, and sending that prompt with no documents is meaningless. Fix: have the card open `#fileInput` through `UI.onWelcomeCard`/the controller, or relabel it "Start" / "Ask about documents". Owner: frontend-developer (label), integration-developer (picker).
- **m4 (Minor) Guard-test gaps.** `VIEW_JS` in test_frontend_guards.py omits notifications.js. The hex-colour, border and keyframe checks never read reference.css. Closed by the new `tests/test_frontend_reference.py` (5 passing, 4 xfail). Owner: qa-reviewer (done).
- **n1 (Nit) Group names vs behaviour.** `Utils.formatDateGroup` puts a chat from 7+ days ago in "Older"; "Previous 7 Days" covers 1-6 days. view-contract 17.3 says "30 days+ is Older" (wrong). Owner: frontend-developer (doc).
- **n2 (Nit) `aria-current="page"` on the top workspace tabs and sidebar mode tabs.** They switch a chat mode rather than navigate. `aria-pressed` or a tab pattern is more accurate. Behaviour and names are fine. Owner: frontend-developer.
- **n3 (Nit) Reduced motion.** No running animations under `prefers-reduced-motion`. The card hover still applies translateY(-2px), instantly, which is not motion.
- **n4 (Nit) 1024x768 welcome.** Cards 3-4 sit under the docked composer and need a scroll. No overflow. Reference tablet is portrait and matches.

### Fidelity differences (screenshots in `/tmp/claude-1000/-home-anubhav-personal-proj-AgenticAI-Chatbot-Using-RAG/caf02d66-07c4-4aa7-91dd-f93346a21f08/scratchpad/shots/`: d1440_light/dark, t768, t1024, m390_light/dark)
Layout, colours, gradient cards, capability strip, composer, rail, dark theme and mobile row-cards all match closely. Concrete gaps:
1. Brand logo is a sparkles glyph in a gradient tile; the reference uses a triangle mark.
2. The hero orb and rings are smaller than the reference (about 84 px core vs about 110 px) and sit lower with a larger empty gap above. Satellites are placed similarly.
3. The model pill and chip use a letter tile ("O"), not a provider logo (reference shows the OpenAI mark). Honest, since no logo data exists.
4. Session rows truncate sooner and show pencil/trash on hover or touch (reference has times only). With a hover-capable pointer the time shows at rest, matching the reference.
5. Sidebar is 256 px; the reference is about 273 px at the same scale. Nav item row heights are a little taller in the reference.
6. Mobile shows 4 tabs (General, Documents, Research, Code) where the reference shows 3. The mobile chip shows only the tile and chevron (the reference shows "GPT-4o"). Mobile dark header shows a sun where the reference shows a moon; this is the "switch to" icon, defensible.
7. The reference top bar has no collapse icon on tablet; ours has a sidebar toggle at 768. The brand tagline "Smarter. Faster. Together." is present.
8. User row shows the real userId ("default" / "User ID"), not the reference's name and email. Intentional, no invented data.

## 4. Verified with evidence (puppeteer-core 23.11.1, axe-core 4.10.2 in scratch dir `.../scratchpad/pkg`; Chrome via `/usr/bin/google-chrome`; hover/pointer-fine emulated with `--blink-settings`; stub fetch; static server on 127.0.0.1:8765, stopped)
- **qa-04 regressions.** Toast now top-right (1280x800: y 64-108; 390x844: y 112-156). `elementFromPoint` at the Stop centre returns the Stop button at both sizes, and a real mouse click on Stop stops the stream (Stop hidden afterwards). Welcome cards: 4 cards x {click, Enter, Space} = **12/12** (composer == data-prompt, mode stored, welcome visible, 0 user messages, 0 non-GET requests). Streaming: textarea stays enabled, Stop visible, Send hidden. Stop gives "Stopped. This answer may be incomplete." Failed send keeps the draft with no user bubble. Failed regenerate keeps the old answer. Regenerate payload `regenerate:true`. Drawer at 390 and 768: focus moves inside, Escape closes, focus returns to `#sidebarToggle`.
- **New controls.**
  - Bell: `aria-expanded` toggles, label becomes "Notifications, 1 unread" after an error toast, dot is hidden until unread. Panel is `role=dialog`, shows "No notifications" when empty, lists the real toast text, Clear all works by keyboard, Esc and outside click close, focus returns to the bell.
  - Search icon: focuses `#sessionSearch`.
  - Filter menu: radio items with `aria-checked`, ArrowDown+Enter selects "Today" and only the Today group remains, Home returns to All, Esc returns focus.
  - Recent toggle: Enter and Space toggle `aria-expanded` and the panel.
  - Gear: menu Light/Dark/System. A pick sets `data-theme` and `localStorage.theme`; the segmented Light/Dark buttons update `aria-pressed`.
  - My Documents: opens the doc list (real `/documents` row), switches mode to document, second click closes.
  - Model chip: Enter opens the listbox, the unavailable model is `aria-disabled`, ArrowDown+Enter updates `#modelSelect` and `localStorage.model` (m2), Esc returns focus, same at 390. The top pill opens the same listbox.
  - Web Search: Space flips `aria-pressed`, and the next `/chat/stream` payload carries `web_search:false`.
  - Top tabs: Enter on Research sets `localStorage.mode=research`, marks both tab sets, and with a selected session sends `PATCH /sessions/s1 {"mode":"research"}`.
  - Grouping: Today / Previous 7 Days / Older from `updated_at` (fixture 1 h, 3 d, 20 d ago); times at rest in the Today group.
  - Accessible names present: Notifications, "Search conversations", "Filter conversations by date", "AI model: GPT-4o. Change model", "Light theme", "Settings: theme (System)". The tabs are named buttons.
- **A11y.** axe 0 violations at 1440x900, 1280x800, 1024x768, 768x1024, 430x932, 390x844, 375x812, 320x700 x light/dark x welcome/chat, and 14 more with bell, model listbox, filter menu, My Documents or drawer open (`axe.txt` in scratchpad). `scrollWidth==clientWidth` and the composer is in view in all 32. Focus ring is 2 px solid rgb(79,70,229) light / rgb(139,131,255) dark on 13 sampled controls. Reduced motion shows 0 running animations (8 running otherwise). Outline contrast computed with `python3 -I` from main.css tokens (m1, m2).
- **Honesty.** No fake sessions, models or notifications in production code (grep: no GPT/OpenAI literals, no lorem, no `href="#"`, `Math.random` only in `utils.uuid`). The notification list holds only toasts and upload errors recorded this page session. The file-type copy "PDF, DOCX, TXT, images and more" is true against backend `SUPPORTED_EXTENSIONS` (pdf docx pptx odt rtf csv xlsx xls txt md rst html htm json + images). "Multiple AI Models" and "Web Search" are backed by `/models` and Tavily. The top-bar avatar is a non-interactive `role=img` (not a dead control). Findings: M1, m3.
- **Security.** Hostile userId, session title, id, mode, model, doc filename and file_type, toast and notification text (`"><img onerror><svg onload>`, quote-breaking attribute) gave `window.__pwn` undefined, no injected img/svg or handler attributes. Session titles go through `escapeHtml` into double-quoted attributes. Notifications use `textContent` only. Model labels render as literal text. No new external origins.

## 5. Checks run
- `find frontend/js -name '*.js' -exec node --check {} \;` exit 0
- `venv/bin/pytest` exit 0: **291 passed, 1 skipped (xlwt missing), 6 deselected (live), 4 xfailed** (all mine, QA-05)
- `venv/bin/ruff check backend tests --select F,E9,B --ignore B008,B905,B011` "All checks passed!"
- New guard file `tests/test_frontend_reference.py`: 5 pass, 4 xfail (documented above). No existing test changed.
- Scratch scripts in `.../scratchpad/scripts` (h.js, t2-t6.js). Servers and browsers I started are closed.

## 6. Not verified
Real backend streaming, upload and URL paths; citation modal and drag-drop in this pass; 1366x650 and 200% zoom (not re-run, covered in qa-04 for the earlier layout); touch and screen-reader behaviour; the Tab-order walk of the whole page; the dark 1024 and 768 fidelity screens (viewed light only); regenerate availability after a Stopped answer (not checked).
