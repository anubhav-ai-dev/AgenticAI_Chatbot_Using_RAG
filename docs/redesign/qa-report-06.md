# QA report 06 - Re-verification after QA-05 fixes

## 1. Verdict: PASS
All QA-05 findings (M1, m1-m4, n1-n4) are closed. No Blocker/Major/Minor open. Evidence: puppeteer-core 23.11.1 + axe-core 4.10.2 in a scratch dir (stubbed fetch, no live LLM calls, servers/browsers closed).

## 2. QA-05 closure
- M1 privacy claim: now "Safe Uploads / File type and size are checked" (index.html:467). True: backend `max_upload_mb` 413 + SUPPORTED_EXTENSIONS. The former xfail guard passes.
- m1 outlines: `--border-ui` #8790AF on #FFF = 3.16:1, #56669E on #10173A = 3.14:1 (python3 -I). Closed.
- m2 card borders: exemption documented; former xfail test now passes (suite 295 passed, 0 xfail).
- m3 Upload Files CTA: real puppeteer `waitForFileChooser`: click, Enter and Space on the document card each opened the chooser (3/3) with 0 non-GET requests before a file was chosen. After accepting a file: `POST /upload` then `GET /documents`. The 3 other cards (3 methods x 3 = 9 runs): no chooser, 0 non-GET, composer filled, mode stored, no user message.
- m4 guards: new file passes. n1: Previous 7 Days = days 2-7 (1 = Yesterday), 8-29 = Previous 30 Days, 30+ Earlier. n2: mode tabs use `aria-pressed`, 0 `aria-current` left. n3: reduced motion, hover transform `none`, 0 running animations (8 otherwise). n4: at 1024x768 all 4 cards sit above the composer (bottom 599 < composer 636).
- Guard-test edit (`tests/test_frontend_guards.py:654`, `.mode-tab[aria-pressed="true"]`): legitimate. It is a selector inside the in-browser harness string, tracking the real markup change (the tabs now use aria-pressed). It asserts the same property (active mode == card mode); no assertion was removed or loosened.

## 3. Regression (QA-04/05 suite re-run)
Toast top-right and does not cover Stop (1280x800 and 390x844, hit-test plus real click stops); welcome cards 12/12 activations (0 sends); streaming keeps the textarea enabled with Stop shown; Stopped notice; failed send keeps draft and no user bubble; failed regenerate keeps the old answer; regenerate payload `regenerate:true`; drawer at 390 and 768 moves focus in, Escape closes and returns focus to `#sidebarToggle`; bell, filter, gear, model listbox, web toggle, top tabs (PATCH mode) all as before. My Documents: mode becomes document, tab aria-pressed follows, `GET /documents`, list shows the real row, second click closes. A forced `/documents` 500 raises an error notification (bell "1 unread").

## 4. A11y / layout
axe wcag2a/2aa/21aa: 0 violations in 32 runs (1440x900, 1280x800, 1024x768, 768x1024, 430x932, 390x844, 375x812, 320x700 x light/dark x welcome/chat) and 14 runs with overlays open. No horizontal overflow, composer in view in all. 1366x650 and 1280x720: no overflow, cards and composer both visible (badge and subtitle hidden as designed). Focus ring 2 px solid in both themes.

## 5. Fidelity vs reference (screens in .../scratchpad/shots6/)
Viewed: desktop light and dark 1440x1024, tablet 768 and 1024 dark, mobile 390 light and dark, 1366x650. Remaining differences (all cosmetic or honest-data choices, no finding): triangle logo vs sparkles tile; orb smaller than the reference and the tablet 1024 view omits the "AI Assistant Pro" badge and subtitle; model tile shows a letter, not a provider logo; sidebar 256 px vs about 273; mobile has 4 tabs vs the reference's 3 and the chip is icon-only; user row shows real userId; tablet 768 keeps a sidebar toggle.

## 6. Observations (not findings)
- Cancelling the file chooser leaves "Summarize the uploaded document" in the composer (editable, never sent). The card aria-label still says "Switches to Documents and fills the message box" and does not mention the chooser; a clearer name would be "Opens file picker". Nit, owner frontend-developer.

## 7. Checks run
- `find frontend/js -name '*.js' -exec node --check {} \;` exit 0
- `venv/bin/pytest` exit 0: 295 passed, 1 skipped (xlwt missing), 6 deselected (live), 0 xfailed
- `venv/bin/pytest tests/test_frontend_*.py` all pass
- `venv/bin/ruff check backend tests --select F,E9,B --ignore B008,B905,B011`: All checks passed
- Scripts: scratchpad/scripts/v6a.js (cards, chooser, My Documents), v6b.js, v6s.js (screens), t2-t5.js (regressions, axe).

## 8. Not verified
Real backend streaming/upload; citation modal and drag-drop; 200% zoom; touch and screen-reader behaviour; full-page Tab-order walk; real OS file chooser content (only accept/cancel via puppeteer).
