# UI/UX implementation notes — workspace dropdown, workspaces, Settings, motion

Reference: `AI Assistant Pro UI_UX Presentation Board.png` (16-section board) plus the written brief that overrides it
where they differ. This file records what was built, what was deliberately **not** built, and why.

## Where the brief and the board disagree (the brief wins)

| Board | Brief | Built |
|---|---|---|
| Sections 05–07 show the four modes as permanent sidebar rows; the older reference shows a segmented tab bar in the top bar | The mode selector is a **dropdown above New Chat**; the modes must not be permanent rows or a header toggle | One `#modeSelect` menu button above `#newChatBtn`. Rows, header tabs and their CSS/JS are removed. The header shows a read-only workspace label only |
| Section 07: a code block with Python / JavaScript / Java / C++ tabs | No fake editor; do not execute code | The tabs became a real **language preference** that only shapes the text put in the composer (fenced template). Generated code is never run |
| Section 05: "Images" accepted, "Max 20MB" | Guidance must come from the real backend | Types and the size limit come from a new read-only `GET /config`; images are supported by the backend (OCR, best effort) |
| Section 10: Profile, Model, Notifications, Integrations, Privacy, Help, "Show line numbers", "Auto-save" | Only settings that can be persisted reliably; no dead controls | Built: Profile (read-only user ID), Appearance, Model preferences, Notifications, Integrations (live `/health`), Privacy, Help. **Not built:** line numbers (would need a new code-render pipeline) and auto-save (conversations always save, so a toggle would be dead) |

## Stack decisions

* **Kept vanilla HTML/CSS/JS.** There is no `package.json`, React, Tailwind or shadcn/ui in the project. Introducing them
  would be a rewrite, which the brief rules out ("do not replace the application architecture"). The shadcn/ui
  primitives the brief names map onto existing view components: `Menu` (dropdown), `Dialog` (dialogs, drawer-style
  panel), the radio-group helper, tabs in Settings, tooltips.
* **Icons:** Lucide (ISC) inline sprite; new icons were generated from the official package, validated to plain shapes.
* **GSAP: yes, narrowly.** `js/components/motion.js` is the only user: a staged workspace entrance and a quick workspace
  transition. Lazy, same-origin, pinned by SHA-256, never loaded under reduced motion. The previous CSS entrance chain
  (up to ~1.5 s, replayed on every workspace switch) was removed. License note: GreenSock "no charge", not OSI.
* **Three.js: no.** The CSS/SVG orb already delivers the visual; a 3D runtime would add ~600 KB for no task benefit.

## Backend change (the only one)

`GET /config` → `{"upload": {"max_mb": int, "extensions": [..]}}`. Additive, read-only, no secrets, no new behaviour.
Needed because the size limit and accepted types were not discoverable, so client-side validation could only guess.
The UI degrades to "no size pre-check" on an older backend. Tests: `tests/test_api.py` (2).

## Behaviour that follows from the backend (not invented)

* **Research always searches the web** (server rule), so the Web Search switch is shown on and locked there.
* **Documents are per user, not per chat**, so opening My Documents does not switch workspace.
* **No duplicate detection on the server**, so the client asks before uploading a same-named file again.
* **No upload progress events**, so there is no percentage anywhere; the drop zone shows a busy state.
* **No pin / download / open endpoints**, so those controls do not exist. Rename and delete do (confirmed).
* **No accounts**: Profile shows the browser-local user ID and says so.

## Checklist

- [x] Mode dropdown above New Chat (keyboard, Esc, outside click, selected state, drawer-safe)
- [x] Welcome dashboard with four task cards, capabilities row, composer
- [x] Documents / Research / Code workspace panels (real actions only)
- [x] Chat: streaming, sources, copy, regenerate, feedback, error + retry, long code / URLs on mobile
- [x] History: grouping, search, rename, delete (confirm), loading + error states, arrow-key navigation
- [x] My Documents panel: list, filter, delete (confirm), upload, add URL, loading / empty / error
- [x] Settings: 7 sections, every control functional
- [x] Light / Dark / System, no flash; text size and spacing preferences
- [x] Motion: GSAP entrance + transition, reduced-motion aware
- [x] Responsive sweep (9 widths × 2 themes × 9 states) and axe WCAG 2.2 AA scan
- [x] Tests: `test_frontend_shell.py`, `test_frontend_workspaces.py`, `test_frontend_motion.py`, `/config` tests
