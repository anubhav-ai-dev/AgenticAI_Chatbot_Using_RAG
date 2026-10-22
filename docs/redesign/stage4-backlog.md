# Stage 4 backlog (controller work: app.js, api.js, js/workflows/)

Recorded decisions. Do not start before Stage 3 is reviewed.

## S4-1 Welcome cards must not auto-send (approved requirement)
- Today `app.js:137` sends `data-prompt` on click. Required end state: clicking or pressing Enter/Space on a card switches to the card's mode, fills the composer and focuses it, and does NOT submit a request.
- Regression tests required: prove click and Enter on every card issue no `createSession` / `chatStream` call (stubbed, no real LLM calls), and that mode, composer value and focus are set.
- Source: Stage 2 QA finding M1. The view currently clicks the matching `.mode-tab` first (ui.js) and relies on app.js to send; adjust the view/controller seam together.

## Other items carried from QA
- Focus lost while streaming (`_setLoading` disables textarea, hides Send): keep a reachable Stop control and restore focus.
- C1 UI.confirm, C2 onRename, C7 models retry, C8 icon names, C9 searchToggle color, C14 `.rst` (controller `ALLOWED` and view `accept` together).
- Input preservation on failure; regenerate must not drop the DOM answer before success.
