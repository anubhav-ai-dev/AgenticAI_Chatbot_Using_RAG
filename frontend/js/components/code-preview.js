/* ============================================================
   components/code-preview.js — sandboxed iframe preview pane
   Only shown in the Code workspace (html[data-workspace="code"]).
   CSS controls Run/Download button visibility; JS manages the pane.

   Public API:
     CodePreview.decorate(wrapper)          — called by Utils.highlightCode after wrapping
     CodePreview.sync(mode)                 — called by UI.setActiveMode
     CodePreview.open(source, lang, opener) — programmatic open
     CodePreview.close()                    — called by Chat.openArtifact
   ============================================================ */
window.CodePreview = (function () {

  const PREVIEWABLE = new Set(["html", "htm", "svg"]);
  let _paneOpen = false;
  let _opener = null;
  let _wired = false;
  const _narrow = window.matchMedia("(max-width: 767px)");

  // ── Helpers ──────────────────────────────────────────────

  function _el(id) { return document.getElementById(id); }

  function _lang(wrapper) {
    const span = wrapper && wrapper.querySelector(".code-block__lang");
    return span ? span.textContent.trim().toLowerCase() : "";
  }

  function _source(wrapper) {
    const code = wrapper && wrapper.querySelector("pre code");
    return code ? (code.innerText || code.textContent || "") : "";
  }

  // ── Decorate: add Run + Download buttons ──────────────────

  /**
   * Tag a code-block wrapper as visited, then add Download and Run buttons when
   * the fence is previewable (html/htm/svg) and lives inside #chatArea.
   * Idempotent: repeated calls on the same wrapper are a no-op.
   * CSS hides the buttons outside the Code workspace.
   */
  function decorate(wrapper) {
    if (!wrapper || wrapper.dataset.cpDone) return;
    wrapper.dataset.cpDone = "1";

    const area = _el("chatArea");
    if (!area || !area.contains(wrapper)) return;  // skip artifact-pane code blocks
    if (!PREVIEWABLE.has(_lang(wrapper))) return;  // non-previewable fences stay copy-only

    _addButtons(wrapper);
  }

  function _addButtons(wrapper) {
    const header = wrapper.querySelector(".code-block__header");
    if (!header || header.querySelector(".code-block__run")) return;  // already wired

    // ── Download ──
    const dlBtn = document.createElement("button");
    dlBtn.className = "code-block__download";
    dlBtn.type = "button";
    dlBtn.setAttribute("aria-label", "Download code");
    dlBtn.innerHTML = Icons.html("download", { size: 16 }) + "<span>Download</span>";
    dlBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const src = _source(wrapper);
      const { filename } = Utils.buildPreviewDoc(src, _lang(wrapper));
      _downloadBlob(src, filename);
    });

    // ── Run ──
    const runBtn = document.createElement("button");
    runBtn.className = "code-block__run";
    runBtn.type = "button";
    runBtn.setAttribute("aria-label", "Run preview");
    // Use icon--filled so the play triangle is solid (matches DeepSeek appearance)
    runBtn.innerHTML =
      '<svg class="icon icon--16 icon--filled" aria-hidden="true"><use href="#i-play"/></svg>' +
      "<span>Run</span>";
    runBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      _launch(_source(wrapper), _lang(wrapper), runBtn);
    });

    header.appendChild(dlBtn);
    header.appendChild(runBtn);
  }

  function _downloadBlob(text, filename) {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  // ── Mode sync ─────────────────────────────────────────────

  /**
   * Called by UI.setActiveMode each time the workspace changes.
   * Run/Download visibility is handled by CSS:
   *   html[data-workspace="code"] .code-block__run { display: inline-flex; }
   * We only need JS to close the pane when leaving Code.
   */
  function sync(mode) {
    if (mode !== "code") close();
  }

  // ── Pane layer sync ───────────────────────────────────────

  function _syncLayer() {
    const pane = _el("codePreview");
    if (!pane || !_paneOpen) return;
    if (_narrow.matches) {
      pane.setAttribute("role", "dialog");
      pane.setAttribute("aria-modal", "true");
    } else {
      pane.setAttribute("role", "complementary");
      pane.removeAttribute("aria-modal");
    }
  }

  // ── Resize (mirrors artifact-pane resize logic) ───────────

  const _W_KEY = "codePreviewWidth";
  const _W_MIN = 280;

  function _wMax() { return Math.max(_W_MIN, Math.round(window.innerWidth * 0.8)); }
  function _clampW(px) { return Math.max(_W_MIN, Math.min(_wMax(), Math.round(px))); }

  function _applyW(px, persist) {
    const pane = _el("codePreview");
    const handle = _el("codePreviewResize");
    if (!pane) return;
    const w = _clampW(px);
    pane.style.setProperty("--code-preview-w", w + "px");
    if (handle) {
      handle.setAttribute("aria-valuenow", String(w));
      handle.setAttribute("aria-valuemin", String(_W_MIN));
      handle.setAttribute("aria-valuemax", String(_wMax()));
    }
    if (persist) { try { localStorage.setItem(_W_KEY, String(w)); } catch (_) {} }
  }

  function _resetW() {
    const pane = _el("codePreview");
    if (pane) pane.style.removeProperty("--code-preview-w");
    try { localStorage.removeItem(_W_KEY); } catch (_) {}
    const handle = _el("codePreviewResize");
    if (handle && pane) {
      const w = Math.round(pane.getBoundingClientRect().width);
      handle.setAttribute("aria-valuenow", String(w));
    }
  }

  function _restoreW() {
    try {
      const v = parseInt(localStorage.getItem(_W_KEY), 10);
      if (Number.isFinite(v)) _applyW(v, false);
    } catch (_) {}
  }

  function _wireResize(handle) {
    if (!handle || handle.dataset.wired) return;
    handle.dataset.wired = "1";
    handle.setAttribute("aria-valuemin", String(_W_MIN));
    handle.addEventListener("pointerdown", (e) => {
      if (_narrow.matches || e.button) return;
      const pane = _el("codePreview");
      if (!pane) return;
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startW = pane.getBoundingClientRect().width;
      const move = (ev) => { _applyW(startW + (startX - ev.clientX), false); };
      const up = (ev) => {
        handle.releasePointerCapture(ev.pointerId);
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        _applyW(pane.getBoundingClientRect().width, true);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", _resetW);
    handle.addEventListener("keydown", (e) => {
      if (_narrow.matches) return;
      const pane = _el("codePreview");
      if (!pane) return;
      const w = pane.getBoundingClientRect().width;
      if (e.key === "ArrowLeft")  { e.preventDefault(); _applyW(w + 16, true); }
      else if (e.key === "ArrowRight") { e.preventDefault(); _applyW(w - 16, true); }
      else if (e.key === "Home")  { e.preventDefault(); _resetW(); }
    });
  }

  // ── Wire pane (once) ─────────────────────────────────────

  function _wirePane() {
    if (_wired) return;
    const pane      = _el("codePreview");
    const closeBtn  = _el("codePreviewClose");
    const scrim     = _el("codePreviewScrim");
    const resize    = _el("codePreviewResize");
    if (!pane || !closeBtn) return;
    _wired = true;
    closeBtn.addEventListener("click", close);
    if (scrim) scrim.addEventListener("click", close);
    document.addEventListener("keydown", (e) => {
      if (_paneOpen && e.key === "Escape") { e.stopPropagation(); close(); }
    }, true);
    if (_narrow.addEventListener) _narrow.addEventListener("change", _syncLayer);
    _wireResize(resize);
  }

  // ── Launch (open the pane) ────────────────────────────────

  function _launch(source, lang, openerBtn) {
    _wirePane();

    // Close the document preview if it happens to be open
    if (window.Chat && typeof Chat.closeArtifact === "function") Chat.closeArtifact();

    const pane    = _el("codePreview");
    const frame   = _el("codePreviewFrame");
    const scrim   = _el("codePreviewScrim");
    const titleEl = _el("codePreviewTitle");
    if (!pane || !frame) return;

    if (!window.Utils || typeof Utils.buildPreviewDoc !== "function") return;
    const { title, html } = Utils.buildPreviewDoc(source, lang);

    if (titleEl) titleEl.textContent = title;

    // Clear first so the iframe fully resets before new content arrives
    frame.removeAttribute("srcdoc");
    frame.srcdoc = html;

    _opener = openerBtn || null;
    _paneOpen = true;
    pane.classList.add("is-open");
    pane.setAttribute("aria-hidden", "false");
    pane.inert = false;
    if (scrim) scrim.classList.add("is-open");
    _syncLayer();
    _restoreW();

    const closeBtn = _el("codePreviewClose");
    if (closeBtn) closeBtn.focus();
  }

  // ── Close ─────────────────────────────────────────────────

  function close() {
    if (!_paneOpen) return;
    const pane  = _el("codePreview");
    const frame = _el("codePreviewFrame");
    const scrim = _el("codePreviewScrim");

    _paneOpen = false;

    if (pane) {
      pane.classList.remove("is-open");
      pane.setAttribute("aria-hidden", "true");
      pane.setAttribute("role", "complementary");
      pane.removeAttribute("aria-modal");
      pane.inert = true;
    }
    // Remove srcdoc to stop any scripts still running in the sandboxed page
    if (frame) frame.removeAttribute("srcdoc");
    if (scrim) scrim.classList.remove("is-open");

    if (_opener && document.contains(_opener) && typeof _opener.focus === "function") {
      _opener.focus();
    }
    _opener = null;
  }

  return { decorate, sync, open: _launch, close };
})();
