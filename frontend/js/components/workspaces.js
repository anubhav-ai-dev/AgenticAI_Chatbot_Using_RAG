/* ============================================================
   components/workspaces.js — the Documents / Research / Code panels
   (the empty state of each workspace, markup in index.html).
   View only. Everything here puts TEXT IN THE COMPOSER and focuses it:
   nothing is ever sent from a panel, and no request is made.
     Workspaces.init()
     Workspaces.setActive(mode)        composer hint + web-search switch for the workspace
     Workspaces.setDocCount(n)         enables / explains the document actions
     Workspaces.setUploadBusy(busy, name)
   ============================================================ */

window.Workspaces = (function () {

  const $ = (id) => document.getElementById(id);

  const PLACEHOLDER = {
    general:  "Ask anything...",
    document: "Ask anything about your documents...",
    research: "Ask a research question...",
    code:     "Describe what to build, or paste code to explain or debug...",
  };

  const fenced = (lead, tail) => `${lead}\n\n\`\`\`\n\n\`\`\`${tail || ""}`;
  const TEMPLATE = {
    write:    () => "Write a function that ",
    explain:  () => fenced("Explain what this code does, step by step:"),
    fix:      () => fenced("Find and fix the bug in this code, and explain what was wrong:", "\n\nError message (if any):\n"),
    optimize: () => fenced("Optimize this code for readability and performance, and explain your changes:"),
    tests:    () => fenced("Write unit tests for this code:"),
    debug:    () => fenced("Debug these logs. Identify the problem, root cause, why it happens, whether it has shown up before, and the fix. Explain in simple terms unless I ask for the technical version."),
  };

  let _docCount = 0;

  /** Put text in the composer (the controller auto-resizes on "input"), focus it, place the caret. Never sends. */
  function fill(text, { select = "", fence = "" } = {}) {
    const input = $("chatInput");
    if (!input) return;
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();
    let start = text.length, end = text.length;
    if (select) {
      const i = text.indexOf(select);
      if (i !== -1) { start = i; end = i + select.length; }       // the placeholder is selected: typing replaces it
    } else if (typeof fence === "string") {
      const marker = "```" + fence + "\n";
      const i = text.indexOf(marker);
      if (i !== -1) start = end = i + marker.length;              // caret on the empty line inside the code fence
    }
    try { input.setSelectionRange(start, end); } catch (_) { /* not selectable */ }
  }

  function _onClick(e) {
    const t = e.target.closest("[data-fill], [data-template]");
    const root = $("welcomeScreen");
    if (!t || !root || !root.contains(t)) return;
    if (t.hasAttribute("data-needs-docs") && t.getAttribute("aria-disabled") === "true") {
      if (window.UI) UI.setLiveStatus("Upload a document first to use this action.");
      const dz = $("docDropzone");
      if (dz) dz.focus();
      return;
    }
    if (t.dataset.template) {
      const make = TEMPLATE[t.dataset.template];
      if (make) fill(make(), { fence: "" });
      return;
    }
    fill(t.dataset.fill || "", { select: t.dataset.select || "" });
  }

  /** Research always searches the web on the server, so the switch shows "on" and cannot be turned off there. */
  function _syncWebToggle(mode) {
    const t = $("searchToggle");
    if (!t) return;
    const locked = mode === "research";
    if (locked && t.getAttribute("aria-disabled") !== "true") {
      t.dataset.userPressed = t.getAttribute("aria-pressed") || "true";
      t.setAttribute("aria-disabled", "true");
      t.setAttribute("aria-pressed", "true");
      t.title = "Web search is always on in Research";
    } else if (!locked && t.getAttribute("aria-disabled") === "true") {
      t.removeAttribute("aria-disabled");
      t.removeAttribute("title");
      t.setAttribute("aria-pressed", t.dataset.userPressed || "true");
      delete t.dataset.userPressed;
    }
  }

  function setActive(mode) {
    const input = $("chatInput");
    if (input) input.placeholder = PLACEHOLDER[mode] || PLACEHOLDER.general;
    _syncWebToggle(mode);
  }

  /** Document actions need at least one indexed document; say why they are unavailable instead of hiding them. */
  function setDocCount(n) {
    _docCount = Number(n) || 0;
    const has = _docCount > 0;
    document.querySelectorAll("[data-needs-docs]").forEach((c) => {
      if (has) c.removeAttribute("aria-disabled"); else c.setAttribute("aria-disabled", "true");
      if (!has) c.setAttribute("aria-describedby", "docActionsNote"); else c.removeAttribute("aria-describedby");
    });
    const note = $("docActionsNote");
    if (note) note.textContent = has
      ? "These fill the message box. Review or edit the text, then press Send."
      : "Upload a document to enable these actions.";
  }

  /** The drop zone shows an in-progress upload and refuses a second one. */
  function setUploadBusy(busy, name) {
    const dz = $("docDropzone");
    if (!dz) return;
    dz.setAttribute("aria-busy", busy ? "true" : "false");
    const title = $("docDropTitle");
    if (title) title.textContent = busy ? `Uploading ${name || "file"}…` : "Drop your files here or click to upload";
  }

  let _hintBase = "";

  function setLimits(maxMb) {
    const hint = $("docDropHint");
    if (!hint) return;
    if (!_hintBase) _hintBase = hint.textContent;
    const n = Number(maxMb);
    hint.textContent = _hintBase + (Number.isFinite(n) && n > 0 ? ` Up to ${n} MB per file.` : "");
  }

  function _wireDropzone() {
    const dz = $("docDropzone");
    if (!dz) return;
    dz.addEventListener("click", () => {
      if (dz.getAttribute("aria-busy") === "true") { if (window.UI) UI.setLiveStatus("An upload is already in progress."); return; }
      const attach = $("attachBtn");
      if (attach) attach.click();                                  // the composer's existing upload path
    });
    // Highlight only. The drop itself is handled once, by the page-level handler (UI.initDropzone)
    let depth = 0;
    dz.addEventListener("dragenter", () => { depth++; dz.classList.add("is-over"); });
    dz.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) dz.classList.remove("is-over"); });
    dz.addEventListener("drop", () => { depth = 0; dz.classList.remove("is-over"); });
  }

  function init() {
    const root = $("welcomeScreen");
    if (root) root.addEventListener("click", _onClick);

    _wireDropzone();

    const addUrl = $("wsAddUrl");
    if (addUrl) addUrl.addEventListener("click", () => { const b = $("addUrlBtn"); if (b) b.click(); });
    const manage = $("wsManageDocs");
    if (manage) manage.addEventListener("click", () => { if (window.DocsView) DocsView.open(); });

    setDocCount(0);
  }

  return { init, setActive, setDocCount, setUploadBusy, setLimits, fill };
})();
