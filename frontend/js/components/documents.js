/* ============================================================
   components/documents.js — DocsView: the document list (Documents workspace +
   My Documents panel) and the panel dialog itself.
   View only: it renders what the controller gives it and reports clicks back
   through callbacks. It never fetches, so a document is only ever shown after
   the server returned it.
     DocsView.render(documents, onDelete)   documents = GET /documents rows
     DocsView.setStatus("loading" | "error" | "ready", { message, onRetry })
     DocsView.open() / DocsView.close()
   Every string from the server is written with textContent.
   ============================================================ */

window.DocsView = (function () {

  const $ = (id) => document.getElementById(id);

  const TYPE_ICON = {
    pdf: "file-text", docx: "file-text", odt: "file-text", rtf: "file-text", txt: "file-text", md: "file-text", rst: "file-text",
    log: "file-text", github: "github",
    pptx: "presentation", csv: "file-spreadsheet", xlsx: "file-spreadsheet", xls: "file-spreadsheet",
    json: "file-code", html: "file-code", htm: "file-code", url: "globe",
    png: "file-image", jpg: "file-image", jpeg: "file-image", webp: "file-image", bmp: "file-image",
    tif: "file-image", tiff: "file-image", gif: "file-image",
  };

  let _docs = [];
  let _onDelete = null;
  let _status = { state: "ready" };

  function _el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function _metaText(doc) {
    const bits = [];
    if (doc.file_type) bits.push(String(doc.file_type).toUpperCase());
    const chunks = Number(doc.chunk_count);
    if (Number.isFinite(chunks) && chunks >= 0) bits.push(`${chunks} ${chunks === 1 ? "chunk" : "chunks"}`);
    const when = Utils.parseDate(doc.indexed_at);
    if (when) {
      const sameYear = when.getFullYear() === new Date().getFullYear();
      bits.push("Added " + when.toLocaleDateString([], sameYear ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric" }));
    }
    return bits.join(" · ");
  }

  function _row(doc) {
    const filename = String(doc.filename == null ? "" : doc.filename);
    const row = _el("div", "doc-row");
    const icon = _el("span", "doc-row__icon");
    icon.setAttribute("aria-hidden", "true");
    icon.appendChild(Icons.el(TYPE_ICON[doc.file_type] || "file", { size: 20 }));
    const main = _el("span", "doc-row__main");
    const name = _el("span", "doc-row__name", filename);
    name.title = filename;                       // the full name stays reachable when truncated
    main.appendChild(name);
    main.appendChild(_el("span", "doc-row__meta", _metaText(doc)));
    row.appendChild(icon);
    row.appendChild(main);
    if (_onDelete) {
      const del = _el("button", "doc-row__del");
      del.type = "button";
      del.title = "Remove document";
      del.setAttribute("aria-label", `Remove ${filename}`);
      del.appendChild(Icons.el("trash-2", { size: 16 }));
      del.addEventListener("click", (e) => { e.stopPropagation(); _onDelete(doc); });
      row.appendChild(del);
    }
    return row;
  }

  function _state(kind, iconName, title, text, action) {
    const box = _el("div", "list-state" + (kind === "error" ? " list-state--error" : ""));
    const icon = Icons.el(iconName, { size: 24, cls: "list-state__icon" });
    box.appendChild(icon);
    box.appendChild(_el("div", "list-state__title", title));
    if (text) box.appendChild(_el("div", "", text));
    if (action) {
      const b = _el("button", "btn btn--secondary btn--sm", action.label);
      b.type = "button";
      b.addEventListener("click", action.run);
      box.appendChild(b);
    }
    return box;
  }

  function _skeleton() {
    const wrap = document.createDocumentFragment();
    for (let i = 0; i < 3; i++) {
      const row = _el("div", "doc-skeleton");
      row.appendChild(_el("span", "skeleton doc-skeleton__icon"));
      const lines = _el("span", "doc-skeleton__lines");
      lines.appendChild(_el("div", "skeleton skeleton-line skeleton-line--70"));
      lines.appendChild(_el("div", "skeleton skeleton-line skeleton-line--60"));
      row.appendChild(lines);
      wrap.appendChild(row);
    }
    return wrap;
  }

  function _fill(container) {
    const full = container.dataset.docList === "full";
    const query = full ? (($("docSearch") && $("docSearch").value) || "").trim().toLowerCase() : "";
    container.textContent = "";
    container.setAttribute("aria-busy", _status.state === "loading" ? "true" : "false");

    if (_status.state === "loading") { container.appendChild(_skeleton()); return; }
    if (_status.state === "error") {
      container.appendChild(_state("error", "circle-alert", "Couldn’t load your documents", _status.message || "", _status.onRetry ? { label: "Try again", run: _status.onRetry } : null));
      return;
    }
    if (!_docs.length) {
      container.appendChild(_state("empty", "folder", "No documents yet",
        "Upload a file or add a web page, then ask questions about it."));
      return;
    }
    const shown = query ? _docs.filter((d) => String(d.filename || "").toLowerCase().includes(query)) : _docs;
    if (!shown.length) {
      container.appendChild(_state("empty", "search", "No matches", `No document name contains “${$("docSearch").value.trim()}”.`));
      return;
    }
    shown.forEach((d) => container.appendChild(_row(d)));
  }

  function _renderAll() {
    document.querySelectorAll("[data-doc-list]").forEach(_fill);
  }

  function _counts() {
    const n = _docs.length;
    const ready = _status.state === "ready";
    const pill = $("wsDocCount");
    if (pill) { pill.textContent = String(n); pill.hidden = !(ready && n); }
    const sub = $("docsDialogSub");
    if (sub) sub.textContent = ready && n ? `${n} ${n === 1 ? "document" : "documents"} indexed.` : "Files and web pages you have indexed.";
    const card = $("docsCardSub");
    if (card) card.textContent = ready && n ? `${n} ${n === 1 ? "file" : "files"}` : "Manage your files";
    if (window.Workspaces) Workspaces.setDocCount(ready ? n : 0);
  }

  /** @param {Array} documents GET /documents rows  @param {(doc)=>void} onDelete */
  function render(documents, onDelete) {
    _docs = Array.isArray(documents) ? documents : [];
    _onDelete = typeof onDelete === "function" ? onDelete : null;
    _status = { state: "ready" };
    _renderAll();
    _counts();
  }

  function setStatus(state, opts) {
    _status = Object.assign({ state: state === "loading" || state === "error" ? state : "ready" }, opts || {});
    _renderAll();
    _counts();
  }

  function open() {
    const dlg = $("docsDialog");
    if (!dlg) return;
    const search = $("docSearch");
    if (search && !dlg.open) { search.value = ""; _renderAll(); }
    Dialog.open(dlg, { initialFocus: search });
  }

  function close() { Dialog.close($("docsDialog")); }

  function init() {
    const dlg = $("docsDialog");
    if (!dlg) return;
    const search = $("docSearch");
    if (search) search.addEventListener("input", _renderAll);
    const closeBtn = $("docsDialogClose");
    if (closeBtn) closeBtn.addEventListener("click", close);
    dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
    // These reuse the composer's existing handlers, so there is one upload path
    const up = $("docsUploadBtn");
    if (up) up.addEventListener("click", () => { const a = $("attachBtn"); if (a) a.click(); });
    const url = $("docsAddUrl");
    if (url) url.addEventListener("click", () => { const a = $("addUrlBtn"); if (a) a.click(); });
    _renderAll();
  }

  return { init, render, setStatus, open, close };
})();
