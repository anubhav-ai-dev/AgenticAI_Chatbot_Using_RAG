/* ============================================================
   ui.js — Sidebar, modals, drag-drop, theme toggle, toasts,
            session list rendering, document list rendering
   ============================================================ */

window.UI = (function () {

  // ── Theme ─────────────────────────────────────────────────

  function initTheme() {
    const saved = localStorage.getItem("theme") || "dark";
    setTheme(saved);
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
    const btn = document.getElementById("themeToggle");
    if (btn) btn.textContent = theme === "dark" ? "🌙" : "☀️";

    // Sync hljs theme
    const link = document.getElementById("hljs-theme");
    if (link) {
      link.href = theme === "dark"
        ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"
        : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css";
    }
  }

  function toggleTheme() {
    const current = document.documentElement.dataset.theme || "dark";
    setTheme(current === "dark" ? "light" : "dark");
  }

  // ── Sidebar ───────────────────────────────────────────────

  function openSidebar() {
    const sb = document.getElementById("sidebar");
    const bd = document.getElementById("sidebarBackdrop");
    if (!sb) return;
    sb.classList.add("mobile-open");
    if (bd) bd.style.display = "block";
    const toggle = document.getElementById("sidebarToggle");
    if (toggle) toggle.setAttribute("aria-expanded", "true");
  }

  function closeSidebar() {
    const sb = document.getElementById("sidebar");
    const bd = document.getElementById("sidebarBackdrop");
    if (!sb) return;
    sb.classList.remove("mobile-open");
    if (bd) bd.style.display = "none";
    const toggle = document.getElementById("sidebarToggle");
    if (toggle) toggle.setAttribute("aria-expanded", "false");
  }

  function toggleSidebar() {
    const sb = document.getElementById("sidebar");
    if (!sb) return;
    sb.classList.contains("mobile-open") ? closeSidebar() : openSidebar();
  }

  // ── Session list ──────────────────────────────────────────

  function renderSessionList(sessions, activeId, onSelect, onDelete) {
    const list = document.getElementById("sessionList");
    if (!list) return;

    if (!sessions || sessions.length === 0) {
      list.innerHTML = `<div class="text-xs text-muted" style="padding:10px 8px">No chats yet</div>`;
      return;
    }

    // Group by date
    const groups = {};
    sessions.forEach((s) => {
      const g = Utils.formatDateGroup(s.updated_at);
      if (!groups[g]) groups[g] = [];
      groups[g].push(s);
    });

    const order = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Earlier"];
    let html = "";

    order.forEach((groupName) => {
      const items = groups[groupName];
      if (!items || items.length === 0) return;
      html += `<div class="session-group-label">${groupName}</div>`;
      items.forEach((s) => {
        const isActive = s.id === activeId ? "active" : "";
        const modeIcons = { general: "💬", document: "📄", research: "🌐", code: "🧮" };
        const icon = modeIcons[s.mode] || "💬";
        const updated = Utils.formatTime(s.updated_at);
        html += `
          <div class="session-item ${isActive}" data-session-id="${s.id}" title="${Utils.escapeHtml(s.title)}">
            <span class="session-item__icon">${icon}</span>
            <div class="session-item__text">
              <div class="session-item__title truncate">${Utils.escapeHtml(s.title)}</div>
              <div class="session-item__meta">${s.model.split("/").pop().substring(0,18)} · ${updated}</div>
            </div>
            <div class="session-item__actions">
              <button class="btn btn--icon-sm delete-session-btn" data-session-id="${s.id}"
                      data-tooltip="Delete" aria-label="Delete chat" title="Delete">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
              </button>
            </div>
          </div>
        `;
      });
    });

    list.innerHTML = html;

    // Bind clicks
    list.querySelectorAll(".session-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".delete-session-btn")) return;
        onSelect(el.dataset.sessionId);
        // Auto-close sidebar on mobile
        if (window.innerWidth <= 768) closeSidebar();
      });
    });

    list.querySelectorAll(".delete-session-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        onDelete(btn.dataset.sessionId);
      });
    });
  }

  // ── Document list ─────────────────────────────────────────

  function renderDocList(documents) {
    const el = document.getElementById("docList");
    if (!el) return;

    if (!documents || documents.length === 0) {
      el.innerHTML = `<div class="text-xs text-muted" style="padding:4px 0">No documents indexed yet</div>`;
      return;
    }

    const typeIcons = { pdf: "📄", docx: "📝", csv: "📊", xlsx: "📊", txt: "📃", url: "🌐" };
    el.innerHTML = documents.map((d) => {
      const icon = typeIcons[d.file_type] || "📎";
      const name = d.filename.length > 22 ? d.filename.substring(0, 20) + "…" : d.filename;
      return `
        <div class="doc-item" title="${Utils.escapeHtml(d.filename)}">
          <span class="doc-item__icon">${icon}</span>
          <span class="doc-item__name">${Utils.escapeHtml(name)}</span>
          <span class="doc-item__type">${(d.file_type || "").toUpperCase()}</span>
        </div>
      `;
    }).join("");
  }

  // ── Upload toast ──────────────────────────────────────────

  function showUploadToast(filename) {
    const toast = document.getElementById("uploadToast");
    const nameEl = document.getElementById("uploadToastName");
    const bar = document.getElementById("uploadProgress");
    if (!toast) return;
    if (nameEl) nameEl.textContent = filename;
    if (bar) {
      bar.style.width = "0%";
      bar.style.animation = "none";
      requestAnimationFrame(() => {
        bar.style.animation = "progressFill 2.5s ease both";
      });
    }
    toast.style.display = "block";
  }

  function hideUploadToast() {
    const toast = document.getElementById("uploadToast");
    if (toast) toast.style.display = "none";
  }

  function showToast(message, type = "info") {
    const colors = { info: "var(--accent-blue)", success: "var(--accent-green)", error: "var(--accent-red)" };
    const icons  = { info: "ℹ️", success: "✅", error: "❌" };

    const el = document.createElement("div");
    el.className = "upload-toast animate-slide-right";
    el.style.minWidth = "220px";
    el.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:500;color:${colors[type]}">
        ${icons[type]} ${Utils.escapeHtml(message)}
      </div>
    `;
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .4s"; }, 2800);
    setTimeout(() => el.remove(), 3200);
  }

  // ── Citation modal ────────────────────────────────────────

  function showCitationModal(citation) {
    if (!citation) return;
    const modal = document.getElementById("citationModal");
    const sourceEl = document.getElementById("citationSource");
    const snippetEl = document.getElementById("citationSnippet");
    if (!modal) return;

    const page = citation.page ? ` · Page ${citation.page}` : "";
    if (sourceEl) sourceEl.innerHTML = `📌 <strong>${Utils.escapeHtml(citation.source)}${page}</strong>`;
    if (snippetEl) snippetEl.textContent = citation.snippet || "No preview available.";
    modal.style.display = "flex";
  }

  function hideCitationModal() {
    const modal = document.getElementById("citationModal");
    if (modal) modal.style.display = "none";
  }

  // ── URL modal ─────────────────────────────────────────────

  function showUrlModal() {
    const modal = document.getElementById("urlModal");
    if (modal) {
      modal.style.display = "flex";
      const inp = document.getElementById("urlInput");
      if (inp) { inp.value = ""; inp.focus(); }
    }
  }

  function hideUrlModal() {
    const modal = document.getElementById("urlModal");
    if (modal) modal.style.display = "none";
  }

  // ── Drag-and-drop zone ────────────────────────────────────

  function initDropzone(onFileDrop) {
    const overlay = document.getElementById("dropzoneOverlay");
    let _dragCount = 0;

    document.addEventListener("dragenter", (e) => {
      if (e.dataTransfer && e.dataTransfer.items.length > 0) {
        _dragCount++;
        if (overlay) overlay.classList.add("active");
      }
    });

    document.addEventListener("dragleave", () => {
      _dragCount--;
      if (_dragCount <= 0) {
        _dragCount = 0;
        if (overlay) overlay.classList.remove("active");
      }
    });

    document.addEventListener("dragover", (e) => e.preventDefault());

    document.addEventListener("drop", (e) => {
      e.preventDefault();
      _dragCount = 0;
      if (overlay) overlay.classList.remove("active");
      const files = Array.from(e.dataTransfer?.files || []);
      if (files.length > 0) onFileDrop(files[0]);
    });
  }

  // ── Active mode tab ───────────────────────────────────────

  function setActiveMode(mode) {
    document.querySelectorAll(".mode-tab").forEach((tab) => {
      const isActive = tab.dataset.mode === mode;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });
  }

  // ── User ID display ───────────────────────────────────────

  function setUserIdDisplay(userId) {
    const el = document.getElementById("userIdDisplay");
    if (el) el.textContent = `user: ${userId || "default"}`;
  }

  return {
    initTheme,
    setTheme,
    toggleTheme,
    openSidebar,
    closeSidebar,
    toggleSidebar,
    renderSessionList,
    renderDocList,
    showUploadToast,
    hideUploadToast,
    showToast,
    showCitationModal,
    hideCitationModal,
    showUrlModal,
    hideUrlModal,
    initDropzone,
    setActiveMode,
    setUserIdDisplay,
  };
})();
