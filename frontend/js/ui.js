/* ============================================================
   ui.js — App shell (sidebar drawer / rail / expanded), modals,
            drag-drop, theme, toasts, session list, document list,
            model selector, welcome prompts
   (view layer: no fetch, no State, no userId/mode/model storage)
   ============================================================ */

window.UI = (function () {

  const $ = (id) => document.getElementById(id);

  // ── Theme ─────────────────────────────────────────────────
  // Preference storage lives in js/theme.js; these are thin wrappers
  // so existing callers (app.js: initTheme) keep working.

  // The top bar has a Light | Dark segmented switch and, on narrow screens, one quick toggle;
  // the full Light / Dark / System choice lives in Settings > Appearance.
  function _syncThemeButton() {
    const dark = Theme.resolved() === "dark";
    const light = $("themeLight"), night = $("themeDark"), quick = $("themeQuick");
    if (light) { light.setAttribute("aria-pressed", String(!dark)); light.classList.toggle("is-on", !dark); }
    if (night) { night.setAttribute("aria-pressed", String(dark)); night.classList.toggle("is-on", dark); }
    if (quick) {
      quick.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      quick.innerHTML = Icons.html(dark ? "sun" : "moon", { size: 20 });
    }
  }

  /** Idempotent. Applies the saved preference; never writes "dark" as a default. */
  function initTheme() {
    Theme.init();
    _syncThemeButton(Theme.get());
  }

  /** @param {"light"|"dark"|"system"} theme */
  function setTheme(theme) {
    Theme.set(theme);
  }

  Theme.onChange(_syncThemeButton);
  _syncThemeButton(Theme.get());

  // ── Sidebar shell ─────────────────────────────────────────
  // Three shells, written to <html data-shell> (the head snippet in index.html
  // sets the first value before paint):
  //   drawer    < 768     off-canvas from the left, opened by #sidebarToggle
  //   rail      768-1023  64px icon rail; #sidebarToggle opens it as an overlay
  //             >= 1024   rail only if the user collapsed it (localStorage.sidebarState)
  //   expanded  >= 1024   272px
  // While opened as an overlay the sidebar gets .mobile-open (legacy class name),
  // role=dialog + aria-modal, a focus trap, and <main> becomes inert.

  const SIDEBAR_KEY = "sidebarState";   // view preference: "expanded" | "rail"
  const MQ_DRAWER = window.matchMedia("(max-width: 767px)");
  const MQ_NARROW = window.matchMedia("(max-width: 1023px)");

  function _storedRail() {
    try { return localStorage.getItem(SIDEBAR_KEY) === "rail"; } catch (_) { return false; }
  }

  function getShell() {
    if (MQ_DRAWER.matches) return "drawer";
    if (MQ_NARROW.matches || _storedRail()) return "rail";
    return "expanded";
  }

  function isSidebarOpen() {
    const sb = $("sidebar");
    return !!sb && sb.classList.contains("mobile-open");
  }

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function _focusables(root) {
    return Array.from(root.querySelectorAll(FOCUSABLE)).filter((n) => n.getClientRects().length > 0);
  }

  function _setMainInert(on) {
    const main = $("main");
    if (!main) return;
    if ("inert" in main) main.inert = on;
    else if (on) main.setAttribute("aria-hidden", "true");
    else main.removeAttribute("aria-hidden");
  }

  /** Labels, icons and ARIA that depend on shell + open state. */
  function _syncSidebarChrome() {
    const shell = getShell();
    const open = isSidebarOpen();
    const toggle = $("sidebarToggle");
    const collapse = $("sidebarCollapse");
    if (toggle) {
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", MQ_NARROW.matches ? "Open navigation" : "Expand sidebar");
    }
    if (collapse) {
      const label = open ? "Close navigation" : "Collapse sidebar";
      collapse.setAttribute("aria-label", label);
      collapse.innerHTML = Icons.html(open ? "x" : "chevrons-left", { size: 20 });
    }
    document.documentElement.dataset.shell = shell;
  }

  function openSidebar() {
    const sb = $("sidebar");
    if (!sb || isSidebarOpen() || !MQ_NARROW.matches) return;   // expanded >= 1024: nothing to open
    sb.classList.add("mobile-open");
    sb.setAttribute("role", "dialog");
    sb.setAttribute("aria-modal", "true");
    sb.setAttribute("aria-label", "Navigation");
    const bd = $("sidebarBackdrop");
    if (bd) bd.classList.add("is-open");
    _setMainInert(true);
    _syncSidebarChrome();
    // The mode dropdown is the first workspace control in the drawer
    const first = $("modeSelect") || _focusables(sb)[0];
    if (first) first.focus();
  }

  /** @param {{restoreFocus?: boolean}} [opts] (an Event passed as first argument is fine) */
  function closeSidebar(opts) {
    const sb = $("sidebar");
    if (!sb || !isSidebarOpen()) return;
    const restore = !(opts && opts.restoreFocus === false);
    const hadFocus = sb.contains(document.activeElement) || document.activeElement === document.body;
    sb.classList.remove("mobile-open");
    sb.removeAttribute("role");
    sb.removeAttribute("aria-modal");
    sb.setAttribute("aria-label", "Sidebar");
    const bd = $("sidebarBackdrop");
    if (bd) bd.classList.remove("is-open");
    _setMainInert(false);
    _syncSidebarChrome();
    const toggle = $("sidebarToggle");
    if (restore && hadFocus && toggle && toggle.getClientRects().length) toggle.focus();
  }

  /**
   * User collapse of the desktop sidebar to the icon rail (>= 1024). Persists
   * localStorage.sidebarState. Below 1024 the rail is automatic and this only stores the preference.
   */
  function setSidebarCollapsed(collapsed, opts) {
    try { localStorage.setItem(SIDEBAR_KEY, collapsed ? "rail" : "expanded"); } catch (_) {}
    _syncSidebarChrome();
    if (opts && opts.focus === false) return;
    const target = $(collapsed ? "sidebarToggle" : "sidebarCollapse");
    if (target && target.getClientRects().length) target.focus();
  }

  function isSidebarCollapsed() { return getShell() === "rail"; }

  function toggleSidebar() {
    if (MQ_NARROW.matches) { isSidebarOpen() ? closeSidebar() : openSidebar(); return; }
    setSidebarCollapsed(getShell() !== "rail");
  }

  function _onShellChange() {
    // Leaving the narrow range with the overlay open (e.g. rotate / resize): drop the overlay state
    if (isSidebarOpen() && !MQ_NARROW.matches) closeSidebar({ restoreFocus: false });
    _syncSidebarChrome();
  }

  // Keeps Tab inside the open drawer / overlay
  function _trapSidebarTab(e) {
    if (e.key !== "Tab" || !isSidebarOpen()) return;
    if (document.querySelector("dialog[open]") || Menu.isOpen()) return;
    const sb = $("sidebar");
    const items = _focusables(sb);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    if (!sb.contains(active)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && active === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && active === last) { e.preventDefault(); first.focus(); }
  }

  // ── Session list ──────────────────────────────────────────

  const MODE_ICONS = { general: "message-circle", document: "file-text", research: "globe", code: "code-xml" };
  // Reference grouping: Today / Previous 7 Days / Older, derived from the real updated_at only
  const GROUP_ORDER = ["Today", "Previous 7 Days", "Older"];
  const GROUP_OF = { "Today": "Today", "Yesterday": "Previous 7 Days", "Previous 7 Days": "Previous 7 Days" };
  let _dateFilter = "all";          // view-only filter: all | one of GROUP_ORDER
  let _lastList = null;             // last renderSessionList arguments (re-rendered when the filter changes)

  function _groupOf(iso) { return GROUP_OF[Utils.formatDateGroup(iso)] || "Older"; }
  const TITLE_MAX = 200;   // SessionUpdate.title max_length (backend/models/api.py)

  /** Time of day for today's chats, a short date for older ones. */
  function _sessionStamp(group, iso) {
    const d = Utils.parseDate(iso);
    if (!d) return "";
    if (group === "Today") return Utils.formatTime(iso);
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  /**
   * @param {Array} sessions     SessionSummary[] (already filtered by the controller)
   * @param {string} activeId
   * @param {(id:string)=>void} onSelect
   * @param {(id:string)=>void} onDelete
   * @param {{onRename?: (id:string, title:string)=>(void|Promise)}} [handlers]
   *        The rename control is rendered only when handlers.onRename is a function.
   */
  function renderSessionList(sessions, activeId, onSelect, onDelete, handlers) {
    const list = $("sessionList");
    if (!list) return;
    const onRename = handlers && typeof handlers.onRename === "function" ? handlers.onRename : null;
    _lastList = { sessions, activeId, onSelect, onDelete, handlers };
    list.setAttribute("aria-busy", "false");

    // Keep keyboard focus across the controller's full re-render
    const ae = document.activeElement;
    let restore = null;
    if (ae && list.contains(ae)) {
      const row = ae.closest(".session-item");
      if (row) restore = { id: row.dataset.sessionId, cls: ["session-item__main", "delete-session-btn", "rename-session-btn"].find((c) => ae.classList.contains(c)) };
    }

    const groups = {};
    (sessions || []).forEach((s) => {
      const g = _groupOf(s.updated_at);
      if (_dateFilter !== "all" && g !== _dateFilter) return;
      (groups[g] = groups[g] || []).push(s);
    });

    if (!Object.keys(groups).length) {
      const search = $("sessionSearch");
      const q = search ? search.value.trim() : "";
      list.textContent = "";
      list.appendChild(Object.assign(document.createElement("div"), {
        className: "sidebar__empty",
        textContent: q ? `No chats match “${q}”`
          : (sessions && sessions.length && _dateFilter !== "all" ? `No chats in “${_dateFilter}”` : "No chats yet"),
      }));
      return;
    }

    let html = "";
    GROUP_ORDER.forEach((groupName, gi) => {
      const items = groups[groupName];
      if (!items || items.length === 0) return;
      const gid = `sessionGroup${gi}`;
      html += `<section class="session-group" aria-labelledby="${gid}">
        <h2 class="session-group-label" id="${gid}">${groupName}</h2><ul class="session-group__list">`;
      items.forEach((s) => {
        const isActive = s.id === activeId;
        const icon = Icons.html(MODE_ICONS[s.mode] || "message-circle", { size: 16 });
        const stamp = _sessionStamp(groupName, s.updated_at);
        const model = String(s.model || "");
        const rawTitle = s.title || "Untitled";
        const title = Utils.escapeHtml(rawTitle);
        const tip = Utils.escapeHtml(model ? `${rawTitle} · ${model}` : rawTitle);
        const id = Utils.escapeHtml(s.id);
        html += `
          <li class="session-item${isActive ? " active" : ""}" data-session-id="${id}">
            <button type="button" class="session-item__main" title="${tip}"${isActive ? ' aria-current="true"' : ""}>
              <span class="session-item__icon">${icon}</span>
              <span class="session-item__title truncate">${title}</span>
              <span class="session-item__time">${Utils.escapeHtml(stamp)}</span>
            </button>
            <div class="session-item__actions">
              ${onRename ? `<button type="button" class="btn btn--icon-sm rename-session-btn" aria-label="Rename chat: ${title}" title="Rename">${Icons.html("pencil", { size: 16 })}</button>` : ""}
              <button type="button" class="btn btn--icon-sm delete-session-btn" data-session-id="${id}"
                      aria-label="Delete chat: ${title}" title="Delete">
                ${Icons.html("trash-2", { size: 16 })}
              </button>
            </div>
          </li>
        `;
      });
      html += `</ul></section>`;
    });

    list.innerHTML = html;

    list.querySelectorAll(".session-item").forEach((row) => {
      const id = row.dataset.sessionId;
      const session = sessions.find((x) => x.id === id) || {};
      row.querySelector(".session-item__main").addEventListener("click", () => {
        onSelect(id);
        closeSidebar();   // no-op unless the drawer / overlay is open
      });
      row.querySelector(".delete-session-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        onDelete(id);
      });
      const ren = row.querySelector(".rename-session-btn");
      if (ren) ren.addEventListener("click", (e) => {
        e.stopPropagation();
        _beginRename(row, session, onRename);
      });
    });

    if (restore) {
      const row = Array.from(list.querySelectorAll(".session-item")).find((r) => r.dataset.sessionId === restore.id);
      const target = row && (row.querySelector("." + restore.cls) || row.querySelector(".session-item__main"));
      if (target) target.focus();
    }
  }

  /**
   * Loading / error state of the history list ("loading" | "error"). renderSessionList replaces it
   * (an empty history gets its own message there). error takes { message, onRetry }.
   */
  function setSessionsStatus(state, opts) {
    const list = $("sessionList");
    if (!list) return;
    const o = opts || {};
    list.textContent = "";
    list.setAttribute("aria-busy", state === "loading" ? "true" : "false");
    if (state === "loading") {
      for (let i = 0; i < 4; i++) {
        const row = document.createElement("div");
        row.className = "doc-skeleton";
        const icon = document.createElement("span");
        icon.className = "skeleton doc-skeleton__icon";
        const lines = document.createElement("span");
        lines.className = "doc-skeleton__lines";
        const line = document.createElement("div");
        line.className = "skeleton skeleton-line skeleton-line--80";
        lines.appendChild(line);
        row.appendChild(icon);
        row.appendChild(lines);
        list.appendChild(row);
      }
      return;
    }
    if (state === "error") {
      const box = document.createElement("div");
      box.className = "list-state list-state--error";
      box.appendChild(Icons.el("circle-alert", { size: 24, cls: "list-state__icon" }));
      const title = document.createElement("div");
      title.className = "list-state__title";
      title.textContent = "Couldn’t load your chats";
      box.appendChild(title);
      if (o.message) { const m = document.createElement("div"); m.textContent = String(o.message); box.appendChild(m); }
      if (typeof o.onRetry === "function") {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "btn btn--secondary btn--sm";
        b.textContent = "Try again";
        b.addEventListener("click", o.onRetry);
        box.appendChild(b);
      }
      list.appendChild(box);
    }
  }

  /** Inline rename (only reachable when the controller supplied onRename). */
  function _beginRename(row, session, onRename) {
    if (row.classList.contains("is-renaming")) return;
    const main = row.querySelector(".session-item__main");
    const actions = row.querySelector(".session-item__actions");
    const original = session.title || "Untitled";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "input session-rename";
    input.value = original;
    input.maxLength = TITLE_MAX;
    input.setAttribute("aria-label", "Chat title");
    row.classList.add("is-renaming");
    main.hidden = true;
    actions.hidden = true;
    row.insertBefore(input, main);
    input.focus();
    input.select();

    let done = false;
    const finish = (commit) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      const changed = commit && value && value !== original;
      const revert = () => {
        input.remove();
        main.hidden = false;
        actions.hidden = false;
        row.classList.remove("is-renaming");
      };
      if (!changed) { revert(); main.focus(); return; }
      input.disabled = true;
      Promise.resolve()
        .then(() => onRename(session.id, value))
        .then(() => {
          const t = main.querySelector(".session-item__title");
          if (t) t.textContent = value;
          revert();
          main.focus();
        })
        .catch(() => { revert(); main.focus(); });
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
  }

  // ── Document list ─────────────────────────────────────────

  /** Render GET /documents rows into every document list (workspace panel + My Documents). */
  function renderDocList(documents, onDelete) {
    if (window.DocsView) DocsView.render(documents, onDelete);
  }

  /** "loading" | "error" | "ready" for the document lists; error takes { message, onRetry }. */
  function setDocsStatus(state, opts) {
    if (window.DocsView) DocsView.setStatus(state, opts);
  }

  // ── Model selector (populated from GET /models) ───────────

  function renderModelSelect(models, selected, opts) {
    const sel = $("modelSelect");
    if (!sel) return;
    const groups = {};
    models.forEach((m) => {
      (groups[m.provider] = groups[m.provider] || []).push(m);
    });
    const names = { groq: "Groq", google: "Google" };
    sel.innerHTML = "";
    Object.keys(groups).forEach((provider) => {
      const og = document.createElement("optgroup");
      og.label = names[provider] || provider;
      groups[provider].forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.label == null || m.label === "" ? m.id : m.label}${m.description ? " — " + m.description : ""}`;
        // Read by components/model-select.js (the native text stays the fallback)
        opt.dataset.label = m.label == null || m.label === "" ? String(m.id == null ? "" : m.id) : String(m.label);
        if (m.description) opt.dataset.description = String(m.description);
        opt.disabled = !m.available;
        if (!m.available && m.reason) { opt.title = m.reason; opt.dataset.reason = String(m.reason); }
        og.appendChild(opt);
      });
      sel.appendChild(og);
    });
    if (selected) sel.value = selected;
    // C7: failure state (distinct from "no models"); the real <select> stays the source of truth
    if (window.ModelSelect) {
      const o = opts || {};
      if (o.error) window.ModelSelect.setError(String(o.error), typeof o.onRetry === "function" ? o.onRetry : null);
      else window.ModelSelect.clearError();
    }
    if (window.SettingsView) SettingsView.syncModels();
  }

  // ── Upload toast ──────────────────────────────────────────
  // The API reports no progress, so the bar is indeterminate (pure CSS).
  // showUploadError (view-contract C6) swaps the same toast into an error state.

  function _clearUploadError() {
    const toast = $("uploadToast");
    if (!toast) return;
    toast.classList.remove("upload-toast--error");
    toast.setAttribute("role", "status");
    toast.removeAttribute("aria-label");
    toast.querySelectorAll(".upload-toast__msg, .upload-toast__actions").forEach((n) => n.remove());
    const bar = toast.querySelector(".progress-bar");
    if (bar) bar.hidden = false;
    const icon = toast.querySelector(".upload-toast__title .icon");
    if (icon) {
      icon.classList.add("icon--spin");
      const use = icon.querySelector("use");
      if (use) use.setAttribute("href", "#i-loader-circle");
    }
    const title = $("uploadToastTitle");
    if (title) title.textContent = "Processing document… large or scanned files can take a minute";
  }

  function showUploadToast(filename) {
    const toast = $("uploadToast");
    const nameEl = $("uploadToastName");
    if (!toast) return;
    _clearUploadError();
    if (nameEl) nameEl.textContent = filename;
    toast.hidden = false;
    if (window.Workspaces) Workspaces.setUploadBusy(true, filename);
  }

  function hideUploadToast() {
    const toast = $("uploadToast");
    if (!toast) return;
    const hadFocus = toast.contains(document.activeElement);
    toast.hidden = true;
    _clearUploadError();
    if (window.Workspaces) Workspaces.setUploadBusy(false);
    if (hadFocus) { const a = $("attachBtn"); if (a) a.focus(); }
  }

  /**
   * Error state of the upload toast.
   * filename: string (textContent). opts: { message?: string, onRetry?: () => void }.
   * Retry is rendered only when onRetry is a function; Dismiss always. Esc inside the toast dismisses.
   * Announced through #liveStatus; focus is not moved.
   */
  function showUploadError(filename, opts) {
    const toast = $("uploadToast");
    if (!toast) return;
    const o = opts || {};
    _clearUploadError();
    toast.classList.add("upload-toast--error");
    toast.setAttribute("role", "group");
    toast.setAttribute("aria-label", "Upload failed");
    const nameEl = $("uploadToastName");
    if (nameEl) nameEl.textContent = filename == null ? "" : String(filename);
    const title = $("uploadToastTitle");
    if (title) title.textContent = "Upload failed";
    const icon = toast.querySelector(".upload-toast__title .icon");
    if (icon) {
      icon.classList.remove("icon--spin");
      const use = icon.querySelector("use");
      if (use) use.setAttribute("href", "#i-circle-alert");
    }
    const bar = toast.querySelector(".progress-bar");
    if (bar) bar.hidden = true;

    const message = o.message == null ? "" : String(o.message);
    if (message) {
      const msg = document.createElement("div");
      msg.className = "upload-toast__msg";
      msg.textContent = message;
      toast.appendChild(msg);
    }
    const actions = document.createElement("div");
    actions.className = "upload-toast__actions";
    if (typeof o.onRetry === "function") {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "btn btn--secondary btn--sm";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => {
        hideUploadToast();
        o.onRetry();
      });
      actions.appendChild(retry);
    }
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "btn btn--ghost btn--sm";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", hideUploadToast);
    actions.appendChild(dismiss);
    toast.appendChild(actions);

    toast.onkeydown = (e) => { if (e.key === "Escape") { e.stopPropagation(); hideUploadToast(); } };
    toast.hidden = false;
    setLiveStatus(`Upload failed: ${filename == null ? "" : filename}.${message ? " " + message : ""}`);
    if (window.Notifications) Notifications.record(`Upload failed: ${filename == null ? "" : filename}.${message ? " " + message : ""}`, "error");
  }

  // ── Toasts ────────────────────────────────────────────────
  // `message` may still contain legacy emoji from app.js; it is shown as plain text.

  const TOAST_ICONS = { info: "info", success: "circle-check", error: "circle-alert" };
  const TOAST_MAX = 3;

  function _toastRegion() {
    let region = document.querySelector(".toast-region");
    if (!region) {
      region = document.createElement("div");
      region.className = "toast-region";
      region.setAttribute("aria-live", "polite");
      region.setAttribute("aria-relevant", "additions");
      document.body.appendChild(region);
    }
    return region;
  }

  function _removeToast(el) {
    if (!el || !el.parentNode) return;
    clearTimeout(el._timer);
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), 130);
  }

  function showToast(message, type = "info") {
    if (!TOAST_ICONS[type]) type = "info";
    // Settings > Notifications: with "Errors only" the other notices skip the pop-up but stay in the bell menu
    if (type !== "error" && window.Prefs && Prefs.get("toasts") === "errors") {
      if (window.Notifications) Notifications.record(message, type);
      return;
    }
    const region = _toastRegion();
    while (region.children.length >= TOAST_MAX) {
      const oldest = region.firstElementChild;
      clearTimeout(oldest._timer);
      oldest.remove();
    }

    const el = document.createElement("div");
    el.className = `toast toast--${type} animate-slide-right`;
    if (type === "error") el.setAttribute("role", "alert");

    const icon = document.createElement("span");
    icon.className = "toast__icon";
    icon.innerHTML = Icons.html(TOAST_ICONS[type], { size: 20 });
    const text = document.createElement("span");
    text.className = "toast__text";
    text.textContent = String(message == null ? "" : message);
    el.appendChild(icon);
    el.appendChild(text);

    if (type === "error") {
      const close = document.createElement("button");
      close.type = "button";
      close.className = "btn btn--icon-sm toast__close";
      close.setAttribute("aria-label", "Dismiss notification");
      close.innerHTML = Icons.html("x", { size: 16 });
      close.addEventListener("click", () => _removeToast(el));
      el.appendChild(close);
    }

    const duration = type === "error" ? 8000 : 4000;
    const start = (ms) => {
      clearTimeout(el._timer);
      el._timer = setTimeout(() => _removeToast(el), ms);
    };
    // Stay while hovered / focused
    el.addEventListener("mouseenter", () => clearTimeout(el._timer));
    el.addEventListener("focusin", () => clearTimeout(el._timer));
    el.addEventListener("mouseleave", () => start(2000));
    el.addEventListener("focusout", () => start(2000));

    region.appendChild(el);
    start(duration);
    if (window.Notifications) Notifications.record(message, type);
  }

  // ── Live status (screen-reader announcements, never per token) ─

  let _liveTimer = null;

  /** Coalesced write to the visually hidden #liveStatus region. Empty string clears it. */
  function setLiveStatus(text) {
    const el = $("liveStatus");
    if (!el) return;
    clearTimeout(_liveTimer);
    if (!text) { el.textContent = ""; return; }
    _liveTimer = setTimeout(() => { el.textContent = String(text); }, 150);
  }

  // ── Citation modal ────────────────────────────────────────

  function showCitationModal(citation) {
    if (!citation) return;
    const modal = $("citationModal");
    const sourceEl = $("citationSource");
    const snippetEl = $("citationSnippet");
    if (!modal) return;

    if (sourceEl) {
      sourceEl.textContent = "";
      const href = Utils.safeUrl(citation.url);
      sourceEl.appendChild(Icons.el(citation.type === "web" ? "globe" : "file-text", { size: 16 }));
      const strong = document.createElement("strong");
      const page = citation.page ? ` · Page ${citation.page}` : "";
      strong.textContent = `${citation.source == null ? "" : citation.source}${page}`;
      sourceEl.appendChild(strong);
      if (href) {
        const a = document.createElement("a");
        a.className = "citation-modal__link";
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "Open " + new URL(href).hostname.replace(/^www\./, "");
        sourceEl.appendChild(a);
      }
    }
    if (snippetEl) snippetEl.textContent = citation.snippet || "No preview available.";
    Dialog.open(modal, { initialFocus: "#citationModalClose" });
  }

  function hideCitationModal() {
    Dialog.close($("citationModal"));
  }

  // ── URL modal ─────────────────────────────────────────────

  function showUrlModal() {
    const modal = $("urlModal");
    if (!modal) return;
    const inp = $("urlInput");
    if (inp && !modal.open) inp.value = "";
    Dialog.open(modal, { initialFocus: inp });
  }

  function hideUrlModal() {
    Dialog.close($("urlModal"));
  }

  // ── Confirmation dialog (replaces window.confirm) ─────────

  /**
   * @param {{title:string, message?:string, confirmLabel?:string, cancelLabel?:string, destructive?:boolean}} opts
   * @returns {Promise<boolean>} true on confirm; false on cancel, Esc or backdrop click
   */
  function confirmDialog(opts) {
    return Dialog.confirm(opts);
  }

  // ── Drag-and-drop zone ────────────────────────────────────

  function initDropzone(onFileDrop) {
    const overlay = $("dropzoneOverlay");
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
      if (files.length > 0) onFileDrop(files);
    });
  }

  // ── Active mode tab ───────────────────────────────────────

  // ── Workspace mode dropdown (sidebar, above New Chat) ─────
  // One trigger (#modeSelect) opens a menu with the four workspaces. The view owns only the
  // presentation: the controller registers UI.onModeChange(fn) and answers with UI.setActiveMode(mode).

  const MODES = [
    { id: "general",  label: "General",   icon: "message-circle" },
    { id: "document", label: "Documents", icon: "file-text" },
    { id: "research", label: "Research",  icon: "globe" },
    { id: "code",     label: "Code",      icon: "code-xml" },
  ];
  const MODE_BY_ID = MODES.reduce((acc, m) => { acc[m.id] = m; return acc; }, {});
  let _mode = "general";
  let _modeApplied = false;     // false until the first setActiveMode (the entrance covers the first view)
  let _modeHandler = null;

  /** Register the controller callback for the mode dropdown: handler(modeId). Pass null to unregister. */
  function onModeChange(handler) {
    _modeHandler = typeof handler === "function" ? handler : null;
  }

  function toggleModeMenu() {
    const btn = $("modeSelect");
    if (!btn) return;
    if (Menu.isOpen(btn)) { Menu.close(); return; }
    Menu.open({
      anchor: btn,
      label: "Workspace mode",
      align: "start",
      matchWidth: true,
      className: "menu--mode",
      items: MODES.map((m) => ({ id: m.id, label: m.label, icon: m.icon, checked: m.id === _mode })),
      onSelect: (id) => {
        if (id === _mode) return;
        if (_modeHandler) _modeHandler(id);
        closeSidebar();   // no-op unless the drawer / overlay is open; focus returns to the menu button
      },
    });
  }

  /** The Menu is a body-level popover: ArrowDown / ArrowUp on the trigger open it (menu-button pattern). */
  function _wireModeSelect() {
    const btn = $("modeSelect");
    if (!btn) return;
    btn.addEventListener("click", toggleModeMenu);
    btn.addEventListener("keydown", (e) => {
      if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !Menu.isOpen(btn)) {
        e.preventDefault();
        toggleModeMenu();
      }
    });
  }

  /** Swap a mode glyph. A later change replays a short fade; the first paint does not. */
  function _paintModeIcon(slot, name, size, animate) {
    if (!slot) return;
    slot.replaceChildren(Icons.el(name, { size }));
    if (!animate) return;
    slot.classList.remove("is-swap");
    requestAnimationFrame(() => slot.classList.add("is-swap"));
  }

  /** Reflect the active workspace everywhere the view shows it. Idempotent. */
  function setActiveMode(mode) {
    const m = MODE_BY_ID[mode] || MODE_BY_ID.general;
    const switched = _modeApplied && m.id !== _mode;
    _modeApplied = true;
    _mode = m.id;

    const btn = $("modeSelect");
    if (btn) {
      btn.dataset.mode = m.id;
      btn.setAttribute("aria-label", `Workspace mode: ${m.label}`);
      btn.setAttribute("data-rail-tip", `Workspace: ${m.label}`);
      const icon = btn.querySelector(".mode-select__icon");
      _paintModeIcon(icon, m.icon, 20, switched);
      const label = $("modeSelectLabel");
      if (label) label.textContent = m.label;
    }

    const title = $("contextTitle");
    if (title) {
      title.dataset.mode = m.id;
      const tIcon = title.querySelector(".topbar__title-icon");
      _paintModeIcon(tIcon, m.icon, 16, switched);
      const name = $("contextName");
      if (name) name.textContent = m.label;
    }

    // Layout / workspace hooks: CSS shows the matching empty-state panel; Code uses a wider column
    document.documentElement.dataset.workspace = m.id;
    if (window.Workspaces) Workspaces.setActive(m.id);
    if (switched && window.Motion) Motion.workspaceChanged();   // the new panel eases in
  }

  // ── User ID display ───────────────────────────────────────

  /** Shows the client-side user id (not an account) and its initial in both avatars. */
  function setUserIdDisplay(userId) {
    const id = String(userId == null || userId === "" ? "default" : userId);
    const initial = (id.trim().charAt(0) || "?").toUpperCase();
    const el = $("userIdDisplay");
    if (el) { el.textContent = id; el.title = id; }
    if (window.SettingsView) SettingsView.setUserId(id);
    ["sidebarAvatar", "topAvatar"].forEach((k) => { const a = $(k); if (a) a.textContent = initial; });
    const top = $("topAvatar");
    if (top) top.setAttribute("aria-label", `User: ${id}`);
  }

  // ── AI visualization (welcome screen instance) ────────────

  /**
   * Drive the welcome-screen visualization: idle | processing | responding | error.
   * Safe to call before/without the component. Message avatars are driven through the
   * renderer returned by Chat.startAssistantMessage() (renderer.setVizState).
   */
  function setVizState(state) {
    if (window.AIViz) AIViz.setState($("welcomeViz"), state);
  }

  // ── Global view wiring (no controller involvement) ────────

  function _initViewWiring() {
    // Welcome mark: replaces the static fallback icon with the animated component
    if (window.AIViz && $("welcomeViz")) AIViz.mount($("welcomeViz"), { size: "hero", state: "idle" });

    // Sidebar: scrim, close / collapse button, navigation closes the overlay
    const bd = $("sidebarBackdrop");
    if (bd) bd.addEventListener("click", closeSidebar);

    const collapse = $("sidebarCollapse");
    if (collapse) collapse.addEventListener("click", () => {
      if (isSidebarOpen()) closeSidebar();
      else setSidebarCollapsed(true);
    });

    _wireModeSelect();
    if (window.DocsView) DocsView.init();
    if (window.Workspaces) Workspaces.init();
    if (window.SettingsView) SettingsView.init();
    if (window.Motion) Motion.enter();
    const newChat = $("newChatBtn");
    if (newChat) newChat.addEventListener("click", () => closeSidebar());

    if (MQ_DRAWER.addEventListener) {
      MQ_DRAWER.addEventListener("change", _onShellChange);
      MQ_NARROW.addEventListener("change", _onShellChange);
    }
    document.addEventListener("keydown", _trapSidebarTab);
    _syncSidebarChrome();

    _wireReferenceShell();

    // Welcome feature cards (Stage 4 seam S4-1). If the controller registered UI.onWelcomeCard,
    // the card click (mouse, Enter and Space all arrive as "click" on a <button>) is reported to it
    // and the legacy card listener in app.js is suppressed (capture phase on the ancestor #welcomeScreen).
    // The view itself never switches mode, fills the composer or sends.
    const cards = $("welcomeScreen");
    if (cards) cards.addEventListener("click", (e) => {
      if (!_welcomeCardHandler) return;                 // legacy path: app.js handles it
      const card = e.target.closest(".feature-card");
      if (!card || !cards.contains(card)) return;
      e.preventDefault();
      e.stopPropagation();                              // capture on the ancestor: the target's own listeners never run
      _welcomeCardHandler({ mode: card.dataset.mode || "", prompt: card.dataset.prompt || "", action: card.dataset.action || "" });
    }, true);

    // History: ArrowUp / ArrowDown move between chats (Tab still reaches rename and delete)
    const history = $("sessionList");
    if (history) history.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      const mains = Array.from(history.querySelectorAll(".session-item__main")).filter((b) => !b.hidden);
      const i = mains.indexOf(document.activeElement);
      if (i === -1) return;
      e.preventDefault();
      mains[Math.max(0, Math.min(mains.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))].focus();
    });

    // Skip link: move focus to the composer
    const skip = document.querySelector(".skip-link");
    if (skip) {
      skip.addEventListener("click", (e) => {
        e.preventDefault();
        const input = $("chatInput");
        if (input) input.focus();
      });
    }

    // Esc closes the drawer / overlay (menus, listbox and dialogs handle their own Esc)
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      if (isSidebarOpen() && !Menu.isOpen() && !document.querySelector("dialog[open]")) {
        e.preventDefault();
        closeSidebar();
      }
    });

    // Esc dismisses the tooltip of the focused control; it returns on the next focus / hover
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const t = document.activeElement;
      if (t && t.hasAttribute && t.hasAttribute("data-tooltip")) t.setAttribute("data-tooltip-hidden", "");
    });
    const showTooltip = (e) => {
      if (e.target && e.target.removeAttribute) e.target.removeAttribute("data-tooltip-hidden");
    };
    document.addEventListener("focusout", showTooltip);
    document.addEventListener("mouseout", showTooltip);

    // Model selector listbox (wraps #modelSelect; the native select stays the source of truth)
    if (window.ModelSelect) { ModelSelect.init($("modelSelect")); ModelSelect.bindChip($("composerModel")); }
  }

  // ── Reference shell: top tabs, recent toggle, filter, My Documents, theme switch, search ──

  let _myDocsHandler = null;

  /** Register the controller callback for the My Documents card: handler(). Pass null to unregister. */
  function onMyDocuments(handler) {
    _myDocsHandler = typeof handler === "function" ? handler : null;
  }

  /** Make sure the sidebar is visible (overlay / expanded rail) and the Recent panel is open, then focus the search box. */
  function focusSessionSearch() {
    const input = $("sessionSearch");
    if (!input) return false;
    if (MQ_NARROW.matches) openSidebar();
    else if (getShell() === "rail") setSidebarCollapsed(false, { focus: false });
    _setRecentOpen(true);
    input.focus();
    return true;
  }

  function _setRecentOpen(open) {
    const btn = $("recentToggle"), panel = $("recentPanel");
    if (!btn || !panel) return;
    btn.setAttribute("aria-expanded", String(open));
    panel.hidden = !open;
  }

  function _wireReferenceShell() {
    const recent = $("recentToggle");
    if (recent) recent.addEventListener("click", () => _setRecentOpen(recent.getAttribute("aria-expanded") !== "true"));

    // Date-range filter for the session list (view-only; applies to the sessions it was given)
    const filter = $("sessionFilterBtn");
    if (filter) filter.addEventListener("click", () => {
      if (Menu.isOpen(filter)) { Menu.close(); return; }
      Menu.open({
        anchor: filter,
        label: "Show conversations from",
        align: "start",
        items: [{ id: "all", label: "All conversations", checked: _dateFilter === "all" }]
          .concat(GROUP_ORDER.map((g) => ({ id: g, label: g, checked: _dateFilter === g }))),
        onSelect: (id) => {
          _dateFilter = id;
          filter.classList.toggle("is-active", id !== "all");
          if (_lastList) renderSessionList(_lastList.sessions, _lastList.activeId, _lastList.onSelect, _lastList.onDelete, _lastList.handlers);
        },
      });
    });

    // My Documents card: opens the file-management panel; the controller refreshes the list
    const docs = $("myDocsBtn");
    if (docs) docs.addEventListener("click", () => {
      if (window.DocsView) DocsView.open();
      if (_myDocsHandler) _myDocsHandler();
    });

    // Theme: Light | Dark segmented switch and the narrow-screen quick toggle (view preference)
    const light = $("themeLight"), night = $("themeDark"), quick = $("themeQuick");
    if (light) light.addEventListener("click", () => Theme.set("light"));
    if (night) night.addEventListener("click", () => Theme.set("dark"));
    if (quick) quick.addEventListener("click", () => Theme.set(Theme.resolved() === "dark" ? "light" : "dark"));
    _syncThemeButton(Theme.get());

    const search = $("topSearchBtn");
    if (search) search.addEventListener("click", focusSessionSearch);

    _wireHeroLook();

    // Composer model chip mirrors the selector
    if (window.ModelSelect) ModelSelect.bindChip($("composerModel"));
  }

  /**
   * The hero orb's eyes follow the pointer (and glance down while the composer has focus).
   * Only writes --look-x / --look-y (-1..1) on the hero; the motion itself is a CSS transition.
   * Skipped for touch input and when the user prefers reduced motion.
   */
  function _wireHeroLook() {
    const motionOk = window.matchMedia("(hover: hover) and (prefers-reduced-motion: no-preference)");
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    let viz = null;
    const hero = () => (viz && viz.isConnected ? viz : (viz = document.querySelector(".ai-viz--hero")));
    const look = (x, y) => {
      const v = hero();
      if (!v) return;
      v.style.setProperty("--look-x", x.toFixed(2));
      v.style.setProperty("--look-y", y.toFixed(2));
    };
    document.addEventListener("pointermove", (e) => {
      if (!motionOk.matches || e.pointerType === "touch") return;
      const v = hero();
      if (!v || !v.offsetParent) return;           // welcome screen hidden
      const r = v.getBoundingClientRect();
      look(clamp((e.clientX - (r.left + r.width / 2)) / 360), clamp((e.clientY - (r.top + r.height / 2)) / 260));
    }, { passive: true });
    document.documentElement.addEventListener("mouseleave", () => look(0, 0));
    const input = $("chatInput");
    if (input) {
      input.addEventListener("focus", () => { if (motionOk.matches) look(0, 1); });
      input.addEventListener("blur", () => look(0, 0));
    }
  }

  /** Show the server's real per-file limit in the Documents drop zone (MB), or nothing when it is unknown. */
  function setUploadLimits(maxMb) {
    if (window.Workspaces) Workspaces.setLimits(maxMb);
  }

  /** Open Settings, optionally on one section: "profile" | "appearance" | "models" | "notifications" | "integrations" | "privacy" | "help". */
  function openSettings(pane) {
    if (window.SettingsView) SettingsView.open(pane);
  }

  /** Controller hook: called with the section id whenever a Settings section is shown (used to load Integrations). */
  function onSettingsPane(handler) {
    if (window.SettingsView) SettingsView.onPane(handler);
  }

  /** Render the Integrations section from a GET /health body, or { loading } / { error }. */
  function renderIntegrations(data, opts) {
    if (window.SettingsView) SettingsView.renderIntegrations(data, opts);
  }

  function openDocuments() {
    if (window.DocsView) DocsView.open();
  }

  /** Record a notice in the bell popover without showing a toast (toasts are recorded automatically). */
  function notify(text, type) {
    if (window.Notifications) Notifications.record(text, type);
  }

  function clearNotifications() {
    if (window.Notifications) Notifications.clear();
  }

  // ── Stage 4 view hooks ─────────────────────────────────────
  let _welcomeCardHandler = null;

  /** Register the controller callback for welcome cards: handler({mode, prompt}). Pass null to unregister. */
  function onWelcomeCard(handler) {
    _welcomeCardHandler = typeof handler === "function" ? handler : null;
  }

  /** Focus the composer (no-op if hidden); optionally put the caret at the end. */
  function focusComposer() {
    const input = $("chatInput");
    if (!input || input.disabled || !input.getClientRects().length) return false;
    input.focus();
    const n = input.value.length;
    try { input.setSelectionRange(n, n); } catch (_) { /* ignore */ }
    return true;
  }

  /**
   * Reflect a running request in the composer. Pure view: toggles #sendBtn/#stopBtn visibility via the
   * `hidden` attribute and aria-busy on the composer, announces nothing, never disables #chatInput.
   * If focus sat on a control that gets hidden (Send or Stop) it is moved to the composer.
   * Optional alternative to the controller writing style.display itself.
   */
  function setComposerBusy(busy) {
    const send = $("sendBtn"), stop = $("stopBtn"), input = $("chatInput");
    const active = document.activeElement;
    const composer = document.querySelector(".composer");
    if (composer) composer.setAttribute("aria-busy", busy ? "true" : "false");
    if (send) { send.hidden = !!busy; send.style.display = busy ? "none" : ""; }
    if (stop) { stop.hidden = !busy; stop.style.display = busy ? "" : "none"; }
    if (!active || active === document.body || active === send || active === stop) {
      if (input && !input.disabled) input.focus();
    }
  }

  _initViewWiring();

  return {
    initTheme,
    setTheme,
    openSidebar,
    closeSidebar,
    toggleSidebar,
    setSidebarCollapsed,
    isSidebarCollapsed,
    isSidebarOpen,
    renderSessionList,
    setSessionsStatus,
    renderDocList,
    setDocsStatus,
    openDocuments,
    setUploadLimits,
    openSettings,
    onSettingsPane,
    renderIntegrations,
    renderModelSelect,
    showUploadToast,
    showUploadError,
    hideUploadToast,
    showToast,
    setLiveStatus,
    showCitationModal,
    hideCitationModal,
    showUrlModal,
    hideUrlModal,
    confirm: confirmDialog,
    initDropzone,
    setActiveMode,
    setUserIdDisplay,
    setVizState,
    onWelcomeCard,
    focusComposer,
    setComposerBusy,
    onMyDocuments,
    onModeChange,
    focusSessionSearch,
    notify,
    clearNotifications,
  };
})();
