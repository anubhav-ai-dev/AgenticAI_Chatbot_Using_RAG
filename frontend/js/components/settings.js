/* ============================================================
   components/settings.js — SettingsView: the Settings dialog.
   Sections: Profile, Appearance, Model preferences, Notifications,
   Integrations, Privacy, Help & support. Every control maps to something real:
     theme                    -> Theme (theme.js)
     text size / spacing /
     pop-up notices           -> Prefs (this browser only)
     default model            -> the existing #modelSelect (its change event reaches the controller)
     integrations             -> GET /health, fetched by the controller (SettingsView.onPane)
   View only: it never fetches and keeps no application state.
     SettingsView.open(pane?) / .setUserId(id) / .syncModels() / .renderIntegrations(data, opts) / .onPane(fn)
   ============================================================ */

window.SettingsView = (function () {

  const $ = (id) => document.getElementById(id);
  const PANES = ["profile", "appearance", "models", "notifications", "integrations", "privacy", "help"];

  let _pane = "appearance";
  let _paneHandler = null;
  let _userId = "default";

  function _el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ── Sections ──────────────────────────────────────────────

  function _tabs() { return Array.from(document.querySelectorAll(".settings__tab")); }

  function select(pane, { focus = false } = {}) {
    if (PANES.indexOf(pane) === -1) pane = "appearance";
    _pane = pane;
    _tabs().forEach((t) => {
      const on = t.dataset.pane === pane;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    PANES.forEach((p) => { const el = $("setPane-" + p); if (el) el.hidden = p !== pane; });
    const body = document.querySelector(".settings__panes");
    if (body) body.scrollTop = 0;
    if (_paneHandler) _paneHandler(pane);
  }

  function _wireTabs() {
    const list = document.querySelector(".settings__nav");
    if (!list) return;
    list.addEventListener("click", (e) => {
      const t = e.target.closest(".settings__tab");
      if (t) select(t.dataset.pane);
    });
    list.addEventListener("keydown", (e) => {
      const tabs = _tabs();
      const i = tabs.indexOf(document.activeElement);
      if (i === -1) return;
      let next = -1;
      switch (e.key) {
        case "ArrowDown":
        case "ArrowRight": next = (i + 1) % tabs.length; break;
        case "ArrowUp":
        case "ArrowLeft":  next = (i - 1 + tabs.length) % tabs.length; break;
        case "Home":       next = 0; break;
        case "End":        next = tabs.length - 1; break;
        default: return;
      }
      e.preventDefault();
      select(tabs[next].dataset.pane, { focus: true });
    });
    // The section list becomes a horizontal strip on narrow screens
    const mq = window.matchMedia("(max-width: 767px)");
    const orient = () => list.setAttribute("aria-orientation", mq.matches ? "horizontal" : "vertical");
    orient();
    if (mq.addEventListener) mq.addEventListener("change", orient);
  }

  // ── Syncing controls with their sources of truth ──────────

  function _syncNotifications() {
    const el = $("setNotifCount");
    if (!el || !window.Notifications) return;
    const n = Notifications.count();
    el.textContent = n ? `${n} ${n === 1 ? "notice" : "notices"} in the bell menu.` : "No notices in the bell menu.";
  }

  /** Mirror #modelSelect (options, groups, disabled state, reason) into the settings select. */
  function syncModels() {
    const src = $("modelSelect"), dst = $("setModel");
    if (!src || !dst) return;
    dst.textContent = "";
    const copy = (opt, parent) => {
      const o = document.createElement("option");
      o.value = opt.value;
      const label = opt.dataset.label || opt.textContent;
      o.textContent = opt.disabled && opt.dataset.reason ? `${label} (unavailable: ${opt.dataset.reason})` : label;
      o.disabled = opt.disabled;
      parent.appendChild(o);
    };
    Array.from(src.children).forEach((child) => {
      if (child.tagName === "OPTGROUP") {
        const g = document.createElement("optgroup");
        g.label = child.label;
        Array.from(child.children).forEach((opt) => copy(opt, g));
        dst.appendChild(g);
      } else if (child.tagName === "OPTION") {
        copy(child, dst);
      }
    });
    dst.value = src.value;
    dst.disabled = !dst.options.length || (dst.options.length === 1 && !dst.options[0].value);
  }

  function _syncAll() {
    if (window.RadioGroup) {
      RadioGroup.set($("setTheme"), Theme.get());
      RadioGroup.set($("setFont"), Prefs.get("fontSize"));
      RadioGroup.set($("setDensity"), Prefs.get("density"));
      RadioGroup.set($("setToasts"), Prefs.get("toasts"));
    }
    const id = $("setUserId");
    if (id) { id.textContent = _userId; id.title = _userId; }
    syncModels();
    _syncNotifications();
  }

  // ── Integrations (data comes from the controller) ─────────

  const SERVICES = [
    { key: "groq",       name: "Groq",          hint: "Chat models" },
    { key: "google",     name: "Google Gemini", hint: "Chat models" },
    { key: "tavily",     name: "Tavily",        hint: "Web search" },
    { key: "embeddings", name: "Embeddings",    hint: "Document search" },
    { key: "reranker",   name: "Reranker",      hint: "Orders search results" },
    { key: "database",   name: "Database",      hint: "Chats and documents" },
  ];

  // A label AND an icon carry the state, so it never depends on colour alone
  function _badge(key, value) {
    const v = String(value == null ? "" : value);
    if (key === "database") return v === "connected" ? ["Connected", "ok", "circle-check"] : ["Unavailable", "bad", "circle-alert"];
    if (key === "groq" || key === "google" || key === "tavily") {
      return v === "configured" ? ["Key configured", "ok", "circle-check"] : ["No API key", "warn", "triangle-alert"];
    }
    if (v === "ready") return ["Ready", "ok", "circle-check"];
    if (v === "loading") return ["Loading", "warn", "loader-circle"];
    if (v === "error") return ["Error", "bad", "circle-alert"];
    if (v === "disabled") return ["Off", "muted", "info"];
    return [v || "Unknown", "muted", "info"];
  }

  /**
   * @param {{status?:string, services?:Object}|null} data  GET /health body, or null
   * @param {{loading?:boolean, error?:string}} [opts]
   */
  function renderIntegrations(data, opts) {
    const list = $("setServices"), note = $("setServicesNote"), btn = $("setRefreshServices");
    if (!list) return;
    const o = opts || {};
    list.textContent = "";
    if (btn) btn.disabled = !!o.loading;
    if (o.loading) { if (note) note.textContent = "Checking services…"; return; }
    if (o.error || !data || !data.services) {
      if (note) note.textContent = `Could not load service status${o.error ? ": " + o.error : ""}.`;
      return;
    }
    SERVICES.forEach((s) => {
      if (!(s.key in data.services)) return;
      const [label, tone, icon] = _badge(s.key, data.services[s.key]);
      const li = _el("li", "status-row");
      const text = _el("span", "status-row__text");
      text.appendChild(_el("span", "status-row__name", s.name));
      text.appendChild(_el("span", "status-row__hint", s.hint));
      const badge = _el("span", "status-badge status-badge--" + tone);
      badge.appendChild(Icons.el(icon, { size: 16 }));
      badge.appendChild(_el("span", "", label));
      li.appendChild(text);
      li.appendChild(badge);
      list.appendChild(li);
    });
    if (note) note.textContent = data.status ? `Overall status: ${data.status}.` : "";
  }

  // ── Public ────────────────────────────────────────────────

  function open(pane) {
    const dlg = $("settingsDialog");
    if (!dlg) return;
    _syncAll();
    select(PANES.indexOf(pane) !== -1 ? pane : _pane);
    const active = _tabs().find((t) => t.getAttribute("aria-selected") === "true");
    Dialog.open(dlg, { initialFocus: active });
  }

  function close() { Dialog.close($("settingsDialog")); }
  function setUserId(id) { _userId = String(id == null || id === "" ? "default" : id); const el = $("setUserId"); if (el) { el.textContent = _userId; el.title = _userId; } }
  function onPane(fn) { _paneHandler = typeof fn === "function" ? fn : null; }

  function init() {
    const dlg = $("settingsDialog");
    if (!dlg) return;
    _wireTabs();

    const gear = $("settingsBtn");
    if (gear) gear.addEventListener("click", () => open());
    const closeBtn = $("settingsClose");
    if (closeBtn) closeBtn.addEventListener("click", close);
    dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });

    // Appearance
    RadioGroup.wire($("setTheme"), (v) => Theme.set(v));
    Theme.onChange((pref) => RadioGroup.set($("setTheme"), pref));            // the top-bar switch changes it too
    RadioGroup.wire($("setFont"), (v) => Prefs.set("fontSize", v));
    RadioGroup.wire($("setDensity"), (v) => Prefs.set("density", v));
    RadioGroup.wire($("setToasts"), (v) => Prefs.set("toasts", v));

    // Model preferences: drive the real selector so the controller's change handler persists the choice
    const model = $("setModel");
    if (model) model.addEventListener("change", () => {
      const main = $("modelSelect");
      if (!main || !model.value) return;
      main.value = model.value;
      main.dispatchEvent(new Event("change", { bubbles: true }));
    });
    const main = $("modelSelect");
    if (main) main.addEventListener("change", () => { if (model) model.value = main.value; });

    // Notifications
    const clear = $("setClearNotif");
    if (clear) clear.addEventListener("click", () => {
      if (window.Notifications) Notifications.clear();
      _syncNotifications();
      if (window.UI) UI.setLiveStatus("Notification history cleared.");
    });

    // Profile
    const copy = $("setCopyUserId");
    if (copy) copy.addEventListener("click", async () => {
      const ok = await Utils.copyText(_userId);
      if (window.UI) UI.showToast(ok ? "User ID copied" : "Could not copy the user ID", ok ? "success" : "error");
    });

    // Integrations + Privacy
    const refresh = $("setRefreshServices");
    if (refresh) refresh.addEventListener("click", () => { if (_paneHandler) _paneHandler("integrations"); });
    const docs = $("setOpenDocs");
    if (docs) docs.addEventListener("click", () => { close(); if (window.DocsView) DocsView.open(); });

    select("appearance");
  }

  return { init, open, close, select, setUserId, syncModels, renderIntegrations, onPane };
})();
