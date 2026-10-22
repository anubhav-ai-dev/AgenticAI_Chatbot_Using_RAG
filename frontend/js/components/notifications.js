/* ============================================================
   components/notifications.js — in-app notifications popover (bell)
   Notifications.record(text, type)   remember a notice (toasts call this)
   Notifications.open() / close() / isOpen() / clear() / count()

   Only real notices are listed: whatever the app showed as a toast or
   upload error during this page session (kept in memory, newest first,
   max 20). Nothing is persisted and nothing is invented; with no notices
   the panel says "No notifications". The dot on the bell shows only while
   unread notices exist; opening the panel marks them read.
   All text is set with textContent. View only: no fetch, no app state.
   ============================================================ */

window.Notifications = (function () {

  const MAX = 20;
  const ICON = { info: "info", success: "circle-check", error: "circle-alert" };

  const items = [];       // { text, type, at: Date }, newest first
  let unread = 0;
  let btn = null;
  let dot = null;
  let panel = null;       // { el, cleanup }

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  function _stamp(d) {
    try { return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }); } catch (_) { return ""; }
  }

  function _syncDot() {
    if (dot) dot.hidden = unread === 0;
    if (btn) {
      btn.setAttribute("aria-label", unread ? `Notifications, ${unread} unread` : "Notifications");
    }
  }

  function _render() {
    if (!panel) return;
    const body = panel.body;
    body.textContent = "";
    panel.clearBtn.hidden = items.length === 0;
    if (!items.length) {
      const empty = el("p", "notif-panel__empty");
      empty.appendChild(Icons.el("bell", { size: 24 }));
      empty.appendChild(el("span", "", "No notifications"));
      body.appendChild(empty);
      return;
    }
    const list = el("ul", "notif-list");
    items.forEach((n) => {
      const li = el("li", "notif-item notif-item--" + n.type);
      li.appendChild(el("span", "notif-item__icon")).appendChild(Icons.el(ICON[n.type] || "info", { size: 16 }));
      li.appendChild(el("span", "notif-item__text", n.text));
      li.appendChild(el("span", "notif-item__time", _stamp(n.at)));
      list.appendChild(li);
    });
    body.appendChild(list);
  }

  /** Rock the bell once (CSS animation on .notif-btn.has-new, see animations.css). */
  function _ring() {
    if (!btn) return;
    btn.classList.remove("has-new");
    void btn.offsetWidth;   // restart the animation if it is already running
    btn.classList.add("has-new");
    btn.addEventListener("animationend", () => btn.classList.remove("has-new"), { once: true });
  }

  /** Remember a notice. Unknown types become "info". */
  function record(text, type) {
    const t = ICON[type] ? type : "info";
    const msg = String(text == null ? "" : text).trim();
    if (!msg) return;
    items.unshift({ text: msg, type: t, at: new Date() });
    if (items.length > MAX) items.length = MAX;
    if (panel) { _render(); } else { unread++; _syncDot(); _ring(); }
  }

  function clear() {
    items.length = 0;
    unread = 0;
    _syncDot();
    _render();
  }

  function isOpen() { return !!panel; }
  function count() { return items.length; }

  function _position() {
    const gap = 8;
    const a = btn.getBoundingClientRect();
    const m = panel.el.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const top = Math.min(a.bottom + gap, Math.max(gap, vh - m.height - gap));
    const left = Math.min(Math.max(gap, a.right - m.width), vw - m.width - gap);
    panel.el.style.top = Math.round(top) + "px";
    panel.el.style.left = Math.round(left) + "px";
  }

  function close(restoreFocus) {
    if (!panel) return;
    const { el: node, cleanup } = panel;
    panel = null;
    cleanup();
    node.remove();
    if (btn) {
      btn.setAttribute("aria-expanded", "false");
      if (restoreFocus !== false && document.contains(btn)) btn.focus();
    }
  }

  function open() {
    if (panel || !btn) return;
    if (window.Menu && Menu.isOpen()) Menu.close({ restoreFocus: false });
    const node = el("div", "notif-panel");
    node.setAttribute("role", "dialog");
    node.setAttribute("aria-label", "Notifications");
    node.tabIndex = -1;

    const head = el("div", "notif-panel__head");
    head.appendChild(el("h2", "notif-panel__title", "Notifications"));
    const clearBtn = el("button", "btn btn--ghost btn--sm notif-panel__clear", "Clear all");
    clearBtn.type = "button";
    clearBtn.addEventListener("click", () => { clear(); node.focus(); });
    head.appendChild(clearBtn);
    node.appendChild(head);

    const body = el("div", "notif-panel__body");
    node.appendChild(body);

    document.body.appendChild(node);
    panel = { el: node, body, clearBtn, cleanup: null };
    btn.setAttribute("aria-expanded", "true");
    unread = 0;
    _syncDot();
    _render();
    _position();

    node.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(true); }
      else if (e.key === "Tab") {
        // Non-modal: leaving the panel with Tab closes it and lets focus continue from the bell
        const f = Array.from(node.querySelectorAll("button:not([hidden])"));
        const first = f[0], last = f[f.length - 1];
        const a = document.activeElement;
        if (!f.length || (!e.shiftKey && a === last) || (e.shiftKey && (a === first || a === node))) close(true);
      }
    });
    const onDown = (e) => { if (!node.contains(e.target) && !btn.contains(e.target)) close(false); };
    const onViewport = (e) => { if (e && e.target && node.contains(e.target)) return; close(false); };
    document.addEventListener("pointerdown", onDown, true);
    window.addEventListener("resize", onViewport);
    window.addEventListener("scroll", onViewport, true);
    panel.cleanup = () => {
      document.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("resize", onViewport);
      window.removeEventListener("scroll", onViewport, true);
    };
    node.focus();
  }

  function init() {
    btn = document.getElementById("notifBtn");
    dot = document.getElementById("notifDot");
    if (!btn) return;
    btn.addEventListener("click", () => { panel ? close(true) : open(); });
    _syncDot();
  }

  init();

  return { record, open, close: () => close(true), isOpen, clear, count };
})();
