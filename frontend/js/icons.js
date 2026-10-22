/* ============================================================
   icons.js — icon helper over the inline SVG sprite in index.html
   Lucide icons (ISC license, see /assets/LICENSE-lucide.txt).
   Icons.html("copy", {size: 16}) -> '<svg class="icon icon--16" ...><use href="#i-copy"/></svg>'

   Unknown names are returned as escaped text, so callers that still
   pass legacy emoji (app.js STAGE_LABELS, toast strings) keep working.
   A few well-known legacy emoji are mapped to their icon.
   ============================================================ */

window.Icons = (function () {

  const SIZES = { 16: "icon--16", 20: "", 24: "icon--24", 32: "icon--32" };

  // Legacy emoji still sent by app.js (view-contract C8) -> icon names
  const LEGACY = {
    "✏": "pencil",          // pencil
    "🔍": "file-search", // magnifier
    "🌐": "globe",       // globe
    "🤖": "sparkles",    // robot
    "✅": "circle-check",      // check mark button
    "⚠": "circle-alert",      // warning
    "❌": "circle-alert",      // cross mark
    "ℹ": "info",              // information
  };

  const _present = {};

  function _esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Resolve a name or legacy emoji to a sprite name, or null. */
  function resolve(name) {
    if (typeof name !== "string" || !name) return null;
    const key = name.replace(/️/g, "").trim();
    const mapped = LEGACY[key] || key;
    if (!/^[a-z0-9-]+$/.test(mapped)) return null;
    if (_present[mapped] === undefined) {
      _present[mapped] = !!document.getElementById("i-" + mapped);
    }
    return _present[mapped] ? mapped : null;
  }

  function has(name) { return resolve(name) !== null; }

  function _classes(size, cls) {
    const parts = ["icon"];
    if (SIZES[size]) parts.push(SIZES[size]);
    if (cls) parts.push(String(cls).replace(/[^\w\- ]/g, ""));
    return parts.join(" ");
  }

  /**
   * HTML string for an icon. Decorative by default (aria-hidden); give the
   * surrounding control an accessible name instead.
   */
  function html(name, { size = 20, cls = "" } = {}) {
    const id = resolve(name);
    if (!id) return name == null ? "" : _esc(name);
    return `<svg class="${_classes(size, cls)}" aria-hidden="true" focusable="false"><use href="#i-${id}"/></svg>`;
  }

  /** Same as html() but returns a Node (SVG element, or a text node for unknown names). */
  function el(name, { size = 20, cls = "" } = {}) {
    const id = resolve(name);
    if (!id) return document.createTextNode(name == null ? "" : String(name));
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", _classes(size, cls));
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    const use = document.createElementNS(NS, "use");
    use.setAttribute("href", "#i-" + id);
    svg.appendChild(use);
    return svg;
  }

  return { html, el, has };
})();
