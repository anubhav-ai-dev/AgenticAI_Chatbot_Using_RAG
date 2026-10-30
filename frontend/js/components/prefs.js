/* ============================================================
   components/prefs.js — view preferences saved in THIS browser only
   (text size, chat spacing, pop-up notices).
   Storage keys are "pref.<name>". Nothing here is sent to the server,
   and the theme has its own module (theme.js).
   Prefs.get(name) / Prefs.set(name, value) / Prefs.onChange(fn) / Prefs.apply()
   Unknown names and values are rejected, so a stale or edited value in
   storage can never reach the DOM.
   ============================================================ */

window.Prefs = (function () {

  const SPEC = {
    fontSize: { values: ["small", "medium", "large"], def: "medium" },
    density:  { values: ["comfortable", "compact"], def: "comfortable" },
    toasts:   { values: ["all", "errors"], def: "all" },
  };
  const KEY = (name) => "pref." + name;
  const _listeners = [];

  function _load(name) {
    try { return localStorage.getItem(KEY(name)); } catch (_) { return null; }
  }

  function get(name) {
    const spec = SPEC[name];
    if (!spec) return null;
    const v = _load(name);
    return spec.values.indexOf(v) !== -1 ? v : spec.def;
  }

  /** Mirror the stored values onto <html> (data-font, data-density). Idempotent. */
  function apply() {
    const root = document.documentElement;
    const font = get("fontSize");
    if (font === "medium") root.removeAttribute("data-font"); else root.dataset.font = font;
    const density = get("density");
    if (density === "comfortable") root.removeAttribute("data-density"); else root.dataset.density = density;
  }

  function set(name, value) {
    const spec = SPEC[name];
    if (!spec || spec.values.indexOf(value) === -1) return false;
    try { localStorage.setItem(KEY(name), value); } catch (_) { /* private mode: the value lasts for this page */ }
    apply();
    _listeners.slice().forEach((fn) => { try { fn(name, value); } catch (_) {} });
    return true;
  }

  /** Subscribe to changes. Returns an unsubscribe function. */
  function onChange(fn) {
    _listeners.push(fn);
    return () => {
      const i = _listeners.indexOf(fn);
      if (i !== -1) _listeners.splice(i, 1);
    };
  }

  // Another tab changed a preference
  window.addEventListener("storage", (e) => {
    if (!e.key || e.key.indexOf("pref.") !== 0) return;
    apply();
    const name = e.key.slice(5);
    _listeners.slice().forEach((fn) => { try { fn(name, get(name)); } catch (_) {} });
  });

  apply();
  return { get, set, onChange, apply };
})();
