/* ============================================================
   theme.js — theme preference (light | dark | system)
   Storage contract: view-contract.md section 6.
     localStorage.theme        "light" | "dark" | "system"  (default "system")
     localStorage.themeVersion "2" once the legacy value was migrated
   DOM outputs: <html data-theme="light|dark"> (resolved),
                data-theme-pref, style.colorScheme, <meta name=theme-color>.
   The inline snippet in index.html <head> does the first paint; this
   module keeps it in sync (OS changes, other tabs, UI.setTheme).
   ============================================================ */

window.Theme = (function () {

  const KEY = "theme";
  const VERSION_KEY = "themeVersion";
  const VERSION = "2";
  const PREFS = ["light", "dark", "system"];
  const META_COLOR = { light: "#FFFFFF", dark: "#0A1024" };  // meta cannot read CSS variables

  const _listeners = [];
  let _pref = "system";
  let _mql = null;
  let _inited = false;

  function _store(key, value) {
    try { localStorage.setItem(key, value); } catch (_) {}
  }

  function _load(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
  }

  /** Legacy default was persisted as "dark" on every load, so only "light" is a real choice. */
  function _migrate() {
    if (_load(VERSION_KEY) === VERSION) return;
    const stored = _load(KEY);
    _store(KEY, stored === "light" ? "light" : "system");
    _store(VERSION_KEY, VERSION);
  }

  function _normalize(value) {
    return PREFS.indexOf(value) !== -1 ? value : "system";
  }

  function _systemDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function resolved() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function get() { return _pref; }

  function _apply() {
    const dark = _pref === "dark" || (_pref === "system" && _systemDark());
    const theme = dark ? "dark" : "light";
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.dataset.themePref = _pref;
    root.style.colorScheme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", META_COLOR[theme]);
    _listeners.slice().forEach((fn) => { try { fn(_pref, theme); } catch (_) {} });
  }

  /** Persist and apply a preference ("light" | "dark" | "system"). */
  function set(pref) {
    _pref = _normalize(pref);
    _store(KEY, _pref);
    _store(VERSION_KEY, VERSION);
    _apply();
  }

  /** Subscribe to preference / resolved-theme changes. Returns an unsubscribe function. */
  function onChange(fn) {
    _listeners.push(fn);
    return () => {
      const i = _listeners.indexOf(fn);
      if (i !== -1) _listeners.splice(i, 1);
    };
  }

  /** Idempotent: migrates storage, applies the preference and wires OS / storage listeners once. */
  function init() {
    _migrate();
    _pref = _normalize(_load(KEY));
    _apply();
    if (_inited) return;
    _inited = true;

    if (window.matchMedia) {
      _mql = window.matchMedia("(prefers-color-scheme: dark)");
      const onOs = () => { if (_pref === "system") _apply(); };
      if (_mql.addEventListener) _mql.addEventListener("change", onOs);
      else if (_mql.addListener) _mql.addListener(onOs);
    }

    window.addEventListener("storage", (e) => {
      if (e.key !== KEY) return;
      _pref = _normalize(e.newValue);
      _apply();
    });
  }

  init();

  return { get, set, resolved, onChange, init };
})();
