/* ============================================================
   components/model-select.js — listbox popover over the native <select>
   ModelSelect.init(selectEl)   idempotent; called by ui.js

   The native <select id="modelSelect"> stays the SOURCE OF TRUTH:
     - choosing an option sets select.value and dispatches a real, bubbling
       "change" event (app.js listens to it);
     - changes made from outside (app.js `sel.value = x`, option rebuilds in
       UI.renderModelSelect) are mirrored into the trigger. `value` and
       `selectedIndex` assignments fire no event, so both setters are wrapped
       on this one element; a MutationObserver covers option changes.
   Only data present in the <select> is shown: option label/description/reason
   (data-label, data-description, data-reason written by UI.renderModelSelect, or
   the option text), provider (optgroup label), disabled. Nothing is invented.
   If this script never runs, the native select simply stays visible.
   All text is set with textContent.
   ============================================================ */

window.ModelSelect = (function () {

  let sel = null;
  let wrap = null;
  let trigger = null;
  let labelEl = null;
  let iconSlot = null;
  let subEl = null;          // provider line under the label (top-bar pill)
  let anchor = null;         // element the open list is positioned against (trigger or a bound chip)
  const chips = [];          // composer chips mirroring the selection (bindChip)
  let pop = null;            // { el, items, optEls, active, cleanup }
  let seq = 0;
  let typeBuf = "";
  let typeTimer = null;
  let syncQueued = false;
  let errMsg = "";          // set by setError (C7); "" = no failure
  let retryBtn = null;
  let retryCb = null;

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  // ── Read model data from the native select ────────────────

  function _item(opt, group) {
    const text = opt.textContent || "";
    return {
      value: opt.value,
      label: opt.dataset.label || text.trim() || opt.value || "Unknown model",
      desc: opt.dataset.description || "",
      reason: opt.dataset.reason || opt.title || "",
      disabled: !!opt.disabled,
      group: group || "",
    };
  }

  function _read() {
    const items = [];
    let placeholder = false;
    Array.from(sel.children).forEach((ch) => {
      const opts = ch.tagName === "OPTGROUP" ? Array.from(ch.children) : [ch];
      opts.forEach((o) => {
        if (o.tagName !== "OPTION") return;
        if (o.value === "") { placeholder = true; return; }
        items.push(_item(o, ch.tagName === "OPTGROUP" ? ch.label : ""));
      });
    });
    let state = "ready";
    if (!items.length) state = placeholder ? "loading" : "empty";
    return { items, state };
  }

  // ── Trigger ───────────────────────────────────────────────

  function _setIcon(name, spin) {
    iconSlot.textContent = "";
    iconSlot.classList.remove("is-logo");
    iconSlot.appendChild(Icons.el(name, { size: 16, cls: spin ? "icon--spin" : "" }));
  }

  /** Provider "logo": a letter tile from the provider name already present in the data (optgroup label). */
  function _initial(item) {
    const src = (item && (item.group || item.label)) || "";
    return src.trim().charAt(0).toUpperCase();
  }

  function _setLogo(slot, item) {
    slot.textContent = _initial(item);
    slot.classList.add("is-logo");
  }

  function _syncChips(text, item, state) {
    chips.forEach((c) => {
      c.label.textContent = text;
      c.logo.textContent = "";
      c.logo.classList.toggle("is-logo", !!item);
      if (item) c.logo.textContent = _initial(item);
      else c.logo.appendChild(Icons.el(state === "loading" ? "loader-circle" : "circle-alert", { size: 16, cls: state === "loading" ? "icon--spin" : "" }));
      c.btn.disabled = state !== "ready" || sel.disabled;
      c.btn.setAttribute("aria-label", item ? "AI model: " + item.label + ". Change model" : text);
      c.btn.classList.toggle("is-error", state !== "ready" && state !== "loading");
    });
  }

  function _syncTrigger() {
    const { items, state } = _read();
    let text;
    let name;
    let sub = "";
    let cur = null;
    trigger.removeAttribute("aria-busy");
    if (errMsg && state !== "ready") {
      text = "Models unavailable"; name = "Couldn't load models. " + errMsg;
      _setIcon("circle-alert", false);
      trigger.title = errMsg;
    } else if (state === "loading") {
      text = "Loading models…"; name = text;
      trigger.setAttribute("aria-busy", "true");
      _setIcon("loader-circle", true);
    } else if (state === "empty") {
      text = "No models available"; name = text;
      _setIcon("circle-alert", false);
    } else {
      cur = items.find((i) => i.value === sel.value) || null;
      text = cur ? cur.label : "Select a model";
      sub = cur ? cur.group : "";
      name = cur ? "AI model: " + cur.label + (cur.group ? " (" + cur.group + ")" : "") : "AI model: none selected";
      if (cur) _setLogo(iconSlot, cur); else _setIcon("sparkles", false);
      trigger.title = text;
    }
    if (state !== "ready" && !errMsg) trigger.title = "";
    labelEl.textContent = text;
    subEl.textContent = sub;
    subEl.hidden = !sub;
    trigger.classList.toggle("has-sub", !!sub);
    _syncChips(text, cur, state);
    trigger.setAttribute("aria-label", name);
    trigger.disabled = state !== "ready" || sel.disabled;
    if (trigger.disabled && pop) close(false);
    if (retryBtn) {
      retryBtn.hidden = !errMsg || !retryCb;
      wrap.classList.toggle("select-wrap--error", !!errMsg);
    }
    return { items, state };
  }

  function _scheduleSync() {
    if (syncQueued) return;
    syncQueued = true;
    Promise.resolve().then(() => {
      syncQueued = false;
      if (pop) close(false);   // options changed under an open list
      _syncTrigger();
    });
  }

  // ── Popover ───────────────────────────────────────────────

  function _position() {
    const gap = 8;
    const a = (anchor || trigger).getBoundingClientRect();
    const m = pop.el.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const below = a.bottom + gap + m.height <= vh - gap;
    const top = below ? a.bottom + gap : Math.max(gap, a.top - gap - m.height);
    let left = a.right - m.width;
    left = Math.min(Math.max(gap, left), vw - m.width - gap);
    pop.el.style.top = Math.round(top) + "px";
    pop.el.style.left = Math.round(left) + "px";
  }

  function _setActive(i, scroll) {
    if (!pop) return;
    const n = pop.items.length;
    if (!n) return;
    i = Math.min(Math.max(i, 0), n - 1);
    if (pop.active >= 0) pop.optEls[pop.active].classList.remove("is-active");
    pop.active = i;
    const o = pop.optEls[i];
    o.classList.add("is-active");
    pop.el.setAttribute("aria-activedescendant", o.id);
    if (scroll !== false) o.scrollIntoView({ block: "nearest" });
  }

  function _choose(i) {
    const item = pop && pop.items[i];
    if (!item || item.disabled) return;
    close(true);
    if (sel.value !== item.value) {
      sel.value = item.value;
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function _typeahead(ch) {
    clearTimeout(typeTimer);
    typeBuf += ch.toLowerCase();
    typeTimer = setTimeout(() => { typeBuf = ""; }, 600);
    const n = pop.items.length;
    const cycle = /^(.)\1*$/.test(typeBuf);          // "aaa" cycles through "a" matches
    const needle = cycle ? typeBuf[0] : typeBuf;
    const start = pop.active + (cycle || typeBuf.length === 1 ? 1 : 0);
    for (let k = 0; k < n; k++) {
      const i = (start + k) % n;
      const it = pop.items[i];
      if (!it.disabled && it.label.toLowerCase().startsWith(needle)) { _setActive(i); return; }
    }
  }

  /** The visible element to anchor to when none was given (top-bar pill, else the first visible chip). */
  function _defaultAnchor() {
    if (trigger.getClientRects().length) return trigger;
    const c = chips.find((x) => x.btn.getClientRects().length);
    return c ? c.btn : trigger;
  }

  function _setExpanded(on) {
    trigger.setAttribute("aria-expanded", on ? "true" : "false");
    chips.forEach((c) => c.btn.setAttribute("aria-expanded", on && anchor === c.btn ? "true" : "false"));
  }

  function open(anchorEl) {
    if (pop || trigger.disabled) return;
    anchor = anchorEl && anchorEl.getClientRects && anchorEl.getClientRects().length ? anchorEl : _defaultAnchor();
    const { items, state } = _read();
    if (state !== "ready") return;

    const id = "ms-" + (++seq);
    const list = el("div", "model-list");
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "AI model");
    list.tabIndex = -1;

    const optEls = [];
    let groupEl = null;
    let lastGroup = null;
    items.forEach((it, i) => {
      if (it.group !== lastGroup) {
        lastGroup = it.group;
        if (it.group) {
          groupEl = el("div", "model-list__group");
          groupEl.setAttribute("role", "group");
          const gl = el("div", "model-list__group-label", it.group);
          gl.id = id + "-g" + i;
          groupEl.setAttribute("aria-labelledby", gl.id);
          groupEl.appendChild(gl);
          list.appendChild(groupEl);
        } else {
          groupEl = null;
        }
      }
      const o = el("div", "model-option");
      o.id = id + "-o" + i;
      o.setAttribute("role", "option");
      const selected = it.value === sel.value;
      o.setAttribute("aria-selected", String(selected));
      if (selected) o.classList.add("is-selected");
      if (it.disabled) { o.setAttribute("aria-disabled", "true"); o.classList.add("is-disabled"); }

      const body = el("span", "model-option__body");
      body.appendChild(el("span", "model-option__label", it.label));
      let sub = it.desc;
      if (it.disabled) sub = "Unavailable" + (it.reason ? " — " + it.reason : "");
      if (sub) body.appendChild(el("span", "model-option__desc", sub));
      o.appendChild(body);
      o.appendChild(Icons.el("check", { size: 16, cls: "model-option__check" }));

      o.addEventListener("mousemove", () => { if (pop && pop.active !== i) _setActive(i, false); });
      o.addEventListener("click", () => _choose(i));
      (groupEl || list).appendChild(o);
      optEls.push(o);
    });

    document.body.appendChild(list);
    _setExpanded(true);
    list.id = id;
    anchor.setAttribute("aria-controls", id);
    pop = { el: list, items, optEls, active: -1 };
    _position();

    list.addEventListener("keydown", _onListKey);

    const onDown = (e) => {
      if (list.contains(e.target) || trigger.contains(e.target) || chips.some((c) => c.btn.contains(e.target))) return;
      close(false);
    };
    const onViewport = (e) => { if (e && e.target && list.contains(e.target)) return; close(false); };
    document.addEventListener("pointerdown", onDown, true);
    window.addEventListener("resize", onViewport);
    window.addEventListener("scroll", onViewport, true);
    list.addEventListener("focusout", (e) => {
      // Focus left the list without a key (e.g. clicking elsewhere): close without stealing focus
      if (pop && !list.contains(e.relatedTarget) && e.relatedTarget !== anchor) close(false);
    });
    pop.cleanup = () => {
      document.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("resize", onViewport);
      window.removeEventListener("scroll", onViewport, true);
    };

    list.focus();
    const cur = items.findIndex((it) => it.value === sel.value);
    _setActive(cur >= 0 ? cur : Math.max(0, items.findIndex((it) => !it.disabled)));
  }

  function close(restoreFocus) {
    if (!pop) return;
    const { el: node, cleanup } = pop;
    pop = null;
    cleanup();
    clearTimeout(typeTimer);
    typeBuf = "";
    node.remove();
    const back = anchor || trigger;
    _setExpanded(false);
    back.removeAttribute("aria-controls");
    if (restoreFocus !== false && document.contains(back)) back.focus();
  }

  function _onListKey(e) {
    if (!pop) return;
    const n = pop.items.length;
    switch (e.key) {
      case "ArrowDown": e.preventDefault(); _setActive(Math.min(pop.active + 1, n - 1)); break;
      case "ArrowUp":   e.preventDefault(); _setActive(Math.max(pop.active - 1, 0)); break;
      case "Home":      e.preventDefault(); _setActive(0); break;
      case "End":       e.preventDefault(); _setActive(n - 1); break;
      case "PageDown":  e.preventDefault(); _setActive(Math.min(pop.active + 5, n - 1)); break;
      case "PageUp":    e.preventDefault(); _setActive(Math.max(pop.active - 5, 0)); break;
      case "Enter":
      case " ":         e.preventDefault(); _choose(pop.active); break;
      case "Escape":    e.preventDefault(); e.stopPropagation(); close(true); break;
      case "Tab":       close(false); break;
      default:
        if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          _typeahead(e.key);
        }
    }
  }

  // ── Init ──────────────────────────────────────────────────

  function _wrapProp(name) {
    const desc = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, name);
    if (!desc || !desc.get || !desc.set) return;
    Object.defineProperty(sel, name, {
      configurable: true,
      get() { return desc.get.call(this); },
      set(v) { desc.set.call(this, v); _syncTrigger(); },
    });
  }

  function init(select) {
    select = select || document.getElementById("modelSelect");
    if (!select || sel === select) return;
    sel = select;
    wrap = sel.parentNode;

    trigger = el("button", "model-trigger");
    trigger.type = "button";
    trigger.id = "modelTrigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    iconSlot = el("span", "model-trigger__icon");
    labelEl = el("span", "model-trigger__label");
    subEl = el("span", "model-trigger__sub");
    subEl.hidden = true;
    const textWrap = el("span", "model-trigger__text");
    textWrap.appendChild(labelEl);
    textWrap.appendChild(subEl);
    trigger.appendChild(iconSlot);
    trigger.appendChild(textWrap);
    trigger.appendChild(Icons.el("chevron-down", { size: 16, cls: "model-trigger__chevron" }));
    wrap.insertBefore(trigger, sel);
    wrap.classList.add("select-wrap--enhanced");

    // The native control stays in the DOM (value + change), out of sight and out of the tab order
    sel.tabIndex = -1;
    sel.setAttribute("aria-hidden", "true");

    trigger.addEventListener("click", () => { pop ? close(true) : open(trigger); });
    trigger.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); open(trigger); }
    });

    _wrapProp("value");
    _wrapProp("selectedIndex");
    sel.addEventListener("change", _syncTrigger);
    new MutationObserver(_scheduleSync).observe(sel, {
      childList: true, subtree: true, attributes: true, characterData: true,
    });

    _syncTrigger();
  }

  /**
   * Mirror the selection into a composer chip and let it open the same listbox.
   * btn must contain .model-chip__logo and .model-chip__label (index.html). Safe to call before init().
   */
  function bindChip(btn) {
    if (!btn || chips.some((c) => c.btn === btn)) return;
    const chip = { btn, logo: btn.querySelector(".model-chip__logo"), label: btn.querySelector(".model-chip__label") };
    if (!chip.logo || !chip.label) return;
    chips.push(chip);
    btn.addEventListener("click", () => { pop ? close(true) : open(btn); });
    btn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); open(btn); }
    });
    if (sel) _syncTrigger();
  }

  /**
   * C7 failure state. msg: short reason (textContent); onRetry: called by the Retry button
   * (button is not rendered without a callback). Distinct from "No models available".
   */
  function setError(msg, onRetry) {
    if (!sel) return;
    errMsg = msg ? String(msg) : "Models unavailable";
    retryCb = typeof onRetry === "function" ? onRetry : null;
    _ensureRetry();
    _syncTrigger();
  }

  function clearError() {
    if (!sel) return;
    errMsg = "";
    retryCb = null;
    _syncTrigger();
  }

  function _ensureRetry() {
    if (retryBtn) return;
    retryBtn = el("button", "model-retry btn btn--secondary btn--sm");
    retryBtn.type = "button";
    retryBtn.setAttribute("aria-label", "Retry loading models");
    retryBtn.title = "Retry loading models";
    retryBtn.appendChild(Icons.el("rotate-ccw", { size: 16 }));
    retryBtn.appendChild(el("span", "model-retry__text", "Retry"));
    retryBtn.hidden = true;
    retryBtn.addEventListener("click", () => { if (retryCb) retryCb(); });
    wrap.insertBefore(retryBtn, sel);
  }

  return { init, bindChip, setError, clearError, open, close: () => close(true), isOpen: () => !!pop };
})();
