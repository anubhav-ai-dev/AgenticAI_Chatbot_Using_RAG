/* ============================================================
   components/dialog.js — native <dialog> helpers
   Dialog.open(dialogEl, { initialFocus })   show modally, remember invoker
   Dialog.close(dialogEl)                    close; focus returns to invoker
   Dialog.confirm({ title, message, confirmLabel, cancelLabel, destructive })
       -> Promise<boolean>   true on confirm; false on cancel / Esc / backdrop.
   showModal() provides the top layer, focus trap, Esc and inert background.
   All text is set with textContent.
   ============================================================ */

window.Dialog = (function () {

  const _invoker = new WeakMap();     // dialog -> element to refocus
  const _wired = new WeakSet();       // dialogs with our listeners attached
  let _pending = null;                // { dlg, resolve } of the open confirm

  function _wire(dlg) {
    if (_wired.has(dlg)) return;
    _wired.add(dlg);

    // A drag that starts inside .modal (e.g. selecting input text) and ends on the
    // backdrop dispatches click on the dialog itself. Swallow that click so the
    // controller's "click on backdrop closes" test (e.target === dialog) only sees
    // real backdrop clicks.
    let downInside = false;
    dlg.addEventListener("pointerdown", (e) => {
      downInside = !!(e.target.closest && e.target.closest(".modal"));
    }, true);
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg && downInside) {
        e.stopImmediatePropagation();
      }
      downInside = false;
    }, true);

    dlg.addEventListener("close", () => {
      const back = _invoker.get(dlg);
      _invoker.delete(dlg);
      if (back && document.contains(back) && typeof back.focus === "function") back.focus();
    });
  }

  function open(dlg, { initialFocus } = {}) {
    if (!dlg || dlg.open) return;
    _wire(dlg);
    _invoker.set(dlg, document.activeElement);
    if (typeof dlg.showModal === "function") {
      dlg.showModal();
    } else {
      // Very old browsers: non-modal fallback (no top layer / trap)
      dlg.setAttribute("open", "");
      dlg.addEventListener("keydown", function esc(e) {
        if (e.key === "Escape") { dlg.removeEventListener("keydown", esc); close(dlg); }
      });
    }
    const target = typeof initialFocus === "string" ? dlg.querySelector(initialFocus) : initialFocus;
    if (target && typeof target.focus === "function") target.focus();
  }

  function close(dlg) {
    if (!dlg || !dlg.open) return;
    if (typeof dlg.close === "function") {
      dlg.close();
    } else {
      dlg.removeAttribute("open");
      dlg.dispatchEvent(new Event("close"));
    }
  }

  /** Accessible replacement for window.confirm(). One at a time: a new call cancels the open one. */
  function confirm({ title = "Are you sure?", message = "", confirmLabel = "Confirm",
                     cancelLabel = "Cancel", destructive = false } = {}) {

    let carry = null;   // invoker of a replaced confirm, so focus still returns to the real trigger
    if (_pending) {
      const prev = _pending;
      carry = _invoker.get(prev.dlg) || null;
      _pending = null;
      prev.resolve(false);
      prev.dlg.remove();
    }

    // No <dialog> support: fall back to the native prompt with the same contract
    const probe = document.createElement("dialog");
    if (typeof probe.showModal !== "function") {
      return Promise.resolve(window.confirm(message ? `${title}\n\n${message}` : title));
    }

    return new Promise((resolve) => {
      const uid = "confirm-" + Utils.uuid();
      const dlg = document.createElement("dialog");
      dlg.className = "dialog";
      dlg.setAttribute("aria-modal", "true");
      dlg.setAttribute("aria-labelledby", uid + "-title");

      const modal = document.createElement("div");
      modal.className = "modal modal--sm modal--confirm";

      const h = document.createElement("h2");
      h.className = "modal__title";
      h.id = uid + "-title";
      h.textContent = title;
      const header = document.createElement("div");
      header.className = "modal__header";
      header.appendChild(h);
      modal.appendChild(header);

      if (message) {
        const p = document.createElement("p");
        p.className = "modal__text";
        p.id = uid + "-desc";
        p.textContent = message;
        dlg.setAttribute("aria-describedby", p.id);
        modal.appendChild(p);
      }

      const actions = document.createElement("div");
      actions.className = "modal__actions";
      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "btn btn--secondary";
      cancelBtn.textContent = cancelLabel;
      const okBtn = document.createElement("button");
      okBtn.type = "button";
      okBtn.className = "btn " + (destructive ? "btn--danger" : "btn--primary");
      okBtn.textContent = confirmLabel;
      actions.appendChild(cancelBtn);
      actions.appendChild(okBtn);
      modal.appendChild(actions);
      dlg.appendChild(modal);

      let result = false;
      const entry = { dlg, resolve };
      _pending = entry;

      cancelBtn.addEventListener("click", () => { result = false; close(dlg); });
      okBtn.addEventListener("click", () => { result = true; close(dlg); });
      // Backdrop click cancels (never confirms)
      dlg.addEventListener("click", (e) => { if (e.target === dlg) { result = false; close(dlg); } });
      // Esc fires "cancel" then "close"; result stays false
      dlg.addEventListener("close", () => {
        if (_pending === entry) _pending = null;
        dlg.remove();
        resolve(result);
      });

      document.body.appendChild(dlg);
      // Safe default: focus Cancel, so Enter never confirms a destructive action by accident
      open(dlg, { initialFocus: cancelBtn });
      if (carry) _invoker.set(dlg, carry);
    });
  }

  return { open, close, confirm };
})();
