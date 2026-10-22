/* ============================================================
   components/menu.js — anchored popover menu with roving focus
   Menu.open({ anchor, label, items, onSelect, align, matchWidth?, className? }) -> handle
     items: [{ id, label, caption?, icon?, checked? }]  (role=menuitemradio)
     matchWidth: panel is at least as wide as the anchor (dropdowns); className: extra modifier class
     onSelect(id) is called after the menu closed and focus returned.
   Menu.close() / Menu.isOpen(anchor?)
   All text is set with textContent. Icons come from js/icons.js.
   ============================================================ */

window.Menu = (function () {

  let _current = null;   // { el, anchor, cleanup }

  function isOpen(anchor) {
    return !!_current && (!anchor || _current.anchor === anchor);
  }

  function close({ restoreFocus = true } = {}) {
    if (!_current) return;
    const { el, anchor, cleanup } = _current;
    _current = null;
    cleanup();
    el.remove();
    if (anchor) {
      anchor.setAttribute("aria-expanded", "false");
      if (restoreFocus && document.contains(anchor)) anchor.focus();
    }
  }

  function _position(menu, anchor, align) {
    const gap = 8;
    const a = anchor.getBoundingClientRect();
    const m = menu.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;

    // Prefer below; flip above when there is not enough room
    const below = a.bottom + gap + m.height <= vh - gap;
    const top = below ? a.bottom + gap : Math.max(gap, a.top - gap - m.height);

    let left = align === "start" ? a.left : a.right - m.width;
    left = Math.min(Math.max(gap, left), vw - m.width - gap);

    menu.style.top = Math.round(top) + "px";
    menu.style.left = Math.round(left) + "px";
  }

  function open({ anchor, label = "", items = [], onSelect, align = "end", matchWidth = false, className = "" } = {}) {
    if (!anchor || !items.length) return null;
    close({ restoreFocus: false });

    const menu = document.createElement("div");
    menu.className = "menu" + (className ? " " + String(className).replace(/[^\w\- ]/g, "") : "");
    menu.setAttribute("role", "menu");
    if (label) menu.setAttribute("aria-label", label);
    // Dropdown panels (the workspace-mode selector) are as wide as their trigger
    // (never narrower than the default panel: a collapsed rail trigger is only 40px wide)
    if (matchWidth) menu.style.minWidth = Math.max(176, Math.round(anchor.getBoundingClientRect().width)) + "px";

    const buttons = items.map((item) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "menu__item";
      b.setAttribute("role", "menuitemradio");
      b.setAttribute("aria-checked", String(!!item.checked));
      b.tabIndex = -1;
      b.dataset.menuId = item.id;

      if (item.icon) b.appendChild(Icons.el(item.icon, { size: 20 }));

      const text = document.createElement("span");
      text.className = "menu__text";
      const lab = document.createElement("span");
      lab.textContent = item.label;
      text.appendChild(lab);
      if (item.caption) {
        const cap = document.createElement("span");
        cap.className = "menu__caption";
        cap.textContent = item.caption;
        text.appendChild(cap);
      }
      b.appendChild(text);
      b.appendChild(Icons.el("check", { size: 16, cls: "menu__check" }));

      b.addEventListener("click", () => {
        close();
        if (onSelect) onSelect(item.id);
      });
      menu.appendChild(b);
      return b;
    });

    document.body.appendChild(menu);
    anchor.setAttribute("aria-expanded", "true");
    _position(menu, anchor, align);

    const focusAt = (i) => {
      const n = buttons.length;
      buttons[((i % n) + n) % n].focus();
    };

    menu.addEventListener("keydown", (e) => {
      const i = buttons.indexOf(document.activeElement);
      switch (e.key) {
        case "ArrowDown": e.preventDefault(); focusAt(i + 1); break;
        case "ArrowUp":   e.preventDefault(); focusAt(i - 1); break;
        case "Home":      e.preventDefault(); focusAt(0); break;
        case "End":       e.preventDefault(); focusAt(buttons.length - 1); break;
        case "Escape":    e.preventDefault(); e.stopPropagation(); close(); break;
        case "Tab":       close({ restoreFocus: false }); break;
        default: break;
      }
    });

    const onPointerDown = (e) => {
      if (menu.contains(e.target) || anchor.contains(e.target)) return;
      close({ restoreFocus: false });
    };
    const onViewport = () => close({ restoreFocus: false });
    document.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("resize", onViewport);
    window.addEventListener("scroll", onViewport, true);

    _current = {
      el: menu,
      anchor,
      cleanup() {
        document.removeEventListener("pointerdown", onPointerDown, true);
        window.removeEventListener("resize", onViewport);
        window.removeEventListener("scroll", onViewport, true);
      },
    };

    // Focus the selected item (or the first one)
    const selected = buttons.findIndex((b) => b.getAttribute("aria-checked") === "true");
    focusAt(selected === -1 ? 0 : selected);
    return { close };
  }

  return { open, close, isOpen };
})();
