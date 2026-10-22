/* ============================================================
   components/radiogroup.js — roving-tabindex radio group
   Markup: <div role="radiogroup"> <button role="radio" aria-checked data-value="…" tabindex> … </div>
   RadioGroup.wire(group, onChange)   click + arrow keys / Home / End (arrows move AND select, per the
                                      ARIA radio pattern); onChange(value) fires for user input only
   RadioGroup.set(group, value)       reflect a value without firing onChange
   RadioGroup.get(group)              current value or null
   View only: it owns no state beyond the DOM attributes.
   ============================================================ */

window.RadioGroup = (function () {

  const radios = (group) => Array.from(group.querySelectorAll('[role="radio"]'));
  const valueOf = (r) => r.dataset.value || "";

  function set(group, value) {
    if (!group) return;
    radios(group).forEach((r) => {
      const on = valueOf(r) === value;
      r.setAttribute("aria-checked", String(on));
      r.tabIndex = on ? 0 : -1;
    });
  }

  function get(group) {
    const on = group && radios(group).find((r) => r.getAttribute("aria-checked") === "true");
    return on ? valueOf(on) : null;
  }

  function wire(group, onChange) {
    if (!group) return;
    const choose = (radio, { focus = false } = {}) => {
      const changed = radio.getAttribute("aria-checked") !== "true";
      set(group, valueOf(radio));
      if (focus) radio.focus();
      if (changed && typeof onChange === "function") onChange(valueOf(radio));
    };

    group.addEventListener("click", (e) => {
      const radio = e.target.closest('[role="radio"]');
      if (radio && group.contains(radio)) choose(radio);
    });

    group.addEventListener("keydown", (e) => {
      const list = radios(group);
      const i = list.indexOf(document.activeElement);
      if (i === -1) return;
      let next = -1;
      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown": next = (i + 1) % list.length; break;
        case "ArrowLeft":
        case "ArrowUp":   next = (i - 1 + list.length) % list.length; break;
        case "Home":      next = 0; break;
        case "End":       next = list.length - 1; break;
        default: return;
      }
      e.preventDefault();
      choose(list[next], { focus: true });
    });
  }

  return { wire, set, get };
})();
