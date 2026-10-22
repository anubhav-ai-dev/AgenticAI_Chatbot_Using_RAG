/* ============================================================
   components/ai-viz.js — AIVisualization (design-spec section 8)
   A small decorative status mark: idle | processing | responding | error.
   View only: it never fetches and owns no app state. The state is a single
   data-state attribute; all motion is CSS (transform / opacity), so there is
   no JavaScript in the animation loop (no timers, no requestAnimationFrame).
   aria-hidden: the meaning is carried by the activity steps and #liveStatus.
   ============================================================ */

window.AIViz = (function () {
  const NS = "http://www.w3.org/2000/svg";
  const STATES = ["idle", "processing", "responding", "error"];

  // 7 nodes (halo, ring, arc, dot, core, alert + the spin group), well under the 8-node budget.
  // Geometry: viewBox 96, ring r=38 (pathLength=100 so dasharray values are percentages).
  const MARKUP =
    '<circle class="ai-viz__halo" cx="48" cy="48" r="44"/>' +
    '<circle class="ai-viz__ring" cx="48" cy="48" r="38"/>' +
    '<g class="ai-viz__spin">' +
      '<circle class="ai-viz__arc" cx="48" cy="48" r="38" pathLength="100"/>' +
      '<circle class="ai-viz__dot" cx="48" cy="10" r="3"/>' +
    '</g>' +
    '<path class="ai-viz__core" d="M48 31c1.2 9.5 7.5 15.8 17 17-9.5 1.2-15.8 7.5-17 17-1.2-9.5-7.5-15.8-17-17 9.5-1.2 15.8-7.5 17-17z"/>' +
    '<path class="ai-viz__alert" d="M48 34v17M48 62h.01"/>';

  /**
   * Welcome hero (reference look): glossy orb with glowing eyes, two gradient orbit rings with a planet
   * each, three satellite chips (document, globe, code) and sparkles. Plain elements, all motion is CSS
   * (transform / opacity). Same data-state contract as the SVG variants, so UI.setVizState works.
   * The eyes follow the pointer through --look-x / --look-y (see ui.js _wireHeroLook); the markup
   * itself has no timers and no randomness.
   */
  function _hero(state) {
    const div = document.createElement("div");
    div.className = "ai-viz ai-viz--hero";
    div.setAttribute("aria-hidden", "true");
    div.setAttribute("data-state", STATES.includes(state) ? state : "idle");
    const span = (cls, kids) => {
      const n = document.createElement("span");
      n.className = cls;
      (kids || []).forEach((k) => n.appendChild(k));
      return n;
    };
    const sat = (cls, icon) => span("hero__sat " + cls, [Icons.el(icon, { size: 20 })]);
    const orbit = (mod) => span("hero__tilt hero__tilt--" + mod, [
      span("hero__orbit hero__orbit--" + mod, [span("hero__ring"), span("hero__planet")]),
    ]);
    const spark = (mod) => span("hero__spark hero__spark--" + mod, [Icons.el("sparkle", { size: 16 })]);
    div.appendChild(span("hero__glow"));
    div.appendChild(orbit("a"));
    div.appendChild(span("hero__orb", [
      span("hero__face"),
      span("hero__look", [span("hero__eyes", [span("hero__eye"), span("hero__eye")])]),
      span("hero__shine"),
    ]));
    div.appendChild(orbit("b"));
    div.appendChild(sat("hero__sat--doc", "file-text"));
    div.appendChild(sat("hero__sat--globe", "globe"));
    div.appendChild(sat("hero__sat--code", "code-xml"));
    div.appendChild(spark("a"));
    div.appendChild(spark("b"));
    div.appendChild(spark("c"));
    return div;
  }

  /** Build the SVG. `size`: "lg" (96px), "sm" (28px, message avatar) or "hero" (welcome orb scene, a div). */
  function create({ size = "sm", state = "idle" } = {}) {
    if (size === "hero") return _hero(state);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "ai-viz ai-viz--" + (size === "lg" ? "lg" : "sm"));
    svg.setAttribute("viewBox", "0 0 96 96");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    svg.setAttribute("data-state", STATES.includes(state) ? state : "idle");
    svg.innerHTML = MARKUP;   // static, trusted constant
    return svg;
  }

  function _find(target) {
    if (!target) return null;
    if (target.classList && target.classList.contains("ai-viz")) return target;
    return target.querySelector ? target.querySelector(".ai-viz") : null;
  }

  /** Set the state on an AIViz svg, or on a container that holds one. Unknown states are ignored. */
  function setState(target, state) {
    const svg = _find(target);
    if (!svg || !STATES.includes(state)) return false;
    if (svg.getAttribute("data-state") !== state) svg.setAttribute("data-state", state);
    return true;
  }

  function getState(target) {
    const svg = _find(target);
    return svg ? svg.getAttribute("data-state") : null;
  }

  /** Replace the children of `container` with a new instance (keeps any fallback icon until JS runs). */
  function mount(container, opts) {
    if (!container) return null;
    const svg = create(opts);
    container.textContent = "";
    container.appendChild(svg);
    return svg;
  }

  // Pause every instance while the tab is hidden (CSS reads html[data-page-hidden]).
  // One event listener, nothing runs while visible.
  function _syncVisibility() {
    document.documentElement.toggleAttribute("data-page-hidden", document.hidden);
  }
  document.addEventListener("visibilitychange", _syncVisibility);
  _syncVisibility();

  return { create, mount, setState, getState, STATES };
})();
