/* ============================================================
   components/motion.js — Motion: the one place GSAP is used.
   Motion.enter()       first welcome entrance (heading, text, cards, composer, orb)
   Motion.welcome()     the same entrance again when a fresh welcome screen returns
   Motion.workspaceChanged()  entrance of whichever panel just became visible
   Hover, focus and the mode menu stay in CSS.
   Rules:
     - reduced motion never loads the library; content stays in its final visible state
     - CSS transitions on transform are suspended only while a timeline is running
       (otherwise they swallow the movement and the screen looks static)
     - a new timeline reverts the previous one
     - clearProps removes inline styles when a tween finishes
     - the idle float is one yoyo on the orb and pauses while the tab is hidden
   View only: no fetch, no application state.
   ============================================================ */

window.Motion = (function () {

  const SRC = "/js/vendor/gsap.min.js";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const root = document.documentElement;

  let _loading = null;
  let _ctx = null;
  let _entered = false;
  let _floatTween = null;

  function _release() { root.classList.remove("motion-pending"); }

  function _load() {
    if (window.gsap) return Promise.resolve(window.gsap);
    if (!_loading) {
      _loading = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = SRC;
        s.async = true;
        s.onload = () => (window.gsap ? resolve(window.gsap) : reject(new Error("gsap missing")));
        s.onerror = () => reject(new Error("gsap failed to load"));
        document.head.appendChild(s);
      });
      _loading.catch(() => { _loading = null; });
    }
    return _loading;
  }

  const $ = (sel) => Array.from(document.querySelectorAll(sel));
  const shown = (el) => el && el.getClientRects().length > 0;

  function _groups() {
    const ws = root.dataset.workspace || "general";
    const panelId = { document: "#wsDocument", research: "#wsResearch", code: "#wsCode" }[ws];
    if (panelId) return { panel: $(panelId + " > *").filter(shown) };
    return {
      viz: $(".welcome__viz").filter(shown),
      badge: $(".welcome__badge").filter(shown),
      title: $(".welcome__title").filter(shown),
      sub: $(".welcome__sub").filter(shown),
      cards: $(".welcome__starters .feature-card").filter(shown),
      caps: $(".capabilities").filter(shown),
    };
  }

  function _stopFloat() {
    if (_floatTween) { _floatTween.kill(); _floatTween = null; }
    root.classList.remove("is-gsap-float");
    const orb = document.querySelector(".hero__orb");
    if (orb && window.gsap) window.gsap.set(orb, { clearProps: "transform" });
  }

  function _revert() {
    if (_ctx) { _ctx.revert(); _ctx = null; }
    root.classList.remove("motion-run");
    _stopFloat();
  }

  /** Gentle 4px float. One tween, paused while the tab is hidden. */
  function _float(gsap) {
    if (_floatTween || reduce.matches) return;
    const orb = document.querySelector(".hero__orb");
    if (!orb) return;
    root.classList.add("is-gsap-float");
    _floatTween = gsap.to(orb, { y: -4, duration: 1.8, ease: "sine.inOut", yoyo: true, repeat: -1 });
  }

  /**
   * Welcome order: orb, heading (16px), supporting text, cards (20px, stagger), composer (12px).
   * About 1.0s including the stagger. fromTo so the start offset is applied before the first paint
   * of the revealed content. clearProps hands the elements back to CSS afterwards.
   */
  function _play(gsap, withComposer) {
    const g = _groups();
    const composer = withComposer ? $(".input-dock .composer").filter(shown) : [];
    const pieces = [].concat(g.panel || [], g.viz || [], g.badge || [], g.title || [], g.sub || [], g.cards || [], g.caps || [], composer);
    if (!pieces.length) { _release(); return; }
    _revert();
    const done = { ease: "power3.out", clearProps: "opacity,transform" };
    _ctx = gsap.context(() => {
      root.classList.add("motion-run");
      if (g.panel) {
        gsap.fromTo(g.panel, { opacity: 0, y: 16 }, Object.assign({ opacity: 1, y: 0, duration: 0.4, stagger: 0.06 }, done));
      } else {
        if (g.viz.length) {
          const vizFrom = { opacity: 0, y: 16 };
          vizFrom.scale = 0.94;
          const vizTo = Object.assign({ opacity: 1, y: 0, duration: 0.55 }, done);
          vizTo.scale = 1;
          gsap.fromTo(g.viz, vizFrom, vizTo);
        }
        if (g.badge.length) gsap.fromTo(g.badge, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.4, delay: 0.04 }, done));
        if (g.title.length) gsap.fromTo(g.title, { opacity: 0, y: 16 }, Object.assign({ opacity: 1, y: 0, duration: 0.5, delay: 0.08 }, done));
        if (g.sub.length) gsap.fromTo(g.sub, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.4, delay: 0.18 }, done));
        if (g.cards.length) gsap.fromTo(g.cards, { opacity: 0, y: 20 }, Object.assign({ opacity: 1, y: 0, duration: 0.45, delay: 0.28, stagger: 0.09 }, done));
        if (g.caps.length) gsap.fromTo(g.caps, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.4, delay: 0.55 }, done));
      }
      if (composer.length) gsap.fromTo(composer, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.4, delay: 0.62 }, done));
      gsap.delayedCall(1.15, () => root.classList.remove("motion-run"));
      _release();
    });
    _float(gsap);
  }

  /** First paint. Later calls are ignored; Motion.welcome() covers a return. */
  function enter() {
    if (reduce.matches) { _release(); return; }
    if (_entered) return;
    _entered = true;
    setTimeout(_release, 1500);
    _load().then((gsap) => _play(gsap, true)).catch(() => { _revert(); _release(); });
  }

  /** A fresh welcome screen (New Chat, or an empty history after the screen was hidden). */
  function welcome() {
    if (reduce.matches) { _release(); return; }
    if (!window.gsap) { _load().then((gsap) => _play(gsap, true)).catch(_release); return; }
    _play(window.gsap, true);
  }

  /** The workspace changed: the panel now on screen eases in. */
  function workspaceChanged() {
    if (reduce.matches) { _revert(); _release(); return; }
    if (!window.gsap) { _load().catch(() => {}); return; }
    const general = (root.dataset.workspace || "general") === "general";
    _play(window.gsap, general);
  }

  let _introCtx = null;
  let _introPlayed = false;

  /**
   * One introduction entrance. A later call (theme change, hash return) does not replay it.
   * The heading starts partly visible so the words stay readable. Scale is applied on the
   * vars object, not as a layout property. Reduced motion never reaches this function.
   */
  function _playIntro(gsap) {
    if (_introPlayed) return;
    const header = document.querySelector(".intro__header");
    const badge = document.querySelector(".intro__badge");
    const lines = Array.from(document.querySelectorAll(".intro__title-line"));
    const lead = document.querySelector(".intro__lead");
    const cta = document.querySelector(".intro__cta");
    const feats = Array.from(document.querySelectorAll(".intro__feature"));
    const stage = document.querySelector(".intro__stage");
    if (!header && !lines.length && !stage) return;
    _introPlayed = true;
    if (_introCtx) { _introCtx.revert(); _introCtx = null; }
    const end = { ease: "power3.out", clearProps: "opacity,transform" };
    _introCtx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      if (header) tl.fromTo(header, { opacity: 0, y: -12 }, Object.assign({ opacity: 1, y: 0, duration: 0.32 }, end), 0);
      if (badge) tl.fromTo(badge, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.3 }, end), 0.08);
      if (lines.length) tl.fromTo(lines, { opacity: 0.55, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.34, stagger: 0.08 }, end), 0.14);
      if (lead) tl.fromTo(lead, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.28 }, end), 0.28);
      if (cta) tl.fromTo(cta, { opacity: 0, y: 10 }, Object.assign({ opacity: 1, y: 0, duration: 0.28 }, end), 0.36);
      if (feats.length) tl.fromTo(feats, { opacity: 0, y: 12 }, Object.assign({ opacity: 1, y: 0, duration: 0.28, stagger: 0.05 }, end), 0.44);
      if (stage) {
        const from = { opacity: 0, y: 16 };
        from.scale = 0.97;
        const to = Object.assign({ opacity: 1, y: 0, duration: 0.46 }, end);
        to.scale = 1;
        tl.fromTo(stage, from, to, 0.2);
      }
    });
  }

  function intro() {
    if (reduce.matches || _introPlayed) return;
    _load().then((gsap) => _playIntro(gsap)).catch(() => {});
  }

  if (reduce.addEventListener) reduce.addEventListener("change", () => {
    if (reduce.matches) {
      _revert();
      _release();
      if (_introCtx) { _introCtx.revert(); _introCtx = null; }
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (!_floatTween) return;
    if (document.hidden) _floatTween.pause();
    else _floatTween.play();
  });
  window.addEventListener("pagehide", _revert);

  return { enter, welcome, workspaceChanged, intro };
})();
