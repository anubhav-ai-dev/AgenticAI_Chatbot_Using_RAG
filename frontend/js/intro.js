/* ============================================================
   intro.js — Introduction page.
   Get Started reveals the existing chatbot (#app). No network.
   Theme changes go through Theme (theme.js), the same store the app uses.
   ============================================================ */

(function () {
  "use strict";

  const root = document.documentElement;
  const nav = document.getElementById("introNav");
  const menuBtn = document.getElementById("introMenu");
  const themeBtn = document.getElementById("introTheme");
  const startBtn = document.getElementById("introStart");
  const shell = document.getElementById("appShell");
  const narrow = window.matchMedia("(max-width: 767px)");

  function syncTheme() {
    if (!window.Theme || !themeBtn) return;
    const dark = Theme.resolved() === "dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    const moon = themeBtn.querySelector(".intro__theme-icon--moon");
    const sun = themeBtn.querySelector(".intro__theme-icon--sun");
    if (moon) moon.hidden = dark;
    if (sun) sun.hidden = !dark;
  }

  function setMenu(open) {
    if (!nav || !menuBtn) return;
    nav.classList.toggle("is-open", open);
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) {
      const first = nav.querySelector("a");
      if (first) first.focus();
    }
  }

  function showApp(focusComposer) {
    root.dataset.view = "app";
    if (shell) shell.removeAttribute("inert");
    setMenu(false);
    if (location.hash !== "#app") history.pushState({ view: "app" }, "", "#app");
    if (focusComposer) {
      const input = document.getElementById("chatInput");
      if (input) input.focus();
    }
  }

  function showIntro() {
    root.dataset.view = "intro";
    if (shell) shell.setAttribute("inert", "");
    if (window.Motion) Motion.intro();
  }

  function boot() {
    syncTheme();
    if (window.Theme) Theme.onChange(syncTheme);
    if (root.dataset.view === "intro" && window.Motion) Motion.intro();

    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        if (!window.Theme) return;
        Theme.set(Theme.resolved() === "dark" ? "light" : "dark");
      });
    }

    if (menuBtn) {
      menuBtn.addEventListener("click", () => {
        setMenu(menuBtn.getAttribute("aria-expanded") !== "true");
      });
    }

    if (nav) {
      nav.addEventListener("click", (e) => {
        if (e.target.closest("a")) setMenu(false);
      });
    }

    if (startBtn) startBtn.addEventListener("click", () => showApp(true));

    document.addEventListener("click", (e) => {
      if (!nav || !nav.classList.contains("is-open")) return;
      if (nav.contains(e.target) || (menuBtn && menuBtn.contains(e.target))) return;
      setMenu(false);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || !nav || !nav.classList.contains("is-open")) return;
      setMenu(false);
      if (menuBtn) menuBtn.focus();
    });

    const onNarrow = () => { if (!narrow.matches) setMenu(false); };
    if (narrow.addEventListener) narrow.addEventListener("change", onNarrow);

    window.addEventListener("hashchange", () => {
      if (location.hash === "#app") showApp(false);
      else showIntro();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
