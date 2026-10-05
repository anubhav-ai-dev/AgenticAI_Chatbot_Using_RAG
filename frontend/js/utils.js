/* ============================================================
   utils.js — Markdown, syntax highlighting, helpers
   ============================================================ */

window.Utils = (function () {
  // Configure marked.js
  if (window.marked) {
    marked.setOptions({
      breaks: true,        // GFM line breaks
      gfm: true,
      headerIds: false,
      mangle: false,
    });
  }

  function renderMarkdown(text) {
    if (!text) return "";
    if (window.marked) {
      try {
        let html = marked.parse(text);
        // Transform [1], [2] into citation badges
        html = html.replace(/\[(\d+)\]/g, '<sup class="citation" data-id="$1">[$1]</sup>');
        return html;
      } catch (e) {
        console.error("Markdown parse error:", e);
      }
    }
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function highlightCode(element) {
    if (!window.hljs || !element) return;
    element.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);

      // Wrap in code-block container with language badge & copy button
      const pre = block.parentElement;
      if (pre.parentElement.classList.contains("code-block")) return; // already wrapped

      const lang = (block.className.match(/language-(\w+)/) || [, "code"])[1];
      const wrapper = document.createElement("div");
      wrapper.className = "code-block";

      wrapper.innerHTML = `
        <div class="code-block__header">
          <span class="code-block__lang">${lang}</span>
          <button class="code-block__copy" aria-label="Copy code">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy
          </button>
        </div>
      `;

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      wrapper.querySelector(".code-block__copy").addEventListener("click", function () {
        copyText(block.innerText);
        this.innerHTML = "✓ Copied!";
        setTimeout(() => {
          this.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy
          `;
        }, 2000);
      });
    });
  }

  function copyText(text) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => {});
    } else {
      const el = document.createElement("textarea");
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatTime(isoString) {
    if (!isoString) return "";
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatDateGroup(isoString) {
    if (!isoString) return "Earlier";
    const d = new Date(isoString);
    const now = new Date();
    const diffDays = Math.floor((now - d) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7)  return "Previous 7 Days";
    if (diffDays < 30) return "Previous 30 Days";
    return "Earlier";
  }

  function uuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0;
      return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
    });
  }

  return {
    renderMarkdown,
    highlightCode,
    copyText,
    escapeHtml,
    formatTime,
    formatDateGroup,
    uuid,
  };
})();
