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

  // DOMPurify policy. Model, web and document text is untrusted: HTML is
  // sanitised, never executed. Only http(s) URLs survive; forms, embeds,
  // styles and inline styles are removed; no data-* / id / name attributes
  // (blocks spoofing of our own hooks and DOM clobbering). class/role are also
  // dropped so model output cannot imitate .citation badges or add tab stops;
  // our own classes/roles are applied after sanitising, via DOM code. The one
  // exception is the Markdown code-fence hint (code.language-xxx), which
  // highlight.js needs; see keepLanguageClass.
  const PURIFY_CONFIG = {
    USE_PROFILES: { html: true },
    FORBID_TAGS: [
      "style", "form", "input", "button", "textarea", "select", "option",
      "iframe", "frame", "frameset", "object", "embed", "link", "meta", "base",
      "video", "audio", "source", "track", "svg", "math", "dialog",
    ],
    FORBID_ATTR: ["style", "srcset", "id", "name", "download", "ping", "class", "role"],
    ALLOW_DATA_ATTR: false,
    ALLOWED_URI_REGEXP: /^https?:/i,
  };

  // Keep only <code class="language-xxx"> (marked's fence output) through the
  // class ban; the value is validated so it cannot carry any other class.
  function keepLanguageClass(node, data) {
    if (data.attrName === "class" && node.nodeName === "CODE" &&
        /^language-[\w+#.-]+$/.test(data.attrValue)) {
      data.forceKeepAttr = true;
    }
  }

  function sanitizeMarkdownHtml(html) {
    DOMPurify.addHook("uponSanitizeAttribute", keepLanguageClass);
    try {
      return DOMPurify.sanitize(html, PURIFY_CONFIG);
    } finally {
      DOMPurify.removeHook("uponSanitizeAttribute", keepLanguageClass);
    }
  }

  /**
   * Return a normalised absolute URL string if `url` is http: or https:,
   * otherwise null. Used for every href that comes from untrusted data.
   */
  function safeUrl(url) {
    if (typeof url !== "string" || !url.trim()) return null;
    try {
      const u = new URL(url.trim());
      return (u.protocol === "http:" || u.protocol === "https:") ? u.href : null;
    } catch (_) {
      return null;
    }
  }

  /**
   * Link/image policy for already-sanitised HTML (operates on an inert template):
   *  - links: http(s) only, opened in a new tab with rel="noopener noreferrer"
   *  - images: remote images are blocked (no request is ever made); replaced by a
   *    text placeholder, with an "open" link when the source is http(s)
   */
  function applyLinkImagePolicy(root) {
    root.querySelectorAll("a").forEach((a) => {
      const href = safeUrl(a.getAttribute("href") || "");
      if (href) {
        a.setAttribute("href", href);
        a.setAttribute("target", "_blank");
        a.setAttribute("rel", "noopener noreferrer");
      } else {
        a.removeAttribute("href");
        a.removeAttribute("target");
        a.removeAttribute("rel");
      }
    });

    // Wide tables scroll inside their own focusable region instead of widening the page
    root.querySelectorAll("table").forEach((table) => {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      wrap.setAttribute("role", "region");
      wrap.setAttribute("aria-label", "Table");
      wrap.tabIndex = 0;
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });

    root.querySelectorAll("img").forEach((img) => {
      const src = safeUrl(img.getAttribute("src") || "");
      const alt = (img.getAttribute("alt") || "").trim();
      const label = "Image blocked" + (alt ? ": " + alt : "");
      let repl;
      if (src) {
        repl = document.createElement("a");
        repl.setAttribute("href", src);
        repl.setAttribute("target", "_blank");
        repl.setAttribute("rel", "noopener noreferrer");
        repl.textContent = label + " (open)";
      } else {
        repl = document.createElement("span");
        repl.textContent = label;
      }
      repl.classList.add("md-image-blocked");
      img.replaceWith(repl);
    });
  }

  /**
   * Replace "[n]" in text nodes (outside code/links) with clickable citation badges,
   * and apply the link/image policy. Works on already-sanitised HTML, so it cannot
   * introduce markup from the model.
   */
  function decorateCitations(html) {
    const tpl = document.createElement("template");
    tpl.innerHTML = html;
    applyLinkImagePolicy(tpl.content);

    const walker = document.createTreeWalker(tpl.content, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach((node) => {
      const parent = node.parentElement;
      if (!/\[\d+\]/.test(node.nodeValue) || (parent && parent.closest("pre, code, a"))) return;
      const frag = document.createDocumentFragment();
      node.nodeValue.split(/(\[\d+\])/).forEach((part) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m) {
          const sup = document.createElement("sup");
          sup.className = "citation";
          sup.dataset.id = m[1];
          sup.setAttribute("role", "button");
          sup.setAttribute("tabindex", "0");
          sup.setAttribute("aria-label", "Source " + m[1]);
          sup.textContent = part;
          frag.appendChild(sup);
        } else if (part) {
          frag.appendChild(document.createTextNode(part));
        }
      });
      node.parentNode.replaceChild(frag, node);
    });

    // Mark genuinely adjacent badges ("[3][4]": nothing, not even a space, between them) so the
    // coarse-pointer hit areas can stop at the shared edge. CSS "+" ignores text nodes, so it
    // cannot tell "[1] text [2]" from "[3][4]".
    tpl.content.querySelectorAll("sup.citation").forEach((b) => {
      const prev = b.previousSibling;
      if (prev && prev.nodeType === 1 && prev.matches("sup.citation")) {
        prev.setAttribute("data-cite-next", "");
        b.setAttribute("data-cite-prev", "");
      }
    });
    return tpl.innerHTML;
  }

  /** Markdown → sanitised HTML. Falls back to escaped text if a library is missing. */
  function renderMarkdown(text) {
    if (!text) return "";
    // Some models cite as 【1】 or [1]†; normalise to [1] so badges are clickable
    text = text.replace(/【(\d+)(?:[^】]*)】/g, "[$1]");
    if (window.marked && window.DOMPurify) {
      try {
        const clean = sanitizeMarkdownHtml(marked.parse(text));
        return decorateCitations(clean);
      } catch (e) {
        console.error("Markdown parse error:", e);
      }
    }
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function highlightCode(element) {
    if (!window.hljs || !element) return;
    element.querySelectorAll("pre code").forEach((block) => {
      // Read the label from the fence BEFORE highlighting: hljs adds a language-xxx class when it
      // auto-detects, and we must not present a guess as what the author wrote.
      const fence = block.className.match(/language-([\w+#.-]+)/);
      hljs.highlightElement(block);

      // Wrap in code-block container with language badge & copy button
      const pre = block.parentElement;
      if (pre.parentElement.classList.contains("code-block")) return; // already wrapped

      const lang = fence ? fence[1].toLowerCase() : "Code";
      const wrapper = document.createElement("div");
      wrapper.className = "code-block";

      wrapper.innerHTML = `
        <div class="code-block__header">
          <span class="code-block__lang">${escapeHtml(lang)}</span>
          <button class="code-block__copy" type="button" aria-label="Copy code">
            ${Icons.html("copy", { size: 16 })}<span>Copy</span>
          </button>
        </div>
      `;

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      // Keyboard users must be able to scroll long lines
      pre.tabIndex = 0;
      pre.setAttribute("role", "region");
      pre.setAttribute("aria-label", fence ? `${lang} code` : "Code");

      const btn = wrapper.querySelector(".code-block__copy");
      let resetTimer = null;
      btn.addEventListener("click", function () {
        copyText(block.innerText).then((ok) => {
          btn.innerHTML = `${Icons.html(ok ? "check" : "circle-alert", { size: 16 })}<span>${ok ? "Copied" : "Copy failed"}</span>`;
          btn.setAttribute("aria-label", ok ? "Copied" : "Copy failed");
          clearTimeout(resetTimer);
          resetTimer = setTimeout(() => {
            btn.innerHTML = `${Icons.html("copy", { size: 16 })}<span>Copy</span>`;
            btn.setAttribute("aria-label", "Copy code");
          }, 2000);
        });
      });
    });
  }

  /** Copy text to the clipboard. Resolves true on success, false on failure. */
  function copyText(text) {
    if (navigator.clipboard) {
      return navigator.clipboard.writeText(text).then(() => true, () => false);
    }
    try {
      const el = document.createElement("textarea");
      el.value = text;
      document.body.appendChild(el);
      el.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(el);
      return Promise.resolve(!!ok);
    } catch (_) {
      return Promise.resolve(false);
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ── Dates ─────────────────────────────────────────────────

  // Server timestamps are naive UTC strings without a zone ("2026-10-20T08:30:00").
  // Date() would read those as LOCAL time, so append "Z" when no zone is present.
  const NAIVE_ISO = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/;

  /** Parse a server timestamp (or Date) into a Date; returns null if invalid. */
  function parseDate(value) {
    if (!value) return null;
    if (value instanceof Date) return isNaN(value) ? null : value;
    let s = String(value).trim();
    if (NAIVE_ISO.test(s)) s = s.replace(" ", "T") + "Z";
    const d = new Date(s);
    return isNaN(d) ? null : d;
  }

  function formatTime(isoString) {
    const d = parseDate(isoString);
    if (!d) return "";
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatDateGroup(isoString) {
    const d = parseDate(isoString);
    if (!d) return "Earlier";
    // Calendar-day difference in the user's local time zone
    const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diffDays = Math.round((startOf(new Date()) - startOf(d)) / 86400000);

    if (diffDays <= 0) return "Today";   // includes small clock skew into the future
    if (diffDays === 1) return "Yesterday";
    if (diffDays <= 7) return "Previous 7 Days";
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
    parseDate,
    safeUrl,
    uuid,
  };
})();
