/* ============================================================
   chat.js — Message DOM creation, streaming, thought cards,
             citations, code highlighting
   ============================================================ */

window.Chat = (function () {

  const BOT_AVATAR = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2z"/>
          <circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/>
        </svg>`;

  // Internal citation registry: msgId -> [{id,source,page,snippet}]
  const _citations = {};

  /**
   * Append a USER message bubble to the chat area.
   * Returns the created element.
   */
  function appendUserMessage(text, userId = "default") {
    const area = document.getElementById("chatArea");
    _hideWelcome();

    const initials = (userId || "U").charAt(0).toUpperCase();
    const time = Utils.formatTime(new Date().toISOString());

    const el = document.createElement("div");
    el.className = "message message--user animate-fade-slide-up";
    el.innerHTML = `
      <div class="message__avatar">${initials}</div>
      <div class="message__body">
        <div class="message__bubble">${Utils.escapeHtml(text)}</div>
        <div class="message__time">${time}</div>
      </div>
    `;

    area.appendChild(el);
    _scrollToBottom(area);
    return el;
  }

  /**
   * Start an ASSISTANT message with thought steps + streaming area.
   * Returns an object with methods to control the ongoing render.
   */
  function startAssistantMessage() {
    const area = document.getElementById("chatArea");
    _hideWelcome();

    const msgId = Utils.uuid();
    _citations[msgId] = [];

    const el = document.createElement("div");
    el.className = "message message--assistant animate-fade-slide-up";
    el.dataset.msgId = msgId;
    el.innerHTML = `
      <div class="message__avatar">
        ${BOT_AVATAR}
      </div>
      <div class="message__body">
        <div class="thought-steps" id="thoughtSteps-${msgId}"></div>
        <div class="message__bubble">
          <div class="md-content stream-cursor" id="streamContent-${msgId}"></div>
        </div>
        <div class="citation-cards" id="citationCards-${msgId}"></div>
        <div class="message__badges" id="msgBadges-${msgId}"></div>
        <div class="message__actions">
          <button class="btn btn--icon-sm" onclick="Chat.copyMessage('${msgId}')" data-tooltip="Copy response" aria-label="Copy response">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>
          <span class="message__extra-actions" id="msgExtra-${msgId}"></span>
          <div class="message__time" id="msgTime-${msgId}"></div>
        </div>
      </div>
    `;

    area.appendChild(el);
    _scrollToBottom(area);

    let _rawText = "";
    let _renderTimer = null;

    const _render = () => {
      _renderTimer = null;
      const target = document.getElementById(`streamContent-${msgId}`);
      if (target) {
        target.innerHTML = Utils.renderMarkdown(_rawText);
        _scrollToBottom(area);
      }
    };

    return {
      msgId,

      /** Add a thought step card */
      addThought(icon, label, status = "pending") {
        const steps = document.getElementById(`thoughtSteps-${msgId}`);
        if (!steps) return null;

        const step = document.createElement("div");
        step.className = `thought-step thought-step--${status === "error" ? "error" : ""}`;
        step.id = `thought-${msgId}-${Utils.uuid()}`;
        step.innerHTML = `
          <span class="thought-step__icon">${status === "pending" ? '<div class="spinner"></div>' : icon}</span>
          <span class="thought-step__label">${Utils.escapeHtml(label)}</span>
          <span class="thought-step__time"></span>
        `;
        steps.appendChild(step);
        return step;
      },

      /** Finalize a thought step (mark done/error, show elapsed) */
      resolveThought(stepEl, { icon, label, elapsed, status = "done" } = {}) {
        if (!stepEl) return;
        stepEl.classList.toggle("thought-step--done", status === "done");
        stepEl.classList.toggle("thought-step--error", status === "error");
        if (icon) stepEl.querySelector(".thought-step__icon").innerHTML = icon;
        if (label) stepEl.querySelector(".thought-step__label").textContent = label;
        if (elapsed) stepEl.querySelector(".thought-step__time").textContent = elapsed;
      },

      /** Stream a token into the content area (re-rendered at most every 60 ms) */
      appendToken(token) {
        _rawText += token;
        if (_renderTimer === null) _renderTimer = setTimeout(_render, 60);
      },

      /** Show model / cache / fallback badges under the bubble */
      setBadges({ model, cached, fellBackFrom } = {}) {
        const el = document.getElementById(`msgBadges-${msgId}`);
        if (!el) return;
        const parts = [];
        if (model) parts.push(`<span class="badge">${Utils.escapeHtml(model)}</span>`);
        if (cached) parts.push(`<span class="badge badge--cache" title="Served from the answer cache">⚡ cached</span>`);
        if (fellBackFrom && fellBackFrom.length) {
          parts.push(`<span class="badge badge--warn" title="Fell back from ${Utils.escapeHtml(fellBackFrom.join(", "))}">↪ fallback</span>`);
        }
        el.innerHTML = parts.join("");
      },

      /** Add regenerate + 👍/👎 once the message has an id */
      enableActions({ messageId, feedback, onRegenerate, onFeedback } = {}) {
        const el = document.getElementById(`msgExtra-${msgId}`);
        if (!el) return;
        _renderActionButtons(el, { messageId, feedback, onRegenerate, onFeedback });
      },

      /** Finalize streaming: apply full markdown + syntax highlighting */
      finalize(citations = []) {
        if (_renderTimer !== null) {
          clearTimeout(_renderTimer);
          _renderTimer = null;
        }
        const el = document.getElementById(`streamContent-${msgId}`);
        if (el) {
          el.classList.remove("stream-cursor");
          el.innerHTML = Utils.renderMarkdown(_rawText);
          Utils.highlightCode(el);

          // Wire citation badge clicks
          el.querySelectorAll(".citation").forEach((badge) => {
            badge.addEventListener("click", () => {
              const id = parseInt(badge.dataset.id);
              const cit = citations.find((c) => c.id === id);
              if (cit) UI.showCitationModal(cit);
            });
          });
        }

        // Set timestamp
        const timeEl = document.getElementById(`msgTime-${msgId}`);
        if (timeEl) timeEl.textContent = Utils.formatTime(new Date().toISOString());

        // Render citation cards below the bubble
        if (citations.length > 0) {
          _citations[msgId] = citations;
          const cardsEl = document.getElementById(`citationCards-${msgId}`);
          if (cardsEl) {
            cardsEl.innerHTML = citations.map((c) => {
              const page = c.page ? ` · p.${c.page}` : "";
              return `
                <div class="citation-card" onclick="UI.showCitationModal(Chat.getCitation('${msgId}',${c.id}))"
                     title="${Utils.escapeHtml(c.snippet || "")}">
                  📌 ${Utils.escapeHtml(c.source)}${page}
                </div>
              `;
            }).join("");
          }
        }

        _scrollToBottom(area);
      },

      /** Replace content with an error message */
      showError(msg) {
        if (_renderTimer !== null) {
          clearTimeout(_renderTimer);
          _renderTimer = null;
        }
        const el = document.getElementById(`streamContent-${msgId}`);
        if (el) {
          el.classList.remove("stream-cursor");
          el.innerHTML = `<span style="color:var(--accent-red)">⚠️ ${Utils.escapeHtml(msg)}</span>`;
        }
      },

      getRawText() { return _rawText; },
    };
  }

  /** Copy message text to clipboard */
  function copyMessage(msgId) {
    const el = document.getElementById(`streamContent-${msgId}`);
    if (el) Utils.copyText(el.innerText);
  }

  /** Retrieve a citation object for a given msgId and citation id */
  function getCitation(msgId, id) {
    return (_citations[msgId] || []).find((c) => c.id === id) || null;
  }

  /** Show a temporary "Thinking…" placeholder skeleton */
  function showThinking() {
    const area = document.getElementById("chatArea");
    const el = document.createElement("div");
    el.id = "thinkingPlaceholder";
    el.className = "message message--assistant animate-fade-in";
    el.innerHTML = `
      <div class="message__avatar">
        <div class="spinner spinner--lg" style="border-top-color:var(--accent-blue)"></div>
      </div>
      <div class="message__body">
        <div class="message__bubble">
          <div class="skeleton" style="height:14px;width:80%;margin-bottom:8px"></div>
          <div class="skeleton" style="height:14px;width:60%;margin-bottom:8px"></div>
          <div class="skeleton" style="height:14px;width:70%"></div>
        </div>
      </div>
    `;
    area.appendChild(el);
    _scrollToBottom(area);
    return el;
  }

  function removeThinking() {
    const el = document.getElementById("thinkingPlaceholder");
    if (el) el.remove();
  }

  /** Restore a past session's messages into the chat area */
  function renderHistory(messages, handlers = {}) {
    const area = document.getElementById("chatArea");
    _resetArea(area);

    messages.forEach((msg) => {
      if (msg.role === "user") {
        const el = document.createElement("div");
        el.className = "message message--user";
        el.innerHTML = `
          <div class="message__avatar">${"U"}</div>
          <div class="message__body">
            <div class="message__bubble">${Utils.escapeHtml(msg.content)}</div>
            <div class="message__time">${Utils.formatTime(msg.created_at)}</div>
          </div>
        `;
        area.appendChild(el);
      } else if (msg.role === "assistant") {
        const el = document.createElement("div");
        el.className = "message message--assistant";
        const histId = Utils.uuid();
        _citations[histId] = msg.citations || [];
        const cits = _citations[histId];
        const meta = msg.meta || {};
        el.innerHTML = `
          <div class="message__avatar">
            ${BOT_AVATAR}
          </div>
          <div class="message__body">
            <div class="message__bubble">
              <div class="md-content">${Utils.renderMarkdown(msg.content)}</div>
            </div>
            ${cits.length > 0 ? `
            <div class="citation-cards">
              ${cits.map(c => `<div class="citation-card" data-cid="${c.id}">📌 ${Utils.escapeHtml(c.source)}${c.page ? ` · p.${c.page}` : ""}</div>`).join("")}
            </div>` : ""}
            <div class="message__badges">${meta.model ? `<span class="badge">${Utils.escapeHtml(meta.model)}</span>` : ""}${meta.cached ? `<span class="badge badge--cache">⚡ cached</span>` : ""}</div>
            <div class="message__actions">
              <span class="message__extra-actions"></span>
              <div class="message__time">${Utils.formatTime(msg.created_at)}</div>
            </div>
          </div>
        `;
        area.appendChild(el);
        Utils.highlightCode(el);

        // Citation badges + cards open the modal (restored history used to be inert)
        const open = (id) => {
          const cit = cits.find((c) => c.id === id);
          if (cit) UI.showCitationModal(cit);
        };
        el.querySelectorAll(".citation").forEach((b) =>
          b.addEventListener("click", () => open(parseInt(b.dataset.id, 10))));
        el.querySelectorAll(".citation-card").forEach((c) =>
          c.addEventListener("click", () => open(parseInt(c.dataset.cid, 10))));

        if (msg.id && handlers.onFeedback) {
          _renderActionButtons(el.querySelector(".message__extra-actions"), {
            messageId: msg.id,
            feedback: msg.feedback,
            onFeedback: handlers.onFeedback,
          });
        }
      }
    });

    if (messages.length === 0) {
      _showWelcome();
    } else {
      _hideWelcome();
      _scrollToBottom(area);
    }
  }

  function clearChat() {
    const area = document.getElementById("chatArea");
    _resetArea(area);
    _showWelcome();
  }

  // ── Private helpers ───────────────────────────────────────

  /** Render regenerate / 👍 / 👎 buttons into `container` (idempotent). */
  function _renderActionButtons(container, { messageId, feedback, onRegenerate, onFeedback }) {
    if (!container) return;
    container.innerHTML = "";
    const mk = (label, title, extraClass = "") => {
      const b = document.createElement("button");
      b.className = `btn btn--icon-sm ${extraClass}`.trim();
      b.type = "button";
      b.textContent = label;
      b.title = title;
      b.setAttribute("aria-label", title);
      return b;
    };

    if (onRegenerate) {
      const regen = mk("↻", "Regenerate response", "message__regen");
      regen.addEventListener("click", () => onRegenerate());
      container.appendChild(regen);
    }
    if (onFeedback && messageId) {
      let current = feedback || 0;
      const up = mk("👍", "Good response");
      const down = mk("👎", "Poor response");
      const paint = () => {
        up.classList.toggle("is-active", current === 1);
        down.classList.toggle("is-active", current === -1);
      };
      const send = async (value) => {
        const next = current === value ? 0 : value;
        try {
          await onFeedback(messageId, next);
          current = next;
          paint();
        } catch (_) {}
      };
      up.addEventListener("click", () => send(1));
      down.addEventListener("click", () => send(-1));
      paint();
      container.appendChild(up);
      container.appendChild(down);
    }
  }

  /** Remove all messages but keep the welcome screen node (it lives inside the chat area). */
  function _resetArea(area) {
    const welcome = document.getElementById("welcomeScreen");
    if (welcome && welcome.parentNode === area) area.removeChild(welcome);
    area.innerHTML = "";
    if (welcome) area.appendChild(welcome);
  }

  function _scrollToBottom(el) {
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }

  function _hideWelcome() {
    const w = document.getElementById("welcomeScreen");
    if (w) w.style.display = "none";
  }

  function _showWelcome() {
    const w = document.getElementById("welcomeScreen");
    if (w) w.style.display = "";
  }

  return {
    appendUserMessage,
    startAssistantMessage,
    showThinking,
    removeThinking,
    renderHistory,
    clearChat,
    copyMessage,
    getCitation,
  };
})();
