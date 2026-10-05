/* ============================================================
   chat.js — Message DOM creation, streaming, thought cards,
             citations, code highlighting
   ============================================================ */

window.Chat = (function () {

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
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2z"/>
          <circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/>
        </svg>
      </div>
      <div class="message__body">
        <div class="thought-steps" id="thoughtSteps-${msgId}"></div>
        <div class="message__bubble">
          <div class="md-content stream-cursor" id="streamContent-${msgId}"></div>
        </div>
        <div class="citation-cards" id="citationCards-${msgId}"></div>
        <div class="message__actions">
          <button class="btn btn--icon-sm" onclick="Chat.copyMessage('${msgId}')" data-tooltip="Copy response">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>
          <div class="message__time" id="msgTime-${msgId}"></div>
        </div>
      </div>
    `;

    area.appendChild(el);
    _scrollToBottom(area);

    let _rawText = "";

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

      /** Stream a token into the content area */
      appendToken(token) {
        _rawText += token;
        const el = document.getElementById(`streamContent-${msgId}`);
        if (el) {
          el.innerHTML = Utils.renderMarkdown(_rawText);
          _scrollToBottom(area);
        }
      },

      /** Finalize streaming: apply full markdown + syntax highlighting */
      finalize(citations = []) {
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
  function renderHistory(messages) {
    const area = document.getElementById("chatArea");
    // Clear existing
    area.innerHTML = "";

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
        const cits = msg.citations || [];
        el.innerHTML = `
          <div class="message__avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2z"/>
              <circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/>
            </svg>
          </div>
          <div class="message__body">
            <div class="message__bubble">
              <div class="md-content">${Utils.renderMarkdown(msg.content)}</div>
            </div>
            ${cits.length > 0 ? `
            <div class="citation-cards">
              ${cits.map(c => `<div class="citation-card">📌 ${Utils.escapeHtml(c.source)}${c.page ? ` · p.${c.page}` : ""}</div>`).join("")}
            </div>` : ""}
            <div class="message__actions">
              <div class="message__time">${Utils.formatTime(msg.created_at)}</div>
            </div>
          </div>
        `;
        // Apply syntax highlighting
        area.appendChild(el);
        Utils.highlightCode(el);
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
    area.innerHTML = "";
    _showWelcome();
  }

  // ── Private helpers ───────────────────────────────────────

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
