/* ============================================================
   chat.js — Message DOM creation, streaming, activity steps,
             citations / sources, smart scroll, code highlighting
   (view layer: no fetch, no State; the controller passes callbacks)
   ============================================================ */

window.Chat = (function () {

  // Follow the stream only while the reader is within this distance of the bottom (design-spec 7.2)
  const STICK_PX = 80;

  // Internal citation registry: msgId -> [{id,type,source,page,url,snippet}]
  const _citations = {};
  // msgId -> raw Markdown (copy uses this, not the rendered DOM, so UI chrome never leaks into the clipboard)
  const _raw = {};

  let _stuck = true;          // true while the reader is following the newest content
  let _scrollBound = false;

  /** Announce a status line to screen readers (coalesced by UI; never per token). */
  function _announce(text) {
    if (window.UI && typeof UI.setLiveStatus === "function") UI.setLiveStatus(text);
  }

  function _el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ── Smart scroll ──────────────────────────────────────────

  function _distanceFromBottom(area) {
    return area.scrollHeight - area.scrollTop - area.clientHeight;
  }

  function _updateJump(area) {
    const jump = document.getElementById("jumpLatest");
    if (!jump) return;
    const show = !_stuck && !!area.querySelector(".message");
    if (jump.hidden === show) jump.hidden = !show;
  }

  /** Bind the scroll listener and the "Jump to latest" control once. */
  function _bindScroll(area) {
    if (_scrollBound || !area) return;
    _scrollBound = true;
    area.addEventListener("scroll", () => {
      _stuck = _distanceFromBottom(area) <= STICK_PX;
      _updateJump(area);
    }, { passive: true });
    const jump = document.getElementById("jumpLatest");
    if (jump) {
      jump.addEventListener("click", () => {
        _scrollNow(area);
        _stuck = true;
        _updateJump(area);
        area.setAttribute("tabindex", "-1");
        area.focus({ preventScroll: true });   // the button hides itself; keep focus somewhere sensible
      });
    }
  }

  function _scrollNow(area) {
    // "instant" overrides the CSS smooth-scroll so rapid streaming never fights the animation
    try { area.scrollTo({ top: area.scrollHeight, behavior: "instant" }); }
    catch (_) { area.scrollTop = area.scrollHeight; }
  }

  /** Follow new content only if the reader has not scrolled away. */
  function _follow(area) {
    _bindScroll(area);
    if (!_stuck) { _updateJump(area); return; }
    requestAnimationFrame(() => { if (_stuck) _scrollNow(area); });
  }

  /** Always go to the bottom (the user just sent a message, or history was restored). */
  function _forceBottom(area) {
    _bindScroll(area);
    _stuck = true;
    requestAnimationFrame(() => { _scrollNow(area); _updateJump(area); });
  }

  // ── Message builders ──────────────────────────────────────

  function _buildUser(text, timeText, live) {
    const el = _el("article", "message message--user" + (live ? " animate-fade-slide-up" : ""));
    el.setAttribute("aria-label", "You said");
    const body = _el("div", "message__body");
    body.appendChild(_el("div", "message__bubble", text == null ? "" : String(text)));
    body.appendChild(_el("div", "message__time", timeText || ""));
    el.appendChild(body);
    return el;
  }

  /** Assistant message skeleton shared by live streaming and restored history. */
  function _buildAssistant(msgId, { live }) {
    const el = _el("article", "message message--assistant" + (live ? " animate-fade-slide-up" : ""));
    el.dataset.msgId = msgId;
    el.setAttribute("aria-label", "Assistant said");
    el.innerHTML = `
      <div class="message__avatar" aria-hidden="true"></div>
      <div class="message__body">
        <div class="activity" id="activity-${msgId}" hidden>
          <button type="button" class="activity__toggle" aria-expanded="true" aria-controls="thoughtSteps-${msgId}" hidden>
            ${Icons.html("chevron-down", { size: 16, cls: "activity__chevron" })}<span class="activity__summary"></span>
          </button>
          <div class="activity__panel">
            <div class="thought-steps" id="thoughtSteps-${msgId}" role="list" aria-label="Activity"></div>
          </div>
        </div>
        <div class="message__loading" aria-hidden="true" hidden>
          <div class="skeleton skeleton-line skeleton-line--80"></div>
          <div class="skeleton skeleton-line skeleton-line--60"></div>
          <div class="skeleton skeleton-line skeleton-line--70"></div>
        </div>
        <div class="message__bubble">
          <div class="md-content" id="streamContent-${msgId}"></div>
        </div>
        <div class="file-card-slot" id="fileCard-${msgId}" hidden></div>
        <div class="sources" id="citationCards-${msgId}" hidden></div>
        <div class="message__badges" id="msgBadges-${msgId}"></div>
        <div class="message__actions">
          <button type="button" class="btn btn--icon-sm message__copy" data-tooltip="Copy response" aria-label="Copy response">
            ${Icons.html("copy", { size: 16 })}
          </button>
          <span class="message__extra-actions" id="msgExtra-${msgId}"></span>
          <div class="message__time" id="msgTime-${msgId}"></div>
        </div>
      </div>
    `;
    if (window.AIViz) AIViz.mount(el.querySelector(".message__avatar"), { size: "sm", state: "idle" });

    el.querySelector(".message__copy").addEventListener("click", () => copyMessage(msgId));
    const toggle = el.querySelector(".activity__toggle");
    toggle.addEventListener("click", () => _setActivityOpen(el, toggle.getAttribute("aria-expanded") !== "true"));
    return el;
  }

  function _setActivityOpen(el, open) {
    const wrap = el.querySelector(".activity");
    const toggle = el.querySelector(".activity__toggle");
    const panel = el.querySelector(".activity__panel");
    if (!wrap || !toggle || !panel) return;
    wrap.classList.toggle("is-collapsed", !open);
    toggle.setAttribute("aria-expanded", String(open));
    panel.inert = !open;
  }

  /** After the answer is complete the activity steps collapse behind a disclosure (design-spec 6.8). */
  function _collapseActivity(el) {
    const steps = el.querySelectorAll(".thought-step").length;
    const toggle = el.querySelector(".activity__toggle");
    if (!steps || !toggle) return;
    el.querySelector(".activity__summary").textContent = `Activity · ${steps} ${steps === 1 ? "step" : "steps"}`;
    toggle.hidden = false;
    _setActivityOpen(el, false);
  }

  function _setViz(el, state) {
    if (window.AIViz) AIViz.setState(el.querySelector(".message__avatar"), state);
  }

  /** One note line under the answer ("Stopped", "interrupted", "no text"). Replaces an earlier note. */
  function _setNote(el, text, detail) {
    let note = el.querySelector(".message-note");
    if (!note) {
      note = _el("div", "message-note");
      el.querySelector(".message__bubble").after(note);
    }
    note.textContent = "";
    note.appendChild(Icons.el("triangle-alert", { size: 16 }));
    const t = _el("span", "message-note__text", text);
    note.appendChild(t);
    if (detail) note.appendChild(_el("span", "message-note__detail", String(detail)));
    return note;
  }

  // ── Public: user message ──────────────────────────────────

  /**
   * Append a USER message to the chat area. `userId` is accepted for
   * compatibility; the redesigned user message has no avatar.
   * Returns the created element.
   */
  function appendUserMessage(text, userId = "default") {
    const area = document.getElementById("chatArea");
    _hideWelcome();
    const el = _buildUser(text, Utils.formatTime(new Date().toISOString()), true);
    area.appendChild(el);
    _forceBottom(area);
    return el;
  }

  // ── Public: live assistant message ────────────────────────

  /**
   * Start an ASSISTANT message with activity steps + streaming area.
   * Returns an object with methods to control the ongoing render.
   */
  function startAssistantMessage() {
    const area = document.getElementById("chatArea");
    _hideWelcome();

    const msgId = Utils.uuid();
    _citations[msgId] = [];
    _raw[msgId] = "";

    const el = _buildAssistant(msgId, { live: true });
    el.setAttribute("aria-busy", "true");
    el.querySelector(".message__loading").hidden = false;
    el.querySelector(".md-content").classList.add("stream-cursor");
    _setViz(el, "processing");   // the request has been submitted

    area.appendChild(el);
    _forceBottom(area);

    let _rawText = "";
    let _renderTimer = null;
    let _responding = false;
    let _doneRendering = false;

    const _content = () => document.getElementById(`streamContent-${msgId}`);
    const _clearLoading = () => {
      const l = el.querySelector(".message__loading");
      if (l && !l.hidden) l.hidden = true;
    };
    const _cancelTimer = () => {
      if (_renderTimer !== null) { clearTimeout(_renderTimer); _renderTimer = null; }
    };

    const _render = () => {
      _renderTimer = null;
      if (_doneRendering) return;
      const target = _content();
      if (target) {
        target.innerHTML = Utils.renderMarkdown(_rawText);
        // [n] markers are not interactive until finalize() binds them to real citations
        target.querySelectorAll(".citation").forEach((b) => {
          b.removeAttribute("role");
          b.removeAttribute("tabindex");
        });
        _follow(area);
      }
    };

    return {
      msgId,

      /**
       * Add an activity step (from a real `status` event).
       * `icon`: an icon name (pencil, file-search, globe, sparkles, circle-check, circle-alert)
       * or legacy text/emoji, which is accepted and shown escaped (or mapped to its icon).
       */
      addThought(icon, label, status = "pending") {
        const steps = document.getElementById(`thoughtSteps-${msgId}`);
        if (!steps) return null;

        _clearLoading();
        el.querySelector(".activity").hidden = false;
        if (!_responding) _setViz(el, "processing");

        const step = document.createElement("div");
        step.className = "thought-step" + (status === "error" ? " thought-step--error" : "") + (status === "done" ? " thought-step--done" : "");
        step.setAttribute("role", "listitem");
        step.id = `thought-${msgId}-${Utils.uuid()}`;
        step.innerHTML = `
          <span class="thought-step__icon">${status === "pending" ? '<span class="spinner" aria-hidden="true"></span>' : Icons.html(icon, { size: 16 })}</span>
          <span class="thought-step__label">${Utils.escapeHtml(label)}</span>
          <span class="thought-step__time"></span>
        `;
        steps.appendChild(step);
        _announce(label);
        _follow(area);
        return step;
      },

      /** Finalize a step (mark done/error, show elapsed) */
      resolveThought(stepEl, { icon, label, elapsed, status = "done" } = {}) {
        if (!stepEl) return;
        stepEl.classList.toggle("thought-step--done", status === "done");
        stepEl.classList.toggle("thought-step--error", status === "error");
        const iconEl = stepEl.querySelector(".thought-step__icon");
        if (iconEl) {
          const name = icon || (status === "error" ? "circle-alert" : "circle-check");
          iconEl.innerHTML = Icons.html(name, { size: 16 });
        }
        if (label) stepEl.querySelector(".thought-step__label").textContent = label;
        if (elapsed) stepEl.querySelector(".thought-step__time").textContent = elapsed;
      },

      /** Stream a token into the content area (re-rendered at most every 60 ms) */
      appendToken(token) {
        if (!_responding) {
          _responding = true;
          _clearLoading();
          _setViz(el, "responding");   // first token
        }
        _rawText += token;
        _raw[msgId] = _rawText;
        if (_renderTimer === null) _renderTimer = setTimeout(_render, 60);
      },

      /** Show model / cache / fallback badges under the answer */
      setBadges({ model, cached, fellBackFrom } = {}) {
        const badgesEl = document.getElementById(`msgBadges-${msgId}`);
        if (!badgesEl) return;
        badgesEl.innerHTML = _badgesHtml({ model, cached, fellBackFrom });
      },

      /** Add regenerate + feedback buttons once the message has an id */
      enableActions({ messageId, feedback, onRegenerate, onFeedback } = {}) {
        const extra = document.getElementById(`msgExtra-${msgId}`);
        if (!extra) return;
        _renderActionButtons(extra, { messageId, feedback, onRegenerate, onFeedback });
      },

      /** Finalize streaming: apply full markdown + syntax highlighting + sources */
      finalize(citations = []) {
        citations = _resolveCitations(citations);
        _citations[msgId] = citations;
        _cancelTimer();
        _doneRendering = true;
        _clearLoading();
        el.removeAttribute("aria-busy");
        const content = _content();
        if (content) {
          content.classList.remove("stream-cursor");
          content.innerHTML = Utils.renderMarkdown(_rawText);
          Utils.highlightCode(content);
          _bindCitationBadges(content, citations);
        }
        if (!_rawText.trim()) _setNote(el, "No response text was received.");

        const timeEl = document.getElementById(`msgTime-${msgId}`);
        if (timeEl) timeEl.textContent = Utils.formatTime(new Date().toISOString());

        const cits = _citations[msgId];
        const cardsEl = document.getElementById(`citationCards-${msgId}`);
        if (cardsEl) _fillSources(cardsEl, cits, msgId);

        _collapseActivity(el);
        _setViz(el, "idle");
        _announce("Response complete");
        _follow(area);
      },

      /**
       * Replace the content with an error message.
       * opts.onRetry (optional): renders a Retry button and a Dismiss button (removes this message).
       */
      showError(msg, opts = {}) {
        _cancelTimer();
        _doneRendering = true;
        _clearLoading();
        el.removeAttribute("aria-busy");
        const content = _content();
        if (content) {
          content.classList.remove("stream-cursor");
          content.innerHTML = "";
          const box = _el("div", "message-error");
          box.setAttribute("role", "alert");
          box.appendChild(Icons.el("circle-alert", { size: 20 }));
          box.appendChild(_el("span", "message-error__text", msg == null ? "" : String(msg)));
          if (typeof opts.onRetry === "function") {
            const bar = _el("span", "message-error__actions");
            const retry = _el("button", "btn btn--secondary btn--sm", "Retry");
            retry.type = "button";
            retry.addEventListener("click", () => { opts.onRetry(); });
            // retryAfter (seconds): Retry stays disabled with a visible countdown that updates each
            // second (not live-announced per tick); announced once when it becomes available.
            const wait = Math.floor(Number(opts.retryAfter));
            if (wait > 0) {
              let left = Math.min(wait, 3600);
              const label = () => `Retry in ${left}s`;
              retry.disabled = true;
              retry.textContent = label();
              const step = () => {
                if (!retry.isConnected) return;
                left -= 1;
                if (left > 0) { retry.textContent = label(); setTimeout(step, 1000); return; }
                retry.disabled = false;
                retry.textContent = "Retry";
                _announce("Retry is available.");
              };
              setTimeout(step, 1000);
            }
            const dismiss = _el("button", "btn btn--ghost btn--sm", "Dismiss");
            dismiss.type = "button";
            dismiss.addEventListener("click", () => { el.remove(); _updateJump(area); });
            bar.appendChild(retry);
            bar.appendChild(dismiss);
            box.appendChild(bar);
          }
          content.appendChild(box);
        }
        // Show the steps that did run (they explain where it failed)
        if (el.querySelector(".thought-step")) el.querySelector(".activity").hidden = false;
        _setViz(el, "error");
        // No _announce here: the role="alert" box above is the single announcement channel.
        _follow(area);
      },

      /**
       * Mark a delivered answer as incomplete.
       * { reason: "stopped" | "error", message? }: "stopped" = the reader pressed Stop;
       * "error" = the stream broke after some text arrived (message = the server's text, shown verbatim).
       */
      markPartial({ reason = "stopped", message } = {}) {
        if (reason === "error") {
          _setNote(el, "Response interrupted. This answer is incomplete.", message);
          _setViz(el, "error");
        } else {
          _setNote(el, "Stopped. This answer may be incomplete.");
        }
        _follow(area);
      },

      /** Drive the avatar visualization directly: idle | processing | responding | error */
      setVizState(state) { _setViz(el, state); },

      /** Attach the document that arrived on the artifact event. An error is a note, not a file card. */
      attachArtifact(data) {
        if (!data) return;
        if (data.error) {
          _setNote(el, String(data.error));
          return;
        }
        const slot = document.getElementById(`fileCard-${msgId}`);
        _mountFileCard(slot, data, data.message_id, data.format);
        const opener = slot && slot.querySelector(".file-card__open");
        openArtifact({
          doc: data, messageId: data.message_id || "", opener: opener || undefined, format: data.format,
        });
      },

      getRawText() { return _rawText; },
    };
  }

  /** Copy the answer's Markdown to the clipboard; shows a "Copied" state on the message's copy button. */
  function copyMessage(msgId) {
    const content = document.getElementById(`streamContent-${msgId}`);
    if (!content) return Promise.resolve(false);
    const btn = document.querySelector(`[data-msg-id="${msgId}"] .message__copy`);
    const text = _raw[msgId] != null && _raw[msgId] !== "" ? _raw[msgId] : content.innerText;
    return Utils.copyText(text).then((ok) => {
      if (btn) _flashCopy(btn, ok);
      return ok;
    });
  }

  /**
   * Resolve citation ids once so badges, Sources rows, the dialog and history restore agree.
   * Explicit positive-integer ids are kept (first occurrence wins); missing, invalid or
   * duplicate ids get the smallest unused positive integer, in citation order.
   * Returns shallow copies with a numeric `id`; the input is not mutated.
   */
  function _resolveCitations(citations) {
    const list = Array.isArray(citations) ? citations.filter((c) => c && typeof c === "object") : [];
    const used = new Set();
    const keep = list.map((c) => {
      const raw = c.id;
      const n = (typeof raw === "number" || (typeof raw === "string" && /^\s*\d+\s*$/.test(raw))) ? Number(raw) : NaN;
      if (Number.isSafeInteger(n) && n > 0 && !used.has(n)) { used.add(n); return n; }
      return null;
    });
    let next = 1;
    return list.map((c, i) => {
      let id = keep[i];
      if (id === null) {
        while (used.has(next)) next++;
        id = next;
        used.add(id);
      }
      return Object.assign({}, c, { id });
    });
  }

  /** Copy handed to the dialog, with a readable name when source is blank. */
  function _modalCitation(c) {
    const href = Utils.safeUrl(c.url);
    const domain = href ? new URL(href).hostname.replace(/^www\./, "") : "";
    return Object.assign({}, c, { source: _citationLabel(c, domain) });
  }

  /** Display name for a citation: source, else url host, else "Source N". */
  function _citationLabel(c, domain) {
    const src = c.source == null ? "" : String(c.source).trim();
    return src || domain || `Source ${c.id}`;
  }

  /** Retrieve a citation object for a given msgId and citation id */
  function getCitation(msgId, id) {
    return (_citations[msgId] || []).find((c) => c.id === Number(id)) || null;
  }

  /** Show a temporary "Thinking…" placeholder (legacy; the live message has its own loading state) */
  function showThinking() {
    const area = document.getElementById("chatArea");
    const el = _el("div", "message message--assistant animate-fade-in");
    el.id = "thinkingPlaceholder";
    el.innerHTML = `
      <div class="message__avatar" aria-hidden="true"></div>
      <div class="message__body">
        <div class="message__loading" aria-hidden="true">
          <div class="skeleton skeleton-line skeleton-line--80"></div>
          <div class="skeleton skeleton-line skeleton-line--60"></div>
          <div class="skeleton skeleton-line skeleton-line--70"></div>
        </div>
      </div>
    `;
    if (window.AIViz) AIViz.mount(el.querySelector(".message__avatar"), { size: "sm", state: "processing" });
    area.appendChild(el);
    _follow(area);
    return el;
  }

  function removeThinking() {
    const el = document.getElementById("thinkingPlaceholder");
    if (el) el.remove();
  }

  // ── Public: restored history ──────────────────────────────

  /**
   * Restore a past session's messages into the chat area, rendered like live ones.
   * handlers: { onFeedback(messageId, value), onRegenerate() }. onRegenerate (optional) is offered on
   * the newest assistant message only. Returns nothing.
   */
  function renderHistory(messages, handlers = {}) {
    const area = document.getElementById("chatArea");
    _resetArea(area);
    messages = Array.isArray(messages) ? messages : [];

    let lastAssistant = -1;
    messages.forEach((m, i) => { if (m && m.role === "assistant") lastAssistant = i; });

    const frag = document.createDocumentFragment();
    messages.forEach((msg, i) => {
      if (!msg) return;
      if (msg.role === "user") {
        frag.appendChild(_buildUser(msg.content, Utils.formatTime(msg.created_at), false));
      } else if (msg.role === "assistant") {
        frag.appendChild(_historyAssistant(msg, {
          onFeedback: handlers.onFeedback,
          onRegenerate: i === lastAssistant ? handlers.onRegenerate : null,
        }));
      }
    });
    area.appendChild(frag);
    // Highlighting needs the nodes in the document tree only for innerText-free work; do it once here
    Utils.highlightCode(area);

    // [n] markers and source rows need the live nodes
    area.querySelectorAll(".message--assistant[data-msg-id]").forEach((el) => {
      _bindCitationBadges(el.querySelector(".md-content"), _citations[el.dataset.msgId] || []);
    });

    if (messages.length === 0) {
      _showWelcome();
    } else {
      _hideWelcome();
      _forceBottom(area);
    }
  }

  function _historyAssistant(msg, { onFeedback, onRegenerate }) {
    const histId = Utils.uuid();
    const cits = _resolveCitations(msg.citations);
    _citations[histId] = cits;
    _raw[histId] = msg.content == null ? "" : String(msg.content);
    const meta = msg.meta || {};

    const el = _buildAssistant(histId, { live: false });
    el.querySelector(".md-content").innerHTML = Utils.renderMarkdown(_raw[histId]);
    el.querySelector(".message__badges").innerHTML = _badgesHtml({
      model: meta.model,
      cached: meta.cached,
      fellBackFrom: Array.isArray(meta.fell_back_from) ? meta.fell_back_from : [],
    });
    el.querySelector(".message__time").textContent = Utils.formatTime(msg.created_at);
    _fillSources(el.querySelector(".sources"), cits, histId);

    // Real persisted flags (chat_service stores partial / stopped in the message meta)
    if (meta.artifact) _mountFileCard(el.querySelector(".file-card-slot"), meta.artifact, msg.id, meta.artifact_format);
    if (meta.stopped) _setNote(el, "Stopped. This answer may be incomplete.");
    else if (meta.partial) _setNote(el, "Response was interrupted. This answer may be incomplete.");

    if ((msg.id && onFeedback) || onRegenerate) {
      _renderActionButtons(el.querySelector(".message__extra-actions"), {
        messageId: msg.id,
        feedback: msg.feedback,
        onFeedback: msg.id ? onFeedback : null,
        onRegenerate,
      });
    }
    return el;
  }

  function clearChat() {
    const area = document.getElementById("chatArea");
    _resetArea(area);
    _showWelcome();
    _stuck = true;
    _updateJump(area);
  }

  // ── Private helpers ───────────────────────────────────────

  function _badgesHtml({ model, cached, fellBackFrom } = {}) {
    const parts = [];
    if (model) parts.push(`<span class="badge">${Utils.escapeHtml(model)}</span>`);
    if (cached) {
      parts.push(`<span class="badge badge--cache" title="Served from the answer cache">${Icons.html("zap", { size: 16 })}Cached</span>`);
    }
    if (fellBackFrom && fellBackFrom.length) {
      const from = Utils.escapeHtml(fellBackFrom.join(", "));
      parts.push(`<span class="badge badge--warn" title="Fell back from ${from}">${Icons.html("shuffle", { size: 16 })}Fell back from ${from}</span>`);
    }
    return parts.join("");
  }

  /**
   * Make [n] citation markers open the source dialog (mouse and keyboard).
   * A marker with no matching citation is turned back into plain text (no dead control).
   */
  function _bindCitationBadges(root, citations) {
    if (!root) return;
    root.querySelectorAll(".citation").forEach((badge) => {
      const id = parseInt(badge.dataset.id, 10);
      const cit = (citations || []).find((c) => Number(c.id) === id);
      if (!cit) {
        badge.replaceWith(document.createTextNode(badge.textContent));
        return;
      }
      badge.setAttribute("role", "button");
      badge.setAttribute("tabindex", "0");
      const open = () => UI.showCitationModal(_modalCitation(cit));
      badge.addEventListener("click", open);
      badge.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      });
    });
  }

  /**
   * Sources block under an answer. Rendered only from the citations passed in.
   * - citation with an http(s) url: a safe link (new tab, noopener noreferrer) + "Show excerpt"
   * - otherwise: a button that opens the source dialog (snippet, page)
   */
  function _fillSources(container, citations, msgId) {
    container.textContent = "";
    if (!citations || !citations.length) { container.hidden = true; return; }
    container.hidden = false;

    const types = new Set(citations.map((c) => c.type));
    const icon = types.size === 1 ? (types.has("web") ? "globe" : "file-text") : "link";
    container.setAttribute("role", "group");
    container.setAttribute("aria-label", `Sources (${citations.length})`);

    const head = _el("div", "sources__head");
    head.appendChild(Icons.el(icon, { size: 16 }));
    head.appendChild(_el("span", "sources__title", `Sources (${citations.length})`));
    container.appendChild(head);

    const list = _el("ul", "sources__list");
    citations.forEach((c, i) => {
      const li = _el("li", "source");
      const href = Utils.safeUrl(c.url);
      const domain = href ? new URL(href).hostname.replace(/^www\./, "") : "";
      const title = _citationLabel(c, domain);

      li.appendChild(_el("span", "source__num", String(c.id)));

      const main = _el("div", "source__main");
      let titleEl;
      if (href) {
        titleEl = _el("a", "source__title");
        titleEl.href = href;
        titleEl.target = "_blank";
        titleEl.rel = "noopener noreferrer";
        titleEl.appendChild(_el("span", "source__title-text", title));
        titleEl.appendChild(Icons.el("external-link", { size: 16 }));
        titleEl.appendChild(_el("span", "sr-only", " (opens in a new tab)"));
      } else {
        titleEl = _el("button", "source__title");
        titleEl.type = "button";
        titleEl.appendChild(_el("span", "source__title-text", title));
        titleEl.addEventListener("click", () => UI.showCitationModal(_modalCitation(c)));
      }
      main.appendChild(titleEl);

      const metaBits = [];
      if (domain && domain !== title) metaBits.push(domain);
      if (c.page !== null && c.page !== undefined && c.page !== "") metaBits.push(`Page ${c.page}`);
      if (metaBits.length) main.appendChild(_el("div", "source__meta", metaBits.join(" · ")));

      if (href && c.snippet) {
        const exId = `${msgId}-excerpt-${i}`;
        const tgl = _el("button", "source__excerpt-toggle", "Show excerpt");
        tgl.type = "button";
        tgl.setAttribute("aria-expanded", "false");
        tgl.setAttribute("aria-controls", exId);
        const ex = _el("blockquote", "source__excerpt", String(c.snippet));
        ex.id = exId;
        ex.hidden = true;
        tgl.addEventListener("click", () => {
          const open = tgl.getAttribute("aria-expanded") !== "true";
          tgl.setAttribute("aria-expanded", String(open));
          tgl.textContent = open ? "Hide excerpt" : "Show excerpt";
          ex.hidden = !open;
        });
        main.appendChild(tgl);
        main.appendChild(ex);
      }

      li.appendChild(main);
      list.appendChild(li);
    });
    container.appendChild(list);
  }

  /** Temporary "Copied" state for a copy button. */
  function _flashCopy(btn, ok) {
    clearTimeout(btn._copyTimer);
    btn.innerHTML = Icons.html(ok ? "check" : "circle-alert", { size: 16 });
    btn.setAttribute("aria-label", ok ? "Copied" : "Copy failed");
    _announce(ok ? "Copied to clipboard" : "Copy failed");
    btn._copyTimer = setTimeout(() => {
      btn.innerHTML = Icons.html("copy", { size: 16 });
      btn.setAttribute("aria-label", "Copy response");
    }, 2000);
  }

  /** Render regenerate / feedback buttons into `container` (idempotent). */
  function _renderActionButtons(container, { messageId, feedback, onRegenerate, onFeedback }) {
    if (!container) return;
    container.innerHTML = "";
    const mk = (icon, title, extraClass = "") => {
      const b = document.createElement("button");
      b.className = `btn btn--icon-sm ${extraClass}`.trim();
      b.type = "button";
      b.innerHTML = Icons.html(icon, { size: 16 });
      b.title = title;
      b.setAttribute("aria-label", title);
      return b;
    };

    if (onRegenerate) {
      const regen = mk("rotate-ccw", "Regenerate response", "message__regen");
      regen.addEventListener("click", () => onRegenerate());
      container.appendChild(regen);
    }
    if (onFeedback && messageId) {
      let current = feedback || 0;
      const up = mk("thumbs-up", "Good response");
      const down = mk("thumbs-down", "Poor response");
      const paint = () => {
        up.classList.toggle("is-active", current === 1);
        down.classList.toggle("is-active", current === -1);
        up.setAttribute("aria-pressed", String(current === 1));
        down.setAttribute("aria-pressed", String(current === -1));
      };
      const send = async (value) => {
        const next = current === value ? 0 : value;
        try {
          await onFeedback(messageId, next);
          current = next;
          paint();
        } catch (_) {
          _announce("Feedback could not be saved");
        }
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

  function _hideWelcome() {
    const w = document.getElementById("welcomeScreen");
    if (w) w.hidden = true;
  }

  function _showWelcome() {
    const w = document.getElementById("welcomeScreen");
    const returning = !!(w && w.hidden);
    if (w) w.hidden = false;
    if (window.UI && typeof UI.setVizState === "function") UI.setVizState("idle");
    if (returning && window.Motion && typeof Motion.welcome === "function") Motion.welcome();
  }

  /**
   * Regenerate support (view hook). Hides the newest assistant answer WITHOUT removing it, so the
   * controller can drop it only after the new stream succeeded, or bring it back on failure/Stop.
   * Returns { restore(), discard() } or null when there is no assistant message.
   * Call it BEFORE Chat.startAssistantMessage(): the new answer is appended after the held node.
   */
  function holdLastAnswer() {
    const nodes = document.querySelectorAll("#chatArea .message--assistant");
    const node = nodes[nodes.length - 1];
    if (!node) return null;
    node.hidden = true;
    let done = false;
    return {
      restore() { if (done) return; done = true; node.hidden = false; },
      discard() { if (done) return; done = true; node.remove(); },
    };
  }

  // ── Document preview (one pane, opened from a file card) ──

  let _artifactOpen = false;
  let _artifactOpener = null;
  let _artifactMessageId = "";
  let _artifactDoc = null;
  let _onArtifactDownload = null;
  const _FORMAT_LABEL = { md: "Markdown", txt: "Plain text", xlsx: "Excel", docx: "Word", pdf: "PDF" };
  let _artifactWired = false;
  const _artifactNarrow = window.matchMedia("(max-width: 767px)");

  function _downloadFormat(value) {
    return ["md", "txt", "xlsx", "docx", "pdf"].indexOf(value) >= 0 ? value : "md";
  }

  function fileKeywordFormat(text) {
    const t = String(text || "");
    if (!t) return "";
    if (/\bpdf\b|\.pdf\b/i.test(t)) return "pdf";
    if (/\b(?:docx|word)\b|\.docx\b/i.test(t)) return "docx";
    if (/\b(?:xlsx|xlxs|xls|excel|spreadsheet|csv)\b|\.(?:xlsx|xlxs|xls|csv)\b/i.test(t)) return "xlsx";
    if (/\b(?:plain\s+text|text\s+files?)\b|\.txt\b/i.test(t)) return "txt";
    if (/\bmarkdown\b|\.md\b/i.test(t)) return "md";
    if (/\bdocument\b/i.test(t)) return "md";
    return "";
  }

  function prepareFilePreview(text) {
    const fmt = fileKeywordFormat(text);
    if (!fmt) return;
    openArtifact({
      doc: { title: "Document", blocks: [{ type: "paragraph", text: "Preparing your file…" }] },
      messageId: "",
      format: fmt,
    });
  }

  function _formatLabel(format) {
    return _FORMAT_LABEL[_downloadFormat(format)];
  }

  function _cardKind(format) {
    return _downloadFormat(format) === "xlsx" ? "Spreadsheet" : "Document";
  }

  function _paneTitle(doc, format) {
    return String((doc && doc.title) || "Document") + " " + _formatLabel(format);
  }

  function _mountFileCard(slot, doc, messageId, format) {
    if (!slot) return;
    slot.textContent = "";
    if (!doc || !doc.title || !Array.isArray(doc.blocks) || !doc.blocks.length) {
      slot.hidden = true;
      return;
    }
    slot.hidden = false;
    const chosen = _downloadFormat(format);
    const card = _el("div", "file-card");
    card.setAttribute("role", "group");
    card.setAttribute("aria-label", doc.title);
    card.appendChild(chosen === "xlsx" ? Icons.el("file-spreadsheet", { size: 20 }) : Icons.el("file-text", { size: 20 }));
    const text = _el("div", "file-card__text");
    text.appendChild(_el("div", "file-card__name", String(doc.title)));
    text.appendChild(_el("div", "file-card__type", _cardKind(chosen) + " · " + _formatLabel(chosen)));
    card.appendChild(text);
    const save = _el("button", "btn btn--icon file-card__save");
    save.type = "button";
    save.setAttribute("aria-label", "Download " + doc.title);
    save.appendChild(Icons.el("arrow-down", { size: 16 }));
    save.addEventListener("click", (event) => {
      event.stopPropagation();
      _saveArtifact(save, messageId || "", chosen);
    });
    const btn = _el("button", "btn btn--icon file-card__open");
    btn.type = "button";
    btn.setAttribute("aria-label", "Open preview of " + doc.title);
    btn.appendChild(Icons.el("arrow-right", { size: 16 }));
    const open = () => openArtifact({ doc, messageId: messageId || "", opener: btn, format: chosen });
    btn.addEventListener("click", (event) => { event.stopPropagation(); open(); });
    card.addEventListener("click", open);
    card.appendChild(save);
    card.appendChild(btn);
    slot.appendChild(card);
  }

  function _saveArtifact(btn, messageId, format) {
    if (typeof _onArtifactDownload !== "function") return;
    btn.disabled = true;
    Promise.resolve(_onArtifactDownload(messageId, format)).finally(() => {
      if (btn.isConnected) btn.disabled = false;
    });
  }

  function _artifactFocusables(root) {
    const sel = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    return Array.from(root.querySelectorAll(sel)).filter((n) => n.getClientRects().length > 0);
  }

  function _setChatCovered(cover) {
    const main = document.getElementById("main");
    if (!main || !("inert" in main)) return;
    const sidebarOpen = window.UI && typeof UI.isSidebarOpen === "function" && UI.isSidebarOpen();
    main.inert = !!(cover || sidebarOpen);
  }

  function _syncArtifactLayer() {
    const pane = document.getElementById("artifactPane");
    if (!pane || !_artifactOpen) return;
    if (_artifactNarrow.matches) {
      pane.setAttribute("role", "dialog");
      pane.setAttribute("aria-modal", "true");
    } else {
      pane.setAttribute("role", "complementary");
      pane.removeAttribute("aria-modal");
    }
    _setChatCovered(_artifactNarrow.matches);
  }

  function _paintArtifactPreview(doc, format) {
    const body = document.getElementById("artifactBody");
    const title = document.getElementById("artifactTitle");
    if (!body || !doc) return;
    const fmt = _downloadFormat(format);
    if (title) title.textContent = _paneTitle(doc, fmt);
    body.className = "artifact-pane__body artifact-pane__body--" + fmt;
    body.innerHTML = Utils.renderArtifactHtml(doc, fmt);
    if (window.Utils && typeof Utils.highlightCode === "function") Utils.highlightCode(body);
  }

  function openArtifact({ doc, messageId, opener, format }) {
    _wireArtifactPane();
    // Close code preview if it happens to be open (only one pane at a time)
    if (window.CodePreview) CodePreview.close();
    const pane = document.getElementById("artifactPane");
    const body = document.getElementById("artifactBody");
    const title = document.getElementById("artifactTitle");
    const scrim = document.getElementById("artifactScrim");
    if (!pane || !body || !title || !doc) return;
    _artifactOpener = opener || null;
    _artifactMessageId = messageId || "";
    _artifactDoc = doc;
    const formatEl = document.getElementById("artifactFormat");
    if (formatEl) formatEl.value = _downloadFormat(format);
    _paintArtifactPreview(doc, format);
    pane.classList.add("is-open");
    pane.setAttribute("aria-hidden", "false");
    pane.inert = false;
    if (scrim) scrim.classList.add("is-open");
    _artifactOpen = true;
    _syncArtifactLayer();
    const closeBtn = document.getElementById("artifactClose");
    if (closeBtn) closeBtn.focus();
  }

  function closeArtifact() {
    if (!_artifactOpen) return;
    const pane = document.getElementById("artifactPane");
    const scrim = document.getElementById("artifactScrim");
    const opener = _artifactOpener;
    if (pane) {
      pane.classList.remove("is-open");
      pane.setAttribute("aria-hidden", "true");
      pane.setAttribute("role", "complementary");
      pane.removeAttribute("aria-modal");
      pane.inert = true;
    }
    if (scrim) scrim.classList.remove("is-open");
    _artifactOpen = false;
    _setChatCovered(false);
    if (opener && document.contains(opener) && typeof opener.focus === "function") opener.focus();
  }

  function _onArtifactKey(e) {
    if (!_artifactOpen) return;
    if (e.key === "Escape") {
      if (document.querySelector("dialog[open]")) return;
      if (window.Menu && typeof Menu.isOpen === "function" && Menu.isOpen()) return;
      e.preventDefault();
      e.stopPropagation();
      closeArtifact();
      return;
    }
    if (e.key !== "Tab" || !_artifactNarrow.matches) return;
    const pane = document.getElementById("artifactPane");
    if (!pane) return;
    const nodes = _artifactFocusables(pane);
    if (!nodes.length) { e.preventDefault(); return; }
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    const active = document.activeElement;
    const inside = pane.contains(active);
    if (!inside || (e.shiftKey && active === first) || (!e.shiftKey && active === last)) {
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  }

  function _downloadArtifact(btn) {
    const formatEl = document.getElementById("artifactFormat");
    const format = formatEl ? formatEl.value : "md";
    if (_artifactDoc) {
      const title = document.getElementById("artifactTitle");
      if (title) title.textContent = _paneTitle(_artifactDoc, format);
    }
    _saveArtifact(btn, _artifactMessageId, format);
  }

  const ARTIFACT_W_KEY = "artifactWidth";
  const ARTIFACT_W_MIN = 280;

  function _artifactMax() {
    return Math.max(ARTIFACT_W_MIN, Math.round(window.innerWidth * 0.75));
  }

  function _clampArtifactWidth(px) {
    return Math.max(ARTIFACT_W_MIN, Math.min(_artifactMax(), Math.round(px)));
  }

  function _applyArtifactWidth(px, persist) {
    const pane = document.getElementById("artifactPane");
    const handle = document.getElementById("artifactResize");
    if (!pane) return;
    const w = _clampArtifactWidth(px);
    pane.style.setProperty("--artifact-w", w + "px");
    if (handle) {
      handle.setAttribute("aria-valuenow", String(w));
      handle.setAttribute("aria-valuemin", String(ARTIFACT_W_MIN));
      handle.setAttribute("aria-valuemax", String(_artifactMax()));
    }
    if (persist) {
      try { localStorage.setItem(ARTIFACT_W_KEY, String(w)); } catch (_) { /* private mode */ }
    }
  }

  function _resetArtifactWidth() {
    const pane = document.getElementById("artifactPane");
    if (pane) pane.style.removeProperty("--artifact-w");
    try { localStorage.removeItem(ARTIFACT_W_KEY); } catch (_) { /* private mode */ }
    const handle = document.getElementById("artifactResize");
    if (handle && pane) {
      const w = Math.round(pane.getBoundingClientRect().width);
      handle.setAttribute("aria-valuenow", String(w));
    }
  }

  function _restoreArtifactWidth() {
    try {
      const v = parseInt(localStorage.getItem(ARTIFACT_W_KEY), 10);
      if (Number.isFinite(v)) _applyArtifactWidth(v, false);
    } catch (_) { /* private mode */ }
  }

  function _wireArtifactResize(handle) {
    if (!handle || handle.dataset.wired) return;
    handle.dataset.wired = "1";
    handle.setAttribute("aria-valuemin", String(ARTIFACT_W_MIN));
    handle.addEventListener("pointerdown", (e) => {
      if (_artifactNarrow.matches || e.button) return;
      const pane = document.getElementById("artifactPane");
      if (!pane) return;
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startW = pane.getBoundingClientRect().width;
      const move = (ev) => { _applyArtifactWidth(startW + (startX - ev.clientX), false); };
      const up = (ev) => {
        handle.releasePointerCapture(ev.pointerId);
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        _applyArtifactWidth(pane.getBoundingClientRect().width, true);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", () => { _resetArtifactWidth(); });
    handle.addEventListener("keydown", (e) => {
      if (_artifactNarrow.matches) return;
      const pane = document.getElementById("artifactPane");
      if (!pane) return;
      const w = pane.getBoundingClientRect().width;
      if (e.key === "ArrowLeft") { e.preventDefault(); _applyArtifactWidth(w + 16, true); }
      else if (e.key === "ArrowRight") { e.preventDefault(); _applyArtifactWidth(w - 16, true); }
      else if (e.key === "Home") { e.preventDefault(); _resetArtifactWidth(); }
    });
  }

  function _wireArtifactPane() {
    if (_artifactWired) return;
    const pane = document.getElementById("artifactPane");
    const scrim = document.getElementById("artifactScrim");
    const closeBtn = document.getElementById("artifactClose");
    const downloadBtn = document.getElementById("artifactDownload");
    if (!pane || !scrim || !closeBtn || !downloadBtn) return;
    _artifactWired = true;
    closeBtn.addEventListener("click", closeArtifact);
    scrim.addEventListener("click", closeArtifact);
    downloadBtn.addEventListener("click", () => _downloadArtifact(downloadBtn));
    const formatEl = document.getElementById("artifactFormat");
    if (formatEl) formatEl.addEventListener("change", () => {
      if (_artifactDoc) _paintArtifactPreview(_artifactDoc, formatEl.value);
    });
    document.addEventListener("keydown", _onArtifactKey, true);
    if (_artifactNarrow.addEventListener) _artifactNarrow.addEventListener("change", _syncArtifactLayer);
    _restoreArtifactWidth();
    _wireArtifactResize(document.getElementById("artifactResize"));
  }

  /** Controller hook: onDownload(messageId, format) builds the file. The pane never fetches. */
  function bindArtifact(handlers) {
    _onArtifactDownload = handlers && typeof handlers.onDownload === "function" ? handlers.onDownload : null;
    _wireArtifactPane();
  }

  _wireArtifactPane();

  return {
    appendUserMessage,
    holdLastAnswer,
    startAssistantMessage,
    showThinking,
    removeThinking,
    renderHistory,
    clearChat,
    copyMessage,
    getCitation,
    bindArtifact,
    closeArtifact,
    fileKeywordFormat,
    prepareFilePreview,
  };
})();
