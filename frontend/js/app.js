/* ============================================================
   app.js — Main state orchestrator & event wiring
   Initializes all modules, manages global state and drives the
   single-round-trip streaming chat pipeline (POST /chat/stream).
   ============================================================ */

(function () {
  "use strict";

  // ── Global state ───────────────────────────────────────────
  const State = {
    userId:     localStorage.getItem("userId")    || "default",
    sessionId:  null,
    mode:       localStorage.getItem("mode")      || "general",
    model:      localStorage.getItem("model")     || "",
    defaultModel: "",
    models:     [],
    sessions:   [],
    documents:  [],
    webSearch:  true,    // "auto": the server only searches for time-sensitive queries
    isLoading:  false,
    abort:      null,    // AbortController of the in-flight request
    failed:     null,    // { userEl, assistantEl, text, sessionId } of the last accepted-then-failed send
    busyToastAt: 0,      // last "response in progress" toast (epoch ms), throttles N1 feedback
    cooldownUntil: 0,    // epoch ms; set from Retry-After / error.retry_after
    cooldownTimer: null,
    uploading:  false,
    sessionsLoaded: false,    // false until the first GET /sessions succeeds (drives the loading / error state)
    documentsLoaded: false,   // false until the first GET /documents succeeds (drives the loading / error state)
    limits:     { maxMb: 0, extensions: null },   // from GET /config; unknown on an older backend
  };

  const MAX_TITLE = 200;  // SessionUpdate.title max_length (backend/models/api.py)

  // ── Init ───────────────────────────────────────────────────
  async function init() {
    UI.initTheme();
    UI.setActiveMode(State.mode);
    UI.setUserIdDisplay(State.userId);
    // Start loading BEFORE the handlers are bound. The requests do not depend on any handler, so a failure while
    // binding (for example, a stale cached script that lacks a function this file expects) can never leave the model
    // list on "Loading models…" or the chat list empty. Each loader reports its own errors.
    const loading = Promise.all([_loadModels(), _loadSessions({ notify: true }), _loadDocuments(), _loadConfig()]);
    Chat.bindArtifact({
      onDownload(messageId, format) {
        if (!messageId) {
          UI.showToast("This document could not be saved.", "error");
          return;
        }
        return API.downloadArtifact(messageId, format, State.userId).catch((err) => {
          UI.showToast(err.message || "Download failed.", "error");
        });
      },
    });
    _bindEvents();
    await loading;
  }

  /** Startup threw: say so, with the fix, instead of leaving a half-working page that looks merely slow. */
  function _startupFailed(err) {
    console.error("[app] Startup failed:", err);
    const msg = "Part of the page failed to start" + (err && err.message ? ` (${err.message})` : "") +
      ". Reload with Ctrl+Shift+R (Cmd+Shift+R on Mac) to load the latest files.";
    if (window.UI && typeof UI.showToast === "function") UI.showToast(msg, "error");
  }

  // ── Models (driven by the server config) ───────────────────
  async function _loadModels() {
    let failure = null;
    try {
      const data = await API.getModels();
      State.models = data.models || [];
      State.defaultModel = data.default || "";
    } catch (err) {
      failure = err;
      State.models = [];
    }
    if (!failure) {
      const usable = State.models.filter((m) => m.available);
      const ids = new Set(usable.map((m) => m.id));
      if (!ids.has(State.model)) {
        State.model = ids.has(State.defaultModel) ? State.defaultModel : (usable[0] ? usable[0].id : "");
        if (State.model) localStorage.setItem("model", State.model);
      }
    }
    // Third argument is the C7 hook (ignored by a view that does not implement it yet)
    UI.renderModelSelect(State.models, State.model, failure
      ? { error: "Models unavailable", onRetry: _loadModels }
      : undefined);
    if (failure) UI.showToast(`Could not load models: ${failure.message}`, "error");
  }

  // ── Event wiring ───────────────────────────────────────────
  function _bindEvents() {

    document.getElementById("sendBtn").addEventListener("click", () => _onSend());
    document.getElementById("stopBtn").addEventListener("click", _onStop);

    // Textarea — Enter to send, Shift+Enter for newline, auto-resize
    const textarea = document.getElementById("chatInput");
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        _onSend();
      }
    });
    textarea.addEventListener("input", () => _autoResize(textarea));

    // Workspace mode: the sidebar dropdown reports the chosen mode (the view never changes it itself)
    UI.onModeChange(_setMode);

    // Model selector
    document.getElementById("modelSelect").addEventListener("change", (e) => {
      State.model = e.target.value;
      localStorage.setItem("model", State.model);
      if (State.sessionId) API.updateSession(State.sessionId, { model: State.model }).catch(() => {});
    });

    document.getElementById("newChatBtn").addEventListener("click", _newChat);
    document.getElementById("sidebarToggle").addEventListener("click", UI.toggleSidebar);

    // Attach file
    document.getElementById("attachBtn").addEventListener("click", () => {
      document.getElementById("fileInput").click();
    });
    document.getElementById("fileInput").addEventListener("change", (e) => {
      const files = Array.from(e.target.files || []);
      e.target.value = "";  // reset so same file can be re-selected
      if (files.length) _handleFiles(files);
    });

    // Add URL
    document.getElementById("addUrlBtn").addEventListener("click", UI.showUrlModal);
    document.getElementById("urlModalClose").addEventListener("click", UI.hideUrlModal);
    document.getElementById("urlModalCancel").addEventListener("click", UI.hideUrlModal);
    document.getElementById("urlModalSubmit").addEventListener("click", _onUrlSubmit);
    document.getElementById("urlInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.isComposing) _onUrlSubmit();
    });
    document.getElementById("urlModal").addEventListener("click", (e) => {
      if (e.target === document.getElementById("urlModal")) UI.hideUrlModal();
    });

    document.getElementById("addGithubBtn").addEventListener("click", UI.showGithubModal);
    document.getElementById("githubModalClose").addEventListener("click", () => { _resetGithubOauth(); UI.hideGithubModal(); });
    document.getElementById("githubModalCancel").addEventListener("click", () => { _resetGithubOauth(); UI.hideGithubModal(); });
    document.getElementById("githubModalSubmit").addEventListener("click", _onGithubSubmit);
    document.getElementById("githubOauth").addEventListener("click", _onGithubOauth);
    document.getElementById("githubUrl").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.isComposing) _onGithubSubmit();
    });
    document.getElementById("githubModal").addEventListener("click", (e) => {
      if (e.target === document.getElementById("githubModal")) { _resetGithubOauth(); UI.hideGithubModal(); }
    });

    // Web search toggle (the view styles [aria-pressed])
    const searchToggle = document.getElementById("searchToggle");
    searchToggle.setAttribute("aria-pressed", String(State.webSearch));
    searchToggle.addEventListener("click", () => {
      // Research always searches the web on the server: the view locks the switch there
      if (searchToggle.getAttribute("aria-disabled") === "true") {
        UI.showToast("Web search is always on in Research.", "info");
        return;
      }
      State.webSearch = !State.webSearch;
      searchToggle.setAttribute("aria-pressed", String(State.webSearch));
      UI.showToast(State.webSearch ? "Web search enabled" : "Web search disabled", "info");
    });

    // Citation modal close
    document.getElementById("citationModalClose").addEventListener("click", UI.hideCitationModal);
    document.getElementById("citationModal").addEventListener("click", (e) => {
      if (e.target === document.getElementById("citationModal")) UI.hideCitationModal();
    });

    // Session search filter
    document.getElementById("sessionSearch").addEventListener("input", () => _renderSessions());

    UI.initDropzone(_handleFiles);

    // Welcome feature cards: the view reports {mode, prompt} via UI.onWelcomeCard (and suppresses
    // its own default handling once a handler is registered). We switch mode and fill the composer;
    // this NEVER creates a session or sends.
    UI.onWelcomeCard(({ mode, prompt }) => {
      if (mode) _setMode(mode);
      _fillComposer(prompt);
      if (mode === "document") _openFilePicker();   // runs inside the click/keypress activation
    });

    // "My Documents" (composer add menu): the view opens the panel; refresh the list from GET /documents.
    // The mode is not switched: documents are searched in every workspace.
    UI.onMyDocuments(() => { _loadDocuments(); });

    // Settings > Integrations shows the live GET /health status
    UI.onSettingsPane((pane) => { if (pane === "integrations") _loadServices(); });
  }

  /** Open the native file chooser via the existing #fileInput. Only call from a user-gesture handler. Never sends. */
  function _openFilePicker() {
    const input = document.getElementById("fileInput");
    if (input) input.click();
  }

  function _setMode(mode) {
    if (!mode) return;
    const changed = mode !== State.mode;
    State.mode = mode;
    localStorage.setItem("mode", mode);
    UI.setActiveMode(mode);
    if (changed && State.sessionId) API.updateSession(State.sessionId, { mode }).catch(() => {});
  }

  /** Put text in the composer, resize and focus it. Sends nothing. */
  function _fillComposer(text) {
    const textarea = document.getElementById("chatInput");
    if (text) textarea.value = text;
    _autoResize(textarea);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }

  // ── Chat pipeline (one request: route → retrieve/search → generate → persist) ──
  const STAGE_LABELS = {
    rewriting:  { icon: "pencil",      pending: "Understanding your follow-up…" },
    retrieving: { icon: "file-search", pending: "Searching documents…" },
    searching:  { icon: "globe",       pending: "Searching the web…" },
    generating: { icon: "sparkles",    pending: "Generating response…" },
  };

  /** Rate-limit cooldown from Retry-After / error.retry_after: blocks tight retry loops. */
  function _startCooldown(seconds, { announce = true } = {}) {
    clearTimeout(State.cooldownTimer);
    State.cooldownTimer = null;
    State.cooldownUntil = 0;
    if (!seconds) return;
    State.cooldownUntil = Date.now() + seconds * 1000;
    State.cooldownTimer = setTimeout(() => {
      State.cooldownUntil = 0;
      State.cooldownTimer = null;
      if (announce) UI.setLiveStatus("You can send again.");
    }, seconds * 1000);
  }

  function _cooldownLeft() {
    return Math.max(0, Math.ceil((State.cooldownUntil - Date.now()) / 1000));
  }

  /** Re-insert text that was not delivered, without overwriting anything typed since. */
  function _restoreDraft(text) {
    const textarea = document.getElementById("chatInput");
    if (!text || textarea.value.includes(text)) return;
    textarea.value = textarea.value.trim() ? `${text}\n\n${textarea.value}` : text;
    _autoResize(textarea);
  }

  function _discardFailed() {
    if (State.failed) {
      [State.failed.userEl, State.failed.assistantEl].forEach((n) => { if (n) n.remove(); });
      State.failed = null;
    }
  }

  function _requestErrorMessage(err, regenerate) {
    let msg;
    if (err.status) {
      msg = err.message || `Request failed (HTTP ${err.status}).`;
      if (err.retryAfter) msg += ` Try again in ${err.retryAfter}s.`;
    } else {
      msg = "Could not reach the server. Check your connection and try again.";
    }
    return msg + (regenerate ? " Your previous answer is unchanged." : " Your message is kept.");
  }

  /** N1: tell the user why Send/Enter/session-select does nothing while a response streams (throttled). */
  function _notifyBusy() {
    const now = Date.now();
    if (now - State.busyToastAt < 4000) return;
    State.busyToastAt = now;
    UI.showToast("A response is in progress. Press Stop to cancel.", "info");
  }

  async function _onSend({ regenerate = false } = {}) {
    if (State.isLoading) { _notifyBusy(); return; }
    const textarea = document.getElementById("chatInput");
    const draft = regenerate ? "" : textarea.value;
    const draftText = draft.trim();
    if (!regenerate && !draftText) return;

    // M1: the user turn of a failed send is already stored server-side. Re-sending the restored draft
    // unedited (same session) goes through the regenerate path, exactly like Retry, so it is never stored twice.
    const prior = State.failed;
    const resend = !regenerate && !!prior && prior.sessionId === State.sessionId && draftText === prior.text;
    if (resend) regenerate = true;
    const text = resend ? "" : draftText;   // the message actually posted ("" for regenerate/resend)

    const wait = _cooldownLeft();
    if (wait > 0) {
      UI.showToast(`The server asked you to wait. Try again in ${wait}s.${regenerate && !resend ? "" : " Your message is kept."}`, "info");
      return;
    }

    // The composer is NOT cleared here: it is cleared only once the server accepts the request.
    _setLoading(true);
    State.abort = new AbortController();
    const signal = State.abort.signal;

    let renderer = null;
    let userEl = resend ? prior.userEl : null;
    let accepted = false;
    let persisted = false;     // true only once an SSE event proved the server stored the user turn (meta is first)
    let oldAnswer = null;      // regenerate: the removed previous answer, kept to put back if the stream is stopped early
    let sentSession = null;

    const steps = {};          // stage -> { el, t0 }
    let citations = [];
    let modelInfo = { model: null, fellBackFrom: [] };
    let doneInfo = null;
    let streamError = null;

    const startStep = (stage) => {
      const cfg = STAGE_LABELS[stage];
      if (!cfg || steps[stage]) return;
      steps[stage] = { el: renderer.addThought(cfg.icon, cfg.pending, "pending"), t0: Date.now() };
    };
    const finishStep = (stage, label, status = "done", elapsedMs = null) => {
      const step = steps[stage];
      if (!step) return;
      const ms = elapsedMs != null ? elapsedMs : Date.now() - step.t0;
      renderer.resolveThought(step.el, {
        icon: status === "error" ? "circle-alert" : "circle-check",
        label,
        elapsed: (ms / 1000).toFixed(1) + "s",
        status,
      });
    };

    // The server accepted the request (HTTP 2xx): now the DOM may change and the draft may go.
    const onAccepted = () => {
      if (resend) {
        // Replace only the failed attempt's error bubble (N2 may have removed it: then remove nothing)
        if (prior.assistantEl) prior.assistantEl.remove();
        State.failed = null;
        if (textarea.value === draft) {
          textarea.value = "";
          _autoResize(textarea);
        }
      } else if (regenerate) {
        // Replace the previous answer only now that a new stream is starting
        const all = document.querySelectorAll("#chatArea .message--assistant");
        if (all.length) {
          const node = all[all.length - 1];
          oldAnswer = { node, parent: node.parentNode, prev: node.previousSibling, next: node.nextSibling };
          node.remove();
        }
        State.failed = null;
      } else {
        _discardFailed();
        userEl = Chat.appendUserMessage(text, State.userId);
        if (textarea.value === draft) {
          textarea.value = "";
          _autoResize(textarea);
        }
      }
      // Only the newest answer can be regenerated
      document.querySelectorAll("#chatArea .message__regen").forEach((b) => b.remove());
      renderer = Chat.startAssistantMessage();
      accepted = true;
      if (text) Chat.prepareFilePreview(text);
    };

    /** Remove this send's assistant bubble and, if the server never stored the turn, the optimistic user bubble. */
    const _dropUnsent = () => {
      if (renderer) {
        const el = document.querySelector(`#chatArea [data-msg-id="${renderer.msgId}"]`);
        if (el) el.remove();
      }
      if (!resend && !persisted && userEl) { userEl.remove(); userEl = null; }
    };

    /** Failure after the stream was accepted and before any text: error + retry, draft restored. */
    const failAfterAccept = (message, { retryable = true, retryAfter = null } = {}) => {
      const wait = Number(retryAfter) > 0 ? Number(retryAfter) : 0;
      if (!regenerate && !persisted) {
        // Accepted (HTTP 200) but no event arrived: the user turn may not be stored. Treat as not sent:
        // drop both bubbles, restore the draft; a resend is a NORMAL send (never regenerate).
        _startCooldown(retryAfter);
        _dropUnsent();
        State.failed = null;
        _restoreDraft(draftText);
        UI.showToast(`${message} Your message is kept.${wait ? ` Try again in ${wait}s.` : ""}`, "error");
        return;
      }
      _startCooldown(retryAfter, { announce: !(retryable && wait) });   // the Retry countdown announces itself
      renderer.showError(message, retryable ? { onRetry: _retryAfterError, retryAfter: wait } : {});
      if (!regenerate || resend) {
        _restoreDraft(draftText);
        const assistantEl = document.querySelector(`#chatArea [data-msg-id="${renderer.msgId}"]`);
        State.failed = { userEl, assistantEl, text: draftText, sessionId: State.sessionId };
      }
    };

    // Every event except `error` is yielded after the server stored the user turn (`meta` is first).
    const _persistedOnEvent = (h) => {
      const out = {};
      Object.keys(h).forEach((k) => {
        out[k] = (d) => { if (k !== "error") persisted = true; return h[k](d); };
      });
      return out;
    };

    try {
      if (!State.sessionId) await _ensureSession(signal);
      if (signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
      sentSession = State.sessionId;

      await API.chatStream(
        {
          session_id: State.sessionId,
          message: text,
          user_id: State.userId,
          mode: State.mode,
          model: State.model || null,
          web_search: State.webSearch,
          regenerate,
        },
        _persistedOnEvent({
          meta: () => { renderer.setVizState("processing"); },  // route/mode/reason: steps come from status events
          status: (d) => {
            if (d.state === "start") {
              startStep(d.stage);
            } else if (d.state === "done") {
              const failed = !!d.error;
              const labels = {
                rewriting:  "Rewrote follow-up as a standalone question",
                retrieving: failed ? "Document search failed" : `Retrieved ${d.detail || "excerpts"}`,
                searching:  failed ? `Web search unavailable (${d.error})` : `Found ${d.detail || "sources"}`,
              };
              finishStep(d.stage, labels[d.stage] || d.detail || "Done", failed ? "error" : "done", d.elapsed_ms);
            } else if (d.state === "first_token") {
              finishStep("generating", `Responding with ${d.detail || "model"}`, "done", d.elapsed_ms);
            }
          },
          citations: (d) => { citations = d.items || []; },
          model: (d) => { modelInfo = { model: d.model, fellBackFrom: d.fell_back_from || [] }; },
          token: (d) => { renderer.appendToken(d.text); },
          artifact: (d) => { renderer.attachArtifact(d); },
          error: (d) => { streamError = d; },
          done: (d) => { doneInfo = d; },
        }),
        signal,
        { onOpen: onAccepted },
      );

      const hasText = !!renderer.getRawText().trim();
      const finish = () => {
        renderer.finalize(citations);
        renderer.setBadges({
          model: modelInfo.model || (doneInfo && doneInfo.model) || null,
          cached: !!(doneInfo && doneInfo.cached),
          fellBackFrom: modelInfo.fellBackFrom.length ? modelInfo.fellBackFrom : ((doneInfo && doneInfo.fell_back_from) || []),
        });
      };

      if (streamError && !hasText) {
        finishStep("generating", "Generation failed", "error");
        failAfterAccept(streamError.message || "Something went wrong. Please try again.", {
          retryable: streamError.retryable !== false,
          retryAfter: streamError.retry_after || null,
        });
      } else if (streamError) {
        // Partial answer: keep it, say it is incomplete, offer actions if it was persisted
        _startCooldown(streamError.retry_after || null);
        finish();
        renderer.markPartial({ reason: "error", message: streamError.message });
        renderer.enableActions({ messageId: streamError.message_id, onRegenerate: _regenerateLast, onFeedback: _sendFeedback });
      } else if (!doneInfo) {
        // Connection closed without a terminal event
        if (hasText) {
          finish();
          renderer.markPartial({ reason: "error", message: "The connection closed before the response finished." });
        } else {
          failAfterAccept("The connection closed before a response arrived.");
        }
      } else {
        finish();
        renderer.enableActions({
          messageId: doneInfo.message_id,
          onRegenerate: _regenerateLast,
          onFeedback: _sendFeedback,
        });
        if (doneInfo.persisted === false) UI.showToast("This answer could not be saved to your chat history.", "info");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        if (accepted && renderer && !renderer.getRawText().trim() && (!regenerate || resend)) {
          // N2: stopped before the first token = pre-token failure. Drop the empty bubble, keep the draft.
          renderer.finalize(citations);
          _dropUnsent();
          if (State.sessionId === sentSession) {
            _restoreDraft(draftText);
            if (resend || persisted) {
              State.failed = { userEl, assistantEl: null, text: draftText, sessionId: sentSession };
            } else {
              State.failed = null;   // nothing proved the turn was stored: a resend is a normal send
            }
            UI.showToast("Generation stopped. Your message is kept; press Send to try again.", "info");
          } else {
            UI.showToast("Generation stopped.", "info");
          }
        } else if (accepted && renderer && !renderer.getRawText().trim()) {
          // Regenerate stopped before the first token: drop the empty bubble and put the previous answer back
          renderer.finalize(citations);
          const el = document.querySelector(`#chatArea [data-msg-id="${renderer.msgId}"]`);
          if (el) el.remove();
          // Restore only into the same, untouched chat; after New chat / select / delete the saved node is discarded
          const sameChat = State.sessionId === sentSession
            && oldAnswer && oldAnswer.parent && oldAnswer.parent.isConnected
            && (!oldAnswer.prev || oldAnswer.prev.parentNode === oldAnswer.parent);
          if (sameChat) {
            const ref = oldAnswer.next && oldAnswer.next.parentNode === oldAnswer.parent ? oldAnswer.next : null;
            oldAnswer.parent.insertBefore(oldAnswer.node, ref);
            UI.showToast("Generation stopped. Your previous answer is restored.", "info");
          } else {
            UI.showToast("Generation stopped.", "info");
          }
          oldAnswer = null;
        } else if (accepted && renderer) {
          // Stop pressed: the server persists the partial answer
          renderer.finalize(citations);
          renderer.setBadges({ model: modelInfo.model, fellBackFrom: modelInfo.fellBackFrom });
          renderer.markPartial({ reason: "stopped" });
          UI.showToast("Generation stopped", "info");
        } else {
          UI.showToast("Request cancelled. Your message is kept.", "info");
        }
      } else if (!accepted) {
        // Session creation / HTTP error (429, 503, 404, network): nothing was sent, nothing in the DOM changed
        console.error("[app] Request failed:", err);
        _startCooldown(err.retryAfter || null);
        if (err.status === 404 && State.sessionId) State.sessionId = null;   // stale session: next send creates a new one
        const prefix = State.sessionId ? "" : "Could not start a chat: ";
        UI.showToast(prefix + _requestErrorMessage(err, regenerate && !resend), "error");
      } else {
        console.error("[app] Pipeline error:", err);
        const message = err.message || "Something went wrong. Please try again.";
        if (renderer.getRawText().trim()) {
          renderer.finalize(citations);
          renderer.setBadges({ model: modelInfo.model, fellBackFrom: modelInfo.fellBackFrom });
          renderer.markPartial({ reason: "error", message });
        } else {
          failAfterAccept(message);
        }
      }
    } finally {
      State.abort = null;
      _setLoading(false);
      _loadSessions();
    }
  }

  function _onStop() {
    if (State.abort) State.abort.abort();
  }

  function _sendFeedback(messageId, value) {
    return API.setFeedback(messageId, value);
  }

  /** Retry from an inline error: the user turn is already on the server, so regenerate instead of re-posting it. */
  function _retryAfterError() {
    const wait = _cooldownLeft();
    if (wait > 0) {
      UI.showToast(`Retry is available in ${wait}s.`, "info");
      return;
    }
    if (State.failed) {
      const textarea = document.getElementById("chatInput");
      if (textarea.value.trim() === State.failed.text) {
        textarea.value = "";
        _autoResize(textarea);
      }
      State.failed = null;   // the user bubble stays: the server has that turn
    }
    return _onSend({ regenerate: true });
  }

  /** Regenerate the last answer. The old answer stays in the DOM until the new stream is accepted. */
  function _regenerateLast() {
    return _onSend({ regenerate: true });
  }

  // ── Session management ─────────────────────────────────────
  async function _ensureSession(signal) {
    const session = await API.createSession({
      user_id: State.userId,
      title: "New Chat",
      model: State.model || null,
      mode: State.mode,
    }, signal);
    // If Stop raced with a creation that completed anyway, the session exists server-side: keep it
    // so the next send reuses it instead of orphaning it (the caller then sees signal.aborted).
    if (!State.sessionId) State.sessionId = session.id;
  }

  async function _newChat() {
    if (State.isLoading) _onStop();
    State.sessionId = null;
    State.failed = null;
    Chat.clearChat();
    try {
      await _ensureSession();
      await _loadSessions();
    } catch (err) {
      console.error("[app] New chat failed:", err);
      UI.notify(`Could not start a new chat: ${err.message}`, "error");
    }
    UI.setActiveMode(State.mode);
  }

  async function _selectSession(sessionId) {
    if (State.isLoading) { _notifyBusy(); return; }
    try {
      const session = await API.getSession(sessionId);
      State.sessionId = sessionId;
      State.failed = null;
      Chat.renderHistory(session.messages || [], { onFeedback: _sendFeedback, onRegenerate: _regenerateLast });
      State.mode = session.mode || "general";
      UI.setActiveMode(State.mode);
      const available = State.models.find((m) => m.id === session.model && m.available);
      if (available) {
        State.model = session.model;
        const sel = document.getElementById("modelSelect");
        if (sel) sel.value = State.model;
      }
      _renderSessions();
    } catch (err) {
      UI.showToast("Failed to load session", "error");
    }
  }

  async function _deleteSession(sessionId) {
    const s = State.sessions.find((x) => x.id === sessionId);
    const ok = await UI.confirm({
      title: "Delete this chat?",
      message: s && s.title ? `"${s.title}" and its messages will be permanently deleted.` : "This chat and its messages will be permanently deleted.",
      confirmLabel: "Delete",
      cancelLabel: "Cancel",
      destructive: true,
    });
    if (!ok) return;
    try {
      if (State.sessionId === sessionId && State.isLoading) _onStop();
      await API.deleteSession(sessionId);
      if (State.sessionId === sessionId) {
        State.sessionId = null;
        Chat.clearChat();
      }
      await _loadSessions();
    } catch (err) {
      UI.showToast("Delete failed", "error");
    }
  }

  /** Rename via PATCH /sessions/{id} {title}. Rejects on failure so the view reverts the inline field. */
  async function _renameSession(sessionId, rawTitle) {
    const title = String(rawTitle == null ? "" : rawTitle).trim();
    if (!title || title.length > MAX_TITLE) {
      UI.showToast(`Chat names must be 1-${MAX_TITLE} characters.`, "error");
      throw new Error("invalid title");
    }
    try {
      const updated = await API.updateSession(sessionId, { title });
      const s = State.sessions.find((x) => x.id === sessionId);
      if (s) s.title = (updated && updated.title) || title;
      _renderSessions();
    } catch (err) {
      UI.showToast(`Could not rename chat: ${err.message}. The old name was kept.`, "error");
      throw err;
    }
  }

  function _renderSessions() {
    const input = document.getElementById("sessionSearch");
    const q = (input && input.value ? input.value : "").toLowerCase();
    const list = q ? State.sessions.filter((s) => (s.title || "").toLowerCase().includes(q)) : State.sessions;
    UI.renderSessionList(list, State.sessionId, _selectSession, _deleteSession, {
      onRename: _renameSession,
      onRetry: () => _loadSessions({ notify: true }),
    });
  }

  async function _loadSessions({ notify = false } = {}) {
    if (!State.sessionsLoaded) UI.setSessionsStatus("loading");
    try {
      const sessions = await API.listSessions(State.userId);
      State.sessions = sessions || [];
      State.sessionsLoaded = true;
      _renderSessions();
    } catch (err) {
      // Keep a list that was already shown; the error state replaces only an empty (never loaded) one
      if (!State.sessionsLoaded) UI.setSessionsStatus("error", { message: err.message, onRetry: () => _loadSessions({ notify: true }) });
      if (notify) UI.showToast(`Could not load chats: ${err.message}`, "error");
    }
  }

  // ── Document handling ──────────────────────────────────────
  const ALLOWED = ["pdf", "docx", "pptx", "odt", "rtf", "csv", "xlsx", "xls", "txt", "md", "rst", "html", "htm", "json", "log", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"];

  function _fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  /** Optional server-advertised limits (GET /config). Without them the server still enforces its own. */
  async function _loadConfig() {
    try {
      const cfg = await API.getConfig();
      const up = cfg && cfg.upload;
      if (up) {
        State.limits = {
          maxMb: Number(up.max_mb) > 0 ? Number(up.max_mb) : 0,
          extensions: Array.isArray(up.extensions) ? up.extensions.map((x) => String(x).toLowerCase()) : null,
        };
        UI.setUploadLimits(State.limits.maxMb);
      }
      UI.setGithubOauth(!!(cfg && cfg.github_oauth));
    } catch (_) {
      // older backend without /config: no client-side size pre-check
    }
  }

  /** Upload several files one after another. Stops at the first failure so its Retry stays on screen. */
  async function _handleFiles(files) {
    const list = Array.from(files || []);
    if (!list.length) return;
    if (State.uploading) { UI.showToast("Wait for the current upload to finish.", "info"); return; }
    if (list.length > 1) UI.showToast(`Uploading ${list.length} files, one at a time.`, "info");
    for (let i = 0; i < list.length; i++) {
      const ok = await _handleFileUpload(list[i]);
      if (ok === false && i < list.length - 1) {
        const left = list.length - i - 1;
        UI.showToast(`Stopped after a problem. ${left} more ${left === 1 ? "file was" : "files were"} not uploaded.`, "info");
        return;
      }
    }
  }

  /** Client-side checks are a convenience only: the server validates type and size again. Returns true on success. */
  async function _handleFileUpload(file) {
    const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    const allowed = State.limits.extensions || ALLOWED;
    if (!allowed.includes(ext)) {
      UI.showToast(ext ? `Unsupported file type: .${ext}` : `"${file.name}" has no file extension.`, "error");
      return false;
    }
    if (file.size === 0) {
      UI.showToast(`"${file.name}" is empty.`, "error");
      return false;
    }
    if (State.limits.maxMb && file.size > State.limits.maxMb * 1024 * 1024) {
      UI.showToast(`"${file.name}" is ${_fmtBytes(file.size)}, over the ${State.limits.maxMb} MB limit.`, "error");
      return false;
    }
    // The backend does not de-duplicate: a second upload of the same name becomes a second document
    if (State.documents.some((d) => d.filename === file.name)) {
      const again = await UI.confirm({
        title: "Upload this file again?",
        message: `"${file.name}" is already in your documents. Uploading it again adds a second copy that answers will also draw from.`,
        confirmLabel: "Upload anyway",
        cancelLabel: "Skip",
      });
      if (!again) return false;
    }
    // The File is kept in the job closure so a failed upload can be retried
    return _runUpload({
      label: file.name,
      send: () => API.uploadFile(file, State.userId),
      success: (r) => `"${r.filename}" indexed (${r.chunk_count} chunks)${r.indexing === "lexical" ? " (keyword index only)" : ""}`,
      failure: "Upload failed",
    });
  }

  async function _onUrlSubmit() {
    const inp = document.getElementById("urlInput");
    const url = inp?.value?.trim();
    if (!url || !/^https?:\/\//i.test(url)) {
      UI.showToast("Please enter a valid URL starting with http(s)://", "error");
      return;
    }
    UI.hideUrlModal();
    return _runUpload({
      label: url,
      send: () => API.uploadUrl(url, State.userId),
      success: (r) => `URL indexed (${r.chunk_count} chunks)`,
      failure: "URL indexing failed",
      // Without a retry control in the view, reopen the dialog with the URL so nothing is retyped
      reopen: () => { UI.showUrlModal(); if (inp) inp.value = url; },
    });
  }

  async function _runUpload(job) {
    if (State.uploading) {
      UI.showToast("Wait for the current upload to finish.", "info");
      return false;
    }
    State.uploading = true;
    UI.showUploadToast(job.label);
    try {
      const result = await job.send();
      UI.hideUploadToast();
      UI.showToast(job.success(result), "success");
      await _loadDocuments();
      return true;
    } catch (err) {
      UI.hideUploadToast();
      const message = `${job.failure}: ${err.message}`;   // the server's own text (e.g. the 413 message)
      if (typeof UI.showUploadError === "function") {
        UI.showUploadError(job.label, { message, onRetry: () => _runUpload(job) });
      } else {
        UI.showToast(message, "error");
        if (job.reopen) job.reopen();
      }
      return false;
    } finally {
      State.uploading = false;
    }
  }

  async function _deleteDocument(doc) {
    const ok = await UI.confirm({
      title: "Remove this document?",
      message: `"${doc.filename}" will be removed from your documents and can no longer be searched.`,
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      destructive: true,
    });
    if (!ok) return;
    try {
      await API.deleteDocument(doc.id, State.userId);
      UI.showToast("Document removed", "success");
      await _loadDocuments();
    } catch (err) {
      UI.showToast(`Could not remove document: ${err.message}`, "error");
    }
  }

  async function _deleteSelectedDocuments(ids) {
    if (!ids || !ids.length) return;
    const n = ids.length;
    const ok = await UI.confirm({
      title: `Remove ${n} ${n === 1 ? "document" : "documents"}?`,
      message: `${n === 1 ? "This document" : `These ${n} documents`} will be removed from your library and can no longer be searched.`,
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      destructive: true,
    });
    if (!ok) return;
    try {
      // Delete in parallel — existing single-delete endpoint handles each
      await Promise.all(ids.map((id) => API.deleteDocument(id, State.userId)));
      UI.showToast(`${n} ${n === 1 ? "document" : "documents"} removed`, "success");
    } catch (err) {
      UI.showToast(`Could not remove some documents: ${err.message}`, "error");
    } finally {
      await _loadDocuments();
    }
  }

  async function _loadDocuments() {
    if (!State.documentsLoaded) UI.setDocsStatus("loading");
    try {
      const docs = await API.listDocuments(State.userId);
      State.documents = docs || [];
      State.documentsLoaded = true;
      UI.renderDocList(State.documents, _deleteDocument, _deleteSelectedDocuments);
    } catch (err) {
      console.error("[app] Could not load documents:", err);
      // Keep a list that was already loaded; show the error state only when there is nothing to show
      if (!State.documentsLoaded) UI.setDocsStatus("error", { message: err.message, onRetry: _loadDocuments });
      UI.notify(`Could not load documents: ${err.message}`, "error");
    }
  }

  // ── Integrations (Settings) ────────────────────────────────
  async function _loadServices() {
    UI.renderIntegrations(null, { loading: true });
    try {
      UI.renderIntegrations(await API.getHealth());
    } catch (err) {
      UI.renderIntegrations(null, { error: err.message });
    }
    _loadGithubRepos();
  }

  async function _loadGithubRepos() {
    try {
      UI.renderGithubRepos(await API.listGithubRepos(State.userId), { onDisconnect: _disconnectGithub });
    } catch (_) {
      UI.renderGithubRepos(null, { error: true });
    }
  }

  async function _disconnectGithub(repoId) {
    if (!repoId) return;
    try {
      await API.deleteGithubRepo(repoId, State.userId);
      UI.showToast("Repository disconnected.", "success");
      _loadGithubRepos();
      _loadDocuments();
    } catch (err) {
      UI.showToast(err.message || "Could not disconnect the repository.", "error");
    }
  }

  async function _onGithubSubmit() {
    const urlEl = document.getElementById("githubUrl");
    const tokenEl = document.getElementById("githubToken");
    const btn = document.getElementById("githubModalSubmit");
    const url = urlEl ? urlEl.value.trim() : "";
    const token = tokenEl ? tokenEl.value.trim() : "";
    if (!url) { UI.showToast("Enter a GitHub repository URL.", "error"); return; }
    if (btn) btn.disabled = true;
    try {
      UI.showToast("Indexing repository…", "info");
      await API.connectGithub(url, token, State.userId);
      if (tokenEl) tokenEl.value = "";
      _resetGithubOauth();
      UI.hideGithubModal();
      UI.showToast("Repository connected.", "success");
      _loadDocuments();
      _loadGithubRepos();
    } catch (err) {
      UI.showToast(err.message || "Could not connect the repository.", "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  let _githubDeviceCode = "";

  function _resetGithubOauth() {
    _githubDeviceCode = "";
    const btn = document.getElementById("githubOauth");
    if (btn) btn.textContent = "Sign in with GitHub";
  }

  async function _onGithubOauth() {
    const help = document.getElementById("githubDeviceHelp");
    const codeEl = document.getElementById("githubUserCode");
    const link = document.getElementById("githubDeviceLink");
    const btn = document.getElementById("githubOauth");
    if (_githubDeviceCode) {
      try {
        const data = await API.pollGithubDevice(_githubDeviceCode, State.userId);
        if (data.status === "connected") {
          _resetGithubOauth();
          UI.showToast("GitHub connected. Paste a repository URL and click Connect.", "success");
          return;
        }
        UI.showToast("GitHub is still waiting. Approve the code, then click I've authorized.", "info");
      } catch (err) {
        _resetGithubOauth();
        UI.showToast(err.message || "GitHub sign-in did not complete.", "error");
      }
      return;
    }
    try {
      const start = await API.startGithubDevice();
      _githubDeviceCode = start.device_code || "";
      if (codeEl) codeEl.textContent = start.user_code || "";
      const href = window.Utils && typeof Utils.safeUrl === "function"
        ? Utils.safeUrl(start.verification_uri) : null;
      if (link && href) {
        link.href = href;
        link.textContent = "Open the device login page";
      }
      if (help) help.hidden = false;
      if (btn) btn.textContent = "I've authorized";
      UI.showToast("Approve the sign-in on GitHub, then click I've authorized.", "info");
    } catch (err) {
      _resetGithubOauth();
      UI.showToast(err.message || "GitHub sign-in did not complete.", "error");
    }
  }

  // ── Helpers ────────────────────────────────────────────────
  // The textarea stays enabled while a request runs (the next message can be typed);
  // only Send is gated (State.isLoading). Stop replaces Send and is reachable by Tab.
  function _setLoading(val) {
    State.isLoading = val;
    const send = document.getElementById("sendBtn");
    const stop = document.getElementById("stopBtn");
    const textarea = document.getElementById("chatInput");
    const active = document.activeElement;
    if (send) send.style.display = val ? "none" : "";
    if (stop) stop.style.display = val ? "" : "none";
    if (textarea) {
      // A hidden control cannot keep focus: hand it back to the composer
      if (!active || active === document.body || active === send || active === stop) textarea.focus();
    }
  }

  function _autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }

  // ── Boot ───────────────────────────────────────────────────
  const boot = () => { init().catch(_startupFailed); };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

})();
