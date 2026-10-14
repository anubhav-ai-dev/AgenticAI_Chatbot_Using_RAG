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
    abort:      null,    // AbortController of the in-flight stream
  };

  // ── Init ───────────────────────────────────────────────────
  async function init() {
    UI.initTheme();
    UI.setActiveMode(State.mode);
    UI.setUserIdDisplay(State.userId);
    _bindEvents();
    await Promise.all([_loadModels(), _loadSessions(), _loadDocuments()]);
  }

  // ── Models (driven by the server config) ───────────────────
  async function _loadModels() {
    try {
      const data = await API.getModels();
      State.models = data.models || [];
      State.defaultModel = data.default || "";
    } catch (_) {
      State.models = [];
    }
    const usable = State.models.filter((m) => m.available);
    const ids = new Set(usable.map((m) => m.id));
    if (!ids.has(State.model)) {
      State.model = ids.has(State.defaultModel) ? State.defaultModel : (usable[0] ? usable[0].id : "");
      if (State.model) localStorage.setItem("model", State.model);
    }
    UI.renderModelSelect(State.models, State.model);
  }

  // ── Event wiring ───────────────────────────────────────────
  function _bindEvents() {

    document.getElementById("sendBtn").addEventListener("click", () => _onSend());
    document.getElementById("stopBtn").addEventListener("click", _onStop);

    // Textarea — Enter to send, Shift+Enter for newline, auto-resize
    const textarea = document.getElementById("chatInput");
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        _onSend();
      }
    });
    textarea.addEventListener("input", () => _autoResize(textarea));

    // Mode tabs
    document.getElementById("modeTabs").addEventListener("click", (e) => {
      const tab = e.target.closest(".mode-tab");
      if (!tab) return;
      const mode = tab.dataset.mode;
      State.mode = mode;
      localStorage.setItem("mode", mode);
      UI.setActiveMode(mode);
      if (State.sessionId) API.updateSession(State.sessionId, { mode }).catch(() => {});
    });

    // Model selector
    document.getElementById("modelSelect").addEventListener("change", (e) => {
      State.model = e.target.value;
      localStorage.setItem("model", State.model);
      if (State.sessionId) API.updateSession(State.sessionId, { model: State.model }).catch(() => {});
    });

    document.getElementById("newChatBtn").addEventListener("click", _newChat);
    document.getElementById("themeToggle").addEventListener("click", UI.toggleTheme);
    document.getElementById("sidebarToggle").addEventListener("click", UI.toggleSidebar);

    // Attach file
    document.getElementById("attachBtn").addEventListener("click", () => {
      document.getElementById("fileInput").click();
    });
    document.getElementById("fileInput").addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) _handleFileUpload(file);
      e.target.value = "";  // reset so same file can be re-selected
    });

    // Add URL
    document.getElementById("addUrlBtn").addEventListener("click", UI.showUrlModal);
    document.getElementById("urlModalClose").addEventListener("click", UI.hideUrlModal);
    document.getElementById("urlModalCancel").addEventListener("click", UI.hideUrlModal);
    document.getElementById("urlModalSubmit").addEventListener("click", _onUrlSubmit);
    document.getElementById("urlInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") _onUrlSubmit();
    });
    document.getElementById("urlModal").addEventListener("click", (e) => {
      if (e.target === document.getElementById("urlModal")) UI.hideUrlModal();
    });

    // Web search toggle
    const searchToggle = document.getElementById("searchToggle");
    searchToggle.addEventListener("click", () => {
      State.webSearch = !State.webSearch;
      searchToggle.style.color = State.webSearch ? "var(--accent-green)" : "var(--text-muted)";
      searchToggle.setAttribute("aria-pressed", String(State.webSearch));
      UI.showToast(State.webSearch ? "Web search enabled" : "Web search disabled", "info");
    });

    // Citation modal close
    document.getElementById("citationModalClose").addEventListener("click", UI.hideCitationModal);
    document.getElementById("citationModal").addEventListener("click", (e) => {
      if (e.target === document.getElementById("citationModal")) UI.hideCitationModal();
    });

    // Session search filter
    document.getElementById("sessionSearch").addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = State.sessions.filter((s) => s.title.toLowerCase().includes(q));
      UI.renderSessionList(filtered, State.sessionId, _selectSession, _deleteSession);
    });

    UI.initDropzone(_handleFileUpload);

    // Starter prompts
    document.getElementById("starterPrompts").addEventListener("click", (e) => {
      const btn = e.target.closest(".welcome__starter");
      if (!btn) return;
      const prompt = btn.dataset.prompt;
      if (!prompt) return;
      const textarea = document.getElementById("chatInput");
      textarea.value = prompt;
      _autoResize(textarea);
      _onSend();
    });
  }

  // ── Chat pipeline (one request: route → retrieve/search → generate → persist) ──
  const STAGE_LABELS = {
    rewriting:  { icon: "✏️", pending: "Understanding your follow-up…" },
    retrieving: { icon: "🔍", pending: "Searching documents…" },
    searching:  { icon: "🌐", pending: "Searching the web…" },
    generating: { icon: "🤖", pending: "Generating response…" },
  };

  async function _onSend({ regenerate = false } = {}) {
    if (State.isLoading) return;
    const textarea = document.getElementById("chatInput");
    const text = regenerate ? "" : textarea.value.trim();
    if (!regenerate && !text) return;

    if (!regenerate) {
      textarea.value = "";
      _autoResize(textarea);
    }
    _setLoading(true);

    try {
      if (!State.sessionId) await _ensureSession();
    } catch (err) {
      _setLoading(false);
      UI.showToast(`Could not start a chat: ${err.message}`, "error");
      return;
    }

    if (!regenerate) Chat.appendUserMessage(text, State.userId);
    // Only the newest answer can be regenerated
    document.querySelectorAll("#chatArea .message__regen").forEach((b) => b.remove());
    const renderer = Chat.startAssistantMessage();

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
        icon: status === "error" ? "⚠️" : "✅",
        label,
        elapsed: (ms / 1000).toFixed(1) + "s",
        status,
      });
    };

    State.abort = new AbortController();
    try {
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
        {
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
          error: (d) => { streamError = d; },
          done: (d) => { doneInfo = d; },
        },
        State.abort.signal,
      );

      if (streamError && !renderer.getRawText()) {
        finishStep("generating", "Generation failed", "error");
        renderer.showError(streamError.message || "Something went wrong. Please try again.");
      } else {
        renderer.finalize(citations);
        renderer.setBadges({
          model: modelInfo.model,
          cached: !!(doneInfo && doneInfo.cached),
          fellBackFrom: modelInfo.fellBackFrom,
        });
        const messageId = (doneInfo && doneInfo.message_id) || (streamError && streamError.message_id);
        renderer.enableActions({
          messageId,
          onRegenerate: _regenerateLast,
          onFeedback: _sendFeedback,
        });
        if (streamError) UI.showToast(streamError.message || "The response was interrupted.", "error");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // Stop pressed: the server persists the partial answer
        renderer.finalize(citations);
        renderer.setBadges({ model: modelInfo.model, fellBackFrom: modelInfo.fellBackFrom });
        UI.showToast("Generation stopped", "info");
      } else {
        console.error("[app] Pipeline error:", err);
        renderer.showError(err.message || "Something went wrong. Please try again.");
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

  /** Regenerate the last answer: drop it from the DOM, then re-run on the server. */
  function _regenerateLast() {
    const assistants = document.querySelectorAll("#chatArea .message--assistant");
    const last = assistants[assistants.length - 1];
    if (last) last.remove();
    return _onSend({ regenerate: true });
  }

  // ── Session management ─────────────────────────────────────
  async function _ensureSession() {
    const session = await API.createSession({
      user_id: State.userId,
      title: "New Chat",
      model: State.model || null,
      mode: State.mode,
    });
    State.sessionId = session.id;
  }

  async function _newChat() {
    if (State.isLoading) _onStop();
    State.sessionId = null;
    Chat.clearChat();
    try {
      await _ensureSession();
      await _loadSessions();
    } catch (_) {}
    UI.setActiveMode(State.mode);
  }

  async function _selectSession(sessionId) {
    if (State.isLoading) return;
    State.sessionId = sessionId;
    try {
      const session = await API.getSession(sessionId);
      Chat.renderHistory(session.messages || [], { onFeedback: _sendFeedback });
      State.mode = session.mode || "general";
      UI.setActiveMode(State.mode);
      const available = State.models.find((m) => m.id === session.model && m.available);
      if (available) {
        State.model = session.model;
        const sel = document.getElementById("modelSelect");
        if (sel) sel.value = State.model;
      }
      UI.renderSessionList(State.sessions, State.sessionId, _selectSession, _deleteSession);
    } catch (err) {
      UI.showToast("Failed to load session", "error");
    }
  }

  async function _deleteSession(sessionId) {
    if (!confirm("Delete this chat?")) return;
    try {
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

  async function _loadSessions() {
    try {
      const sessions = await API.listSessions(State.userId);
      State.sessions = sessions || [];
      UI.renderSessionList(State.sessions, State.sessionId, _selectSession, _deleteSession);
    } catch (_) {}
  }

  // ── Document handling ──────────────────────────────────────
  async function _handleFileUpload(file) {
    const ALLOWED = ["pdf", "docx", "csv", "xlsx", "txt", "md"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      UI.showToast(`Unsupported file type: .${ext}`, "error");
      return;
    }

    UI.showUploadToast(file.name);
    try {
      const result = await API.uploadFile(file, State.userId);
      UI.hideUploadToast();
      const note = result.indexing === "lexical" ? " (keyword index only)" : "";
      UI.showToast(`✅ "${result.filename}" indexed (${result.chunk_count} chunks)${note}`, "success");
      await _loadDocuments();
    } catch (err) {
      UI.hideUploadToast();
      UI.showToast(`Upload failed: ${err.message}`, "error");
    }
  }

  async function _onUrlSubmit() {
    const inp = document.getElementById("urlInput");
    const url = inp?.value?.trim();
    if (!url || !/^https?:\/\//i.test(url)) {
      UI.showToast("Please enter a valid URL starting with http(s)://", "error");
      return;
    }
    UI.hideUrlModal();
    UI.showUploadToast(url);
    try {
      const result = await API.uploadUrl(url, State.userId);
      UI.hideUploadToast();
      UI.showToast(`✅ URL indexed (${result.chunk_count} chunks)`, "success");
      await _loadDocuments();
    } catch (err) {
      UI.hideUploadToast();
      UI.showToast(`URL indexing failed: ${err.message}`, "error");
    }
  }

  async function _deleteDocument(doc) {
    if (!confirm(`Remove "${doc.filename}" from your documents?`)) return;
    try {
      await API.deleteDocument(doc.id, State.userId);
      UI.showToast("Document removed", "success");
      await _loadDocuments();
    } catch (err) {
      UI.showToast(`Could not remove document: ${err.message}`, "error");
    }
  }

  async function _loadDocuments() {
    try {
      const docs = await API.listDocuments(State.userId);
      State.documents = docs || [];
      UI.renderDocList(State.documents, _deleteDocument);
    } catch (_) {}
  }

  // ── Helpers ────────────────────────────────────────────────
  function _setLoading(val) {
    State.isLoading = val;
    const send = document.getElementById("sendBtn");
    const stop = document.getElementById("stopBtn");
    const textarea = document.getElementById("chatInput");
    if (send) send.style.display = val ? "none" : "";
    if (stop) stop.style.display = val ? "" : "none";
    if (textarea) {
      textarea.disabled = val;
      if (!val) textarea.focus();
    }
  }

  function _autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }

  // ── Boot ───────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();
