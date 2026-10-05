/* ============================================================
   app.js — Main state orchestrator & event wiring
   Initializes all modules, manages global state, handles
   the full send → retrieve/search → stream pipeline.
   ============================================================ */

(function () {
  "use strict";

  // ── Global state ───────────────────────────────────────────
  const State = {
    userId:     localStorage.getItem("userId")    || "default",
    sessionId:  null,
    mode:       localStorage.getItem("mode")      || "general",
    model:      localStorage.getItem("model")     || "claude-sonnet-4-5",
    sessions:   [],
    documents:  [],
    webSearch:  true,    // toggle via search button
    isLoading:  false,
  };

  // ── Init ───────────────────────────────────────────────────
  async function init() {
    UI.initTheme();

    // Restore model and mode selections
    const modelSel = document.getElementById("modelSelect");
    if (modelSel) modelSel.value = State.model;

    UI.setActiveMode(State.mode);
    UI.setUserIdDisplay(State.userId);

    // Wire event listeners
    _bindEvents();

    // Load sessions and documents
    await Promise.all([_loadSessions(), _loadDocuments()]);
  }

  // ── Event wiring ───────────────────────────────────────────
  function _bindEvents() {

    // Send button
    document.getElementById("sendBtn").addEventListener("click", _onSend);

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

    // New chat button
    document.getElementById("newChatBtn").addEventListener("click", _newChat);

    // Theme toggle
    document.getElementById("themeToggle").addEventListener("click", UI.toggleTheme);

    // Sidebar toggle (mobile)
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

    // Drag and drop
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

  // ── Send pipeline ──────────────────────────────────────────
  async function _onSend() {
    const textarea = document.getElementById("chatInput");
    const text = textarea.value.trim();
    if (!text || State.isLoading) return;

    // Clear input
    textarea.value = "";
    _autoResize(textarea);
    _setLoading(true);

    // Ensure a session exists
    if (!State.sessionId) await _ensureSession();

    // Append user message to UI
    Chat.appendUserMessage(text, State.userId);

    // Save user message to DB (non-blocking)
    API.saveMessage(State.sessionId, { role: "user", content: text }).catch(() => {});

    // Start assistant message render
    const renderer = Chat.startAssistantMessage();

    try {
      let ragChunks   = [];
      let searchResults = [];
      const citations = [];

      // ── Step 1: RAG retrieval (if mode = document or has docs) ──
      if (State.mode === "document" && State.documents.length > 0) {
        const ragStep = renderer.addThought("🔍", "Searching documents…", "pending");
        const t0 = Date.now();
        try {
          const result = await API.retrieve(text, State.userId, 5, 0.30);
          const elapsed = ((Date.now() - t0) / 1000).toFixed(1) + "s";
          if (result.used_rag && result.chunks.length > 0) {
            ragChunks = result.chunks;
            result.chunks.forEach((c, i) => {
              citations.push({ id: i + 1, source: c.source, page: c.page, snippet: c.snippet });
              renderer.addThought("📄", `Found excerpt from ${c.source}${c.page ? ` (p.${c.page})` : ""}`, "done");
            });
            renderer.resolveThought(ragStep, {
              icon: "✅",
              label: `Retrieved ${ragChunks.length} relevant excerpt${ragChunks.length > 1 ? "s" : ""}`,
              elapsed,
              status: "done",
            });
          } else {
            renderer.resolveThought(ragStep, {
              icon: "ℹ️",
              label: "No matching document excerpts found",
              elapsed,
              status: "done",
            });
          }
        } catch (err) {
          renderer.resolveThought(ragStep, { icon: "❌", label: "Document search failed", status: "error" });
        }
      }

      // ── Step 2: Web search (if mode = research or general + search enabled) ──
      if (
        (State.mode === "research" || (State.mode === "general" && State.webSearch)) &&
        State.documents.length === 0 || State.mode === "research"
      ) {
        const searchStep = renderer.addThought("🌐", "Searching the web…", "pending");
        const t1 = Date.now();
        try {
          const result = await API.search(text, 5);
          const elapsed = ((Date.now() - t1) / 1000).toFixed(1) + "s";
          if (result.results && result.results.length > 0) {
            searchResults = result.results;
            const offset = citations.length;
            result.results.slice(0, 3).forEach((r, i) => {
              citations.push({ id: offset + i + 1, source: r.title || r.url, page: null, snippet: r.snippet });
            });
            renderer.resolveThought(searchStep, {
              icon: "✅",
              label: `Found ${result.results.length} web sources`,
              elapsed,
              status: "done",
            });
          } else {
            renderer.resolveThought(searchStep, {
              icon: "ℹ️",
              label: "No web results found",
              elapsed,
              status: "done",
            });
          }
        } catch (err) {
          renderer.resolveThought(searchStep, { icon: "❌", label: "Web search failed", status: "error" });
        }
      }

      // ── Step 3: Build conversation history for context ──────
      let sessionMessages = [];
      try {
        const sess = await API.getSession(State.sessionId);
        const hist = (sess.messages || []).slice(-10);  // last 10 messages
        sessionMessages = hist
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => ({ role: m.role, content: m.content }));
      } catch (_) {}

      // Add current user message at the end
      sessionMessages.push({ role: "user", content: text });

      // ── Step 4: Build system prompt ─────────────────────────
      const systemPrompt = PuterClient.buildSystemPrompt(State.mode, ragChunks, searchResults);

      // ── Step 5: Stream via Puter.js ─────────────────────────
      const thinkStep = renderer.addThought("🤖", "Generating response…", "pending");
      const t2 = Date.now();

      await new Promise((resolve, reject) => {
        PuterClient.streamChat({
          model: State.model,
          messages: sessionMessages,
          systemPrompt,
          onToken: (token) => renderer.appendToken(token),
          onDone: () => {
            const elapsed = ((Date.now() - t2) / 1000).toFixed(1) + "s";
            renderer.resolveThought(thinkStep, {
              icon: "✅",
              label: `Response generated in ${elapsed}`,
              elapsed,
              status: "done",
            });
            resolve();
          },
          onError: reject,
        });
      });

      // ── Step 6: Finalize render + save to DB ────────────────
      renderer.finalize(citations);

      API.saveMessage(State.sessionId, {
        role: "assistant",
        content: renderer.getRawText(),
        citations: citations.length > 0 ? citations : null,
      }).catch(() => {});

      // Refresh session list title
      await _loadSessions();

    } catch (err) {
      console.error("[app] Pipeline error:", err);
      renderer.showError(err.message || "Something went wrong. Please try again.");
    } finally {
      _setLoading(false);
    }
  }

  // ── Session management ─────────────────────────────────────
  async function _ensureSession() {
    const session = await API.createSession({
      user_id: State.userId,
      title: "New Chat",
      model: State.model,
      mode: State.mode,
    });
    State.sessionId = session.id;
  }

  async function _newChat() {
    State.sessionId = null;
    Chat.clearChat();
    await _ensureSession();
    await _loadSessions();
    UI.setActiveMode(State.mode);
  }

  async function _selectSession(sessionId) {
    State.sessionId = sessionId;
    try {
      const session = await API.getSession(sessionId);
      Chat.renderHistory(session.messages || []);
      // Sync mode and model from session
      State.mode = session.mode || "general";
      State.model = session.model || "claude-sonnet-4-5";
      UI.setActiveMode(State.mode);
      const sel = document.getElementById("modelSelect");
      if (sel) sel.value = State.model;
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
    const ALLOWED = ["pdf","docx","csv","xlsx","txt","md"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      UI.showToast(`Unsupported file type: .${ext}`, "error");
      return;
    }

    UI.showUploadToast(file.name);
    try {
      const result = await API.uploadFile(file, State.userId);
      UI.hideUploadToast();
      UI.showToast(`✅ "${result.filename}" indexed (${result.chunk_count} chunks)`, "success");
      await _loadDocuments();
    } catch (err) {
      UI.hideUploadToast();
      UI.showToast(`Upload failed: ${err.message}`, "error");
    }
  }

  async function _onUrlSubmit() {
    const inp = document.getElementById("urlInput");
    const url = inp?.value?.trim();
    if (!url || !url.startsWith("http")) {
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

  async function _loadDocuments() {
    try {
      const docs = await API.listDocuments(State.userId);
      State.documents = docs || [];
      UI.renderDocList(State.documents);
    } catch (_) {}
  }

  // ── Helpers ────────────────────────────────────────────────
  function _setLoading(val) {
    State.isLoading = val;
    const btn = document.getElementById("sendBtn");
    const textarea = document.getElementById("chatInput");
    if (btn) btn.disabled = val;
    if (textarea) textarea.disabled = val;
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
