/* ============================================================
   api.js — REST + SSE calls to the Python backend
   ============================================================ */

window.API = (function () {
  function getBaseUrl() {
    return (window.CONFIG && window.CONFIG.BACKEND_URL) || "";
  }

  async function _errorMessage(resp) {
    let msg = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) msg = data.detail[0].msg;
      else if (data.message) msg = data.message;
    } catch (_) {}
    return msg;
  }

  /** Error for a non-2xx response: carries `status` and `retryAfter` (seconds, from Retry-After) when present. */
  async function _httpError(resp) {
    const err = new Error(await _errorMessage(resp));
    err.status = resp.status;
    const ra = parseInt(resp.headers && resp.headers.get && resp.headers.get("Retry-After"), 10);
    err.retryAfter = Number.isFinite(ra) && ra > 0 ? ra : null;
    return err;
  }

  async function request(path, options = {}) {
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";   // FormData: browser sets the boundary
    }

    try {
      const resp = await fetch(`${getBaseUrl()}${path}`, { ...options, headers });
      if (!resp.ok) throw await _httpError(resp);
      if (resp.status === 204) return null;
      return await resp.json();
    } catch (err) {
      if (err.name !== "AbortError") console.error(`[API] Error on ${path}:`, err);
      throw err;
    }
  }

  // ── Sessions ──────────────────────────────────────────────

  function listSessions(userId = "default") {
    return request(`/sessions?user_id=${encodeURIComponent(userId)}`);
  }

  function createSession(data, signal) {
    return request("/sessions", { method: "POST", body: JSON.stringify(data), ...(signal ? { signal } : {}) });
  }

  function getSession(sessionId) {
    return request(`/sessions/${sessionId}`);
  }

  function updateSession(sessionId, data) {
    return request(`/sessions/${sessionId}`, { method: "PATCH", body: JSON.stringify(data) });
  }

  function deleteSession(sessionId) {
    return request(`/sessions/${sessionId}`, { method: "DELETE" });
  }

  function setFeedback(messageId, value) {
    return request(`/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ value }),
    });
  }

  // ── Documents ─────────────────────────────────────────────

  function listDocuments(userId = "default") {
    return request(`/documents?user_id=${encodeURIComponent(userId)}`);
  }

  function uploadFile(file, userId = "default") {
    const form = new FormData();
    form.append("file", file);
    return request(`/upload?user_id=${encodeURIComponent(userId)}`, { method: "POST", body: form });
  }

  function uploadUrl(url, userId = "default") {
    return request(`/upload?url=${encodeURIComponent(url)}&user_id=${encodeURIComponent(userId)}`, {
      method: "POST",
    });
  }

  function deleteDocument(docId, userId = "default") {
    return request(`/documents/${docId}?user_id=${encodeURIComponent(userId)}`, { method: "DELETE" });
  }

  // ── Models ────────────────────────────────────────────────

  /** Service status (database, providers, embeddings…). Read-only. */
  function getHealth() {
    return request("/health");
  }

  /** Upload limits advertised by the server: { upload: { max_mb, extensions } }. 404 on an older backend. */
  function getConfig() {
    return request("/config");
  }

  function getModels() {
    return request("/models");
  }

  // ── Chat (server-sent events) ─────────────────────────────

  /**
   * POST /chat/stream and dispatch each SSE event to handlers[eventName](data).
   * Events: meta, status, citations, model, token, error, done.
   * Resolves when the stream ends; rejects on HTTP errors (err.status, err.retryAfter) or abort.
   * opts.onOpen() is called once the server has accepted the request (HTTP 2xx, before any event).
   */
  async function chatStream(payload, handlers, signal, opts = {}) {
    const resp = await fetch(`${getBaseUrl()}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!resp.ok) throw await _httpError(resp);
    if (!resp.body) throw new Error("Streaming is not supported by this browser.");
    if (typeof opts.onOpen === "function") opts.onOpen();

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const dispatch = (block) => {
      let event = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) return;
      let data;
      try {
        data = JSON.parse(dataLines.join("\n"));
      } catch (_) {
        return;
      }
      // Own-property check: an event named __proto__/constructor/toString must not reach Object.prototype members.
      // Framing is LF-only ("\n\n"); CRLF is unsupported because the backend emits LF only (backend/utils/sse.py).
      if (Object.prototype.hasOwnProperty.call(handlers, event) && typeof handlers[event] === "function") handlers[event](data);
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        dispatch(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 2);
      }
    }
    if (buffer.trim()) dispatch(buffer);
  }

  return {
    listSessions,
    createSession,
    getSession,
    updateSession,
    deleteSession,
    setFeedback,
    listDocuments,
    uploadFile,
    uploadUrl,
    deleteDocument,
    getModels,
    getHealth,
    getConfig,
    chatStream,
  };
})();
