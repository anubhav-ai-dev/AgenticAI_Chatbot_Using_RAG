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
      else if (data.message) msg = data.message;
    } catch (_) {}
    return msg;
  }

  async function request(path, options = {}) {
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";   // FormData: browser sets the boundary
    }

    try {
      const resp = await fetch(`${getBaseUrl()}${path}`, { ...options, headers });
      if (!resp.ok) throw new Error(await _errorMessage(resp));
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

  function createSession(data) {
    return request("/sessions", { method: "POST", body: JSON.stringify(data) });
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

  function getModels() {
    return request("/models");
  }

  // ── Chat (server-sent events) ─────────────────────────────

  /**
   * POST /chat/stream and dispatch each SSE event to handlers[eventName](data).
   * Events: meta, status, citations, model, token, error, done.
   * Resolves when the stream ends; rejects on HTTP errors or abort.
   */
  async function chatStream(payload, handlers, signal) {
    const resp = await fetch(`${getBaseUrl()}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
    if (!resp.ok) throw new Error(await _errorMessage(resp));
    if (!resp.body) throw new Error("Streaming is not supported by this browser.");

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
      const handler = handlers[event];
      if (handler) handler(data);
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
    chatStream,
  };
})();
