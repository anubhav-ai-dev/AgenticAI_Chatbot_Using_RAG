/* ============================================================
   api.js — REST calls to the Python backend
   ============================================================ */

window.API = (function () {
  function getBaseUrl() {
    return (window.CONFIG && window.CONFIG.BACKEND_URL) || "";
  }

  async function request(path, options = {}) {
    const url = `${getBaseUrl()}${path}`;
    const defaultHeaders = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };

    if (options.body instanceof FormData) {
      delete defaultHeaders["Content-Type"]; // let browser set boundary
    }

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {}),
      },
    };

    try {
      const resp = await fetch(url, config);
      if (!resp.ok) {
        let errorMsg = `HTTP ${resp.status}`;
        try {
          const errData = await resp.json();
          errorMsg = errData.detail || errData.message || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }
      if (resp.status === 204) return null;
      return await resp.json();
    } catch (err) {
      console.error(`[API] Error on ${path}:`, err);
      throw err;
    }
  }

  // ── Sessions ──────────────────────────────────────────────

  function listSessions(userId = "default") {
    return request(`/sessions?user_id=${encodeURIComponent(userId)}`);
  }

  function createSession(data) {
    return request("/sessions", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  function getSession(sessionId) {
    return request(`/sessions/${sessionId}`);
  }

  function updateSession(sessionId, data) {
    return request(`/sessions/${sessionId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  function deleteSession(sessionId) {
    return request(`/sessions/${sessionId}`, {
      method: "DELETE",
    });
  }

  function saveMessage(sessionId, message) {
    return request(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify(message),
    });
  }

  // ── Documents ─────────────────────────────────────────────

  function listDocuments(userId = "default") {
    return request(`/documents?user_id=${encodeURIComponent(userId)}`);
  }

  function uploadFile(file, userId = "default") {
    const form = new FormData();
    form.append("file", file);
    return request(`/upload?user_id=${encodeURIComponent(userId)}`, {
      method: "POST",
      body: form,
    });
  }

  function uploadUrl(url, userId = "default") {
    return request(`/upload?url=${encodeURIComponent(url)}&user_id=${encodeURIComponent(userId)}`, {
      method: "POST",
    });
  }

  // ── RAG Retrieval ─────────────────────────────────────────

  function retrieve(query, userId = "default", topK = 5, threshold = 0.35) {
    return request("/retrieve", {
      method: "POST",
      body: JSON.stringify({
        query,
        user_id: userId,
        top_k: topK,
        threshold,
      }),
    });
  }

  // ── Search ────────────────────────────────────────────────

  function search(query, maxResults = 5) {
    return request("/search", {
      method: "POST",
      body: JSON.stringify({ query, max_results: maxResults }),
    });
  }

  // ── Health ────────────────────────────────────────────────

  function health() {
    return request("/health");
  }

  return {
    listSessions,
    createSession,
    getSession,
    updateSession,
    deleteSession,
    saveMessage,
    listDocuments,
    uploadFile,
    uploadUrl,
    retrieve,
    search,
    health,
  };
})();
