/* ============================================================
   puter_client.js — Wraps puter.ai.chat() with auth, streaming,
   context injection, and graceful error handling.
   ============================================================ */

window.PuterClient = (function () {

  let _initialized = false;

  function init() {
    if (_initialized) return;
    // Puter.js SDK is loaded via <script src="https://js.puter.com/v2/">
    // Auth token is injected via puter.auth.signIn if needed
    _initialized = true;
  }

  /**
   * Stream a chat response from a Puter.js model.
   *
   * @param {object} opts
   * @param {string}   opts.model      — puter model id (e.g. "claude-sonnet-4-5")
   * @param {Array}    opts.messages   — [{role:"user"|"assistant"|"system", content:"..."}]
   * @param {string}   opts.systemPrompt — system instruction string
   * @param {function} opts.onToken    — called for each streamed token (string)
   * @param {function} opts.onDone     — called when stream finishes
   * @param {function} opts.onError    — called on error (Error)
   */
  async function streamChat({ model, messages, systemPrompt, onToken, onDone, onError }) {
    init();

    if (!window.puter || !puter.ai) {
      onError(new Error("Puter.js SDK not loaded. Check your internet connection."));
      return;
    }

    try {
      // Build message list with system prompt at front
      const fullMessages = [];
      if (systemPrompt) {
        fullMessages.push({ role: "system", content: systemPrompt });
      }
      fullMessages.push(...messages);

      const response = await puter.ai.chat(
        fullMessages[fullMessages.length - 1].content,
        {
          model,
          stream: true,
          messages: fullMessages,
        }
      );

      for await (const part of response.textStream) {
        if (part) onToken(part);
      }

      onDone();
    } catch (err) {
      console.error("[PuterClient] Stream error:", err);
      onError(err);
    }
  }

  /**
   * Non-streaming chat — returns full response text.
   */
  async function chat({ model, messages, systemPrompt }) {
    init();

    if (!window.puter || !puter.ai) {
      throw new Error("Puter.js SDK not loaded.");
    }

    const fullMessages = [];
    if (systemPrompt) {
      fullMessages.push({ role: "system", content: systemPrompt });
    }
    fullMessages.push(...messages);

    const response = await puter.ai.chat(
      fullMessages[fullMessages.length - 1].content,
      {
        model,
        stream: false,
        messages: fullMessages,
      }
    );

    return response.text || "";
  }

  /**
   * Build the system prompt for the current mode + context.
   *
   * @param {string} mode          — "general"|"document"|"research"|"code"
   * @param {Array}  ragChunks     — [{source,page,snippet,relevance}]
   * @param {Array}  searchResults — [{title,url,snippet}]
   * @returns {string}
   */
  function buildSystemPrompt(mode, ragChunks = [], searchResults = []) {
    const timestamp = new Date().toUTCString();
    let system = `You are AI Assistant Pro, an expert multipurpose AI assistant. Current time: ${timestamp}.\n\n`;

    switch (mode) {
      case "document":
        system += `## Document Expert Mode\nYou answer questions using ONLY the document excerpts provided below. Always cite page numbers inline as [1], [2], etc. If the excerpts don't contain enough information, say so clearly. End every answer with a **References** section.\n\n`;
        break;
      case "research":
        system += `## Deep Research Mode\nYou synthesize information from multiple web sources provided below. Always cite your sources inline using numbered references. Provide a comprehensive, well-structured answer with clear section headings.\n\n`;
        break;
      case "code":
        system += `## Code & Data Analyst Mode\nYou are an expert software engineer and data scientist. Provide clean, well-commented, production-quality code. Explain your approach and any edge cases. Use markdown code blocks with language tags.\n\n`;
        break;
      default:
        system += `## General Assistant Mode\nYou are a helpful, accurate, and concise AI assistant. Answer clearly and directly. Format your response in clean markdown.\n\n`;
    }

    // Inject RAG context
    if (ragChunks.length > 0) {
      system += `## Document Context (use these excerpts to answer):\n\n`;
      ragChunks.forEach((chunk, i) => {
        const page = chunk.page ? ` · Page ${chunk.page}` : "";
        system += `[${i + 1}] **${chunk.source}${page}** (relevance: ${(chunk.relevance || 0).toFixed(2)})\n${chunk.snippet}\n\n`;
      });
      system += `---\n\n`;
    }

    // Inject web search context
    if (searchResults.length > 0) {
      system += `## Web Search Results (use these to answer):\n\n`;
      searchResults.forEach((r, i) => {
        system += `[${i + 1 + ragChunks.length}] **${r.title}** — ${r.url}\n${r.snippet}\n\n`;
      });
      system += `---\n\n`;
    }

    return system;
  }

  return { streamChat, chat, buildSystemPrompt };
})();
