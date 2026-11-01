"use strict";

const fs = require("fs");
const path = require("path");

const root = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG";
const payload = JSON.parse(fs.readFileSync(path.join(root, ".ua/tmp/batch-prompt-1.json"), "utf8"));

const nodes = [];
const edges = [];
const ids = new Set();
const edgeKeys = new Set();

function complexityFromLines(lines) {
  if (lines > 200) return "complex";
  if (lines >= 50) return "moderate";
  return "simple";
}

function addNode(node) {
  if (!node.summary || !node.summary.trim()) throw new Error("empty summary " + node.id);
  if (!Array.isArray(node.tags) || node.tags.length < 3 || node.tags.length > 5) {
    throw new Error("tags " + node.id + " " + JSON.stringify(node.tags));
  }
  if (ids.has(node.id)) throw new Error("dup " + node.id);
  ids.add(node.id);
  nodes.push(node);
}

function addEdge(source, target, type, weight) {
  if (source === target) throw new Error("self " + type + " " + source);
  const key = source + "|" + type + "|" + target;
  if (edgeKeys.has(key)) return;
  edgeKeys.add(key);
  edges.push({ source, target, type, direction: "forward", weight });
}

function fileNode(filePath, summary, tags, nonEmpty, languageNotes) {
  const node = {
    id: "file:" + filePath,
    type: "file",
    name: filePath.split("/").pop(),
    filePath,
    summary,
    tags,
    complexity: complexityFromLines(nonEmpty),
  };
  if (languageNotes) node.languageNotes = languageNotes;
  addNode(node);
}

function symbolNode(filePath, type, name, start, end, summary, tags, exported) {
  const lines = end - start + 1;
  const node = {
    id: type + ":" + filePath + ":" + name,
    type,
    name,
    filePath,
    lineRange: [start, end],
    summary,
    tags,
    complexity: complexityFromLines(lines),
  };
  addNode(node);
  addEdge("file:" + filePath, node.id, "contains", 1.0);
  if (exported) addEdge("file:" + filePath, node.id, "exports", 0.8);
}

function endpointNode(method, routePath, summary, tags) {
  const filePath = "backend/api/routes.py";
  const name = method + " " + routePath;
  const node = {
    id: "endpoint:" + filePath + ":" + name,
    type: "endpoint",
    name,
    filePath,
    summary,
    tags,
    complexity: "simple",
  };
  addNode(node);
  addEdge("file:" + filePath, node.id, "contains", 1.0);
}

fileNode(
  "backend/api/mcp_server.py",
  "Optional stdio MCP server that exposes document search, web search, GitHub connect, and the full chat pipeline to MCP clients.",
  ["entry-point", "mcp", "service", "api-handler"],
  93,
  "Tools are registered on MCPServer and resolve a lazily built ServiceContainer inside the server event loop."
);
fileNode(
  "backend/api/routes.py",
  "FastAPI router of thin HTTP handlers for chat streaming, sessions, uploads, GitHub, retrieval, and web search.",
  ["api-handler", "entry-point", "http", "streaming"],
  328,
  "Handlers stay thin: FastAPI Depends injects services from the container, and POST /chat/stream serializes pipeline events as SSE."
);
fileNode(
  "backend/client/search_client.py",
  "Async HTTP clients for Tavily and Serper web search, each with a pooled client and one retry on transient failures.",
  ["service", "http-client", "web-search"],
  77
);
fileNode(
  "backend/core/container.py",
  "Composition root that constructs every service once, wires constructor dependencies, and exposes FastAPI Depends getters.",
  ["dependency-injection", "factory", "service", "singleton"],
  155,
  "An overrides dict lets tests and the MCP server swap embedders, the LLM client, or other collaborators without reading the environment."
);
fileNode(
  "backend/main.py",
  "FastAPI application factory that configures CORS, mounts the SPA, and owns container startup and shutdown.",
  ["entry-point", "fastapi", "application"],
  92,
  "When launched as a script, the module rewrites sys.path so the backend package stays importable and local folders are not treated as top-level modules."
);
fileNode(
  "backend/models/api.py",
  "Pydantic request and response schemas for sessions, chat, uploads, retrieval, search, GitHub, and health.",
  ["data-model", "validation", "api-schema"],
  177
);
fileNode(
  "backend/services/artifact.py",
  "Validates a structured download document and renders it to Markdown, text, Excel, Word, or PDF only when a file is requested.",
  ["service", "document-generation", "validation"],
  1063,
  "File bytes are produced at download time from one validated block document and are not stored on disk."
);
fileNode(
  "backend/services/chat_service.py",
  "Chat pipeline that routes a turn, checks the semantic cache, retrieves and searches in parallel, then streams a model answer with failover.",
  ["service", "streaming", "rag", "orchestration"],
  485
);
fileNode(
  "backend/services/search_service.py",
  "Web search policy with TTL caching, a hard deadline, Tavily-to-Serper fallback, and parallel sub-queries for research mode.",
  ["service", "web-search", "caching"],
  115
);
fileNode(
  "backend/utils/cache.py",
  "Thread-safe in-process caches: a bounded TTL LRU cache and a per-scope cosine similarity cache of previous answers.",
  ["utility", "caching", "concurrency"],
  141,
  "SemanticCache stores L2-normalized embeddings and only returns a hit inside the same scope tuple, so answers do not leak across users or a changed index."
);
fileNode(
  "backend/utils/sse.py",
  "Encodes a pipeline event dict as one Server-Sent Events frame with an event name and JSON data.",
  ["utility", "serialization", "streaming"],
  8
);
fileNode(
  "tests/test_api.py",
  "Offline HTTP tests that exercise every API route, the SSE chat contract, uploads, search fallback, and rate limits.",
  ["test", "api-handler", "integration"],
  406
);
fileNode(
  "tests/test_live.py",
  "Opt-in live tests that measure real model streaming, hybrid retrieval quality, and Tavily latency against provider keys.",
  ["test", "live", "latency"],
  197
);

// --- mcp_server ---
const mcp = "backend/api/mcp_server.py";
symbolNode(mcp, "class", "_Runtime", 28, 44,
  "Lazily builds and starts one ServiceContainer inside the MCP server event loop, guarded by an asyncio lock.",
  ["singleton", "lifecycle", "dependency-injection"], false);
symbolNode(mcp, "function", "search_documents", 51, 56,
  "MCP tool that runs hybrid document search for a user and returns ranked source, page, relevance, and text.",
  ["mcp", "retrieval", "api-handler"], true);
symbolNode(mcp, "function", "list_documents", 60, 65,
  "MCP tool that lists the user's indexed documents with id, filename, type, and chunk count.",
  ["mcp", "documents", "api-handler"], true);
symbolNode(mcp, "function", "web_search", 69, 74,
  "MCP tool that searches the web and returns status plus title, URL, and snippet results.",
  ["mcp", "web-search", "api-handler"], true);
symbolNode(mcp, "function", "connect_github_repo", 78, 84,
  "MCP tool that indexes a GitHub repository into the user's document store and returns the repo brief.",
  ["mcp", "github", "api-handler"], true);
symbolNode(mcp, "function", "list_github_repos", 88, 93,
  "MCP tool that lists GitHub repositories already connected for a user.",
  ["mcp", "github", "api-handler"], true);
symbolNode(mcp, "function", "github_context", 97, 100,
  "MCP tool that returns the stored briefs for the user's connected GitHub repositories.",
  ["mcp", "github", "api-handler"], true);
symbolNode(mcp, "function", "ask", 104, 115,
  "MCP tool that opens a session and runs the full chat pipeline, returning answer, model, route, and citations.",
  ["mcp", "chat", "api-handler"], true);
symbolNode(mcp, "function", "main", 118, 120,
  "Starts the MCP server on stdio after sending logs to stderr so stdout stays the protocol channel.",
  ["entry-point", "mcp", "cli"], true);

// --- routes functions ---
const routes = "backend/api/routes.py";
const routeFns = [
  ["ping", 126, 128, "Returns a lightweight pong payload for uptime monitors.", ["api-handler", "health", "http"]],
  ["health", 132, 134, "Reports container health status, per-service state, and diagnostic details.", ["api-handler", "health", "http"]],
  ["public_config", 138, 142, "Returns upload size and extension limits plus whether GitHub OAuth is configured.", ["api-handler", "configuration", "http"]],
  ["list_models", 146, 151, "Lists configured chat models and the default model, optionally refreshing provider discovery.", ["api-handler", "llm", "http"]],
  ["chat_stream", 157, 174, "Streams a chat turn as SSE after rate limiting and confirming the session exists.", ["api-handler", "streaming", "chat"]],
  ["chat_complete", 178, 194, "Runs the same chat pipeline as the stream endpoint and returns one JSON document.", ["api-handler", "chat", "http"]],
  ["generate", 198, 217, "Legacy completion endpoint that calls the LLM with model fallback and no RAG or history.", ["api-handler", "llm", "http"]],
  ["list_sessions", 223, 224, "Lists session summaries for a user.", ["api-handler", "sessions", "http"]],
  ["create_session", 229, 230, "Creates a chat session and returns the full session schema.", ["api-handler", "sessions", "http"]],
  ["get_session", 234, 238, "Loads one session with its messages, or 404 when the id is unknown.", ["api-handler", "sessions", "http"]],
  ["update_session", 242, 247, "Patches a session title, model, or mode.", ["api-handler", "sessions", "http"]],
  ["delete_session", 251, 256, "Deletes a session and returns 204, or 404 when it is missing.", ["api-handler", "sessions", "http"]],
  ["save_message", 261, 266, "Appends a message to a session and returns the stored message.", ["api-handler", "sessions", "http"]],
  ["download_message", 274, 292, "Renders a stored message artifact to the requested format and returns the file bytes.", ["api-handler", "download", "documents"]],
  ["message_feedback", 296, 301, "Stores thumbs feedback on a message and returns the updated message.", ["api-handler", "sessions", "http"]],
  ["upload_document", 307, 328, "Ingests an uploaded file or URL, enforcing the size limit, and returns the index result.", ["api-handler", "documents", "ingestion"]],
  ["list_documents", 332, 333, "Lists documents indexed for a user.", ["api-handler", "documents", "http"]],
  ["delete_document", 337, 341, "Deletes one user document and its index entries, or 404 when it is missing.", ["api-handler", "documents", "http"]],
  ["github_connect", 352, 358, "Connects a GitHub repository URL and returns the stored repo row.", ["api-handler", "github", "http"]],
  ["github_repos", 362, 363, "Lists GitHub repositories connected for a user.", ["api-handler", "github", "http"]],
  ["github_delete", 367, 371, "Disconnects a GitHub repository for a user.", ["api-handler", "github", "http"]],
  ["github_device_start", 375, 379, "Starts the GitHub device OAuth flow and returns the user code and verification URL.", ["api-handler", "github", "oauth"]],
  ["github_device_poll", 383, 388, "Polls the GitHub device flow and connects the repository when the user approves.", ["api-handler", "github", "oauth"]],
  ["retrieve", 394, 403, "Runs hybrid retrieval with reranking and returns ranked chunks plus timing.", ["api-handler", "retrieval", "rag"]],
  ["web_search", 407, 410, "Runs a web search and returns results with an ok, disabled, or error status.", ["api-handler", "web-search", "http"]],
];
for (const [name, start, end, summary, tags] of routeFns) {
  symbolNode(routes, "function", name, start, end, summary, tags, true);
}

const endpoints = [
  ["GET", "/ping", "Liveness route used by uptime monitors.", ["http", "health", "api"]],
  ["GET", "/health", "Service health route returning status, dependency checks, and details.", ["http", "health", "api"]],
  ["GET", "/config", "Public configuration route for upload limits and GitHub OAuth readiness.", ["http", "configuration", "api"]],
  ["GET", "/models", "Model catalog route listing available chat models and the default id.", ["http", "llm", "api"]],
  ["POST", "/chat/stream", "Streaming chat route that emits SSE events from the chat pipeline.", ["http", "streaming", "chat"]],
  ["POST", "/chat", "Non-streaming chat route that returns the finished answer as JSON.", ["http", "chat", "api"]],
  ["POST", "/generate", "Legacy generate route for a plain completion without retrieval or history.", ["http", "llm", "api"]],
  ["GET", "/sessions", "Lists chat session summaries for a user.", ["http", "sessions", "api"]],
  ["POST", "/sessions", "Creates a chat session.", ["http", "sessions", "api"]],
  ["GET", "/sessions/{session_id}", "Fetches one session including its messages.", ["http", "sessions", "api"]],
  ["PATCH", "/sessions/{session_id}", "Updates session title, model, or mode.", ["http", "sessions", "api"]],
  ["DELETE", "/sessions/{session_id}", "Deletes a session.", ["http", "sessions", "api"]],
  ["POST", "/sessions/{session_id}/messages", "Appends a message to a session.", ["http", "sessions", "messages"]],
  ["GET", "/messages/{message_id}/download", "Downloads a message artifact as Markdown, text, Excel, Word, or PDF.", ["http", "download", "documents"]],
  ["POST", "/messages/{message_id}/feedback", "Records feedback on an assistant message.", ["http", "messages", "api"]],
  ["POST", "/upload", "Uploads a file or URL for parsing and indexing.", ["http", "documents", "ingestion"]],
  ["GET", "/documents", "Lists indexed documents for a user.", ["http", "documents", "api"]],
  ["DELETE", "/documents/{doc_id}", "Deletes an indexed document.", ["http", "documents", "api"]],
  ["POST", "/github/connect", "Connects and indexes a GitHub repository.", ["http", "github", "api"]],
  ["GET", "/github/repos", "Lists connected GitHub repositories.", ["http", "github", "api"]],
  ["DELETE", "/github/repos/{repo_id}", "Removes a connected GitHub repository.", ["http", "github", "api"]],
  ["POST", "/github/oauth/device", "Starts GitHub device-code OAuth.", ["http", "github", "oauth"]],
  ["POST", "/github/oauth/poll", "Polls GitHub device-code OAuth until it completes or expires.", ["http", "github", "oauth"]],
  ["POST", "/retrieve", "Hybrid RAG retrieval route with optional reranking.", ["http", "retrieval", "rag"]],
  ["POST", "/search", "Web search route backed by Tavily with a Serper fallback.", ["http", "web-search", "api"]],
];
for (const [method, routePath, summary, tags] of endpoints) {
  endpointNode(method, routePath, summary, tags);
}

// --- search clients ---
const searchClient = "backend/client/search_client.py";
symbolNode(searchClient, "class", "TavilyClient", 10, 43,
  "Posts a Tavily search request and returns raw result dicts, retrying once on 5xx or transport errors.",
  ["http-client", "web-search", "service"], true);
symbolNode(searchClient, "class", "SerperClient", 46, 88,
  "Fallback search client that queries Serper and maps organic results into the same raw dict shape as Tavily.",
  ["http-client", "web-search", "fallback"], true);

// --- container ---
const container = "backend/core/container.py";
symbolNode(container, "class", "ServiceContainer", 42, 162,
  "Builds the database, models, retrieval, LLM, search, sessions, caches, GitHub, and chat services, then runs startup, warmup, shutdown, and health.",
  ["dependency-injection", "lifecycle", "service"], true);
const getters = [
  ["get_container", 167, 168, "FastAPI dependency that returns the container stored on application state."],
  ["get_chat_service", 171, 172, "FastAPI dependency that returns the shared ChatService."],
  ["get_session_service", 175, 176, "FastAPI dependency that returns the shared SessionService."],
  ["get_document_service", 179, 180, "FastAPI dependency that returns the shared DocumentService."],
  ["get_retrieval_service", 183, 184, "FastAPI dependency that returns the shared RetrievalService."],
  ["get_search_service", 187, 188, "FastAPI dependency that returns the shared WebSearchService."],
  ["get_llm_service", 191, 192, "FastAPI dependency that returns the shared LLMService."],
  ["get_github_service", 195, 196, "FastAPI dependency that returns the shared GitHubService."],
];
for (const [name, start, end, summary] of getters) {
  symbolNode(container, "function", name, start, end, summary, ["dependency-injection", "fastapi", "factory"], true);
}

// --- main ---
const main = "backend/main.py";
symbolNode(main, "class", "RevalidatingStaticFiles", 56, 62,
  "Static file mount that adds Cache-Control: no-cache so the browser revalidates SPA assets after a deploy.",
  ["middleware", "static-files", "http"], true);
symbolNode(main, "function", "create_app", 65, 109,
  "Builds the FastAPI app, attaches CORS and the router, mounts frontend assets, and starts the service container on lifespan.",
  ["entry-point", "factory", "fastapi"], true);

// --- models ---
const models = "backend/models/api.py";
const modelClasses = [
  ["SessionCreate", 15, 19, "Request body for creating a session, with user id, title, model, and mode."],
  ["SessionUpdate", 22, 25, "Partial update for a session title, model, or mode."],
  ["CitationSchema", 28, 34, "Citation attached to a message, with source type, page or URL, and snippet."],
  ["MessageSchema", 37, 47, "Stored chat message including role, content, citations, feedback, and metadata."],
  ["SessionSchema", 50, 60, "Full session record with timestamps and its message list."],
  ["SessionSummary", 63, 72, "Session list item without the message body."],
  ["MessageCreate", 77, 80, "Request body for appending a role, content, and citations to a session."],
  ["FeedbackRequest", 83, 84, "Request body carrying the feedback value for a message."],
  ["ChatRequest", 89, 96, "Chat turn input: session, message, user, mode, model, web search toggle, and regenerate flag."],
  ["ChatResponse", 99, 106, "Completed chat payload with message id, answer, model, route, cache flag, citations, and timings."],
  ["UploadResponse", 111, 117, "Result of an ingest, including document id, filename, type, chunk count, and indexing state."],
  ["RetrieveRequest", 122, 126, "Retrieval query with user id, relevance threshold, and top-k."],
  ["RetrievedChunk", 129, 133, "One retrieved excerpt with source, page, snippet, and relevance."],
  ["RetrieveResponse", 136, 142, "Retrieval result list plus whether RAG and reranking ran, and elapsed time."],
  ["SearchRequest", 147, 149, "Web search query and maximum result count."],
  ["SearchResult", 152, 156, "One web hit with title, URL, snippet, and score."],
  ["SearchResponse", 159, 162, "Web search payload with results, the original query, and a status string."],
  ["DocumentSchema", 167, 174, "Indexed document summary with filename, type, chunk count, and indexed time."],
  ["ModelInfo", 179, 186, "Chat model descriptor with provider, availability, reason, and default flag."],
  ["ChatMessage", 189, 191, "Role and content pair used by the legacy generate request."],
  ["GenerateRequest", 194, 197, "Legacy completion request with model id, messages, and an optional system prompt."],
  ["GenerateResponse", 200, 202, "Legacy completion response with text and the model that produced it."],
  ["UploadLimits", 207, 209, "Public upload ceiling in megabytes and the allowed extensions."],
  ["PublicConfig", 212, 215, "Client-visible config bundling upload limits and GitHub OAuth readiness."],
  ["GithubConnectRequest", 218, 221, "Request to connect a repository URL with an optional token and user id."],
  ["GithubRepoSchema", 224, 232, "Connected repository row with owner, name, branch, file count, and connected time."],
  ["GithubDeviceStart", 235, 240, "Device-flow start payload with user code, verification URI, expiry, and interval."],
  ["GithubDevicePoll", 243, 245, "Device-flow poll body carrying the device code and user id."],
  ["HealthResponse", 250, 253, "Health payload with overall status, per-service states, and diagnostic details."],
];
for (const [name, start, end, summary] of modelClasses) {
  symbolNode(models, "class", name, start, end, summary, ["data-model", "validation", "api-schema"], true);
}

// --- artifact ---
const artifact = "backend/services/artifact.py";
symbolNode(artifact, "class", "ArtifactError", 87, 92,
  "Raised when a download document is empty, over the caps, or not the expected JSON.",
  ["validation", "error", "documents"], true);
const artifactFns = [
  ["wants_file", 145, 152, "Detects a download request from a named format, extension, or restyle follow-up.", ["validation", "documents", "intent"], true],
  ["is_restyle", 155, 160, "Detects a short follow-up that only asks to change formatting of the previous document.", ["validation", "documents", "intent"], true],
  ["style_note", 163, 167, "Returns a formatting instruction when the user message names style words such as APA or landscape.", ["prompt", "documents", "formatting"], true],
  ["is_bare_save", 182, 197, "Detects a save-only request such as 'i want a pdf' with no new topic left in the text.", ["validation", "documents", "intent"], true],
  ["topic_title", 200, 211, "Chooses a file title, falling back to the earlier user question for a bare save or restyle.", ["documents", "formatting", "utility"], true],
  ["should_author_document", 214, 225, "Decides that this turn should attach a download, and skips short clarifying questions.", ["validation", "documents", "intent"], true],
  ["document_source", 228, 238, "Picks the previous assistant answer when the user only asked to save what was just written.", ["documents", "history", "utility"], true],
  ["document_from_markdown", 241, 250, "Converts markdown the model already wrote into a validated block document without a second model call.", ["documents", "parsing", "validation"], true],
  ["preferred_format", 253, 267, "Maps the user's wording to pdf, docx, xlsx, txt, or markdown, defaulting to markdown.", ["documents", "formatting", "utility"], true],
  ["fallback_title", 270, 273, "Builds a short title by stripping format words from the user text.", ["documents", "formatting", "utility"], true],
  ["artifact_messages", 276, 283, "Builds the system and user messages that ask the model for one JSON document.", ["prompt", "documents", "llm"], true],
  ["parse_model_document", 286, 297, "Parses model JSON, stripping fences and recovering a surrounding object, or raises ArtifactError.", ["parsing", "validation", "documents"], true],
  ["validate_document", 300, 317, "Drops unknown fields, strips control characters, and rejects empty or over-cap documents.", ["validation", "documents", "sanitization"], true],
  ["render_download", 320, 340, "Renders an already validated document to bytes, media type, and filename for one format.", ["documents", "rendering", "download"], true],
  ["content_disposition", 343, 348, "Builds a Content-Disposition header with an ASCII fallback and a UTF-8 filename.", ["http", "download", "utility"], true],
  ["safe_filename", 351, 366, "Makes a short download name from the heading, at most four words joined by underscores.", ["documents", "sanitization", "utility"], true],
  ["strip_markdown_inline", 369, 376, "Removes inline markdown emphasis so non-markdown formats do not keep leftover asterisks.", ["documents", "sanitization", "formatting"], true],
  ["render_markdown", 379, 407, "Renders the block document as a Markdown string.", ["rendering", "markdown", "documents"], true],
  ["render_text", 410, 435, "Renders the block document as plain text.", ["rendering", "documents", "formatting"], true],
  ["render_xlsx", 438, 545, "Renders headings, paragraphs, lists, and tables into an Excel workbook.", ["rendering", "spreadsheet", "documents"], true],
  ["render_docx", 548, 611, "Renders the block document into a styled Word file.", ["rendering", "documents", "formatting"], true],
  ["render_pdf", 614, 695, "Renders the block document into a styled PDF.", ["rendering", "pdf", "documents"], true],
  ["_block", 698, 728, "Normalizes one heading, paragraph, list, code, or table block and rejects unknown shapes.", ["validation", "documents", "parsing"], false],
  ["_list_block", 731, 747, "Validates a list block's items and caps.", ["validation", "documents", "parsing"], false],
  ["_table", 750, 770, "Validates table columns and rows against the cell and size caps.", ["validation", "documents", "parsing"], false],
  ["_use_previous", 819, 832, "Decides that a short generate-file request should file the last answer instead of an export how-to.", ["documents", "intent", "history"], false],
  ["_choose_title", 843, 853, "Keeps a markdown opening heading, or uses the user's question for a prose save.", ["documents", "formatting", "parsing"], false],
  ["_promote_sections", 856, 890, "Splits label-style prose into a heading plus a paragraph so saved answers gain structure.", ["parsing", "documents", "formatting"], false],
  ["_parse_markdown", 908, 992, "Parses markdown into heading, paragraph, list, code, and table blocks.", ["parsing", "markdown", "documents"], false],
  ["_fit_blocks", 1009, 1062, "Trims parsed blocks to the document caps before validation.", ["validation", "documents", "sanitization"], false],
  ["_style_docx_heading", 1096, 1106, "Applies navy heading font, size, and color to a Word paragraph.", ["rendering", "formatting", "documents"], false],
  ["_style_docx_body", 1109, 1120, "Applies body font styling to a Word paragraph.", ["rendering", "formatting", "documents"], false],
  ["_docx_cell", 1133, 1146, "Writes text into a Word table cell, with header shading when requested.", ["rendering", "documents", "formatting"], false],
  ["_add_runs", 1149, 1170, "Splits inline markdown into bold, italic, and code runs inside a Word paragraph.", ["rendering", "markdown", "formatting"], false],
  ["_pdf_table", 1173, 1196, "Draws a styled table into the PDF renderer.", ["rendering", "pdf", "documents"], false],
  ["log_ready", 1199, 1201, "Logs that a document was accepted without writing the body.", ["logging", "documents", "utility"], true],
];
for (const [name, start, end, summary, tags, exported] of artifactFns) {
  symbolNode(artifact, "function", name, start, end, summary, tags, exported);
}

// --- chat ---
const chat = "backend/services/chat_service.py";
symbolNode(chat, "class", "RouteDecision", 51, 61,
  "Records whether a turn uses RAG, web search, and research mode, and exposes a rag, web, rag+web, or direct label.",
  ["routing", "data-model", "chat"], true);
symbolNode(chat, "class", "QueryRouter", 81, 108,
  "Heuristic router that classifies smalltalk, fresh-info, and follow-ups without calling an LLM.",
  ["routing", "chat", "heuristic"], true);
symbolNode(chat, "class", "ChatCommand", 114, 120,
  "Input to the chat pipeline: session, message, mode, model, web-search toggle, and regenerate flag.",
  ["data-model", "chat", "command"], true);
symbolNode(chat, "class", "ChatService", 131, 538,
  "Streams or completes a turn: persist the user message, route, rewrite follow-ups, check the semantic cache, retrieve and search in parallel, prompt, generate with failover, attach artifacts, and save the answer.",
  ["service", "streaming", "rag", "orchestration"], true);

// --- search service ---
const searchSvc = "backend/services/search_service.py";
symbolNode(searchSvc, "class", "SearchOutcome", 27, 31,
  "Web search result with hits, a status of ok, disabled, error, or timeout, elapsed time, and a cache flag.",
  ["data-model", "web-search", "service"], true);
symbolNode(searchSvc, "class", "WebSearchService", 34, 125,
  "Searches via Tavily with a Serper fallback, caches hits in a TTL cache, and fans out sub-queries for research mode.",
  ["service", "web-search", "caching"], true);

// --- cache ---
const cache = "backend/utils/cache.py";
symbolNode(cache, "class", "TTLCache", 20, 76,
  "Bounded thread-safe LRU cache whose entries expire after a TTL.",
  ["caching", "utility", "concurrency"], true);
symbolNode(cache, "class", "CachedAnswer", 80, 86,
  "Cached chat answer with text, citations, model, route, and an optional artifact.",
  ["data-model", "caching", "chat"], true);
symbolNode(cache, "class", "SemanticCache", 96, 167,
  "Returns a stored answer when a query embedding is close enough to a previous one in the same scope.",
  ["caching", "embeddings", "chat"], true);

// --- sse ---
symbolNode("backend/utils/sse.py", "function", "format_sse", 9, 12,
  "Serializes an event name and JSON data payload into one SSE frame.",
  ["serialization", "streaming", "utility"], true);

// --- tests ---
const testApi = "tests/test_api.py";
const apiTests = [
  ["new_session", 20, 24, "Test helper that creates a session through POST /sessions and returns the JSON body.", ["test", "helper", "sessions"]],
  ["chat", 27, 31, "Test helper that posts a message to /chat/stream and parses the SSE body.", ["test", "helper", "streaming"]],
  ["names", 34, 35, "Test helper that collects SSE event names from a parsed stream.", ["test", "helper", "streaming"]],
  ["text_of", 38, 39, "Test helper that concatenates token text from an SSE stream.", ["test", "helper", "streaming"]],
  ["test_ping_health_models_and_spa", 44, 55, "Checks ping, health, the model list, and that the SPA index is served.", ["test", "health", "api"]],
  ["test_health_degraded_without_llm_keys", 58, 75, "Checks that health is degraded when no LLM provider key is configured.", ["test", "health", "llm"]],
  ["test_session_crud_and_cascade", 80, 103, "Covers session create, read, update, delete, and cascade removal of messages.", ["test", "sessions", "api"]],
  ["test_session_title_auto_generated_and_ordering", 106, 113, "Checks automatic session titles and list ordering.", ["test", "sessions", "api"]],
  ["test_upload_list_retrieve_delete_flow", 118, 135, "Uploads a document, lists it, retrieves against it, then deletes it.", ["test", "documents", "retrieval"]],
  ["test_upload_rejections", 143, 146, "Parametrized check that rejected uploads return the expected status.", ["test", "documents", "validation"]],
  ["test_upload_requires_file_or_url_and_blocks_ssrf", 149, 154, "Requires a file or URL on upload and blocks private or non-http targets.", ["test", "documents", "security"]],
  ["test_upload_oversize_returns_413", 157, 161, "Checks that a file over the configured limit returns 413.", ["test", "documents", "validation"]],
  ["test_upload_csv_and_markdown", 164, 173, "Uploads CSV and Markdown and checks they are indexed.", ["test", "documents", "parsing"]],
  ["test_user_id_validation", 176, 179, "Checks that an invalid user id is rejected.", ["test", "validation", "api"]],
  ["test_search_endpoint_ok_disabled_and_error", 184, 194, "Covers web search success, a disabled provider, and an error status.", ["test", "web-search", "api"]],
  ["test_search_falls_back_to_serper_when_tavily_fails", 197, 204, "Checks that a Tavily failure falls back to Serper.", ["test", "web-search", "fallback"]],
  ["test_search_uses_serper_when_tavily_is_unconfigured", 207, 214, "Checks that Serper is used when Tavily has no API key.", ["test", "web-search", "fallback"]],
  ["test_chat_direct_route_streams_and_persists", 219, 233, "Streams a direct chat turn and checks the answer is persisted.", ["test", "chat", "streaming"]],
  ["test_chat_rag_route_uses_documents_and_returns_citations", 236, 250, "Checks a RAG turn uses uploaded documents and returns citations.", ["test", "rag", "citations"]],
  ["test_chat_web_route_and_graceful_search_failure", 253, 284, "Checks the web route and that a search failure still completes the turn.", ["test", "web-search", "chat"]],
  ["test_chat_research_mode_always_searches_multi_query", 287, 293, "Checks research mode always runs a multi-query web search.", ["test", "web-search", "chat"]],
  ["test_chat_document_mode_without_documents_tells_the_model", 296, 301, "Checks document mode still prompts the model when the user has no documents.", ["test", "rag", "chat"]],
  ["test_chat_history_is_used_without_duplicating_current_turn", 304, 314, "Checks history is sent to the model without duplicating the current user turn.", ["test", "chat", "sessions"]],
  ["test_semantic_cache_hit_skips_llm", 317, 330, "Checks a semantic cache hit replays the answer without another LLM call.", ["test", "caching", "chat"]],
  ["test_regenerate_replaces_last_answer", 333, 343, "Checks regenerate drops the last exchange and stores a new answer.", ["test", "chat", "sessions"]],
  ["test_failover_is_reported_to_client", 346, 353, "Checks model failover is reported to the client in the stream.", ["test", "llm", "streaming"]],
  ["test_all_models_down_yields_error_event_not_500", 356, 366, "Checks a total provider outage yields an SSE error instead of HTTP 500.", ["test", "llm", "streaming"]],
  ["test_mid_stream_error_keeps_partial_answer", 369, 376, "Checks a mid-stream failure keeps the partial answer that was already emitted.", ["test", "streaming", "chat"]],
  ["test_chat_validation_and_errors", 379, 386, "Checks chat validation failures and error responses.", ["test", "validation", "chat"]],
  ["test_non_streaming_chat_and_legacy_generate", 389, 400, "Covers POST /chat JSON and the legacy POST /generate endpoint.", ["test", "chat", "api"]],
  ["test_feedback_roundtrip", 403, 414, "Stores message feedback and reads it back on the session.", ["test", "sessions", "api"]],
  ["test_rate_limit_returns_429", 417, 424, "Checks the per-user rate limit returns 429.", ["test", "security", "api"]],
  ["test_client_disconnect_persists_partial_answer", 427, 441, "Checks a disconnected client still persists the partial answer.", ["test", "streaming", "sessions"]],
  ["test_migration_adds_columns_to_legacy_database", 444, 468, "Checks startup migration adds missing columns on a legacy database file.", ["test", "database", "migration"]],
  ["test_public_config_advertises_the_real_upload_limits", 471, 483, "Checks GET /config reports the real upload size and extensions.", ["test", "configuration", "documents"]],
  ["test_public_config_limit_follows_the_setting", 486, 488, "Checks the advertised upload limit follows the settings value.", ["test", "configuration", "documents"]],
  ["test_frontend_files_must_be_revalidated_so_a_page_never_mixes_old_and_new_scripts", 491, 503, "Checks SPA assets are served with no-cache so old and new scripts are not mixed.", ["test", "static-files", "http"]],
];
for (const [name, start, end, summary, tags] of apiTests) {
  symbolNode(testApi, "function", name, start, end, summary, tags, true);
}

const testLive = "tests/test_live.py";
const liveTests = [
  ["test_every_registered_model_streams_or_reports_why", 81, 106, "Live check that every registered model streams tokens or reports why it cannot.", ["test", "live", "llm"]],
  ["test_invalid_model_id_falls_back_without_error", 109, 124, "Live check that an unknown model id falls back to the default without an error event.", ["test", "live", "llm"]],
  ["test_retrieval_quality_hybrid_beats_single_signals", 127, 154, "Live check that hybrid retrieval ranks the expected corpus file for paraphrased and exact queries.", ["test", "live", "retrieval"]],
  ["test_end_to_end_chat_latency_report", 157, 187, "Live latency report for a full chat turn against the loaded corpus.", ["test", "live", "latency"]],
  ["test_retrieval_latency_distribution", 190, 212, "Live distribution of hybrid retrieval latency.", ["test", "live", "latency"]],
  ["test_tavily_basic_vs_advanced_latency", 215, 224, "Live comparison of Tavily basic and advanced search latency.", ["test", "live", "web-search"]],
];
for (const [name, start, end, summary, tags] of liveTests) {
  symbolNode(testLive, "function", name, start, end, summary, tags, true);
}

// --- imports: one edge per batchImportData path ---
let importCount = 0;
for (const [filePath, targets] of Object.entries(payload.batchImportData)) {
  for (const target of targets) {
    addEdge("file:" + filePath, "file:" + target, "imports", 0.7);
    importCount++;
  }
}

function fn(file, name) { return "function:" + file + ":" + name; }
function cl(file, name) { return "class:" + file + ":" + name; }

// --- calls ---
const calls = [
  [fn(mcp, "main"), fn("backend/core/config.py", "configure_logging")],
  [cl(mcp, "_Runtime"), fn("backend/core/config.py", "get_settings")],
  [cl(mcp, "_Runtime"), cl(container, "ServiceContainer")],
  [fn(mcp, "ask"), cl(models, "SessionCreate")],
  [fn(mcp, "ask"), cl(chat, "ChatCommand")],
  [fn(mcp, "ask"), cl(chat, "ChatService")],
  [fn(mcp, "search_documents"), cl("backend/services/retrieval_service.py", "RetrievalService")],
  [fn(mcp, "list_documents"), cl("backend/services/document_service.py", "DocumentService")],
  [fn(mcp, "web_search"), cl(searchSvc, "WebSearchService")],
  [fn(mcp, "connect_github_repo"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(mcp, "list_github_repos"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(mcp, "github_context"), cl("backend/services/github_service.py", "GitHubService")],

  [fn(routes, "health"), cl(container, "ServiceContainer")],
  [fn(routes, "public_config"), cl(models, "PublicConfig")],
  [fn(routes, "list_models"), cl("backend/services/llm_service.py", "LLMService")],
  [fn(routes, "chat_stream"), fn("backend/utils/sse.py", "format_sse")],
  [fn(routes, "chat_stream"), cl(chat, "ChatService")],
  [fn(routes, "chat_stream"), cl(chat, "ChatCommand")],
  [fn(routes, "chat_stream"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "chat_complete"), cl(chat, "ChatService")],
  [fn(routes, "chat_complete"), cl(models, "ChatResponse")],
  [fn(routes, "generate"), cl("backend/services/llm_service.py", "LLMService")],
  [fn(routes, "generate"), cl(models, "GenerateResponse")],
  [fn(routes, "list_sessions"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "create_session"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "get_session"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "update_session"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "delete_session"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "save_message"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "download_message"), fn(artifact, "render_download")],
  [fn(routes, "download_message"), fn(artifact, "validate_document")],
  [fn(routes, "download_message"), fn(artifact, "content_disposition")],
  [fn(routes, "message_feedback"), cl("backend/services/session_service.py", "SessionService")],
  [fn(routes, "upload_document"), cl("backend/services/document_service.py", "DocumentService")],
  [fn(routes, "list_documents"), cl("backend/services/document_service.py", "DocumentService")],
  [fn(routes, "delete_document"), cl("backend/services/document_service.py", "DocumentService")],
  [fn(routes, "github_connect"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(routes, "github_repos"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(routes, "github_delete"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(routes, "github_device_start"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(routes, "github_device_poll"), cl("backend/services/github_service.py", "GitHubService")],
  [fn(routes, "retrieve"), cl("backend/services/retrieval_service.py", "RetrievalService")],
  [fn(routes, "web_search"), cl(searchSvc, "WebSearchService")],

  [cl(container, "ServiceContainer"), fn("backend/client/http.py", "build_http_client")],
  [cl(container, "ServiceContainer"), cl(searchClient, "TavilyClient")],
  [cl(container, "ServiceContainer"), cl(searchClient, "SerperClient")],
  [cl(container, "ServiceContainer"), cl("backend/client/web_client.py", "WebPageClient")],
  [cl(container, "ServiceContainer"), cl("backend/core/database.py", "Database")],
  [cl(container, "ServiceContainer"), cl("backend/services/embedding_service.py", "EmbeddingService")],
  [cl(container, "ServiceContainer"), cl("backend/services/rerank_service.py", "RerankService")],
  [cl(container, "ServiceContainer"), cl("backend/services/index_store.py", "HybridIndexStore")],
  [cl(container, "ServiceContainer"), cl("backend/services/retrieval_service.py", "RetrievalService")],
  [cl(container, "ServiceContainer"), cl("backend/services/parser.py", "DocumentParser")],
  [cl(container, "ServiceContainer"), cl("backend/services/document_service.py", "DocumentService")],
  [cl(container, "ServiceContainer"), cl("backend/utils/rate_limiter.py", "QuotaGuard")],
  [cl(container, "ServiceContainer"), cl("backend/utils/rate_limiter.py", "RateLimiter")],
  [cl(container, "ServiceContainer"), cl("backend/services/llm_service.py", "LLMService")],
  [cl(container, "ServiceContainer"), cl(searchSvc, "WebSearchService")],
  [cl(container, "ServiceContainer"), cl("backend/services/prompt_builder.py", "PromptBuilder")],
  [cl(container, "ServiceContainer"), cl("backend/services/session_service.py", "SessionService")],
  [cl(container, "ServiceContainer"), cl(cache, "SemanticCache")],
  [cl(container, "ServiceContainer"), cl("backend/services/github_service.py", "GitHubService")],
  [cl(container, "ServiceContainer"), cl(chat, "ChatService")],
  [cl(container, "ServiceContainer"), cl(chat, "QueryRouter")],
  [cl(container, "ServiceContainer"), fn("backend/services/parser.py", "ocr_available")],

  [fn(container, "get_chat_service"), fn(container, "get_container")],
  [fn(container, "get_session_service"), fn(container, "get_container")],
  [fn(container, "get_document_service"), fn(container, "get_container")],
  [fn(container, "get_retrieval_service"), fn(container, "get_container")],
  [fn(container, "get_search_service"), fn(container, "get_container")],
  [fn(container, "get_llm_service"), fn(container, "get_container")],
  [fn(container, "get_github_service"), fn(container, "get_container")],

  [fn(main, "create_app"), fn("backend/core/config.py", "get_settings")],
  [fn(main, "create_app"), fn("backend/core/config.py", "configure_logging")],
  [fn(main, "create_app"), cl(container, "ServiceContainer")],

  [fn(artifact, "wants_file"), fn(artifact, "is_restyle")],
  [fn(artifact, "is_bare_save"), fn(artifact, "wants_file")],
  [fn(artifact, "topic_title"), fn(artifact, "is_bare_save")],
  [fn(artifact, "topic_title"), fn(artifact, "is_restyle")],
  [fn(artifact, "topic_title"), fn(artifact, "fallback_title")],
  [fn(artifact, "should_author_document"), fn(artifact, "wants_file")],
  [fn(artifact, "document_source"), fn(artifact, "_use_previous")],
  [fn(artifact, "document_from_markdown"), fn(artifact, "_parse_markdown")],
  [fn(artifact, "document_from_markdown"), fn(artifact, "_promote_sections")],
  [fn(artifact, "document_from_markdown"), fn(artifact, "_fit_blocks")],
  [fn(artifact, "document_from_markdown"), fn(artifact, "_choose_title")],
  [fn(artifact, "document_from_markdown"), fn(artifact, "validate_document")],
  [fn(artifact, "parse_model_document"), cl(artifact, "ArtifactError")],
  [fn(artifact, "render_download"), fn(artifact, "render_markdown")],
  [fn(artifact, "render_download"), fn(artifact, "render_text")],
  [fn(artifact, "render_download"), fn(artifact, "render_xlsx")],
  [fn(artifact, "render_download"), fn(artifact, "render_docx")],
  [fn(artifact, "render_download"), fn(artifact, "render_pdf")],
  [fn(artifact, "render_download"), fn(artifact, "safe_filename")],
  [fn(artifact, "validate_document"), fn(artifact, "_block")],
  [fn(artifact, "_block"), fn(artifact, "_list_block")],
  [fn(artifact, "_block"), fn(artifact, "_table")],
  [fn(artifact, "render_docx"), fn(artifact, "_style_docx_heading")],
  [fn(artifact, "render_docx"), fn(artifact, "_style_docx_body")],
  [fn(artifact, "render_docx"), fn(artifact, "_docx_cell")],
  [fn(artifact, "render_docx"), fn(artifact, "strip_markdown_inline")],
  [fn(artifact, "_style_docx_body"), fn(artifact, "_add_runs")],
  [fn(artifact, "render_pdf"), fn(artifact, "_pdf_table")],

  [cl(chat, "ChatService"), cl(chat, "QueryRouter")],
  [cl(chat, "ChatService"), cl(chat, "RouteDecision")],
  [cl(chat, "ChatService"), cl("backend/services/llm_service.py", "LLMService")],
  [cl(chat, "ChatService"), cl("backend/services/retrieval_service.py", "RetrievalService")],
  [cl(chat, "ChatService"), cl(searchSvc, "WebSearchService")],
  [cl(chat, "ChatService"), cl("backend/services/prompt_builder.py", "PromptBuilder")],
  [cl(chat, "ChatService"), fn("backend/services/prompt_builder.py", "select_specialist")],
  [cl(chat, "ChatService"), cl("backend/services/session_service.py", "SessionService")],
  [cl(chat, "ChatService"), cl(cache, "SemanticCache")],
  [cl(chat, "ChatService"), cl(cache, "CachedAnswer")],
  [cl(chat, "ChatService"), fn(artifact, "wants_file")],
  [cl(chat, "ChatService"), fn(artifact, "is_restyle")],
  [cl(chat, "ChatService"), fn(artifact, "is_bare_save")],
  [cl(chat, "ChatService"), fn(artifact, "style_note")],
  [cl(chat, "ChatService"), fn(artifact, "topic_title")],
  [cl(chat, "ChatService"), fn(artifact, "should_author_document")],
  [cl(chat, "ChatService"), fn(artifact, "document_source")],
  [cl(chat, "ChatService"), fn(artifact, "document_from_markdown")],
  [cl(chat, "ChatService"), fn(artifact, "preferred_format")],
  [cl(chat, "ChatService"), fn(artifact, "log_ready")],

  [cl(searchSvc, "WebSearchService"), cl(searchClient, "TavilyClient")],
  [cl(searchSvc, "WebSearchService"), cl(searchClient, "SerperClient")],
  [cl(searchSvc, "WebSearchService"), cl(cache, "TTLCache")],
  [cl(searchSvc, "WebSearchService"), cl(searchSvc, "SearchOutcome")],
  [cl(searchSvc, "WebSearchService"), fn("backend/utils/text.py", "normalize_query")],
  [cl(searchSvc, "WebSearchService"), cl(models, "SearchResult")],
];
for (const [source, target] of calls) addEdge(source, target, "calls", 0.8);

// MCP search/list/github tools call services that are not in mcp_server's neighborMap.
// Those class symbols ARE in other files' neighbor maps (routes, container), so they validate.
// DocumentService is in routes neighborMap. RetrievalService and GitHubService too.

const testedBy = [
  ["backend/api/routes.py", testApi],
  ["backend/main.py", testApi],
  ["backend/models/api.py", testApi],
  ["backend/services/chat_service.py", testApi],
  ["backend/core/container.py", testApi],
  ["backend/services/search_service.py", testApi],
  ["backend/client/search_client.py", testApi],
  ["backend/utils/cache.py", testApi],
  ["backend/utils/sse.py", testApi],
  ["backend/core/container.py", testLive],
  ["backend/services/chat_service.py", testLive],
  ["backend/models/api.py", testLive],
  ["backend/core/config.py", testLive],
  ["backend/services/search_service.py", testLive],
  ["backend/client/search_client.py", testLive],
  ["backend/services/retrieval_service.py", testLive],
];
for (const [prod, test] of testedBy) addEdge("file:" + prod, "file:" + test, "tested_by", 0.5);

// inherits: RevalidatingStaticFiles extends fastapi.staticfiles.StaticFiles (external) — no project edge.

// --- validate ---
const allowedFiles = new Set();
for (const [filePath, targets] of Object.entries(payload.batchImportData)) {
  allowedFiles.add(filePath);
  for (const t of targets) allowedFiles.add(t);
}
const allowedSymbols = new Set();
for (const neighbors of Object.values(payload.neighborMap)) {
  for (const n of neighbors) {
    allowedFiles.add(n.path);
    for (const s of n.symbols) allowedSymbols.add(n.path + ":" + s);
  }
}
for (const f of payload.files) allowedFiles.add(f.path);

const problems = [];
for (const e of edges) {
  for (const end of [e.source, e.target]) {
    if (ids.has(end)) continue;
    if (end.startsWith("file:")) {
      const p = end.slice(5);
      if (!allowedFiles.has(p)) problems.push("bad file " + end + " via " + e.type);
      continue;
    }
    const m = end.match(/^(function|class):(.+):([^:]+)$/);
    if (!m) {
      problems.push("bad id " + end);
      continue;
    }
    if (!allowedSymbols.has(m[2] + ":" + m[3])) problems.push("bad symbol " + end + " via " + e.source + " " + e.type);
  }
}

const expectedImports = Object.values(payload.batchImportData).reduce((n, a) => n + a.length, 0);
const actualImports = edges.filter((e) => e.type === "imports").length;
if (actualImports !== expectedImports) problems.push("imports " + actualImports + " != " + expectedImports);

const outPath = path.join(root, ".ua/intermediate/batch-1.json");
const body = { nodes, edges };
fs.writeFileSync(outPath, JSON.stringify(body, null, 2));
const counts = {};
for (const n of nodes) counts[n.type] = (counts[n.type] || 0) + 1;
const edgeCounts = {};
for (const e of edges) edgeCounts[e.type] = (edgeCounts[e.type] || 0) + 1;
console.log(JSON.stringify({
  nodes: nodes.length,
  edges: edges.length,
  nodeTypes: counts,
  edgeTypes: edgeCounts,
  importCount,
  problems,
}, null, 2));
if (problems.length) process.exit(1);
