"use strict";
const fs = require("fs");
const path = require("path");

const outDir = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG/.ua/intermediate";
const nodes = [];
const edges = [];

function fid(file, name) { return `function:${file}:${name}`; }
function fileId(file) { return `file:${file}`; }

function addFile(file, type, summary, tags, complexity, languageNotes) {
  const id = type === "document" ? `document:${file}` : `${type}:${file}`;
  const node = {
    id,
    type,
    name: file.split("/").pop(),
    filePath: file,
    summary,
    tags,
    complexity,
  };
  if (languageNotes) node.languageNotes = languageNotes;
  nodes.push(node);
  return id;
}

function spanComplexity(start, end) {
  const n = end - start + 1;
  if (n >= 80) return "complex";
  if (n >= 20) return "moderate";
  return "simple";
}

function addFns(file, list) {
  const parent = fileId(file);
  for (const [name, start, end, exported, summary, tags] of list) {
    if (!summary || tags.length < 3 || tags.length > 5) {
      throw new Error(`bad meta ${file} ${name} tags=${tags && tags.length}`);
    }
    const id = fid(file, name);
    nodes.push({
      id,
      type: "function",
      name,
      filePath: file,
      lineRange: [start, end],
      summary,
      tags,
      complexity: spanComplexity(start, end),
    });
    edges.push({ source: parent, target: id, type: "contains", direction: "forward", weight: 1.0 });
    if (exported) {
      edges.push({ source: parent, target: id, type: "exports", direction: "forward", weight: 0.8 });
    }
  }
}

function edge(source, target, type, weight) {
  if (source === target) throw new Error("self edge " + source);
  edges.push({ source, target, type, direction: "forward", weight });
}

addFile("Pipfile", "file",
  "Pipenv manifest that points at PyPI and pins Python 3.13, with empty runtime and development package lists.",
  ["configuration", "dependency-manifest", "python", "pipenv"], "simple");

addFile("backend/__init__.py", "file",
  "Empty package marker for the backend so Python treats the directory as a package.",
  ["package-marker", "backend", "python"], "simple");

addFile("backend/api/__init__.py", "file",
  "Package marker for the delivery layer, described as FastAPI HTTP routes plus an optional MCP server.",
  ["package-marker", "api", "http", "backend"], "simple");

addFile("backend/client/__init__.py", "file",
  "Package marker for external adapters (LLM providers, Tavily, embeddings, web pages), stating that clients own transport and policy stays in backend.services.",
  ["package-marker", "client", "integration", "backend"], "simple");

addFile("backend/config.py", "file",
  "Staged leftover settings module that loads dotenv and defines data and frontend paths, provider API keys, server bind and CORS, the SQLite URL, Cohere and FAISS retrieval constants, Tavily and LLM defaults, Groq and Puter model catalogs, four agent modes, and a 20-message session history limit.",
  ["configuration", "settings", "rag", "llm"], "moderate",
  "Module-level constants and os.getenv reads only; the file defines no functions or classes.");

addFile("backend/core/__init__.py", "file",
  "Package marker for settings, the database, shared exceptions, and the DI container, left empty so importing the container here does not cycle with services.",
  ["package-marker", "core", "dependency-injection", "backend"], "simple");

addFile("backend/models/__init__.py", "file",
  "Empty package marker for the backend models package.",
  ["package-marker", "data-model", "backend"], "simple");

addFile("backend/utils/__init__.py", "file",
  "Package marker for pure helpers that depend only on the standard library or numpy, not on the rest of the backend.",
  ["package-marker", "utility", "backend"], "simple");

addFile("data/.gitkeep", "file",
  "Empty placeholder that keeps the data directory present in version control.",
  ["placeholder", "data-directory", "git"], "simple");

addFile("docs/AI_ASSISTANT_PRO_MASTER_TASK.md", "document",
  "Implementation brief for turning AI Assistant Pro into a production frontend, in 18 phases covering inspection, the design system, layout, theme, animated identity, welcome and chat, composer, model selection, document, research, and code workspaces, accessibility, and verification.",
  ["documentation", "design-system", "frontend", "product-spec"], "complex");

addFile("frontend/__init__.py", "file",
  "Empty package marker so the frontend directory is importable as a Python package.",
  ["package-marker", "frontend", "python"], "simple");

addFile("frontend/assets/LICENSE-lucide.txt", "document",
  "ISC license for the Lucide icon set vendored as an inline SVG sprite in frontend/index.html (lucide-static v1.60.0, unmodified path data).",
  ["documentation", "license", "icons", "third-party"], "simple");

addFile("frontend/index.html", "file",
  "Single-page shell for AI Assistant Pro. An inline script sets theme, shell, and workspace before first paint, then the page loads the local stylesheets and classic scripts, with app.js last.",
  ["entry-point", "markup", "spa", "chat-ui"], "complex",
  "The head script is an anonymous IIFE that mirrors js/theme.js and the shell rules in js/ui.js so the first paint does not flash.");

addFile("frontend/js/api.js", "file",
  "Browser HTTP client exposed as window.API. It wraps JSON REST calls for sessions, documents, GitHub, models, and health, and it opens the POST /chat/stream SSE channel plus artifact downloads.",
  ["api-client", "http", "sse", "spa"], "complex",
  "Classic script IIFE, not an ES module. SSE frames are split on LF blank lines because the backend emits LF only.");

addFile("frontend/js/app.js", "file",
  "Browser entry and state owner. It boots the page, wires composer, session, upload, and GitHub controls, and drives the single POST /chat/stream round trip through window.API and window.Chat.",
  ["entry-point", "spa", "state", "streaming"], "complex");

addFile("frontend/js/chat.js", "file",
  "Chat view layer exposed as window.Chat. It builds user and assistant message DOM, streams Markdown, renders citations and file cards, and opens the artifact pane. It does not call fetch.",
  ["view", "chat-ui", "streaming", "markdown"], "complex");

addFile("frontend/js/components/ai-viz.js", "file",
  "Decorative AI status mark (window.AIViz) with idle, processing, responding, and error states. Motion is CSS-only; the module never fetches and holds no app state.",
  ["component", "visualization", "view", "accessibility"], "moderate");

addFile("frontend/js/components/code-preview.js", "file",
  "Sandboxed iframe preview pane (window.CodePreview) for HTML and SVG in the Code workspace, including Run and Download actions and a resizable pane.",
  ["component", "code-preview", "sandbox", "view"], "complex");

addFile("frontend/js/components/dialog.js", "file",
  "Native dialog helpers (window.Dialog) for modal open and close with focus return, plus an accessible confirm() that replaces window.confirm.",
  ["component", "dialog", "accessibility", "focus"], "moderate");

addFile("frontend/js/components/documents.js", "file",
  "Document list view (window.DocsView) for the Documents workspace and the My Documents dialog. It renders rows the controller supplies and reports delete clicks; it does not fetch.",
  ["component", "documents", "view", "list"], "complex");

addFile("frontend/js/components/menu.js", "file",
  "Anchored popover menu (window.Menu) with roving focus, menuitem and menuitemradio rows, and separators. Labels are set with textContent.",
  ["component", "menu", "accessibility", "popover"], "moderate");

addFile("frontend/js/components/model-select.js", "file",
  "Listbox popover (window.ModelSelect) over the native model select, which stays the source of truth. Choices dispatch a real change event that app.js already listens for.",
  ["component", "model-select", "listbox", "accessibility"], "complex");

addFile("frontend/js/components/motion.js", "file",
  "The only GSAP user (window.Motion). It plays the welcome, workspace, and intro entrances, skips animation when reduced motion is requested, and never fetches application data.",
  ["component", "animation", "gsap", "view"], "moderate");

addFile("frontend/js/components/notifications.js", "file",
  "In-memory notification popover (window.Notifications) for toasts and upload errors from this page session, with an unread dot on the bell. It does not persist or fetch.",
  ["component", "notifications", "popover", "view"], "moderate");

addFile("frontend/js/components/prefs.js", "file",
  "Browser-local view preferences (window.Prefs) for text size, chat density, and which toasts appear. Values are stored under pref.* keys and mirrored onto the html element; nothing is sent to the server.",
  ["component", "preferences", "local-storage", "view"], "moderate");

addFns("frontend/js/api.js", [
  ["_errorMessage", 10, 19, false, "Reads a non-OK response body and prefers a FastAPI detail string, the first validation message, or data.message, otherwise the HTTP status.", ["error-handling", "http", "api-client"]],
  ["request", 30, 45, false, "Shared JSON fetch: sets Accept and Content-Type (except FormData), throws on non-2xx, and returns parsed JSON or null for 204.", ["http", "api-client", "json"]],
  ["listSessions", 49, 51, true, "GET /sessions for a user id and returns the session list JSON.", ["api-client", "http", "sessions"]],
  ["createSession", 53, 55, true, "POST /sessions with a JSON body to create a chat session, forwarding an optional abort signal.", ["api-client", "http", "sessions"]],
  ["getSession", 57, 59, true, "GET /sessions/{sessionId} and returns that session's JSON.", ["api-client", "http", "sessions"]],
  ["updateSession", 61, 63, true, "PATCH /sessions/{sessionId} with a JSON body to update title, model, or other session fields.", ["api-client", "http", "sessions"]],
  ["deleteSession", 65, 67, true, "DELETE /sessions/{sessionId} to remove a chat session.", ["api-client", "http", "sessions"]],
  ["setFeedback", 69, 74, true, "POST /messages/{messageId}/feedback with a JSON value for thumbs feedback on an answer.", ["api-client", "http", "feedback"]],
  ["listDocuments", 78, 80, true, "GET /documents for a user id and returns the uploaded document list.", ["api-client", "http", "documents"]],
  ["uploadFile", 82, 86, true, "POST /upload as multipart FormData for one file, scoped by user id.", ["api-client", "http", "upload"]],
  ["uploadUrl", 88, 92, true, "POST /upload with a url query parameter so the backend ingests a remote page.", ["api-client", "http", "upload"]],
  ["deleteDocument", 94, 96, true, "DELETE /documents/{docId} for a user id to remove one ingested document.", ["api-client", "http", "documents"]],
  ["listGithubRepos", 98, 100, true, "GET /github/repos for a user id and returns connected repositories.", ["api-client", "http", "github"]],
  ["connectGithub", 102, 106, true, "POST /github/connect with a repository URL, optional token, and user id.", ["api-client", "http", "github"]],
  ["deleteGithubRepo", 108, 112, true, "DELETE /github/repos/{repoId} for a user id to disconnect a repository.", ["api-client", "http", "github"]],
  ["startGithubDevice", 114, 116, true, "POST /github/oauth/device to begin a GitHub device-code login.", ["api-client", "http", "oauth"]],
  ["pollGithubDevice", 118, 123, true, "POST /github/oauth/poll with the device code and user id until the login completes.", ["api-client", "http", "oauth"]],
  ["getHealth", 128, 130, true, "GET /health and returns read-only service status for the database, providers, and embeddings.", ["api-client", "http", "health"]],
  ["getConfig", 133, 135, true, "GET /config and returns advertised upload limits (max size and extensions).", ["api-client", "http", "configuration"]],
  ["getModels", 137, 139, true, "GET /models and returns the model catalog the composer can offer.", ["api-client", "http", "models"]],
  ["chatStream", 149, 194, true, "POST /chat/stream and reads the SSE body, dispatching meta, status, citations, model, token, artifact, error, and done events until the stream ends.", ["api-client", "sse", "streaming", "chat"]],
  ["dispatch", 164, 181, false, "Parses one LF-delimited SSE block into an event name and JSON data, then invokes the matching own-property handler.", ["sse", "parser", "streaming"]],
  ["downloadArtifact", 207, 221, true, "GET /messages/{messageId}/download and starts a browser download using the Content-Disposition filename.", ["api-client", "http", "download"]],
]);

addFns("frontend/js/app.js", [
  ["init", 36, 57, false, "Boots theme, mode, and user id, loads models, sessions, documents, and config in parallel, then binds artifact download and DOM events.", ["entry-point", "bootstrap", "state"]],
  ["_loadModels", 68, 91, false, "GET /models via API.getModels, keeps usable ids, persists the selection, and asks the UI to render the model select.", ["models", "state", "bootstrap"]],
  ["_bindEvents", 94, 196, false, "Wires send, stop, composer keys, mode and model changes, new chat, sidebar, file and URL upload, and the GitHub connect controls.", ["event-handler", "composer", "wiring"]],
  ["_startCooldown", 231, 242, false, "Starts a send cooldown from a Retry-After duration and announces when the user can send again.", ["rate-limit", "composer", "state"]],
  ["_requestErrorMessage", 263, 272, false, "Turns a failed request error into the toast text shown for a send that never started.", ["error-handling", "composer", "http"]],
  ["_onSend", 282, 559, false, "Runs one chat turn: ensures a session, POSTs /chat/stream through API.chatStream, and maps SSE events onto the Chat renderer until done, stop, or error.", ["streaming", "chat", "orchestrator", "sse"]],
  ["_retryAfterError", 570, 585, false, "Retries the last failed send after the server cooldown, using the regenerate path when the user turn was already stored.", ["error-handling", "streaming", "retry"]],
  ["_ensureSession", 593, 603, false, "Creates a session with API.createSession when the composer has no session id yet, and stores the returned id.", ["sessions", "state", "http"]],
  ["_newChat", 605, 618, false, "Clears the current transcript and session id so the next send starts a new conversation.", ["sessions", "chat", "state"]],
  ["_selectSession", 620, 639, false, "Loads one session with API.getSession and renders its history in the transcript.", ["sessions", "history", "http"]],
  ["_deleteSession", 641, 662, false, "Confirms and DELETE /sessions via API.deleteSession, then refreshes the sidebar list.", ["sessions", "http", "state"]],
  ["_renameSession", 665, 680, false, "PATCH /sessions with a new title via API.updateSession and refreshes the session row.", ["sessions", "http", "state"]],
  ["_loadSessions", 692, 704, false, "GET /sessions via API.listSessions and stores the sidebar list, marking sessions as loaded.", ["sessions", "http", "bootstrap"]],
  ["_loadConfig", 716, 731, false, "GET /config via API.getConfig and stores the upload size and extension limits used by the composer.", ["configuration", "upload", "http"]],
  ["_handleFiles", 734, 747, false, "Accepts the file input selection and starts an upload for each chosen file.", ["upload", "files", "event-handler"]],
  ["_handleFileUpload", 750, 782, false, "Builds an upload job that POSTs one file through API.uploadFile and runs it with the shared upload helper.", ["upload", "files", "http"]],
  ["_onUrlSubmit", 784, 800, false, "Reads the URL modal and POSTs the address through API.uploadUrl as an ingest job.", ["upload", "url", "http"]],
  ["_runUpload", 802, 828, false, "Runs one upload job, shows progress and success or the server error, then reloads the document list.", ["upload", "error-handling", "documents"]],
  ["_deleteDocument", 830, 846, false, "Confirms and DELETE /documents via API.deleteDocument for a single row, then reloads the list.", ["documents", "http", "delete"]],
  ["_deleteSelectedDocuments", 848, 868, false, "Deletes every selected document id with API.deleteDocument and then reloads the list.", ["documents", "http", "delete"]],
  ["_loadDocuments", 870, 883, false, "GET /documents via API.listDocuments and stores the rows the document view renders.", ["documents", "http", "bootstrap"]],
  ["_disconnectGithub", 904, 914, false, "DELETE /github/repos via API.deleteGithubRepo and refreshes the connected-repo list.", ["github", "http", "delete"]],
  ["_onGithubSubmit", 916, 938, false, "POSTs a repository URL and optional token through API.connectGithub, then reloads connected repos.", ["github", "http", "connect"]],
  ["_onGithubOauth", 948, 985, false, "Starts GitHub device login with API.startGithubDevice and polls API.pollGithubDevice until it completes or fails.", ["github", "oauth", "http"]],
  ["_setLoading", 990, 1002, false, "Toggles the send and stop buttons and keeps focus on the composer while a stream is in flight.", ["composer", "ui-state", "streaming"]],
]);

addFns("frontend/js/chat.js", [
  ["_bindScroll", 46, 63, false, "Tracks whether the reader is near the bottom of the transcript so streaming auto-scroll can stick or release.", ["scroll", "chat-ui", "streaming"]],
  ["_buildAssistant", 98, 139, false, "Builds the assistant message shell: avatar, activity steps, Markdown body, sources, and action row.", ["dom", "chat-ui", "message"]],
  ["_setNote", 166, 178, false, "Writes a status note on an assistant message, used for interruptions, partial answers, and artifact errors.", ["chat-ui", "status", "dom"]],
  ["appendUserMessage", 187, 194, true, "Appends a user bubble to the transcript, hides the welcome screen, and scrolls to the new message.", ["chat-ui", "message", "dom"]],
  ["startAssistantMessage", 202, 440, true, "Starts a live assistant message and returns the controller for thoughts, tokens, citations, badges, errors, and artifact cards.", ["streaming", "chat-ui", "sse", "markdown"]],
  ["copyMessage", 443, 452, true, "Copies the stored Markdown for a message to the clipboard and flashes the copy control.", ["clipboard", "markdown", "chat-ui"]],
  ["_resolveCitations", 460, 479, false, "Normalizes citation records from a stream or history payload into the shape the badges and source list use.", ["citations", "normalization", "chat-ui"]],
  ["getCitation", 495, 497, true, "Looks up one citation object by message id and citation number.", ["citations", "lookup", "chat-ui"]],
  ["showThinking", 500, 518, true, "Inserts a temporary Thinking placeholder with a processing AIViz mark until the live assistant message replaces it.", ["chat-ui", "loading", "placeholder"]],
  ["removeThinking", 520, 523, true, "Removes the Thinking placeholder element if it is still in the transcript.", ["chat-ui", "loading", "dom"]],
  ["renderHistory", 532, 567, true, "Rebuilds the transcript from stored messages, highlights code once, and binds citation badges.", ["history", "chat-ui", "markdown"]],
  ["_historyAssistant", 569, 600, false, "Renders one stored assistant message, including its Markdown, citations, and action row.", ["history", "chat-ui", "message"]],
  ["clearChat", 602, 608, true, "Empties the transcript and shows the welcome screen again.", ["chat-ui", "state", "dom"]],
  ["_badgesHtml", 612, 623, false, "Builds the model, cache, and fallback badges shown on an assistant message.", ["badges", "chat-ui", "models"]],
  ["_bindCitationBadges", 629, 649, false, "Makes inline citation markers open the matching source details.", ["citations", "event-handler", "chat-ui"]],
  ["_fillSources", 656, 726, false, "Renders the source list under an answer from the resolved citation records.", ["citations", "sources", "chat-ui"]],
  ["_flashCopy", 729, 738, false, "Temporarily changes a copy button label to show that the clipboard write succeeded or failed.", ["clipboard", "feedback", "chat-ui"]],
  ["_renderActionButtons", 741, 785, false, "Builds copy, feedback, and regenerate controls for an assistant message and wires their callbacks.", ["chat-ui", "feedback", "actions"]],
  ["holdLastAnswer", 814, 824, true, "Keeps the previous assistant answer on screen while a regenerate request is still being accepted.", ["streaming", "regenerate", "chat-ui"]],
  ["fileKeywordFormat", 841, 851, true, "Detects a requested download format from phrases in the user text, used before a file card is shown.", ["artifacts", "format", "parsing"]],
  ["prepareFilePreview", 853, 861, true, "Prepares the transcript for a file preview when the user message asks for a generated document.", ["artifacts", "preview", "chat-ui"]],
  ["_mountFileCard", 875, 910, false, "Inserts the file card for an artifact, including open and save actions, into the message slot.", ["artifacts", "file-card", "dom"]],
  ["_syncArtifactLayer", 932, 943, false, "Updates the artifact pane's dialog versus side-panel layer to match the viewport.", ["artifacts", "layout", "dialog"]],
  ["_paintArtifactPreview", 945, 954, false, "Fills the artifact pane body from the document payload and runs syntax highlighting.", ["artifacts", "preview", "markdown"]],
  ["openArtifact", 956, 979, true, "Opens the artifact pane for a generated document, closing the code preview so only one pane is open.", ["artifacts", "preview", "pane"]],
  ["closeArtifact", 981, 997, true, "Closes the artifact pane, restores inertness, and returns focus to the control that opened it.", ["artifacts", "pane", "focus"]],
  ["_onArtifactKey", 999, 1022, false, "Handles keyboard input while the artifact pane is open, including Escape to close.", ["artifacts", "keyboard", "accessibility"]],
  ["_applyArtifactWidth", 1045, 1059, false, "Applies a resized width to the artifact pane and optionally stores it for the next visit.", ["artifacts", "resize", "layout"]],
  ["_resetArtifactWidth", 1061, 1070, false, "Clears the stored artifact pane width and returns the pane to its default size.", ["artifacts", "resize", "layout"]],
  ["_wireArtifactResize", 1079, 1111, false, "Drags the artifact pane handle and clamps the width while the pointer is down.", ["artifacts", "resize", "pointer"]],
  ["_wireArtifactPane", 1113, 1132, false, "Binds the artifact pane's close, format, download, and resize controls once.", ["artifacts", "event-handler", "pane"]],
  ["bindArtifact", 1135, 1138, true, "Stores the download callback from app.js and wires the artifact pane controls.", ["artifacts", "wiring", "callback"]],
]);

addFns("frontend/js/components/ai-viz.js", [
  ["_hero", 33, 64, false, "Builds the large welcome-orb markup, a div scene rather than the small SVG mark.", ["visualization", "dom", "welcome"]],
  ["create", 67, 77, true, "Builds an AIViz node for a size and state: the hero scene, or an SVG mark with the static path markup.", ["visualization", "factory", "svg"]],
  ["setState", 86, 91, true, "Sets data-state on an AIViz node or a container that holds one, ignoring unknown state names.", ["visualization", "state", "dom"]],
  ["getState", 93, 96, true, "Reads the current data-state from an AIViz node or from a container that holds one.", ["visualization", "state", "dom"]],
  ["mount", 99, 105, true, "Replaces a container's children with a new AIViz instance and returns that node.", ["visualization", "dom", "mount"]],
]);

addFns("frontend/js/components/code-preview.js", [
  ["decorate", 42, 51, true, "Adds Run and Download controls to a highlighted code block after Utils.highlightCode wraps it.", ["code-preview", "dom", "code-block"]],
  ["_addButtons", 53, 86, false, "Creates the Run and Download buttons for one code block and wires their click handlers.", ["code-preview", "dom", "event-handler"]],
  ["_downloadBlob", 88, 98, false, "Starts a browser download of the code block source as a text file.", ["download", "code-preview", "blob"]],
  ["sync", 108, 110, true, "Closes the preview pane when the workspace leaves Code. Called from UI.setActiveMode.", ["code-preview", "workspace", "pane"]],
  ["_syncLayer", 114, 124, false, "Switches the open preview pane between a modal dialog on a narrow viewport and a side pane.", ["code-preview", "layout", "dialog"]],
  ["_applyW", 134, 146, false, "Applies a width to the code preview pane and can persist it for the next open.", ["code-preview", "resize", "layout"]],
  ["_resetW", 148, 157, false, "Clears the stored preview width and restores the pane's default size.", ["code-preview", "resize", "layout"]],
  ["_wireResize", 166, 198, false, "Drags the preview pane resize handle and clamps the width to the viewport.", ["code-preview", "resize", "pointer"]],
  ["_wirePane", 202, 217, false, "Binds close, scrim, and resize controls on the code preview pane once.", ["code-preview", "event-handler", "pane"]],
  ["_launch", 221, 253, true, "Opens the sandboxed iframe preview for HTML or SVG source. Exposed as CodePreview.open.", ["code-preview", "sandbox", "iframe"]],
  ["close", 257, 280, true, "Closes the preview pane, drops the iframe srcdoc so scripts stop, and returns focus to the opener.", ["code-preview", "pane", "focus"]],
]);

addFns("frontend/js/components/dialog.js", [
  ["_wire", 17, 41, false, "Attaches backdrop-click filtering and focus return on a dialog element, once per element.", ["dialog", "focus", "event-handler"]],
  ["open", 43, 58, true, "Shows a dialog modally, remembers the invoker, and moves focus to the requested element.", ["dialog", "focus", "modal"]],
  ["close", 60, 68, true, "Closes a dialog with the native close() or by removing the open attribute.", ["dialog", "modal", "focus"]],
  ["confirm", 71, 152, true, "Shows one confirm dialog and resolves true only when the confirm button is used. Cancel, Escape, and backdrop resolve false.", ["dialog", "confirm", "accessibility"]],
]);

addFns("frontend/js/components/documents.js", [
  ["_metaText", 39, 50, false, "Builds the secondary line for a document row from its type, size, and status fields.", ["documents", "formatting", "list"]],
  ["_updateBulkBar", 53, 67, false, "Shows or hides the bulk-delete bar from the current selection size.", ["documents", "selection", "dom"]],
  ["_row", 69, 112, false, "Builds one document row with its type icon, label, selection checkbox, and delete control.", ["documents", "dom", "list"]],
  ["_state", 114, 127, false, "Builds an empty, error, or other status block for the document list.", ["documents", "empty-state", "dom"]],
  ["_skeleton", 129, 141, false, "Builds the loading skeleton shown before the first document list arrives.", ["documents", "loading", "dom"]],
  ["_fill", 143, 165, false, "Fills a document container from the current rows, filter, and selection.", ["documents", "list", "render"]],
  ["_counts", 171, 181, false, "Updates the document count labels from the rows currently held by the view.", ["documents", "list", "dom"]],
  ["render", 188, 199, true, "Replaces the view's document rows and delete callbacks, then redraws the list. The controller passes server rows.", ["documents", "render", "view"]],
  ["clearSelection", 202, 206, true, "Clears the multi-select set and redraws the list and bulk bar.", ["documents", "selection", "view"]],
  ["setStatus", 208, 212, true, "Sets the list to loading, error, or ready and redraws, including an optional retry action.", ["documents", "status", "view"]],
  ["open", 214, 220, true, "Opens the My Documents dialog through Dialog.open, resetting the search when it was closed.", ["documents", "dialog", "view"]],
  ["init", 224, 266, true, "Binds search, close, upload, URL, select-all, and bulk-delete controls on the documents dialog.", ["documents", "event-handler", "wiring"]],
]);

addFns("frontend/js/components/menu.js", [
  ["isOpen", 17, 19, true, "Reports whether the popover is open, optionally only for a given anchor.", ["menu", "popover", "state"]],
  ["close", 21, 31, true, "Removes the open menu, marks the anchor collapsed, and can return focus to it.", ["menu", "popover", "focus"]],
  ["_position", 33, 50, false, "Places the menu next to its anchor, flipping alignment so it stays in the viewport.", ["menu", "positioning", "popover"]],
  ["open", 52, 152, true, "Opens an anchored menu from the item list, handles pointer and keyboard dismissal, and returns a close handle.", ["menu", "popover", "keyboard", "accessibility"]],
]);

addFns("frontend/js/components/model-select.js", [
  ["_item", 47, 57, false, "Reads one option into the listbox item shape: value, label, description, reason, disabled, and group.", ["model-select", "parsing", "listbox"]],
  ["_read", 59, 73, false, "Reads the native select's options and optgroups into the items the popover displays.", ["model-select", "parsing", "select"]],
  ["_syncChips", 94, 105, false, "Mirrors the current model label and logo onto the composer chips.", ["model-select", "dom", "chips"]],
  ["_syncTrigger", 107, 147, false, "Copies the selected option's label, logo, and disabled state onto the custom trigger.", ["model-select", "dom", "select"]],
  ["_position", 161, 173, false, "Positions the listbox popover under the trigger or the chip that opened it.", ["model-select", "positioning", "popover"]],
  ["_setActive", 175, 186, false, "Moves the roving active option in the open listbox and scrolls it into view.", ["model-select", "keyboard", "listbox"]],
  ["_typeahead", 198, 211, false, "Jumps the active option to the next label that starts with the typed character.", ["model-select", "keyboard", "typeahead"]],
  ["open", 225, 307, true, "Opens the model listbox, builds option and group rows from the native select, and handles typeahead and keys.", ["model-select", "listbox", "popover", "accessibility"]],
  ["close", 309, 321, true, "Closes the model listbox and can return focus to the trigger or chip.", ["model-select", "popover", "focus"]],
  ["_onListKey", 323, 343, false, "Handles arrow, Home, End, Enter, and typeahead keys while the listbox is open.", ["model-select", "keyboard", "listbox"]],
  ["init", 357, 398, true, "Enhances one native select: wraps value writes, observes option changes, and keeps the custom trigger in sync.", ["model-select", "init", "select"]],
  ["bindChip", 404, 414, true, "Connects a composer model chip so it shows the selection and opens the same listbox.", ["model-select", "chips", "event-handler"]],
  ["setError", 420, 426, true, "Shows a short failure reason and an optional Retry button when the model list could not load.", ["model-select", "error-handling", "ui-state"]],
  ["clearError", 428, 433, true, "Clears the model-list failure state and refreshes the trigger.", ["model-select", "error-handling", "ui-state"]],
  ["_ensureRetry", 435, 446, false, "Creates the Retry button used by the model-list failure state, once.", ["model-select", "dom", "retry"]],
]);

addFns("frontend/js/components/motion.js", [
  ["_load", 30, 44, false, "Resolves window.gsap if it is already present, otherwise injects /js/vendor/gsap.min.js and waits for it.", ["animation", "gsap", "loader"]],
  ["_groups", 49, 61, false, "Collects the welcome elements that the entrance timeline animates.", ["animation", "welcome", "dom"]],
  ["_play", 90, 120, false, "Runs the welcome entrance timeline and then the idle float, reverting any previous timeline first.", ["animation", "gsap", "welcome"]],
  ["enter", 123, 129, true, "Plays the first welcome entrance, or releases the pending class immediately when reduced motion is on.", ["animation", "welcome", "gsap"]],
  ["welcome", 132, 136, true, "Replays the welcome entrance when a fresh welcome screen is shown again.", ["animation", "welcome", "gsap"]],
  ["workspaceChanged", 139, 144, true, "Reverts the previous timeline and plays the entrance for the workspace panel that just became visible.", ["animation", "workspace", "gsap"]],
  ["_playIntro", 154, 183, false, "Runs the introduction-page entrance timeline on the intro hero and its sections.", ["animation", "intro", "gsap"]],
  ["intro", 185, 188, true, "Plays the introduction entrance once, and does nothing when reduced motion is requested.", ["animation", "intro", "gsap"]],
]);

addFns("frontend/js/components/notifications.js", [
  ["_render", 43, 64, false, "Redraws the panel body: an empty state, or the in-memory notices newest first.", ["notifications", "render", "dom"]],
  ["record", 76, 83, true, "Stores one notice from a toast or upload error, caps the list at 20, and updates the unread dot.", ["notifications", "state", "toast"]],
  ["clear", 85, 90, true, "Drops every stored notice and redraws the empty panel.", ["notifications", "state", "dom"]],
  ["_position", 95, 105, false, "Places the notification panel next to the bell so it stays inside the viewport.", ["notifications", "positioning", "popover"]],
  ["close", 107, 117, true, "Removes the notification panel and can return focus to the bell.", ["notifications", "popover", "focus"]],
  ["open", 119, 167, true, "Opens the bell panel, closing any Menu popover first, and marks the current notices read.", ["notifications", "popover", "menu"]],
]);

addFns("frontend/js/components/prefs.js", [
  ["get", 25, 30, true, "Returns the stored preference if it is one of the allowed values, otherwise the default. Unknown names return null.", ["preferences", "local-storage", "validation"]],
  ["apply", 33, 39, true, "Mirrors font size and density onto the html element, removing the attribute when the value is the default.", ["preferences", "dom", "theme"]],
  ["set", 41, 48, true, "Stores an allowed preference, applies it to the document, and notifies subscribers. Rejects unknown names and values.", ["preferences", "local-storage", "validation"]],
  ["onChange", 51, 57, true, "Subscribes to preference changes and returns an unsubscribe function.", ["preferences", "events", "subscription"]],
]);

// index.html loads these project assets.
const pageAssets = [
  "frontend/css/main.css",
  "frontend/css/animations.css",
  "frontend/css/components.css",
  "frontend/css/ai-viz.css",
  "frontend/css/reference.css",
  "frontend/css/workspace.css",
  "frontend/css/intro.css",
  "frontend/css/syntax.css",
  "frontend/assets/logo.svg",
  "frontend/js/icons.js",
  "frontend/js/utils.js",
  "frontend/js/api.js",
  "frontend/js/chat.js",
  "frontend/js/theme.js",
  "frontend/js/components/menu.js",
  "frontend/js/components/dialog.js",
  "frontend/js/components/model-select.js",
  "frontend/js/components/ai-viz.js",
  "frontend/js/components/notifications.js",
  "frontend/js/components/prefs.js",
  "frontend/js/components/radiogroup.js",
  "frontend/js/components/documents.js",
  "frontend/js/components/workspaces.js",
  "frontend/js/components/settings.js",
  "frontend/js/components/motion.js",
  "frontend/js/components/code-preview.js",
  "frontend/js/intro.js",
  "frontend/js/ui.js",
  "frontend/js/app.js",
];
for (const asset of pageAssets) {
  edge("file:frontend/index.html", `file:${asset}`, "depends_on", 0.6);
}

edge("document:frontend/assets/LICENSE-lucide.txt", "file:frontend/index.html", "documents", 0.5);
edge("file:frontend/js/api.js", "file:backend/main.py", "depends_on", 0.6);
edge("file:frontend/js/app.js", "file:backend/main.py", "depends_on", 0.6);
edge("file:frontend/js/app.js", "file:frontend/js/api.js", "depends_on", 0.6);
edge("file:frontend/js/app.js", "file:frontend/js/chat.js", "depends_on", 0.6);
edge("file:frontend/js/chat.js", "file:frontend/js/components/code-preview.js", "depends_on", 0.6);
edge("file:frontend/js/chat.js", "file:frontend/js/components/ai-viz.js", "depends_on", 0.6);
edge("file:frontend/js/components/documents.js", "file:frontend/js/components/dialog.js", "depends_on", 0.6);
edge("file:frontend/js/components/notifications.js", "file:frontend/js/components/menu.js", "depends_on", 0.6);
edge("file:frontend/js/components/motion.js", "file:frontend/js/vendor/gsap.min.js", "depends_on", 0.6);

const calls = [
  ["frontend/js/app.js", "init", "frontend/js/chat.js", "bindArtifact"],
  ["frontend/js/app.js", "_loadModels", "frontend/js/api.js", "getModels"],
  ["frontend/js/app.js", "_bindEvents", "frontend/js/api.js", "updateSession"],
  ["frontend/js/app.js", "_onSend", "frontend/js/api.js", "chatStream"],
  ["frontend/js/app.js", "_onSend", "frontend/js/chat.js", "appendUserMessage"],
  ["frontend/js/app.js", "_onSend", "frontend/js/chat.js", "startAssistantMessage"],
  ["frontend/js/app.js", "_onSend", "frontend/js/chat.js", "prepareFilePreview"],
  ["frontend/js/app.js", "_ensureSession", "frontend/js/api.js", "createSession"],
  ["frontend/js/app.js", "_selectSession", "frontend/js/api.js", "getSession"],
  ["frontend/js/app.js", "_deleteSession", "frontend/js/api.js", "deleteSession"],
  ["frontend/js/app.js", "_renameSession", "frontend/js/api.js", "updateSession"],
  ["frontend/js/app.js", "_loadSessions", "frontend/js/api.js", "listSessions"],
  ["frontend/js/app.js", "_loadConfig", "frontend/js/api.js", "getConfig"],
  ["frontend/js/app.js", "_handleFileUpload", "frontend/js/api.js", "uploadFile"],
  ["frontend/js/app.js", "_onUrlSubmit", "frontend/js/api.js", "uploadUrl"],
  ["frontend/js/app.js", "_deleteDocument", "frontend/js/api.js", "deleteDocument"],
  ["frontend/js/app.js", "_deleteSelectedDocuments", "frontend/js/api.js", "deleteDocument"],
  ["frontend/js/app.js", "_loadDocuments", "frontend/js/api.js", "listDocuments"],
  ["frontend/js/app.js", "_disconnectGithub", "frontend/js/api.js", "deleteGithubRepo"],
  ["frontend/js/app.js", "_onGithubSubmit", "frontend/js/api.js", "connectGithub"],
  ["frontend/js/app.js", "_onGithubOauth", "frontend/js/api.js", "startGithubDevice"],
  ["frontend/js/app.js", "_onGithubOauth", "frontend/js/api.js", "pollGithubDevice"],
  ["frontend/js/chat.js", "openArtifact", "frontend/js/components/code-preview.js", "close"],
  ["frontend/js/chat.js", "showThinking", "frontend/js/components/ai-viz.js", "mount"],
  ["frontend/js/components/documents.js", "open", "frontend/js/components/dialog.js", "open"],
  ["frontend/js/components/notifications.js", "open", "frontend/js/components/menu.js", "isOpen"],
  ["frontend/js/components/notifications.js", "open", "frontend/js/components/menu.js", "close"],
];
for (const [sf, sn, tf, tn] of calls) {
  edge(fid(sf, sn), fid(tf, tn), "calls", 0.8);
}

const ids = new Set(nodes.map((n) => n.id));
if (ids.size !== nodes.length) throw new Error("duplicate node ids");
const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

const external = new Set([
  "file:backend/main.py",
  "file:frontend/js/vendor/gsap.min.js",
  ...pageAssets.filter((p) => !ids.has(`file:${p}`)).map((p) => `file:${p}`),
]);

for (const e of edges) {
  if (!byId[e.source]) throw new Error("missing source " + e.source);
  const targetOk = byId[e.target] || external.has(e.target);
  if (!targetOk) throw new Error("missing target " + e.target);
  if (e.source === e.target) throw new Error("self " + e.source);
}

const filePaths = [...new Set(nodes.map((n) => n.filePath))].sort();
const nodeCount = nodes.length;
const edgeCount = edges.length;
const parts = Math.ceil(Math.max(nodeCount / 60, edgeCount / 120));
fs.mkdirSync(outDir, { recursive: true });

// Remove a single-file output if a previous run left one, so the merge does not double-count.
const single = path.join(outDir, "batch-11.json");
if (fs.existsSync(single)) fs.unlinkSync(single);

if (parts <= 1) {
  fs.writeFileSync(single, JSON.stringify({ nodes, edges }, null, 2));
  console.log(JSON.stringify({ files: ["batch-11.json"], nodeCount, edgeCount, filePaths: filePaths.length }, null, 2));
} else {
  const chunk = Math.ceil(filePaths.length / parts);
  const summary = [];
  for (let i = 0; i < parts; i++) {
    const slice = new Set(filePaths.slice(i * chunk, (i + 1) * chunk));
    const partNodes = nodes.filter((n) => slice.has(n.filePath));
    const partIds = new Set(partNodes.map((n) => n.id));
    const partEdges = edges.filter((e) => partIds.has(e.source));
    const name = `batch-11-part-${i + 1}.json`;
    fs.writeFileSync(path.join(outDir, name), JSON.stringify({ nodes: partNodes, edges: partEdges }, null, 2));
    summary.push({ name, nodes: partNodes.length, edges: partEdges.length, files: [...slice] });
  }
  console.log(JSON.stringify({ parts, nodeCount, edgeCount, summary }, null, 2));
}
