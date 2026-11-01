# -*- coding: utf-8 -*-
"""Build batch-4 knowledge graph fragment."""
import json
from pathlib import Path

nodes = []

def add_node(**kwargs):
    nodes.append(kwargs)

# --- files ---
add_node(
    id="file:backend/core/database.py",
    type="file",
    name="database.py",
    filePath="backend/core/database.py",
    summary="Owns the SQLAlchemy engine and session factory for chat persistence. Defines the declarative Base and a Database wrapper that creates tables, runs short transactions, and applies additive column migrations.",
    tags=["database", "sqlalchemy", "session-management", "persistence"],
    complexity="moderate",
    languageNotes="SQLite connections enable foreign keys, WAL, and a busy timeout; other engines use pool_pre_ping. init() lazily imports the ORM models so tables register before create_all.",
)
add_node(
    id="file:backend/models/db.py",
    type="file",
    name="db.py",
    filePath="backend/models/db.py",
    summary="SQLAlchemy ORM models for chat sessions, messages, uploaded documents, GitHub credentials, and connected repository briefs.",
    tags=["data-model", "sqlalchemy", "orm", "persistence"],
    complexity="moderate",
    languageNotes="Message rows cascade-delete with their session. Primary keys are UUID strings, and timestamps default to a naive UTC value.",
)
add_node(
    id="file:backend/services/document_service.py",
    type="file",
    name="document_service.py",
    filePath="backend/services/document_service.py",
    summary="Ingests files, URLs, and raw text by parsing, chunking, embedding, and writing a hybrid search index plus a document row. Also lists and deletes documents and re-embeds chunks when the embedding model changes.",
    tags=["service", "ingestion", "embedding", "rag"],
    complexity="complex",
)
add_node(
    id="file:backend/services/github_service.py",
    type="file",
    name="github_service.py",
    filePath="backend/services/github_service.py",
    summary="Connects a GitHub repository, indexes selected source files into the RAG store, and keeps a per-user token plus an LLM-written repository brief. Supports personal access tokens and the OAuth device flow.",
    tags=["service", "github", "oauth", "rag"],
    complexity="complex",
    languageNotes="Device-flow OAuth posts to github.com/login. File bodies are fetched from api.github.com and decoded from base64.",
)
add_node(
    id="file:backend/services/session_service.py",
    type="file",
    name="session_service.py",
    filePath="backend/services/session_service.py",
    summary="Creates, lists, updates, and deletes chat sessions and messages in short database transactions, returning pydantic schemas so callers never hold ORM objects.",
    tags=["service", "persistence", "chat", "session"],
    complexity="moderate",
)
add_node(
    id="file:backend/utils/timeutil.py",
    type="file",
    name="timeutil.py",
    filePath="backend/utils/timeutil.py",
    summary="Provides a naive UTC timestamp for database defaults and parses duration strings such as 2.28s, 1m26.4s, and 120ms into seconds.",
    tags=["utility", "datetime", "parsing"],
    complexity="simple",
)
add_node(
    id="file:tests/test_github.py",
    type="file",
    name="test_github.py",
    filePath="tests/test_github.py",
    summary="Tests GitHub URL parsing, source-path selection, repository connect and delete, OAuth device flow, code-mode chat that receives the repo brief, and MCP GitHub tool exposure.",
    tags=["test", "github", "oauth", "integration"],
    complexity="moderate",
)

# --- classes ---
classes = [
    ("backend/core/database.py", "Base", [21, 22], "simple",
     "Declarative SQLAlchemy base that ORM models inherit so their tables register on one metadata object.",
     ["data-model", "sqlalchemy", "orm"], True),
    ("backend/core/database.py", "Database", [25, 91], "moderate",
     "Creates the SQLAlchemy engine and session factory, initializes tables, migrates new nullable columns, and checks connectivity.",
     ["database", "sqlalchemy", "session-management"], True),
    ("backend/models/db.py", "ChatSession", [20, 35], "simple",
     "ORM model for a chat session, storing user, title, model, and mode, with messages that cascade-delete.",
     ["data-model", "sqlalchemy", "chat"], True),
    ("backend/models/db.py", "Message", [38, 50], "simple",
     "ORM model for one chat turn, including role, content, citations, feedback, and JSON metadata, keyed to a session.",
     ["data-model", "sqlalchemy", "chat"], True),
    ("backend/models/db.py", "Document", [53, 61], "simple",
     "ORM model for an ingested file or URL, recording user, filename, file type, chunk count, and index time.",
     ["data-model", "sqlalchemy", "document"], True),
    ("backend/models/db.py", "GithubCredential", [64, 70], "simple",
     "Per-user GitHub token row, distinguished as a personal access token or an OAuth token, and never returned to clients.",
     ["data-model", "sqlalchemy", "github", "security"], True),
    ("backend/models/db.py", "RepoContext", [73, 83], "simple",
     "ORM model for a connected repository brief, including owner, name, branch, summary text, and indexed file count.",
     ["data-model", "sqlalchemy", "github"], True),
    ("backend/services/document_service.py", "IngestResult", [35, 40], "simple",
     "Dataclass returned after ingestion, carrying the document id, filename, file type, chunk count, and whether indexing was hybrid or lexical.",
     ["data-model", "ingestion", "dto"], True),
    ("backend/services/document_service.py", "DocumentService", [43, 247], "complex",
     "Parses uploads, URLs, and text into chunks, embeds them when possible, writes the hybrid index and a document row, and reindexes after an embedding-model change.",
     ["service", "ingestion", "embedding", "rag"], True),
    ("backend/services/github_service.py", "GitHubError", [55, 56], "simple",
     "User-facing ingest error raised when a GitHub URL, token, or API response cannot be completed.",
     ["error-handling", "github", "validation"], True),
    ("backend/services/github_service.py", "GitHubService", [59, 293], "complex",
     "Fetches a repository tree, indexes selected source files, stores the user's token, and writes or returns a repository brief for the coding assistant.",
     ["service", "github", "oauth", "rag"], True),
    ("backend/services/session_service.py", "SessionService", [31, 156], "moderate",
     "Persists chat sessions and messages, auto-titles new chats from the first user message, and records assistant feedback.",
     ["service", "persistence", "chat", "session"], True),
]
for path, name, line_range, complexity, summary, tags, exported in classes:
    add_node(
        id=f"class:{path}:{name}",
        type="class",
        name=name,
        filePath=path,
        lineRange=line_range,
        summary=summary,
        tags=tags,
        complexity=complexity,
        _exported=exported,
    )

# --- functions ---
functions = [
    ("backend/core/database.py", "__init__", [26, 40], "simple",
     "Builds the SQLAlchemy engine for the given URL and attaches SQLite pragmas plus a non-expiring session factory.",
     ["database", "sqlalchemy", "initialization"], False),
    ("backend/core/database.py", "session", [52, 62], "simple",
     "Opens a short-lived SQLAlchemy session that commits on success, rolls back on error, and always closes.",
     ["database", "session-management", "transaction"], False),
    ("backend/core/database.py", "_ensure_columns", [71, 84], "simple",
     "Adds nullable columns that exist on the ORM metadata but are missing from an already created table.",
     ["database", "migration", "sqlalchemy"], False),
    ("backend/models/db.py", "_uuid", [16, 17], "simple",
     "Returns a new UUID string used as the default primary key for session, message, document, and repo rows.",
     ["utility", "identifier", "database"], True),
    ("backend/services/document_service.py", "__init__", [44, 55], "simple",
     "Stores the database, parser, embedder, and index, and configures the recursive text splitter from settings.",
     ["service", "initialization", "ingestion"], False),
    ("backend/services/document_service.py", "ingest_file", [59, 85], "moderate",
     "Validates an upload, parses it off the event loop, and hands the extracted pages to the shared indexing path.",
     ["ingestion", "upload", "parsing"], False),
    ("backend/services/document_service.py", "_chunk", [100, 113], "simple",
     "Splits page text into overlapping chunks and wraps each piece as a ChunkIn with page and order metadata.",
     ["ingestion", "chunking", "rag"], False),
    ("backend/services/document_service.py", "_index_pages", [115, 156], "moderate",
     "Chunks pages, embeds them, writes the hybrid index, records the document, and replaces any earlier upload with the same filename.",
     ["ingestion", "embedding", "indexing"], False),
    ("backend/services/document_service.py", "_embed", [158, 168], "simple",
     "Embeds chunk texts after warming the model, falling back to lexical-only indexing when embeddings fail.",
     ["embedding", "rag", "fallback"], False),
    ("backend/services/document_service.py", "_record_document", [170, 181], "simple",
     "Inserts the document row and deletes earlier rows for the same user and filename, returning those replaced ids.",
     ["persistence", "document", "database"], False),
    ("backend/services/document_service.py", "delete_document", [190, 203], "simple",
     "Deletes a user's document row and, when the row existed, removes its chunks from the search index.",
     ["document", "deletion", "indexing"], False),
    ("backend/services/document_service.py", "reindex", [213, 233], "moderate",
     "Re-embeds unembedded chunks for every user, and all chunks when the stored embedding model no longer matches.",
     ["embedding", "reindex", "rag"], False),
    ("backend/services/document_service.py", "reindex_if_needed", [235, 247], "simple",
     "Startup hook that reindexes when the embedding model changed or lexical-only chunks remain, and skips quietly on failure.",
     ["embedding", "reindex", "startup"], False),
    ("backend/services/github_service.py", "delete_repo", [91, 111], "moderate",
     "Deletes a connected repository row and removes the GitHub documents that were indexed under that owner and repo.",
     ["github", "deletion", "document"], False),
    ("backend/services/github_service.py", "connect", [113, 144], "moderate",
     "Parses a repository URL, fetches the default-branch tree, indexes selected files, and stores an LLM brief plus file count.",
     ["github", "ingestion", "rag"], False),
    ("backend/services/github_service.py", "start_device_flow", [146, 166], "simple",
     "Starts GitHub's OAuth device flow and returns the user code, verification URL, and device code.",
     ["github", "oauth", "authentication"], False),
    ("backend/services/github_service.py", "poll_device_flow", [168, 197], "moderate",
     "Polls GitHub for the device-flow access token and stores it as an OAuth credential when sign-in completes.",
     ["github", "oauth", "authentication"], False),
    ("backend/services/github_service.py", "_brief", [199, 224], "moderate",
     "Asks the LLM for a coding-assistant summary of the indexed tree and samples, or returns a layout fallback.",
     ["github", "llm", "summarization"], False),
    ("backend/services/github_service.py", "_file_text", [226, 236], "simple",
     "Fetches one repository file from the GitHub contents API and decodes base64 bodies to text.",
     ["github", "http-client", "decoding"], False),
    ("backend/services/github_service.py", "_get", [238, 257], "simple",
     "GETs the GitHub REST API with an optional bearer token and maps HTTP failures to GitHubError.",
     ["github", "http-client", "error-handling"], False),
    ("backend/services/github_service.py", "_upsert_context", [274, 293], "moderate",
     "Inserts or updates the user's repository brief, branch, and indexed file count, then detaches the row.",
     ["github", "persistence", "database"], False),
    ("backend/services/github_service.py", "parse_repo_url", [296, 302], "simple",
     "Extracts owner and repository name from an HTTPS or SSH GitHub URL and rejects anything else.",
     ["github", "parsing", "validation"], True),
    ("backend/services/github_service.py", "_select_paths", [305, 322], "simple",
     "Chooses source and manifest paths from a Git tree, skipping vendored directories, binaries, and oversized files.",
     ["github", "filtering", "ingestion"], True),
    ("backend/services/session_service.py", "update_session", [69, 81], "simple",
     "Updates a session title, model, or mode and returns the reloaded session schema, or raises when the id is missing.",
     ["session", "persistence", "update"], False),
    ("backend/services/session_service.py", "add_message", [92, 111], "moderate",
     "Appends a message, bumps the session timestamp, and titles a new chat from the first few words of a user message.",
     ["chat", "persistence", "message"], False),
    ("backend/services/session_service.py", "history", [113, 123], "simple",
     "Returns the most recent user and assistant messages for a session, oldest first, optionally dropping the latest user turn.",
     ["chat", "history", "persistence"], False),
    ("backend/services/session_service.py", "pop_last_exchange", [125, 135], "simple",
     "Deletes the trailing assistant reply so a turn can be regenerated, and returns the last user text.",
     ["chat", "regenerate", "persistence"], False),
    ("backend/services/session_service.py", "get_owned_message", [137, 146], "simple",
     "Loads an assistant message only when it belongs to the given user, for rebuilding a download.",
     ["chat", "authorization", "message"], False),
    ("backend/utils/timeutil.py", "utcnow", [13, 15], "simple",
     "Returns the current UTC time with tzinfo removed so new rows match timestamps stored before timezone support.",
     ["utility", "datetime", "database"], True),
    ("backend/utils/timeutil.py", "parse_duration", [18, 33], "simple",
     "Parses a bare number of seconds or a combined hours, minutes, seconds, and milliseconds string into a float.",
     ["utility", "parsing", "datetime"], True),
    ("tests/test_github.py", "github_transport", [18, 56], "moderate",
     "Mock HTTP transport that fakes GitHub device-flow, repository, tree, and base64 file responses for the tests.",
     ["test", "github", "http-mock"], True),
    ("tests/test_github.py", "test_parse_repo_url_and_path_selection", [59, 74], "simple",
     "Checks owner/repo parsing, rejection of non-GitHub URLs, and that path selection keeps source files while dropping vendored and binary paths.",
     ["test", "github", "parsing"], True),
    ("tests/test_github.py", "test_connect_indexes_repo_and_hides_the_token", [77, 98], "moderate",
     "Connects a sample repository, asserts indexed GitHub documents, and checks that the token is absent from API responses before delete.",
     ["test", "github", "security"], True),
    ("tests/test_github.py", "test_connect_rejects_unknown_and_non_github_urls", [101, 105], "simple",
     "Expects HTTP 400 for a non-GitHub URL and HTTP 404 when the repository is missing.",
     ["test", "github", "validation"], True),
    ("tests/test_github.py", "test_oauth_device_flow", [108, 117], "simple",
     "Starts the device flow with a configured client id and polls until the mocked token exchange reports connected.",
     ["test", "oauth", "github"], True),
    ("tests/test_github.py", "test_oauth_pending_and_unconfigured", [120, 127], "simple",
     "Expects HTTP 501 when OAuth is unconfigured and a pending status while GitHub has not issued a token yet.",
     ["test", "oauth", "github"], True),
    ("tests/test_github.py", "test_code_chat_gets_the_repo_brief", [130, 145], "moderate",
     "Connects a repository, streams a code-mode chat, and checks that the system prompt includes the repository brief and source path.",
     ["test", "github", "chat"], True),
    ("tests/test_github.py", "test_mcp_exposes_github_tools", [148, 153], "simple",
     "Asserts the MCP server exposes callable wrappers for connecting a repo, listing repos, reading context, and searching documents.",
     ["test", "mcp", "github"], True),
]
for path, name, line_range, complexity, summary, tags, exported in functions:
    add_node(
        id=f"function:{path}:{name}",
        type="function",
        name=name,
        filePath=path,
        lineRange=line_range,
        summary=summary,
        tags=tags,
        complexity=complexity,
        _exported=exported,
    )

tables = [
    ("sessions", [20, 35], "simple",
     "Chat sessions keyed by user, with title, model, mode, and created/updated timestamps. Deleting a session cascades to its messages.",
     ["database", "chat", "sqlalchemy"]),
    ("messages", [38, 50], "simple",
     "Chat turns belonging to a session, storing role, content, citations, thumbs feedback, and JSON metadata.",
     ["database", "chat", "sqlalchemy"]),
    ("documents", [53, 61], "simple",
     "Ingested uploads, URLs, and GitHub files per user, with file type, chunk count, and index time.",
     ["database", "document", "sqlalchemy"]),
    ("github_credentials", [64, 70], "simple",
     "One GitHub token per user, marked as a personal access token or an OAuth token, with an updated timestamp.",
     ["database", "github", "security"]),
    ("repo_contexts", [73, 83], "simple",
     "Connected repositories for a user, including owner, name, branch, generated brief, and indexed file count.",
     ["database", "github", "sqlalchemy"]),
]
for name, line_range, complexity, summary, tags in tables:
    add_node(
        id=f"table:backend/models/db.py:{name}",
        type="table",
        name=name,
        filePath="backend/models/db.py",
        lineRange=line_range,
        summary=summary,
        tags=tags,
        complexity=complexity,
        _exported=False,
    )

imports = {
    "backend/core/database.py": [],
    "backend/models/db.py": ["backend/core/database.py", "backend/utils/timeutil.py"],
    "backend/services/document_service.py": [
        "backend/core/config.py", "backend/core/database.py", "backend/core/exceptions.py",
        "backend/models/db.py", "backend/services/embedding_service.py",
        "backend/services/index_store.py", "backend/services/parser.py",
    ],
    "backend/services/github_service.py": [
        "backend/core/config.py", "backend/core/database.py", "backend/core/exceptions.py",
        "backend/models/db.py", "backend/services/document_service.py",
        "backend/services/llm_service.py", "backend/utils/timeutil.py",
    ],
    "backend/services/session_service.py": [
        "backend/core/config.py", "backend/core/database.py", "backend/core/exceptions.py",
        "backend/models/api.py", "backend/models/db.py", "backend/utils/timeutil.py",
    ],
    "backend/utils/timeutil.py": [],
    "tests/test_github.py": ["backend/services/github_service.py", "tests/conftest.py"],
}

edges = []

def edge(source, target, etype, weight):
    edges.append({
        "source": source,
        "target": target,
        "type": etype,
        "direction": "forward",
        "weight": weight,
    })

for n in nodes:
    fp = n.get("filePath")
    if n["type"] in ("function", "class", "table"):
        edge(f"file:{fp}", n["id"], "contains", 1.0)
    if n.get("_exported"):
        edge(f"file:{fp}", n["id"], "exports", 0.8)

for src, targets in imports.items():
    for t in targets:
        edge(f"file:{src}", f"file:{t}", "imports", 0.7)

for model in ("ChatSession", "Message", "Document", "GithubCredential", "RepoContext"):
    edge(f"class:backend/models/db.py:{model}", "class:backend/core/database.py:Base", "inherits", 0.9)
edge("class:backend/services/github_service.py:GitHubError", "class:backend/core/exceptions.py:IngestError", "inherits", 0.9)

edge("file:backend/core/database.py", "file:backend/models/db.py", "depends_on", 0.6)
edge("file:backend/services/github_service.py", "file:tests/test_github.py", "tested_by", 0.5)

# calls — document service
D = "function:backend/services/document_service.py"
edge(f"{D}:ingest_file", f"{D}:_index_pages", "calls", 0.8)
edge(f"{D}:ingest_file", "class:backend/core/exceptions.py:IngestError", "calls", 0.8)
edge(f"{D}:ingest_file", "class:backend/services/parser.py:DocumentParser", "calls", 0.8)
edge(f"{D}:_chunk", "class:backend/services/index_store.py:ChunkIn", "calls", 0.8)
edge(f"{D}:_index_pages", f"{D}:_chunk", "calls", 0.8)
edge(f"{D}:_index_pages", f"{D}:_embed", "calls", 0.8)
edge(f"{D}:_index_pages", f"{D}:_record_document", "calls", 0.8)
edge(f"{D}:_index_pages", "class:backend/core/exceptions.py:IngestError", "calls", 0.8)
edge(f"{D}:_index_pages", "class:backend/services/index_store.py:HybridIndexStore", "calls", 0.8)
edge(f"{D}:_index_pages", "class:backend/services/document_service.py:IngestResult", "calls", 0.8)
edge(f"{D}:_embed", "class:backend/services/embedding_service.py:EmbeddingService", "calls", 0.8)
edge(f"{D}:_record_document", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{D}:delete_document", "class:backend/services/index_store.py:HybridIndexStore", "calls", 0.8)
edge(f"{D}:reindex", "class:backend/services/embedding_service.py:EmbeddingService", "calls", 0.8)
edge(f"{D}:reindex", "class:backend/services/index_store.py:HybridIndexStore", "calls", 0.8)
edge(f"{D}:reindex_if_needed", f"{D}:reindex", "calls", 0.8)
edge(f"{D}:reindex_if_needed", "class:backend/services/embedding_service.py:EmbeddingService", "calls", 0.8)

# calls — github service
G = "function:backend/services/github_service.py"
edge(f"{G}:connect", f"{G}:parse_repo_url", "calls", 0.8)
edge(f"{G}:connect", f"{G}:_select_paths", "calls", 0.8)
edge(f"{G}:connect", f"{G}:_file_text", "calls", 0.8)
edge(f"{G}:connect", f"{G}:_get", "calls", 0.8)
edge(f"{G}:connect", f"{G}:_brief", "calls", 0.8)
edge(f"{G}:connect", f"{G}:_upsert_context", "calls", 0.8)
edge(f"{G}:connect", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:connect", "class:backend/services/document_service.py:DocumentService", "calls", 0.8)
edge(f"{G}:delete_repo", f"{D}:delete_document", "calls", 0.8)
edge(f"{G}:delete_repo", "class:backend/services/document_service.py:DocumentService", "calls", 0.8)
edge(f"{G}:_file_text", f"{G}:_get", "calls", 0.8)
edge(f"{G}:_file_text", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:_get", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:_brief", "class:backend/services/llm_service.py:LLMService", "calls", 0.8)
edge(f"{G}:parse_repo_url", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:start_device_flow", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:poll_device_flow", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{G}:_upsert_context", "function:backend/utils/timeutil.py:utcnow", "calls", 0.8)
edge(f"{G}:_upsert_context", "function:backend/core/database.py:session", "calls", 0.8)

# calls — session service
S = "function:backend/services/session_service.py"
edge(f"{S}:update_session", "class:backend/core/exceptions.py:SessionNotFound", "calls", 0.8)
edge(f"{S}:update_session", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{S}:add_message", "class:backend/core/exceptions.py:SessionNotFound", "calls", 0.8)
edge(f"{S}:add_message", "function:backend/utils/timeutil.py:utcnow", "calls", 0.8)
edge(f"{S}:add_message", "class:backend/models/api.py:MessageSchema", "calls", 0.8)
edge(f"{S}:add_message", "class:backend/models/api.py:CitationSchema", "calls", 0.8)
edge(f"{S}:add_message", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{S}:history", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{S}:pop_last_exchange", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{S}:get_owned_message", "function:backend/core/database.py:session", "calls", 0.8)
edge(f"{S}:get_owned_message", "class:backend/models/api.py:MessageSchema", "calls", 0.8)

# calls — tests
T = "function:tests/test_github.py"
edge(f"{T}:test_parse_repo_url_and_path_selection", f"{G}:parse_repo_url", "calls", 0.8)
edge(f"{T}:test_parse_repo_url_and_path_selection", f"{G}:_select_paths", "calls", 0.8)
edge(f"{T}:test_parse_repo_url_and_path_selection", "class:backend/services/github_service.py:GitHubError", "calls", 0.8)
edge(f"{T}:test_connect_indexes_repo_and_hides_the_token", f"{T}:github_transport", "calls", 0.8)
edge(f"{T}:test_connect_indexes_repo_and_hides_the_token", "class:tests/conftest.py:FakeProviderClient", "calls", 0.8)
edge(f"{T}:test_connect_indexes_repo_and_hides_the_token", "function:tests/conftest.py:client_factory", "calls", 0.8)
edge(f"{T}:test_connect_rejects_unknown_and_non_github_urls", f"{T}:github_transport", "calls", 0.8)
edge(f"{T}:test_connect_rejects_unknown_and_non_github_urls", "function:tests/conftest.py:client_factory", "calls", 0.8)
edge(f"{T}:test_oauth_device_flow", f"{T}:github_transport", "calls", 0.8)
edge(f"{T}:test_oauth_device_flow", "function:tests/conftest.py:client_factory", "calls", 0.8)
edge(f"{T}:test_oauth_pending_and_unconfigured", f"{T}:github_transport", "calls", 0.8)
edge(f"{T}:test_oauth_pending_and_unconfigured", "function:tests/conftest.py:client_factory", "calls", 0.8)
edge(f"{T}:test_code_chat_gets_the_repo_brief", f"{T}:github_transport", "calls", 0.8)
edge(f"{T}:test_code_chat_gets_the_repo_brief", "class:tests/conftest.py:FakeProviderClient", "calls", 0.8)
edge(f"{T}:test_code_chat_gets_the_repo_brief", "function:tests/conftest.py:parse_sse", "calls", 0.8)
edge(f"{T}:test_code_chat_gets_the_repo_brief", "function:tests/conftest.py:client_factory", "calls", 0.8)

for n in nodes:
    n.pop("_exported", None)

out = {"nodes": nodes, "edges": edges}
dest = Path(r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua\intermediate\batch-4.json")
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"nodes={len(nodes)} edges={len(edges)}")
print("imports", sum(1 for e in edges if e["type"] == "imports"))
print("contains", sum(1 for e in edges if e["type"] == "contains"))
print("exports", sum(1 for e in edges if e["type"] == "exports"))
