import json
from pathlib import Path

out_path = Path(r"F:\Personal Projects\AgenticAI_Chatbot_Using_RAG\.ua\intermediate\batch-3.json")

def N(id, type, name, summary, tags, complexity, filePath=None, lineRange=None, languageNotes=None):
    node = {
        "id": id,
        "type": type,
        "name": name,
        "summary": summary,
        "tags": tags,
        "complexity": complexity,
    }
    if filePath is not None:
        node["filePath"] = filePath
    if lineRange is not None:
        node["lineRange"] = lineRange
    if languageNotes:
        node["languageNotes"] = languageNotes
    return node

def E(source, target, type, weight):
    return {
        "source": source,
        "target": target,
        "type": type,
        "direction": "forward",
        "weight": weight,
    }

nodes = []

# --- files ---
nodes.append(N(
    "file:backend/client/llm_client.py", "file", "llm_client.py",
    "OpenAI-compatible transport for Groq and Google chat models: opens streamed completions, lists model ids, and maps SDK failures to normalised LLMError codes.",
    ["service", "llm", "streaming", "error-handling"],
    "moderate",
    "backend/client/llm_client.py",
    languageNotes="Uses the OpenAI SDK raw response so rate-limit headers stay available to the quota guard.",
))
nodes.append(N(
    "file:backend/core/exceptions.py", "file", "exceptions.py",
    "Shared application exception hierarchy whose messages are safe to show to clients and never echo upstream provider text.",
    ["error-handling", "data-model", "validation"],
    "simple",
    "backend/core/exceptions.py",
    languageNotes="Every user-facing failure subclasses AppError; LLMError carries a machine-readable code plus a fixed safe message.",
))
nodes.append(N(
    "file:backend/services/llm_service.py", "file", "llm_service.py",
    "LLM policy layer that selects a model chain, fails over before the first token, and applies a circuit breaker plus quota guard around provider clients.",
    ["service", "llm", "failover", "streaming"],
    "complex",
    "backend/services/llm_service.py",
    languageNotes="Failover runs only before the first visible token; a second pass ignores soft cooldowns when every model was skipped.",
))
nodes.append(N(
    "file:backend/utils/rate_limiter.py", "file", "rate_limiter.py",
    "Per-user token-bucket throttling and a per-model rolling quota guard that lets the LLM layer skip exhausted free-tier models before they return 429.",
    ["utility", "rate-limiting", "quota"],
    "moderate",
    "backend/utils/rate_limiter.py",
))
nodes.append(N(
    "file:tests/conftest.py", "file", "conftest.py",
    "Offline pytest fixtures: deterministic embedder and reranker fakes, a scripted LLM provider, and a ServiceContainer factory that builds the FastAPI test client.",
    ["test", "fixture", "factory"],
    "complex",
    "tests/conftest.py",
))
nodes.append(N(
    "file:tests/test_artifacts.py", "file", "test_artifacts.py",
    "Exercises document validation, markdown-to-file rendering, and the chat SSE path that emits an artifact and serves every download format.",
    ["test", "artifact", "validation", "api-handler"],
    "complex",
    "tests/test_artifacts.py",
))
nodes.append(N(
    "file:tests/test_llm.py", "file", "test_llm.py",
    "Covers LLMService failover, circuit-breaker trips, quota-guard skips, think-block filtering, and classify_exception status mapping.",
    ["test", "llm", "failover", "streaming"],
    "moderate",
    "tests/test_llm.py",
))
nodes.append(N(
    "file:tests/test_retrieval.py", "file", "test_retrieval.py",
    "Offline checks for hybrid ingest and search, owner isolation, embedder and reranker degradation, and atomic index rollback.",
    ["test", "retrieval", "indexing"],
    "moderate",
    "tests/test_retrieval.py",
))

# --- llm_client symbols ---
nodes.append(N(
    "class:backend/client/llm_client.py:OpenedStream", "class", "OpenedStream",
    "Dataclass holding response headers, the SDK stream, and its async iterator after a completion has started.",
    ["data-model", "streaming", "llm"],
    "simple",
    "backend/client/llm_client.py", [25, 30],
))
nodes.append(N(
    "class:backend/client/llm_client.py:ProviderClient", "class", "ProviderClient",
    "Wraps one OpenAI-compatible provider endpoint and opens streamed chat completions or lists model ids under a timeout.",
    ["service", "llm", "streaming"],
    "simple",
    "backend/client/llm_client.py", [33, 57],
))
nodes.append(N(
    "function:backend/client/llm_client.py:build_provider_clients", "function", "build_provider_clients",
    "Builds one ProviderClient per provider whose API key is configured, sharing a pooled HTTP client and disabling SDK retries.",
    ["factory", "llm", "configuration"],
    "simple",
    "backend/client/llm_client.py", [60, 66],
))
nodes.append(N(
    "function:backend/client/llm_client.py:wrap_clients", "function", "wrap_clients",
    "Normalises a map of SDK clients or existing ProviderClients so tests can inject doubles beside real clients.",
    ["factory", "llm", "test"],
    "simple",
    "backend/client/llm_client.py", [69, 71],
))
nodes.append(N(
    "function:backend/client/llm_client.py:chunk_text", "function", "chunk_text",
    "Returns the visible content delta of a chat chunk and ignores reasoning-only deltas.",
    ["streaming", "llm", "utility"],
    "simple",
    "backend/client/llm_client.py", [76, 80],
))
nodes.append(N(
    "function:backend/client/llm_client.py:chunk_usage", "function", "chunk_usage",
    "Extracts prompt, completion, and total token counts from a stream chunk when the provider includes usage.",
    ["streaming", "llm", "utility"],
    "simple",
    "backend/client/llm_client.py", [83, 91],
))
nodes.append(N(
    "function:backend/client/llm_client.py:close_stream", "function", "close_stream",
    "Closes an SDK stream best-effort, awaiting the closer when it returns a coroutine and swallowing shutdown errors.",
    ["streaming", "llm", "utility"],
    "simple",
    "backend/client/llm_client.py", [94, 104],
))
nodes.append(N(
    "function:backend/client/llm_client.py:classify_exception", "function", "classify_exception",
    "Maps OpenAI SDK and HTTP failures to an LLMError code, including retry-after hints parsed from rate-limit headers.",
    ["error-handling", "llm", "validation"],
    "moderate",
    "backend/client/llm_client.py", [109, 134],
))

# --- exceptions ---
exc = "backend/core/exceptions.py"
nodes.append(N("class:backend/core/exceptions.py:AppError", "class", "AppError",
    "Base exception for expected failures whose message is safe to present to the user.",
    ["error-handling", "data-model", "validation"], "simple", exc, [12, 13]))
nodes.append(N("class:backend/core/exceptions.py:DocumentParseError", "class", "DocumentParseError",
    "Raised when a document is unreadable, unsupported, or rejected as unsafe input.",
    ["error-handling", "document", "validation"], "simple", exc, [18, 19]))
nodes.append(N("class:backend/core/exceptions.py:FetchError", "class", "FetchError",
    "Raised when a web page cannot be fetched because of a bad URL, a blocked address, an HTTP error, or size limits.",
    ["error-handling", "document", "validation"], "simple", exc, [22, 23]))
nodes.append(N("class:backend/core/exceptions.py:IngestError", "class", "IngestError",
    "Ingestion failure that carries a user-safe message and an HTTP status hint, defaulting to 422.",
    ["error-handling", "document", "validation"], "simple", exc, [26, 32]))
nodes.append(N("class:backend/core/exceptions.py:SessionNotFound", "class", "SessionNotFound",
    "Raised when a session or message id does not exist.",
    ["error-handling", "session", "validation"], "simple", exc, [37, 38]))
nodes.append(N("class:backend/core/exceptions.py:EmbeddingUnavailable", "class", "EmbeddingUnavailable",
    "Raised when no embedding backend can serve the request.",
    ["error-handling", "embedding", "retrieval"], "simple", exc, [41, 42]))
nodes.append(N("class:backend/core/exceptions.py:LLMError", "class", "LLMError",
    "Normalised LLM failure with a machine-readable code, optional model and retry metadata, and a fixed user-safe message.",
    ["error-handling", "llm", "data-model"], "simple", exc, [63, 76]))

# --- llm_service ---
svc = "backend/services/llm_service.py"
nodes.append(N("class:backend/services/llm_service.py:LLMResult", "class", "LLMResult",
    "Completed generation: joined text, chosen model, fallback list, time-to-first-token, and optional usage.",
    ["data-model", "llm", "streaming"], "simple", svc, [43, 48]))
nodes.append(N("class:backend/services/llm_service.py:LLMStream", "class", "LLMStream",
    "Async iterator of visible text deltas that strips think blocks, records usage, and closes the upstream stream when finished.",
    ["streaming", "llm", "service"], "moderate", svc, [51, 103]))
nodes.append(N("class:backend/services/llm_service.py:CircuitBreaker", "class", "CircuitBreaker",
    "Per-model cooldown tracker with soft trips that failover may override and hard trips that hide a model until reset.",
    ["service", "llm", "failover"], "moderate", svc, [108, 141]))
nodes.append(N("class:backend/services/llm_service.py:LLMService", "class", "LLMService",
    "Owns the model registry, discovery, and failover chain, opening a stream on the first model that yields visible text.",
    ["service", "llm", "failover", "streaming"], "complex", svc, [146, 399]))

# --- rate limiter ---
rl = "backend/utils/rate_limiter.py"
nodes.append(N("class:backend/utils/rate_limiter.py:RateLimiter", "class", "RateLimiter",
    "Thread-safe token bucket keyed by caller that returns whether a request is allowed and how long to wait.",
    ["rate-limiting", "utility", "middleware"], "moderate", rl, [18, 49]))
nodes.append(N("class:backend/utils/rate_limiter.py:QuotaGuard", "class", "QuotaGuard",
    "Rolling 60-second request and token accounting per model, updated from local usage and provider rate-limit headers.",
    ["rate-limiting", "llm", "quota"], "moderate", rl, [52, 107]))

# --- conftest ---
cf = "tests/conftest.py"
nodes.append(N("class:tests/conftest.py:FakeEmbedder", "class", "FakeEmbedder",
    "Deterministic hashing bag-of-words embedder with a synonym table, used so retrieval tests run without a model download.",
    ["test", "embedding", "fixture"], "moderate", cf, [43, 93]))
nodes.append(N("class:tests/conftest.py:FakeReranker", "class", "FakeReranker",
    "Deterministic reranker that scores passages by token overlap and can be switched into a failure mode.",
    ["test", "retrieval", "fixture"], "simple", cf, [96, 127]))
nodes.append(N("class:tests/conftest.py:_FakeStream", "class", "_FakeStream",
    "Async iterator of scripted chat chunks that can raise a canned API error on a chosen step.",
    ["test", "streaming", "llm"], "simple", cf, [150, 175]))
nodes.append(N("class:tests/conftest.py:FakeProviderClient", "class", "FakeProviderClient",
    "Scripted OpenAI-shaped provider that records calls and returns text, errors, or hangs according to a behaviour list.",
    ["test", "llm", "fixture"], "moderate", cf, [178, 233]))
nodes.append(N("function:tests/conftest.py:settings", "function", "settings",
    "Pytest fixture that builds Settings with isolated temp paths and a cleared provider environment.",
    ["test", "fixture", "configuration"], "simple", cf, [239, 249]))
nodes.append(N("function:tests/conftest.py:tavily_transport", "function", "tavily_transport",
    "Builds an httpx mock transport that replays Tavily and Serper search responses for offline web-search tests.",
    ["test", "fixture", "search"], "simple", cf, [252, 272]))
nodes.append(N("function:tests/conftest.py:build_container", "function", "build_container",
    "Constructs a ServiceContainer with fake embeddings, reranking, and LLM clients, and disposes it after the test.",
    ["test", "factory", "fixture"], "moderate", cf, [281, 306]))
nodes.append(N("function:tests/conftest.py:client_factory", "function", "client_factory",
    "Returns a callable that mounts the FastAPI app on a TestClient wired to a fresh offline container.",
    ["test", "factory", "api-handler"], "simple", cf, [310, 324]))
nodes.append(N("function:tests/conftest.py:parse_sse", "function", "parse_sse",
    "Parses a Server-Sent Events payload into a list of event names and JSON data objects.",
    ["test", "utility", "streaming"], "simple", cf, [327, 339]))

# --- artifact tests (10+ lines) ---
ta = "tests/test_artifacts.py"
art_tests = [
    ("test_validator_rejects_empty_and_over_cap_documents", [75, 91],
     "Asserts validate_document rejects empty documents and documents that exceed the block cap.",
     ["test", "validation", "artifact"]),
    ("test_when_a_document_is_and_is_not_authored", [94, 121],
     "Checks the predicates that decide whether a user turn should author, restyle, or save a file and which format they want.",
     ["test", "artifact", "validation"]),
    ("test_five_formats_are_built_from_the_same_document", [134, 162],
     "Renders one validated document to markdown, text, xlsx, docx, and pdf and checks filenames and content types.",
     ["test", "artifact", "serialization"]),
    ("test_awkward_answers_still_become_every_format", [185, 202],
     "Turns awkward markdown into a document and confirms every download format still builds.",
     ["test", "artifact", "serialization"]),
    ("test_a_prose_answer_becomes_a_titled_sectioned_document", [216, 236],
     "Converts a prose answer into a titled, sectioned document and checks the downloadable markdown.",
     ["test", "artifact", "serialization"]),
    ("test_generate_pdf_saves_the_previous_answer", [239, 260],
     "Streams a chat turn then downloads a PDF of the previous answer through the session API.",
     ["test", "artifact", "api-handler"]),
    ("test_markdown_becomes_a_document_without_a_second_model_call", [263, 284],
     "Shows document_source reuses existing markdown or the previous answer instead of calling the model again.",
     ["test", "artifact", "llm"]),
    ("test_file_request_streams_the_answer_then_the_artifact", [287, 318],
     "Asserts a file request streams answer tokens, then an artifact event, and that the saved file downloads.",
     ["test", "artifact", "streaming"]),
    ("test_save_request_files_the_previous_answer_not_export_instructions", [321, 336],
     "A bare save request files the previous answer rather than the export instructions themselves.",
     ["test", "artifact", "api-handler"]),
    ("test_i_want_a_pdf_opens_a_file_even_when_the_model_refuses", [339, 355],
     "A PDF request still produces a download when the model replies with a refusal.",
     ["test", "artifact", "api-handler"]),
    ("test_every_mode_downloads_every_format", [359, 387],
     "Parametrized check that each chat mode can download markdown, text, spreadsheet, docx, and pdf.",
     ["test", "artifact", "api-handler"]),
    ("test_thin_answer_still_downloads_instead_of_failing", [390, 400],
     "A very short answer still produces a downloadable file instead of an error.",
     ["test", "artifact", "validation"]),
    ("test_download_rate_limit_is_separate_from_chat", [412, 425],
     "Shows the download limiter can return 429 while a following chat post is still accepted.",
     ["test", "rate-limiting", "artifact"]),
    ("test_inline_markdown_is_stripped_in_non_markdown_formats", [428, 450],
     "Inline markdown markers are removed from text, spreadsheet, docx, and pdf renders.",
     ["test", "artifact", "serialization"]),
]
for name, span, summary, tags in art_tests:
    nodes.append(N(f"function:{ta}:{name}", "function", name, summary, tags, "simple", ta, span))

# --- llm tests (10+ lines) ---
tl = "tests/test_llm.py"
llm_tests = [
    ("test_failover_on_rate_limit_and_5xx_and_timeout", [48, 62],
     "A rate limit, a 5xx, and a timeout each move the stream to the next model before any token is yielded.",
     ["test", "failover", "llm"]),
    ("test_404_hard_disables_model_and_hides_it", [79, 92],
     "A 404 hard-disables the model so later availability and the public model list hide it.",
     ["test", "failover", "llm"]),
    ("test_all_models_failing_raises_normalised_error", [95, 104],
     "When every model in the chain fails, open_stream raises a normalised all_models_failed LLMError.",
     ["test", "error-handling", "llm"]),
]
for name, span, summary, tags in llm_tests:
    nodes.append(N(f"function:{tl}:{name}", "function", name, summary, tags, "simple", tl, span))

# --- retrieval tests (10+ lines) ---
tr = "tests/test_retrieval.py"
ret_tests = [
    ("test_delete_document_removes_dense_and_lexical", [56, 65],
     "Deleting a document drops both dense and lexical hits and leaves search empty for that owner.",
     ["test", "retrieval", "indexing"]),
    ("test_user_id_cannot_escape_index_dir", [95, 104],
     "A path-like user id is hashed into the index directory and cannot write outside it.",
     ["test", "security", "indexing"]),
    ("test_index_store_atomic_rollback_on_dim_mismatch", [169, 178],
     "Adding chunks with the wrong embedding dimension rolls the hybrid index back and leaves lexical search empty.",
     ["test", "indexing", "retrieval"]),
]
for name, span, summary, tags in ret_tests:
    nodes.append(N(f"function:{tr}:{name}", "function", name, summary, tags, "simple", tr, span))

# --- edges ---
edges = []

imports = {
    "backend/client/llm_client.py": [
        "backend/core/config.py",
        "backend/core/exceptions.py",
        "backend/utils/timeutil.py",
    ],
    "backend/services/llm_service.py": [
        "backend/client/llm_client.py",
        "backend/core/config.py",
        "backend/core/exceptions.py",
        "backend/utils/rate_limiter.py",
        "backend/utils/streaming.py",
        "backend/utils/text.py",
    ],
    "backend/utils/rate_limiter.py": ["backend/utils/timeutil.py"],
    "tests/conftest.py": [
        "backend/core/config.py",
        "backend/core/container.py",
        "backend/main.py",
        "backend/services/llm_service.py",
        "backend/utils/rate_limiter.py",
    ],
    "tests/test_artifacts.py": [
        "backend/services/artifact.py",
        "backend/utils/rate_limiter.py",
        "tests/conftest.py",
    ],
    "tests/test_llm.py": [
        "backend/client/llm_client.py",
        "backend/core/exceptions.py",
        "backend/services/llm_service.py",
        "backend/utils/rate_limiter.py",
        "tests/conftest.py",
    ],
    "tests/test_retrieval.py": [
        "backend/core/exceptions.py",
        "backend/services/index_store.py",
        "tests/conftest.py",
    ],
}
for src, targets in imports.items():
    for t in targets:
        edges.append(E(f"file:{src}", f"file:{t}", "imports", 0.7))

# contains + exports for every symbol node
for node in nodes:
    if node["type"] in ("function", "class"):
        edges.append(E(f"file:{node['filePath']}", node["id"], "contains", 1.0))
        edges.append(E(f"file:{node['filePath']}", node["id"], "exports", 0.8))

inherits = [
    ("class:backend/core/exceptions.py:DocumentParseError", "class:backend/core/exceptions.py:AppError"),
    ("class:backend/core/exceptions.py:FetchError", "class:backend/core/exceptions.py:DocumentParseError"),
    ("class:backend/core/exceptions.py:IngestError", "class:backend/core/exceptions.py:AppError"),
    ("class:backend/core/exceptions.py:SessionNotFound", "class:backend/core/exceptions.py:AppError"),
    ("class:backend/core/exceptions.py:EmbeddingUnavailable", "class:backend/core/exceptions.py:AppError"),
    ("class:backend/core/exceptions.py:LLMError", "class:backend/core/exceptions.py:AppError"),
]
for s, t in inherits:
    edges.append(E(s, t, "inherits", 0.9))

tested_by = [
    ("file:backend/core/config.py", "file:tests/conftest.py"),
    ("file:backend/core/container.py", "file:tests/conftest.py"),
    ("file:backend/main.py", "file:tests/conftest.py"),
    ("file:backend/services/llm_service.py", "file:tests/conftest.py"),
    ("file:backend/utils/rate_limiter.py", "file:tests/conftest.py"),
    ("file:backend/services/artifact.py", "file:tests/test_artifacts.py"),
    ("file:backend/utils/rate_limiter.py", "file:tests/test_artifacts.py"),
    ("file:backend/client/llm_client.py", "file:tests/test_llm.py"),
    ("file:backend/core/exceptions.py", "file:tests/test_llm.py"),
    ("file:backend/services/llm_service.py", "file:tests/test_llm.py"),
    ("file:backend/utils/rate_limiter.py", "file:tests/test_llm.py"),
    ("file:backend/core/exceptions.py", "file:tests/test_retrieval.py"),
    ("file:backend/services/index_store.py", "file:tests/test_retrieval.py"),
]
for s, t in tested_by:
    edges.append(E(s, t, "tested_by", 0.5))

def fn(path, name):
    return f"function:{path}:{name}"

def cl(path, name):
    return f"class:{path}:{name}"

calls = [
    (fn("backend/client/llm_client.py", "classify_exception"), fn("backend/utils/timeutil.py", "parse_duration")),
    (fn("backend/client/llm_client.py", "classify_exception"), cl("backend/core/exceptions.py", "LLMError")),
    (cl("backend/services/llm_service.py", "LLMStream"), fn("backend/client/llm_client.py", "chunk_text")),
    (cl("backend/services/llm_service.py", "LLMStream"), fn("backend/client/llm_client.py", "chunk_usage")),
    (cl("backend/services/llm_service.py", "LLMStream"), fn("backend/client/llm_client.py", "classify_exception")),
    (cl("backend/services/llm_service.py", "LLMStream"), fn("backend/client/llm_client.py", "close_stream")),
    (cl("backend/services/llm_service.py", "LLMStream"), cl("backend/core/exceptions.py", "LLMError")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "build_provider_clients")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "wrap_clients")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "chunk_text")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "chunk_usage")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "classify_exception")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/client/llm_client.py", "close_stream")),
    (cl("backend/services/llm_service.py", "LLMService"), cl("backend/client/llm_client.py", "ProviderClient")),
    (cl("backend/services/llm_service.py", "LLMService"), cl("backend/utils/streaming.py", "ThinkFilter")),
    (cl("backend/services/llm_service.py", "LLMService"), fn("backend/utils/text.py", "estimate_tokens")),
    (cl("backend/services/llm_service.py", "LLMService"), cl("backend/utils/rate_limiter.py", "QuotaGuard")),
    (cl("backend/services/llm_service.py", "LLMService"), cl("backend/core/exceptions.py", "LLMError")),
    (cl("backend/utils/rate_limiter.py", "QuotaGuard"), fn("backend/utils/timeutil.py", "parse_duration")),
    (fn("tests/conftest.py", "settings"), cl("backend/core/config.py", "Settings")),
    (fn("tests/conftest.py", "build_container"), cl("backend/services/llm_service.py", "LLMService")),
    (fn("tests/conftest.py", "build_container"), cl("backend/utils/rate_limiter.py", "QuotaGuard")),
    (fn("tests/conftest.py", "build_container"), cl("backend/core/container.py", "ServiceContainer")),
    (fn("tests/conftest.py", "client_factory"), fn("backend/main.py", "create_app")),
]

art = "backend/services/artifact.py"
calls += [
    (fn(ta, "test_validator_rejects_empty_and_over_cap_documents"), fn(art, "validate_document")),
    (fn(ta, "test_when_a_document_is_and_is_not_authored"), fn(art, "should_author_document")),
    (fn(ta, "test_when_a_document_is_and_is_not_authored"), fn(art, "wants_file")),
    (fn(ta, "test_when_a_document_is_and_is_not_authored"), fn(art, "is_bare_save")),
    (fn(ta, "test_when_a_document_is_and_is_not_authored"), fn(art, "preferred_format")),
    (fn(ta, "test_when_a_document_is_and_is_not_authored"), fn(art, "is_restyle")),
    (fn(ta, "test_five_formats_are_built_from_the_same_document"), fn(art, "validate_document")),
    (fn(ta, "test_five_formats_are_built_from_the_same_document"), fn(art, "render_download")),
    (fn(ta, "test_five_formats_are_built_from_the_same_document"), fn(art, "safe_filename")),
    (fn(ta, "test_five_formats_are_built_from_the_same_document"), fn(art, "content_disposition")),
    (fn(ta, "test_awkward_answers_still_become_every_format"), fn(art, "document_from_markdown")),
    (fn(ta, "test_awkward_answers_still_become_every_format"), fn(art, "render_download")),
    (fn(ta, "test_a_prose_answer_becomes_a_titled_sectioned_document"), fn(art, "topic_title")),
    (fn(ta, "test_a_prose_answer_becomes_a_titled_sectioned_document"), fn(art, "document_from_markdown")),
    (fn(ta, "test_a_prose_answer_becomes_a_titled_sectioned_document"), fn(art, "render_download")),
    (fn(ta, "test_generate_pdf_saves_the_previous_answer"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_markdown_becomes_a_document_without_a_second_model_call"), fn(art, "document_from_markdown")),
    (fn(ta, "test_markdown_becomes_a_document_without_a_second_model_call"), fn(art, "document_source")),
    (fn(ta, "test_file_request_streams_the_answer_then_the_artifact"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_save_request_files_the_previous_answer_not_export_instructions"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_i_want_a_pdf_opens_a_file_even_when_the_model_refuses"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_every_mode_downloads_every_format"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_thin_answer_still_downloads_instead_of_failing"), cl(cf, "FakeProviderClient")),
    (fn(ta, "test_download_rate_limit_is_separate_from_chat"), cl(rl, "RateLimiter")),
    (fn(ta, "test_inline_markdown_is_stripped_in_non_markdown_formats"), fn(art, "document_from_markdown")),
    (fn(ta, "test_inline_markdown_is_stripped_in_non_markdown_formats"), fn(art, "render_download")),
    (fn(ta, "test_inline_markdown_is_stripped_in_non_markdown_formats"), fn(art, "strip_markdown_inline")),
    (fn(tl, "test_failover_on_rate_limit_and_5xx_and_timeout"), cl(svc, "LLMService")),
    (fn(tl, "test_failover_on_rate_limit_and_5xx_and_timeout"), cl(cf, "FakeProviderClient")),
    (fn(tl, "test_404_hard_disables_model_and_hides_it"), cl(svc, "LLMService")),
    (fn(tl, "test_all_models_failing_raises_normalised_error"), cl(svc, "LLMService")),
    (fn(tl, "test_all_models_failing_raises_normalised_error"), cl(exc, "LLMError")),
    (fn(tr, "test_index_store_atomic_rollback_on_dim_mismatch"), cl("backend/services/index_store.py", "HybridIndexStore")),
    (fn(tr, "test_index_store_atomic_rollback_on_dim_mismatch"), cl("backend/services/index_store.py", "ChunkIn")),
]
for s, t in calls:
    edges.append(E(s, t, "calls", 0.8))

# build_provider_clients -> Settings is a method call on a Settings instance (provider_configs).
# Neighbor symbol Settings exists. Keep it.

graph = {"nodes": nodes, "edges": edges}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

ids = [n["id"] for n in nodes]
print("nodes", len(nodes))
print("edges", len(edges))
print("dup_nodes", len(ids) - len(set(ids)))
from collections import Counter
print("edge_types", dict(Counter(e["type"] for e in edges)))
print("imports", sum(1 for e in edges if e["type"] == "imports"))
self_e = [e for e in edges if e["source"] == e["target"]]
print("self", len(self_e))
bad_tags = [n["id"] for n in nodes if not (3 <= len(n["tags"]) <= 5)]
print("bad_tags", bad_tags)
empty = [n["id"] for n in nodes if not n["summary"] or not n["tags"]]
print("empty", empty)
