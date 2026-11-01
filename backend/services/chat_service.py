"""
Chat pipeline: route → (rewrite) → semantic cache → retrieval ∥ web search →
prompt → streamed generation with model failover → persistence.

`ChatService.stream()` yields plain event dicts ({"event": str, "data": dict});
the HTTP layer only has to serialise them (SSE) or aggregate them (JSON).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from backend.core.config import MODES, Settings
from backend.core.exceptions import LLMError
from backend.services.artifact import (
    ArtifactError,
    FILE_ANSWER_NOTE,
    FILE_RESTYLE_NOTE,
    FILE_SAVE_NOTE,
    document_from_markdown,
    document_source,
    is_bare_save,
    is_restyle,
    log_ready,
    preferred_format,
    should_author_document,
    style_note,
    topic_title,
    wants_file,
)
from backend.services.llm_service import LLMService
from backend.services.prompt_builder import PromptBuilder, select_specialist
from backend.services.retrieval_service import RetrievalResult, RetrievalService
from backend.services.search_service import SearchOutcome, WebSearchService
from backend.services.session_service import SessionService
from backend.utils.cache import CachedAnswer, SemanticCache

logger = logging.getLogger(__name__)

Event = dict[str, Any]


# ── routing ──────────────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    use_rag: bool
    use_web: bool
    research: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        if self.use_rag and self.use_web:
            return "rag+web"
        return "rag" if self.use_rag else ("web" if self.use_web else "direct")


_SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|thanks?( you)?|thx|ok(ay)?|cool|great|nice|bye|goodbye|"
    r"good (morning|afternoon|evening|night)|how are you\??|who are you\??|what can you do\??)\W*$",
    re.IGNORECASE)
_FRESH = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|latest|newest|recent(ly)?|current(ly)?|right now|news|"
    r"breaking|trending|price|prices|stock|stocks|weather|forecast|score|scores|"
    r"release[sd]?|update[sd]?|this (week|month|year)|20(2[4-9]|3\d)|who (is|are) the|"
    r"search|google|look ?up|online|https?://)\b", re.IGNORECASE)
_FOLLOWUP_START = re.compile(
    r"^\s*(and|also|but|so|then|what about|how about|why|how come|what if|tell me more|more|"
    r"elaborate|explain (that|it|this)|can you (explain|elaborate|expand|summari[sz]e)|"
    r"it|its|this|that|those|these|they|them)\b", re.IGNORECASE)
_PRONOUN = re.compile(r"\b(it|its|this|that|those|these|they|them|he|she|his|her|their|there)\b",
                      re.IGNORECASE)


class QueryRouter:
    """Heuristic routing — no LLM call, so it costs microseconds."""

    @staticmethod
    def is_smalltalk(query: str) -> bool:
        return bool(_SMALLTALK.match(query))

    @staticmethod
    def needs_fresh_info(query: str) -> bool:
        return bool(_FRESH.search(query))

    @staticmethod
    def is_followup(query: str, has_history: bool) -> bool:
        if not has_history:
            return False
        words = query.split()
        if len(words) <= 3 or _FOLLOWUP_START.match(query):
            return True
        return len(words) <= 8 and bool(_PRONOUN.search(query))

    def decide(self, *, mode: str, query: str, has_docs: bool, web_toggle: bool,
               web_available: bool = True) -> RouteDecision:
        if self.is_smalltalk(query):
            return RouteDecision(False, False, reason="smalltalk")
        if mode == "research":
            return RouteDecision(has_docs, web_available, research=True, reason="research mode")
        fresh = web_toggle and web_available and self.needs_fresh_info(query)
        return RouteDecision(has_docs, fresh, reason="fresh info" if fresh else "default")


# ── pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class ChatCommand:
    session_id: str
    message: str = ""
    mode: str = "general"
    model: Optional[str] = None
    web_search: bool = True
    regenerate: bool = False


def _event(name: str, **data: Any) -> Event:
    return {"event": name, "data": data}


def _error(code: str, message: str, retryable: bool = True, **extra: Any) -> Event:
    return _event("error", code=code, message=message, retryable=retryable, **extra)


class ChatService:
    def __init__(self, settings: Settings, llm: LLMService, retrieval: RetrievalService,
                 search: WebSearchService, prompts: PromptBuilder, sessions: SessionService,
                 semantic_cache: SemanticCache, router: Optional[QueryRouter] = None,
                 github: Any = None) -> None:
        self._s = settings
        self._llm = llm
        self._retrieval = retrieval
        self._search = search
        self._prompts = prompts
        self._sessions = sessions
        self._cache = semantic_cache
        self._router = router or QueryRouter()
        self._github = github

    # ── public entry points ──────────────────────────────────────────────────

    async def stream(self, cmd: ChatCommand) -> AsyncIterator[Event]:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        meta = await asyncio.to_thread(self._sessions.session_meta, cmd.session_id)
        if meta is None:
            yield _error("session_not_found", "Chat session not found.", retryable=False)
            return
        user_id = meta["user_id"]
        mode = cmd.mode if cmd.mode in MODES else meta["mode"]
        model = cmd.model if cmd.model and self._llm.has_model(cmd.model) else \
            (meta["model"] if self._llm.has_model(meta["model"]) else self._s.llm_default_model)

        # persist the user turn (server is the single source of truth for history)
        if cmd.regenerate:
            text = await asyncio.to_thread(self._sessions.pop_last_exchange, cmd.session_id)
            if not text:
                yield _error("nothing_to_regenerate", "There is no previous message to regenerate.",
                             retryable=False)
                return
        else:
            text = cmd.message.strip()
            if not text:
                yield _error("empty_message", "Message is empty.", retryable=False)
                return
            await asyncio.to_thread(self._sessions.add_message, cmd.session_id, "user", text)

        history = await asyncio.to_thread(
            self._sessions.history, cmd.session_id, self._s.history_max_messages, True)
        has_docs = await asyncio.to_thread(self._retrieval.has_documents, user_id)
        decision = self._router.decide(mode=mode, query=text, has_docs=has_docs,
                                       web_toggle=cmd.web_search,
                                       web_available=self._search.available)
        yield _event("meta", route=decision.label, mode=mode, reason=decision.reason,
                     requested_model=model)

        # Compute file_intent early — needed to gate rephrase and semantic cache.
        file_intent = wants_file(text)

        # ── Query rephrase (code mode only) ──────────────────────────────────
        # Launched right after meta so it overlaps with the follow-up rewrite and
        # RAG / web fetches that happen below. Collected just before prompt build.
        # History always stores the original message; only the model sees the rephrased query.
        # Skipped for file/artifact requests — the artifact service needs the raw query.
        query_for_model = text
        rephrase_task: Optional[asyncio.Task] = None
        if (mode == "code" and not cmd.regenerate
                and self._s.query_rewrite_enabled and not file_intent):
            yield _event("status", stage="understanding", state="start")
            rephrase_task = asyncio.create_task(self._rephrase_query(text))

        # follow-up → standalone query (only when retrieval/search will use it)
        followup = self._router.is_followup(text, bool(history))
        search_query = text
        if followup and (decision.use_rag or decision.use_web):
            yield _event("status", stage="rewriting", state="start")
            t = time.perf_counter()
            search_query = await self._rewrite(text, history)
            timings["rewrite_ms"] = (time.perf_counter() - t) * 1000
            yield _event("status", stage="rewriting", state="done",
                         elapsed_ms=timings["rewrite_ms"], detail=search_query[:120])

        # semantic answer cache (standalone, non-live queries only). File turns never
        # replay a cached chat answer — that would skip the preview attachment.
        qvec = None
        cache_scope = None
        if (self._s.semantic_cache_enabled and not followup and not decision.use_web
                and not cmd.regenerate and not file_intent
                and not self._router.needs_fresh_info(text)):
            qvec = await self._retrieval.embed_query(text)
            if qvec is not None:
                cache_scope = (user_id, mode, model, self._retrieval.version(user_id), has_docs)
                hit = self._cache.lookup(cache_scope, qvec)
                if hit is not None:
                    async for ev in self._replay_cached(cmd, hit, timings, t_start):
                        yield ev
                    return

        # gather context in parallel
        rag: Optional[RetrievalResult] = None
        web: Optional[SearchOutcome] = None
        tasks: dict[asyncio.Task, str] = {}
        if decision.use_rag:
            yield _event("status", stage="retrieving", state="start")
            tasks[asyncio.create_task(self._retrieval.search(user_id, search_query))] = "retrieving"
        if decision.use_web:
            yield _event("status", stage="searching", state="start")
            tasks[asyncio.create_task(self._web(decision, search_query))] = "searching"
        try:
            while tasks:
                done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    stage = tasks.pop(task)
                    try:
                        res = task.result()
                    except Exception:  # noqa: BLE001 — context is best-effort
                        logger.exception("%s branch failed", stage)
                        res = None
                    if stage == "retrieving":
                        rag = res
                        r = res or RetrievalResult(error="failed")
                        timings["retrieve_ms"] = r.timings.get("total_ms", 0.0)
                        yield _event("status", stage=stage, state="done",
                                     elapsed_ms=timings["retrieve_ms"],
                                     detail=f"{len(r.items)} excerpt(s)" + (" · cached" if r.cached else ""),
                                     error=r.error, mode=r.mode, reranked=r.reranked)
                    else:
                        web = res
                        o = res or SearchOutcome(status="error")
                        timings["search_ms"] = o.elapsed_ms
                        yield _event("status", stage=stage, state="done", elapsed_ms=o.elapsed_ms,
                                     detail=f"{len(o.results)} source(s)" + (" · cached" if o.cached else ""),
                                     error=None if o.status == "ok" else o.status)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

        rag_items = rag.items if rag and rag.used_rag else []
        web_results = web.results if web and web.status == "ok" else []
        effective_route = RouteDecision(bool(rag_items), bool(web_results)).label

        notes: list[str] = []
        has_prev = any((m.get("role") == "assistant" and (m.get("content") or "").strip())
                       for m in history)
        if file_intent:
            if is_restyle(text) and has_prev:
                notes.append(FILE_RESTYLE_NOTE)
            elif is_bare_save(text) and has_prev:
                notes.append(FILE_SAVE_NOTE)
            else:
                notes.append(FILE_ANSWER_NOTE)
            extra_style = style_note(text)
            if extra_style:
                notes.append(extra_style)
        if mode == "document" and not has_docs:
            notes.append("The user has not uploaded any documents. Say so and suggest uploading "
                         "one; you may still answer from general knowledge if you state that clearly.")
        elif mode == "document" and decision.use_rag and not rag_items:
            notes.append("No relevant passages were found in the user's documents for this question. "
                         "Say that clearly before offering any general-knowledge answer.")
        if decision.use_web and web is not None and web.status != "ok":
            notes.append("Live web search was unavailable; say you could not verify current information.")
        elif decision.use_web and web is not None and not web_results:
            notes.append("Web search returned no results; answer from your own knowledge and say so.")

        # ── Collect rephrase result (started after meta, runs in parallel with RAG/web) ──
        if rephrase_task is not None:
            t_rephrase = time.perf_counter()
            try:
                query_for_model = await asyncio.wait_for(
                    asyncio.shield(rephrase_task), timeout=2.0)
            except Exception as exc:   # noqa: BLE001 — always fall back to original
                logger.info("Query rephrase skipped (%s)", type(exc).__name__)
                if not rephrase_task.done():
                    rephrase_task.cancel()
                query_for_model = text
            timings["rephrase_ms"] = (time.perf_counter() - t_rephrase) * 1000
            yield _event("status", stage="understanding", state="done",
                         elapsed_ms=timings["rephrase_ms"],
                         detail=query_for_model[:200])

        repo_brief = ""
        if self._github is not None:
            try:
                repo_brief = await asyncio.to_thread(self._github.combined_brief, user_id)
            except Exception:  # noqa: BLE001 — chat still answers without the brief
                logger.info("repo brief skipped")
        specialist = select_specialist(
            mode, query_for_model, file_intent=file_intent, has_repo=bool(repo_brief))
        prompt = self._prompts.build(mode=mode, query=query_for_model, history=history,
                                     rag_items=rag_items, web_results=web_results, notes=notes,
                                     specialist=specialist, repo_brief=repo_brief)
        if prompt.citations:
            yield _event("citations", items=prompt.citations)

        # generate
        yield _event("status", stage="generating", state="start")
        file_tokens = min(8192, max(self._s.llm_max_tokens, 4096)) if file_intent else None
        try:
            llm_stream = await self._llm.open_stream(prompt.messages, model=model,
                                                     max_tokens=file_tokens)
        except LLMError as err:
            yield _error(err.code, err.message, retryable=err.retryable,
                         attempts=err.attempts, retry_after=err.retry_after)
            return
        timings["ttft_ms"] = (time.perf_counter() - t_start) * 1000
        yield _event("status", stage="generating", state="first_token",
                     elapsed_ms=llm_stream.ttft_ms, detail=llm_stream.model)
        yield _event("model", model=llm_stream.model, fell_back_from=llm_stream.fell_back_from)

        parts: list[str] = []
        finalized = False
        stream_error: Optional[LLMError] = None
        try:
            try:
                async for piece in llm_stream:
                    parts.append(piece)
                    yield _event("token", text=piece)
            except LLMError as err:
                stream_error = err
            answer = "".join(parts)
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            artifact = None
            artifact_error = None
            if stream_error is None and answer.strip():
                artifact, artifact_error = self._maybe_artifact(text, answer, history)
            msg_meta = {
                "model": llm_stream.model, "route": effective_route, "mode": mode,
                "fell_back_from": llm_stream.fell_back_from, "cached": False,
                "timings": {k: round(v, 1) for k, v in timings.items()},
                "usage": llm_stream.usage,
                "partial": stream_error is not None,
            }
            file_format = preferred_format(text) if artifact is not None else None
            if artifact is not None:
                msg_meta["artifact"] = artifact
                msg_meta["artifact_format"] = file_format
            message_id = None
            if answer.strip():
                message_id = await self._persist(cmd.session_id, answer, prompt.citations, msg_meta)
                finalized = True
            if stream_error is not None:
                yield _error(stream_error.code, stream_error.message, retryable=True,
                             partial=bool(answer.strip()), message_id=message_id)
                return
            if artifact is not None:
                yield _event("artifact", message_id=message_id, title=artifact["title"],
                             blocks=artifact["blocks"], format=file_format)
            elif artifact_error:
                yield _event("artifact", error=artifact_error)
            if (cache_scope is not None and qvec is not None and answer.strip()
                    and not llm_stream.fell_back_from and artifact_error is None):
                self._cache.store(cache_scope, qvec, CachedAnswer(
                    answer, prompt.citations, llm_stream.model, effective_route, artifact,
                    file_format))
            yield _event("done", message_id=message_id, model=llm_stream.model,
                         fell_back_from=llm_stream.fell_back_from, cached=False,
                         route=effective_route, persisted=message_id is not None,
                         timings=msg_meta["timings"], usage=llm_stream.usage)
        finally:
            await llm_stream.aclose()
            if not finalized and parts:                    # client disconnected / Stop pressed
                partial = "".join(parts)
                if partial.strip():
                    self._persist_sync(cmd.session_id, partial, prompt.citations, {
                        "model": llm_stream.model, "route": effective_route, "mode": mode,
                        "stopped": True, "partial": True})

    async def complete(self, cmd: ChatCommand) -> dict[str, Any]:
        """Non-streaming wrapper: aggregates the event stream."""
        text: list[str] = []
        citations: list[dict[str, Any]] = []
        out: dict[str, Any] = {"route": "direct", "cached": False, "model": None,
                               "message_id": None, "timings": {}}
        async for ev in self.stream(cmd):
            name, data = ev["event"], ev["data"]
            if name == "token":
                text.append(data["text"])
            elif name == "citations":
                citations = data["items"]
            elif name == "meta":
                out["route"] = data["route"]
            elif name == "done":
                out.update(message_id=data.get("message_id"), model=data.get("model"),
                           cached=data.get("cached", False), timings=data.get("timings", {}))
            elif name == "error":
                out["error"] = data
        out["response"] = "".join(text)
        out["citations"] = citations
        return out

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _replay_cached(self, cmd: ChatCommand, hit: CachedAnswer,
                             timings: dict[str, float], t_start: float) -> AsyncIterator[Event]:
        if hit.citations:
            yield _event("citations", items=hit.citations)
        yield _event("model", model=hit.model, fell_back_from=[])
        for i in range(0, len(hit.text), 48):
            yield _event("token", text=hit.text[i:i + 48])
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        meta = {"model": hit.model, "route": hit.route, "cached": True,
                "timings": {k: round(v, 1) for k, v in timings.items()}}
        if hit.artifact:
            meta["artifact"] = hit.artifact
            if hit.artifact_format:
                meta["artifact_format"] = hit.artifact_format
        message_id = await self._persist(cmd.session_id, hit.text, hit.citations, meta)
        if hit.artifact:
            yield _event("artifact", message_id=message_id, title=hit.artifact["title"],
                         blocks=hit.artifact["blocks"], format=hit.artifact_format or "md")
        yield _event("done", message_id=message_id, model=hit.model, fell_back_from=[],
                     cached=True, route=hit.route, persisted=message_id is not None,
                     timings=meta["timings"], usage=None)

    async def _web(self, decision: RouteDecision, query: str) -> SearchOutcome:
        if decision.research:
            return await self._search.multi_search(
                self._search.generate_sub_queries(query),
                max_results_each=3, depth=self._s.tavily_research_depth)
        return await self._search.search(query)

    async def _rewrite(self, query: str, history: list[dict[str, str]]) -> str:
        """Standalone retrieval query for a follow-up; falls back to simple concatenation."""
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        fallback = f"{last_user[:200]} {query}".strip() if last_user else query
        if not self._s.query_rewrite_enabled or not self._llm.configured:
            return fallback
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        messages = [
            {"role": "system", "content": (
                "Rewrite the user's latest message as ONE standalone search query that includes "
                "any needed context from the conversation. Output only the query.")},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest message: {query}\n\n"
                                        "Standalone query:"},
        ]
        try:
            res = await asyncio.wait_for(
                self._llm.complete(messages, fast=True, max_tokens=200, temperature=0.0), 4.0)
            rewritten = res.text.strip().splitlines()[0].strip().strip('"') if res.text.strip() else ""
            return rewritten[:300] or fallback
        except Exception as exc:  # noqa: BLE001
            logger.info("Query rewrite skipped (%s)", type(exc).__name__)
            return fallback

    async def _rephrase_query(self, query: str) -> str:
        """Rephrase the user's query to be technically precise without changing its intent.

        Used in code mode to improve model output quality.  A tight timeout and silent
        fallback mean this never blocks the main response even if the fast model is slow.
        """
        if not self._llm.configured:
            return query
        messages = [
            {"role": "system", "content": (
                "You are a technical query reformulator for a coding AI assistant. "
                "Rephrase the user's message to be precise and unambiguous.\n\n"
                "Rules (follow strictly):\n"
                "1. Preserve the EXACT intent — do not add new requirements or constraints.\n"
                "2. Fix grammar and make phrasing natural and professional.\n"
                "3. For UI / frontend requests: specify the expected output format naturally "
                "   (e.g. 'as a single self-contained HTML file with all CSS and JS inline').\n"
                "4. For backend / script requests: name the language or framework explicitly "
                "   if it is clear from context.\n"
                "5. Output ONLY the rephrased query — no explanation, no preamble."
            )},
            {"role": "user", "content": query},
        ]
        try:
            res = await self._llm.complete(
                messages, fast=True, max_tokens=120, temperature=0.0)
            rephrased = (res.text or "").strip().splitlines()[0].strip().strip('"\'')
            return rephrased if rephrased else query
        except Exception as exc:   # noqa: BLE001
            logger.info("_rephrase_query failed (%s)", type(exc).__name__)
            return query

    def _maybe_artifact(self, user_text: str, answer: str,
                        history: list[dict[str, str]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Return (document, None) or (None, error). Built from the answer, not a second model call."""
        if not should_author_document(user_text, answer):
            return None, None
        try:
            source = document_source(user_text, answer, history)
            doc = document_from_markdown(source, fallback_title=topic_title(user_text, history))
        except ArtifactError as err:
            logger.info("artifact skipped (%s)", type(err).__name__)
            return None, err.message
        except Exception as exc:  # noqa: BLE001 — the chat answer still completes
            logger.info("artifact skipped (%s)", type(exc).__name__)
            return None, "The document could not be created."
        log_ready(doc)
        return doc, None

    async def _persist(self, session_id: str, text: str, citations: list[dict[str, Any]],
                       meta: dict[str, Any]) -> Optional[str]:
        try:
            msg = await asyncio.to_thread(self._sessions.add_message, session_id, "assistant",
                                          text, citations or None, meta)
            return msg.id
        except Exception:  # noqa: BLE001 — the answer was already delivered
            logger.exception("Failed to persist assistant message")
            return None

    def _persist_sync(self, session_id: str, text: str, citations: list[dict[str, Any]],
                      meta: dict[str, Any]) -> None:
        try:
            self._sessions.add_message(session_id, "assistant", text, citations or None, meta)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist partial assistant message")
