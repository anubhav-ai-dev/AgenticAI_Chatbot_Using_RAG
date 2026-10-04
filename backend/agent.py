"""
AI agent: LangGraph workflow + RAG (Cohere embeddings / FAISS) + in-memory session history.

Graph topology:
    router ──► rag ──► agent ──► END
           └──────────────────►

- router: decides whether to use RAG (similarity score vs threshold) or plain LLM/search.
- rag:    builds an enriched prompt with numbered excerpts and page citations.
- agent:  runs the LLM; optionally wraps it in a Tavily ReAct search agent.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, List, Dict, Optional

import cohere
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from backend.config import (
    GROQ_API_KEY,
    OPENAI_API_KEY,
    TAVILY_API_KEY,
    COHERE_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    RETRIEVAL_INITIAL_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    SESSION_HISTORY_LIMIT,
    LLM_TEMPERATURE,
    TAVILY_MAX_RESULTS,
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    use_rag: bool
    similarity_threshold: float
    retrieved_docs: List[Document]


# ── Memory manager ────────────────────────────────────────────────────────────

class MemoryManager:
    """In-memory store for per-session chat history."""

    def __init__(self) -> None:
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, message: Dict[str, Any]) -> None:
        self._sessions.setdefault(session_id, []).append(
            {**message, "timestamp": datetime.now().isoformat()}
        )

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ── RAG manager ───────────────────────────────────────────────────────────────

class RAGManager:
    """Manages FAISS vector stores and Cohere embedding / reranking per user."""

    def __init__(self) -> None:
        self._cohere_client: Optional[cohere.Client] = None
        self._embeddings: Optional[CohereEmbeddings] = None
        self.cohere_available: bool = False
        self._vector_stores: Dict[str, FAISS] = {}

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        if COHERE_API_KEY:
            try:
                self._cohere_client = cohere.Client(COHERE_API_KEY)
                self._embeddings = CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model=COHERE_EMBED_MODEL,
                )
                self.cohere_available = True
                print("[agent] Cohere initialised successfully.")
            except Exception as exc:
                print(f"[agent] Warning: Cohere init failed — {exc}. RAG disabled.")
        else:
            print("[agent] Warning: COHERE_API_KEY not set. RAG disabled.")

    # ── public API ──────────────────────────────────────────────────────────

    def process_pdf(self, user_id: str, pages: List[Dict[str, Any]], filename: str) -> bool:
        """Chunk page texts and store in the user's FAISS index."""
        if not self.cohere_available or not self._embeddings:
            print("[agent] RAG not available — skipping PDF processing.")
            return False

        documents: List[Document] = []
        for page in pages:
            page_num: int = page["page_number"]
            text: str = page.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            for i, chunk in enumerate(self._text_splitter.split_text(text)):
                if len(chunk.strip()) < 20:
                    continue
                documents.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata={
                            "source": filename,
                            "page_number": page_num,
                            "user_id": user_id,
                            "chunk_id": i,
                            "timestamp": datetime.now().isoformat(),
                            "char_count": len(chunk),
                        },
                    )
                )

        if not documents:
            print("[agent] No valid chunks extracted from PDF.")
            return False

        try:
            if user_id in self._vector_stores:
                self._vector_stores[user_id].add_documents(documents)
            else:
                self._vector_stores[user_id] = FAISS.from_documents(documents, self._embeddings)
            print(f"[agent] Stored {len(documents)} chunks for user '{user_id}'.")
            return True
        except Exception as exc:
            print(f"[agent] Error storing documents: {exc}")
            traceback.print_exc()
            return False

    def similarity_score(self, user_id: str, query: str) -> float:
        """Return a 0-1 similarity score between the query and the user's docs."""
        if user_id not in self._vector_stores:
            return 0.0
        try:
            results = self._vector_stores[user_id].similarity_search_with_score(query, k=1)
            if results:
                _, distance = results[0]
                return float(1.0 / (1.0 + distance))
        except Exception as exc:
            print(f"[agent] Similarity score error: {exc}")
        return 0.0

    def retrieve(self, user_id: str, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        """Two-stage retrieval: FAISS recall → Cohere rerank."""
        if user_id not in self._vector_stores:
            return []

        initial_k = min(k * 4, RETRIEVAL_INITIAL_K)
        try:
            results = self._vector_stores[user_id].similarity_search_with_score(query, k=initial_k)
        except Exception as exc:
            print(f"[agent] FAISS search error: {exc}")
            return []

        if not results:
            return []

        docs = [doc for doc, _ in results]

        # Rerank with Cohere when available
        if self.cohere_available and self._cohere_client:
            try:
                rerank_resp = self._cohere_client.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=query,
                    documents=[d.page_content for d in docs],
                    top_n=k,
                    return_documents=False,
                )
                reranked: List[Document] = []
                for result in rerank_resp.results:
                    doc = docs[result.index]
                    doc.metadata["relevance_score"] = result.relevance_score
                    reranked.append(doc)
                print(f"[agent] Reranked to top {len(reranked)} docs.")
                return reranked
            except Exception as exc:
                print(f"[agent] Reranking failed ({exc}), falling back to FAISS order.")

        # Fallback: use FAISS similarity scores
        for doc, dist in results[:k]:
            doc.metadata["relevance_score"] = float(1.0 / (1.0 + dist))
        return [doc for doc, _ in results[:k]]

    def list_user_documents(self, user_id: str) -> List[str]:
        """Return unique document filenames uploaded by this user."""
        if user_id not in self._vector_stores:
            return []
        try:
            # Empty-string search with high k to enumerate all metadata
            docs = self._vector_stores[user_id].similarity_search("", k=10_000)
            return list({d.metadata.get("source", "Unknown") for d in docs})
        except Exception as exc:
            print(f"[agent] list_user_documents error: {exc}")
            return []

    def has_documents(self, user_id: str) -> bool:
        return user_id in self._vector_stores


# ── Module-level singletons ───────────────────────────────────────────────────

memory_manager = MemoryManager()
rag_manager = RAGManager()


# ── Graph nodes ───────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Decide RAG vs LLM/Search path based on query-document similarity."""
    query = state["messages"][-1].content if state["messages"] else ""
    user_id = state.get("user_id", "default")
    threshold = state.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)

    print(f"\n[agent] Router — query: {query[:80]}…")

    if rag_manager.cohere_available and rag_manager.has_documents(user_id):
        score = rag_manager.similarity_score(user_id, query)
        print(f"[agent] Similarity: {score:.4f}, threshold: {threshold}")
        if score > threshold:
            docs = rag_manager.retrieve(user_id, query)
            state["retrieved_docs"] = docs
            state["use_rag"] = True
            print(f"[agent] → RAG ({len(docs)} docs)")
            return state

    state["use_rag"] = False
    state["retrieved_docs"] = []
    print("[agent] → LLM/Search")
    return state


def rag_node(state: AgentState) -> AgentState:
    """Replace the last user message with a context-enriched prompt."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return state

    context_parts: List[str] = []
    for idx, doc in enumerate(docs, 1):
        page = doc.metadata.get("page_number", "?")
        source = doc.metadata.get("source", "?")
        relevance = doc.metadata.get("relevance_score", 0.0)
        context_parts.append(
            f"[EXCERPT {idx}] (Source: {source}, Page: {page}, Relevance: {relevance:.3f})\n"
            f"{doc.page_content}"
        )

    context = "\n\n" + ("=" * 80 + "\n\n").join(context_parts) + "\n\n" + "=" * 80
    query = state["messages"][-1].content

    enriched_prompt = f"""You are a precise document analysis assistant. Answer ONLY from the excerpts below.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Use ONLY the excerpts above — do not add outside knowledge.
2. Cite page numbers inline (e.g. "According to page 5…").
3. If the excerpts lack sufficient information, say so clearly.
4. End with a **References** section listing every page you cited.

Your answer:"""

    state["messages"] = state["messages"][:-1] + [HumanMessage(content=enriched_prompt)]
    print(f"[agent] RAG context built — {len(context)} chars, {len(docs)} excerpts.")
    return state


def _build_agent_node(llm, tools: list, use_search: bool):
    """Return an agent_node closure bound to the given LLM and tools."""

    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]

        if state.get("use_rag", False):
            print("[agent] Generating RAG response…")
            response = llm.invoke(messages)
            return {"messages": [response]}

        if use_search and tools:
            try:
                print("[agent] Generating search-enabled response…")
                from langgraph.prebuilt import create_react_agent
                react_agent = create_react_agent(llm, tools)
                result = react_agent.invoke({"messages": messages})
                return {"messages": result["messages"]}
            except Exception as exc:
                print(f"[agent] ReAct agent failed ({exc}), falling back to plain LLM.")

        print("[agent] Generating plain LLM response…")
        response = llm.invoke(messages)
        return {"messages": [response]}

    return agent_node


def _build_graph(llm, tools: list, use_search: bool):
    """Compile and return the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("agent", _build_agent_node(llm, tools, use_search))
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda s: "rag" if s.get("use_rag", False) else "agent",
        {"rag": "rag", "agent": "agent"},
    )
    workflow.add_edge("rag", "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()


# ── Public API (called by main.py) ────────────────────────────────────────────

def get_response(
    llm_id: str,
    query: List[str],
    allow_search: bool,
    system_prompt: str,
    provider: str,
    user_id: str = "default",
    session_id: str = "default",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> str:
    """Run the full agent pipeline and return the AI's text response."""

    # Build LLM
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, temperature=LLM_TEMPERATURE)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, temperature=LLM_TEMPERATURE)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    # Build tools
    tools: list = []
    if allow_search and TAVILY_API_KEY:
        try:
            from langchain_tavily import TavilySearch
            tools = [TavilySearch(max_results=TAVILY_MAX_RESULTS)]
        except Exception as exc:
            print(f"[agent] Tavily init failed: {exc}")

    # Assemble messages: system + history + new query
    history = memory_manager.get_history(session_id)[-SESSION_HISTORY_LIMIT:]
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["type"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    messages.extend(HumanMessage(content=q) for q in query)

    # Run graph
    graph = _build_graph(llm, tools, allow_search)
    state: AgentState = {
        "messages": messages,
        "user_id": user_id,
        "session_id": session_id,
        "use_rag": False,
        "similarity_threshold": similarity_threshold,
        "retrieved_docs": [],
    }

    print(f"[agent] Invoking graph — user={user_id}, session={session_id}")
    result = graph.invoke(state)

    ai_messages = [m.content for m in result.get("messages", []) if isinstance(m, AIMessage)]
    final_response = ai_messages[-1] if ai_messages else "No response generated."
    print(f"[agent] Response: {len(final_response)} chars.")

    # Persist to memory
    for q in query:
        memory_manager.append(session_id, {"type": "human", "content": q})
    memory_manager.append(session_id, {"type": "ai", "content": final_response})

    return final_response


def process_pdf(user_id: str, pdf_path: str, filename: str) -> bool:
    """Extract text from a PDF file and index it for the user."""
    pages = _extract_pdf_pages(pdf_path)
    if not pages:
        print(f"[agent] No text extracted from {filename}.")
        return False
    return rag_manager.process_pdf(user_id, pages, filename)


def _extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Return a list of {page_number, text} dicts from a PDF file."""
    # Primary: pdfplumber
    try:
        import pdfplumber
        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if len(text.strip()) >= 50:
                    pages.append({"page_number": num, "text": text.strip()})
        print(f"[agent] Extracted {len(pages)} pages via pdfplumber.")
        return pages
    except ImportError:
        pass
    except Exception as exc:
        print(f"[agent] pdfplumber failed: {exc}")

    # Fallback: PyPDF2
    try:
        import PyPDF2
        pages = []
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if len(text.strip()) >= 50:
                    pages.append({"page_number": num, "text": text.strip()})
        print(f"[agent] Extracted {len(pages)} pages via PyPDF2.")
        return pages
    except Exception as exc:
        print(f"[agent] PyPDF2 failed: {exc}")

    return []


# ── Convenience wrappers for main.py ──────────────────────────────────────────

def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    return memory_manager.get_history(session_id)


def clear_chat_history(session_id: str) -> None:
    memory_manager.clear(session_id)


def get_user_documents(user_id: str) -> List[str]:
    return rag_manager.list_user_documents(user_id)
