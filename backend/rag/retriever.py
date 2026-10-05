"""
Hybrid keyword + vector retriever with Cohere Reranking.
"""

from __future__ import annotations

import cohere
from langchain_core.documents import Document

from backend.config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    COHERE_RERANK_TOP_N,
    RETRIEVAL_INITIAL_K,
)
from backend.rag.embeddings import embeddings_provider
from backend.services.vector_store import VectorStoreManager

store_manager = VectorStoreManager(embeddings_provider)

# Init Cohere client for reranking
_cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


def process_and_index(user_id: str, pages: list[dict], filename: str) -> bool:
    """Chunk and store document pages in FAISS."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = []
    for page in pages:
        text = page.get("text", "")
        if not text:
            continue
        num = page.get("page_number", "?")

        for chunk_idx, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": filename,
                        "page": num,
                        "chunk": chunk_idx,
                        "user_id": user_id,
                    }
                )
            )

    if not docs:
        return False

    return store_manager.upsert_documents(user_id, docs)


def retrieve_documents(user_id: str, query: str, top_k: int = COHERE_RERANK_TOP_N) -> list[Document]:
    """Retrieve documents using FAISS similarity, then rerank with Cohere."""
    store = store_manager.get_store(user_id)
    if not store:
        return []

    # 1. Recall (FAISS)
    try:
        results = store.similarity_search_with_score(query, k=RETRIEVAL_INITIAL_K)
    except Exception as exc:
        print(f"[retriever] FAISS error: {exc}")
        return []

    if not results:
        return []

    docs = [doc for doc, _ in results]

    # 2. Rerank (Cohere)
    if _cohere_client:
        try:
            resp = _cohere_client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=[d.page_content for d in docs],
                top_n=top_k,
                return_documents=False,
            )
            reranked = []
            for item in resp.results:
                doc = docs[item.index]
                doc.metadata["relevance"] = item.relevance_score
                reranked.append(doc)
            return reranked
        except Exception as exc:
            print(f"[retriever] Cohere rerank failed: {exc}")

    # Fallback to FAISS order
    for doc, dist in results[:top_k]:
        doc.metadata["relevance"] = float(1.0 / (1.0 + dist))
    return [doc for doc, _ in results[:top_k]]
