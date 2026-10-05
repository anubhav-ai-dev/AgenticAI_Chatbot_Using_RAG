"""
Manager for persistent FAISS vector stores.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import FAISS_DIR


class VectorStoreManager:
    """Manages loading/saving FAISS indexes to disk (one per user_id)."""

    def __init__(self, embeddings_provider):
        self.embeddings = embeddings_provider
        self.base_dir = FAISS_DIR

        # Cache of loaded stores: {user_id: [FAISS, index_path]}
        self._stores: dict[str, tuple[FAISS, Path]] = {}

    def get_store(self, user_id: str) -> FAISS | None:
        """Get or load a user's FAISS store."""
        if user_id in self._stores:
            return self._stores[user_id][0]

        index_path = self.base_dir / f"{user_id}_vector"
        if not index_path.exists():
            return None

        try:
            store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True  # Required since picking local files
            )
            self._stores[user_id] = (store, index_path)
            return store
        except Exception as exc:
            print(f"[vector_store] Failed to load store for '{user_id}': {exc}")
            return None

    def upsert_documents(self, user_id: str, documents: list[Document]) -> bool:
        """Add documents to a user's store and save to disk."""
        if not documents:
            return False

        store = self.get_store(user_id)
        index_path = self.base_dir / f"{user_id}_vector"

        try:
            if store:
                store.add_documents(documents)
            else:
                store = FAISS.from_documents(documents, self.embeddings)

            # Persist to disk
            store.save_local(str(index_path))
            self._stores[user_id] = (store, index_path)

            print(f"[vector_store] Stored {len(documents)} chunks, saved to {index_path.name}")
            return True
        except Exception as exc:
            print(f"[vector_store] Upsert failed: {exc}")
            return False

    def list_sources(self, user_id: str) -> list[str]:
        """Return unique document filenames indexed by this user."""
        store = self.get_store(user_id)
        if not store:
            return []

        try:
            # Document metadata is stored in FAISS docstore mapping
            mapping = store.docstore._dict
            sources = set()
            for doc in mapping.values():
                source = doc.metadata.get("source", "Unknown")
                sources.add(source)
            return list(sources)
        except Exception as exc:
            print(f"[vector_store] list_sources failed: {exc}")
            return []


# Global instance will be created in retriever.py
