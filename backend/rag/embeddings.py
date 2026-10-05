"""
Cohere-backed embeddings wrapper.
Falls back to Google Gemini embeddings if Cohere fails.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings

from backend.config import (
    COHERE_API_KEY,
    GOOGLE_API_KEY,
    COHERE_EMBED_MODEL,
)


class HybridEmbeddings(Embeddings):
    """
    Tries Cohere first. If it fails or key is missing, tries Google Gemini.
    """

    def __init__(self):
        self.cohere = None
        self.google = None

        if COHERE_API_KEY:
            try:
                from langchain_cohere import CohereEmbeddings
                self.cohere = CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model=COHERE_EMBED_MODEL,
                )
            except Exception as e:
                print(f"[embeddings] Cohere init failed: {e}")

        if GOOGLE_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.google = GoogleGenerativeAIEmbeddings(
                    google_api_key=GOOGLE_API_KEY,
                    model="models/text-embedding-004",
                )
            except Exception as e:
                print(f"[embeddings] Google init failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.cohere:
            try:
                return self.cohere.embed_documents(texts)
            except Exception as e:
                print(f"[embeddings] Cohere docs failed: {e}")

        if self.google:
            return self.google.embed_documents(texts)

        raise RuntimeError("No embedding provider available.")

    def embed_query(self, text: str) -> List[float]:
        if self.cohere:
            try:
                return self.cohere.embed_query(text)
            except Exception as e:
                print(f"[embeddings] Cohere query failed: {e}")

        if self.google:
            return self.google.embed_query(text)

        raise RuntimeError("No embedding provider available.")


# Global instance
embeddings_provider = HybridEmbeddings()
