"""
Python client utilities for calling backend services.
"""

from __future__ import annotations

import os
import requests
from typing import Optional

from backend.config import BACKEND_URL, COHERE_API_KEY


class BackendClient:
    """Client for calling the FastAPI backend from Python code."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

    def _get(self, path: str, params: Optional[dict] = None):
        return requests.get(f"{self.base_url}{path}", params=params, timeout=120)

    def _post(self, path: str, json: Optional[dict] = None, files: Optional[dict] = None):
        return requests.post(f"{self.base_url}{path}", json=json, files=files, timeout=120)

    def health(self):
        return self._get("/health").json()

    def get_models(self):
        return self._get("/models").json()

    def list_sessions(self, user_id: str = "default"):
        return self._get("/sessions", params={"user_id": user_id}).json()

    def create_session(self, user_id: str = "default", title: str = "New Chat", model: str = "claude-sonnet-4-5", mode: str = "general"):
        return self._post(
            "/sessions",
            json={"user_id": user_id, "title": title, "model": model, "mode": mode},
        ).json()

    def get_session(self, session_id: str):
        return self._get(f"/sessions/{session_id}").json()

    def update_session(self, session_id: str, **updates):
        return self._patch(f"/sessions/{session_id}", json=updates).json()

    def save_message(self, session_id: str, role: str, content: str, citations=None):
        return self._post(
            f"/sessions/{session_id}/messages",
            json={"role": role, "content": content, "citations": citations or []},
        ).json()

    def upload_file(self, file_path: str, user_id: str = "default", filename: Optional[str] = None):
        with open(file_path, "rb") as fh:
            return self._post(
                "/upload",
                files={"file": (filename or os.path.basename(file_path), fh, "application/octet-stream")},
                data={"user_id": user_id},
            ).json()

    def upload_url(self, url: str, user_id: str = "default"):
        return self._post(
            "/upload",
            data={"url": url, "user_id": user_id},
        ).json()

    def retrieve(self, query: str, user_id: str = "default", top_k: int = 5, threshold: float = 0.4):
        return self._post(
            "/retrieve",
            json={"query": query, "user_id": user_id, "top_k": top_k, "threshold": threshold},
        ).json()

    def search(self, query: str, max_results: int = 5):
        return self._post(
            "/search",
            json={"query": query, "max_results": max_results},
        ).json()

    def list_documents(self, user_id: str = "default"):
        return self._get("/documents", params={"user_id": user_id}).json()


# Global instance
client = BackendClient()
