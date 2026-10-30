"""
Connect a GitHub repository, index its source into the existing RAG store,
and keep a repo brief so the coding assistant does not need a recap each turn.

Tokens are stored per user and never returned to clients or written to logs.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Any, Optional
from urllib.parse import quote

import httpx

from backend.core.config import Settings
from backend.core.database import Database
from backend.core.exceptions import IngestError
from backend.models.db import GithubCredential, RepoContext
from backend.services.document_service import DocumentService
from backend.services.llm_service import LLMService
from backend.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

API = "https://api.github.com"
OAUTH = "https://github.com/login"
MAX_FILES = 150
MAX_FILE_BYTES = 400_000
SKIP_DIRS = {
    "node_modules", "dist", "vendor", ".git", "build", "__pycache__", ".venv", "venv",
    "target", "coverage", ".next", "out", "bin", "obj", ".idea", ".vscode",
}
SOURCE_EXT = {
    "py", "js", "ts", "tsx", "jsx", "mjs", "cjs", "go", "rs", "java", "kt", "rb", "php",
    "c", "h", "cpp", "cc", "hpp", "cs", "swift", "html", "css", "scss", "vue", "svelte",
    "json", "yml", "yaml", "toml", "md", "rst", "txt", "sh", "bash", "sql", "xml",
    "gradle", "kts", "dockerfile",
}
PRIORITY_NAMES = {
    "readme", "readme.md", "readme.rst", "package.json", "pyproject.toml",
    "requirements.txt", "cargo.toml", "go.mod", "dockerfile", "composer.json",
    "pom.xml", "makefile", "gemfile",
}
_REPO_URL = re.compile(
    r"(?:https?://github\.com/|git@github\.com:)(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?(?:/|$)",
    re.IGNORECASE,
)


class GitHubError(IngestError):
    """User-facing GitHub failure."""


class GitHubService:
    def __init__(self, settings: Settings, db: Database, http: httpx.AsyncClient,
                 documents: DocumentService, llm: LLMService) -> None:
        self._s = settings
        self._db = db
        self._http = http
        self._documents = documents
        self._llm = llm

    @property
    def oauth_configured(self) -> bool:
        return bool(self._s.github_client_id)

    def combined_brief(self, user_id: str) -> str:
        with self._db.session() as session:
            rows = (session.query(RepoContext)
                    .filter(RepoContext.user_id == user_id)
                    .order_by(RepoContext.connected_at.desc())
                    .all())
            parts = [f"### {r.owner}/{r.repo} ({r.branch})\n{r.brief}" for r in rows if r.brief]
        return "\n\n".join(parts)[:8000]

    def list_repos(self, user_id: str) -> list[RepoContext]:
        with self._db.session() as session:
            rows = (session.query(RepoContext)
                    .filter(RepoContext.user_id == user_id)
                    .order_by(RepoContext.connected_at.desc())
                    .all())
            for row in rows:
                session.expunge(row)
            return rows

    async def delete_repo(self, user_id: str, repo_id: str) -> bool:
        def _drop() -> Optional[tuple[str, str]]:
            with self._db.session() as session:
                row = (session.query(RepoContext)
                       .filter(RepoContext.id == repo_id, RepoContext.user_id == user_id)
                       .first())
                if row is None:
                    return None
                owner, repo = row.owner, row.repo
                session.delete(row)
                return owner, repo

        pair = await asyncio.to_thread(_drop)
        if pair is None:
            return False
        prefix = f"{pair[0]}/{pair[1]}/"
        docs = await asyncio.to_thread(self._documents.list_documents, user_id)
        for doc in docs:
            if doc.file_type == "github" and doc.filename.startswith(prefix):
                await self._documents.delete_document(user_id, doc.id)
        return True

    async def connect(self, user_id: str, url: str, token: Optional[str] = None) -> RepoContext:
        owner, repo = parse_repo_url(url)
        auth = (token or "").strip() or self._load_token(user_id)
        if token and token.strip():
            await asyncio.to_thread(self._store_token, user_id, token.strip(), "pat")
        info = await self._get(f"/repos/{owner}/{repo}", auth)
        branch = str(info.get("default_branch") or "main")
        tree = await self._get(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", auth)
        paths = _select_paths(tree.get("tree") or [])
        indexed = 0
        samples: list[str] = []
        for path in paths:
            try:
                body = await self._file_text(owner, repo, path, auth)
            except GitHubError:
                continue
            if not body.strip():
                continue
            labeled = f"# {path}\n{body}"
            try:
                await self._documents.ingest_text(
                    user_id, f"{owner}/{repo}/{path}", labeled, "github")
            except IngestError:
                continue
            indexed += 1
            if len(samples) < 12:
                samples.append(labeled[:1500])
        if indexed == 0:
            raise GitHubError("No readable source files were found in that repository.", 422)
        brief = await self._brief(owner, repo, branch, paths, samples)
        return await asyncio.to_thread(self._upsert_context, user_id, owner, repo, branch,
                                       brief, indexed)

    async def start_device_flow(self) -> dict[str, Any]:
        if not self._s.github_client_id:
            raise GitHubError("GitHub OAuth is not configured on this server.", 501)
        resp = await self._http.post(
            f"{OAUTH}/device/code",
            data={"client_id": self._s.github_client_id, "scope": "repo read:user"},
            headers={"Accept": "application/json"},
            timeout=20.0,
        )
        if resp.status_code >= 400:
            raise GitHubError("Could not start GitHub sign-in.", 502)
        data = resp.json()
        if not data.get("device_code") or not data.get("user_code"):
            raise GitHubError("Could not start GitHub sign-in.", 502)
        return {
            "user_code": data["user_code"],
            "verification_uri": data.get("verification_uri") or "https://github.com/login/device",
            "expires_in": int(data.get("expires_in") or 900),
            "interval": int(data.get("interval") or 5),
            "device_code": data["device_code"],
        }

    async def poll_device_flow(self, user_id: str, device_code: str) -> dict[str, str]:
        if not self._s.github_client_id:
            raise GitHubError("GitHub OAuth is not configured on this server.", 501)
        payload = {
            "client_id": self._s.github_client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        if self._s.github_client_secret:
            payload["client_secret"] = self._s.github_client_secret
        resp = await self._http.post(
            f"{OAUTH}/oauth/access_token", data=payload,
            headers={"Accept": "application/json"}, timeout=20.0)
        data = {}
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            data = {}
        err = data.get("error")
        if err == "authorization_pending":
            return {"status": "pending"}
        if err == "slow_down":
            return {"status": "slow_down"}
        if err == "expired_token":
            raise GitHubError("The GitHub sign-in code expired. Start again.", 410)
        if err or not data.get("access_token"):
            raise GitHubError("GitHub sign-in did not complete.", 400)
        token = str(data["access_token"])
        await asyncio.to_thread(self._store_token, user_id, token, "oauth")
        return {"status": "connected"}

    async def _brief(self, owner: str, repo: str, branch: str, paths: list[str],
                     samples: list[str]) -> str:
        tree = "\n".join(paths[:80])
        excerpt = "\n\n".join(samples)[:8000]
        fallback = (
            f"{owner}/{repo} on branch {branch}. {len(paths)} files indexed.\n"
            f"Layout:\n{tree}"
        )
        if not self._llm.configured:
            return fallback
        messages = [
            {"role": "system", "content": (
                "Summarize this repository for a coding assistant. Cover languages, layout, "
                "entry points, conventions, and how to change things. Be specific. "
                "400-800 words. No secrets."
            )},
            {"role": "user", "content": f"Repo {owner}/{repo} ({branch})\n\nFiles:\n{tree}\n\n{excerpt}"},
        ]
        try:
            result = await asyncio.wait_for(
                self._llm.complete(messages, fast=True, max_tokens=700, temperature=0.2), 20.0)
            text = (result.text or "").strip()
            return text or fallback
        except Exception:  # noqa: BLE001
            logger.info("repo brief generation skipped")
            return fallback

    async def _file_text(self, owner: str, repo: str, path: str, token: str) -> str:
        data = await self._get(f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}", token)
        if isinstance(data, list):
            return ""
        raw = data.get("content") or ""
        if data.get("encoding") == "base64":
            try:
                return base64.b64decode(raw).decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001
                raise GitHubError("A file in the repository could not be read.", 422) from exc
        return str(raw)

    async def _get(self, path: str, token: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "AI-Assistant-Pro",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = await self._http.get(f"{API}{path}", headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            raise GitHubError("Could not reach GitHub.", 502) from exc
        if resp.status_code == 401:
            raise GitHubError("GitHub rejected the token.", 401)
        if resp.status_code == 403:
            raise GitHubError("GitHub rate-limited or forbade this request.", 403)
        if resp.status_code == 404:
            raise GitHubError("Repository not found. Check the URL or provide a token.", 404)
        if resp.status_code >= 400:
            raise GitHubError("GitHub could not complete this request.", 502)
        return resp.json()

    def _load_token(self, user_id: str) -> str:
        with self._db.session() as session:
            row = session.get(GithubCredential, user_id)
            return row.token if row else ""

    def _store_token(self, user_id: str, token: str, source: str) -> None:
        with self._db.session() as session:
            row = session.get(GithubCredential, user_id)
            if row is None:
                session.add(GithubCredential(user_id=user_id, token=token, source=source))
            else:
                row.token = token
                row.source = source
                row.updated_at = utcnow()

    def _upsert_context(self, user_id: str, owner: str, repo: str, branch: str,
                        brief: str, file_count: int) -> RepoContext:
        with self._db.session() as session:
            row = (session.query(RepoContext)
                   .filter(RepoContext.user_id == user_id,
                           RepoContext.owner == owner, RepoContext.repo == repo)
                   .first())
            if row is None:
                row = RepoContext(user_id=user_id, owner=owner, repo=repo, branch=branch,
                                  brief=brief, file_count=file_count)
                session.add(row)
            else:
                row.branch = branch
                row.brief = brief
                row.file_count = file_count
                row.connected_at = utcnow()
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return row


def parse_repo_url(url: str) -> tuple[str, str]:
    text = (url or "").strip()
    match = _REPO_URL.search(text)
    if not match:
        raise GitHubError("Enter a GitHub repository URL like https://github.com/owner/repo.", 400)
    repo = match.group("repo").removesuffix(".git")
    return match.group("owner"), repo


def _select_paths(tree: list[dict[str, Any]]) -> list[str]:
    priority: list[str] = []
    rest: list[str] = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = str(node.get("path") or "")
        if not path or any(part in SKIP_DIRS for part in path.split("/")):
            continue
        size = int(node.get("size") or 0)
        if size > MAX_FILE_BYTES:
            continue
        name = path.rsplit("/", 1)[-1].lower()
        ext = name.rsplit(".", 1)[-1] if "." in name else name
        if name not in PRIORITY_NAMES and ext not in SOURCE_EXT:
            continue
        (priority if name in PRIORITY_NAMES else rest).append(path)
    return (priority + rest)[:MAX_FILES]
