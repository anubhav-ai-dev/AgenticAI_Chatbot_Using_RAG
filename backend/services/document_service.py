"""
Document ingestion: parse → chunk → embed → index (+ DB record), deletion,
listing and embedding-model reindexing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import Settings
from backend.core.database import Database
from backend.core.exceptions import DocumentParseError, IngestError
from backend.models.db import Document as DBDocument
from backend.services.embedding_service import EmbeddingService
from backend.services.index_store import ChunkIn, HybridIndexStore
from backend.services.parser import DocumentParser

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOCUMENT = 5000
MIN_CHUNK_CHARS = 20


@dataclass
class IngestResult:
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    indexing: str          # "hybrid" (vectors + BM25) | "lexical" (BM25 only)


class DocumentService:
    def __init__(self, settings: Settings, db: Database, parser: DocumentParser,
                 embedder: EmbeddingService, store: HybridIndexStore) -> None:
        self._settings = settings
        self._db = db
        self._parser = parser
        self._embedder = embedder
        self._store = store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── ingestion ────────────────────────────────────────────────────────────

    async def ingest_file(self, user_id: str, filename: str, data: bytes) -> IngestResult:
        if not data:
            raise IngestError("The uploaded file is empty.", 400)
        if len(data) > self._settings.max_upload_mb * 1024 * 1024:
            raise IngestError(f"File exceeds the {self._settings.max_upload_mb} MB limit.", 413)
        safe_name = Path(filename or "upload").name
        ext = Path(safe_name).suffix.lower().lstrip(".")

        def _parse() -> list[dict[str, Any]]:
            fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}" if ext else "")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                return self._parser.parse_file(tmp_path, safe_name)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        t0 = time.perf_counter()
        try:
            pages = await asyncio.to_thread(_parse)
        except DocumentParseError as exc:
            raise IngestError(str(exc), 415 if "Unsupported file type" in str(exc) else 422) from exc
        return await self._index_pages(user_id, safe_name, ext or "bin", pages,
                                       {"parse": time.perf_counter() - t0})

    async def ingest_url(self, user_id: str, url: str) -> IngestResult:
        try:
            pages = await asyncio.to_thread(self._parser.parse_url, url)
        except DocumentParseError as exc:
            raise IngestError(str(exc), 422) from exc
        return await self._index_pages(user_id, url, "url", pages)

    def _chunk(self, pages: list[dict[str, Any]]) -> list[ChunkIn]:
        chunks: list[ChunkIn] = []
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            meta = page.get("metadata") or {}
            for piece in self._splitter.split_text(text):
                piece = piece.strip()
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(ChunkIn(text=piece, page=page.get("page_number"),
                                      ord=len(chunks), meta=meta))
        return chunks

    async def _index_pages(self, user_id: str, source: str, file_type: str,
                           pages: list[dict[str, Any]],
                           timings: Optional[dict[str, float]] = None) -> IngestResult:
        timings = timings if timings is not None else {}
        started = time.perf_counter()
        t = started
        chunks = await asyncio.to_thread(self._chunk, pages)
        timings["chunk"], t = time.perf_counter() - t, time.perf_counter()
        if not chunks:
            raise IngestError(f"No readable text could be extracted from '{source}'.", 422)
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise IngestError(
                f"Document is too large ({len(chunks)} chunks; limit {MAX_CHUNKS_PER_DOCUMENT}).", 413)

        doc_id = str(uuid.uuid4())
        vectors, indexing = await self._embed(user_id, [c.text for c in chunks])
        timings["embed"], t = time.perf_counter() - t, time.perf_counter()

        try:
            await asyncio.to_thread(
                self._store.add_chunks, user_id, doc_id, source, chunks,
                vectors, self._embedder.model_id if vectors is not None else None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Index write failed for %s", source)
            raise IngestError("Failed to build the search index for this document.", 500) from exc
        timings["index"], t = time.perf_counter() - t, time.perf_counter()

        try:
            replaced = await asyncio.to_thread(self._record_document, user_id, doc_id, source,
                                               file_type, len(chunks))
        except Exception as exc:  # noqa: BLE001 — keep DB and index consistent
            await asyncio.to_thread(self._store.delete_document, user_id, doc_id)
            logger.exception("Document DB write failed for %s", source)
            raise IngestError("Failed to record the document.", 500) from exc

        for old_id in replaced:                       # re-upload replaces the previous copy
            await asyncio.to_thread(self._store.delete_document, user_id, old_id)
        timings["record"] = time.perf_counter() - t
        timings["total"] = sum(timings.values())
        logger.info("Upload timings %s: chunks=%d pages=%d indexing=%s %s", file_type, len(chunks),
                    len(pages), indexing, " ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
        return IngestResult(doc_id, source, file_type, len(chunks), indexing)

    async def _embed(self, user_id: str, texts: list[str]):
        """Embed chunk texts; any failure degrades this document to lexical-only."""
        if not await self._embedder.warmup():
            return None, "lexical"
        try:
            await asyncio.to_thread(self._ensure_index_current)
            vectors = await self._embedder.aembed_documents(texts)
            return vectors, "hybrid"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding failed for user upload (%s) — indexing lexically", exc)
            return None, "lexical"

    def _record_document(self, user_id: str, doc_id: str, filename: str,
                         file_type: str, chunk_count: int) -> list[str]:
        """Insert the DB row; returns ids of earlier uploads with the same filename."""
        with self._db.session() as s:
            old = (s.query(DBDocument)
                   .filter(DBDocument.user_id == user_id, DBDocument.filename == filename).all())
            old_ids = [d.id for d in old]
            for d in old:
                s.delete(d)
            s.add(DBDocument(id=doc_id, user_id=user_id, filename=filename,
                             file_type=file_type, chunk_count=chunk_count))
        return old_ids

    # ── management ───────────────────────────────────────────────────────────

    def list_documents(self, user_id: str) -> list[DBDocument]:
        with self._db.session() as s:
            return (s.query(DBDocument).filter(DBDocument.user_id == user_id)
                    .order_by(DBDocument.indexed_at.desc()).all())

    async def delete_document(self, user_id: str, doc_id: str) -> bool:
        def _delete_row() -> bool:
            with self._db.session() as s:
                row = (s.query(DBDocument)
                       .filter(DBDocument.id == doc_id, DBDocument.user_id == user_id).first())
                if row is None:
                    return False
                s.delete(row)
                return True

        existed = await asyncio.to_thread(_delete_row)
        if existed:
            await asyncio.to_thread(self._store.delete_document, user_id, doc_id)
        return existed

    # ── embedding-model reindex ──────────────────────────────────────────────

    def _ensure_index_current(self) -> None:
        stored = self._store.embed_model()
        if stored is not None and stored != self._embedder.model_id:
            logger.info("Embedding model changed (%s → %s): reindexing", stored, self._embedder.model_id)
            self.reindex()

    def reindex(self) -> int:
        """Blocking. Re-embeds chunks lacking vectors (all of them if the model changed)."""
        if not self._embedder.ready and self._embedder.dim is None:
            self._embedder.embed_query("warmup")       # may raise EmbeddingUnavailable
        current = self._embedder.model_id
        stored = self._store.embed_model()
        changed = stored is not None and stored != current
        total = 0
        for user_id in self._store.users():
            if changed:
                self._store.mark_all_unembedded(user_id)
            pending = self._store.chunk_texts(user_id, only_unembedded=True)
            if not pending:
                continue
            ids = [i for i, _ in pending]
            vectors = self._embedder.embed_documents([t for _, t in pending])
            self._store.set_vectors(user_id, ids, vectors, current, rebuild=changed)
            total += len(ids)
        if total:
            logger.info("Reindexed %d chunks", total)
        return total

    async def reindex_if_needed(self) -> int:
        """Startup hook: embeds lexical-only chunks / re-embeds after a model switch."""
        stored = self._store.embed_model()
        changed = stored is not None and stored != self._embedder.model_id
        if not changed and self._store.unembedded_count() == 0:
            return 0
        if not await self._embedder.warmup():
            return 0
        try:
            return await asyncio.to_thread(self.reindex)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reindex skipped: %s", exc)
            return 0
