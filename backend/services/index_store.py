"""
Persistent hybrid index: SQLite (chunk text + FTS5/BM25) and per-user FAISS
(cosine via inner product on normalised vectors).

* Chunk ids are the FAISS ids, so a hit in either index resolves to one row.
* User ids never touch the filesystem directly (index files are named by hash),
  and FAISS files are loaded with `read_index` (no pickle).
* Writes are atomic: rows + vectors are committed together or rolled back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from backend.utils.text import build_fts_query

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    doc_id   TEXT NOT NULL,
    source   TEXT NOT NULL,
    page     INTEGER,
    ord      INTEGER NOT NULL DEFAULT 0,
    text     TEXT NOT NULL,
    meta     TEXT,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON chunks(user_id, doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class ChunkIn:
    text: str
    page: Optional[int] = None
    ord: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    id: int
    user_id: str
    doc_id: str
    source: str
    page: Optional[int]
    ord: int
    text: str
    meta: dict[str, Any]


class HybridIndexStore:
    def __init__(self, index_dir: Path, max_loaded_indexes: int = 32) -> None:
        self._dir = Path(index_dir)
        self._faiss_dir = self._dir / "faiss"
        self._faiss_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._indexes: "OrderedDict[str, faiss.Index]" = OrderedDict()
        self._max_loaded = max_loaded_indexes
        self._versions: dict[str, int] = {}

        self._conn = sqlite3.connect(
            str(self._dir / "rag.sqlite3"), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _index_path(self, user_id: str) -> Path:
        return self._faiss_dir / f"{hashlib.sha1(user_id.encode('utf-8')).hexdigest()[:24]}.index"

    def _load_index(self, user_id: str) -> Optional[faiss.Index]:
        if user_id in self._indexes:
            self._indexes.move_to_end(user_id)
            return self._indexes[user_id]
        path = self._index_path(user_id)
        if not path.exists():
            return None
        try:
            idx = faiss.read_index(str(path))
        except Exception as exc:  # noqa: BLE001 — corrupt file: lexical search still works
            logger.error("Could not read FAISS index for a user (%s)", exc)
            return None
        self._indexes[user_id] = idx
        while len(self._indexes) > self._max_loaded:
            self._indexes.popitem(last=False)
        return idx

    def _save_index(self, user_id: str, idx: faiss.Index) -> None:
        path = self._index_path(user_id)
        tmp = path.with_suffix(".tmp")
        faiss.write_index(idx, str(tmp))
        os.replace(tmp, path)

    def _bump(self, user_id: str) -> None:
        self._versions[user_id] = self._versions.get(user_id, 0) + 1

    def version(self, user_id: str) -> int:
        with self._lock:
            return self._versions.get(user_id, 0)

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
        meta: dict[str, Any] = {}
        if row["meta"]:
            try:
                meta = json.loads(row["meta"])
            except (TypeError, ValueError):
                meta = {}
        return ChunkRecord(row["id"], row["user_id"], row["doc_id"], row["source"],
                           row["page"], row["ord"], row["text"], meta)

    # ── writes ───────────────────────────────────────────────────────────────

    def add_chunks(self, user_id: str, doc_id: str, source: str, chunks: list[ChunkIn],
                   vectors: Optional[np.ndarray] = None, embed_model: Optional[str] = None) -> list[int]:
        """Insert chunks (and vectors, when given) atomically. Returns chunk ids."""
        if not chunks:
            return []
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        embedded = 1 if vectors is not None else 0
        with self._lock:
            ids: list[int] = []
            idx: Optional[faiss.Index] = None
            added_vectors = False
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for c in chunks:
                    cur = self._conn.execute(
                        "INSERT INTO chunks(user_id, doc_id, source, page, ord, text, meta, embedded) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (user_id, doc_id, source, c.page, c.ord, c.text,
                         json.dumps(c.meta) if c.meta else None, embedded),
                    )
                    ids.append(int(cur.lastrowid))
                if vectors is not None:
                    vecs = np.ascontiguousarray(vectors, dtype=np.float32)
                    idx = self._load_index(user_id)
                    if idx is None:
                        idx = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
                        self._indexes[user_id] = idx
                    if idx.d != vecs.shape[1]:
                        raise ValueError(f"embedding dim {vecs.shape[1]} != index dim {idx.d}")
                    idx.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
                    added_vectors = True
                    self._save_index(user_id, idx)
                    if embed_model:
                        self._set_meta("embed_model", embed_model)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                if added_vectors and idx is not None:
                    idx.remove_ids(np.asarray(ids, dtype=np.int64))
                    self._indexes.pop(user_id, None)    # force reload from last good file
                raise
            self._bump(user_id)
            return ids

    def delete_document(self, user_id: str, doc_id: str) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM chunks WHERE user_id=? AND doc_id=?", (user_id, doc_id)).fetchall()
            ids = [int(r["id"]) for r in rows]
            if not ids:
                return 0
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM chunks WHERE user_id=? AND doc_id=?", (user_id, doc_id))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            idx = self._load_index(user_id)
            if idx is not None:
                idx.remove_ids(np.asarray(ids, dtype=np.int64))
                self._save_index(user_id, idx)
            self._bump(user_id)
            return len(ids)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE user_id=?", (user_id,))
            self._indexes.pop(user_id, None)
            path = self._index_path(user_id)
            if path.exists():
                path.unlink()
            self._bump(user_id)

    # ── reads ────────────────────────────────────────────────────────────────

    def dense_search(self, user_id: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            idx = self._load_index(user_id)
            if idx is None or idx.ntotal == 0:
                return []
            q = np.ascontiguousarray(np.asarray(qvec, dtype=np.float32).reshape(1, -1))
            if q.shape[1] != idx.d:
                logger.warning("Query dim %s != index dim %s (reindex pending?)", q.shape[1], idx.d)
                return []
            scores, ids = idx.search(q, min(k, idx.ntotal))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def cosines(self, user_id: str, qvec: np.ndarray, ids: Iterable[int]) -> dict[int, float]:
        with self._lock:
            idx = self._load_index(user_id)
            if idx is None:
                return {}
            q = np.asarray(qvec, dtype=np.float32).reshape(-1)
            out: dict[int, float] = {}
            for i in ids:
                try:
                    out[int(i)] = float(np.dot(q, idx.reconstruct(int(i))))
                except RuntimeError:        # id has no vector (lexical-only chunk)
                    continue
            return out

    def lexical_search(self, user_id: str, query: str, k: int) -> list[tuple[int, float]]:
        match = build_fts_query(query)
        if not match:
            return []
        sql = (
            "SELECT c.id AS id, bm25(chunks_fts) AS s FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.user_id = ? ORDER BY s LIMIT ?"
        )
        with self._lock:
            try:
                rows = self._conn.execute(sql, (match, user_id, k)).fetchall()
            except sqlite3.OperationalError as exc:
                logger.warning("FTS query failed (%s) for %r", exc, match)
                return []
        return [(int(r["id"]), -float(r["s"])) for r in rows]   # bm25() is lower-is-better

    def get_chunks(self, ids: Iterable[int]) -> dict[int, ChunkRecord]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", ids).fetchall()
        return {int(r["id"]): self._row_to_chunk(r) for r in rows}

    def count(self, user_id: str) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE user_id=?", (user_id,)).fetchone()[0])

    def has_documents(self, user_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM chunks WHERE user_id=? LIMIT 1", (user_id,)).fetchone() is not None

    def list_sources(self, user_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT source FROM chunks WHERE user_id=? ORDER BY source", (user_id,)).fetchall()
        return [r["source"] for r in rows]

    def doc_chunk_count(self, user_id: str, doc_id: str) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE user_id=? AND doc_id=?", (user_id, doc_id)).fetchone()[0])

    # ── embedding-model bookkeeping / reindex ────────────────────────────────

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO index_meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def embed_model(self) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM index_meta WHERE key='embed_model'").fetchone()
        return row["value"] if row else None

    def users(self) -> list[str]:
        with self._lock:
            return [r["user_id"] for r in self._conn.execute("SELECT DISTINCT user_id FROM chunks")]

    def chunk_texts(self, user_id: str, only_unembedded: bool = False) -> list[tuple[int, str]]:
        sql = "SELECT id, text FROM chunks WHERE user_id=?"
        if only_unembedded:
            sql += " AND embedded=0"
        with self._lock:
            return [(int(r["id"]), r["text"]) for r in self._conn.execute(sql + " ORDER BY id", (user_id,))]

    def set_vectors(self, user_id: str, ids: list[int], vectors: np.ndarray,
                    embed_model: str, rebuild: bool = False) -> None:
        """Attach vectors to existing chunks; `rebuild` replaces the user's index."""
        if not ids:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        with self._lock:
            idx = None if rebuild else self._load_index(user_id)
            if idx is None:
                idx = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
            elif idx.d != vecs.shape[1]:
                idx = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
            arr = np.asarray(ids, dtype=np.int64)
            idx.remove_ids(arr)                         # idempotent re-embedding
            idx.add_with_ids(vecs, arr)
            self._indexes[user_id] = idx
            self._save_index(user_id, idx)
            marks = ",".join("?" * len(ids))
            self._conn.execute(f"UPDATE chunks SET embedded=1 WHERE id IN ({marks})", ids)
            self._set_meta("embed_model", embed_model)
            self._bump(user_id)

    def mark_all_unembedded(self, user_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE chunks SET embedded=0 WHERE user_id=?", (user_id,))
            self._indexes.pop(user_id, None)
            path = self._index_path(user_id)
            if path.exists():
                path.unlink()
            self._bump(user_id)

    def unembedded_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded=0").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return {"chunks": total, "loaded_indexes": len(self._indexes),
                "embed_model": self.embed_model(), "unembedded": self.unembedded_count()}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
