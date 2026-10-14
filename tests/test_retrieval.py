"""Hybrid index + retrieval + ingestion behaviour (offline, deterministic fakes)."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from backend.core.exceptions import IngestError
from backend.services.index_store import ChunkIn, HybridIndexStore
from tests.conftest import FakeEmbedder, FakeReranker

HANDBOOK = (
    "Refund policy: Enterprise customers may request a full refund within 30 days of purchase.\n\n"
    "Support hours: Our support team is available Monday through Friday from 9am to 6pm Pacific.\n\n"
    "Security: All customer data is encrypted at rest using AES-256 and TLS 1.3 in transit.\n\n"
    "Parental leave: Employees are entitled to 16 weeks of paid parental leave.\n\n"
    + ("Filler paragraph about nothing in particular. " * 20)
)


async def ingest(c, user="u1", name="handbook.txt", text=HANDBOOK):
    return await c.documents.ingest_file(user, name, text.encode())


async def test_ingest_search_and_isolation(build_container):
    c = build_container()
    res = await ingest(c)
    assert res.indexing == "hybrid" and res.chunk_count >= 2 and res.file_type == "txt"
    hit = await c.retrieval.search("u1", "enterprise refund within 30 days")
    assert hit.used_rag and "Refund policy" in hit.items[0].text and hit.items[0].source == "handbook.txt"
    assert hit.mode == "hybrid" and hit.reranked
    other = await c.retrieval.search("u2", "enterprise refund within 30 days")
    assert not other.used_rag and other.items == []                  # users are isolated


async def test_lexical_and_dense_both_contribute(build_container):
    c = build_container(reranker=FakeReranker(enabled=False))
    await ingest(c)
    # exact rare token only BM25 can key on
    exact = await c.retrieval.search("u1", "AES-256", min_relevance=0.0)
    assert exact.items and "AES-256" in exact.items[0].text
    # synonym-only query: no shared surface term, the (synonym-aware) dense side must find it
    para = await c.retrieval.search("u1", "reimbursement vacation", min_relevance=0.05)
    assert para.items


async def test_irrelevant_query_returns_nothing(build_container):
    c = build_container()
    await ingest(c)
    res = await c.retrieval.search("u1", "how to bake sourdough bread")
    assert not res.used_rag


async def test_delete_document_removes_dense_and_lexical(build_container):
    c = build_container()
    res = await ingest(c)
    assert c.store.count("u1") > 0
    assert await c.documents.delete_document("u1", res.document_id)
    assert c.store.count("u1") == 0 and not c.store.has_documents("u1")
    assert c.store.lexical_search("u1", "refund", 5) == []
    assert c.store.dense_search("u1", c.embedder.embed_query("refund"), 5) == []
    assert not await c.documents.delete_document("u1", res.document_id)      # already gone
    assert not (await c.retrieval.search("u1", "refund")).used_rag


async def test_delete_is_scoped_to_owner(build_container):
    c = build_container()
    res = await ingest(c, user="owner")
    assert not await c.documents.delete_document("intruder", res.document_id)
    assert c.store.count("owner") > 0


async def test_reupload_replaces_previous_copy(build_container):
    c = build_container()
    await ingest(c, text="Alpha policy about widgets. " * 30)
    first = c.store.count("u1")
    await ingest(c, text="Beta policy about gadgets. " * 30)
    docs = c.documents.list_documents("u1")
    assert len(docs) == 1 and c.store.count("u1") <= first + 1
    assert not (await c.retrieval.search("u1", "widgets", min_relevance=0.3)).used_rag


async def test_cache_invalidated_by_index_change(build_container):
    c = build_container()
    await ingest(c)
    a = await c.retrieval.search("u1", "refund policy enterprise")
    b = await c.retrieval.search("u1", "refund policy enterprise")
    assert b.cached and not a.cached
    await ingest(c, name="second.txt", text="Refund addendum: refunds for training courses are 7 days. " * 5)
    assert not (await c.retrieval.search("u1", "refund policy enterprise")).cached


async def test_user_id_cannot_escape_index_dir(build_container, tmp_path):
    c = build_container()
    evil = "../../../etc/passwd"
    await ingest(c, user=evil)
    root = c.settings.index_dir
    files = [p for p in root.rglob("*") if p.is_file()]
    assert all(root in p.parents for p in files)
    assert not (tmp_path / "etc").exists()
    faiss_name = hashlib.sha1(evil.encode()).hexdigest()[:24] + ".index"
    assert (root / "faiss" / faiss_name).exists()


async def test_embedder_failure_degrades_to_lexical_only(build_container):
    c = build_container(embedder=FakeEmbedder(fail=True))
    res = await ingest(c)
    assert res.indexing == "lexical"
    hit = await c.retrieval.search("u1", "enterprise refund")
    assert hit.used_rag and hit.mode == "lexical"
    junk = await c.retrieval.search("u1", "sourdough bread recipe")
    assert not junk.used_rag


async def test_reranker_failure_falls_back_to_fused_order(build_container):
    c = build_container(reranker=FakeReranker(fail=True))
    await ingest(c)
    hit = await c.retrieval.search("u1", "enterprise refund within 30 days")
    assert hit.used_rag and hit.reranked is False and "Refund" in hit.items[0].text


async def test_reindex_after_embedding_model_change(build_container):
    c = build_container()
    await ingest(c)
    assert c.store.embed_model() == "fake:bow-v1"
    c.embedder.model_id = "fake:bow-v2"                     # operator switched models
    assert await c.documents.reindex_if_needed() > 0
    assert c.store.embed_model() == "fake:bow-v2" and c.store.unembedded_count() == 0
    assert (await c.retrieval.search("u1", "enterprise refund")).used_rag


async def test_lexical_only_chunks_get_embedded_on_recovery(build_container):
    emb = FakeEmbedder(fail=True)
    c = build_container(embedder=emb)
    await ingest(c)
    assert c.store.unembedded_count() > 0
    emb.fail = False                                       # provider recovered
    assert await c.documents.reindex_if_needed() > 0 and c.store.unembedded_count() == 0
    assert (await c.retrieval.search("u1", "enterprise refund policy")).mode == "hybrid"


@pytest.mark.parametrize("data,name,status", [
    (b"", "empty.txt", 400),
    (b"x" * (21 * 1024 * 1024), "big.txt", 413),
    (b"hello world this is text", "tool.exe", 415),
    (b"\x00\x01\x02", "blank.txt", 422),
    (b"%PDF-1.4 broken", "broken.pdf", 422),
])
async def test_ingest_rejects_bad_files(build_container, data, name, status):
    c = build_container()
    with pytest.raises(IngestError) as ei:
        await c.documents.ingest_file("u1", name, data)
    assert ei.value.status_code == status


async def test_concurrent_ingest_and_queries_are_consistent(build_container):
    c = build_container()
    texts = [f"Topic{i}: unique-token-{i} details. " * 40 for i in range(6)]
    await asyncio.gather(*(ingest(c, name=f"d{i}.txt", text=t) for i, t in enumerate(texts)))
    assert len(c.documents.list_documents("u1")) == 6
    results = await asyncio.gather(*(c.retrieval.search("u1", f"unique-token-{i}", min_relevance=0.0)
                                     for i in range(6)))
    for i, r in enumerate(results):
        assert r.items and f"unique-token-{i}" in r.items[0].text


def test_index_store_atomic_rollback_on_dim_mismatch(tmp_path):
    import numpy as np

    store = HybridIndexStore(tmp_path)
    store.add_chunks("u", "d1", "a.txt", [ChunkIn("alpha beta gamma delta")], np.ones((1, 8), np.float32), "m")
    with pytest.raises(ValueError):
        store.add_chunks("u", "d2", "b.txt", [ChunkIn("epsilon zeta eta theta")], np.ones((1, 4), np.float32), "m")
    assert store.count("u") == 1                              # failed insert rolled back, nothing orphaned
    assert store.lexical_search("u", "epsilon", 5) == []
    store.close()
