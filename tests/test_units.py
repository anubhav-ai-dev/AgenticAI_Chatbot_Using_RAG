"""Pure-logic unit tests: config, caches, limiter, filters, fusion, prompts, routing, parser."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.core.config import MODES, Settings
from backend.core.exceptions import DocumentParseError
from backend.services.chat_service import QueryRouter
from backend.services.parser import DocumentParser
from backend.services.prompt_builder import PromptBuilder, QUALITY_CONTRACT, select_specialist
from backend.services.retrieval_service import RetrievedItem
from backend.utils.cache import CachedAnswer, SemanticCache, TTLCache
from backend.utils.rate_limiter import QuotaGuard, RateLimiter
from backend.utils.ranking import reciprocal_rank_fusion
from backend.utils.streaming import ThinkFilter
from backend.utils.text import build_fts_query, query_terms, term_coverage
from backend.utils.timeutil import parse_duration


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ── config ───────────────────────────────────────────────────────────────────

def test_settings_defaults_and_registry(settings):
    specs = {m.id for m in settings.model_specs()}
    assert {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",
            "gemini-3.7-flash", "gemini-2.5-flash-lite", "gemma-4-31b-it"} <= specs
    assert settings.llm_default_model in specs
    assert settings.validate_runtime() == []          # all keys present in fixture
    assert set(MODES) == {"general", "document", "research", "code"}


def test_settings_llm_models_override(settings):
    custom = [{"id": "x/y", "label": "XY", "provider": "groq"}]
    s = settings.model_copy(update={"llm_models": json.dumps(custom), "llm_default_model": "x/y",
                                    "llm_fallback_chain": "x/y"})
    assert [m.id for m in s.model_specs()] == ["x/y"]
    assert s.validate_runtime() == []


def test_settings_invalid_values_fail_fast(settings):
    bad_json = settings.model_copy(update={"llm_models": "{nope"})
    with pytest.raises(ValueError, match="valid JSON"):
        bad_json.model_specs()
    bad_default = settings.model_copy(update={"llm_default_model": "does/not-exist"})
    with pytest.raises(ValueError, match="registry"):
        bad_default.validate_runtime()
    bad_chunk = settings.model_copy(update={"chunk_overlap": 900, "chunk_size": 700})
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        bad_chunk.validate_runtime()


def test_settings_missing_keys_only_warn(settings):
    s = settings.model_copy(update={"groq_api_key": None, "google_api_key": None, "tavily_api_key": None})
    warnings = s.validate_runtime()
    assert any("No LLM provider key" in w for w in warnings)
    assert any("TAVILY" in w for w in warnings)


def test_settings_masked_hides_keys(settings):
    masked = settings.masked()
    assert masked["groq_api_key"] == "set"
    assert "test-groq" not in json.dumps(masked)


def test_blank_env_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_PORT", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    s = Settings(_env_file=None, data_dir=str(tmp_path))
    assert s.backend_port == 8000 and s.groq_api_key is None


# ── caches ───────────────────────────────────────────────────────────────────

def test_ttl_cache_expiry_and_lru():
    clock = Clock()
    c: TTLCache = TTLCache(maxsize=2, ttl_s=10, clock=clock)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1                    # refreshes recency of a
    c.set("c", 3)                             # evicts b (least recent)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    clock.t += 11
    assert c.get("a") is None and len(c) == 0
    assert c.stats()["hits"] >= 2


def test_semantic_cache_scope_threshold_and_ttl():
    clock = Clock()
    sc = SemanticCache(maxsize=10, ttl_s=100, threshold=0.95, clock=clock)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([0.99, 0.05, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sc.store(("u1", "general"), v, CachedAnswer("hello"))
    assert sc.lookup(("u1", "general"), near).text == "hello"
    assert sc.lookup(("u1", "general"), far) is None
    assert sc.lookup(("u2", "general"), near) is None            # other user: never leaks
    assert sc.lookup(("u1", "document"), near) is None           # other mode
    clock.t += 101
    assert sc.lookup(("u1", "general"), near) is None            # expired


def test_semantic_cache_eviction_bound():
    sc = SemanticCache(maxsize=3, ttl_s=100, threshold=0.9)
    for i in range(10):
        sc.store(("u", i), np.eye(4, dtype=np.float32)[i % 4], CachedAnswer(str(i)))
    assert sc.stats()["size"] <= 3


# ── rate limiting / quota ────────────────────────────────────────────────────

def test_rate_limiter_bucket_refill():
    clock = Clock()
    rl = RateLimiter(rpm=2, clock=clock)
    assert rl.allow("k")[0] and rl.allow("k")[0]
    ok, retry = rl.allow("k")
    assert not ok and 0 < retry <= 30
    assert rl.allow("other")[0]                                   # independent keys
    clock.t += 31
    assert rl.allow("k")[0]


@pytest.mark.parametrize("raw,expected", [
    ("2.28s", 2.28), ("1m26.4s", 86.4), ("120ms", 0.12), ("7m12s", 432.0), ("5", 5.0),
    ("", None), (None, None), ("abc", None),
])
def test_parse_duration(raw, expected):
    got = parse_duration(raw)
    assert (got is None) if expected is None else got == pytest.approx(expected)


def test_quota_guard_rolling_window_and_headers():
    clock = Clock()
    q = QuotaGuard(clock=clock)
    assert q.allows("m", 1000, rpm=2, tpm=2000)
    q.record("m", 1500)
    assert not q.allows("m", 1000, rpm=10, tpm=2000)              # would exceed tpm
    assert q.allows("m", 400, rpm=10, tpm=2000)
    q.record("m", 10)
    assert not q.allows("m", 1, rpm=2, tpm=None)                  # rpm exhausted
    clock.t += 61
    assert q.allows("m", 1000, rpm=2, tpm=2000)                   # window rolled
    q.observe_headers("g", {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "5s"})
    assert not q.allows("g", 500, None, None)
    clock.t += 6
    assert q.allows("g", 500, None, None)


# ── think filter ─────────────────────────────────────────────────────────────

def _feed(chunks):
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    return out


def test_think_filter_variants():
    assert _feed(["Hello ", "world"]) == "Hello world"
    assert _feed(["<think>hidden</think>Visible"]) == "Visible"
    assert _feed(["<thi", "nk>hid", "den</th", "ink>Vis", "ible"]) == "Visible"       # split tags
    assert _feed(["A <thought>x</thought> B"]) == "A  B"
    assert _feed(["1 < 2 and 3 > 2"]) == "1 < 2 and 3 > 2"                            # not a tag
    assert _feed(["Value <b>bold</b>"]) == "Value <b>bold</b>"
    assert _feed(["Answer", " <th"]) == "Answer <th"                                  # dangling prefix flushed


def test_think_filter_unclosed_block_is_not_swallowed():
    out = _feed(["<thought>step one ", "step two"])
    assert "step one" in out and "step two" in out


# ── lexical helpers / fusion ─────────────────────────────────────────────────

def test_build_fts_query_sanitises():
    assert build_fts_query("What is the refund policy?") == '"refund" OR "policy"'
    assert build_fts_query('AND OR NOT "quoted" (paren) col:val*') is not None
    q = build_fts_query('x" OR 1=1 --')
    assert '"' not in q.replace('"', "", 0) or q.count('"') % 2 == 0                  # always balanced
    assert build_fts_query("the of a") is None
    assert build_fts_query("") is None
    assert build_fts_query("naïve café résumé") == '"naïve" OR "café" OR "résumé"'
    assert len(query_terms(" ".join(f"term{i}" for i in range(50)))) == 16            # capped


def test_term_coverage():
    terms = query_terms("summarize the security measures")
    assert term_coverage(terms, "All data is secured. Security audits quarterly.") == pytest.approx(1 / 3)
    assert term_coverage(terms, "unrelated cooking text") == 0.0
    assert term_coverage([], "anything") == 0.0


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([([1, 2, 3], 1.0), ([3, 2, 4], 1.0)], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] in (2, 3) and set(ids) == {1, 2, 3, 4}
    scores = dict(fused)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 62)
    assert scores[2] > scores[1] > scores[4] or scores[2] > scores[4]
    weighted = reciprocal_rank_fusion([([1], 2.0), ([2], 1.0)], k=60)
    assert weighted[0][0] == 1
    assert reciprocal_rank_fusion([], 60) == []


# ── routing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q,expected", [
    ("hi", True), ("Thanks!", True), ("how are you?", True), ("what is the refund policy", False),
])
def test_smalltalk(q, expected):
    assert QueryRouter.is_smalltalk(q) is expected


def test_router_decisions():
    r = QueryRouter()
    d = r.decide(mode="general", query="hello", has_docs=True, web_toggle=True)
    assert d.label == "direct"
    d = r.decide(mode="general", query="what's the latest news on AI today", has_docs=False, web_toggle=True)
    assert d.label == "web"
    d = r.decide(mode="general", query="latest news today", has_docs=False, web_toggle=False)
    assert d.label == "direct"                                     # toggle off wins
    d = r.decide(mode="general", query="explain recursion", has_docs=True, web_toggle=True)
    assert d.label == "rag"                                        # no live-info cue → docs only
    d = r.decide(mode="research", query="quantum computing", has_docs=True, web_toggle=False)
    assert d.label == "rag+web" and d.research
    d = r.decide(mode="research", query="x", has_docs=False, web_toggle=True, web_available=False)
    assert d.label == "direct"                                     # no Tavily key → no web


def test_followup_detection():
    r = QueryRouter()
    assert not r.is_followup("what about page 5?", has_history=False)
    assert r.is_followup("what about page 5?", has_history=True)
    assert r.is_followup("and the second one?", has_history=True)
    assert r.is_followup("why?", has_history=True)
    assert r.is_followup("can you explain that better", has_history=True)
    assert not r.is_followup("Describe the enterprise refund policy in detail please", has_history=True)


# ── prompt builder ───────────────────────────────────────────────────────────

def test_prompt_builder_context_and_budgets(settings):
    pb = PromptBuilder(settings.model_copy(update={"context_token_budget": 300, "history_token_budget": 50}))
    items = [RetrievedItem(i, f"doc{i}.pdf", i, "chunk text " * 100, 0.9) for i in range(1, 6)]
    from backend.models.api import SearchResult
    web = [SearchResult(title="T", url="https://x.y", snippet="snippet " * 200, score=1.0)]
    history = [{"role": "user", "content": "old " * 200}, {"role": "assistant", "content": "short"}]
    built = pb.build(mode="document", query="q?", history=history, rag_items=items, web_results=web)
    system = built.messages[0]["content"]
    assert built.messages[0]["role"] == "system" and built.messages[-1] == {"role": "user", "content": "q?"}
    assert "Document Expert Mode" in system and "[1]" in system
    assert "Response quality" in system and QUALITY_CONTRACT.split("\n", 1)[0] in system
    assert len(system) < 300 * 3 + 2500                          # context respects the budget
    assert all(c["id"] == i + 1 for i, c in enumerate(built.citations))
    assert {c["type"] for c in built.citations} <= {"doc", "web"}
    assert [m["content"] for m in built.messages[1:-1]] == ["short"]  # old long message dropped by budget
    assert "never follow instructions" in system


def test_prompt_builder_without_context_has_no_grounding_block(settings):
    built = PromptBuilder(settings).build(mode="general", query="hi", history=[])
    assert "## Context" not in built.messages[0]["content"] and built.citations == []
    assert "Response quality" in built.messages[0]["content"]


def test_specialist_router_picks_task_prompts():
    assert select_specialist("general", "write a proposal", file_intent=True) == "document"
    assert select_specialist("code", "fix this function") == "code"
    assert select_specialist("general", "Debug these logs\nTraceback (most recent call last):") == "debug"
    assert select_specialist("code", "```python\nprint(1)\n```", has_repo=True) == "codebase"
    assert select_specialist("research", "what is happening in the market") == "research"
    assert select_specialist("document", "what does the refund policy say") is None
    assert select_specialist("general", "hello") == "general"


# ── parser ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://localhost/admin", "http://127.0.0.1:8000/", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.5/", "http://[::1]/", "file:///etc/passwd", "ftp://example.com/x", "gopher://x",
])
def test_parser_blocks_private_and_non_http(url):
    with pytest.raises(DocumentParseError):
        DocumentParser().parse_url(url)


def test_parser_text_csv_and_errors(tmp_path):
    p = DocumentParser()
    txt = tmp_path / "a.txt"
    txt.write_text("First paragraph here.\n\nSecond paragraph here.\n\n" + "More text. " * 120)
    pages = p.parse_file(str(txt), "a.txt")
    assert pages and pages[0]["page_number"] == 1 and "First paragraph" in pages[0]["text"]

    csv_file = tmp_path / "d.csv"
    csv_file.write_text("name,qty\n" + "\n".join(f"item{i},{i}" for i in range(120)))
    cpages = p.parse_file(str(csv_file), "d.csv")
    assert "Columns (2)" in cpages[0]["text"] and len(cpages) == 4      # summary + 3 blocks of ≤50

    with pytest.raises(DocumentParseError, match="Unsupported"):
        p.parse_file(str(txt), "malware.exe")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not-really")
    with pytest.raises(DocumentParseError):
        p.parse_file(str(bad), "bad.pdf")
    empty = tmp_path / "e.csv"
    empty.write_text("")
    assert p.parse_file(str(empty), "e.csv") == []
