"""
Tavily web search service — supports single and multi-query search.
"""

from __future__ import annotations

from backend.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH
from backend.models.api import SearchResponse, SearchResult


def search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> SearchResponse:
    """Run a single Tavily search and return structured results."""
    if not TAVILY_API_KEY:
        return SearchResponse(results=[], query=query)

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=max_results,
            include_answer=False,
        )
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in response.get("results", [])
        ]
        return SearchResponse(results=results, query=query)

    except Exception as exc:
        print(f"[search] Tavily error: {exc}")
        return SearchResponse(results=[], query=query)


def multi_search(queries: list[str], max_results_each: int = 3) -> SearchResponse:
    """
    Run multiple search queries (for deep research mode) and deduplicate by URL.
    """
    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []

    for q in queries[:4]:  # cap at 4 sub-queries to avoid rate limits
        resp = search(q, max_results=max_results_each)
        for r in resp.results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

    # Sort by score descending
    all_results.sort(key=lambda r: r.score, reverse=True)
    combined_query = " | ".join(queries)
    return SearchResponse(results=all_results[:TAVILY_MAX_RESULTS], query=combined_query)


def generate_sub_queries(query: str) -> list[str]:
    """
    Decompose a complex research question into 2-3 focused search queries.
    Uses simple heuristics — the LLM prompt in puter_client.js does the heavy lifting.
    """
    q = query.strip()
    queries = [q]

    # Add a "latest / 2025" variant for recency
    if not any(w in q.lower() for w in ("2024", "2025", "2026", "latest", "recent")):
        queries.append(f"{q} 2025")

    # Add a "how to" variant if it looks like a what/why question
    if q.lower().startswith(("what is", "what are", "explain")):
        queries.append(f"{q} examples")

    return queries[:3]
