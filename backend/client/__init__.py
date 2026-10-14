"""Adapters to external systems (LLM providers, Tavily, embedding/rerank backends, web pages).

Clients own transport concerns (HTTP, SDKs, retries, error normalisation). Policy such as
fallback chains, caching and gating lives in `backend.services`.
"""
