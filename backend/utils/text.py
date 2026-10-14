"""Text helpers: token estimates, query normalisation and lexical query building."""

from __future__ import annotations

import re
from typing import Optional

CHARS_PER_TOKEN = 3          # conservative: budgets are expressed in tokens

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is it its of on or "
    "our so such than that the their then there these they this to was we were what when "
    "where which who whom why will with you your do does did can could should would about "
    "tell me give show please explain".split()
)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def normalize_query(text: str) -> str:
    """Lower-cased, whitespace-collapsed form used as a cache key."""
    return " ".join(text.lower().split())


def query_terms(text: str, max_terms: int = 16) -> list[str]:
    """Distinct content terms of a query (lower-cased, stop-words removed)."""
    seen: set[str] = set()
    terms: list[str] = []
    for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        if tok in seen or tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def build_fts_query(text: str, max_terms: int = 16) -> Optional[str]:
    """Turn free text into a safe FTS5 OR-query (quoted terms, stop-words removed)."""
    terms = query_terms(text, max_terms)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def term_coverage(terms: list[str], text: str) -> float:
    """Fraction of query terms present in `text` (crude 6-char stemming)."""
    if not terms:
        return 0.0
    stems = {tok[:6] for tok in re.findall(r"\w+", text.lower(), flags=re.UNICODE)}
    return sum(1 for t in terms if t[:6] in stems) / len(terms)
