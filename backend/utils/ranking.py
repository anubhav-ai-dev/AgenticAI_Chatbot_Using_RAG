"""Rank-fusion helpers."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[tuple[list[int], float]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked id lists. Each entry is (ids_best_first, weight). Returns (id, score) desc."""
    scores: dict[int, float] = {}
    for ids, weight in rankings:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
