"""Reciprocal Rank Fusion for Dense and BM25 candidate lists."""

from __future__ import annotations

from dataclasses import dataclass

from rag_demo.models.retrieval import FusedCandidate, ScoredCandidate

__all__ = ["FusedCandidate", "ScoredCandidate", "reciprocal_rank_fusion"]


@dataclass(slots=True)
class _MutableFusedCandidate:
    chunk_id: int
    first_seen: int
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0


def reciprocal_rank_fusion(
    dense: tuple[ScoredCandidate, ...],
    bm25: tuple[ScoredCandidate, ...],
    *,
    rank_constant: int,
) -> tuple[FusedCandidate, ...]:
    """Fuse two ordered rankings while retaining their original evidence."""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    candidates: dict[int, _MutableFusedCandidate] = {}  # 融合分暂存
    next_seen = 0  # 用于解决同分时的排序问题

    for branch, ranking in (("dense", dense), ("bm25", bm25)):
        seen_in_branch: set[int] = set()  # 防止同一个分支内重复计算
        for rank, candidate in enumerate(ranking, start=1):
            if candidate.chunk_id in seen_in_branch:
                continue
            seen_in_branch.add(candidate.chunk_id)

            fused = candidates.get(candidate.chunk_id)
            if fused is None:
                fused = _MutableFusedCandidate(
                    chunk_id=candidate.chunk_id,
                    first_seen=next_seen,
                )
                candidates[candidate.chunk_id] = fused
                next_seen += 1

            fused.rrf_score += 1 / (rank_constant + rank)  # 纳入分数
            if branch == "dense":
                fused.dense_rank = rank
                fused.dense_score = candidate.score
            else:
                fused.bm25_rank = rank
                fused.bm25_score = candidate.score

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.rrf_score, candidate.first_seen),
    )
    return tuple(
        FusedCandidate(
            chunk_id=candidate.chunk_id,
            dense_rank=candidate.dense_rank,
            dense_score=candidate.dense_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            rrf_score=candidate.rrf_score,
        )
        for candidate in ordered
    )
