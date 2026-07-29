import pytest

from rag_demo.rrf import ScoredCandidate, reciprocal_rank_fusion


def test_rrf_combines_rankings_and_preserves_branch_details() -> None:
    dense = (
        ScoredCandidate(chunk_id=10, score=0.91),
        ScoredCandidate(chunk_id=20, score=0.82),
        ScoredCandidate(chunk_id=30, score=0.73),
    )
    bm25 = (
        ScoredCandidate(chunk_id=20, score=12.4),
        ScoredCandidate(chunk_id=40, score=9.1),
        ScoredCandidate(chunk_id=10, score=7.2),
    )

    fused = reciprocal_rank_fusion(dense, bm25, rank_constant=60)

    assert [candidate.chunk_id for candidate in fused] == [20, 10, 40, 30]
    assert fused[0].dense_rank == 2
    assert fused[0].dense_score == pytest.approx(0.82)
    assert fused[0].bm25_rank == 1
    assert fused[0].bm25_score == pytest.approx(12.4)
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[2].dense_rank is None
    assert fused[3].bm25_rank is None


def test_rrf_ignores_repeated_chunk_in_the_same_branch() -> None:
    dense = (
        ScoredCandidate(chunk_id=1, score=0.9),
        ScoredCandidate(chunk_id=1, score=0.8),
        ScoredCandidate(chunk_id=2, score=0.7),
    )

    fused = reciprocal_rank_fusion(dense, (), rank_constant=60)

    assert [candidate.chunk_id for candidate in fused] == [1, 2]
    assert fused[0].dense_rank == 1
    assert fused[0].dense_score == pytest.approx(0.9)
    assert fused[1].dense_rank == 3


def test_rrf_rejects_invalid_rank_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion((), (), rank_constant=0)
