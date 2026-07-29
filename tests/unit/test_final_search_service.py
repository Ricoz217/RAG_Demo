from collections.abc import Sequence

import pytest

from rag_demo.bm25_retriever import BM25SearchResponse, BM25SearchResult
from rag_demo.dense_retriever import (
    DenseQueryResponse,
    DenseSearchMode,
    DenseSearchResult,
)
from rag_demo.rrf import ScoredCandidate, reciprocal_rank_fusion
from rag_demo.search_service import (
    HybridRecallResponse,
    HybridSearchService,
    SearchRequest,
)


def _dense(rank: int, chunk_id: int, score: float) -> DenseSearchResult:
    return DenseSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=f"文档 {chunk_id}",
        source_repo="https://example.test",
        source_commit="commit",
        source_path=f"docs/{chunk_id}.md",
        language="zh",
        heading_path=("章节", str(chunk_id)),
        content_raw=f"正文 {chunk_id}",
        retrieval_text=f"检索文本 {chunk_id}",
        metadata={"chunk": chunk_id},
        cosine_distance=1 - score,
        cosine_similarity=score,
    )


def _bm25(rank: int, chunk_id: int, score: float) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=f"文档 {chunk_id}",
        source_repo="https://example.test",
        source_commit="commit",
        source_path=f"docs/{chunk_id}.md",
        language="zh",
        heading_path=("章节", str(chunk_id)),
        content_raw=f"正文 {chunk_id}",
        retrieval_text=f"检索文本 {chunk_id}",
        metadata={"chunk": chunk_id},
        score=score,
    )


def _recall_response() -> HybridRecallResponse:
    dense_results = (
        _dense(1, 10, 0.9),
        _dense(2, 20, 0.8),
        _dense(3, 30, 0.7),
    )
    bm25_results = (
        _bm25(1, 20, 12.0),
        _bm25(2, 30, 9.0),
        _bm25(3, 40, 7.0),
    )
    fused = reciprocal_rank_fusion(
        tuple(ScoredCandidate(item.chunk_id, item.cosine_similarity) for item in dense_results),
        tuple(ScoredCandidate(item.chunk_id, item.score) for item in bm25_results),
        rank_constant=60,
    )
    return HybridRecallResponse(
        query="测试查询",
        dense=DenseQueryResponse(
            query="测试查询",
            mode=DenseSearchMode.HNSW,
            results=dense_results,
            query_embedding_ms=1.0,
            dense_search_ms=2.0,
            total_ms=3.0,
        ),
        bm25=BM25SearchResponse(results=bm25_results, search_ms=1.5),
        fused=fused,
        rrf_ms=0.2,
        total_ms=3.5,
    )


class FakeRecallService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def recall(
        self,
        query: str,
        *,
        dense_top_k: int,
        bm25_top_k: int,
        rank_constant: int,
        dense_mode: DenseSearchMode,
    ) -> HybridRecallResponse:
        self.calls.append((query, dense_top_k, bm25_top_k, rank_constant, dense_mode))
        return _recall_response()


class FakeReranker:
    model = "test-reranker"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[float, ...]:
        self.calls.append((query, tuple(documents)))
        return (0.1, 0.9, 0.2)


def _request(*, use_reranker: bool) -> SearchRequest:
    return SearchRequest(
        query="测试查询",
        dense_top_k=3,
        bm25_top_k=3,
        rrf_rank_constant=60,
        rerank_top_k=3,
        final_top_k=2,
        dense_mode=DenseSearchMode.HNSW,
        use_reranker=use_reranker,
        debug=True,
    )


@pytest.mark.asyncio
async def test_hybrid_search_reranks_rrf_top_k_and_preserves_all_scores() -> None:
    recall = FakeRecallService()
    reranker = FakeReranker()
    service = HybridSearchService(
        recall_service=recall,
        reranker_client=reranker,
    )

    response = await service.search(_request(use_reranker=True))

    assert reranker.calls == [
        (
            "测试查询",
            ("检索文本 20", "检索文本 30", "检索文本 10"),
        )
    ]
    assert [item.chunk_id for item in response.reranked] == [30, 10, 20]
    assert [item.chunk_id for item in response.results] == [30, 10]
    assert [item.final_rank for item in response.results] == [1, 2]
    assert response.results[0].dense_rank == 3
    assert response.results[0].bm25_rank == 2
    assert response.results[0].rrf_rank == 2
    assert response.results[0].rerank_score == pytest.approx(0.9)
    assert response.timings.query_embedding_ms == pytest.approx(1.0)
    assert response.timings.dense_search_ms == pytest.approx(2.0)
    assert response.timings.bm25_search_ms == pytest.approx(1.5)
    assert response.timings.rrf_ms == pytest.approx(0.2)
    assert response.timings.rerank_ms >= 0
    assert response.counts.dense_candidate_count == 3
    assert response.counts.bm25_candidate_count == 3
    assert response.counts.union_candidate_count == 4
    assert response.reranker_used is True


@pytest.mark.asyncio
async def test_hybrid_search_can_skip_reranker_and_keep_rrf_order() -> None:
    recall = FakeRecallService()
    reranker = FakeReranker()
    service = HybridSearchService(
        recall_service=recall,
        reranker_client=reranker,
    )

    response = await service.search(_request(use_reranker=False))

    assert reranker.calls == []
    assert response.reranked == ()
    assert [item.chunk_id for item in response.results] == [20, 30]
    assert all(item.rerank_score is None for item in response.results)
    assert response.reranker_used is False
    assert response.timings.rerank_ms == 0


def test_search_request_rejects_inconsistent_candidate_limits() -> None:
    with pytest.raises(ValueError, match="final_top_k"):
        SearchRequest(
            query="query",
            dense_top_k=3,
            bm25_top_k=3,
            rrf_rank_constant=60,
            rerank_top_k=2,
            final_top_k=3,
        )
