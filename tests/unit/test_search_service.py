import asyncio

import pytest

from rag_demo.bm25_retriever import BM25SearchResponse, BM25SearchResult
from rag_demo.dense_retriever import (
    DenseQueryResponse,
    DenseSearchMode,
    DenseSearchResult,
)
from rag_demo.search_service import HybridRecallService


def _dense_result(rank: int, chunk_id: int, score: float) -> DenseSearchResult:
    return DenseSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=1,
        title="测试",
        source_repo="https://example.test",
        source_commit="commit",
        source_path=f"docs/{chunk_id}.md",
        language="zh",
        heading_path=("章节",),
        content_raw=f"dense-{chunk_id}",
        retrieval_text=f"dense-{chunk_id}",
        metadata={},
        cosine_distance=1 - score,
        cosine_similarity=score,
    )


def _bm25_result(rank: int, chunk_id: int, score: float) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=1,
        title="测试",
        source_repo="https://example.test",
        source_commit="commit",
        source_path=f"docs/{chunk_id}.md",
        language="zh",
        heading_path=("章节",),
        content_raw=f"bm25-{chunk_id}",
        retrieval_text=f"bm25-{chunk_id}",
        metadata={},
        score=score,
    )


class FakeDenseService:
    def __init__(self) -> None:
        self.started = False
        self.other: FakeBM25Retriever | None = None

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        mode: DenseSearchMode,
    ) -> DenseQueryResponse:
        self.started = True
        await asyncio.sleep(0)
        assert self.other is not None and self.other.started
        assert query == "测试查询"
        assert top_k == 2
        assert mode is DenseSearchMode.HNSW
        return DenseQueryResponse(
            query=query,
            mode=mode,
            results=(
                _dense_result(1, 10, 0.9),
                _dense_result(2, 20, 0.8),
            ),
            query_embedding_ms=1.0,
            dense_search_ms=2.0,
            total_ms=3.0,
        )


class FakeBM25Retriever:
    def __init__(self) -> None:
        self.started = False
        self.other: FakeDenseService | None = None

    async def search(self, query: str, *, top_k: int) -> BM25SearchResponse:
        self.started = True
        await asyncio.sleep(0)
        assert self.other is not None and self.other.started
        assert query == "测试查询"
        assert top_k == 2
        return BM25SearchResponse(
            results=(
                _bm25_result(1, 20, 12.0),
                _bm25_result(2, 30, 8.0),
            ),
            search_ms=1.5,
        )


@pytest.mark.asyncio
async def test_hybrid_recall_runs_branches_concurrently_and_fuses_debug_evidence() -> None:
    dense = FakeDenseService()
    bm25 = FakeBM25Retriever()
    dense.other = bm25
    bm25.other = dense
    service = HybridRecallService(
        dense_service=dense,
        bm25_retriever=bm25,
    )

    response = await service.recall(
        "测试查询",
        dense_top_k=2,
        bm25_top_k=2,
        rank_constant=60,
        dense_mode=DenseSearchMode.HNSW,
    )

    assert [candidate.chunk_id for candidate in response.fused] == [20, 10, 30]
    assert response.fused[0].dense_rank == 2
    assert response.fused[0].bm25_rank == 1
    assert response.dense.results[0].chunk_id == 10
    assert response.bm25.results[0].chunk_id == 20
    assert response.union_candidate_count == 3
    assert response.rrf_ms >= 0
    assert response.total_ms >= 0
