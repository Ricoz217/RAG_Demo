"""Concurrent Dense/BM25 recall and observable RRF fusion."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from rag_demo.bm25_retriever import BM25SearchResponse
from rag_demo.dense_retriever import DenseQueryResponse, DenseSearchMode
from rag_demo.rrf import FusedCandidate, ScoredCandidate, reciprocal_rank_fusion


class DenseQueryProvider(Protocol):
    """Dense query behavior consumed by hybrid recall."""

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        mode: DenseSearchMode,
    ) -> DenseQueryResponse:
        """Embed and search one query."""


class BM25QueryProvider(Protocol):
    """BM25 query behavior consumed by hybrid recall."""

    async def search(self, query: str, *, top_k: int) -> BM25SearchResponse:
        """Search one query."""


class RerankerProvider(Protocol):
    """Cross-Encoder behavior consumed by final search."""

    @property
    def model(self) -> str:
        """Return the configured reranker model identifier."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[float, ...]:
        """Return scores aligned with the input documents."""


@dataclass(frozen=True, slots=True)
class HybridRecallResponse:
    """Dense, BM25, and RRF stages retained for debug output."""

    query: str
    dense: DenseQueryResponse
    bm25: BM25SearchResponse
    fused: tuple[FusedCandidate, ...]
    rrf_ms: float
    total_ms: float

    @property
    def union_candidate_count(self) -> int:
        return len(self.fused)


class HybridRecallProvider(Protocol):
    """Hybrid recall behavior consumed by final search."""

    async def recall(
        self,
        query: str,
        *,
        dense_top_k: int,
        bm25_top_k: int,
        rank_constant: int,
        dense_mode: DenseSearchMode,
    ) -> HybridRecallResponse:
        """Return observable Dense, BM25, and RRF stages."""


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Validated core search parameters shared by CLI and REST."""

    query: str
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_rank_constant: int = 60
    rerank_top_k: int = 20
    final_top_k: int = 5
    dense_mode: DenseSearchMode = DenseSearchMode.HNSW
    use_reranker: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        for name, value in (
            ("dense_top_k", self.dense_top_k),
            ("bm25_top_k", self.bm25_top_k),
            ("rrf_rank_constant", self.rrf_rank_constant),
            ("rerank_top_k", self.rerank_top_k),
            ("final_top_k", self.final_top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k must not exceed rerank_top_k")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Hydrated candidate retaining every ranking stage."""

    chunk_id: int
    document_id: int
    title: str | None
    source_repo: str
    source_commit: str
    source_path: str
    language: str
    heading_path: tuple[str, ...]
    content_raw: str
    retrieval_text: str
    metadata: Mapping[str, Any]
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_rank: int
    rrf_score: float
    rerank_score: float | None
    final_rank: int | None


@dataclass(frozen=True, slots=True)
class SearchTimings:
    """Per-stage search latency in milliseconds."""

    query_embedding_ms: float
    dense_search_ms: float
    bm25_search_ms: float
    rrf_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class SearchCandidateCounts:
    """Candidate counts at branch and union boundaries."""

    dense_candidate_count: int
    bm25_candidate_count: int
    union_candidate_count: int


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Final results plus complete intermediate evidence."""

    query: str
    results: tuple[SearchCandidate, ...]
    reranked: tuple[SearchCandidate, ...]
    recall: HybridRecallResponse
    timings: SearchTimings
    counts: SearchCandidateCounts
    reranker_used: bool


class HybridRecallService:
    """Run Dense and BM25 concurrently, then fuse their ordered candidates."""

    def __init__(
        self,
        *,
        dense_service: DenseQueryProvider,
        bm25_retriever: BM25QueryProvider,
    ) -> None:
        self._dense_service = dense_service
        self._bm25_retriever = bm25_retriever

    async def recall(
        self,
        query: str,
        *,
        dense_top_k: int,
        bm25_top_k: int,
        rank_constant: int,
        dense_mode: DenseSearchMode = DenseSearchMode.HNSW,
    ) -> HybridRecallResponse:
        """Return all branch rankings and their RRF union."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if dense_top_k <= 0:
            raise ValueError("dense_top_k must be positive")
        if bm25_top_k <= 0:
            raise ValueError("bm25_top_k must be positive")
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")

        total_started = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            dense_task = group.create_task(
                self._dense_service.search(
                    query,
                    top_k=dense_top_k,
                    mode=dense_mode,
                )
            )
            bm25_task = group.create_task(self._bm25_retriever.search(query, top_k=bm25_top_k))

        dense = dense_task.result()
        bm25 = bm25_task.result()
        rrf_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            tuple(
                ScoredCandidate(
                    chunk_id=result.chunk_id,
                    score=result.cosine_similarity,
                )
                for result in dense.results
            ),
            tuple(
                ScoredCandidate(chunk_id=result.chunk_id, score=result.score)
                for result in bm25.results
            ),
            rank_constant=rank_constant,
        )
        rrf_ms = (time.perf_counter() - rrf_started) * 1000
        return HybridRecallResponse(
            query=query,
            dense=dense,
            bm25=bm25,
            fused=fused,
            rrf_ms=rrf_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )


class HybridSearchService:
    """Run hybrid recall and optionally rerank the RRF candidate prefix."""

    def __init__(
        self,
        *,
        recall_service: HybridRecallProvider,
        reranker_client: RerankerProvider | None,
    ) -> None:
        self._recall_service = recall_service
        self._reranker_client = reranker_client

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute the complete retrieval pipeline without answer generation."""
        total_started = time.perf_counter()
        recall = await self._recall_service.recall(
            request.query,
            dense_top_k=request.dense_top_k,
            bm25_top_k=request.bm25_top_k,
            rank_constant=request.rrf_rank_constant,
            dense_mode=request.dense_mode,
        )
        candidates = _hydrate_fused_candidates(recall)

        rerank_ms = 0.0
        reranked: tuple[SearchCandidate, ...] = ()
        reranker_used = request.use_reranker
        if request.use_reranker and candidates:
            if self._reranker_client is None:
                raise RuntimeError("reranker is enabled but no client is configured")
            rerank_input = candidates[: request.rerank_top_k]
            rerank_started = time.perf_counter()
            scores = await self._reranker_client.rerank(
                request.query,
                tuple(candidate.retrieval_text for candidate in rerank_input),
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            if len(scores) != len(rerank_input):
                raise RuntimeError("reranker returned an unexpected score count")
            if not all(math.isfinite(score) for score in scores):
                raise RuntimeError("reranker returned a non-finite score")
            reranked = tuple(
                candidate
                for _, candidate in sorted(
                    (
                        (
                            original_position,
                            replace(candidate, rerank_score=score),
                        )
                        for original_position, (candidate, score) in enumerate(
                            zip(rerank_input, scores, strict=True)
                        )
                    ),
                    key=lambda item: (
                        -item[1].rerank_score if item[1].rerank_score is not None else math.inf,
                        item[0],
                    ),
                )
            )
            final_base = reranked[: request.final_top_k]
        else:
            final_base = candidates[: request.final_top_k]

        results = tuple(
            replace(candidate, final_rank=rank)
            for rank, candidate in enumerate(final_base, start=1)
        )
        return SearchResponse(
            query=request.query,
            results=results,
            reranked=reranked,
            recall=recall,
            timings=SearchTimings(
                query_embedding_ms=recall.dense.query_embedding_ms,
                dense_search_ms=recall.dense.dense_search_ms,
                bm25_search_ms=recall.bm25.search_ms,
                rrf_ms=recall.rrf_ms,
                rerank_ms=rerank_ms,
                total_ms=(time.perf_counter() - total_started) * 1000,
            ),
            counts=SearchCandidateCounts(
                dense_candidate_count=recall.dense.candidate_count,
                bm25_candidate_count=recall.bm25.candidate_count,
                union_candidate_count=recall.union_candidate_count,
            ),
            reranker_used=reranker_used,
        )


def _hydrate_fused_candidates(
    recall: HybridRecallResponse,
) -> tuple[SearchCandidate, ...]:
    dense_by_id = {result.chunk_id: result for result in recall.dense.results}
    bm25_by_id = {result.chunk_id: result for result in recall.bm25.results}
    hydrated: list[SearchCandidate] = []

    for rrf_rank, fused in enumerate(recall.fused, start=1):
        source = dense_by_id.get(fused.chunk_id) or bm25_by_id.get(fused.chunk_id)
        if source is None:
            raise RuntimeError(f"RRF candidate {fused.chunk_id} has no source Chunk")
        hydrated.append(
            SearchCandidate(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                title=source.title,
                source_repo=source.source_repo,
                source_commit=source.source_commit,
                source_path=source.source_path,
                language=source.language,
                heading_path=source.heading_path,
                content_raw=source.content_raw,
                retrieval_text=source.retrieval_text,
                metadata=source.metadata,
                dense_rank=fused.dense_rank,
                dense_score=fused.dense_score,
                bm25_rank=fused.bm25_rank,
                bm25_score=fused.bm25_score,
                rrf_rank=rrf_rank,
                rrf_score=fused.rrf_score,
                rerank_score=None,
                final_rank=None,
            )
        )
    return tuple(hydrated)
