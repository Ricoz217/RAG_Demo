"""Concurrent Dense/BM25 recall and observable RRF fusion."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

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
