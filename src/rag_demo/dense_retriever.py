"""Exact and HNSW dense retrieval backed by PostgreSQL pgvector."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from psycopg import AsyncConnection

from rag_demo.db import Database, Row
from rag_demo.embedding_client import EmbeddingVector
from rag_demo.models.retrieval import (
    DenseQueryPlan,
    DenseQueryResponse,
    DenseRetrievalResponse,
    DenseSearchMode,
    DenseSearchResult,
)

__all__ = [
    "DenseQueryPlan",
    "DenseQueryResponse",
    "DenseQueryService",
    "DenseRetrievalResponse",
    "DenseRetriever",
    "DenseSearchMode",
    "DenseSearchResult",
    "QueryEmbeddingProvider",
]


class QueryEmbeddingProvider(Protocol):
    """Minimal Embedding contract needed by the query service."""

    @property
    def model(self) -> str:
        """Return the vector-space model identifier."""

    @property
    def dimensions(self) -> int:
        """Return the vector-space dimensions."""

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Embed query texts in input order."""


# 在初步筛选向量里面，漏了 commit 版本号
# 数一下，一共 6 个参数，说实话，写得有点屎
_DENSE_SQL = """
SELECT
    ranked_chunks.chunk_id,
    ranked_chunks.document_id,
    documents.title,
    documents.source_repo,
    documents.source_commit,
    documents.source_path,
    documents.language,
    ranked_chunks.heading_path,
    ranked_chunks.content_raw,
    ranked_chunks.retrieval_text,
    ranked_chunks.metadata,
    ranked_chunks.cosine_distance,
    ranked_chunks.cosine_similarity
FROM (
    SELECT
        chunks.id AS chunk_id,
        chunks.document_id,
        chunks.heading_path,
        chunks.content_raw,
        chunks.retrieval_text,
        chunks.metadata,
        chunks.embedding <=> %s AS cosine_distance,
        1 - (chunks.embedding <=> %s) AS cosine_similarity
    FROM chunks
    WHERE chunks.embedding_model = %s
      AND chunks.embedding_dimensions = %s
    ORDER BY chunks.embedding <=> %s
    LIMIT %s
) AS ranked_chunks
JOIN documents ON documents.id = ranked_chunks.document_id
ORDER BY ranked_chunks.cosine_distance
"""


class DenseRetriever:
    """Run isolated exact or HNSW cosine searches over stored Chunks."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_model: str,
        embedding_dimensions: int,
        hnsw_ef_search: int,
    ) -> None:
        if not embedding_model:
            raise ValueError("embedding_model must not be empty")
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        if hnsw_ef_search <= 0:
            raise ValueError("hnsw_ef_search must be positive")
        self._database = database
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._hnsw_ef_search = hnsw_ef_search

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_dimensions

    @property
    def hnsw_ef_search(self) -> int:
        return self._hnsw_ef_search

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        top_k: int,
        mode: DenseSearchMode,
    ) -> DenseRetrievalResponse:
        """
        Return cosine-ranked Chunks using the requested execution mode.
        就是向量检索喵
        """
        vector = self._validate_vector(query_vector)
        self._validate_top_k(top_k)  # 就一个简单合规性判断
        started = time.perf_counter()

        async with self._database.connection() as connection:
            await self._configure_mode(connection, mode)
            cursor = await connection.execute(
                _DENSE_SQL,
                self._query_parameters(vector, top_k),
            )
            rows = await cursor.fetchall()

        elapsed_ms = (time.perf_counter() - started) * 1000
        return DenseRetrievalResponse(
            mode=mode,
            results=tuple(
                self._result_from_row(row, rank=rank) for rank, row in enumerate(rows, start=1)
            ),
            search_ms=elapsed_ms,
        )

    async def explain(
        self,
        query_vector: EmbeddingVector,
        *,
        top_k: int,
        mode: DenseSearchMode,
    ) -> DenseQueryPlan:
        """
        Execute and return an ANALYZE/BUFFERS plan for demonstration.
        Demo 演示用，生产不用
        """
        vector = self._validate_vector(query_vector)
        self._validate_top_k(top_k)

        async with self._database.connection() as connection:
            # Demo 手动切换是否使用 HNSW；生产交给 PostgreSQL Planner 自动选择。
            await self._configure_mode(connection, mode)
            cursor = await connection.execute(
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {_DENSE_SQL}",
                self._query_parameters(vector, top_k),
            )
            rows = await cursor.fetchall()

        return DenseQueryPlan(
            mode=mode,
            lines=tuple(str(next(iter(row.values()))) for row in rows),
            ef_search=self._hnsw_ef_search if mode is DenseSearchMode.HNSW else None,
        )

    async def _configure_mode(
        self,
        connection: AsyncConnection[Row],
        mode: DenseSearchMode,
    ) -> None:
        if mode is DenseSearchMode.EXACT:
            await connection.execute("SET LOCAL enable_indexscan = off")
            await connection.execute("SET LOCAL enable_bitmapscan = off")
            return
        if mode is DenseSearchMode.HNSW:
            await connection.execute("SET LOCAL enable_seqscan = off")
            await connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(self._hnsw_ef_search),),
            )
            return
        raise ValueError(f"unsupported dense search mode: {mode}")

    def _validate_vector(self, vector: EmbeddingVector) -> EmbeddingVector:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self._embedding_dimensions:
            raise ValueError(
                f"query vector must have exactly {self._embedding_dimensions} dimensions"
            )
        if not np.isfinite(array).all():
            raise ValueError("query vector must contain only finite values")
        if math.isclose(float(np.linalg.norm(array)), 0.0):
            raise ValueError("query vector must not be a zero vector")
        return array

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

    def _query_parameters(
        self,
        vector: EmbeddingVector,
        top_k: int,
    ) -> tuple[object, ...]:
        """写得有点屎说实话"""
        return (
            vector,
            vector,
            self._embedding_model,
            self._embedding_dimensions,
            vector,
            top_k,
        )

    @staticmethod
    def _result_from_row(row: Row, *, rank: int) -> DenseSearchResult:
        return DenseSearchResult(
            rank=rank,
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            title=row["title"],
            source_repo=row["source_repo"],
            source_commit=row["source_commit"],
            source_path=row["source_path"],
            language=row["language"],
            heading_path=tuple(row["heading_path"]),
            content_raw=row["content_raw"],
            retrieval_text=row["retrieval_text"],
            metadata=row["metadata"],
            cosine_distance=float(row["cosine_distance"]),
            cosine_similarity=float(row["cosine_similarity"]),
        )


class DenseQueryService:
    """Embed one text query and execute Dense retrieval."""

    def __init__(
        self,
        *,
        embedding_client: QueryEmbeddingProvider,
        retriever: DenseRetriever,
    ) -> None:
        if embedding_client.model != retriever.embedding_model:
            raise ValueError("query and corpus embedding models must match")
        if embedding_client.dimensions != retriever.embedding_dimensions:
            raise ValueError("query and corpus embedding dimensions must match")
        self._embedding_client = embedding_client
        self._retriever = retriever

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        mode: DenseSearchMode = DenseSearchMode.HNSW,
    ) -> DenseQueryResponse:
        """
        Embed and search one non-empty query.
        检索服务入口
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        total_started = time.perf_counter()
        embedding_started = time.perf_counter()
        vectors = await self._embedding_client.embed((query,))  # 获取嵌入向量
        query_embedding_ms = (time.perf_counter() - embedding_started) * 1000
        if len(vectors) != 1:
            raise RuntimeError("query embedding service must return exactly one vector")

        retrieval = await self._retriever.search(
            vectors[0],
            top_k=top_k,
            mode=mode,
        )
        return DenseQueryResponse(
            query=query,
            mode=mode,
            results=retrieval.results,
            query_embedding_ms=query_embedding_ms,
            dense_search_ms=retrieval.search_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
