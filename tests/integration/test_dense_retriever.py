from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import pytest
from psycopg.types.json import Jsonb

from rag_demo.chunker import MarkdownChunker
from rag_demo.config import Settings
from rag_demo.db import Database
from rag_demo.dense_retriever import (
    DenseQueryService,
    DenseRetriever,
    DenseSearchMode,
)
from rag_demo.embedding_client import EmbeddingClient, EmbeddingVector
from rag_demo.ingest_service import DocumentIngestor

pytestmark = pytest.mark.integration
SOURCE_REPO = "https://example.test/dense-fixture"
MODEL = "dense-test-model"
DIMENSIONS = 1024


class QueryEmbeddingClient:
    model = MODEL
    dimensions = DIMENSIONS
    batch_size = 8

    def __init__(self, vector: EmbeddingVector) -> None:
        self.vector = vector
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.calls.append(tuple(texts))
        return (self.vector,)


def _vector(first: float, second: float) -> EmbeddingVector:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[0] = first
    vector[1] = second
    return vector


async def _cleanup(
    database: Database,
    *,
    idempotency_keys: Sequence[str] = (),
) -> None:
    async with database.connection() as connection:
        if idempotency_keys:
            await connection.execute(
                "DELETE FROM ingestion_requests WHERE idempotency_key = ANY(%s)",
                (list(idempotency_keys),),
            )
        await connection.execute(
            "DELETE FROM documents WHERE source_repo = %s",
            (SOURCE_REPO,),
        )


async def _insert_fixture(database: Database) -> None:
    async with database.connection() as connection:
        document_cursor = await connection.execute(
            """
            INSERT INTO documents (
                source_repo,
                source_commit,
                source_path,
                language,
                title,
                content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                SOURCE_REPO,
                "dense-commit",
                "docs/dense.md",
                "zh",
                "Dense 测试",
                "document-hash",
            ),
        )
        document = await document_cursor.fetchone()
        assert document is not None

        rows = (
            (0, "最近结果", _vector(1.0, 0.0), MODEL),
            (1, "中间结果", _vector(0.0, 1.0), MODEL),
            (2, "最远结果", _vector(-1.0, 0.0), MODEL),
            (3, "其他模型结果", _vector(1.0, 0.0), "other-model"),
        )
        for chunk_index, content, embedding, model in rows:
            await connection.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    chunker_version,
                    heading_path,
                    content_raw,
                    retrieval_text,
                    char_count,
                    content_hash,
                    metadata,
                    embedding_model,
                    embedding_dimensions,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    document["id"],
                    chunk_index,
                    "dense-fixture-v1",
                    ["测试", content],
                    content,
                    f"[文档：Dense 测试]\n[章节：测试 / {content}]\n{content}",
                    len(content),
                    f"chunk-hash-{chunk_index}",
                    Jsonb({"fixture": True}),
                    model,
                    DIMENSIONS,
                    embedding,
                ),
            )


@pytest.mark.asyncio
async def test_exact_dense_search_returns_true_cosine_order_and_source_metadata() -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database)
        await _insert_fixture(database)
        retriever = DenseRetriever(
            database=database,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            hnsw_ef_search=100,
        )
        try:
            response = await retriever.search(
                _vector(1.0, 0.0),
                top_k=3,
                mode=DenseSearchMode.EXACT,
            )
        finally:
            await _cleanup(database)

    assert [result.content_raw for result in response.results] == [
        "最近结果",
        "中间结果",
        "最远结果",
    ]
    assert [result.rank for result in response.results] == [1, 2, 3]
    assert response.results[0].cosine_similarity == pytest.approx(1.0)
    assert response.results[1].cosine_similarity == pytest.approx(0.0)
    assert response.results[2].cosine_similarity == pytest.approx(-1.0)
    assert response.results[0].source_path == "docs/dense.md"
    assert response.results[0].heading_path == ("测试", "最近结果")
    assert response.results[0].metadata == {"fixture": True}
    assert response.mode is DenseSearchMode.EXACT
    assert response.candidate_count == 3
    assert response.search_ms >= 0


@pytest.mark.asyncio
async def test_exact_and_hnsw_modes_use_the_expected_postgresql_plans() -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database)
        await _insert_fixture(database)
        retriever = DenseRetriever(
            database=database,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            hnsw_ef_search=100,
        )
        try:
            exact_plan = await retriever.explain(
                _vector(1.0, 0.0),
                top_k=3,
                mode=DenseSearchMode.EXACT,
            )
            hnsw_plan = await retriever.explain(
                _vector(1.0, 0.0),
                top_k=3,
                mode=DenseSearchMode.HNSW,
            )
            hnsw_response = await retriever.search(
                _vector(1.0, 0.0),
                top_k=1,
                mode=DenseSearchMode.HNSW,
            )
        finally:
            await _cleanup(database)

    assert exact_plan.uses_hnsw is False
    assert "Seq Scan" in exact_plan.text
    assert hnsw_plan.uses_hnsw is True, hnsw_plan.text
    assert "chunks_embedding_hnsw" in hnsw_plan.text
    assert hnsw_plan.ef_search == 100
    assert hnsw_response.results[0].content_raw == "最近结果"


@pytest.mark.asyncio
async def test_dense_query_service_embeds_one_query_and_reports_stage_timings() -> None:
    settings = Settings()
    query = "哪一个结果最接近？"
    query_vector = _vector(1.0, 0.0)
    embedding_client = QueryEmbeddingClient(query_vector)

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database)
        await _insert_fixture(database)
        retriever = DenseRetriever(
            database=database,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            hnsw_ef_search=100,
        )
        service = DenseQueryService(
            embedding_client=embedding_client,
            retriever=retriever,
        )
        try:
            response = await service.search(
                query,
                top_k=1,
                mode=DenseSearchMode.HNSW,
            )
        finally:
            await _cleanup(database)

    assert embedding_client.calls == [(query,)]
    assert response.query == query
    assert response.query_embedding_ms >= 0
    assert response.dense_search_ms >= 0
    assert response.total_ms >= response.query_embedding_ms
    assert response.results[0].content_raw == "最近结果"


@pytest.mark.asyncio
async def test_dense_search_rejects_invalid_query_vector() -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        retriever = DenseRetriever(
            database=database,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
            hnsw_ef_search=100,
        )

        with pytest.raises(ValueError, match="1024 dimensions"):
            await retriever.search(
                np.ones(3, dtype=np.float32),
                top_k=1,
                mode=DenseSearchMode.EXACT,
            )
        invalid = _vector(1.0, 0.0)
        invalid[10] = np.nan
        with pytest.raises(ValueError, match="finite"):
            await retriever.search(
                invalid,
                top_k=1,
                mode=DenseSearchMode.EXACT,
            )


@pytest.mark.asyncio
async def test_real_embedding_to_exact_and_hnsw_dense_search(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "dense.md").write_text(
        """
# FastAPI 教程

## 请求体

使用 Pydantic 模型让接口接收 JSON 对象。

## 安全

OAuth2PasswordBearer 用于从请求头提取 Bearer Token。
""".strip(),
        encoding="utf-8",
    )
    idempotency_key = "real-dense-end-to-end"
    settings = Settings()
    timeout = httpx.Timeout(
        settings.embedding_timeout_seconds,
        connect=5.0,
        write=15.0,
        pool=5.0,
    )

    async with (
        Database(settings.database_url.get_secret_value()) as database,
        httpx.AsyncClient(timeout=timeout) as http_client,
    ):
        await _cleanup(database, idempotency_keys=(idempotency_key,))
        embedding_client = EmbeddingClient.from_settings(http_client, settings)
        ingestor = DocumentIngestor(
            database=database,
            embedding_client=embedding_client,
            chunker=MarkdownChunker(
                target_chars=80,
                max_chars=160,
                overlap_chars=20,
            ),
            source_repo=SOURCE_REPO,
            source_commit="real-dense-commit",
            source_root=tmp_path,
            language="zh",
        )
        retriever = DenseRetriever(
            database=database,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            hnsw_ef_search=settings.hnsw_ef_search,
        )
        service = DenseQueryService(
            embedding_client=embedding_client,
            retriever=retriever,
        )
        try:
            ingested = await ingestor.ingest_path(
                source,
                idempotency_key=idempotency_key,
            )
            async with database.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT id, retrieval_text
                    FROM chunks
                    WHERE embedding_model = %s
                      AND retrieval_text LIKE %s
                    """,
                    (settings.embedding_model, "%Pydantic%"),
                )
                target = await cursor.fetchone()
                assert target is not None

            exact = await service.search(
                target["retrieval_text"],
                top_k=1,
                mode=DenseSearchMode.EXACT,
            )
            hnsw = await service.search(
                target["retrieval_text"],
                top_k=1,
                mode=DenseSearchMode.HNSW,
            )
        finally:
            await _cleanup(database, idempotency_keys=(idempotency_key,))

    assert ingested.chunk_count == 2
    assert exact.results[0].chunk_id == target["id"]
    assert hnsw.results[0].chunk_id == target["id"]
    assert exact.results[0].cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert hnsw.results[0].cosine_similarity == pytest.approx(1.0, abs=1e-5)
