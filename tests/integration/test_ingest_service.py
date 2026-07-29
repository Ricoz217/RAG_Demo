import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import pytest

from rag_demo.chunker import MarkdownChunker
from rag_demo.config import Settings
from rag_demo.db import Database
from rag_demo.embedding_client import EmbeddingClient, EmbeddingVector
from rag_demo.ingest_service import (
    DocumentIngestor,
    IdempotencyConflictError,
    IdempotencyInProgressError,
)
from rag_demo.migrations import apply_migrations

pytestmark = pytest.mark.integration
SOURCE_REPO = "https://example.test/ingestion-fixture"


class FakeEmbeddingClient:
    model = "test-embedding-model"
    dimensions = 1024
    batch_size = 8

    def __init__(
        self,
        *,
        model: str = "test-embedding-model",
        failures: int = 0,
        delay_seconds: float = 0,
    ) -> None:
        self.model = model
        self.calls: list[tuple[str, ...]] = []
        self.failures = failures
        self.delay_seconds = delay_seconds

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.calls.append(tuple(texts))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("simulated embedding failure")
        return tuple(
            np.full(self.dimensions, index + 1, dtype=np.float32) for index, _ in enumerate(texts)
        )


def _ingestor(
    database: Database,
    embedding_client: FakeEmbeddingClient,
    source_root: Path,
    *,
    source_commit: str,
) -> DocumentIngestor:
    return DocumentIngestor(
        database=database,
        embedding_client=embedding_client,
        chunker=MarkdownChunker(
            target_chars=40,
            max_chars=80,
            overlap_chars=10,
        ),
        source_repo=SOURCE_REPO,
        source_commit=source_commit,
        source_root=source_root,
        language="zh",
    )


async def _cleanup(
    database: Database,
    *,
    idempotency_keys: Sequence[str],
) -> None:
    async with database.connection() as connection:
        await connection.execute(
            "DELETE FROM ingestion_requests WHERE idempotency_key = ANY(%s)",
            (list(idempotency_keys),),
        )
        await connection.execute(
            "DELETE FROM documents WHERE source_repo = %s",
            (SOURCE_REPO,),
        )


async def _source_counts(database: Database) -> tuple[int, int]:
    async with database.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT
                count(DISTINCT documents.id) AS documents,
                count(chunks.id) AS chunks
            FROM documents
            LEFT JOIN chunks ON chunks.document_id = documents.id
            WHERE documents.source_repo = %s
            """,
            (SOURCE_REPO,),
        )
        row = await cursor.fetchone()
        assert row is not None
        return row["documents"], row["chunks"]


@pytest.mark.asyncio
async def test_repeated_ingestion_is_idempotent_and_skips_unchanged_embeddings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text(
        "# 教程\n\n## 请求体\n\n第一段内容。\n\n第二段内容。",
        encoding="utf-8",
    )
    keys = ("repeat-key", "new-key-same-body")
    settings = Settings()
    await apply_migrations(settings.database_url.get_secret_value())

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=keys)
        embeddings = FakeEmbeddingClient()
        ingestor = _ingestor(database, embeddings, tmp_path, source_commit="commit-repeat")
        try:
            first = await ingestor.ingest_path(source, idempotency_key=keys[0])
            repeated = await ingestor.ingest_path(source, idempotency_key=keys[0])
            same_body = await ingestor.ingest_path(source, idempotency_key=keys[1])
            counts = await _source_counts(database)
        finally:
            await _cleanup(database, idempotency_keys=keys)

    assert repeated == first
    assert first.document_count == 1
    assert first.chunk_count > 0
    assert first.embedded_chunk_count == first.chunk_count
    assert same_body.embedded_chunk_count == 0
    assert same_body.skipped_embedding_count == same_body.chunk_count
    assert len(embeddings.calls) == 1
    assert counts == (1, first.chunk_count)


@pytest.mark.asyncio
async def test_same_idempotency_key_rejects_different_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text("# 标题\n\n正文。", encoding="utf-8")
    key = "conflict-key"
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=(key,))
        embeddings = FakeEmbeddingClient()
        first = _ingestor(database, embeddings, tmp_path, source_commit="commit-a")
        conflicting = _ingestor(database, embeddings, tmp_path, source_commit="commit-b")
        try:
            await first.ingest_path(source, idempotency_key=key)
            with pytest.raises(IdempotencyConflictError, match="different request"):
                await conflicting.ingest_path(source, idempotency_key=key)
        finally:
            await _cleanup(database, idempotency_keys=(key,))


@pytest.mark.asyncio
async def test_ingestion_deletes_stale_chunks_in_the_same_source_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    markdown = source / "body.md"
    markdown.write_text(
        "# 标题\n\n## 章节\n\n" + "甲" * 55 + "\n\n" + "乙" * 55 + "\n\n" + "丙" * 55,
        encoding="utf-8",
    )
    keys = ("stale-first", "stale-second")
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=keys)
        embeddings = FakeEmbeddingClient()
        ingestor = _ingestor(database, embeddings, tmp_path, source_commit="commit-stale")
        try:
            first = await ingestor.ingest_path(source, idempotency_key=keys[0])
            markdown.write_text("# 标题\n\n## 章节\n\n短正文。", encoding="utf-8")
            second = await ingestor.ingest_path(source, idempotency_key=keys[1])
            counts = await _source_counts(database)
        finally:
            await _cleanup(database, idempotency_keys=keys)

    assert first.chunk_count > second.chunk_count
    assert second.deleted_chunk_count == first.chunk_count - second.chunk_count
    assert counts == (1, second.chunk_count)


@pytest.mark.asyncio
async def test_failed_idempotency_key_can_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text("# 标题\n\n正文。", encoding="utf-8")
    key = "retry-key"
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=(key,))
        embeddings = FakeEmbeddingClient(failures=1)
        ingestor = _ingestor(database, embeddings, tmp_path, source_commit="commit-retry")
        try:
            with pytest.raises(RuntimeError, match="simulated embedding failure"):
                await ingestor.ingest_path(source, idempotency_key=key)
            retried = await ingestor.ingest_path(source, idempotency_key=key)

            async with database.connection() as connection:
                cursor = await connection.execute(
                    "SELECT status FROM ingestion_requests WHERE idempotency_key = %s",
                    (key,),
                )
                row = await cursor.fetchone()
                assert row is not None
        finally:
            await _cleanup(database, idempotency_keys=(key,))

    assert retried.embedded_chunk_count == retried.chunk_count
    assert row["status"] == "completed"
    assert len(embeddings.calls) == 2


@pytest.mark.asyncio
async def test_concurrent_ingestion_of_same_source_is_serialized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text(
        "# 标题\n\n## 章节\n\n并发摄取正文。",
        encoding="utf-8",
    )
    keys = ("concurrent-a", "concurrent-b")
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=keys)
        embeddings = FakeEmbeddingClient(delay_seconds=0.1)
        ingestor = _ingestor(database, embeddings, tmp_path, source_commit="commit-concurrent")
        try:
            results = await asyncio.gather(
                ingestor.ingest_path(source, idempotency_key=keys[0]),
                ingestor.ingest_path(source, idempotency_key=keys[1]),
            )
            counts = await _source_counts(database)
        finally:
            await _cleanup(database, idempotency_keys=keys)

    assert counts == (1, results[0].chunk_count)
    assert sum(result.embedded_chunk_count for result in results) == results[0].chunk_count
    assert sum(result.skipped_embedding_count for result in results) == results[0].chunk_count


@pytest.mark.asyncio
async def test_same_idempotency_key_reports_request_in_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text("# 标题\n\n正文。", encoding="utf-8")
    key = "processing-key"
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=(key,))
        embeddings = FakeEmbeddingClient(delay_seconds=0.2)
        ingestor = _ingestor(database, embeddings, tmp_path, source_commit="commit-processing")
        try:
            first = asyncio.create_task(ingestor.ingest_path(source, idempotency_key=key))
            await asyncio.sleep(0.05)
            with pytest.raises(IdempotencyInProgressError, match="already processing"):
                await ingestor.ingest_path(source, idempotency_key=key)
            await first
        finally:
            await _cleanup(database, idempotency_keys=(key,))


@pytest.mark.asyncio
async def test_switching_embedding_model_reembeds_existing_chunks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text("# 标题\n\n正文。", encoding="utf-8")
    keys = ("model-a-key", "model-b-key")
    settings = Settings()

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database, idempotency_keys=keys)
        first_client = FakeEmbeddingClient(model="model-a")
        second_client = FakeEmbeddingClient(model="model-b")
        first_ingestor = _ingestor(
            database,
            first_client,
            tmp_path,
            source_commit="commit-model-switch",
        )
        second_ingestor = _ingestor(
            database,
            second_client,
            tmp_path,
            source_commit="commit-model-switch",
        )
        try:
            first = await first_ingestor.ingest_path(source, idempotency_key=keys[0])
            second = await second_ingestor.ingest_path(source, idempotency_key=keys[1])

            async with database.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT DISTINCT chunks.embedding_model
                    FROM chunks
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE documents.source_repo = %s
                    """,
                    (SOURCE_REPO,),
                )
                models = {row["embedding_model"] for row in await cursor.fetchall()}
        finally:
            await _cleanup(database, idempotency_keys=keys)

    assert first.embedded_chunk_count == first.chunk_count
    assert second.embedded_chunk_count == second.chunk_count
    assert models == {"model-b"}


@pytest.mark.asyncio
async def test_real_embedding_service_ingests_one_markdown_into_pgvector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "body.md").write_text(
        "# FastAPI 教程\n\n## 请求体\n\n使用 Pydantic 模型声明 JSON 请求体。",
        encoding="utf-8",
    )
    key = "real-embedding-ingestion"
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
        await _cleanup(database, idempotency_keys=(key,))
        embedding_client = EmbeddingClient.from_settings(http_client, settings)
        ingestor = DocumentIngestor(
            database=database,
            embedding_client=embedding_client,
            chunker=MarkdownChunker(
                target_chars=settings.chunk_target_chars,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ),
            source_repo=SOURCE_REPO,
            source_commit="real-embedding-commit",
            source_root=tmp_path,
            language="zh",
        )
        try:
            result = await ingestor.ingest_path(source, idempotency_key=key)
            async with database.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT
                        vector_dims(chunks.embedding) AS dimensions,
                        chunks.embedding_model
                    FROM chunks
                    JOIN documents ON documents.id = chunks.document_id
                    WHERE documents.source_repo = %s
                    """,
                    (SOURCE_REPO,),
                )
                rows = await cursor.fetchall()
        finally:
            await _cleanup(database, idempotency_keys=(key,))

    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.embedding_http_request_count == 1
    assert len(rows) == 1
    assert rows[0]["dimensions"] == 1024
    assert rows[0]["embedding_model"] == settings.embedding_model
