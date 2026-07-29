import pytest
from numpy import float32, ones

from rag_demo.config import Settings
from rag_demo.db import Database
from rag_demo.migrations import apply_migrations

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_migrations_are_idempotent_and_schema_is_ready() -> None:
    settings = Settings()

    await apply_migrations(settings.database_url.get_secret_value())
    reapplied = await apply_migrations(settings.database_url.get_secret_value())

    assert reapplied == ()

    async with Database(settings.database_url.get_secret_value()) as database:
        status = await database.status()

    assert status.current_user == "rag_app"
    assert status.current_database == "rag_demo"
    assert status.vector_version == "0.8.5"
    assert status.embedding_dimensions == 1024
    assert status.hnsw_index_present is True
    assert status.applied_migrations == ("001_enable_vector", "002_create_tables")


@pytest.mark.asyncio
async def test_pool_adapts_1024_dimension_vectors_without_leaving_rows() -> None:
    settings = Settings()
    embedding = ones(settings.embedding_dimensions, dtype=float32)

    async with Database(settings.database_url.get_secret_value()) as database:
        before = await database.status()

        async with database.connection() as connection:
            async with connection.transaction(force_rollback=True):
                document = await connection.execute(
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
                        "https://example.test/repo",
                        "test-commit",
                        "docs/test.md",
                        "zh",
                        "Test",
                        "document-hash",
                    ),
                )
                document_row = await document.fetchone()
                assert document_row is not None
                document_id = document_row["id"]

                chunk = await connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_index,
                        chunker_version,
                        content_raw,
                        retrieval_text,
                        char_count,
                        content_hash,
                        embedding_model,
                        embedding_dimensions,
                        embedding
                    )
                    VALUES (%s, 0, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING
                        vector_dims(embedding) AS dimensions,
                        1 - (embedding <=> %s) AS cosine_similarity
                    """,
                    (
                        document_id,
                        "test-v1",
                        "raw",
                        "retrieval",
                        3,
                        "chunk-hash",
                        settings.embedding_model,
                        settings.embedding_dimensions,
                        embedding,
                        embedding,
                    ),
                )
                inserted = await chunk.fetchone()
                assert inserted is not None

        after = await database.status()

    assert inserted["dimensions"] == 1024
    assert inserted["cosine_similarity"] == pytest.approx(1.0)
    assert after.document_count == before.document_count
    assert after.chunk_count == before.chunk_count
