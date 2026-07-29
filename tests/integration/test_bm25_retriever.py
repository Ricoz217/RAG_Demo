from pathlib import Path

import numpy as np
import pytest
from psycopg.types.json import Jsonb

import rag_demo.bm25_retriever as bm25_module
from rag_demo.bm25_retriever import (
    BM25IndexManager,
    BM25IndexNotFoundError,
)
from rag_demo.config import Settings
from rag_demo.db import Database

pytestmark = pytest.mark.integration
SOURCE_REPO = "https://example.test/bm25-fixture"
MODEL = "bm25-test-model"
DIMENSIONS = 1024


async def _cleanup(database: Database) -> None:
    async with database.connection() as connection:
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
                "bm25-commit",
                "docs/bm25.md",
                "zh",
                "BM25 测试",
                "document-hash",
            ),
        )
        document = await document_cursor.fetchone()
        assert document is not None

        contents = (
            ("请求体", "使用 Pydantic 模型让接口接收 JSON 对象。"),
            ("OAuth2", "OAuth2PasswordBearer 从 HTTP 请求头读取 Bearer Token。"),
            ("后台任务", "BackgroundTasks 可以在返回响应以后执行任务。"),
        )
        embedding = np.ones(DIMENSIONS, dtype=np.float32)
        for chunk_index, (heading, content) in enumerate(contents):
            retrieval_text = f"[文档：BM25 测试]\n[章节：{heading}]\n{content}"
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
                    "bm25-fixture-v1",
                    [heading],
                    content,
                    retrieval_text,
                    len(content),
                    f"chunk-hash-{chunk_index}",
                    Jsonb({"fixture": True}),
                    MODEL,
                    DIMENSIONS,
                    embedding,
                ),
            )


@pytest.mark.asyncio
async def test_bm25_rebuild_load_and_search_preserve_chunk_identity(
    tmp_path: Path,
) -> None:
    settings = Settings()
    index_root = tmp_path / "bm25"

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database)
        await _insert_fixture(database)
        manager = BM25IndexManager(
            database=database,
            index_root=index_root,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
        )
        try:
            build = await manager.rebuild()
            retriever = await manager.load()
            identifier = await retriever.search("OAuth2PasswordBearer", top_k=3)
            chinese = await retriever.search("接口接收 JSON 对象", top_k=3)
            missing = await retriever.search("绝对不存在的独特词条 xyznotfound", top_k=3)
        finally:
            await _cleanup(database)

    assert build.chunk_count == 3
    assert build.generation.startswith("generation-")
    assert build.build_ms >= 0
    assert (index_root / "CURRENT").read_text(encoding="utf-8").strip() == build.generation
    assert identifier.results[0].heading_path == ("OAuth2",)
    assert identifier.results[0].source_path == "docs/bm25.md"
    assert identifier.results[0].metadata == {"fixture": True}
    assert identifier.results[0].score > 0
    assert chinese.results[0].heading_path == ("请求体",)
    assert missing.results == ()
    assert identifier.candidate_count >= 1
    assert identifier.search_ms >= 0


@pytest.mark.asyncio
async def test_failed_rebuild_keeps_previous_generation_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    index_root = tmp_path / "bm25"

    async with Database(settings.database_url.get_secret_value()) as database:
        await _cleanup(database)
        await _insert_fixture(database)
        manager = BM25IndexManager(
            database=database,
            index_root=index_root,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
        )
        try:
            first = await manager.rebuild()

            def fail_build(*args: object, **kwargs: object) -> None:
                raise RuntimeError("simulated BM25 build failure")

            monkeypatch.setattr(bm25_module, "_write_generation", fail_build)
            with pytest.raises(RuntimeError, match="simulated BM25 build failure"):
                await manager.rebuild()

            loaded = await manager.load()
            result = await loaded.search("BackgroundTasks", top_k=1)
        finally:
            await _cleanup(database)

    assert (index_root / "CURRENT").read_text(encoding="utf-8").strip() == first.generation
    assert result.results[0].heading_path == ("后台任务",)


@pytest.mark.asyncio
async def test_loading_missing_bm25_index_reports_clear_error(tmp_path: Path) -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        manager = BM25IndexManager(
            database=database,
            index_root=tmp_path / "missing",
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
        )

        with pytest.raises(BM25IndexNotFoundError, match="not been built"):
            await manager.load()
