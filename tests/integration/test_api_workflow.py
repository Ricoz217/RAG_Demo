import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import rag_demo.bm25_retriever as bm25_module
from rag_demo.api import create_app
from rag_demo.cli import app as cli_app
from rag_demo.config import Settings
from rag_demo.db import Database
from rag_demo.search_service import SearchResponse

pytestmark = pytest.mark.integration
SOURCE_REPO = "https://example.test/api-fixture"
INGESTION_KEY = "api-workflow-ingestion"
BM25_KEY = "api-workflow-bm25"
CONCURRENT_BM25_KEY = "api-workflow-bm25-concurrent"
runner = CliRunner()


async def _cleanup() -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        async with database.connection() as connection:
            await connection.execute(
                """
                DELETE FROM ingestion_requests
                WHERE idempotency_key = %s
                   OR idempotency_key = %s
                   OR idempotency_key = %s
                """,
                (
                    INGESTION_KEY,
                    f"bm25-rebuild:{BM25_KEY}",
                    f"bm25-rebuild:{CONCURRENT_BM25_KEY}",
                ),
            )
            await connection.execute(
                "DELETE FROM documents WHERE source_repo = %s",
                (SOURCE_REPO,),
            )


@pytest.mark.asyncio
async def test_rest_workflow_and_cli_return_the_same_final_chunk_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "api.md").write_text(
        """
# FastAPI 教程

## 请求体

使用 Pydantic 模型声明 JSON 请求体。

## 安全

OAuth2PasswordBearer 从请求头读取 Bearer Token。
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25-index"))
    settings = Settings()
    await _cleanup()

    ingest_body = {
        "source": str(source),
        "source_repo": SOURCE_REPO,
        "source_commit": "api-commit",
        "source_root": str(tmp_path),
        "language": "zh",
    }
    captured_cli: list[SearchResponse] = []

    try:
        api = create_app(settings)
        async with (
            api.router.lifespan_context(api),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api),
                base_url="http://test",
            ) as client,
        ):
            live = await client.get("/health/live")
            assert live.status_code == 200
            assert live.json() == {"status": "ok"}

            not_ready = await client.get("/health/ready")
            assert not_ready.status_code == 503

            ingest = await client.post(
                "/v1/ingest",
                headers={"Idempotency-Key": INGESTION_KEY},
                json=ingest_body,
            )
            assert ingest.status_code == 200, ingest.text
            assert ingest.json()["document_count"] == 1
            assert ingest.json()["chunk_count"] == 2

            repeated_ingest = await client.post(
                "/v1/ingest",
                headers={"Idempotency-Key": INGESTION_KEY},
                json=ingest_body,
            )
            assert repeated_ingest.status_code == 200
            assert repeated_ingest.json() == ingest.json()

            conflicting_body = {**ingest_body, "source_commit": "different-commit"}
            conflict = await client.post(
                "/v1/ingest",
                headers={"Idempotency-Key": INGESTION_KEY},
                json=conflicting_body,
            )
            assert conflict.status_code == 409

            rebuild = await client.post(
                "/v1/bm25/rebuild",
                headers={"Idempotency-Key": BM25_KEY},
            )
            assert rebuild.status_code == 200, rebuild.text
            repeated_rebuild = await client.post(
                "/v1/bm25/rebuild",
                headers={"Idempotency-Key": BM25_KEY},
            )
            assert repeated_rebuild.status_code == 200
            assert repeated_rebuild.json() == rebuild.json()

            ready = await client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json()["status"] == "ready"

            stats = await client.get("/v1/stats")
            assert stats.status_code == 200
            assert stats.json()["document_count"] == 1
            assert stats.json()["chunk_count"] == 2
            assert stats.json()["bm25"]["chunk_count"] == 2

            query = "OAuth2PasswordBearer 有什么作用？"
            search_body = {
                "query": query,
                "dense_top_k": 30,
                "bm25_top_k": 30,
                "rerank_top_k": 20,
                "final_top_k": 2,
                "debug": True,
                "use_reranker": True,
            }
            rest_search = await client.post("/v1/search", json=search_body)
            assert rest_search.status_code == 200, rest_search.text
            payload: dict[str, Any] = rest_search.json()
            assert payload["results"]
            assert payload["debug"]["dense"]
            assert payload["debug"]["bm25"]
            assert payload["debug"]["rrf"]
            assert payload["debug"]["reranker"]

            original_write = bm25_module._write_generation

            def slow_write(*args: Any, **kwargs: Any) -> None:
                time.sleep(0.2)
                original_write(*args, **kwargs)

            monkeypatch.setattr(bm25_module, "_write_generation", slow_write)
            concurrent = await asyncio.gather(
                client.post(
                    "/v1/bm25/rebuild",
                    headers={"Idempotency-Key": CONCURRENT_BM25_KEY},
                ),
                client.post(
                    "/v1/bm25/rebuild",
                    headers={"Idempotency-Key": CONCURRENT_BM25_KEY},
                ),
            )
            assert sorted(response.status_code for response in concurrent) == [200, 409]

            def capture(response: SearchResponse, *, debug: bool) -> None:
                assert debug is True
                captured_cli.append(response)

            monkeypatch.setattr("rag_demo.cli._print_search_response", capture)
            cli = await asyncio.to_thread(
                runner.invoke,
                cli_app,
                [
                    "search",
                    query,
                    "--final-top-k",
                    "2",
                    "--debug",
                ],
            )
            assert cli.exit_code == 0, cli.output

        assert len(captured_cli) == 1
        rest_ids = [item["chunk_id"] for item in payload["results"]]
        cli_ids = [item.chunk_id for item in captured_cli[0].results]
        assert cli_ids == rest_ids
    finally:
        await _cleanup()
