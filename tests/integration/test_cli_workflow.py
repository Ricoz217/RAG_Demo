from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_demo.asyncio_compat import run_async
from rag_demo.cli import app
from rag_demo.config import Settings
from rag_demo.db import Database

pytestmark = pytest.mark.integration
SOURCE_REPO = "https://example.test/cli-fixture"
IDEMPOTENCY_KEY = "cli-workflow-ingestion"
runner = CliRunner()


async def _cleanup() -> None:
    settings = Settings()
    async with Database(settings.database_url.get_secret_value()) as database:
        async with database.connection() as connection:
            await connection.execute(
                "DELETE FROM ingestion_requests WHERE idempotency_key = %s",
                (IDEMPOTENCY_KEY,),
            )
            await connection.execute(
                "DELETE FROM documents WHERE source_repo = %s",
                (SOURCE_REPO,),
            )


def test_real_cli_ingest_rebuild_search_compare_and_doctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "demo.md").write_text(
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
    run_async(_cleanup())

    try:
        ingest = runner.invoke(
            app,
            [
                "ingest",
                str(source),
                "--source-repo",
                SOURCE_REPO,
                "--source-commit",
                "cli-commit",
                "--source-root",
                str(tmp_path),
                "--language",
                "zh",
                "--idempotency-key",
                IDEMPOTENCY_KEY,
            ],
        )
        assert ingest.exit_code == 0, ingest.output
        assert "Documents" in ingest.output
        assert "2" in ingest.output

        rebuild = runner.invoke(app, ["bm25", "rebuild"])
        assert rebuild.exit_code == 0, rebuild.output
        assert "BM25 generation" in rebuild.output

        search = runner.invoke(
            app,
            [
                "search",
                "FastAPI 如何接收 JSON 请求体？",
                "--no-rerank",
                "--debug",
            ],
        )
        assert search.exit_code == 0, search.output
        assert "Final results" in search.output
        assert "Dense Top-K" in search.output
        assert "BM25 Top-K" in search.output
        assert "RRF Top-K" in search.output
        assert "demo.md" in search.output

        rewritten_search = runner.invoke(
            app,
            [
                "search",
                "如何声明请求体？",
                "--rewrite",
                "--no-rerank",
            ],
        )
        assert rewritten_search.exit_code == 0, rewritten_search.output
        assert "Rewrite: enabled" in rewritten_search.output
        assert "alias:请求体" in rewritten_search.output
        assert "Request Body" in rewritten_search.output

        compare = runner.invoke(
            app,
            ["compare", "OAuth2PasswordBearer 有什么作用？"],
        )
        assert compare.exit_code == 0, compare.output
        assert "BM25 only" in compare.output
        assert "Dense only" in compare.output
        assert "Hybrid RRF" in compare.output
        assert "Hybrid RRF + Reranker" in compare.output

        rewrite_comparison = runner.invoke(
            app,
            ["compare-rewrite", "如何声明请求体？", "--no-rerank"],
        )
        assert rewrite_comparison.exit_code == 0, rewrite_comparison.output
        assert "Final Top-K without Rewrite" in rewrite_comparison.output
        assert "Final Top-K with Rewrite" in rewrite_comparison.output
        assert "membership and rank" in rewrite_comparison.output
        assert "comparison" in rewrite_comparison.output
        assert "Same final order:" in rewrite_comparison.output

        doctor = runner.invoke(app, ["doctor"])
        assert doctor.exit_code == 0, doctor.output
        assert "Embedding API" in doctor.output
        assert "Reranker API" in doctor.output
        assert "BM25 index" in doctor.output
        assert "PASS" in doctor.output
    finally:
        run_async(_cleanup())
