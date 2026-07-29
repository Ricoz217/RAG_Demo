import httpx
import pytest

from rag_demo.config import Settings
from rag_demo.reranker_client import RerankerClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_reranker_service_scores_relevant_document_higher() -> None:
    settings = Settings()
    timeout = httpx.Timeout(
        settings.reranker_timeout_seconds,
        connect=5.0,
        write=15.0,
        pool=5.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        client = RerankerClient.from_settings(http_client, settings)
        scores = await client.rerank(
            "FastAPI 如何接收 JSON 请求体？",
            [
                "使用 Pydantic 模型声明 JSON 请求体。",
                "这是一段无关的数据库迁移说明。",
            ],
        )

    assert len(scores) == 2
    assert scores[0] > scores[1]
