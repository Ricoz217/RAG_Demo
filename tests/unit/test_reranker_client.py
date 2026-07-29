import json
from collections.abc import Callable

import httpx
import pytest

from rag_demo.reranker_client import RerankerClient, RerankerClientError


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[RerankerClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = RerankerClient(
        http_client=http_client,
        endpoint="http://reranker.test/reranking",
        api_key="test-key",
        model="test-reranker",
        timeout_seconds=60,
    )
    return reranker, http_client


@pytest.mark.asyncio
async def test_reranker_restores_scores_to_input_document_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload == {
            "query": "测试问题",
            "documents": ["first", "second", "third"],
        }
        return httpx.Response(
            200,
            json={
                "model": "test-reranker",
                "results": [
                    {"index": 2, "relevance_score": 0.3},
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.2},
                ],
            },
        )

    client, http_client = _client(handler)
    async with http_client:
        scores = await client.rerank("测试问题", ["first", "second", "third"])

    assert scores == pytest.approx((0.1, 0.2, 0.3))
    assert client.model == "test-reranker"


@pytest.mark.asyncio
async def test_reranker_empty_documents_do_not_make_http_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client, http_client = _client(handler)
    async with http_client:
        scores = await client.rerank("query", [])

    assert scores == ()


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([{"index": 0, "relevance_score": 1.0}], "score count"),
        (
            [
                {"index": 0, "relevance_score": 1.0},
                {"index": 0, "relevance_score": 2.0},
            ],
            "duplicate reranker index",
        ),
        (
            [
                {"index": 0, "relevance_score": float("nan")},
                {"index": 1, "relevance_score": 2.0},
            ],
            "finite",
        ),
        (
            [
                {"index": "0", "relevance_score": 1.0},
                {"index": 1, "relevance_score": 2.0},
            ],
            "index must be an integer",
        ),
    ],
)
@pytest.mark.asyncio
async def test_reranker_rejects_invalid_response(
    results: list[dict[str, object]],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"results": results}).encode(),
            headers={"Content-Type": "application/json"},
        )

    client, http_client = _client(handler)
    async with http_client:
        with pytest.raises(RerankerClientError, match=message):
            await client.rerank("query", ["first", "second"])


@pytest.mark.asyncio
async def test_reranker_wraps_http_failure_without_exposing_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client, http_client = _client(handler)
    async with http_client:
        with pytest.raises(RerankerClientError) as captured:
            await client.rerank("query", ["document"])

    assert "503" in str(captured.value)
    assert "test-key" not in str(captured.value)
