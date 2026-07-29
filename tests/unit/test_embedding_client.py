import json
from collections.abc import Callable

import httpx
import numpy as np
import pytest

from rag_demo.embedding_client import EmbeddingClient, EmbeddingClientError


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    dimensions: int = 3,
    batch_size: int = 2,
) -> tuple[EmbeddingClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedding_client = EmbeddingClient(
        http_client=http_client,
        endpoint="http://embedding.test/v1/embeddings",
        api_key="test-key",
        model="test-model",
        dimensions=dimensions,
        batch_size=batch_size,
        timeout_seconds=60,
    )
    return embedding_client, http_client


@pytest.mark.asyncio
async def test_embed_batches_requests_and_restores_response_index_order() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        data = [
            {
                "index": index,
                "embedding": [float(len(text))] * 3,
            }
            for index, text in reversed(list(enumerate(payload["input"])))
        ]
        return httpx.Response(
            200,
            json={"object": "list", "model": "test-model", "data": data},
        )

    client, http_client = _client(handler)
    async with http_client:
        vectors = await client.embed(["a", "bb", "ccc"])

    assert [request["input"] for request in requests] == [["a", "bb"], ["ccc"]]
    assert all(request["encoding_format"] == "float" for request in requests)
    assert all(request["model"] == "test-model" for request in requests)
    assert [vector.tolist() for vector in vectors] == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ]
    assert all(vector.dtype == np.float32 for vector in vectors)


@pytest.mark.asyncio
async def test_embed_empty_input_does_not_make_http_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client, http_client = _client(handler)
    async with http_client:
        vectors = await client.embed([])

    assert vectors == ()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([{"index": 0, "embedding": [1.0, 2.0, 3.0]}], "vector count"),
        (
            [
                {"index": 0, "embedding": [1.0, 2.0, 3.0]},
                {"index": 0, "embedding": [4.0, 5.0, 6.0]},
            ],
            "duplicate embedding index",
        ),
        (
            [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 1, "embedding": [4.0, 5.0, 6.0]},
            ],
            "expected 3 embedding dimensions",
        ),
        (
            [
                {"index": 0, "embedding": [1.0, 2.0, float("nan")]},
                {"index": 1, "embedding": [4.0, 5.0, 6.0]},
            ],
            "non-finite embedding",
        ),
    ],
)
@pytest.mark.asyncio
async def test_embed_rejects_invalid_response_data(
    data: list[dict[str, object]],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"object": "list", "model": "test-model", "data": data}).encode(),
            headers={"Content-Type": "application/json"},
        )

    client, http_client = _client(handler)
    async with http_client:
        with pytest.raises(EmbeddingClientError, match=message):
            await client.embed(["first", "second"])


@pytest.mark.asyncio
async def test_embed_wraps_http_failure_without_exposing_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client, http_client = _client(handler)
    async with http_client:
        with pytest.raises(EmbeddingClientError) as captured:
            await client.embed(["text"])

    assert "503" in str(captured.value)
    assert "test-key" not in str(captured.value)
