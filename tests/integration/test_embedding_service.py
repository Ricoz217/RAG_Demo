import httpx
import numpy as np
import pytest

from rag_demo.config import Settings
from rag_demo.embedding_client import EmbeddingClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_embedding_service_returns_ordered_1024_dimension_vectors() -> None:
    settings = Settings()
    timeout = httpx.Timeout(
        settings.embedding_timeout_seconds,
        connect=5.0,
        write=15.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        client = EmbeddingClient.from_settings(http_client, settings)
        vectors = await client.embed(["FastAPI 请求体", "OAuth2PasswordBearer"])

    assert len(vectors) == 2
    assert all(vector.shape == (1024,) for vector in vectors)
    assert all(vector.dtype == np.float32 for vector in vectors)
    assert all(np.isfinite(vector).all() for vector in vectors)
