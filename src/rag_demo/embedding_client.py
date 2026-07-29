"""Thin asynchronous client for llama.cpp's OpenAI-compatible embeddings API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from rag_demo.config import Settings

EmbeddingVector = NDArray[np.float32]


class EmbeddingClientError(RuntimeError):
    """Raised when the embedding service fails or returns an invalid payload."""


class EmbeddingClient:
    """Batch text inputs and validate ordered finite embedding vectors."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        endpoint: str,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int,
        timeout_seconds: float,
    ) -> None:
        self._http_client = http_client
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._timeout = httpx.Timeout(
            timeout_seconds,
            connect=5.0,
            write=15.0,
            pool=5.0,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @classmethod
    def from_settings(
        cls,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> EmbeddingClient:
        """Build a client from validated shared application settings."""
        return cls(
            http_client=http_client,
            endpoint=settings.embedding_endpoint,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        """Embed inputs in configured batches while preserving input order."""
        if not texts:
            return ()

        vectors: list[EmbeddingVector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return tuple(vectors)

    async def embed_query(self, query: str) -> EmbeddingVector:
        """Embed one query using the same model and validation path."""
        return (await self.embed([query]))[0]

    async def _embed_batch(self, texts: list[str]) -> tuple[EmbeddingVector, ...]:
        try:
            response = await self._http_client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": texts,
                    "encoding_format": "float",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingClientError(
                f"embedding request failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingClientError(f"embedding request failed: {type(exc).__name__}") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise EmbeddingClientError("embedding response is not valid JSON") from exc
        return self._validate_batch(payload, expected_count=len(texts))

    def _validate_batch(
        self,
        payload: Any,
        *,
        expected_count: int,
    ) -> tuple[EmbeddingVector, ...]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise EmbeddingClientError("embedding response must contain a data list")

        data: list[Any] = payload["data"]
        if len(data) != expected_count:
            raise EmbeddingClientError(
                f"embedding vector count {len(data)} does not match input count {expected_count}"
            )

        ordered: list[EmbeddingVector | None] = [None] * expected_count
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingClientError("embedding data item must be an object")

            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise EmbeddingClientError("embedding index must be an integer")
            if index < 0 or index >= expected_count:
                raise EmbeddingClientError(f"embedding index {index} is out of range")
            if ordered[index] is not None:
                raise EmbeddingClientError(f"duplicate embedding index {index}")

            try:
                vector = np.asarray(item.get("embedding"), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise EmbeddingClientError(f"embedding at index {index} is not numeric") from exc
            if vector.ndim != 1 or vector.shape[0] != self._dimensions:
                raise EmbeddingClientError(
                    f"expected {self._dimensions} embedding dimensions at index {index}"
                )
            if not np.isfinite(vector).all():
                raise EmbeddingClientError(f"non-finite embedding at index {index}")
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise EmbeddingClientError("embedding response has missing indexes")
        return tuple(vector for vector in ordered if vector is not None)
