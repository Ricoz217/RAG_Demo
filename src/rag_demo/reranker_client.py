"""Thin asynchronous client for llama.cpp's reranking endpoint."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from rag_demo.config import Settings


class RerankerClientError(RuntimeError):
    """Raised when the reranker fails or returns an invalid payload."""


class RerankerClient:
    """Map llama.cpp relevance scores back to input document order."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self._http_client = http_client
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout = httpx.Timeout(
            timeout_seconds,
            connect=5.0,
            write=15.0,
            pool=5.0,
        )

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def from_settings(
        cls,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> RerankerClient:
        """Build a client from validated shared application settings."""
        return cls(
            http_client=http_client,
            endpoint=settings.reranker_endpoint,
            api_key=settings.reranker_api_key.get_secret_value(),
            model=settings.reranker_model,
            timeout_seconds=settings.reranker_timeout_seconds,
        )

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> tuple[float, ...]:
        """Return finite relevance scores aligned with input documents."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if not documents:
            return ()

        try:
            response = await self._http_client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "query": query,
                    "documents": list(documents),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RerankerClientError(
                f"reranker request failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RerankerClientError(f"reranker request failed: {type(exc).__name__}") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise RerankerClientError("reranker response is not valid JSON") from exc
        return self._validate_response(payload, expected_count=len(documents))

    @staticmethod
    def _validate_response(
        payload: Any,
        *,
        expected_count: int,
    ) -> tuple[float, ...]:
        """看一下结果数量对不对，看一下结果的 index 和 relevance_score 对不对"""

        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RerankerClientError("reranker response must contain a results list")

        results: list[Any] = payload["results"]
        if len(results) != expected_count:
            raise RerankerClientError(
                f"reranker score count {len(results)} does not match "
                f"document count {expected_count}"
            )

        ordered: list[float | None] = [None] * expected_count
        for item in results:
            if not isinstance(item, dict):
                raise RerankerClientError("reranker result item must be an object")
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RerankerClientError("reranker index must be an integer")
            if index < 0 or index >= expected_count:
                raise RerankerClientError(f"reranker index {index} is out of range")
            if ordered[index] is not None:
                raise RerankerClientError(f"duplicate reranker index {index}")

            score_value = item.get("relevance_score")
            if (
                not isinstance(score_value, (int, float))
                or isinstance(score_value, bool)
                or not math.isfinite(float(score_value))
            ):
                raise RerankerClientError(f"reranker score at index {index} must be finite")
            ordered[index] = float(score_value)

        if any(score is None for score in ordered):
            raise RerankerClientError("reranker response has missing indexes")
        return tuple(score for score in ordered if score is not None)
