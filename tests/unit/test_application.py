import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from rag_demo.application import ApplicationNotStartedError, RAGApplication
from rag_demo.config import Settings
from rag_demo.search_service import SearchRequest, SearchResponse


def test_application_rejects_resource_access_before_start() -> None:
    application = RAGApplication(Settings())

    with pytest.raises(ApplicationNotStartedError, match="not started"):
        _ = application.database


@pytest.mark.asyncio
async def test_application_rewrite_boundary_preserves_original_and_searches_effective_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dictionary = tmp_path / "aliases.json"
    dictionary.write_text(
        json.dumps({"version": 1, "aliases": {"PG": "PostgreSQL"}}),
        encoding="utf-8",
    )
    settings = Settings().model_copy(update={"query_aliases_path": dictionary})
    application = RAGApplication(settings)
    searched_queries: list[str] = []

    async def fake_search(request: SearchRequest) -> SearchResponse:
        searched_queries.append(request.query)
        return cast(
            SearchResponse,
            SimpleNamespace(query=request.query, results=()),
        )

    monkeypatch.setattr(application, "search", fake_search)

    execution = await application.search_query(SearchRequest(query="PG 事务"), rewrite=True)

    assert searched_queries == ["PostgreSQL (PG) 事务"]
    assert execution.rewrite_enabled is True
    assert execution.rewrite.original_query == "PG 事务"
    assert execution.response.query == "PostgreSQL (PG) 事务"


@pytest.mark.asyncio
async def test_application_disabled_rewrite_does_not_require_dictionary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings().model_copy(
        update={"query_aliases_path": tmp_path / "does-not-exist.json"}
    )
    application = RAGApplication(settings)

    async def fake_search(request: SearchRequest) -> SearchResponse:
        return cast(
            SearchResponse,
            SimpleNamespace(query=request.query, results=()),
        )

    monkeypatch.setattr(application, "search", fake_search)

    execution = await application.search_query(SearchRequest(query="原始查询"), rewrite=False)

    assert execution.rewrite_enabled is False
    assert execution.rewrite.effective_query == "原始查询"
