import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_demo import RAG, AsyncRAG, DenseSearchMode, SearchResponse, Settings
from rag_demo.application import DoctorReport
from rag_demo.bm25_retriever import BM25BuildResult
from rag_demo.ingest_service import IngestResult
from rag_demo.query_rewriter import QueryRewriteExperiment, QuerySearchResponse
from rag_demo.sdk import RAGNotStartedError, RAGSyncInAsyncContextError
from rag_demo.search_service import SearchRequest


def _search_response(query: str) -> SearchResponse:
    return cast(
        SearchResponse,
        SimpleNamespace(query=query, results=()),
    )


class FakeIngestor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []

    async def ingest_path(
        self,
        source: Path,
        *,
        idempotency_key: str | None = None,
    ) -> IngestResult:
        self.calls.append((source, idempotency_key))
        return cast(IngestResult, SimpleNamespace(document_count=1))


class FakeApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.open_count = 0
        self.close_count = 0
        self.loop_ids: list[int] = []
        self.search_requests: list[SearchRequest] = []
        self.query_requests: list[tuple[SearchRequest, bool]] = []
        self.compare_requests: list[SearchRequest] = []
        self.ingestor_arguments: list[dict[str, Any]] = []
        self.ingestor = FakeIngestor()

    async def open(self) -> None:
        self.open_count += 1
        self.loop_ids.append(id(asyncio.get_running_loop()))

    async def close(self) -> None:
        self.close_count += 1
        self.loop_ids.append(id(asyncio.get_running_loop()))

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.search_requests.append(request)
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return _search_response(request.query)

    async def search_query(
        self,
        request: SearchRequest,
        *,
        rewrite: bool = False,
    ) -> QuerySearchResponse:
        self.query_requests.append((request, rewrite))
        return cast(
            QuerySearchResponse,
            SimpleNamespace(response=_search_response(request.query), rewrite_enabled=rewrite),
        )

    async def compare_query_rewrite(self, request: SearchRequest) -> QueryRewriteExperiment:
        self.compare_requests.append(request)
        return cast(QueryRewriteExperiment, SimpleNamespace())

    def create_ingestor(
        self,
        *,
        source_repo: str,
        source_commit: str,
        source_root: Path,
        language: str,
    ) -> FakeIngestor:
        self.ingestor_arguments.append(
            {
                "source_repo": source_repo,
                "source_commit": source_commit,
                "source_root": source_root,
                "language": language,
            }
        )
        return self.ingestor

    async def rebuild_bm25(self) -> BM25BuildResult:
        return cast(BM25BuildResult, SimpleNamespace(generation="generation-1"))

    async def load_bm25(self) -> object:
        return object()

    async def doctor(self) -> DoctorReport:
        return DoctorReport(checks=())


@pytest.mark.asyncio
async def test_async_rag_context_uses_settings_defaults_and_closes() -> None:
    settings = Settings().model_copy(
        update={
            "dense_top_k": 11,
            "bm25_top_k": 12,
            "rrf_rank_constant": 33,
            "rerank_top_k": 9,
            "final_top_k": 4,
        }
    )
    application = FakeApplication(settings)

    async with AsyncRAG(settings, application=cast(Any, application)) as rag:
        response = await rag.search("如何声明请求体？", dense_mode="exact")

        assert response.query == "如何声明请求体？"
        assert rag.is_open is True

    request = application.search_requests[0]
    assert request.dense_top_k == 11
    assert request.bm25_top_k == 12
    assert request.rrf_rank_constant == 33
    assert request.rerank_top_k == 9
    assert request.final_top_k == 4
    assert request.dense_mode is DenseSearchMode.EXACT
    assert application.open_count == 1
    assert application.close_count == 1
    assert rag.is_open is False


@pytest.mark.asyncio
async def test_async_rag_exposes_rewrite_compare_and_ingestion(tmp_path: Path) -> None:
    settings = Settings()
    application = FakeApplication(settings)
    source = tmp_path / "docs"
    source.mkdir()

    async with AsyncRAG(settings, application=cast(Any, application)) as rag:
        rewritten = await rag.search_query("PG 事务", rewrite=True, use_reranker=False)
        comparison = await rag.compare_rewrite("PG 事务", final_top_k=3)
        ingested = await rag.ingest(
            source,
            source_repo="local-notes",
            source_commit="demo-1",
            language="zh-CN",
            idempotency_key="sdk-test",
        )
        rebuilt = await rag.rebuild_bm25()
        loaded = await rag.reload_bm25()
        report = await rag.doctor()

    assert rewritten.rewrite_enabled is True
    assert application.query_requests[0][0].use_reranker is False
    assert application.query_requests[0][1] is True
    assert comparison is not None
    assert application.compare_requests[0].final_top_k == 3
    assert application.ingestor_arguments == [
        {
            "source_repo": "local-notes",
            "source_commit": "demo-1",
            "source_root": source.resolve(),
            "language": "zh-CN",
        }
    ]
    assert application.ingestor.calls == [(source, "sdk-test")]
    assert ingested.document_count == 1
    assert rebuilt.generation == "generation-1"
    assert loaded is not None
    assert report.passed is True


@pytest.mark.asyncio
async def test_async_rag_requires_explicit_lifecycle() -> None:
    settings = Settings()
    rag = AsyncRAG(settings, application=cast(Any, FakeApplication(settings)))

    with pytest.raises(RAGNotStartedError, match="async with AsyncRAG"):
        await rag.search("query")


@pytest.mark.asyncio
async def test_async_rag_initializes_database_before_opening_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    application = FakeApplication(settings)
    calls: list[str] = []

    async def fake_apply_migrations(conninfo: str) -> tuple[str, ...]:
        calls.append(conninfo)
        return ("001_enable_vector", "002_create_tables")

    monkeypatch.setattr("rag_demo.sdk.apply_migrations", fake_apply_migrations)
    rag = AsyncRAG(settings, application=cast(Any, application))

    applied = await rag.init_database()

    assert applied == ("001_enable_vector", "002_create_tables")
    assert calls == [settings.database_url.get_secret_value()]
    assert application.open_count == 0
    assert rag.is_open is False


def test_sync_rag_reuses_one_event_loop_for_multiple_operations() -> None:
    settings = Settings()
    application = FakeApplication(settings)

    with RAG(settings, application=cast(Any, application)) as rag:
        first = rag.search("first")
        second = rag.search_query("second", rewrite=True)

        assert first.query == "first"
        assert second.rewrite_enabled is True
        assert rag.is_open is True

    assert application.open_count == 1
    assert application.close_count == 1
    assert len(set(application.loop_ids)) == 1
    assert rag.is_open is False


def test_sync_rag_initializes_database_without_opening_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    application = FakeApplication(settings)
    loop_ids: list[int] = []

    async def fake_apply_migrations(conninfo: str) -> tuple[str, ...]:
        assert conninfo == settings.database_url.get_secret_value()
        loop_ids.append(id(asyncio.get_running_loop()))
        return ()

    monkeypatch.setattr("rag_demo.sdk.apply_migrations", fake_apply_migrations)
    rag = RAG(settings, application=cast(Any, application))

    applied = rag.init_database()
    rag.open()
    rag.close()

    assert applied == ()
    assert application.open_count == 1
    assert loop_ids == [application.loop_ids[0]]


@pytest.mark.asyncio
async def test_sync_rag_rejects_calls_from_a_running_event_loop() -> None:
    settings = Settings()
    rag = RAG(settings, application=cast(Any, FakeApplication(settings)))

    with pytest.raises(RAGSyncInAsyncContextError, match="AsyncRAG"):
        rag.open()

    rag.close()
