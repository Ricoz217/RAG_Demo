"""Public synchronous and asynchronous Python SDK facades."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from rag_demo.application import DoctorReport, RAGApplication
from rag_demo.asyncio_compat import create_compatible_event_loop
from rag_demo.bm25_retriever import BM25BuildResult, BM25Retriever
from rag_demo.config import Settings
from rag_demo.dense_retriever import DenseSearchMode
from rag_demo.ingest_service import DocumentIngestor, IngestResult
from rag_demo.migrations import apply_migrations
from rag_demo.query_rewriter import QueryRewriteExperiment, QuerySearchResponse
from rag_demo.search_service import SearchRequest, SearchResponse


class RAGSDKError(RuntimeError):
    """Base error for incorrect SDK lifecycle usage."""


class RAGNotStartedError(RAGSDKError):
    """Raised when an SDK operation is attempted before opening the engine."""


class RAGClosedError(RAGSDKError):
    """Raised when a permanently closed synchronous engine is reused."""


class RAGSyncInAsyncContextError(RAGSDKError):
    """Raised when the synchronous facade is used inside a running event loop."""


class _Application(Protocol):
    """Application behavior required by the public facades."""

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def search(self, request: SearchRequest) -> SearchResponse: ...

    async def search_query(
        self,
        request: SearchRequest,
        *,
        rewrite: bool = False,
    ) -> QuerySearchResponse: ...

    async def compare_query_rewrite(self, request: SearchRequest) -> QueryRewriteExperiment: ...

    def create_ingestor(
        self,
        *,
        source_repo: str,
        source_commit: str,
        source_root: Path,
        language: str,
    ) -> DocumentIngestor: ...

    async def rebuild_bm25(self) -> BM25BuildResult: ...

    async def load_bm25(self) -> BM25Retriever: ...

    async def doctor(self) -> DoctorReport: ...


class AsyncRAG:
    """Long-lived asynchronous RAG engine for agents and async web applications."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        application: _Application | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._application = application or RAGApplication(self.settings)
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Return whether the managed application resources are open."""
        return self._is_open

    async def open(self) -> Self:
        """Open shared database and HTTP resources once."""
        if not self._is_open:
            await self._application.open()
            self._is_open = True
        return self

    async def close(self) -> None:
        """Close all shared resources; repeated calls are harmless."""
        if not self._is_open:
            return
        try:
            await self._application.close()
        finally:
            self._is_open = False

    async def __aenter__(self) -> Self:
        return await self.open()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def init_database(self) -> tuple[str, ...]:
        """Apply pending database migrations without opening application resources."""
        return await apply_migrations(self.settings.database_url.get_secret_value())

    async def search(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> SearchResponse:
        """Run hybrid retrieval and return the complete typed search response."""
        self._require_open()
        request = self._search_request(
            query,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_rank_constant=rrf_rank_constant,
            rerank_top_k=rerank_top_k,
            final_top_k=final_top_k,
            dense_mode=dense_mode,
            use_reranker=use_reranker,
            debug=debug,
        )
        return await self._application.search(request)

    async def search_query(
        self,
        query: str,
        *,
        rewrite: bool = False,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> QuerySearchResponse:
        """
        Optionally rewrite one query and retain the rewrite evidence.
        这命名的是一个什么玩意啊
        """
        self._require_open()
        request = self._search_request(
            query,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_rank_constant=rrf_rank_constant,
            rerank_top_k=rerank_top_k,
            final_top_k=final_top_k,
            dense_mode=dense_mode,
            use_reranker=use_reranker,
            debug=debug,
        )
        return await self._application.search_query(request, rewrite=rewrite)

    async def compare_rewrite(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> QueryRewriteExperiment:
        """Run the same query without and with deterministic rewrite."""
        self._require_open()
        request = self._search_request(
            query,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_rank_constant=rrf_rank_constant,
            rerank_top_k=rerank_top_k,
            final_top_k=final_top_k,
            dense_mode=dense_mode,
            use_reranker=use_reranker,
            debug=debug,
        )
        return await self._application.compare_query_rewrite(request)

    async def ingest(
        self,
        source: Path,
        *,
        source_repo: str,
        source_commit: str,
        source_root: Path | None = None,
        language: str = "zh",
        idempotency_key: str | None = None,
    ) -> IngestResult:
        """Ingest a Markdown file or tree using one explicit source identity."""
        self._require_open()
        root = source_root or _default_source_root(source)
        ingestor = self._application.create_ingestor(
            source_repo=source_repo,
            source_commit=source_commit,
            source_root=root.resolve(),
            language=language,
        )
        return await ingestor.ingest_path(source, idempotency_key=idempotency_key)

    async def rebuild_bm25(self) -> BM25BuildResult:
        """Rebuild BM25 from PostgreSQL and load the published generation."""
        self._require_open()
        return await self._application.rebuild_bm25()

    async def reload_bm25(self) -> BM25Retriever:
        """Reload CURRENT after another process publishes a BM25 generation."""
        self._require_open()
        return await self._application.load_bm25()

    async def doctor(self) -> DoctorReport:
        """Run all configured local infrastructure checks."""
        self._require_open()
        return await self._application.doctor()

    def _require_open(self) -> None:
        if not self._is_open:
            raise RAGNotStartedError(
                "RAG engine is not open; use 'async with AsyncRAG() as rag:' or await rag.open()"
            )

    def _search_request(
        self,
        query: str,
        *,
        dense_top_k: int | None,
        bm25_top_k: int | None,
        rrf_rank_constant: int | None,
        rerank_top_k: int | None,
        final_top_k: int | None,
        dense_mode: DenseSearchMode | str,
        use_reranker: bool,
        debug: bool,
    ) -> SearchRequest:
        return SearchRequest(
            query=query,
            dense_top_k=self.settings.dense_top_k if dense_top_k is None else dense_top_k,
            bm25_top_k=self.settings.bm25_top_k if bm25_top_k is None else bm25_top_k,
            rrf_rank_constant=(
                self.settings.rrf_rank_constant if rrf_rank_constant is None else rrf_rank_constant
            ),
            rerank_top_k=self.settings.rerank_top_k if rerank_top_k is None else rerank_top_k,
            final_top_k=self.settings.final_top_k if final_top_k is None else final_top_k,
            dense_mode=DenseSearchMode(dense_mode),
            use_reranker=use_reranker,
            debug=debug,
        )


class RAG:
    """Synchronous RAG engine for scripts; not thread-safe or async-context safe."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        application: _Application | None = None,
    ) -> None:
        self._async_rag = AsyncRAG(settings, application=application)
        self._runner = asyncio.Runner(loop_factory=create_compatible_event_loop)
        self._closed = False

    @property
    def settings(self) -> Settings:
        """Return the validated engine settings."""
        return self._async_rag.settings

    @property
    def is_open(self) -> bool:
        """Return whether the managed resources are open."""
        return self._async_rag.is_open

    def open(self) -> Self:
        """Open the engine on its persistent private event loop."""
        self._prepare_call(require_open=False)
        self._runner.run(self._async_rag.open())
        return self

    def close(self) -> None:
        """Close resources and permanently dispose of the private event loop."""
        if self._closed:
            return
        if self.is_open:
            self._ensure_no_running_loop()
            try:
                self._runner.run(self._async_rag.close())
            finally:
                self._runner.close()
                self._closed = True
            return
        self._runner.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def init_database(self) -> tuple[str, ...]:
        """Synchronously apply pending database migrations."""
        self._prepare_call(require_open=False)
        return self._runner.run(self._async_rag.init_database())

    def search(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> SearchResponse:
        """Synchronously run hybrid retrieval."""
        self._prepare_call()
        return self._runner.run(
            self._async_rag.search(
                query,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                rrf_rank_constant=rrf_rank_constant,
                rerank_top_k=rerank_top_k,
                final_top_k=final_top_k,
                dense_mode=dense_mode,
                use_reranker=use_reranker,
                debug=debug,
            )
        )

    def search_query(
        self,
        query: str,
        *,
        rewrite: bool = False,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> QuerySearchResponse:
        """Synchronously search with optional observable rewrite."""
        self._prepare_call()
        return self._runner.run(
            self._async_rag.search_query(
                query,
                rewrite=rewrite,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                rrf_rank_constant=rrf_rank_constant,
                rerank_top_k=rerank_top_k,
                final_top_k=final_top_k,
                dense_mode=dense_mode,
                use_reranker=use_reranker,
                debug=debug,
            )
        )

    def compare_rewrite(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_rank_constant: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        dense_mode: DenseSearchMode | str = DenseSearchMode.HNSW,
        use_reranker: bool = True,
        debug: bool = False,
    ) -> QueryRewriteExperiment:
        """Synchronously compare retrieval without and with rewrite."""
        self._prepare_call()
        return self._runner.run(
            self._async_rag.compare_rewrite(
                query,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                rrf_rank_constant=rrf_rank_constant,
                rerank_top_k=rerank_top_k,
                final_top_k=final_top_k,
                dense_mode=dense_mode,
                use_reranker=use_reranker,
                debug=debug,
            )
        )

    def ingest(
        self,
        source: Path,
        *,
        source_repo: str,
        source_commit: str,
        source_root: Path | None = None,
        language: str = "zh",
        idempotency_key: str | None = None,
    ) -> IngestResult:
        """Synchronously ingest one Markdown file or directory tree."""
        self._prepare_call()
        return self._runner.run(
            self._async_rag.ingest(
                source,
                source_repo=source_repo,
                source_commit=source_commit,
                source_root=source_root,
                language=language,
                idempotency_key=idempotency_key,
            )
        )

    def rebuild_bm25(self) -> BM25BuildResult:
        """Synchronously rebuild and load BM25."""
        self._prepare_call()
        return self._runner.run(self._async_rag.rebuild_bm25())

    def reload_bm25(self) -> BM25Retriever:
        """Synchronously reload the published BM25 generation."""
        self._prepare_call()
        return self._runner.run(self._async_rag.reload_bm25())

    def doctor(self) -> DoctorReport:
        """Synchronously run infrastructure checks."""
        self._prepare_call()
        return self._runner.run(self._async_rag.doctor())

    def _prepare_call(self, *, require_open: bool = True) -> None:
        self._ensure_no_running_loop()
        if self._closed:
            raise RAGClosedError("synchronous RAG engine is already closed; create a new RAG()")
        if require_open and not self.is_open:
            raise RAGNotStartedError(
                "RAG engine is not open; use 'with RAG() as rag:' or call rag.open()"
            )

    @staticmethod
    def _ensure_no_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RAGSyncInAsyncContextError(
            "RAG is synchronous and cannot run inside an active event loop; use AsyncRAG instead"
        )


def _default_source_root(source: Path) -> Path:
    """Use a directory as its own root and a file's parent as its root."""
    return source if source.is_dir() else source.parent
