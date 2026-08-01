"""Shared application resource lifecycle for CLI and FastAPI."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import httpx
from psycopg.types.json import Jsonb

from rag_demo.bm25_retriever import (
    BM25BuildResult,
    BM25IndexError,
    BM25IndexManager,
    BM25Retriever,
)
from rag_demo.chunker import MarkdownChunker
from rag_demo.config import Settings
from rag_demo.db import Database
from rag_demo.dense_retriever import DenseQueryService, DenseRetriever
from rag_demo.embedding_client import EmbeddingClient
from rag_demo.ingest_service import (
    DocumentIngestor,
    IdempotencyConflictError,
    IdempotencyInProgressError,
)
from rag_demo.query_rewriter import (
    QueryRewriteExperiment,
    QueryRewriter,
    QueryRewriteResult,
    QueryRewriteSearchService,
    QuerySearchResponse,
)
from rag_demo.reranker_client import RerankerClient
from rag_demo.search_service import (
    HybridRecallService,
    HybridSearchService,
    SearchRequest,
    SearchResponse,
)


class ApplicationNotStartedError(RuntimeError):
    """Raised when a managed resource is accessed outside its lifecycle."""


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One observable infrastructure check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Complete infrastructure readiness report."""

    checks: tuple[DoctorCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class RAGApplication:
    """Own shared I/O resources and compose all core services."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._database: Database | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._reranker_client: RerankerClient | None = None
        self._chunker: MarkdownChunker | None = None
        self._dense_retriever: DenseRetriever | None = None
        self._dense_service: DenseQueryService | None = None
        self._bm25_manager: BM25IndexManager | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._search_service: HybridSearchService | None = None
        self._query_rewrite_search_service: QueryRewriteSearchService | None = None

    async def open(self) -> None:
        """Open the database pool and reusable HTTP client once."""
        if self._database is not None:
            return

        database = Database(self.settings.database_url.get_secret_value())
        http_client = httpx.AsyncClient()
        try:
            await database.open()
        except BaseException:
            await http_client.aclose()
            await database.close()
            raise

        self._database = database
        self._http_client = http_client
        self._embedding_client = EmbeddingClient.from_settings(http_client, self.settings)
        self._reranker_client = RerankerClient.from_settings(http_client, self.settings)
        self._chunker = MarkdownChunker(
            target_chars=self.settings.chunk_target_chars,
            max_chars=self.settings.chunk_max_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        self._dense_retriever = DenseRetriever(
            database=database,
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.settings.embedding_dimensions,
            hnsw_ef_search=self.settings.hnsw_ef_search,
        )
        self._dense_service = DenseQueryService(
            embedding_client=self._embedding_client,
            retriever=self._dense_retriever,
        )
        self._bm25_manager = BM25IndexManager(
            database=database,
            index_root=self.settings.bm25_index_path,
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.settings.embedding_dimensions,
        )

    async def close(self) -> None:
        """Close shared resources and discard loaded derived state."""
        database = self._database
        http_client = self._http_client
        self._search_service = None
        self._query_rewrite_search_service = None
        self._bm25_retriever = None
        self._bm25_manager = None
        self._dense_service = None
        self._dense_retriever = None
        self._chunker = None
        self._reranker_client = None
        self._embedding_client = None
        self._http_client = None
        self._database = None

        if http_client is not None:
            await http_client.aclose()
        if database is not None:
            await database.close()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    @property
    def database(self) -> Database:
        return self._required(self._database)

    @property
    def embedding_client(self) -> EmbeddingClient:
        return self._required(self._embedding_client)

    @property
    def reranker_client(self) -> RerankerClient:
        return self._required(self._reranker_client)

    @property
    def bm25_manager(self) -> BM25IndexManager:
        return self._required(self._bm25_manager)

    @property
    def dense_service(self) -> DenseQueryService:
        return self._required(self._dense_service)

    @property
    def bm25_retriever(self) -> BM25Retriever | None:
        return self._bm25_retriever

    @property
    def bm25_ready(self) -> bool:
        return self._bm25_retriever is not None and self._search_service is not None

    def create_ingestor(
        self,
        *,
        source_repo: str,
        source_commit: str,
        source_root: Path,
        language: str,
    ) -> DocumentIngestor:
        """Create a source-specific ingestor over shared resources."""
        return DocumentIngestor(
            database=self.database,
            embedding_client=self.embedding_client,
            chunker=self._required(self._chunker),
            source_repo=source_repo,
            source_commit=source_commit,
            source_root=source_root,
            language=language,
        )

    async def rebuild_bm25(self) -> BM25BuildResult:
        """Publish and immediately load a fresh BM25 generation."""
        result = await self.bm25_manager.rebuild()
        await self.load_bm25()
        return result

    async def rebuild_bm25_idempotent(
        self,
        *,
        idempotency_key: str,
    ) -> BM25BuildResult:
        """Persist REST rebuild idempotency while publishing atomically."""
        request_hash = self._bm25_request_hash()
        stored_key = f"bm25-rebuild:{idempotency_key}"
        cached = await self._claim_bm25_rebuild(
            stored_key=stored_key,
            request_hash=request_hash,
            external_key=idempotency_key,
        )
        if cached is not None:
            await self.load_bm25()
            return cached

        try:
            result = await self.rebuild_bm25()
            await self._complete_bm25_rebuild(
                stored_key=stored_key,
                request_hash=request_hash,
                result=result,
            )
            return result
        except Exception:
            await self._fail_bm25_rebuild(
                stored_key=stored_key,
                request_hash=request_hash,
            )
            raise

    async def load_bm25(self) -> BM25Retriever:
        """Load CURRENT and refresh the final search service."""
        retriever = await self.bm25_manager.load()
        self._bm25_retriever = retriever
        recall = HybridRecallService(
            dense_service=self._required(self._dense_service),
            bm25_retriever=retriever,
        )
        self._search_service = HybridSearchService(
            recall_service=recall,
            reranker_client=self.reranker_client,
        )
        return retriever

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search with CURRENT, loading it lazily on first use."""
        if self._search_service is None:
            await self.load_bm25()
        return await self._required(self._search_service).search(request)

    async def search_query(
        self,
        request: SearchRequest,
        *,
        rewrite: bool = False,
    ) -> QuerySearchResponse:
        """Optionally rewrite immediately before the unchanged retrieval pipeline."""
        if not rewrite:
            response = await self.search(request)
            return QuerySearchResponse(
                rewrite_enabled=False,
                rewrite=QueryRewriteResult.disabled(request.query),
                response=response,
            )
        return await self._rewrite_search_service().search(request, rewrite=True)

    async def compare_query_rewrite(
        self,
        request: SearchRequest,
    ) -> QueryRewriteExperiment:
        """Run a neutral without/with Rewrite experiment for one query."""
        return await self._rewrite_search_service().compare(request)

    def _rewrite_search_service(self) -> QueryRewriteSearchService:
        if self._query_rewrite_search_service is None:
            rewriter = QueryRewriter.from_json(self.settings.query_aliases_path)
            self._query_rewrite_search_service = QueryRewriteSearchService(
                search_provider=self,
                rewriter=rewriter,
            )
        return self._query_rewrite_search_service

    async def doctor(self) -> DoctorReport:
        """Exercise every configured local dependency without exposing secrets."""
        checks: list[DoctorCheck] = [
            DoctorCheck(
                name="Python",
                passed=sys.version_info[:2] == (3, 12),
                detail=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            )
        ]

        try:
            status = await self.database.status()
            checks.append(
                DoctorCheck(
                    name="PostgreSQL + pgvector",
                    passed=status.hnsw_index_present
                    and status.embedding_dimensions == self.settings.embedding_dimensions,
                    detail=(
                        f"PostgreSQL {status.server_version}, pgvector "
                        f"{status.vector_version}, chunks={status.chunk_count}"
                    ),
                )
            )
        except Exception as exc:
            checks.append(_failed_check("PostgreSQL + pgvector", exc))

        try:
            vector = await self.embedding_client.embed_query("RAG doctor test")
            checks.append(
                DoctorCheck(
                    name="Embedding API",
                    passed=len(vector) == self.settings.embedding_dimensions,
                    detail=(f"{self.embedding_client.model}, dimensions={len(vector)}"),
                )
            )
        except Exception as exc:
            checks.append(_failed_check("Embedding API", exc))

        try:
            scores = await self.reranker_client.rerank(
                "FastAPI 请求体",
                ("Pydantic 请求体模型", "无关文本"),
            )
            checks.append(
                DoctorCheck(
                    name="Reranker API",
                    passed=len(scores) == 2,
                    detail=f"{self.reranker_client.model}, scores={len(scores)}",
                )
            )
        except Exception as exc:
            checks.append(_failed_check("Reranker API", exc))

        try:
            bm25 = await self.load_bm25()
            checks.append(
                DoctorCheck(
                    name="BM25 index",
                    passed=True,
                    detail=f"{bm25.generation}, chunks={bm25.chunk_count}",
                )
            )
        except BM25IndexError as exc:
            checks.append(
                DoctorCheck(
                    name="BM25 index",
                    passed=False,
                    detail=str(exc),
                )
            )

        checks.append(await self._corpus_model_check())
        return DoctorReport(checks=tuple(checks))

    async def _corpus_model_check(self) -> DoctorCheck:
        try:
            async with self.database.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT DISTINCT embedding_model, embedding_dimensions
                    FROM chunks
                    ORDER BY embedding_model, embedding_dimensions
                    """
                )
                rows = await cursor.fetchall()
            if not rows:
                return DoctorCheck(
                    name="Corpus vector space",
                    passed=False,
                    detail="corpus is empty",
                )
            spaces = {(row["embedding_model"], row["embedding_dimensions"]) for row in rows}
            expected = (
                self.settings.embedding_model,
                self.settings.embedding_dimensions,
            )
            return DoctorCheck(
                name="Corpus vector space",
                passed=spaces == {expected},
                detail=", ".join(f"{model}/{dimensions}" for model, dimensions in spaces),
            )
        except Exception as exc:
            return _failed_check("Corpus vector space", exc)

    def _bm25_request_hash(self) -> str:
        canonical = json.dumps(
            {
                "operation": "bm25-rebuild",
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "index_root": self.settings.bm25_index_path.resolve().as_posix(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _claim_bm25_rebuild(
        self,
        *,
        stored_key: str,
        request_hash: str,
        external_key: str,
    ) -> BM25BuildResult | None:
        if not external_key.strip():
            raise ValueError("idempotency_key must not be empty")
        async with self.database.connection() as connection:
            async with connection.transaction():
                inserted = await connection.execute(
                    """
                    INSERT INTO ingestion_requests (
                        idempotency_key,
                        request_hash,
                        status
                    )
                    VALUES (%s, %s, 'processing')
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING idempotency_key
                    """,
                    (stored_key, request_hash),
                )
                if await inserted.fetchone() is not None:
                    return None

                cursor = await connection.execute(
                    """
                    SELECT request_hash, status, result
                    FROM ingestion_requests
                    WHERE idempotency_key = %s
                    FOR UPDATE
                    """,
                    (stored_key,),
                )
                existing = await cursor.fetchone()
                if existing is None:
                    raise RuntimeError("BM25 idempotency row disappeared during claim")
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different BM25 rebuild"
                    )
                if existing["status"] == "completed":
                    return BM25BuildResult.from_json(existing["result"])
                if existing["status"] == "processing":
                    raise IdempotencyInProgressError(
                        "idempotent BM25 rebuild is already processing"
                    )
                if existing["status"] != "failed":
                    raise RuntimeError(f"unknown BM25 rebuild status: {existing['status']}")
                await connection.execute(
                    """
                    UPDATE ingestion_requests
                    SET status = 'processing',
                        result = NULL,
                        updated_at = now()
                    WHERE idempotency_key = %s
                    """,
                    (stored_key,),
                )
                return None

    async def _complete_bm25_rebuild(
        self,
        *,
        stored_key: str,
        request_hash: str,
        result: BM25BuildResult,
    ) -> None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'completed',
                    result = %s,
                    updated_at = now()
                WHERE idempotency_key = %s
                  AND request_hash = %s
                  AND status = 'processing'
                """,
                (Jsonb(result.to_json()), stored_key, request_hash),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("BM25 idempotency completion did not update one row")

    async def _fail_bm25_rebuild(
        self,
        *,
        stored_key: str,
        request_hash: str,
    ) -> None:
        async with self.database.connection() as connection:
            await connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'failed',
                    updated_at = now()
                WHERE idempotency_key = %s
                  AND request_hash = %s
                  AND status = 'processing'
                """,
                (stored_key, request_hash),
            )

    @staticmethod
    def _required[ResourceT](resource: ResourceT | None) -> ResourceT:
        if resource is None:
            raise ApplicationNotStartedError("RAG application is not started")
        return resource


def _failed_check(name: str, error: Exception) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        passed=False,
        detail=f"{type(error).__name__}: check failed",
    )
