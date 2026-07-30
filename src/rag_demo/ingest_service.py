"""Transactional, idempotent Markdown ingestion into PostgreSQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from rag_demo.chunker import Chunk, MarkdownChunker
from rag_demo.db import Database, Row
from rag_demo.embedding_client import EmbeddingVector
from rag_demo.markdown_parser import parse_markdown


class IdempotencyConflictError(RuntimeError):
    """Raised when one idempotency key is reused for another request."""


class IdempotencyInProgressError(RuntimeError):
    """Raised when the same idempotent request is already running."""


class EmbeddingProvider(Protocol):
    """Minimal embedding behavior required by the ingestion service."""

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def batch_size(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]: ...


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Observable counts and timings for one idempotent ingestion request."""

    document_count: int
    chunk_count: int
    embedded_chunk_count: int
    skipped_embedding_count: int
    deleted_chunk_count: int
    embedding_http_request_count: int
    parse_ms: float
    embedding_ms: float
    database_insert_ms: float
    total_ms: float

    def to_json(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Any) -> IngestResult:
        if not isinstance(value, dict):
            raise RuntimeError("stored ingestion result is not an object")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class _DocumentResult:
    chunk_count: int
    embedded_chunk_count: int
    skipped_embedding_count: int
    deleted_chunk_count: int
    embedding_http_request_count: int
    parse_ms: float
    embedding_ms: float
    database_insert_ms: float


class DocumentIngestor:
    """Parse, embed, and atomically upsert one Markdown path."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_client: EmbeddingProvider,
        chunker: MarkdownChunker,
        source_repo: str,
        source_commit: str,
        source_root: Path,
        language: str,
    ) -> None:
        self._database = database
        self._embedding_client = embedding_client
        self._chunker = chunker
        self._source_repo = source_repo
        self._source_commit = source_commit
        self._source_root = source_root.resolve()
        self._language = language

    async def ingest_path(
        self,
        source: Path,
        *,
        idempotency_key: str,
    ) -> IngestResult:
        """Ingest one Markdown file or a directory tree exactly once per key."""
        started = time.perf_counter()
        files = _markdown_files(source)
        request_hash = self._request_hash(source)
        cached = await self._claim_idempotency(idempotency_key, request_hash)
        if cached is not None:
            return cached

        try:
            document_results = [await self._ingest_document(path) for path in files]
            result = IngestResult(
                document_count=len(document_results),
                chunk_count=sum(item.chunk_count for item in document_results),
                embedded_chunk_count=sum(item.embedded_chunk_count for item in document_results),
                skipped_embedding_count=sum(
                    item.skipped_embedding_count for item in document_results
                ),
                deleted_chunk_count=sum(item.deleted_chunk_count for item in document_results),
                embedding_http_request_count=sum(
                    item.embedding_http_request_count for item in document_results
                ),
                parse_ms=sum(item.parse_ms for item in document_results),
                embedding_ms=sum(item.embedding_ms for item in document_results),
                database_insert_ms=sum(item.database_insert_ms for item in document_results),
                total_ms=(time.perf_counter() - started) * 1000,
            )
            await self._complete_idempotency(idempotency_key, request_hash, result)
            return result
        except Exception:
            await self._fail_idempotency(idempotency_key, request_hash)
            raise

    def _request_hash(self, source: Path) -> str:
        canonical = json.dumps(
            {
                "source": source.resolve().as_posix(),
                "source_repo": self._source_repo,
                "source_commit": self._source_commit,
                "source_root": self._source_root.as_posix(),
                "language": self._language,
                "chunker_version": self._chunker.version,
                "embedding_model": self._embedding_client.model,
                "embedding_dimensions": self._embedding_client.dimensions,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _claim_idempotency(
        self,
        idempotency_key: str,
        request_hash: str,
    ) -> IngestResult | None:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")

        async with self._database.connection() as connection:
            async with connection.transaction():
                inserted_cursor = await connection.execute(
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
                    (idempotency_key, request_hash),
                )
                if await inserted_cursor.fetchone() is not None:
                    return None

                existing_cursor = await connection.execute(
                    """
                    SELECT request_hash, status, result
                    FROM ingestion_requests
                    WHERE idempotency_key = %s
                    FOR UPDATE
                    """,
                    (idempotency_key,),
                )
                existing = await existing_cursor.fetchone()
                if existing is None:
                    raise RuntimeError("idempotency row disappeared during claim")
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different request"
                    )
                if existing["status"] == "completed":
                    return IngestResult.from_json(existing["result"])
                if existing["status"] == "processing":
                    raise IdempotencyInProgressError(
                        "idempotent ingestion request is already processing"
                    )
                if existing["status"] != "failed":
                    raise RuntimeError(f"unknown ingestion request status: {existing['status']}")

                await connection.execute(
                    """
                    UPDATE ingestion_requests
                    SET status = 'processing',
                        result = NULL,
                        updated_at = now()
                    WHERE idempotency_key = %s
                    """,
                    (idempotency_key,),
                )
                return None

    async def _complete_idempotency(
        self,
        idempotency_key: str,
        request_hash: str,
        result: IngestResult,
    ) -> None:
        async with self._database.connection() as connection:
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
                (Jsonb(result.to_json()), idempotency_key, request_hash),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("idempotency completion did not update one row")

    async def _fail_idempotency(
        self,
        idempotency_key: str,
        request_hash: str,
    ) -> None:
        async with self._database.connection() as connection:
            await connection.execute(
                """
                UPDATE ingestion_requests
                SET status = 'failed',
                    updated_at = now()
                WHERE idempotency_key = %s
                  AND request_hash = %s
                  AND status = 'processing'
                """,
                (idempotency_key, request_hash),
            )

    async def _ingest_document(self, path: Path) -> _DocumentResult:
        parse_started = time.perf_counter()
        markdown = await asyncio.to_thread(path.read_text, encoding="utf-8")
        document = parse_markdown(markdown, source_path=path)
        chunks = self._chunker.chunk(document)
        parse_ms = (time.perf_counter() - parse_started) * 1000
        source_path = self._relative_source_path(path)

        transaction_started = time.perf_counter()
        embedding_ms = 0.0
        embedded_count = 0
        skipped_count = 0
        deleted_count = 0
        request_count = 0

        async with self._database.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (self._source_lock_key(source_path),),
                )
                document_id = await self._upsert_document(
                    connection,
                    source_path=source_path,
                    title=document.title,
                    content_hash=document.content_hash,
                )
                existing = await self._existing_chunks(connection, document_id)
                changed = [
                    chunk
                    for chunk in chunks
                    if not self._can_reuse_embedding(chunk, existing.get(chunk.chunk_index))
                ]
                skipped_count = len(chunks) - len(changed)

                vectors_by_index: dict[int, EmbeddingVector] = {}
                if changed:
                    embedding_started = time.perf_counter()
                    vectors = await self._embedding_client.embed(
                        [chunk.retrieval_text for chunk in changed]
                    )
                    embedding_ms = (time.perf_counter() - embedding_started) * 1000
                    if len(vectors) != len(changed):
                        raise RuntimeError("embedding provider returned the wrong vector count")
                    vectors_by_index = {
                        chunk.chunk_index: vector
                        for chunk, vector in zip(changed, vectors, strict=True)
                    }
                    embedded_count = len(changed)
                    request_count = math.ceil(len(changed) / self._embedding_client.batch_size)

                for chunk in changed:
                    vector = vectors_by_index[chunk.chunk_index]
                    self._validate_vector(vector)
                    await self._upsert_chunk(
                        connection,
                        document_id=document_id,
                        source_path=source_path,
                        chunk=chunk,
                        embedding=vector,
                    )

                deleted_cursor = await connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = %s
                      AND chunk_index >= %s
                    """,
                    (document_id, len(chunks)),
                )
                deleted_count = deleted_cursor.rowcount

        transaction_ms = (time.perf_counter() - transaction_started) * 1000
        return _DocumentResult(
            chunk_count=len(chunks),
            embedded_chunk_count=embedded_count,
            skipped_embedding_count=skipped_count,
            deleted_chunk_count=deleted_count,
            embedding_http_request_count=request_count,
            parse_ms=parse_ms,
            embedding_ms=embedding_ms,
            database_insert_ms=max(0.0, transaction_ms - embedding_ms),
        )

    async def _upsert_document(
        self,
        connection: AsyncConnection[Row],
        *,
        source_path: str,
        title: str | None,
        content_hash: str,
    ) -> int:
        cursor = await connection.execute(
            """
            INSERT INTO documents (
                source_repo,
                source_commit,
                source_path,
                language,
                title,
                content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_repo, source_commit, source_path)
            DO UPDATE SET
                language = EXCLUDED.language,
                title = EXCLUDED.title,
                content_hash = EXCLUDED.content_hash
            RETURNING id
            """,
            (
                self._source_repo,
                self._source_commit,
                source_path,
                self._language,
                title,
                content_hash,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("document upsert returned no id")
        return int(row["id"])

    async def _existing_chunks(
        self,
        connection: AsyncConnection[Row],
        document_id: int,
    ) -> dict[int, Row]:
        cursor = await connection.execute(
            """
            SELECT
                chunk_index,
                content_hash,
                chunker_version,
                embedding_model,
                embedding_dimensions
            FROM chunks
            WHERE document_id = %s
            """,
            (document_id,),
        )
        return {int(row["chunk_index"]): row for row in await cursor.fetchall()}

    def _can_reuse_embedding(self, chunk: Chunk, existing: Row | None) -> bool:
        return bool(
            existing is not None
            and existing["content_hash"] == chunk.content_hash
            and existing["chunker_version"] == chunk.chunker_version
            and existing["embedding_model"] == self._embedding_client.model
            and existing["embedding_dimensions"] == self._embedding_client.dimensions
        )

    async def _upsert_chunk(
        self,
        connection: AsyncConnection[Row],
        *,
        document_id: int,
        source_path: str,
        chunk: Chunk,
        embedding: EmbeddingVector,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO chunks (
                document_id,
                chunk_index,
                chunker_version,
                heading_path,
                content_raw,
                retrieval_text,
                char_count,
                content_hash,
                metadata,
                embedding_model,
                embedding_dimensions,
                embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE SET
                chunker_version = EXCLUDED.chunker_version,
                heading_path = EXCLUDED.heading_path,
                content_raw = EXCLUDED.content_raw,
                retrieval_text = EXCLUDED.retrieval_text,
                char_count = EXCLUDED.char_count,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dimensions = EXCLUDED.embedding_dimensions,
                embedding = EXCLUDED.embedding
            """,
            (
                document_id,
                chunk.chunk_index,
                chunk.chunker_version,
                list(chunk.heading_path),
                chunk.content_raw,
                chunk.retrieval_text,
                chunk.char_count,
                chunk.content_hash,
                Jsonb(
                    {
                        "source_repo": self._source_repo,
                        "source_commit": self._source_commit,
                        "source_path": source_path,
                        "language": self._language,
                    }
                ),
                self._embedding_client.model,
                self._embedding_client.dimensions,
                embedding,
            ),
        )

    def _relative_source_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._source_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"source file {path} is outside source root {self._source_root}"
            ) from exc

    def _source_lock_key(self, source_path: str) -> str:
        return json.dumps(
            [self._source_repo, self._source_commit, source_path],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _validate_vector(self, vector: EmbeddingVector) -> None:
        if vector.shape != (self._embedding_client.dimensions,):
            raise RuntimeError(
                f"embedding has shape {vector.shape}, expected "
                f"({self._embedding_client.dimensions},)"
            )
        if not np.isfinite(vector).all():
            raise RuntimeError("embedding contains non-finite values")


def _markdown_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        if source.suffix.lower() != ".md":
            raise ValueError(f"unsupported source file: {source}")
        return (source,)
    if source.is_dir():
        return tuple(
            sorted(
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() == ".md"
            )
        )
    raise FileNotFoundError(source)
