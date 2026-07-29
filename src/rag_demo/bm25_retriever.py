"""Chinese-aware BM25 index derived from PostgreSQL Chunks."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bm25s  # type: ignore[import-untyped]
import jieba  # type: ignore[import-untyped]

from rag_demo.db import Database, Row

_INDEX_FORMAT_VERSION = 1
_TOKENIZER_VERSION = "jieba-search-code-v1"
_SEGMENT_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
    r"|\d+(?:\.\d+)?"
)


class BM25IndexError(RuntimeError):
    """Base class for local BM25 index failures."""


class BM25IndexNotFoundError(BM25IndexError):
    """Raised when no published BM25 generation exists."""


class BM25IndexCompatibilityError(BM25IndexError):
    """Raised when a saved index does not match the active vector corpus."""


class EmptyBM25CorpusError(BM25IndexError):
    """Raised when PostgreSQL has no matching Chunks to index."""


@dataclass(frozen=True, slots=True)
class BM25BuildResult:
    """Observable result from a full index rebuild."""

    generation: str
    chunk_count: int
    build_ms: float

    def to_json(self) -> dict[str, str | int | float]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Any) -> BM25BuildResult:
        if not isinstance(value, dict):
            raise BM25IndexError("stored BM25 rebuild result is not an object")
        try:
            return cls(
                generation=str(value["generation"]),
                chunk_count=int(value["chunk_count"]),
                build_ms=float(value["build_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BM25IndexError("stored BM25 rebuild result is invalid") from exc


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    """One BM25-ranked Chunk with PostgreSQL source metadata."""

    rank: int
    chunk_id: int
    document_id: int
    title: str | None
    source_repo: str
    source_commit: str
    source_path: str
    language: str
    heading_path: tuple[str, ...]
    content_raw: str
    retrieval_text: str
    metadata: Mapping[str, Any]
    score: float


@dataclass(frozen=True, slots=True)
class BM25SearchResponse:
    """Observable output from one BM25 query."""

    results: tuple[BM25SearchResult, ...]
    search_ms: float

    @property
    def candidate_count(self) -> int:
        return len(self.results)


@dataclass(frozen=True, slots=True)
class _IndexManifest:
    generation: str
    chunk_ids: tuple[int, ...]
    embedding_model: str
    embedding_dimensions: int


@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    chunk_id: int
    score: float


def tokenize_bm25(text: str) -> tuple[str, ...]:
    """Tokenize Chinese for search while preserving technical identifiers."""
    tokens: list[str] = []
    for match in _SEGMENT_PATTERN.finditer(text):
        segment = match.group(0)
        if _is_chinese_segment(segment):
            tokens.extend(
                token.strip() for token in jieba.lcut_for_search(segment) if token.strip()
            )
        else:
            tokens.append(segment.lower())
    return tuple(tokens)


def _is_chinese_segment(segment: str) -> bool:
    first = ord(segment[0])
    return 0x3400 <= first <= 0x4DBF or 0x4E00 <= first <= 0x9FFF


class BM25IndexManager:
    """Build, publish, and load immutable BM25 generations."""

    def __init__(
        self,
        *,
        database: Database,
        index_root: Path,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        if not embedding_model:
            raise ValueError("embedding_model must not be empty")
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        self._database = database
        self._index_root = index_root
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    async def rebuild(self) -> BM25BuildResult:
        """Rebuild from PostgreSQL and atomically publish a generation pointer."""
        rows = await self._read_corpus()
        if not rows:
            raise EmptyBM25CorpusError(
                "PostgreSQL has no Chunks for the configured embedding model"
            )

        generation = f"generation-{uuid.uuid4().hex}"
        started = time.perf_counter()
        await asyncio.to_thread(
            _publish_generation,
            self._index_root,
            generation,
            rows,
            self._embedding_model,
            self._embedding_dimensions,
        )
        return BM25BuildResult(
            generation=generation,
            chunk_count=len(rows),
            build_ms=(time.perf_counter() - started) * 1000,
        )

    async def load(self) -> BM25Retriever:
        """Load the currently published generation without blocking the event loop."""
        index, manifest = await asyncio.to_thread(_load_generation, self._index_root)
        if manifest.embedding_model != self._embedding_model:
            raise BM25IndexCompatibilityError(
                "BM25 index embedding model does not match current configuration"
            )
        if manifest.embedding_dimensions != self._embedding_dimensions:
            raise BM25IndexCompatibilityError(
                "BM25 index embedding dimensions do not match current configuration"
            )
        return BM25Retriever(
            database=self._database,
            index=index,
            manifest=manifest,
        )

    async def _read_corpus(self) -> tuple[Row, ...]:
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT chunks.id AS chunk_id, chunks.retrieval_text
                FROM chunks
                WHERE chunks.embedding_model = %s
                  AND chunks.embedding_dimensions = %s
                ORDER BY chunks.id
                """,
                (self._embedding_model, self._embedding_dimensions),
            )
            return tuple(await cursor.fetchall())


class BM25Retriever:
    """Search one immutable BM25 generation and hydrate Chunks from PostgreSQL."""

    def __init__(
        self,
        *,
        database: Database,
        index: bm25s.BM25,
        manifest: _IndexManifest,
    ) -> None:
        self._database = database
        self._index = index
        self._manifest = manifest

    @property
    def generation(self) -> str:
        return self._manifest.generation

    @property
    def chunk_count(self) -> int:
        return len(self._manifest.chunk_ids)

    async def search(self, query: str, *, top_k: int) -> BM25SearchResponse:
        """Return positive-score BM25 candidates with current DB metadata."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        started = time.perf_counter()
        scored = await asyncio.to_thread(
            _retrieve_scores,
            self._index,
            self._manifest.chunk_ids,
            query,
            top_k,
        )
        rows_by_id = await self._fetch_chunks(tuple(item.chunk_id for item in scored))
        results = tuple(
            _result_from_row(rows_by_id[item.chunk_id], score=item.score, rank=rank)
            for rank, item in enumerate(
                (item for item in scored if item.chunk_id in rows_by_id),
                start=1,
            )
        )
        return BM25SearchResponse(
            results=results,
            search_ms=(time.perf_counter() - started) * 1000,
        )

    async def _fetch_chunks(self, chunk_ids: tuple[int, ...]) -> dict[int, Row]:
        if not chunk_ids:
            return {}
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    chunks.id AS chunk_id,
                    chunks.document_id,
                    documents.title,
                    documents.source_repo,
                    documents.source_commit,
                    documents.source_path,
                    documents.language,
                    chunks.heading_path,
                    chunks.content_raw,
                    chunks.retrieval_text,
                    chunks.metadata
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id = ANY(%s)
                """,
                (list(chunk_ids),),
            )
            rows = await cursor.fetchall()
        return {row["chunk_id"]: row for row in rows}


def _publish_generation(
    index_root: Path,
    generation: str,
    rows: Sequence[Row],
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    generations_root = index_root / "generations"
    generations_root.mkdir(parents=True, exist_ok=True)
    staging = generations_root / f".build-{uuid.uuid4().hex}"
    final = generations_root / generation
    staging.mkdir()

    try:
        _write_generation(
            staging,
            generation,
            rows,
            embedding_model,
            embedding_dimensions,
        )
        os.replace(staging, final)
        _replace_current_pointer(index_root, generation)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _write_generation(
    directory: Path,
    generation: str,
    rows: Sequence[Row],
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    chunk_ids = tuple(int(row["chunk_id"]) for row in rows)
    tokenized_corpus = [list(tokenize_bm25(row["retrieval_text"])) for row in rows]
    index = bm25s.BM25(method="lucene")
    index.index(tokenized_corpus, create_empty_token=True, show_progress=False)
    index.save(str(directory), show_progress=False)

    manifest = {
        "format_version": _INDEX_FORMAT_VERSION,
        "tokenizer_version": _TOKENIZER_VERSION,
        "generation": generation,
        "chunk_ids": chunk_ids,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _replace_current_pointer(index_root: Path, generation: str) -> None:
    pointer = index_root / "CURRENT"
    temporary = index_root / f".CURRENT-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{generation}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_generation(index_root: Path) -> tuple[bm25s.BM25, _IndexManifest]:
    pointer = index_root / "CURRENT"
    if not pointer.is_file():
        raise BM25IndexNotFoundError("BM25 index has not been built")

    generation = pointer.read_text(encoding="utf-8").strip()
    if not generation.startswith("generation-") or Path(generation).name != generation:
        raise BM25IndexError("BM25 CURRENT pointer is invalid")
    directory = index_root / "generations" / generation
    if not directory.is_dir():
        raise BM25IndexNotFoundError("published BM25 generation directory is missing")

    manifest_path = directory / "manifest.json"
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("manifest root must be an object")
        if raw["format_version"] != _INDEX_FORMAT_VERSION:
            raise BM25IndexCompatibilityError("unsupported BM25 index format")
        if raw["tokenizer_version"] != _TOKENIZER_VERSION:
            raise BM25IndexCompatibilityError("unsupported BM25 tokenizer version")
        manifest = _IndexManifest(
            generation=str(raw["generation"]),
            chunk_ids=tuple(int(chunk_id) for chunk_id in raw["chunk_ids"]),
            embedding_model=str(raw["embedding_model"]),
            embedding_dimensions=int(raw["embedding_dimensions"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BM25IndexError("BM25 manifest is invalid") from exc
    if manifest.generation != generation:
        raise BM25IndexError("BM25 manifest generation does not match CURRENT")

    index = bm25s.BM25.load(
        str(directory),
        load_corpus=False,
        show_progress=False,
    )
    return index, manifest


def _retrieve_scores(
    index: bm25s.BM25,
    chunk_ids: tuple[int, ...],
    query: str,
    top_k: int,
) -> tuple[_ScoredChunk, ...]:
    tokens = tokenize_bm25(query)
    if not tokens or not chunk_ids:
        return ()
    result = index.retrieve(
        [list(tokens)],
        corpus=list(chunk_ids),
        k=min(top_k, len(chunk_ids)),
        show_progress=False,
    )
    documents = result.documents[0].tolist()
    scores = result.scores[0].tolist()
    return tuple(
        _ScoredChunk(chunk_id=int(chunk_id), score=float(score))
        for chunk_id, score in zip(documents, scores, strict=True)
        if math.isfinite(float(score)) and float(score) > 0
    )


def _result_from_row(row: Row, *, score: float, rank: int) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        title=row["title"],
        source_repo=row["source_repo"],
        source_commit=row["source_commit"],
        source_path=row["source_path"],
        language=row["language"],
        heading_path=tuple(row["heading_path"]),
        content_raw=row["content_raw"],
        retrieval_text=row["retrieval_text"],
        metadata=row["metadata"],
        score=score,
    )
