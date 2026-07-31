from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rag_demo.chunker import MarkdownChunker
from rag_demo.db import Database
from rag_demo.embedding_client import EmbeddingVector
from rag_demo.ingest_service import DocumentIngestor


class _EmbeddingProvider:
    model = "test-embedding"
    dimensions = 3
    batch_size = 8

    async def embed(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        raise AssertionError("embedding is not used while generating a key")


def _ingestor(
    source_root: Path,
    *,
    chunker_version: str,
    target_chars: int = 100,
) -> DocumentIngestor:
    return DocumentIngestor(
        database=cast(Database, object()),
        embedding_client=_EmbeddingProvider(),
        chunker=MarkdownChunker(
            target_chars=target_chars,
            max_chars=150,
            overlap_chars=20,
            version=chunker_version,
        ),
        source_repo="https://example.test/docs.git",
        source_commit="abc123",
        source_root=source_root,
        language="zh",
    )


def test_generated_idempotency_key_is_stable_for_the_same_logical_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    ingestor = _ingestor(tmp_path, chunker_version="markdown-structure-v2")

    first = ingestor.idempotency_key_for(source)
    repeated = ingestor.idempotency_key_for(source)

    assert first == repeated
    assert first.startswith("ingest-")


def test_generated_idempotency_key_changes_with_pipeline_version(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()

    version_one = _ingestor(
        tmp_path,
        chunker_version="markdown-structure-v1",
    ).idempotency_key_for(source)
    version_two = _ingestor(
        tmp_path,
        chunker_version="markdown-structure-v2",
    ).idempotency_key_for(source)

    assert version_one != version_two


def test_generated_key_is_independent_of_checkout_location(tmp_path: Path) -> None:
    first_root = tmp_path / "first-checkout"
    second_root = tmp_path / "second-checkout"
    first_source = first_root / "docs"
    second_source = second_root / "docs"
    first_source.mkdir(parents=True)
    second_source.mkdir(parents=True)

    first = _ingestor(
        first_root,
        chunker_version="markdown-structure-v2",
    ).idempotency_key_for(first_source)
    second = _ingestor(
        second_root,
        chunker_version="markdown-structure-v2",
    ).idempotency_key_for(second_source)

    assert first == second


def test_generated_key_changes_with_chunk_configuration(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()

    target_100 = _ingestor(
        tmp_path,
        chunker_version="markdown-structure-v2",
        target_chars=100,
    ).idempotency_key_for(source)
    target_120 = _ingestor(
        tmp_path,
        chunker_version="markdown-structure-v2",
        target_chars=120,
    ).idempotency_key_for(source)

    assert target_100 != target_120
