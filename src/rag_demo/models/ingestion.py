"""Ingestion operation data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
