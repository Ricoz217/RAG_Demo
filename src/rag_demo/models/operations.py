"""Infrastructure and operational result data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    """Observable database facts used by CLI and health checks."""

    current_user: str
    current_database: str
    server_version: str
    vector_version: str
    embedding_dimensions: int
    hnsw_index_present: bool
    document_count: int
    chunk_count: int
    applied_migrations: tuple[str, ...]

    @property
    def embedding_column_type(self) -> str:
        return f"vector({self.embedding_dimensions})"


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


@dataclass(frozen=True, slots=True)
class CorpusInfo:
    """Observable Git and Markdown facts for one corpus checkout."""

    path: Path
    repository_url: str
    commit: str
    zh_markdown_count: int
    en_markdown_count: int

    def source_directory(self, language: str) -> Path:
        if language not in {"zh", "en"}:
            raise ValueError("FastAPI corpus language must be 'zh' or 'en'")
        return self.path / "docs" / language / "docs"
