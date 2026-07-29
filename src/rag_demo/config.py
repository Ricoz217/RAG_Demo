"""Application configuration loaded from environment variables."""

from typing import Self

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration shared by CLI, REST, and core services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: SecretStr = Field(validation_alias="DATABASE_URL")

    embedding_base_url: AnyHttpUrl
    embedding_api_key: SecretStr
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    reranker_base_url: AnyHttpUrl
    reranker_api_key: SecretStr
    reranker_model: str = Field(min_length=1)
    reranker_timeout_seconds: float = Field(default=60.0, gt=0)

    chunk_target_chars: int = Field(default=1400, gt=0)
    chunk_max_chars: int = Field(default=2200, gt=0)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    dense_top_k: int = Field(default=30, gt=0)
    bm25_top_k: int = Field(default=30, gt=0)
    rrf_rank_constant: int = Field(default=60, gt=0)
    rerank_top_k: int = Field(default=20, gt=0)
    final_top_k: int = Field(default=5, gt=0)
    hnsw_ef_search: int = Field(default=100, gt=0)

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        """Require a PostgreSQL DSN while retaining secret-safe representation."""
        try:
            TypeAdapter(PostgresDsn).validate_python(value.get_secret_value())
        except ValidationError as exc:
            raise ValueError("DATABASE_URL must be a valid PostgreSQL DSN") from exc
        return value

    @property
    def embedding_endpoint(self) -> str:
        """Return the OpenAI-compatible embeddings endpoint."""
        return f"{str(self.embedding_base_url).rstrip('/')}/v1/embeddings"

    @property
    def reranker_endpoint(self) -> str:
        """Return the llama.cpp reranking endpoint."""
        return f"{str(self.reranker_base_url).rstrip('/')}/reranking"

    @model_validator(mode="after")
    def validate_related_limits(self) -> Self:
        """Validate settings whose constraints depend on another field."""
        if self.chunk_target_chars > self.chunk_max_chars:
            raise ValueError("chunk_target_chars must not exceed chunk_max_chars")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_target_chars")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k must not exceed rerank_top_k")
        return self
