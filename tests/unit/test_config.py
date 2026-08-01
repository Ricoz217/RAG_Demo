from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_demo.config import Settings


def _set_required_environment(
    monkeypatch: pytest.MonkeyPatch,
    overrides: Mapping[str, str] | None = None,
) -> None:
    values = {
        "DATABASE_URL": "postgresql://rag_app:database-secret@127.0.0.1:5432/rag_demo",
        "EMBEDDING_BASE_URL": "http://127.0.0.1:8081/",
        "EMBEDDING_API_KEY": "embedding-secret",
        "EMBEDDING_MODEL": "bge-m3",
        "EMBEDDING_DIMENSIONS": "1024",
        "RERANKER_BASE_URL": "http://127.0.0.1:8082/",
        "RERANKER_API_KEY": "reranker-secret",
        "RERANKER_MODEL": "bge-reranker-v2-m3",
    }
    values.update(overrides or {})
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_settings_load_required_services_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_environment(monkeypatch)

    settings = Settings()

    assert settings.embedding_endpoint == "http://127.0.0.1:8081/v1/embeddings"
    assert settings.reranker_endpoint == "http://127.0.0.1:8082/reranking"
    assert settings.embedding_dimensions == 1024
    assert settings.embedding_timeout_seconds == 60.0
    assert settings.embedding_batch_size == 32
    assert settings.reranker_timeout_seconds == 60.0
    assert settings.chunk_target_chars == 1400
    assert settings.chunk_max_chars == 2200
    assert settings.chunk_overlap_chars == 200
    assert settings.dense_top_k == 30
    assert settings.bm25_top_k == 30
    assert settings.rrf_rank_constant == 60
    assert settings.rerank_top_k == 20
    assert settings.final_top_k == 5
    assert settings.hnsw_ef_search == 100
    assert settings.query_aliases_path == Path("data/query_aliases.json")


def test_settings_repr_does_not_expose_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_environment(monkeypatch)

    representation = repr(Settings())

    assert "database-secret" not in representation
    assert "embedding-secret" not in representation
    assert "reranker-secret" not in representation


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"EMBEDDING_DIMENSIONS": "0"}, "embedding_dimensions"),
        ({"EMBEDDING_BATCH_SIZE": "0"}, "embedding_batch_size"),
        ({"CHUNK_TARGET_CHARS": "2201"}, "chunk_target_chars"),
        ({"CHUNK_OVERLAP_CHARS": "1400"}, "chunk_overlap_chars"),
        ({"FINAL_TOP_K": "21"}, "final_top_k"),
    ],
)
def test_settings_reject_invalid_numeric_relationships(
    monkeypatch: pytest.MonkeyPatch,
    overrides: Mapping[str, str],
    message: str,
) -> None:
    _set_required_environment(monkeypatch, overrides)

    with pytest.raises(ValidationError, match=message):
        Settings()


def test_settings_reject_non_postgresql_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_environment(monkeypatch, {"DATABASE_URL": "sqlite:///rag.db"})

    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings()
