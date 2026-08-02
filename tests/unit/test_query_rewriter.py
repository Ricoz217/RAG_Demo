import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from rag_demo.query_rewriter import (
    QueryRewriteConfigurationError,
    QueryRewriter,
    QueryRewriteSearchService,
    compare_result_rankings,
)
from rag_demo.search_service import SearchRequest, SearchResponse


def _write_dictionary(path: Path, aliases: dict[str, str]) -> Path:
    path.write_text(
        json.dumps({"version": 1, "aliases": aliases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_rewriter_normalizes_and_expands_aliases_without_discarding_original_terms(
    tmp_path: Path,
) -> None:
    dictionary = _write_dictionary(
        tmp_path / "aliases.json",
        {
            "PG": "PostgreSQL",
            "WAL": "Write-Ahead Log",
            "依赖注入": "Dependency Injection",
        },
    )
    rewriter = QueryRewriter.from_json(dictionary)

    result = rewriter.rewrite("  ＰＧ  的 WAL 和依赖注入  ")

    assert result.original_query == "  ＰＧ  的 WAL 和依赖注入  "
    assert result.normalized_query == "PG 的 WAL 和依赖注入"
    assert result.effective_query == (
        "PostgreSQL (PG) 的 Write-Ahead Log (WAL) 和Dependency Injection (依赖注入)"
    )
    assert result.applied_rules == (
        "unicode_nfkc",
        "whitespace",
        "alias:PG",
        "alias:WAL",
        "alias:依赖注入",
    )
    assert result.changed is True
    assert result.rewrite_ms >= 0


def test_alias_matching_is_case_insensitive_non_cascading_and_token_aware(
    tmp_path: Path,
) -> None:
    dictionary = _write_dictionary(
        tmp_path / "aliases.json",
        {
            "PG": "PostgreSQL",
            "SQL": "Structured Query Language",
            "API": "Application Programming Interface",
        },
    )
    rewriter = QueryRewriter.from_json(dictionary)

    result = rewriter.rewrite("pg 与 FastAPI、API接口")

    assert result.effective_query == (
        "PostgreSQL (pg) 与 FastAPI、Application Programming Interface (API)接口"
    )
    assert result.applied_rules == ("alias:PG", "alias:API")


def test_rewriter_reports_an_unchanged_query_without_claiming_improvement(
    tmp_path: Path,
) -> None:
    rewriter = QueryRewriter.from_json(
        _write_dictionary(tmp_path / "aliases.json", {"PG": "PostgreSQL"})
    )

    result = rewriter.rewrite("PostgreSQL 事务")

    assert result.effective_query == "PostgreSQL 事务"
    assert result.applied_rules == ()
    assert result.changed is False


def test_rewriter_is_idempotent_for_its_own_alias_expansion(tmp_path: Path) -> None:
    rewriter = QueryRewriter.from_json(
        _write_dictionary(tmp_path / "aliases.json", {"PG": "PostgreSQL"})
    )

    first = rewriter.rewrite("PG 事务")
    second = rewriter.rewrite(first.effective_query)

    assert second.effective_query == first.effective_query
    assert second.applied_rules == ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"version": 2, "aliases": {"PG": "PostgreSQL"}}, "version"),
        ({"version": 1, "aliases": {}}, "aliases"),
        ({"version": 1, "aliases": {"PG": ""}}, "canonical"),
        ({"version": 1, "aliases": {"PG": "PostgreSQL", "pg": "postgres"}}, "duplicate"),
    ],
)
def test_dictionary_rejects_invalid_or_ambiguous_configuration(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QueryRewriteConfigurationError, match=message):
        QueryRewriter.from_json(path)


def test_dictionary_wraps_missing_and_malformed_json_errors(tmp_path: Path) -> None:
    with pytest.raises(QueryRewriteConfigurationError, match="read"):
        QueryRewriter.from_json(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(QueryRewriteConfigurationError, match="JSON"):
        QueryRewriter.from_json(malformed)


def test_result_comparison_reports_membership_and_rank_changes_neutrally() -> None:
    comparison = compare_result_rankings(
        without_rewrite=(10, 20, 30),
        with_rewrite=(20, 40, 10),
    )

    assert comparison.same_order is False
    assert comparison.overlap_chunk_ids == (10, 20)
    assert comparison.only_without_chunk_ids == (30,)
    assert comparison.only_with_chunk_ids == (40,)
    assert [
        (change.chunk_id, change.without_rank, change.with_rank, change.rank_delta)
        for change in comparison.rank_changes
    ] == [
        (10, 1, 3, -2),
        (20, 2, 1, 1),
        (30, 3, None, None),
        (40, None, 2, None),
    ]


def test_result_comparison_can_report_no_effect() -> None:
    comparison = compare_result_rankings(
        without_rewrite=(10, 20),
        with_rewrite=(10, 20),
    )

    assert comparison.same_order is True
    assert comparison.only_without_chunk_ids == ()
    assert comparison.only_with_chunk_ids == ()
    assert all(change.rank_delta == 0 for change in comparison.rank_changes)


@pytest.mark.asyncio
async def test_comparison_executes_both_runs_even_when_effective_query_is_unchanged(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakeSearchProvider:
        async def search(self, request: SearchRequest) -> SearchResponse:
            calls.append(request.query)
            return cast(
                SearchResponse,
                SimpleNamespace(results=(SimpleNamespace(chunk_id=10),)),
            )

    service = QueryRewriteSearchService(
        search_provider=FakeSearchProvider(),
        rewriter=QueryRewriter.from_json(
            _write_dictionary(tmp_path / "aliases.json", {"PG": "PostgreSQL"})
        ),
    )

    experiment = await service.compare(SearchRequest(query="没有命中的查询"))

    assert calls == ["没有命中的查询", "没有命中的查询"]
    assert experiment.with_rewrite.rewrite.changed is False
    assert experiment.comparison.same_order is True
