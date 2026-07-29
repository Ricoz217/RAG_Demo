import json
from pathlib import Path

import pytest

from rag_demo.evaluation import (
    EvaluationHit,
    EvaluationQuery,
    QueryMeasurement,
    calculate_metrics,
    load_evaluation_queries,
)


def test_load_evaluation_queries_validates_and_normalizes_json(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "怎样接收 JSON？",
                    "relevant_path_contains": ["tutorial/body"],
                    "relevant_heading_contains": ["请求体"],
                    "category": "semantic-query",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    queries = load_evaluation_queries(path)

    assert queries == (
        EvaluationQuery(
            query="怎样接收 JSON？",
            relevant_path_contains=("tutorial/body",),
            relevant_heading_contains=("请求体",),
            category="semantic-query",
        ),
    )


def test_metrics_compute_recall_mrr_average_and_p95() -> None:
    first = EvaluationQuery(
        query="first",
        relevant_path_contains=("expected/path",),
        relevant_heading_contains=(),
        category="exact-term",
    )
    second = EvaluationQuery(
        query="second",
        relevant_path_contains=(),
        relevant_heading_contains=("目标章节",),
        category="semantic-query",
    )
    measurements = (
        QueryMeasurement(
            query=first,
            hits=(
                EvaluationHit("expected/path.md", ("其他",)),
                *(EvaluationHit(f"irrelevant/{index}.md", ("其他",)) for index in range(9)),
            ),
            latency_ms=10,
        ),
        QueryMeasurement(
            query=second,
            hits=(
                *(EvaluationHit(f"irrelevant/{index}.md", ("其他",)) for index in range(5)),
                EvaluationHit("another.md", ("目标章节",)),
            ),
            latency_ms=30,
        ),
    )

    metrics = calculate_metrics(measurements)

    assert metrics.recall_at_5 == pytest.approx(0.5)
    assert metrics.recall_at_10 == pytest.approx(1.0)
    assert metrics.mrr_at_10 == pytest.approx((1 + 1 / 6) / 2)
    assert metrics.average_latency_ms == pytest.approx(20)
    assert metrics.p95_latency_ms == pytest.approx(29)
    assert metrics.query_count == 2
