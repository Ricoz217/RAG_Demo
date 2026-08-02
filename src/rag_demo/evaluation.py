"""Small observable retrieval benchmark for the demo corpus."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from rag_demo.application import RAGApplication
from rag_demo.models.evaluation import (
    BenchmarkReport,
    EvaluationHit,
    EvaluationMethod,
    EvaluationMetrics,
    EvaluationQuery,
    MethodEvaluation,
    QueryMeasurement,
)
from rag_demo.models.retrieval import DenseSearchMode, SearchRequest

__all__ = [
    "BenchmarkReport",
    "BenchmarkService",
    "EvaluationHit",
    "EvaluationMethod",
    "EvaluationMetrics",
    "EvaluationQuery",
    "MethodEvaluation",
    "QueryMeasurement",
    "calculate_metrics",
    "load_evaluation_queries",
]


def load_evaluation_queries(path: Path) -> tuple[EvaluationQuery, ...]:
    """Load and validate the committed evaluation JSON."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evaluation queries: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluation query file must contain a non-empty list")

    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each evaluation query must be an object")
        try:
            query = _required_string(item, "query")
            category = _required_string(item, "category")
            paths = _string_tuple(item, "relevant_path_contains")
            headings = _string_tuple(item, "relevant_heading_contains")
        except (KeyError, TypeError) as exc:
            raise ValueError("evaluation query has invalid fields") from exc
        if not paths and not headings:
            raise ValueError("evaluation query must contain a relevance hint")
        if query in seen:
            raise ValueError(f"duplicate evaluation query: {query}")
        seen.add(query)
        queries.append(
            EvaluationQuery(
                query=query,
                relevant_path_contains=paths,
                relevant_heading_contains=headings,
                category=category,
            )
        )
    return tuple(queries)


def calculate_metrics(
    measurements: tuple[QueryMeasurement, ...],
) -> EvaluationMetrics:
    """Calculate binary query Recall, reciprocal rank, and latency."""
    if not measurements:
        raise ValueError("measurements must not be empty")
    ranks = tuple(measurement.first_relevant_rank for measurement in measurements)
    latencies = tuple(measurement.latency_ms for measurement in measurements)
    query_count = len(measurements)
    return EvaluationMetrics(
        recall_at_5=sum(rank is not None and rank <= 5 for rank in ranks) / query_count,
        recall_at_10=sum(rank is not None and rank <= 10 for rank in ranks) / query_count,
        mrr_at_10=sum(0.0 if rank is None else 1 / rank for rank in ranks) / query_count,
        average_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        query_count=query_count,
    )


class BenchmarkService:
    """Run the five required retrieval variants over fixed queries."""

    def __init__(self, application: RAGApplication) -> None:
        self._application = application

    async def benchmark(
        self,
        queries: tuple[EvaluationQuery, ...],
    ) -> BenchmarkReport:
        if not queries:
            raise ValueError("queries must not be empty")
        bm25 = await self._application.load_bm25()
        dense = self._application.dense_service
        settings = self._application.settings
        branch_top_k = max(10, settings.dense_top_k, settings.bm25_top_k)
        rerank_top_k = max(10, settings.rerank_top_k)

        by_method: dict[EvaluationMethod, list[QueryMeasurement]] = {
            method: [] for method in EvaluationMethod
        }
        for evaluation_query in queries:
            exact = await dense.search(
                evaluation_query.query,
                top_k=branch_top_k,
                mode=DenseSearchMode.EXACT,
            )
            by_method[EvaluationMethod.DENSE_EXACT].append(
                QueryMeasurement(
                    query=evaluation_query,
                    hits=tuple(
                        EvaluationHit(result.source_path, result.heading_path)
                        for result in exact.results[:10]
                    ),
                    latency_ms=exact.total_ms,
                )
            )

            hnsw = await dense.search(
                evaluation_query.query,
                top_k=branch_top_k,
                mode=DenseSearchMode.HNSW,
            )
            by_method[EvaluationMethod.DENSE_HNSW].append(
                QueryMeasurement(
                    query=evaluation_query,
                    hits=tuple(
                        EvaluationHit(result.source_path, result.heading_path)
                        for result in hnsw.results[:10]
                    ),
                    latency_ms=hnsw.total_ms,
                )
            )

            sparse = await bm25.search(
                evaluation_query.query,
                top_k=branch_top_k,
            )
            by_method[EvaluationMethod.BM25].append(
                QueryMeasurement(
                    query=evaluation_query,
                    hits=tuple(
                        EvaluationHit(result.source_path, result.heading_path)
                        for result in sparse.results[:10]
                    ),
                    latency_ms=sparse.search_ms,
                )
            )

            hybrid = await self._application.search(
                SearchRequest(
                    query=evaluation_query.query,
                    dense_top_k=branch_top_k,
                    bm25_top_k=branch_top_k,
                    rrf_rank_constant=settings.rrf_rank_constant,
                    rerank_top_k=rerank_top_k,
                    final_top_k=10,
                    dense_mode=DenseSearchMode.HNSW,
                    use_reranker=False,
                )
            )
            by_method[EvaluationMethod.HYBRID_RRF].append(
                QueryMeasurement(
                    query=evaluation_query,
                    hits=tuple(
                        EvaluationHit(result.source_path, result.heading_path)
                        for result in hybrid.results
                    ),
                    latency_ms=hybrid.timings.total_ms,
                )
            )

            reranked = await self._application.search(
                SearchRequest(
                    query=evaluation_query.query,
                    dense_top_k=branch_top_k,
                    bm25_top_k=branch_top_k,
                    rrf_rank_constant=settings.rrf_rank_constant,
                    rerank_top_k=rerank_top_k,
                    final_top_k=10,
                    dense_mode=DenseSearchMode.HNSW,
                    use_reranker=True,
                )
            )
            by_method[EvaluationMethod.HYBRID_RERANKER].append(
                QueryMeasurement(
                    query=evaluation_query,
                    hits=tuple(
                        EvaluationHit(result.source_path, result.heading_path)
                        for result in reranked.results
                    ),
                    latency_ms=reranked.timings.total_ms,
                )
            )

        return BenchmarkReport(
            evaluations=tuple(
                MethodEvaluation(
                    method=method,
                    measurements=tuple(by_method[method]),
                    metrics=calculate_metrics(tuple(by_method[method])),
                )
                for method in EvaluationMethod
            )
        )


def _required_string(item: dict[str, Any], key: str) -> str:
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _string_tuple(item: dict[str, Any], key: str) -> tuple[str, ...]:
    value = item[key]
    if not isinstance(value, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in value
    ):
        raise TypeError(f"{key} must be a list of non-empty strings")
    return tuple(value)
