"""Retrieval benchmark data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvaluationMethod(StrEnum):
    """Retrieval variants required by the project acceptance criteria."""

    BM25 = "bm25"
    DENSE_EXACT = "dense_exact"
    DENSE_HNSW = "dense_hnsw"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANKER = "hybrid_rrf_reranker"


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One hand-labelled query and its relevance hints."""

    query: str
    relevant_path_contains: tuple[str, ...]
    relevant_heading_contains: tuple[str, ...]
    category: str


@dataclass(frozen=True, slots=True)
class EvaluationHit:
    """Minimal source evidence used to judge relevance."""

    source_path: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryMeasurement:
    """One method's ordered hits and latency for one query."""

    query: EvaluationQuery
    hits: tuple[EvaluationHit, ...]
    latency_ms: float

    @property
    def first_relevant_rank(self) -> int | None:
        for rank, hit in enumerate(self.hits[:10], start=1):
            if _is_relevant(self.query, hit):
                return rank
        return None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate retrieval quality and latency metrics."""

    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    average_latency_ms: float
    p95_latency_ms: float
    query_count: int


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    """Measurements and aggregate metrics for one retrieval method."""

    method: EvaluationMethod
    measurements: tuple[QueryMeasurement, ...]
    metrics: EvaluationMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """All required method comparisons over the same labelled queries."""

    evaluations: tuple[MethodEvaluation, ...]


def _is_relevant(query: EvaluationQuery, hit: EvaluationHit) -> bool:
    normalized_path = hit.source_path.lower()
    heading = " / ".join(hit.heading_path).lower()
    return any(
        expected.lower() in normalized_path for expected in query.relevant_path_contains
    ) or any(expected.lower() in heading for expected in query.relevant_heading_contains)
