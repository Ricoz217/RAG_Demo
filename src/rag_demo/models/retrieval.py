"""Dense, sparse, fusion, and final-search data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class BM25IndexError(RuntimeError):
    """Base class for local BM25 index failures."""


class DenseSearchMode(StrEnum):
    """Supported pgvector execution modes."""

    EXACT = "exact"
    HNSW = "hnsw"


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    """One ranked chunk returned by pgvector."""

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
    cosine_distance: float
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class DenseRetrievalResponse:
    """Observable output from the database retrieval stage."""

    mode: DenseSearchMode
    results: tuple[DenseSearchResult, ...]
    search_ms: float

    @property
    def candidate_count(self) -> int:
        return len(self.results)


@dataclass(frozen=True, slots=True)
class DenseQueryPlan:
    """PostgreSQL execution plan evidence for one dense mode."""

    mode: DenseSearchMode
    lines: tuple[str, ...]
    ef_search: int | None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def uses_hnsw(self) -> bool:
        return "chunks_embedding_hnsw" in self.text


@dataclass(frozen=True, slots=True)
class DenseQueryResponse:
    """Dense query output including embedding and retrieval timings."""

    query: str
    mode: DenseSearchMode
    results: tuple[DenseSearchResult, ...]
    query_embedding_ms: float
    dense_search_ms: float
    total_ms: float

    @property
    def candidate_count(self) -> int:
        return len(self.results)


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
    """One BM25-ranked chunk with PostgreSQL source metadata."""

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
class ScoredCandidate:
    """One branch-specific ranked candidate."""

    chunk_id: int
    score: float


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """One union candidate with branch evidence and an RRF score."""

    chunk_id: int
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_score: float


@dataclass(frozen=True, slots=True)
class HybridRecallResponse:
    """Dense, BM25, and RRF stages retained for debug output."""

    query: str
    dense: DenseQueryResponse
    bm25: BM25SearchResponse
    fused: tuple[FusedCandidate, ...]
    rrf_ms: float
    total_ms: float

    @property
    def union_candidate_count(self) -> int:
        return len(self.fused)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Validated core search parameters shared by CLI and REST."""

    query: str
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_rank_constant: int = 60
    rerank_top_k: int = 20
    final_top_k: int = 5
    dense_mode: DenseSearchMode = DenseSearchMode.HNSW
    use_reranker: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        for name, value in (
            ("dense_top_k", self.dense_top_k),
            ("bm25_top_k", self.bm25_top_k),
            ("rrf_rank_constant", self.rrf_rank_constant),
            ("rerank_top_k", self.rerank_top_k),
            ("final_top_k", self.final_top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k must not exceed rerank_top_k")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Hydrated candidate retaining every ranking stage."""

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
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_rank: int
    rrf_score: float
    rerank_score: float | None
    final_rank: int | None


@dataclass(frozen=True, slots=True)
class SearchTimings:
    """Per-stage search latency in milliseconds."""

    query_embedding_ms: float
    dense_search_ms: float
    bm25_search_ms: float
    rrf_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class SearchCandidateCounts:
    """Candidate counts at branch and union boundaries."""

    dense_candidate_count: int
    bm25_candidate_count: int
    union_candidate_count: int


class SearchConfidenceStatus(StrEnum):
    """Whether final retrieval confidence was assessed and flagged."""

    NOT_FLAGGED = "not_flagged"
    LOW = "low"
    UNASSESSED = "unassessed"


@dataclass(frozen=True, slots=True)
class SearchConfidence:
    """Top-result confidence evidence without filtering retrieval results."""

    status: SearchConfidenceStatus
    score: float | None
    threshold: float
    warning: str | None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Final results plus complete intermediate evidence."""

    query: str
    results: tuple[SearchCandidate, ...]
    reranked: tuple[SearchCandidate, ...]
    fused_candidates: tuple[SearchCandidate, ...]
    recall: HybridRecallResponse
    timings: SearchTimings
    counts: SearchCandidateCounts
    reranker_used: bool
    confidence: SearchConfidence
