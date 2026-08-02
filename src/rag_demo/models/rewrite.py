"""Query rewrite and comparison data contracts."""

from __future__ import annotations

from dataclasses import dataclass

from rag_demo.models.retrieval import SearchResponse


@dataclass(frozen=True, slots=True)
class AliasRule:
    """One case-insensitive alias-to-canonical expansion rule."""

    alias: str
    canonical: str


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """Observable result of deterministic query rewriting."""

    original_query: str
    normalized_query: str
    effective_query: str
    applied_rules: tuple[str, ...]
    rewrite_ms: float

    @property
    def changed(self) -> bool:
        """Return whether the effective query differs from caller input."""
        return self.effective_query != self.original_query

    @classmethod
    def disabled(cls, query: str) -> QueryRewriteResult:
        """Describe a deliberately bypassed rewrite stage."""
        return cls(
            original_query=query,
            normalized_query=query,
            effective_query=query,
            applied_rules=(),
            rewrite_ms=0.0,
        )

    def to_json(self) -> dict[str, str | bool | float | list[str]]:
        """Return a transport-friendly debug representation."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "effective_query": self.effective_query,
            "applied_rules": list(self.applied_rules),
            "changed": self.changed,
            "rewrite_ms": self.rewrite_ms,
        }


@dataclass(frozen=True, slots=True)
class ResultRankChange:
    """Before/after final rank for one chunk, without quality judgement."""

    chunk_id: int
    without_rank: int | None
    with_rank: int | None
    rank_delta: int | None


@dataclass(frozen=True, slots=True)
class ResultRankingComparison:
    """Observable final Top-K membership and ordering difference."""

    same_order: bool
    overlap_chunk_ids: tuple[int, ...]
    only_without_chunk_ids: tuple[int, ...]
    only_with_chunk_ids: tuple[int, ...]
    rank_changes: tuple[ResultRankChange, ...]


@dataclass(frozen=True, slots=True)
class QuerySearchResponse:
    """One search response plus the query transformation that preceded it."""

    rewrite_enabled: bool
    rewrite: QueryRewriteResult
    response: SearchResponse


@dataclass(frozen=True, slots=True)
class QueryRewriteExperiment:
    """Two real retrieval runs and their neutral final-ranking difference."""

    without_rewrite: QuerySearchResponse
    with_rewrite: QuerySearchResponse
    comparison: ResultRankingComparison
