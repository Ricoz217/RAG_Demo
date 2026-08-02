"""Deterministic query normalization, alias expansion, and neutral A/B comparison."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from rag_demo.models.retrieval import SearchRequest, SearchResponse
from rag_demo.models.rewrite import (
    AliasRule,
    QueryRewriteExperiment,
    QueryRewriteResult,
    QuerySearchResponse,
    ResultRankChange,
    ResultRankingComparison,
)

__all__ = [
    "AliasRule",
    "QueryRewriteConfigurationError",
    "QueryRewriteExperiment",
    "QueryRewriter",
    "QueryRewriteResult",
    "QueryRewriteSearchService",
    "QuerySearchResponse",
    "ResultRankChange",
    "ResultRankingComparison",
    "SearchProvider",
    "compare_result_rankings",
]


class QueryRewriteConfigurationError(ValueError):
    """Raised when the committed alias dictionary cannot be loaded safely."""


class SearchProvider(Protocol):
    """Core search behavior wrapped by the optional rewrite stage."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute one retrieval request."""


class QueryRewriter:
    """Apply deterministic normalization and non-cascading alias expansion."""

    def __init__(self, rules: Sequence[AliasRule]) -> None:
        if not rules:
            raise QueryRewriteConfigurationError("aliases must not be empty")
        self._rules_by_folded_alias = {rule.alias.casefold(): rule for rule in rules}
        alternatives = sorted(
            (self._alias_pattern(rule.alias) for rule in rules),
            key=len,
            reverse=True,
        )
        self._pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    @classmethod
    def from_json(cls, path: Path) -> QueryRewriter:
        """Load and validate a versioned UTF-8 alias dictionary."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise QueryRewriteConfigurationError(
                f"could not read query alias dictionary: {path}"
            ) from exc
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise QueryRewriteConfigurationError(
                f"query alias dictionary is not valid JSON: {path}"
            ) from exc
        return cls(_parse_dictionary(payload))

    def rewrite(self, query: str) -> QueryRewriteResult:
        """Normalize and expand a query while retaining every matched alias."""
        started = time.perf_counter()
        normalized = unicodedata.normalize("NFKC", query)
        applied_rules: list[str] = []
        if normalized != query:
            applied_rules.append("unicode_nfkc")

        compact = " ".join(normalized.split())
        if compact != normalized:
            applied_rules.append("whitespace")
        if not compact:
            raise ValueError("query must not be empty")

        seen_aliases: set[str] = set()

        def expand(match: re.Match[str]) -> str:
            surface = match.group(0)
            folded = surface.casefold()
            rule = self._rules_by_folded_alias[folded]
            if self._already_expanded(compact, match, rule):
                return surface
            if rule.canonical.casefold() == folded:
                return surface
            if folded not in seen_aliases:
                applied_rules.append(f"alias:{rule.alias}")
                seen_aliases.add(folded)
            return f"{rule.canonical} ({surface})"

        effective = self._pattern.sub(expand, compact)
        return QueryRewriteResult(
            original_query=query,
            normalized_query=compact,
            effective_query=effective,
            applied_rules=tuple(applied_rules),
            rewrite_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _alias_pattern(alias: str) -> str:
        escaped = re.escape(alias)
        prefix = r"(?<![A-Za-z0-9])" if alias[0].isascii() and alias[0].isalnum() else ""
        suffix = r"(?![A-Za-z0-9])" if alias[-1].isascii() and alias[-1].isalnum() else ""
        return f"{prefix}{escaped}{suffix}"

    @staticmethod
    def _already_expanded(text: str, match: re.Match[str], rule: AliasRule) -> bool:
        prefix = f"{rule.canonical} ("
        return (
            match.start() >= len(prefix)
            and text[match.start() - len(prefix) : match.start()].casefold() == prefix.casefold()
            and match.end() < len(text)
            and text[match.end()] == ")"
        )


class QueryRewriteSearchService:
    """Place deterministic rewriting immediately before the retrieval pipeline."""

    def __init__(
        self,
        *,
        search_provider: SearchProvider,
        rewriter: QueryRewriter,
    ) -> None:
        self._search_provider = search_provider
        self._rewriter = rewriter

    async def search(
        self,
        request: SearchRequest,
        *,
        rewrite: bool,
    ) -> QuerySearchResponse:
        """Run one search with an explicit enabled/disabled rewrite stage."""
        rewrite_result = (
            self._rewriter.rewrite(request.query)
            if rewrite
            else QueryRewriteResult.disabled(request.query)
        )
        effective_request = replace(request, query=rewrite_result.effective_query)
        response = await self._search_provider.search(effective_request)
        return QuerySearchResponse(
            rewrite_enabled=rewrite,
            rewrite=rewrite_result,
            response=response,
        )

    async def compare(self, request: SearchRequest) -> QueryRewriteExperiment:
        """Execute both variants even when rewriting leaves the query unchanged."""
        without_rewrite = await self.search(request, rewrite=False)
        with_rewrite = await self.search(request, rewrite=True)
        comparison = compare_result_rankings(
            without_rewrite=tuple(
                candidate.chunk_id for candidate in without_rewrite.response.results
            ),
            with_rewrite=tuple(candidate.chunk_id for candidate in with_rewrite.response.results),
        )
        return QueryRewriteExperiment(
            without_rewrite=without_rewrite,
            with_rewrite=with_rewrite,
            comparison=comparison,
        )


def compare_result_rankings(
    *,
    without_rewrite: Sequence[int],
    with_rewrite: Sequence[int],
) -> ResultRankingComparison:
    """Compare two final result lists without labelling either one as better."""
    without = tuple(without_rewrite)
    with_ = tuple(with_rewrite)
    without_ranks = {chunk_id: rank for rank, chunk_id in enumerate(without, start=1)}
    with_ranks = {chunk_id: rank for rank, chunk_id in enumerate(with_, start=1)}
    ordered_union = (*without, *(chunk_id for chunk_id in with_ if chunk_id not in without_ranks))

    changes = tuple(
        ResultRankChange(
            chunk_id=chunk_id,
            without_rank=without_ranks.get(chunk_id),
            with_rank=with_ranks.get(chunk_id),
            rank_delta=(
                without_ranks[chunk_id] - with_ranks[chunk_id]
                if chunk_id in without_ranks and chunk_id in with_ranks
                else None
            ),
        )
        for chunk_id in ordered_union
    )
    return ResultRankingComparison(
        same_order=without == with_,
        overlap_chunk_ids=tuple(chunk_id for chunk_id in without if chunk_id in with_ranks),
        only_without_chunk_ids=tuple(
            chunk_id for chunk_id in without if chunk_id not in with_ranks
        ),
        only_with_chunk_ids=tuple(chunk_id for chunk_id in with_ if chunk_id not in without_ranks),
        rank_changes=changes,
    )


def _parse_dictionary(payload: Any) -> tuple[AliasRule, ...]:
    if not isinstance(payload, Mapping):
        raise QueryRewriteConfigurationError("query alias dictionary root must be an object")
    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise QueryRewriteConfigurationError("unsupported query alias dictionary version")
    aliases = payload.get("aliases")
    if not isinstance(aliases, Mapping) or not aliases:
        raise QueryRewriteConfigurationError("aliases must be a non-empty object")

    rules: list[AliasRule] = []
    seen: set[str] = set()
    for alias, canonical in aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            raise QueryRewriteConfigurationError("alias must be a non-empty string")
        if not isinstance(canonical, str) or not canonical.strip():
            raise QueryRewriteConfigurationError(
                f"canonical term for alias {alias!r} must be a non-empty string"
            )
        clean_alias = alias.strip()
        folded = clean_alias.casefold()
        if folded in seen:
            raise QueryRewriteConfigurationError(f"duplicate case-insensitive alias: {clean_alias}")
        seen.add(folded)
        rules.append(AliasRule(alias=clean_alias, canonical=canonical.strip()))
    return tuple(rules)
