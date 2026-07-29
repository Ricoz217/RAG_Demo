"""Command-line interface for the Hybrid RAG demo."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_demo import __version__
from rag_demo.application import DoctorReport, RAGApplication
from rag_demo.asyncio_compat import run_async
from rag_demo.config import Settings
from rag_demo.db import Database, DatabaseStatus
from rag_demo.dense_retriever import DenseSearchMode
from rag_demo.ingest_service import IngestResult
from rag_demo.migrations import apply_migrations
from rag_demo.search_service import SearchCandidate, SearchRequest, SearchResponse

app = typer.Typer(
    name="rag-demo",
    help="A minimal, observable Hybrid RAG retrieval demo.",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Initialize and inspect PostgreSQL.")
bm25_app = typer.Typer(help="Build and inspect the local BM25 index.")
app.add_typer(db_app, name="db")
app.add_typer(bm25_app, name="bm25")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = False,
) -> None:
    """Run the observable Hybrid RAG retrieval demo."""


@db_app.command("init")
def db_init() -> None:
    """Apply pending PostgreSQL migrations."""
    settings = Settings()
    applied = run_async(apply_migrations(settings.database_url.get_secret_value()))

    if applied:
        console.print(f"Applied migrations: {', '.join(applied)}")
    else:
        console.print("Database schema is up to date.")


async def _database_status(settings: Settings) -> DatabaseStatus:
    async with Database(settings.database_url.get_secret_value()) as database:
        return await database.status()


@db_app.command("status")
def db_status() -> None:
    """Display PostgreSQL, pgvector, schema, and corpus status."""
    status = run_async(_database_status(Settings()))
    table = Table(title="Database status")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("User", status.current_user)
    table.add_row("Database", status.current_database)
    table.add_row("PostgreSQL", status.server_version)
    table.add_row("pgvector", status.vector_version)
    table.add_row("Embedding column", status.embedding_column_type)
    table.add_row("HNSW", "HNSW ready" if status.hnsw_index_present else "HNSW missing")
    table.add_row("Documents", str(status.document_count))
    table.add_row("Chunks", str(status.chunk_count))
    table.add_row("Migrations", ", ".join(status.applied_migrations))
    console.print(table)


async def _run_doctor(settings: Settings) -> DoctorReport:
    async with RAGApplication(settings) as application:
        return await application.doctor()


@app.command("doctor")
def doctor() -> None:
    """Exercise PostgreSQL, model APIs, BM25, and corpus compatibility."""
    report = run_async(_run_doctor(Settings()))
    table = Table(title="Infrastructure doctor")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row("PASS" if check.passed else "FAIL", check.name, check.detail)
    console.print(table)
    if not report.passed:
        raise typer.Exit(code=1)


async def _ingest(
    settings: Settings,
    *,
    source: Path,
    source_repo: str,
    source_commit: str,
    source_root: Path,
    language: str,
    idempotency_key: str,
) -> IngestResult:
    async with RAGApplication(settings) as application:
        ingestor = application.create_ingestor(
            source_repo=source_repo,
            source_commit=source_commit,
            source_root=source_root,
            language=language,
        )
        return await ingestor.ingest_path(
            source,
            idempotency_key=idempotency_key,
        )


@app.command("ingest")
def ingest(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Markdown file or directory to ingest.",
        ),
    ],
    source_repo: Annotated[
        str,
        typer.Option(help="Source repository URL or stable local identifier."),
    ],
    source_commit: Annotated[
        str,
        typer.Option(help="Git commit SHA or stable source revision."),
    ],
    source_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Root used to derive stored relative source paths.",
        ),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option(help="Stable key for this exact ingestion request."),
    ],
    language: Annotated[
        str,
        typer.Option(help="BCP-47-style source language label."),
    ] = "zh",
) -> None:
    """Parse, embed, and atomically upsert Markdown documents."""
    result = run_async(
        _ingest(
            Settings(),
            source=source,
            source_repo=source_repo,
            source_commit=source_commit,
            source_root=source_root,
            language=language,
            idempotency_key=idempotency_key,
        )
    )
    table = Table(title="Ingestion result")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, value in (
        ("Documents", result.document_count),
        ("Chunks", result.chunk_count),
        ("Embedded", result.embedded_chunk_count),
        ("Embedding skipped", result.skipped_embedding_count),
        ("Stale chunks deleted", result.deleted_chunk_count),
        ("Embedding HTTP requests", result.embedding_http_request_count),
        ("Total ms", f"{result.total_ms:.2f}"),
    ):
        table.add_row(label, str(value))
    console.print(table)


async def _rebuild_bm25(settings: Settings) -> tuple[str, int, float]:
    async with RAGApplication(settings) as application:
        result = await application.rebuild_bm25()
        return result.generation, result.chunk_count, result.build_ms


@bm25_app.command("rebuild")
def bm25_rebuild() -> None:
    """Rebuild BM25 from PostgreSQL and atomically publish CURRENT."""
    generation, chunk_count, build_ms = run_async(_rebuild_bm25(Settings()))
    table = Table(title="BM25 rebuild")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("BM25 generation", generation)
    table.add_row("Chunks", str(chunk_count))
    table.add_row("Build ms", f"{build_ms:.2f}")
    console.print(table)


async def _execute_search(
    settings: Settings,
    request: SearchRequest,
) -> SearchResponse:
    async with RAGApplication(settings) as application:
        return await application.search(request)


def _search_request(
    settings: Settings,
    *,
    query: str,
    dense_top_k: int | None,
    bm25_top_k: int | None,
    rerank_top_k: int | None,
    final_top_k: int | None,
    dense_mode: DenseSearchMode,
    rerank: bool,
    debug: bool,
) -> SearchRequest:
    return SearchRequest(
        query=query,
        dense_top_k=dense_top_k or settings.dense_top_k,
        bm25_top_k=bm25_top_k or settings.bm25_top_k,
        rrf_rank_constant=settings.rrf_rank_constant,
        rerank_top_k=rerank_top_k or settings.rerank_top_k,
        final_top_k=final_top_k or settings.final_top_k,
        dense_mode=dense_mode,
        use_reranker=rerank,
        debug=debug,
    )


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Natural-language search query.")],
    dense_top_k: Annotated[
        int | None,
        typer.Option(min=1, help="Override Dense candidate count."),
    ] = None,
    bm25_top_k: Annotated[
        int | None,
        typer.Option(min=1, help="Override BM25 candidate count."),
    ] = None,
    rerank_top_k: Annotated[
        int | None,
        typer.Option(min=1, help="Override Reranker candidate count."),
    ] = None,
    final_top_k: Annotated[
        int | None,
        typer.Option(min=1, help="Override final result count."),
    ] = None,
    dense_mode: Annotated[
        DenseSearchMode,
        typer.Option(help="Dense execution mode: exact or hnsw."),
    ] = DenseSearchMode.HNSW,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Enable Cross-Encoder reranking."),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option(help="Display every intermediate ranking."),
    ] = False,
) -> None:
    """Run the complete Hybrid RAG retrieval pipeline."""
    settings = Settings()
    request = _search_request(
        settings,
        query=query,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        rerank_top_k=rerank_top_k,
        final_top_k=final_top_k,
        dense_mode=dense_mode,
        rerank=rerank,
        debug=debug,
    )
    response = run_async(_execute_search(settings, request))
    _print_search_response(response, debug=debug)


@app.command("compare")
def compare(
    query: Annotated[str, typer.Argument(help="Query used for all four rankings.")],
) -> None:
    """Compare BM25, Dense, Hybrid RRF, and reranked results."""
    settings = Settings()
    request = _search_request(
        settings,
        query=query,
        dense_top_k=None,
        bm25_top_k=None,
        rerank_top_k=None,
        final_top_k=None,
        dense_mode=DenseSearchMode.HNSW,
        rerank=True,
        debug=True,
    )
    response = run_async(_execute_search(settings, request))
    console.print(f"Query: {response.query}")
    _print_branch_results(
        "BM25 only",
        (
            (
                result.rank,
                result.chunk_id,
                result.score,
                result.source_path,
                result.heading_path,
            )
            for result in response.recall.bm25.results[: settings.final_top_k]
        ),
    )
    _print_branch_results(
        "Dense only",
        (
            (
                result.rank,
                result.chunk_id,
                result.cosine_similarity,
                result.source_path,
                result.heading_path,
            )
            for result in response.recall.dense.results[: settings.final_top_k]
        ),
    )
    _print_candidate_table(
        "Hybrid RRF",
        response.fused_candidates[: settings.final_top_k],
        score_name="RRF score",
        score_getter=lambda item: item.rrf_score,
    )
    _print_candidate_table(
        "Hybrid RRF + Reranker",
        response.results,
        score_name="Rerank score",
        score_getter=lambda item: item.rerank_score,
    )
    _print_timings(response)


def _print_search_response(response: SearchResponse, *, debug: bool) -> None:
    console.print(f"Query: {response.query}")
    _print_candidate_table(
        "Final results",
        response.results,
        score_name="Final score",
        score_getter=lambda item: (
            item.rerank_score if item.rerank_score is not None else item.rrf_score
        ),
    )
    if debug:
        _print_branch_results(
            "Dense Top-K",
            (
                (
                    result.rank,
                    result.chunk_id,
                    result.cosine_similarity,
                    result.source_path,
                    result.heading_path,
                )
                for result in response.recall.dense.results
            ),
        )
        _print_branch_results(
            "BM25 Top-K",
            (
                (
                    result.rank,
                    result.chunk_id,
                    result.score,
                    result.source_path,
                    result.heading_path,
                )
                for result in response.recall.bm25.results
            ),
        )
        _print_candidate_table(
            "RRF Top-K",
            response.fused_candidates,
            score_name="RRF score",
            score_getter=lambda item: item.rrf_score,
        )
        if response.reranker_used:
            _print_candidate_table(
                "Reranker Top-K",
                response.reranked,
                score_name="Rerank score",
                score_getter=lambda item: item.rerank_score,
            )
    _print_timings(response)


def _print_candidate_table(
    title: str,
    candidates: tuple[SearchCandidate, ...],
    *,
    score_name: str,
    score_getter: Callable[[SearchCandidate], float | None],
) -> None:
    table = Table(title=title)
    table.add_column("Rank", justify="right")
    table.add_column("Chunk")
    table.add_column("Title / Heading")
    table.add_column("Source")
    table.add_column(score_name, justify="right")
    table.add_column("Preview")
    for display_rank, candidate in enumerate(candidates, start=1):
        score = score_getter(candidate)
        table.add_row(
            str(display_rank),
            str(candidate.chunk_id),
            _title_heading(candidate.title, candidate.heading_path),
            candidate.source_path,
            "" if score is None else f"{score:.6f}",
            _preview(candidate.content_raw),
        )
    console.print(table)


def _print_branch_results(
    title: str,
    rows: Iterable[tuple[int, int, float, str, tuple[str, ...]]],
) -> None:
    table = Table(title=title)
    table.add_column("Rank", justify="right")
    table.add_column("Chunk")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Heading")
    for rank, chunk_id, score, source_path, heading_path in rows:
        table.add_row(
            str(rank),
            str(chunk_id),
            f"{score:.6f}",
            source_path,
            " / ".join(heading_path),
        )
    console.print(table)


def _print_timings(response: SearchResponse) -> None:
    table = Table(title="Stage timings")
    table.add_column("Stage")
    table.add_column("Milliseconds", justify="right")
    for name, value in (
        ("Query embedding", response.timings.query_embedding_ms),
        ("Dense search", response.timings.dense_search_ms),
        ("BM25 search", response.timings.bm25_search_ms),
        ("RRF", response.timings.rrf_ms),
        ("Reranker", response.timings.rerank_ms),
        ("Total", response.timings.total_ms),
    ):
        table.add_row(name, f"{value:.2f}")
    console.print(table)


def _title_heading(title: str | None, heading_path: tuple[str, ...]) -> str:
    parts = [title or "(untitled)"]
    if heading_path:
        parts.append(" / ".join(heading_path))
    return "\n".join(parts)


def _preview(content: str, *, limit: int = 100) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"
