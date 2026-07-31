"""FastAPI adapter over the shared asynchronous RAG application."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_demo.application import RAGApplication
from rag_demo.bm25_retriever import BM25IndexError
from rag_demo.config import Settings
from rag_demo.dense_retriever import DenseSearchMode
from rag_demo.ingest_service import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
)
from rag_demo.search_service import SearchRequest, SearchResponse


class SearchBody(BaseModel):
    """REST search request with Settings-backed optional overrides."""

    query: str = Field(min_length=1)
    dense_top_k: int | None = Field(default=None, gt=0)
    bm25_top_k: int | None = Field(default=None, gt=0)
    rrf_rank_constant: int | None = Field(default=None, gt=0)
    rerank_top_k: int | None = Field(default=None, gt=0)
    final_top_k: int | None = Field(default=None, gt=0)
    dense_mode: DenseSearchMode = DenseSearchMode.HNSW
    use_reranker: bool = True
    debug: bool = False


class IngestBody(BaseModel):
    """Local-only Markdown ingestion request."""

    source: Path
    source_repo: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    source_root: Path | None = None
    language: str = Field(default="zh", min_length=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI app with one shared RAGApplication lifespan."""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        application = RAGApplication(resolved_settings)
        await application.open()
        app.state.rag_application = application
        try:
            try:
                await application.load_bm25()
            except BM25IndexError:
                pass
            yield
        finally:
            await application.close()

    app = FastAPI(
        title="Minimal Hybrid RAG Demo",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        application = _application(request)
        try:
            database_status = await application.database.status()
            ready = (
                database_status.hnsw_index_present
                and database_status.chunk_count > 0
                and application.bm25_ready
            )
            payload = {
                "status": "ready" if ready else "not_ready",
                "chunk_count": database_status.chunk_count,
                "bm25_ready": application.bm25_ready,
            }
        except Exception:
            ready = False
            payload = {
                "status": "not_ready",
                "chunk_count": 0,
                "bm25_ready": False,
            }
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    @app.get("/v1/stats")
    async def stats_endpoint(request: Request) -> dict[str, Any]:
        application = _application(request)
        database_status = await application.database.status()
        bm25 = application.bm25_retriever
        return {
            "document_count": database_status.document_count,
            "chunk_count": database_status.chunk_count,
            "embedding_model": resolved_settings.embedding_model,
            "embedding_dimensions": resolved_settings.embedding_dimensions,
            "hnsw_ready": database_status.hnsw_index_present,
            "bm25": (
                {
                    "ready": True,
                    "generation": bm25.generation,
                    "chunk_count": bm25.chunk_count,
                }
                if bm25 is not None
                else {
                    "ready": False,
                    "generation": None,
                    "chunk_count": 0,
                }
            ),
        }

    @app.post("/v1/search")
    async def search_endpoint(
        body: SearchBody,
        request: Request,
    ) -> dict[str, Any]:
        application = _application(request)
        try:
            core_request = SearchRequest(
                query=body.query,
                dense_top_k=body.dense_top_k or resolved_settings.dense_top_k,
                bm25_top_k=body.bm25_top_k or resolved_settings.bm25_top_k,
                rrf_rank_constant=(body.rrf_rank_constant or resolved_settings.rrf_rank_constant),
                rerank_top_k=body.rerank_top_k or resolved_settings.rerank_top_k,
                final_top_k=body.final_top_k or resolved_settings.final_top_k,
                dense_mode=body.dense_mode,
                use_reranker=body.use_reranker,
                debug=body.debug,
            )
            response = await application.search(core_request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BM25IndexError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _search_payload(response, debug=body.debug)

    @app.post("/v1/ingest")
    async def ingest_endpoint(
        body: IngestBody,
        request: Request,
        response: Response,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", min_length=1),
        ] = None,
    ) -> dict[str, int | float]:
        application = _application(request)
        source = body.source.resolve()
        source_root = (
            body.source_root.resolve()
            if body.source_root is not None
            else _infer_source_root(source)
        )
        ingestor = application.create_ingestor(
            source_repo=body.source_repo,
            source_commit=body.source_commit,
            source_root=source_root,
            language=body.language,
        )
        effective_key = (
            ingestor.idempotency_key_for(source)
            if idempotency_key is None
            else idempotency_key
        )
        try:
            result = await ingestor.ingest_path(
                source,
                idempotency_key=effective_key,
            )
        except (IdempotencyConflictError, IdempotencyInProgressError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="source path not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response.headers["Idempotency-Key"] = effective_key
        return result.to_json()

    @app.post("/v1/bm25/rebuild")
    async def bm25_rebuild_endpoint(
        request: Request,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1),
        ],
    ) -> dict[str, str | int | float]:
        application = _application(request)
        try:
            result = await application.rebuild_bm25_idempotent(
                idempotency_key=idempotency_key,
            )
        except (IdempotencyConflictError, IdempotencyInProgressError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result.to_json()

    return app


def _application(request: Request) -> RAGApplication:
    application: RAGApplication = request.app.state.rag_application
    return application


def _infer_source_root(source: Path) -> Path:
    current = source if source.is_dir() else source.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _search_payload(response: SearchResponse, *, debug: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": response.query,
        "results": [asdict(candidate) for candidate in response.results],
        "timings": asdict(response.timings),
        "counts": asdict(response.counts),
        "reranker_used": response.reranker_used,
    }
    if debug:
        payload["debug"] = {
            "dense": [asdict(result) for result in response.recall.dense.results],
            "bm25": [asdict(result) for result in response.recall.bm25.results],
            "rrf": [asdict(candidate) for candidate in response.fused_candidates],
            "reranker": [asdict(candidate) for candidate in response.reranked],
        }
    encoded: dict[str, Any] = jsonable_encoder(payload)
    return encoded
