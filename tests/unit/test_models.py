"""Public model organization and backwards-compatibility contracts."""

from rag_demo.application import DoctorReport as LegacyDoctorReport
from rag_demo.bm25_retriever import BM25BuildResult as LegacyBM25BuildResult
from rag_demo.chunker import Chunk as LegacyChunk
from rag_demo.db import DatabaseStatus as LegacyDatabaseStatus
from rag_demo.dense_retriever import DenseSearchResult as LegacyDenseSearchResult
from rag_demo.evaluation import BenchmarkReport as LegacyBenchmarkReport
from rag_demo.ingest_service import IngestResult as LegacyIngestResult
from rag_demo.markdown_parser import ParsedMarkdownDocument as LegacyParsedMarkdownDocument
from rag_demo.models import (
    BenchmarkReport,
    BM25BuildResult,
    Chunk,
    DatabaseStatus,
    DenseSearchResult,
    DoctorReport,
    IngestResult,
    ParsedMarkdownDocument,
    QueryRewriteExperiment,
    SearchResponse,
)
from rag_demo.query_rewriter import QueryRewriteExperiment as LegacyQueryRewriteExperiment
from rag_demo.search_service import SearchResponse as LegacySearchResponse


def test_public_models_are_defined_in_the_models_package() -> None:
    public_models = (
        ParsedMarkdownDocument,
        Chunk,
        DenseSearchResult,
        SearchResponse,
        IngestResult,
        QueryRewriteExperiment,
        DatabaseStatus,
        DoctorReport,
        BM25BuildResult,
        BenchmarkReport,
    )

    assert all(model.__module__.startswith("rag_demo.models.") for model in public_models)


def test_legacy_module_imports_reexport_the_same_model_classes() -> None:
    assert LegacyParsedMarkdownDocument is ParsedMarkdownDocument
    assert LegacyChunk is Chunk
    assert LegacyDenseSearchResult is DenseSearchResult
    assert LegacySearchResponse is SearchResponse
    assert LegacyIngestResult is IngestResult
    assert LegacyQueryRewriteExperiment is QueryRewriteExperiment
    assert LegacyDatabaseStatus is DatabaseStatus
    assert LegacyDoctorReport is DoctorReport
    assert LegacyBM25BuildResult is BM25BuildResult
    assert LegacyBenchmarkReport is BenchmarkReport
