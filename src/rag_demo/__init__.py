"""Public Python API for the Minimal Hybrid RAG retrieval demo."""

from rag_demo.application import DoctorReport
from rag_demo.bm25_retriever import BM25BuildResult
from rag_demo.config import Settings
from rag_demo.dense_retriever import DenseSearchMode
from rag_demo.ingest_service import IngestResult
from rag_demo.query_rewriter import QueryRewriteExperiment, QuerySearchResponse
from rag_demo.sdk import (
    RAG,
    AsyncRAG,
    RAGClosedError,
    RAGNotStartedError,
    RAGSDKError,
    RAGSyncInAsyncContextError,
)
from rag_demo.search_service import SearchCandidate, SearchRequest, SearchResponse

__version__ = "0.1.0"

__all__ = [
    "AsyncRAG",
    "BM25BuildResult",
    "DenseSearchMode",
    "DoctorReport",
    "IngestResult",
    "QueryRewriteExperiment",
    "QuerySearchResponse",
    "RAG",
    "RAGClosedError",
    "RAGNotStartedError",
    "RAGSDKError",
    "RAGSyncInAsyncContextError",
    "SearchCandidate",
    "SearchRequest",
    "SearchResponse",
    "Settings",
    "__version__",
]
