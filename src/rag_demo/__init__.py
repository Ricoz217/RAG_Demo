"""Public Python API for the Minimal Hybrid RAG retrieval demo."""

from rag_demo.config import Settings
from rag_demo.models import (
    BM25BuildResult,
    DenseSearchMode,
    DoctorReport,
    IngestResult,
    QueryRewriteExperiment,
    QuerySearchResponse,
    SearchCandidate,
    SearchConfidence,
    SearchConfidenceStatus,
    SearchRequest,
    SearchResponse,
)
from rag_demo.sdk import (
    RAG,
    AsyncRAG,
    RAGClosedError,
    RAGNotStartedError,
    RAGSDKError,
    RAGSyncInAsyncContextError,
)

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
    "SearchConfidence",
    "SearchConfidenceStatus",
    "SearchRequest",
    "SearchResponse",
    "Settings",
    "__version__",
]
