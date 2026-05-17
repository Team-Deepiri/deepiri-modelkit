"""
Universal RAG Module for Deepiri Platform
Reusable across all industry niches: Insurance, Manufacturing, Property Management, Healthcare, etc.
"""

from .advanced_retrieval import (
    AdvancedRetrievalPipeline,
    MultiQueryRetriever,
    QueryCache,
    QueryExpander,
    RephraseQueryExpander,
    SynonymQueryExpander,
)
from .async_processing import (
    AsyncBatchProcessor,
    AsyncDocumentIndexer,
    AsyncDocumentProcessor,
    BatchProcessingConfig,
    BatchProcessingResult,
)
from .base import (
    Document,
    DocumentType,
    IndustryNiche,
    RAGConfig,
    RAGQuery,
    RetrievalResult,
    UniversalRAGEngine,
)
from .caching import AdvancedCacheManager, EmbeddingCache, QueryResultCache
from .monitoring import (
    IndexingMetrics,
    PerformanceTimer,
    RAGMonitor,
    RetrievalMetrics,
    SystemMetrics,
)
from .processors import (
    DocumentProcessor,
    HistoricalDataProcessor,
    KnowledgeBaseProcessor,
    ManualProcessor,
    RegulationProcessor,
    get_processor,
)
from .retrievers import (
    ContextualRetriever,
    HybridRetriever,
    MultiModalRetriever,
    get_retriever,
)

__all__ = [
    # Core
    "UniversalRAGEngine",
    "Document",
    "DocumentType",
    "IndustryNiche",
    "RAGConfig",
    "RAGQuery",
    "RetrievalResult",
    # Processors
    "DocumentProcessor",
    "RegulationProcessor",
    "HistoricalDataProcessor",
    "KnowledgeBaseProcessor",
    "ManualProcessor",
    "get_processor",
    # Retrievers
    "MultiModalRetriever",
    "HybridRetriever",
    "ContextualRetriever",
    "get_retriever",
    # Advanced retrieval
    "AdvancedRetrievalPipeline",
    "QueryExpander",
    "SynonymQueryExpander",
    "RephraseQueryExpander",
    "MultiQueryRetriever",
    "QueryCache",
    # Caching
    "AdvancedCacheManager",
    "EmbeddingCache",
    "QueryResultCache",
    # Monitoring
    "RAGMonitor",
    "RetrievalMetrics",
    "IndexingMetrics",
    "SystemMetrics",
    "PerformanceTimer",
    # Async processing
    "AsyncBatchProcessor",
    "AsyncDocumentIndexer",
    "AsyncDocumentProcessor",
    "BatchProcessingConfig",
    "BatchProcessingResult",
]
