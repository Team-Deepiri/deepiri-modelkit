"""
Universal RAG Module for Deepiri Platform
Reusable across all industry niches: Insurance, Manufacturing, Property Management, Healthcare, etc.
"""

from .base import (
    UniversalRAGEngine,
    Document,
    DocumentType,
    IndustryNiche,
    RAGConfig,
    RAGQuery,
    RetrievalResult,
)
from .processors import (
    DocumentProcessor,
    RegulationProcessor,
    HistoricalDataProcessor,
    KnowledgeBaseProcessor,
    ManualProcessor,
    get_processor,
)
from .retrievers import (
    MultiModalRetriever,
    HybridRetriever,
    ContextualRetriever,
    get_retriever,
)

# Advanced features (optional imports)
try:
    from .advanced_retrieval import (
        AdvancedRetrievalPipeline,
        QueryExpander,
        SynonymQueryExpander,
        RephraseQueryExpander,
        MultiQueryRetriever,
        QueryCache,
    )
    HAS_ADVANCED_RETRIEVAL = True
except ImportError:
    HAS_ADVANCED_RETRIEVAL = False
    AdvancedRetrievalPipeline = None
    QueryExpander = None
    SynonymQueryExpander = None
    RephraseQueryExpander = None
    MultiQueryRetriever = None
    QueryCache = None

try:
    from .caching import (
        AdvancedCacheManager,
        EmbeddingCache,
        QueryResultCache,
    )
    HAS_CACHING = True
except ImportError:
    HAS_CACHING = False
    AdvancedCacheManager = None
    EmbeddingCache = None
    QueryResultCache = None

try:
    from .monitoring import (
        RAGMonitor,
        RetrievalMetrics,
        IndexingMetrics,
        SystemMetrics,
        PerformanceTimer,
    )
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    RAGMonitor = None
    RetrievalMetrics = None
    IndexingMetrics = None
    SystemMetrics = None
    PerformanceTimer = None

try:
    from .async_processing import (
        AsyncBatchProcessor,
        AsyncDocumentIndexer,
        AsyncDocumentProcessor,
        BatchProcessingConfig,
        BatchProcessingResult,
    )
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False
    AsyncBatchProcessor = None
    AsyncDocumentIndexer = None
    AsyncDocumentProcessor = None
    BatchProcessingConfig = None
    BatchProcessingResult = None

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
]

# Conditionally add advanced features
if HAS_ADVANCED_RETRIEVAL:
    __all__.extend([
        "AdvancedRetrievalPipeline",
        "QueryExpander",
        "SynonymQueryExpander",
        "RephraseQueryExpander",
        "MultiQueryRetriever",
        "QueryCache",
    ])

if HAS_CACHING:
    __all__.extend([
        "AdvancedCacheManager",
        "EmbeddingCache",
        "QueryResultCache",
    ])

if HAS_MONITORING:
    __all__.extend([
        "RAGMonitor",
        "RetrievalMetrics",
        "IndexingMetrics",
        "SystemMetrics",
        "PerformanceTimer",
    ])

if HAS_ASYNC:
    __all__.extend([
        "AsyncBatchProcessor",
        "AsyncDocumentIndexer",
        "AsyncDocumentProcessor",
        "BatchProcessingConfig",
        "BatchProcessingResult",
    ])

