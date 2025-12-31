"""
Universal RAG Module for Deepiri Platform
Reusable across all industry niches: Insurance, Manufacturing, Property Management, Healthcare, etc.
"""

from .base import (
    UniversalRAGEngine,
    DocumentType,
    IndustryNiche,
    RAGConfig,
)
from .processors import (
    DocumentProcessor,
    RegulationProcessor,
    HistoricalDataProcessor,
    KnowledgeBaseProcessor,
)
from .retrievers import (
    MultiModalRetriever,
    HybridRetriever,
)

__all__ = [
    "UniversalRAGEngine",
    "DocumentType",
    "IndustryNiche",
    "RAGConfig",
    "DocumentProcessor",
    "RegulationProcessor",
    "HistoricalDataProcessor",
    "KnowledgeBaseProcessor",
    "MultiModalRetriever",
    "HybridRetriever",
]

