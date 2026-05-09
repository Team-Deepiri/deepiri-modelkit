"""
Universal RAG Base Classes
Core abstractions for retrieval-augmented generation across all industries
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class DocumentType(Enum):
    """Types of documents that can be indexed"""
    REGULATION = "regulation"  # Laws, regulations, compliance documents
    POLICY = "policy"  # Insurance policies, company policies
    MANUAL = "manual"  # Equipment manuals, operation guides
    CONTRACT = "contract"  # Legal contracts, agreements
    WORK_ORDER = "work_order"  # Maintenance work orders, service requests
    CLAIM_RECORD = "claim_record"  # Insurance claims, warranty claims
    MAINTENANCE_LOG = "maintenance_log"  # Equipment maintenance history
    FAQ = "faq"  # Frequently asked questions
    KNOWLEDGE_BASE = "knowledge_base"  # General knowledge articles
    REPORT = "report"  # Inspection reports, audit reports
    PROCEDURE = "procedure"  # Standard operating procedures
    SAFETY_GUIDELINE = "safety_guideline"  # Safety protocols and guidelines
    TECHNICAL_SPEC = "technical_spec"  # Technical specifications
    INVOICE = "invoice"  # Billing and invoices
    OTHER = "other"  # Catch-all for other document types


class IndustryNiche(Enum):
    """Supported industry niches"""
    INSURANCE = "insurance"  # Property & casualty insurance
    MANUFACTURING = "manufacturing"  # Industrial manufacturing
    PROPERTY_MANAGEMENT = "property_management"  # Real estate management
    HEALTHCARE = "healthcare"  # Healthcare providers
    CONSTRUCTION = "construction"  # Construction industry
    AUTOMOTIVE = "automotive"  # Automotive services
    ENERGY = "energy"  # Energy & utilities
    LOGISTICS = "logistics"  # Transportation & logistics
    RETAIL = "retail"  # Retail operations
    HOSPITALITY = "hospitality"  # Hotels & hospitality
    GENERIC = "generic"  # Cross-industry


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    # Industry configuration
    industry: IndustryNiche = IndustryNiche.GENERIC
    
    # Vector database configuration
    vector_db_type: str = "milvus"  # milvus, pinecone, weaviate, memory
    collection_name: str = "deepiri_universal_rag"
    vector_db_host: str = "milvus"
    vector_db_port: int = 19530
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval configuration
    top_k: int = 5  # Number of documents to retrieve
    similarity_threshold: float = 0.7  # Minimum similarity score
    use_reranking: bool = True  # Use cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking configuration
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    
    # Metadata filtering
    enable_metadata_filtering: bool = True
    date_range_filtering: bool = True
    
    # Multi-modal support
    support_images: bool = False
    support_tables: bool = True
    support_code: bool = False


@dataclass
class Document:
    """Universal document representation"""
    id: str
    content: str
    doc_type: DocumentType
    industry: IndustryNiche
    
    # Metadata
    title: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    version: Optional[str] = None
    
    # Industry-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "doc_type": self.doc_type.value,
            "industry": self.industry.value,
            "title": self.title,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "author": self.author,
            "version": self.version,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            doc_type=DocumentType(data["doc_type"]),
            industry=IndustryNiche(data["industry"]),
            title=data.get("title"),
            source=data.get("source"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            author=data.get("author"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
            chunk_index=data.get("chunk_index"),
            total_chunks=data.get("total_chunks"),
        )


@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    document: Document
    score: float  # Similarity score
    rerank_score: Optional[float] = None  # Reranking score if applicable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rerank_score": self.rerank_score,
        }


@dataclass
class RAGQuery:
    """Query for RAG system"""
    query: str
    industry: Optional[IndustryNiche] = None
    doc_types: Optional[List[DocumentType]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "industry": self.industry.value if self.industry else None,
            "doc_types": [dt.value for dt in self.doc_types] if self.doc_types else None,
            "date_range": [dr.isoformat() for dr in self.date_range] if self.date_range else None,
            "metadata_filters": self.metadata_filters,
            "top_k": self.top_k,
        }


class UniversalRAGEngine(ABC):
    """
    Abstract base class for universal RAG engine
    Implements common RAG patterns across all industries
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize RAG components (vector store, embeddings, etc.)"""
        pass
    
    @abstractmethod
    def index_document(self, document: Document) -> bool:
        """
        Index a single document
        
        Args:
            document: Document to index
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Index multiple documents in batch
        
        Args:
            documents: List of documents to index
            
        Returns:
            Statistics about the indexing operation
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query with filters and parameters
            
        Returns:
            List of retrieval results with scores
        """
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        llm_prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents for context
            llm_prompt_template: Optional custom prompt template
            
        Returns:
            Generated response with metadata
        """
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        pass
    
    def search(
        self,
        query: str,
        industry: Optional[IndustryNiche] = None,
        doc_types: Optional[List[DocumentType]] = None,
        top_k: Optional[int] = None,
        **filters
    ) -> List[RetrievalResult]:
        """
        Convenience method for simple search
        
        Args:
            query: Search query
            industry: Filter by industry
            doc_types: Filter by document types
            top_k: Number of results
            **filters: Additional metadata filters
            
        Returns:
            List of retrieval results
        """
        rag_query = RAGQuery(
            query=query,
            industry=industry,
            doc_types=doc_types,
            top_k=top_k or self.config.top_k,
            metadata_filters=filters if filters else None
        )
        return self.retrieve(rag_query)

