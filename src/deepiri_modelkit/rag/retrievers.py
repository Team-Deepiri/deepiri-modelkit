"""
Retrieval Components for Universal RAG
Implements various retrieval strategies for different use cases
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import Document, RetrievalResult, RAGQuery


class BaseRetriever(ABC):
    """Base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """Retrieve relevant documents for query"""
        pass


class MultiModalRetriever(BaseRetriever):
    """
    Multi-modal retriever supporting text, images, tables, and code
    Useful for technical manuals, equipment guides, etc.
    """
    
    def __init__(
        self,
        text_embeddings,
        image_embeddings=None,
        table_embeddings=None,
        code_embeddings=None
    ):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.table_embeddings = table_embeddings
        self.code_embeddings = code_embeddings
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """
        Retrieve documents across multiple modalities
        
        Currently focuses on text, but can be extended for other modalities
        """
        # For now, delegate to text retrieval
        # Future: Add image, table, code retrieval
        results = self._retrieve_text(query)
        return results
    
    def _retrieve_text(self, query: RAGQuery) -> List[RetrievalResult]:
        """Retrieve text documents"""
        # This will be implemented by the concrete RAG engine
        # This is a placeholder for the interface
        return []


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining:
    - Semantic search (vector similarity)
    - Keyword search (BM25)
    - Metadata filtering
    
    Provides better recall than pure semantic search
    """
    
    def __init__(
        self,
        vector_retriever,
        keyword_retriever=None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """
        Retrieve using hybrid approach
        
        Combines vector and keyword search results
        """
        results = []
        
        # Vector search
        vector_results = self._retrieve_vector(query)
        results.extend(vector_results)
        
        # Keyword search (if available)
        if self.keyword_retriever:
            keyword_results = self._retrieve_keyword(query)
            results.extend(keyword_results)
        
        # Merge and re-score
        merged_results = self._merge_results(vector_results, keyword_results if self.keyword_retriever else [])
        
        return merged_results
    
    def _retrieve_vector(self, query: RAGQuery) -> List[RetrievalResult]:
        """Vector similarity search"""
        # Placeholder - implemented by concrete engine
        return []
    
    def _retrieve_keyword(self, query: RAGQuery) -> List[RetrievalResult]:
        """Keyword search using BM25 or similar"""
        # Placeholder - implemented by concrete engine
        return []
    
    def _merge_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Merge results from different retrievers using weighted scoring
        
        Uses Reciprocal Rank Fusion (RRF) for combining rankings
        """
        # Create a dictionary to store combined scores
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.document.id
            # RRF score: 1 / (k + rank) where k=60 is common
            rrf_score = 1.0 / (60 + rank + 1)
            doc_scores[doc_id] = {
                'document': result.document,
                'vector_score': result.score,
                'vector_rrf': rrf_score,
                'keyword_rrf': 0.0,
                'keyword_score': 0.0,
            }
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result.document.id
            rrf_score = 1.0 / (60 + rank + 1)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['keyword_rrf'] = rrf_score
                doc_scores[doc_id]['keyword_score'] = result.score
            else:
                doc_scores[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'vector_rrf': 0.0,
                    'keyword_rrf': rrf_score,
                    'keyword_score': result.score,
                }
        
        # Calculate combined scores
        merged = []
        for doc_id, scores in doc_scores.items():
            combined_rrf = (
                self.vector_weight * scores['vector_rrf'] +
                self.keyword_weight * scores['keyword_rrf']
            )
            
            result = RetrievalResult(
                document=scores['document'],
                score=combined_rrf,
                rerank_score=None,
            )
            merged.append(result)
        
        # Sort by combined score
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged


class ContextualRetriever(BaseRetriever):
    """
    Contextual retriever that considers:
    - User context (role, history, preferences)
    - Temporal context (recent vs historical)
    - Industry context (specific to niche)
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        use_user_context: bool = True,
        use_temporal_context: bool = True,
        use_industry_context: bool = True
    ):
        self.base_retriever = base_retriever
        self.use_user_context = use_user_context
        self.use_temporal_context = use_temporal_context
        self.use_industry_context = use_industry_context
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """
        Retrieve with contextual awareness
        """
        # Get base results
        results = self.base_retriever.retrieve(query)
        
        # Apply contextual reranking
        if self.use_temporal_context:
            results = self._apply_temporal_boost(results, query)
        
        if self.use_industry_context:
            results = self._apply_industry_boost(results, query)
        
        return results
    
    def _apply_temporal_boost(
        self,
        results: List[RetrievalResult],
        query: RAGQuery
    ) -> List[RetrievalResult]:
        """
        Boost recent documents (useful for regulations, updates)
        """
        import time
        from datetime import datetime
        
        current_time = datetime.now().timestamp()
        
        for result in results:
            if result.document.updated_at or result.document.created_at:
                doc_time = (result.document.updated_at or result.document.created_at).timestamp()
                # Calculate age in days
                age_days = (current_time - doc_time) / 86400
                
                # Apply decay: more recent = higher boost
                # Documents within 30 days get full boost
                # Older documents gradually lose boost
                if age_days < 30:
                    temporal_boost = 1.0
                elif age_days < 180:  # 6 months
                    temporal_boost = 0.9
                elif age_days < 365:  # 1 year
                    temporal_boost = 0.8
                else:
                    temporal_boost = 0.7
                
                result.score *= temporal_boost
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _apply_industry_boost(
        self,
        results: List[RetrievalResult],
        query: RAGQuery
    ) -> List[RetrievalResult]:
        """
        Boost documents matching the query's industry
        """
        if not query.industry:
            return results
        
        for result in results:
            if result.document.industry == query.industry:
                result.score *= 1.1  # 10% boost for industry match
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def get_retriever(retriever_type: str, **kwargs) -> BaseRetriever:
    """
    Factory function to get retriever
    
    Args:
        retriever_type: Type of retriever ('hybrid', 'multimodal', 'contextual')
        **kwargs: Configuration for retriever
        
    Returns:
        Configured retriever
    """
    retriever_map = {
        'hybrid': HybridRetriever,
        'multimodal': MultiModalRetriever,
        'contextual': ContextualRetriever,
    }
    
    retriever_class = retriever_map.get(retriever_type, HybridRetriever)
    return retriever_class(**kwargs)

