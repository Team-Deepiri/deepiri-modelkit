"""
Advanced Retrieval Strategies for Universal RAG
Query expansion, multi-query retrieval, and advanced search techniques
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json

from .base import Document, RetrievalResult, RAGQuery


@dataclass
class ExpandedQuery:
    """Expanded query with multiple variations"""
    original_query: str
    expanded_queries: List[str]
    query_type: str  # "synonym", "rephrase", "keyword", etc.
    confidence: float


class QueryExpander(ABC):
    """Base class for query expansion strategies"""
    
    @abstractmethod
    def expand(self, query: str, max_expansions: int = 3) -> ExpandedQuery:
        """Expand a query into multiple variations"""
        pass


class SynonymQueryExpander(QueryExpander):
    """Expand queries using synonyms and related terms"""
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        self.synonym_dict = synonym_dict or self._default_synonyms()
    
    def expand(self, query: str, max_expansions: int = 3) -> ExpandedQuery:
        """Expand query using synonyms"""
        words = query.lower().split()
        expanded = [query]  # Always include original
        
        for word in words:
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word][:max_expansions]
                for synonym in synonyms:
                    expanded_query = query.replace(word, synonym)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded[:max_expansions + 1],
            query_type="synonym",
            confidence=0.8
        )
    
    def _default_synonyms(self) -> Dict[str, List[str]]:
        """Default synonym dictionary"""
        return {
            "repair": ["fix", "maintain", "service", "restore"],
            "maintenance": ["service", "upkeep", "repair", "inspection"],
            "claim": ["request", "application", "report", "filing"],
            "policy": ["coverage", "plan", "insurance", "agreement"],
            "regulation": ["rule", "standard", "requirement", "guideline"],
            "procedure": ["process", "method", "protocol", "steps"],
            "equipment": ["machine", "device", "tool", "apparatus"],
            "safety": ["security", "protection", "precaution"],
            "inspection": ["examination", "review", "check", "audit"],
            "documentation": ["record", "file", "document", "paperwork"],
        }


class RephraseQueryExpander(QueryExpander):
    """Expand queries by rephrasing"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def expand(self, query: str, max_expansions: int = 3) -> ExpandedQuery:
        """Rephrase query using LLM or templates"""
        if self.llm_client:
            return self._llm_rephrase(query, max_expansions)
        else:
            return self._template_rephrase(query, max_expansions)
    
    def _template_rephrase(self, query: str, max_expansions: int) -> ExpandedQuery:
        """Rephrase using templates"""
        templates = [
            f"What is {query}?",
            f"How to {query}?",
            f"Information about {query}",
            f"Details on {query}",
        ]
        
        expanded = [query] + templates[:max_expansions]
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded,
            query_type="rephrase",
            confidence=0.7
        )
    
    def _llm_rephrase(self, query: str, max_expansions: int) -> ExpandedQuery:
        """Rephrase using LLM (if available)"""
        # Placeholder for LLM-based rephrasing
        return self._template_rephrase(query, max_expansions)


class KeywordExtractor:
    """Extract keywords from queries for hybrid search"""
    
    def __init__(self):
        # Common stop words
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "must", "can", "this", "that", "these", "those"
        }
    
    def extract(self, query: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from query"""
        words = query.lower().split()
        keywords = [
            word.strip(".,!?;:()[]{}")
            for word in words
            if word.strip(".,!?;:()[]{}") not in self.stop_words
            and len(word.strip(".,!?;:()[]{}")) > 2
        ]
        return keywords[:max_keywords]


class MultiQueryRetriever:
    """
    Multi-query retrieval strategy
    Generates multiple query variations and combines results
    """
    
    def __init__(
        self,
        base_retriever,
        query_expander: Optional[QueryExpander] = None,
        fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "mean"
    ):
        self.base_retriever = base_retriever
        self.query_expander = query_expander or SynonymQueryExpander()
        self.fusion_method = fusion_method
    
    def retrieve(
        self,
        query: RAGQuery,
        num_queries: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve using multiple query variations
        
        Args:
            query: Original RAG query
            num_queries: Number of query variations to generate
            
        Returns:
            Fused retrieval results
        """
        # Expand query
        expanded = self.query_expander.expand(query.query, max_expansions=num_queries - 1)
        
        # Retrieve for each query variation
        all_results: Dict[str, List[RetrievalResult]] = {}
        
        for expanded_query in expanded.expanded_queries:
            # Create new query with expanded text
            expanded_rag_query = RAGQuery(
                query=expanded_query,
                industry=query.industry,
                doc_types=query.doc_types,
                date_range=query.date_range,
                metadata_filters=query.metadata_filters,
                top_k=query.top_k or 10
            )
            
            # Retrieve results
            results = self.base_retriever.retrieve(expanded_rag_query)
            all_results[expanded_query] = results
        
        # Fuse results
        fused = self._fuse_results(all_results, query.top_k or 10)
        
        return fused
    
    def _fuse_results(
        self,
        all_results: Dict[str, List[RetrievalResult]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Fuse results from multiple queries"""
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(all_results, top_k)
        else:
            return self._mean_score_fusion(all_results, top_k)
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict[str, List[RetrievalResult]],
        top_k: int,
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF)
        Combines rankings from multiple queries
        """
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for query_text, results in all_results.items():
            for rank, result in enumerate(results):
                doc_id = result.document.id
                rrf_score = 1.0 / (k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "document": result.document,
                        "rrf_score": 0.0,
                        "max_score": result.score,
                        "count": 0
                    }
                
                doc_scores[doc_id]["rrf_score"] += rrf_score
                doc_scores[doc_id]["max_score"] = max(
                    doc_scores[doc_id]["max_score"],
                    result.score
                )
                doc_scores[doc_id]["count"] += 1
        
        # Create fused results
        fused = []
        for doc_id, scores in doc_scores.items():
            fused.append(RetrievalResult(
                document=scores["document"],
                score=scores["rrf_score"],  # Use RRF score
                rerank_score=scores["max_score"]  # Store max original score
            ))
        
        # Sort by RRF score
        fused.sort(key=lambda x: x.score, reverse=True)
        
        return fused[:top_k]
    
    def _mean_score_fusion(
        self,
        all_results: Dict[str, List[RetrievalResult]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Fuse results using mean score"""
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for query_text, results in all_results.items():
            for result in results:
                doc_id = result.document.id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "document": result.document,
                        "scores": [],
                        "count": 0
                    }
                
                doc_scores[doc_id]["scores"].append(result.score)
                doc_scores[doc_id]["count"] += 1
        
        # Calculate mean scores
        fused = []
        for doc_id, scores in doc_scores.items():
            mean_score = sum(scores["scores"]) / len(scores["scores"])
            fused.append(RetrievalResult(
                document=scores["document"],
                score=mean_score,
                rerank_score=max(scores["scores"])
            ))
        
        # Sort by mean score
        fused.sort(key=lambda x: x.score, reverse=True)
        
        return fused[:top_k]


class QueryCache:
    """Cache for query results"""
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.cache_ttl = 3600  # 1 hour
    
    def get_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        query_dict = query.to_dict()
        query_str = json.dumps(query_dict, sort_keys=True)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        return f"rag:query:{query_hash}"
    
    def get(self, query: RAGQuery) -> Optional[List[RetrievalResult]]:
        """Get cached results"""
        if not self.cache_manager:
            return None
        
        cache_key = self.get_cache_key(query)
        cached = self.cache_manager.get(cache_key)
        
        if cached:
            # Reconstruct RetrievalResult objects
            results = []
            for item in cached:
                doc = Document.from_dict(item["document"])
                result = RetrievalResult(
                    document=doc,
                    score=item["score"],
                    rerank_score=item.get("rerank_score")
                )
                results.append(result)
            return results
        
        return None
    
    def set(self, query: RAGQuery, results: List[RetrievalResult]):
        """Cache results"""
        if not self.cache_manager:
            return
        
        cache_key = self.get_cache_key(query)
        # Serialize results
        serialized = [
            {
                "document": r.document.to_dict(),
                "score": r.score,
                "rerank_score": r.rerank_score
            }
            for r in results
        ]
        
        self.cache_manager.set(cache_key, serialized, ttl=self.cache_ttl)


class AdvancedRetrievalPipeline:
    """
    Advanced retrieval pipeline with:
    - Query expansion
    - Multi-query retrieval
    - Result caching
    - Hybrid search
    """
    
    def __init__(
        self,
        base_retriever,
        query_expander: Optional[QueryExpander] = None,
        use_multi_query: bool = True,
        use_cache: bool = True,
        cache_manager=None
    ):
        self.base_retriever = base_retriever
        self.query_expander = query_expander or SynonymQueryExpander()
        self.use_multi_query = use_multi_query
        self.use_cache = use_cache
        self.query_cache = QueryCache(cache_manager) if use_cache else None
        
        if use_multi_query:
            self.multi_query_retriever = MultiQueryRetriever(
                base_retriever=base_retriever,
                query_expander=query_expander
            )
        else:
            self.multi_query_retriever = None
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """Retrieve with advanced strategies"""
        # Check cache
        if self.use_cache and self.query_cache:
            cached = self.query_cache.get(query)
            if cached:
                return cached
        
        # Retrieve using multi-query or base retriever
        if self.use_multi_query and self.multi_query_retriever:
            results = self.multi_query_retriever.retrieve(query)
        else:
            results = self.base_retriever.retrieve(query)
        
        # Cache results
        if self.use_cache and self.query_cache:
            self.query_cache.set(query, results)
        
        return results

