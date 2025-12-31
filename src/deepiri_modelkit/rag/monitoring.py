"""
Monitoring and Observability for Universal RAG
Metrics, performance tracking, and analytics
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json

from .base import RAGQuery, RetrievalResult


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation"""
    query_id: str
    query_text: str
    timestamp: datetime
    retrieval_time_ms: float
    num_results: int
    top_score: Optional[float] = None
    cache_hit: bool = False
    reranking_used: bool = False
    query_expansion_used: bool = False
    industry: Optional[str] = None
    doc_types: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "timestamp": self.timestamp.isoformat(),
            "retrieval_time_ms": self.retrieval_time_ms,
            "num_results": self.num_results,
            "top_score": self.top_score,
            "cache_hit": self.cache_hit,
            "reranking_used": self.reranking_used,
            "query_expansion_used": self.query_expansion_used,
            "industry": self.industry,
            "doc_types": self.doc_types,
        }


@dataclass
class IndexingMetrics:
    """Metrics for indexing operations"""
    operation_id: str
    timestamp: datetime
    operation_type: str  # "index", "update", "delete"
    num_documents: int
    processing_time_ms: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "operation_id": self.operation_id,
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type,
            "num_documents": self.num_documents,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_queries: int = 0
    total_indexed_documents: int = 0
    cache_hit_rate: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_indexing_time_ms: float = 0.0
    error_rate: float = 0.0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_queries": self.total_queries,
            "total_indexed_documents": self.total_indexed_documents,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_indexing_time_ms": self.avg_indexing_time_ms,
            "error_rate": self.error_rate,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class RAGMonitor:
    """
    Monitor RAG system performance and collect metrics
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Metrics storage
        self.retrieval_metrics: List[RetrievalMetrics] = []
        self.indexing_metrics: List[IndexingMetrics] = []
        
        # Aggregated stats
        self.system_metrics = SystemMetrics()
        
        # Time windows for analysis
        self.hourly_stats: Dict[str, Dict] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict] = defaultdict(dict)
    
    def record_retrieval(
        self,
        query: RAGQuery,
        results: List[RetrievalResult],
        retrieval_time_ms: float,
        cache_hit: bool = False,
        reranking_used: bool = False,
        query_expansion_used: bool = False
    ):
        """Record retrieval metrics"""
        query_id = f"q_{int(time.time() * 1000)}"
        
        metric = RetrievalMetrics(
            query_id=query_id,
            query_text=query.query,
            timestamp=datetime.now(),
            retrieval_time_ms=retrieval_time_ms,
            num_results=len(results),
            top_score=results[0].score if results else None,
            cache_hit=cache_hit,
            reranking_used=reranking_used,
            query_expansion_used=query_expansion_used,
            industry=query.industry.value if query.industry and hasattr(query.industry, 'value') else str(query.industry),
            doc_types=[dt.value if hasattr(dt, 'value') else str(dt) for dt in query.doc_types] if query.doc_types else None,
        )
        
        self.retrieval_metrics.append(metric)
        
        # Trim history
        if len(self.retrieval_metrics) > self.max_history:
            self.retrieval_metrics = self.retrieval_metrics[-self.max_history:]
        
        # Update aggregated stats
        self._update_system_metrics()
    
    def record_indexing(
        self,
        operation_type: str,
        num_documents: int,
        processing_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record indexing metrics"""
        operation_id = f"idx_{int(time.time() * 1000)}"
        
        metric = IndexingMetrics(
            operation_id=operation_id,
            timestamp=datetime.now(),
            operation_type=operation_type,
            num_documents=num_documents,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error
        )
        
        self.indexing_metrics.append(metric)
        
        # Trim history
        if len(self.indexing_metrics) > self.max_history:
            self.indexing_metrics = self.indexing_metrics[-self.max_history:]
        
        # Update aggregated stats
        self._update_system_metrics()
    
    def _update_system_metrics(self):
        """Update system-wide aggregated metrics"""
        if not self.retrieval_metrics:
            return
        
        # Calculate averages
        total_queries = len(self.retrieval_metrics)
        cache_hits = sum(1 for m in self.retrieval_metrics if m.cache_hit)
        total_retrieval_time = sum(m.retrieval_time_ms for m in self.retrieval_metrics)
        
        self.system_metrics.total_queries = total_queries
        self.system_metrics.cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0
        self.system_metrics.avg_retrieval_time_ms = total_retrieval_time / total_queries if total_queries > 0 else 0.0
        
        if self.indexing_metrics:
            total_indexed = sum(m.num_documents for m in self.indexing_metrics if m.success)
            total_indexing_time = sum(m.processing_time_ms for m in self.indexing_metrics)
            total_indexing_ops = len(self.indexing_metrics)
            
            self.system_metrics.total_indexed_documents = total_indexed
            self.system_metrics.avg_indexing_time_ms = total_indexing_time / total_indexing_ops if total_indexing_ops > 0 else 0.0
        
        # Error rate
        total_ops = total_queries + len(self.indexing_metrics)
        errors = sum(1 for m in self.indexing_metrics if not m.success)
        self.system_metrics.error_rate = errors / total_ops if total_ops > 0 else 0.0
        
        self.system_metrics.last_updated = datetime.now()
    
    def get_retrieval_stats(
        self,
        time_window_minutes: Optional[int] = None,
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get retrieval statistics for time window"""
        metrics = self.retrieval_metrics
        
        # Filter by time window
        if time_window_minutes:
            cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        # Filter by industry
        if industry:
            metrics = [m for m in metrics if m.industry == industry]
        
        if not metrics:
            return {
                "count": 0,
                "avg_time_ms": 0.0,
                "cache_hit_rate": 0.0,
            }
        
        return {
            "count": len(metrics),
            "avg_time_ms": sum(m.retrieval_time_ms for m in metrics) / len(metrics),
            "min_time_ms": min(m.retrieval_time_ms for m in metrics),
            "max_time_ms": max(m.retrieval_time_ms for m in metrics),
            "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics),
            "avg_results": sum(m.num_results for m in metrics) / len(metrics),
            "avg_top_score": sum(m.top_score for m in metrics if m.top_score) / len([m for m in metrics if m.top_score]),
        }
    
    def get_indexing_stats(
        self,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get indexing statistics"""
        metrics = self.indexing_metrics
        
        # Filter by time window
        if time_window_minutes:
            cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        if not metrics:
            return {
                "count": 0,
                "total_documents": 0,
                "avg_time_ms": 0.0,
                "success_rate": 0.0,
            }
        
        successful = [m for m in metrics if m.success]
        
        return {
            "count": len(metrics),
            "total_documents": sum(m.num_documents for m in successful),
            "avg_time_ms": sum(m.processing_time_ms for m in metrics) / len(metrics),
            "success_rate": len(successful) / len(metrics) if metrics else 0.0,
            "error_count": len([m for m in metrics if not m.success]),
        }
    
    def get_top_queries(
        self,
        limit: int = 10,
        time_window_minutes: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get most frequent queries"""
        metrics = self.retrieval_metrics
        
        # Filter by time window
        if time_window_minutes:
            cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        # Count query frequencies
        query_counts: Dict[str, int] = defaultdict(int)
        for m in metrics:
            query_counts[m.query_text] += 1
        
        # Sort by frequency
        top_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "query": query,
                "count": count
            }
            for query, count in top_queries
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "system_metrics": self.system_metrics.to_dict(),
            "retrieval_stats_1h": self.get_retrieval_stats(time_window_minutes=60),
            "retrieval_stats_24h": self.get_retrieval_stats(time_window_minutes=1440),
            "indexing_stats_1h": self.get_indexing_stats(time_window_minutes=60),
            "indexing_stats_24h": self.get_indexing_stats(time_window_minutes=1440),
            "top_queries_24h": self.get_top_queries(limit=10, time_window_minutes=1440),
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            "retrieval_metrics": [m.to_dict() for m in self.retrieval_metrics[-1000:]],  # Last 1000
            "indexing_metrics": [m.to_dict() for m in self.indexing_metrics[-1000:]],  # Last 1000
            "system_metrics": self.system_metrics.to_dict(),
            "exported_at": datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: Optional[RAGMonitor] = None, operation_name: str = "operation"):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        return False
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        elif self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0.0

