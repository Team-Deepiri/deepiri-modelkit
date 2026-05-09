"""
Testing Utilities for Universal RAG
Comprehensive test helpers, fixtures, and evaluation tools
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from .base import Document, DocumentType, IndustryNiche, RAGQuery, RetrievalResult


@dataclass
class TestCase:
    """Test case for RAG evaluation"""
    query: str
    expected_doc_ids: List[str]  # Document IDs that should be retrieved
    expected_doc_types: Optional[List[DocumentType]] = None
    min_score: float = 0.7  # Minimum similarity score
    top_k: int = 5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestResult:
    """Result of a test case"""
    test_case: TestCase
    retrieved_doc_ids: List[str]
    retrieved_scores: List[float]
    precision: float
    recall: float
    f1_score: float
    passed: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.test_case.query,
            "expected_doc_ids": self.test_case.expected_doc_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "passed": self.passed,
            "error": self.error,
        }


class RAGEvaluator:
    """
    Evaluator for RAG system performance
    """
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
    
    def evaluate(
        self,
        test_cases: List[TestCase],
        industry: Optional[IndustryNiche] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on test cases
        
        Args:
            test_cases: List of test cases
            industry: Industry context
            
        Returns:
            Evaluation results with metrics
        """
        results = []
        
        for test_case in test_cases:
            result = self._evaluate_test_case(test_case, industry)
            results.append(result)
        
        # Calculate aggregate metrics
        total_precision = sum(r.precision for r in results) / len(results) if results else 0.0
        total_recall = sum(r.recall for r in results) / len(results) if results else 0.0
        total_f1 = sum(r.f1_score for r in results) / len(results) if results else 0.0
        passed_count = sum(1 for r in results if r.passed)
        
        return {
            "total_tests": len(test_cases),
            "passed": passed_count,
            "failed": len(test_cases) - passed_count,
            "avg_precision": total_precision,
            "avg_recall": total_recall,
            "avg_f1_score": total_f1,
            "pass_rate": passed_count / len(test_cases) if test_cases else 0.0,
            "results": [r.to_dict() for r in results],
        }
    
    def _evaluate_test_case(
        self,
        test_case: TestCase,
        industry: Optional[IndustryNiche]
    ) -> TestResult:
        """Evaluate a single test case"""
        try:
            # Build query
            query = RAGQuery(
                query=test_case.query,
                industry=industry,
                doc_types=test_case.expected_doc_types,
                top_k=test_case.top_k
            )
            
            # Retrieve documents
            results = self.rag_engine.retrieve(query)
            
            # Extract IDs and scores
            retrieved_doc_ids = [r.document.id for r in results]
            retrieved_scores = [r.score for r in results]
            
            # Calculate precision, recall, F1
            expected_set = set(test_case.expected_doc_ids)
            retrieved_set = set(retrieved_doc_ids)
            
            if not retrieved_set:
                precision = 0.0
                recall = 0.0
                f1_score = 0.0
            else:
                # Precision: relevant retrieved / total retrieved
                relevant_retrieved = len(expected_set & retrieved_set)
                precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0.0
                
                # Recall: relevant retrieved / total relevant
                recall = relevant_retrieved / len(expected_set) if expected_set else 0.0
                
                # F1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Check if passed
            passed = (
                precision >= 0.7 and
                recall >= 0.7 and
                f1_score >= 0.7 and
                (not retrieved_scores or max(retrieved_scores) >= test_case.min_score)
            )
            
            return TestResult(
                test_case=test_case,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_scores=retrieved_scores,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                passed=passed
            )
        
        except Exception as e:
            return TestResult(
                test_case=test_case,
                retrieved_doc_ids=[],
                retrieved_scores=[],
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                passed=False,
                error=str(e)
            )


class RAGTestFixture:
    """
    Test fixture for creating test data and scenarios
    """
    
    @staticmethod
    def create_test_documents(
        industry: IndustryNiche = IndustryNiche.MANUFACTURING,
        num_documents: int = 10
    ) -> List[Document]:
        """Create test documents"""
        documents = []
        
        for i in range(num_documents):
            doc = Document(
                id=f"test_doc_{i}",
                content=f"Test document {i} content. This is sample content for testing RAG retrieval.",
                doc_type=DocumentType.MANUAL if i % 2 == 0 else DocumentType.MAINTENANCE_LOG,
                industry=industry,
                title=f"Test Document {i}",
                source="test_fixture",
                metadata={"test_index": i}
            )
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def create_test_cases(
        num_cases: int = 5
    ) -> List[TestCase]:
        """Create test cases"""
        test_cases = []
        
        for i in range(num_cases):
            test_case = TestCase(
                query=f"test query {i}",
                expected_doc_ids=[f"test_doc_{i}", f"test_doc_{i+1}"],
                top_k=5
            )
            test_cases.append(test_case)
        
        return test_cases


class PerformanceBenchmark:
    """
    Performance benchmarking for RAG operations
    """
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
    
    def benchmark_retrieval(
        self,
        queries: List[str],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval performance
        
        Args:
            queries: List of test queries
            iterations: Number of iterations per query
            
        Returns:
            Performance metrics
        """
        import time
        
        total_time = 0.0
        total_queries = 0
        times = []
        
        for query_text in queries:
            query = RAGQuery(query=query_text, top_k=5)
            
            for _ in range(iterations):
                start = time.time()
                results = self.rag_engine.retrieve(query)
                elapsed = time.time() - start
                
                total_time += elapsed
                total_queries += 1
                times.append(elapsed * 1000)  # Convert to ms
        
        avg_time_ms = (total_time / total_queries) * 1000 if total_queries > 0 else 0.0
        min_time_ms = min(times) if times else 0.0
        max_time_ms = max(times) if times else 0.0
        
        return {
            "total_queries": total_queries,
            "avg_time_ms": avg_time_ms,
            "min_time_ms": min_time_ms,
            "max_time_ms": max_time_ms,
            "queries_per_second": total_queries / total_time if total_time > 0 else 0.0,
        }
    
    def benchmark_indexing(
        self,
        documents: List[Document],
        batch_sizes: List[int] = [1, 10, 100]
    ) -> Dict[str, Any]:
        """
        Benchmark indexing performance
        
        Args:
            documents: Documents to index
            batch_sizes: Different batch sizes to test
            
        Returns:
            Performance metrics for each batch size
        """
        import time
        
        results = {}
        
        for batch_size in batch_sizes:
            batches = [
                documents[i:i + batch_size]
                for i in range(0, len(documents), batch_size)
            ]
            
            start = time.time()
            for batch in batches:
                self.rag_engine.index_documents(batch)
            elapsed = time.time() - start
            
            results[f"batch_size_{batch_size}"] = {
                "total_documents": len(documents),
                "num_batches": len(batches),
                "total_time_seconds": elapsed,
                "avg_time_per_doc_ms": (elapsed / len(documents)) * 1000 if documents else 0.0,
                "docs_per_second": len(documents) / elapsed if elapsed > 0 else 0.0,
            }
        
        return results


def create_evaluation_dataset(
    industry: IndustryNiche,
    num_documents: int = 100,
    num_queries: int = 20
) -> Tuple[List[Document], List[TestCase]]:
    """
    Create a complete evaluation dataset
    
    Args:
        industry: Industry for documents
        num_documents: Number of documents to create
        num_queries: Number of test queries
        
    Returns:
        Tuple of (documents, test_cases)
    """
    documents = RAGTestFixture.create_test_documents(industry, num_documents)
    test_cases = RAGTestFixture.create_test_cases(num_queries)
    
    return documents, test_cases

