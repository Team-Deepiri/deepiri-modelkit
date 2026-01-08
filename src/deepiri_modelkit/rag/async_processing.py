"""
Async Batch Processing for Universal RAG
High-performance async document processing and indexing
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
# Fix for Python < 3.9 compatibility
try:
    from collections.abc import AsyncIterator
except ImportError:
    from typing import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import time

from .base import Document, DocumentType, IndustryNiche


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    batch_size: int = 100
    max_concurrent_batches: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_progress: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation"""
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    processing_time_seconds: float
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "processing_time_seconds": self.processing_time_seconds,
            "success_rate": self.success_rate,
            "errors": self.errors,
        }


class AsyncBatchProcessor:
    """
    Async batch processor for high-performance document processing
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_batches)
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchProcessingResult:
        """
        Process items in batches asynchronously
        
        Args:
            items: List of items to process
            processor_func: Async function to process each item
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchProcessingResult with statistics
        """
        start_time = time.time()
        total_items = len(items)
        successful_items = 0
        failed_items = 0
        errors = []
        
        # Split into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, total_items, self.config.batch_size)
        ]
        
        # Process batches concurrently
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._process_batch_with_semaphore(
                batch,
                batch_idx,
                processor_func,
                progress_callback
            )
            tasks.append(task)
        
        # Wait for all batches
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in batch_results:
            if isinstance(result, Exception):
                failed_items += self.config.batch_size
                errors.append({
                    "error": str(result),
                    "type": type(result).__name__
                })
            else:
                successful_items += result["successful"]
                failed_items += result["failed"]
                errors.extend(result.get("errors", []))
        
        processing_time = time.time() - start_time
        
        return BatchProcessingResult(
            total_items=total_items,
            processed_items=total_items,
            successful_items=successful_items,
            failed_items=failed_items,
            processing_time_seconds=processing_time,
            errors=errors
        )
    
    async def _process_batch_with_semaphore(
        self,
        batch: List[Any],
        batch_idx: int,
        processor_func: Callable[[Any], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> Dict[str, Any]:
        """Process a single batch with semaphore control"""
        async with self.semaphore:
            return await self._process_single_batch(
                batch,
                batch_idx,
                processor_func,
                progress_callback
            )
    
    async def _process_single_batch(
        self,
        batch: List[Any],
        batch_idx: int,
        processor_func: Callable[[Any], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> Dict[str, Any]:
        """Process a single batch"""
        successful = 0
        failed = 0
        errors = []
        
        # Process items in batch concurrently
        tasks = [processor_func(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                errors.append({
                    "item_index": batch_idx * self.config.batch_size + idx,
                    "error": str(result),
                    "type": type(result).__name__
                })
            else:
                successful += 1
        
        # Progress callback
        if progress_callback:
            total_processed = (batch_idx + 1) * self.config.batch_size
            progress_callback(total_processed, len(batch) * (batch_idx + 1))
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors
        }


class AsyncDocumentIndexer:
    """
    Async document indexer with batching and retry logic
    """
    
    def __init__(
        self,
        index_func: Callable[[Document], Awaitable[bool]],
        config: Optional[BatchProcessingConfig] = None
    ):
        self.index_func = index_func
        self.config = config or BatchProcessingConfig()
        self.batch_processor = AsyncBatchProcessor(self.config)
    
    async def index_documents(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchProcessingResult:
        """
        Index documents asynchronously in batches
        
        Args:
            documents: List of documents to index
            progress_callback: Optional callback for progress
            
        Returns:
            BatchProcessingResult with statistics
        """
        async def index_document(doc: Document) -> bool:
            """Index a single document with retry"""
            for attempt in range(self.config.max_retries):
                try:
                    result = await self.index_func(doc)
                    if result:
                        return True
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(
                            self.config.retry_delay_seconds * (attempt + 1)
                        )
                        continue
                    raise
            
            return False
        
        return await self.batch_processor.process_batch(
            documents,
            index_document,
            progress_callback
        )
    
    async def index_documents_streaming(
        self,
        document_stream: AsyncIterator[Document],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchProcessingResult:
        """
        Index documents from async stream
        
        Args:
            document_stream: Async iterator of documents
            progress_callback: Optional callback for progress
            
        Returns:
            BatchProcessingResult with statistics
        """
        batch = []
        total_processed = 0
        successful = 0
        failed = 0
        errors = []
        start_time = time.time()
        
        async def process_current_batch():
            nonlocal successful, failed, errors
            
            if not batch:
                return
            
            result = await self.index_documents(batch, progress_callback)
            successful += result.successful_items
            failed += result.failed_items
            errors.extend(result.errors)
            batch.clear()
        
        async for document in document_stream:
            batch.append(document)
            total_processed += 1
            
            if len(batch) >= self.config.batch_size:
                await process_current_batch()
        
        # Process remaining batch
        if batch:
            await process_current_batch()
        
        processing_time = time.time() - start_time
        
        return BatchProcessingResult(
            total_items=total_processed,
            processed_items=total_processed,
            successful_items=successful,
            failed_items=failed,
            processing_time_seconds=processing_time,
            errors=errors
        )


class AsyncDocumentProcessor:
    """
    Async document processor for parallel document processing
    """
    
    def __init__(
        self,
        processor_func: Callable[[str, Dict], List[Document]],
        config: Optional[BatchProcessingConfig] = None
    ):
        self.processor_func = processor_func
        self.config = config or BatchProcessingConfig()
        self.batch_processor = AsyncBatchProcessor(self.config)
    
    async def process_documents(
        self,
        raw_documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> tuple[List[Document], BatchProcessingResult]:
        """
        Process raw documents asynchronously
        
        Args:
            raw_documents: List of dicts with 'content' and 'metadata'
            progress_callback: Optional callback for progress
            
        Returns:
            Tuple of (processed_documents, processing_result)
        """
        processed_docs = []
        errors = []
        
        async def process_document(item: Dict[str, Any]) -> List[Document]:
            """Process a single document"""
            try:
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                docs = await asyncio.to_thread(
                    self.processor_func,
                    content,
                    metadata
                )
                return docs
            except Exception as e:
                errors.append({
                    "item": item.get("id", "unknown"),
                    "error": str(e)
                })
                return []
        
        result = await self.batch_processor.process_batch(
            raw_documents,
            process_document,
            progress_callback
        )
        
        # Collect all processed documents
        tasks = [process_document(item) for item in raw_documents]
        doc_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        for doc_list in doc_lists:
            if isinstance(doc_list, list):
                processed_docs.extend(doc_list)
        
        return processed_docs, result

