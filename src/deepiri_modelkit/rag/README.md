# Universal RAG Module

**Reusable Retrieval-Augmented Generation system for all industry niches**

## Overview

The Universal RAG module provides a common foundation for document indexing, retrieval, and generation across all industries supported by the Deepiri platform.

## Architecture

### Components

```
deepiri_modelkit/rag/
├── __init__.py           # Public API exports
├── base.py               # Base classes and abstractions
├── processors.py         # Document processors
├── retrievers.py         # Retrieval strategies
└── README.md            # This file
```

### Design Principles

1. **Industry Agnostic** - Works across insurance, manufacturing, healthcare, etc.
2. **Document Type Polymorphism** - Supports regulations, manuals, logs, FAQs, etc.
3. **Extensible** - Easy to add new processors and retrievers
4. **Production Ready** - Battle-tested with Milvus vector store

## Quick Start

```python
from deepiri_modelkit.rag import (
    Document,
    DocumentType,
    IndustryNiche,
    RAGQuery,
    UniversalRAGEngine,
)

# Subclass and implement for your vector store
class MyRAGEngine(UniversalRAGEngine):
    def _initialize(self):
        # Initialize your vector store
        pass
    
    def index_document(self, document: Document) -> bool:
        # Index to your vector store
        pass
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        # Retrieve from your vector store
        pass
    
    # ... implement other methods
```

## Core Classes

### Document

Represents a document to be indexed:

```python
document = Document(
    id="doc_123",
    content="Document content...",
    doc_type=DocumentType.REGULATION,
    industry=IndustryNiche.MANUFACTURING,
    title="OSHA Safety Standards",
    source="osha.gov",
    metadata={"regulation_number": "29 CFR 1910"}
)
```

### RAGQuery

Query with filters and parameters:

```python
query = RAGQuery(
    query="What are the fire safety requirements?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.REGULATION, DocumentType.SAFETY_GUIDELINE],
    top_k=5,
    metadata_filters={"topic": "fire_safety"}
)
```

### DocumentProcessor

Process raw content into structured documents:

```python
from deepiri_modelkit.rag.processors import get_processor

processor = get_processor(
    doc_type=DocumentType.REGULATION,
    chunk_size=1000,
    chunk_overlap=200
)

chunks = processor.process(
    raw_content=long_regulation_text,
    metadata={"title": "OSHA 1910"}
)
```

## Supported Industries

- `INSURANCE` - Claims, policies, regulations
- `MANUFACTURING` - Equipment, maintenance, safety
- `PROPERTY_MANAGEMENT` - Maintenance, codes, inspections
- `HEALTHCARE` - Protocols, regulations, equipment
- `CONSTRUCTION` - Specs, safety, inspections
- `AUTOMOTIVE` - Service records, manuals
- `ENERGY` - Equipment, regulations, procedures
- `LOGISTICS` - Operations, regulations
- `RETAIL` - Operations, policies
- `HOSPITALITY` - Procedures, standards
- `GENERIC` - Cross-industry

## Supported Document Types

- `REGULATION` - Laws and regulations
- `POLICY` - Insurance/company policies
- `MANUAL` - Equipment manuals
- `CONTRACT` - Legal contracts
- `WORK_ORDER` - Service requests
- `CLAIM_RECORD` - Insurance/warranty claims
- `MAINTENANCE_LOG` - Equipment history
- `FAQ` - Frequently asked questions
- `KNOWLEDGE_BASE` - Knowledge articles
- `REPORT` - Inspection/audit reports
- `PROCEDURE` - Standard procedures
- `SAFETY_GUIDELINE` - Safety protocols
- `TECHNICAL_SPEC` - Technical specifications
- `INVOICE` - Billing documents
- `OTHER` - Catch-all

## Document Processors

### RegulationProcessor

Handles regulations and compliance documents:
- Extracts sections and subsections
- Preserves document structure
- Adds section metadata

### HistoricalDataProcessor

Handles work orders, logs, claims:
- Minimal chunking (data already structured)
- Preserves record metadata
- Supports status/priority fields

### KnowledgeBaseProcessor

Handles FAQs and knowledge articles:
- Extracts Q&A pairs
- Chunks articles intelligently
- Adds category metadata

### ManualProcessor

Handles equipment manuals and guides:
- Extracts chapters and sections
- Preserves version information
- Adds equipment metadata

## Retrieval Strategies

### HybridRetriever

Combines semantic + keyword search:
- Vector similarity (embeddings)
- Keyword search (BM25)
- Reciprocal Rank Fusion for merging

### MultiModalRetriever

Supports multiple content types:
- Text documents
- Images (future)
- Tables (future)
- Code (future)

### ContextualRetriever

Adds context awareness:
- Temporal boosting (recent documents)
- Industry boosting (matching industry)
- User context (future)

## Implementation Example

See `diri-cyrex/app/integrations/universal_rag_engine.py` for a complete production implementation using Milvus.

## Documentation

- **User Guide**: `docs/UNIVERSAL_RAG_GUIDE.md`
- **Examples**: `docs/UNIVERSAL_RAG_EXAMPLES.md`
- **API Reference**: `http://localhost:8000/docs` (when Cyrex is running)

## Testing

```bash
# Run example
cd diri-cyrex
python examples/universal_rag_example.py

# Test API
curl http://localhost:8000/api/v1/universal-rag/health
```

## License

See LICENSE.md in project root.

