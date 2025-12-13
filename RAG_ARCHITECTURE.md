# RAG System Architecture

## Overview

The RAG (Retrieval-Augmented Generation) system is split between **production runtime** (in Cyrex) and **training pipelines** (in Helox).

---

## Production RAG System (Cyrex Runtime)

**Location:** `deepiri/diri-cyrex/app/`

### Core RAG Files

1. **`app/integrations/rag_pipeline.py`** ✅ PRIMARY RAG SYSTEM
   - Production RAG pipeline for challenge generation
   - Vector search, reranking, and context management
   - Used by API routes for runtime queries
   - Classes:
     - `RAGPipeline` - Main production RAG system
     - `RAGDataPipeline` - Data indexing
     - `initialize_rag_system()` - Factory function
   
2. **`app/routes/rag.py`** ✅ RAG API ENDPOINTS
   - REST API endpoints for RAG operations
   - Endpoints:
     - `POST /rag/query` - Query for relevant challenges
     - `POST /rag/index` - Index challenges into vector DB
     - `POST /rag/generate` - Generate with RAG context

3. **`app/integrations/rag_bridge.py`** ✅ LANGCHAIN BRIDGE
   - Bridges `KnowledgeRetrievalEngine` with new LangChain orchestration
   - Unified interface for RAG operations
   - Seamless integration between old and new systems

4. **`app/services/enhanced_rag_service.py`** ✅ ENHANCED RAG
   - Pinecone/Weaviate integration
   - Cross-modal retrieval (text, images, code)
   - Semantic search with reranking

5. **`app/integrations/milvus_store.py`** ✅ VECTOR DATABASE
   - Milvus vector store integration
   - LangChain-compatible vector operations
   - Used by RAG pipeline for embeddings

6. **`app/services/knowledge_retrieval_engine.py`** ✅ KNOWLEDGE ENGINE
   - Legacy knowledge retrieval system
   - Bridged with new RAG via `rag_bridge.py`

### Tests

- **`tests/ai/test_rag.py`** - RAG system tests

---

## ML Training Pipelines (Helox)

**Location:** `deepiri/diri-helox/pipelines/training/`

### Training Files

1. **`pipelines/training/rag_training_pipeline.py`** ✅ RAG TRAINING
   - Trains embedding models for RAG
   - Trains rerankers (cross-encoders)
   - Evaluates RAG system performance
   - Class: `RAGTrainingPipeline`
   - Methods:
     - `train_embedding_model()` - Fine-tune embeddings
     - `train_reranker()` - Train cross-encoder
     - `evaluate_rag_system()` - Measure accuracy
   - **Note:** Imports production `RAGPipeline` from `diri-cyrex/app/integrations/rag_pipeline.py` for evaluation

---

## Key Distinctions

| Aspect | Cyrex (Production) | Helox (Training) |
|--------|-------------------|------------------|
| **Purpose** | Runtime RAG queries | Train RAG models |
| **Location** | `diri-cyrex/app/` | `diri-helox/pipelines/training/` |
| **Used by** | FastAPI routes, services | Training scripts |
| **Classes** | `RAGPipeline`, `RAGBridge` | `RAGTrainingPipeline` |
| **Dependencies** | Milvus, LangChain, Sentence-Transformers | PyTorch, Transformers, training libs |
| **When runs** | Every API request | During training jobs |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Helox)                          │
│                                                              │
│  1. rag_training_pipeline.py                                │
│     └─> train_embedding_model()                             │
│     └─> train_reranker()                                    │
│     └─> Fine-tuned models saved to disk                     │
│                                                              │
│                          │                                   │
│                          ▼                                   │
│                  Trained Model Files                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           │ Load models
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION (Cyrex)                          │
│                                                              │
│  1. User sends query                                        │
│     └─> POST /rag/query                                     │
│         └─> routes/rag.py                                   │
│             └─> integrations/rag_pipeline.py                │
│                 └─> RAGPipeline.retrieve()                  │
│                     └─> Milvus vector search                │
│                     └─> Cross-encoder reranking             │
│                     └─> Return relevant challenges          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

### Cyrex Runtime

```bash
# Milvus connection
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Model paths (trained models from Helox)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Helox Training

```bash
# Training config
TRAINING_EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=2e-5

# Output paths
MODEL_OUTPUT_PATH=/models/rag/
```

---

## What Was Fixed

### Before (Incorrect)

```
diri-cyrex/app/train/infrastructure/rag_pipeline.py    ❌ DUPLICATE
diri-cyrex/app/train/pipelines/rag_training_pipeline.py ❌ WRONG LOCATION
diri-cyrex/app/integrations/rag_pipeline.py             ✅ CORRECT
```

### After (Correct)

```
diri-cyrex/app/integrations/rag_pipeline.py             ✅ PRODUCTION RAG
diri-helox/pipelines/training/rag_training_pipeline.py  ✅ TRAINING PIPELINE
```

**Deleted:**
- `diri-cyrex/app/train/infrastructure/rag_pipeline.py` (duplicate)
- `diri-cyrex/app/train/pipelines/rag_training_pipeline.py` (moved to Helox)

---

## Summary

✅ **RAG production system is in Cyrex** (`app/integrations/rag_pipeline.py`)  
✅ **RAG training pipeline is in Helox** (`pipelines/training/rag_training_pipeline.py`)  
✅ **All duplicates removed**  
✅ **Runtime imports from `app/integrations/`, not `app/train/`**  

The RAG system is correctly separated: **runtime in Cyrex, training in Helox**.

