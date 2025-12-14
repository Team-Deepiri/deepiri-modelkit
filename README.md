# Deepiri ModelKit

**Shared contracts, interfaces, and utilities for Deepiri AI/ML services.**

## What is ModelKit?

ModelKit is the **shared library** that connects all Deepiri ML services. It provides standardized interfaces, event schemas, and utilities that ensure seamless communication between:

- **Helox** (ML Training) - Trains models and publishes them
- **Cyrex** (Runtime) - Loads models and serves inference
- **Platform Services** - Consume model events and predictions

Think of ModelKit as the **"contract layer"** - it defines how models should behave, how events should be structured, and how services should communicate. This ensures that when Helox trains a model, Cyrex can automatically discover, download, and use it without manual intervention.

### Core Purpose

1. **Model Contracts** - Standard `AIModel` interface that all models must implement
2. **Event-Driven Architecture** - Redis Streams integration for model lifecycle events
3. **Model Registry** - Unified MLflow + S3/MinIO client for model storage
4. **Shared Logging** - Structured JSON logging used across all services
5. **Type Safety** - Pydantic models for all contracts and events

## Purpose

ModelKit provides:
- **Model Contracts**: Interfaces that all models must implement
- **Event Schemas**: Standardized event formats for streaming
- **Streaming Client**: Redis Streams client for event-driven architecture
- **Model Registry Client**: Unified client for MLflow, S3/MinIO, and local storage
- **Shared Logging**: Structured JSON logging for all services

## Installation

```bash
cd deepiri-modelkit
pip install -e .
```

## Usage

### Shared Logging

```python
from deepiri_modelkit import get_logger, get_error_logger

# Service logger
logger = get_logger("my_service")
logger.info("service_started", port=8000, version="1.0")
logger.error("connection_failed", host="redis", reason="timeout")

# Error logger with context
error_logger = get_error_logger()
error_logger.log_api_error(e, request_id="123", endpoint="/predict")
error_logger.log_model_error(e, model_name="classifier", input_data={...})
error_logger.log_training_error(e, pipeline="qlora", config={...})
```

### Model Contracts

```python
from deepiri_modelkit import AIModel, ModelInput, ModelOutput

class MyModel:
    def predict(self, input: ModelInput) -> ModelOutput:
        # Implement prediction
        return ModelOutput(prediction=result)
    
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(name="my-model", version="v1.0")
```

### Streaming Client

```python
from deepiri_modelkit import StreamingClient, ModelReadyEvent

# Publish event
client = StreamingClient()
await client.publish("model-events", {
    "event": "model-ready",
    "model_name": "task-classifier",
    "version": "v1.0",
    "registry_path": "s3://models/task-classifier/v1.0"
})

# Subscribe to events
async for event in client.subscribe("model-events"):
    print(f"Received: {event}")
```

### Model Registry

```python
from deepiri_modelkit import ModelRegistryClient

# Initialize
registry = ModelRegistryClient(
    registry_type="mlflow",
    mlflow_tracking_uri="http://mlflow:5000"
)

# Register model
registry.register_model(
    model_name="task-classifier",
    version="v1.0",
    model_path="./model.pkl",
    metadata={"accuracy": 0.95}
)

# Get model
model_info = registry.get_model("task-classifier", "v1.0")
```

## Structure

```
deepiri-modelkit/
├── src/
│   └── deepiri_modelkit/
│       ├── contracts/      # Model and service contracts
│       ├── streaming/      # Streaming client
│       ├── registry/       # Model registry client
│       ├── logging.py      # Shared logging utilities
│       └── utils/          # Common utilities
└── pyproject.toml
```

## Related

- `diri-cyrex`: Runtime AI services (uses ModelKit)
- `diri-helox`: ML training pipelines (uses ModelKit)
- `cyrex-agi`: AGI system (uses ModelKit)

