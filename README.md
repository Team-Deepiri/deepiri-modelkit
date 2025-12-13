# Deepiri ModelKit

Shared contracts, interfaces, and utilities for Deepiri AI/ML services.

## Purpose

ModelKit provides:
- **Model Contracts**: Interfaces that all models must implement
- **Event Schemas**: Standardized event formats for streaming
- **Streaming Client**: Redis Streams client for event-driven architecture
- **Model Registry Client**: Unified client for MLflow, S3/MinIO, and local storage

## Installation

```bash
pip install -e .
```

## Usage

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
│       ├── streaming/       # Streaming client
│       ├── registry/        # Model registry client
│       └── utils/           # Common utilities
└── pyproject.toml
```

## Related

- `diri-cyrex`: Runtime AI services (uses ModelKit)
- `diri-helox`: ML training pipelines (uses ModelKit)
- `cyrex-agi`: AGI system (uses ModelKit)

