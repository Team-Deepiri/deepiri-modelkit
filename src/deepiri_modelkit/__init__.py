"""
Deepiri ModelKit - Shared contracts, interfaces, and utilities
"""

__version__ = "0.2.0"

from .contracts.models import (
    AIModel,
    AIModelPydantic,
    ModelInput,
    ModelOutput,
    ModelMetadata,
)
from .contracts.events import (
    ModelReadyEvent,
    InferenceEvent,
    PlatformEvent,
    AGIDecisionEvent,
    TrainingEvent,
)
from .contracts.training import (
    AgentTrainingJob,
    DatasetManifest,
    TrainingPriority,
    TrainingRunRequest,
)
from .streaming.event_stream import StreamingClient
from .registry.model_registry import ModelRegistryClient
from .registry.adapters import LocalAdapter, MLflowAdapter, S3Adapter
from .training import (
    TrainingJobQueue,
    TRAINING_JOBS_STREAM,
    read_manifest,
    write_manifest,
    publish_training_event,
    register_model_ready,
    training_jobs_stream_name,
)
from .logging import get_logger, get_error_logger, ErrorLogger

__all__ = [
    "__version__",
    "AIModel",
    "AIModelPydantic",
    "ModelInput",
    "ModelOutput",
    "ModelMetadata",
    "DatasetManifest",
    "TrainingRunRequest",
    "AgentTrainingJob",
    "TrainingPriority",
    "ModelReadyEvent",
    "InferenceEvent",
    "PlatformEvent",
    "AGIDecisionEvent",
    "TrainingEvent",
    "StreamingClient",
    "ModelRegistryClient",
    "MLflowAdapter",
    "S3Adapter",
    "LocalAdapter",
    "TrainingJobQueue",
    "TRAINING_JOBS_STREAM",
    "read_manifest",
    "write_manifest",
    "publish_training_event",
    "register_model_ready",
    "training_jobs_stream_name",
    "get_logger",
    "get_error_logger",
    "ErrorLogger",
]
