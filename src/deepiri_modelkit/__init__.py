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
from .data import validate_manifest_against_path, validate_manifest_file
from .training import (
    TrainingJobQueue,
    TRAINING_JOBS_STREAM,
    TrainingRunContext,
    read_manifest,
    write_manifest,
    publish_training_event,
    register_model_ready,
    training_jobs_stream_name,
    build_training_event,
    build_training_run_request,
    emit_training_lifecycle_event,
    create_run_context,
    persist_manifest_for_dataset,
    pipeline_metadata,
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
    "TrainingRunContext",
    "validate_manifest_against_path",
    "validate_manifest_file",
    "build_training_event",
    "build_training_run_request",
    "emit_training_lifecycle_event",
    "create_run_context",
    "persist_manifest_for_dataset",
    "pipeline_metadata",
    "get_logger",
    "get_error_logger",
    "ErrorLogger",
]
