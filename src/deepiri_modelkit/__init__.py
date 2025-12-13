"""
Deepiri ModelKit - Shared contracts, interfaces, and utilities
"""

__version__ = "0.1.0"

from .contracts.models import AIModel, AIModelPydantic, ModelInput, ModelOutput, ModelMetadata
from .contracts.events import (
    ModelReadyEvent,
    InferenceEvent,
    PlatformEvent,
    AGIDecisionEvent,
    TrainingEvent,
)
from .streaming.event_stream import StreamingClient
from .registry.model_registry import ModelRegistryClient

__all__ = [
    "AIModel",  # Protocol interface for type checking
    "AIModelPydantic",  # Pydantic-compatible type for use in BaseModel fields
    "ModelInput",
    "ModelOutput",
    "ModelMetadata",
    "ModelReadyEvent",
    "InferenceEvent",
    "PlatformEvent",
    "AGIDecisionEvent",
    "TrainingEvent",
    "StreamingClient",
    "ModelRegistryClient",
]

