"""
Stream topic definitions.
"""

from enum import Enum


class StreamTopics(str, Enum):
    """Redis Stream topics."""

    MODEL_EVENTS = "model-events"
    INFERENCE_EVENTS = "inference-events"
    PLATFORM_EVENTS = "platform-events"
    AGI_DECISIONS = "agi-decisions"
    TRAINING_EVENTS = "training-events"

    # LIS document routing streams (document.* namespace).
    DOCUMENT_VECTORIZE = "document.vectorize"
    DOCUMENT_TRAINING = "document.training"
    DOCUMENT_STRUCTURED = "document.structured"
    DOCUMENT_ARTIFACTS = "document.artifacts"

    # Cyrex runtime training signals for Helox (pipeline.* namespace).
    HELOX_TRAINING_RAW = "pipeline.helox-training.raw"
    HELOX_TRAINING_STRUCTURED = "pipeline.helox-training.structured"

    @classmethod
    def all(cls) -> list[str]:
        """Get all topic names."""
        return [topic.value for topic in cls]
