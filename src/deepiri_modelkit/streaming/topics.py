"""
Stream topic definitions
"""

from enum import Enum


class StreamTopics(str, Enum):
    """Redis Stream topics"""

    MODEL_EVENTS = "model-events"
    INFERENCE_EVENTS = "inference-events"
    PLATFORM_EVENTS = "platform-events"
    AGI_DECISIONS = "agi-decisions"
    TRAINING_EVENTS = "training-events"
    TRAINING_JOBS = "training-jobs"
    # LIS document routing streams (document.* namespace).
    DOCUMENT_VECTORIZE = "document.vectorize"
    DOCUMENT_TRAINING = "document.training"
    DOCUMENT_STRUCTURED = "document.structured"
    DOCUMENT_ARTIFACTS = "document.artifacts"
    # Cyrex runtime training signals consumed by Helox.
    HELOX_TRAINING_RAW = "pipeline.helox-training.raw"
    HELOX_TRAINING_STRUCTURED = "pipeline.helox-training.structured"
    # Cyrex AGI pipeline bus (artifact engine → observers / Canvas / Helox).
    PIPELINE_PRESSURE_EVENTS = "pipeline.pressure.events"
    PIPELINE_ARTIFACT_INVALIDATION = "pipeline.artifact.invalidation"
    PIPELINE_SPLICE_EVENTS = "pipeline.splice.events"
    PIPELINE_DEAD_LETTER = "pipeline.dead-letter"
    PIPELINE_METRICS = "pipeline.metrics"

    @classmethod
    def all(cls) -> list:
        """Get all topic names"""
        return [topic.value for topic in cls]

    @classmethod
    def sugar_glider_allowlist(cls) -> list:
        """Canonical SIDECAR_PUBLISH/CONSUME stream list for compose."""
        return cls.all()
