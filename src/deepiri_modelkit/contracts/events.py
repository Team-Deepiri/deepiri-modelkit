"""
Event schemas for streaming service
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Event type enumeration"""
    MODEL_READY = "model-ready"
    MODEL_LOADED = "model-loaded"
    MODEL_FAILED = "model-failed"
    INFERENCE_COMPLETE = "inference-complete"
    INFERENCE_FAILED = "inference-failed"
    USER_INTERACTION = "user-interaction"
    TASK_CREATED = "task-created"
    TASK_COMPLETED = "task-completed"
    AGI_DECISION = "agi-decision"
    AGI_ACTION = "agi-action"
    TRAINING_STARTED = "training-started"
    TRAINING_COMPLETE = "training-complete"
    TRAINING_FAILED = "training-failed"


class BaseEvent(BaseModel):
    """Base event schema"""
    event: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str
    correlation_id: Optional[str] = None


class ModelReadyEvent(BaseEvent):
    """Event published when model is trained and ready"""
    event: str = EventType.MODEL_READY
    model_name: str
    version: str
    registry_path: str  # S3/MLflow path
    metadata: Dict[str, Any]
    model_type: Optional[str] = None
    accuracy: Optional[float] = None
    size_mb: Optional[float] = None


class ModelLoadedEvent(BaseEvent):
    """Event published when model is loaded in runtime"""
    event: str = EventType.MODEL_LOADED
    model_name: str
    version: str
    load_time_ms: float
    cache_location: Optional[str] = None


class InferenceEvent(BaseEvent):
    """Event published after inference completes"""
    event: str = EventType.INFERENCE_COMPLETE
    model_name: str
    version: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    latency_ms: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    confidence: Optional[float] = None
    success: bool = True


class PlatformEvent(BaseEvent):
    """Event published by platform services"""
    event: str  # user-interaction, task-created, etc.
    service: str
    user_id: Optional[str] = None
    action: str
    data: Dict[str, Any]
    organization_id: Optional[str] = None


class AGIDecisionEvent(BaseEvent):
    """Event published by Cyrex-AGI for autonomous decisions"""
    event: str = EventType.AGI_DECISION
    decision_type: str
    target_service: Optional[str] = None
    action: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


class TrainingEvent(BaseEvent):
    """Event published during training"""
    event: str  # training-started, training-complete, training-failed
    experiment_id: str
    model_name: str
    status: str
    progress: Optional[float] = None  # 0.0 to 1.0
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

