"""
Training contracts shared across Helox, Cyrex, and the training orchestrator.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TrainingPriority(str, Enum):
    """Job scheduling priority for training runs."""

    LIVE = "live"
    BATCH = "batch"


class DatasetManifest(BaseModel):
    """Immutable description of a versioned dataset used for training."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    version: str
    path: str
    content_hash: str
    row_count: int
    dataset_schema: Dict[str, Any] = Field(alias="schema")
    produced_by: str
    metadata: Optional[Dict[str, Any]] = None


class TrainingRunRequest(BaseModel):
    """Request to enqueue or start a training run."""

    experiment_id: str
    model_name: str
    fingerprint: str
    dataset_manifest: DatasetManifest
    priority: TrainingPriority = TrainingPriority.BATCH
    id: str = Field(default_factory=lambda: str(uuid4()))
    hyperparameters: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None


class AgentTrainingJob(BaseModel):
    """Agent-initiated training job for live correction and adapter fine-tuning."""

    correlation_id: str
    user_id: str
    learning_artifact_ids: List[str]
    adapter_target: str
    training_run_request: Optional[TrainingRunRequest] = None
    metadata: Optional[Dict[str, Any]] = None
