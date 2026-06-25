"""Helox-oriented helpers built on modelkit contracts and streaming."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..contracts.events import EventType, TrainingEvent
from ..contracts.training import DatasetManifest, TrainingPriority, TrainingRunRequest
from .integration import publish_training_event
from .run_context import TrainingRunContext


def build_training_event(
    event_type: str,
    ctx: TrainingRunContext,
    *,
    status: str = "running",
    progress: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> TrainingEvent:
    """Map Helox-style event names to typed TrainingEvent."""
    normalized = event_type
    if event_type in ("started", "start"):
        normalized = EventType.TRAINING_STARTED.value
    elif event_type in ("completed", "complete", "done"):
        normalized = EventType.TRAINING_COMPLETE.value
    elif event_type in ("failed", "error"):
        normalized = EventType.TRAINING_FAILED.value
    elif event_type in ("progress", "checkpoint"):
        normalized = EventType.TRAINING_STARTED.value

    return TrainingEvent(
        event=normalized,
        source=ctx.source,
        correlation_id=ctx.correlation_id,
        experiment_id=ctx.experiment_id,
        model_name=ctx.model_name,
        status=status,
        training_run_request_id=ctx.request.id if ctx.request else None,
        progress=progress,
        metrics=metrics,
        error=error,
    )


def emit_training_lifecycle_event(
    event_type: str,
    ctx: TrainingRunContext,
    *,
    status: str = "running",
    progress: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    redis_url: Optional[str] = None,
) -> Optional[str]:
    """Publish a typed training lifecycle event; returns Redis message id."""
    event = build_training_event(
        event_type,
        ctx,
        status=status,
        progress=progress,
        metrics=metrics,
        error=error,
    )
    try:
        return publish_training_event(event, redis_url=redis_url)
    except Exception:
        return None


def build_training_run_request(
    ctx: TrainingRunContext,
    *,
    priority: TrainingPriority = TrainingPriority.BATCH,
    hyperparameters: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> TrainingRunRequest:
    """Create a TrainingRunRequest from a run context."""
    if ctx.manifest is None:
        raise ValueError("TrainingRunContext.manifest is required to build a request")
    fingerprint = ctx.fingerprint or "pending"
    request = TrainingRunRequest(
        experiment_id=ctx.experiment_id,
        model_name=ctx.model_name,
        fingerprint=fingerprint,
        dataset_manifest=ctx.manifest,
        priority=priority,
        hyperparameters=hyperparameters,
        tags=tags,
    )
    ctx.request = request
    return request
