"""
Integration helpers for publishing training events and registering trained models.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import redis

from ..contracts.events import ModelReadyEvent, TrainingEvent
from ..registry.model_registry import ModelRegistryClient
from ..streaming.topics import StreamTopics
from .job_queue import DEFAULT_REDIS_URL, TRAINING_JOBS_STREAM

TRAINING_EVENTS_STREAM = StreamTopics.TRAINING_EVENTS.value


def _get_redis_client(
    redis_client: Optional[redis.Redis] = None,
    redis_url: Optional[str] = None,
) -> redis.Redis:
    if redis_client is not None:
        return redis_client
    url: str = redis_url or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
    client: redis.Redis = redis.from_url(url, decode_responses=True)
    return client


def _flatten_event_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value)
        else:
            flat[key] = str(value)
    return flat


def publish_training_event(
    event: TrainingEvent,
    redis_client: Optional[redis.Redis] = None,
    redis_url: Optional[str] = None,
    stream_name: str = TRAINING_EVENTS_STREAM,
    max_length: int = 10000,
) -> str:
    """Publish a typed training event to the training-events Redis stream."""
    client = _get_redis_client(redis_client, redis_url)
    payload = event.model_dump(mode="json")
    message_id: str = client.xadd(
        stream_name,
        _flatten_event_payload(payload),
        maxlen=max_length,
        approximate=True,
    )
    return message_id


def publish_model_ready_event(
    event: ModelReadyEvent,
    redis_client: Optional[redis.Redis] = None,
    redis_url: Optional[str] = None,
    stream_name: str = StreamTopics.MODEL_EVENTS.value,
    max_length: int = 10000,
) -> str:
    """Publish a model-ready event to the model-events Redis stream."""
    client = _get_redis_client(redis_client, redis_url)
    payload = event.model_dump(mode="json")
    message_id: str = client.xadd(
        stream_name,
        _flatten_event_payload(payload),
        maxlen=max_length,
        approximate=True,
    )
    return message_id


def register_model_ready(
    registry_client: ModelRegistryClient,
    model_name: str,
    version: str,
    model_path: str,
    metadata: Dict[str, Any],
    *,
    source: str = "deepiri-modelkit",
    correlation_id: Optional[str] = None,
    model_type: Optional[str] = None,
    accuracy: Optional[float] = None,
    size_mb: Optional[float] = None,
    redis_client: Optional[redis.Redis] = None,
    redis_url: Optional[str] = None,
    publish_event: bool = True,
) -> bool:
    """Register a model and optionally publish a ModelReadyEvent."""
    registered = registry_client.register_model(
        model_name=model_name,
        version=version,
        model_path=model_path,
        metadata=metadata,
    )
    if not registered or not publish_event:
        return registered

    registry_path = metadata.get("registry_path", model_path)
    ready_event = ModelReadyEvent(
        source=source,
        correlation_id=correlation_id,
        model_name=model_name,
        version=version,
        registry_path=registry_path,
        metadata=metadata,
        model_type=model_type,
        accuracy=accuracy,
        size_mb=size_mb,
    )
    publish_model_ready_event(
        ready_event,
        redis_client=redis_client,
        redis_url=redis_url,
    )
    return registered


def training_jobs_stream_name() -> str:
    """Return the canonical Redis stream name for training jobs."""
    return TRAINING_JOBS_STREAM
