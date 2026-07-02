"""
Redis stream producer/consumer for training job requests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import redis
from redis.exceptions import ResponseError

from ..contracts.training import TrainingRunRequest

TRAINING_JOBS_STREAM = "training-jobs"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
PAYLOAD_FIELD = "payload"


class TrainingJobQueue:
    """Redis Streams queue for TrainingRunRequest messages."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        redis_url: Optional[str] = None,
        stream_name: str = TRAINING_JOBS_STREAM,
    ) -> None:
        if redis_client is not None:
            self.redis = redis_client
        else:
            url: str = redis_url or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
            self.redis = redis.from_url(url, decode_responses=True)
        self.stream_name = stream_name

    def enqueue(self, request: TrainingRunRequest, max_length: int = 10000) -> str:
        """Publish a training run request to the jobs stream."""
        payload = request.model_dump_json()
        message_id: str = self.redis.xadd(
            self.stream_name,
            {PAYLOAD_FIELD: payload},
            maxlen=max_length,
            approximate=True,
        )
        return message_id

    def read_messages(
        self,
        count: int = 10,
        last_id: str = "0",
        block_ms: Optional[int] = None,
    ) -> List[Tuple[str, TrainingRunRequest]]:
        """Read training requests from the stream."""
        if block_ms is None:
            messages = self.redis.xread({self.stream_name: last_id}, count=count)
        else:
            messages = self.redis.xread(
                {self.stream_name: last_id},
                count=count,
                block=block_ms,
            )

        results: List[Tuple[str, TrainingRunRequest]] = []
        for _stream_name, stream_messages in messages:
            for message_id, fields in stream_messages:
                request = self._decode_payload(fields)
                results.append((message_id, request))
        return results

    def consume(
        self,
        consumer_group: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 1000,
        auto_ack: bool = True,
    ) -> Iterator[Tuple[str, TrainingRunRequest]]:
        """Consume training requests from a consumer group."""
        self.ensure_consumer_group(consumer_group)
        while True:
            messages = self.redis.xreadgroup(
                consumer_group,
                consumer_name,
                {self.stream_name: ">"},
                count=count,
                block=block_ms,
            )
            if not messages:
                continue

            for _stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    request = self._decode_payload(fields)
                    yield message_id, request
                    if auto_ack:
                        self.redis.xack(self.stream_name, consumer_group, message_id)

    def ensure_consumer_group(self, consumer_group: str) -> None:
        """Create the consumer group if it does not already exist."""
        try:
            self.redis.xgroup_create(
                self.stream_name,
                consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def stream_length(self) -> int:
        """Return the number of messages in the jobs stream."""
        return int(self.redis.xlen(self.stream_name))

    @staticmethod
    def _decode_payload(fields: Dict[str, Any]) -> TrainingRunRequest:
        raw_payload = fields.get(PAYLOAD_FIELD)
        if raw_payload is None:
            raise ValueError(f"Missing '{PAYLOAD_FIELD}' field in training job message")
        return TrainingRunRequest.model_validate_json(raw_payload)
