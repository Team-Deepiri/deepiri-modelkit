"""Training job queue tests."""

import fakeredis

from deepiri_modelkit.contracts.training import (
    DatasetManifest,
    TrainingPriority,
    TrainingRunRequest,
)
from deepiri_modelkit.training.integration import (
    publish_training_event,
    training_jobs_stream_name,
)
from deepiri_modelkit.training.job_queue import TRAINING_JOBS_STREAM, TrainingJobQueue
from deepiri_modelkit.contracts.events import EventType, TrainingEvent


def _sample_request(
    priority: TrainingPriority = TrainingPriority.BATCH,
) -> TrainingRunRequest:
    manifest = DatasetManifest(
        id="ds-queue",
        version="1.0.0",
        path="/tmp/data.jsonl",
        content_hash="queue-hash",
        row_count=10,
        schema={"text": "string"},
        produced_by="tests",
    )
    return TrainingRunRequest(
        experiment_id="exp-queue",
        model_name="queue-model",
        fingerprint="fp-queue",
        dataset_manifest=manifest,
        priority=priority,
    )


def test_enqueue_and_read_messages() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    queue = TrainingJobQueue(redis_client=fake)

    request = _sample_request(TrainingPriority.LIVE)
    message_id = queue.enqueue(request)

    assert message_id
    assert queue.stream_length() == 1

    messages = queue.read_messages()
    assert len(messages) == 1
    read_id, read_request = messages[0]
    assert read_id == message_id
    assert read_request.model_name == "queue-model"
    assert read_request.priority == TrainingPriority.LIVE
    assert read_request.dataset_manifest.id == "ds-queue"


def test_consumer_group_consume() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    queue = TrainingJobQueue(redis_client=fake)
    queue.enqueue(_sample_request())

    consumer = queue.consume("workers", "worker-1", block_ms=1, auto_ack=True)
    message_id, request = next(consumer)

    assert message_id
    assert request.experiment_id == "exp-queue"


def test_publish_training_event() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    event = TrainingEvent(
        event=EventType.TRAINING_COMPLETE.value,
        source="tests",
        experiment_id="exp-event",
        model_name="event-model",
        status="complete",
        correlation_id="corr-event",
        training_run_request_id="run-event",
        metrics={"loss": 0.1},
    )
    message_id = publish_training_event(event, redis_client=fake)
    assert message_id

    messages = fake.xread({"training-events": "0"})
    assert messages
    _stream, stream_messages = messages[0]
    _msg_id, fields = stream_messages[0]
    assert fields["correlation_id"] == "corr-event"
    assert fields["training_run_request_id"] == "run-event"


def test_training_jobs_stream_name() -> None:
    assert training_jobs_stream_name() == TRAINING_JOBS_STREAM
