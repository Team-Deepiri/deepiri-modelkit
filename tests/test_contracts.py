"""Contract serialization tests."""

from deepiri_modelkit.contracts.events import EventType, ModelReadyEvent, TrainingEvent
from deepiri_modelkit.contracts.training import (
    AgentTrainingJob,
    DatasetManifest,
    TrainingPriority,
    TrainingRunRequest,
)


def _sample_manifest() -> DatasetManifest:
    return DatasetManifest(
        id="ds-001",
        version="1.0.0",
        path="/data/train.jsonl",
        content_hash="abc123",
        row_count=100,
        schema={"text": "string", "label": "string"},
        produced_by="deepiri-dataset-processor",
    )


def test_dataset_manifest_round_trip() -> None:
    manifest = _sample_manifest()
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_training_run_request_round_trip() -> None:
    request = TrainingRunRequest(
        experiment_id="exp-1",
        model_name="adapter-a",
        fingerprint="fp-123",
        dataset_manifest=_sample_manifest(),
        priority=TrainingPriority.LIVE,
        hyperparameters={"lr": 1e-4},
    )
    restored = TrainingRunRequest.model_validate(request.model_dump())
    assert restored.model_name == "adapter-a"
    assert restored.priority == TrainingPriority.LIVE
    assert restored.dataset_manifest.id == "ds-001"
    assert restored.id


def test_agent_training_job_round_trip() -> None:
    request = TrainingRunRequest(
        experiment_id="exp-2",
        model_name="adapter-b",
        fingerprint="fp-456",
        dataset_manifest=_sample_manifest(),
    )
    job = AgentTrainingJob(
        correlation_id="corr-1",
        user_id="user-9",
        learning_artifact_ids=["artifact-1", "artifact-2"],
        adapter_target="lora-rank-8",
        training_run_request=request,
    )
    restored = AgentTrainingJob.model_validate_json(job.model_dump_json())
    assert restored.correlation_id == "corr-1"
    assert restored.training_run_request is not None
    assert restored.training_run_request.experiment_id == "exp-2"


def test_training_event_extended_fields() -> None:
    event = TrainingEvent(
        event=EventType.TRAINING_STARTED.value,
        source="helox",
        experiment_id="exp-3",
        model_name="model-x",
        status="running",
        correlation_id="corr-99",
        training_run_request_id="run-42",
        progress=0.25,
    )
    payload = event.model_dump()
    assert payload["correlation_id"] == "corr-99"
    assert payload["training_run_request_id"] == "run-42"
    assert payload["progress"] == 0.25


def test_model_ready_event_round_trip() -> None:
    event = ModelReadyEvent(
        source="helox",
        model_name="model-y",
        version="2.0.0",
        registry_path="s3://models/model-y/2.0.0",
        metadata={"accuracy": 0.91},
        correlation_id="corr-100",
    )
    restored = ModelReadyEvent.model_validate(event.model_dump())
    assert restored.registry_path.startswith("s3://")
    assert restored.metadata["accuracy"] == 0.91
