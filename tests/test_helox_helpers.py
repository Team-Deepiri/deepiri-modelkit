from deepiri_modelkit.contracts.training import DatasetManifest, TrainingPriority
from deepiri_modelkit.training.helox_helpers import (
    build_training_run_request,
    emit_training_lifecycle_event,
)
from deepiri_modelkit.training.run_context import TrainingRunContext


def test_build_training_run_request():
    manifest = DatasetManifest(
        id="ds",
        version="1",
        path="/data",
        content_hash="hash",
        row_count=10,
        schema={},
        produced_by="test",
    )
    ctx = TrainingRunContext(
        experiment_id="exp",
        model_name="m",
        fingerprint="fp123",
        manifest=manifest,
    )
    req = build_training_run_request(ctx, priority=TrainingPriority.BATCH)
    assert req.experiment_id == "exp"
    assert req.dataset_manifest.id == "ds"


def test_emit_training_lifecycle_event_no_redis():
    ctx = TrainingRunContext(experiment_id="e", model_name="m")
    # Should not raise when redis unavailable
    result = emit_training_lifecycle_event("started", ctx, status="running")
    assert result is None or isinstance(result, str)
