"""Factory helpers for standard training pipeline setup."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..data.manifest_validator import validate_manifest_against_path
from ..logging import get_logger
from .manifest_io import read_manifest, write_manifest
from .run_context import TrainingRunContext

logger = get_logger("modelkit.pipeline_factory")


def create_run_context(
    experiment_id: str,
    model_name: str,
    *,
    source: str = "helox",
    fingerprint: Optional[str] = None,
    correlation_id: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> TrainingRunContext:
    """Build a TrainingRunContext, optionally loading a manifest from disk."""
    manifest = read_manifest(manifest_path) if manifest_path else None
    return TrainingRunContext(
        experiment_id=experiment_id,
        model_name=model_name,
        source=source,
        fingerprint=fingerprint,
        correlation_id=correlation_id,
        manifest=manifest,
    )


def persist_manifest_for_dataset(
    manifest: Any,
    output_dir: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """Write manifest JSON and optionally validate against dataset path."""
    from ..contracts.training import DatasetManifest

    if not isinstance(manifest, DatasetManifest):
        manifest = DatasetManifest.model_validate(manifest)
    path = write_manifest(manifest, Path(output_dir) / f"{manifest.id}.manifest.json")
    if validate:
        report = validate_manifest_against_path(manifest)
        if not report["valid"]:
            logger.warning("manifest_validation_failed", report=report)
    return path


def pipeline_metadata(
    ctx: TrainingRunContext, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Standard metadata blob for registry + events."""
    data: Dict[str, Any] = {
        "experiment_id": ctx.experiment_id,
        "model_name": ctx.model_name,
        "source": ctx.source,
    }
    if ctx.fingerprint:
        data["fingerprint"] = ctx.fingerprint
    if ctx.manifest:
        data["dataset_manifest"] = ctx.manifest.model_dump(mode="json")
    if extra:
        data.update(extra)
    return data
