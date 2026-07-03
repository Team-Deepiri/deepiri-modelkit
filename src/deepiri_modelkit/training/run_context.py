"""Shared training run context for Helox, Cyrex, and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..contracts.training import DatasetManifest, TrainingRunRequest


@dataclass
class TrainingRunContext:
    """Lifecycle context carried across training pipelines."""

    experiment_id: str
    model_name: str
    source: str = "deepiri"
    fingerprint: Optional[str] = None
    correlation_id: Optional[str] = None
    manifest: Optional[DatasetManifest] = None
    request: Optional[TrainingRunRequest] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> Dict[str, str]:
        params = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "source": self.source,
        }
        if self.fingerprint:
            params["fingerprint"] = self.fingerprint
        if self.correlation_id:
            params["correlation_id"] = self.correlation_id
        if self.manifest:
            params["dataset_id"] = self.manifest.id
            params["dataset_version"] = self.manifest.version
            params["dataset_hash"] = self.manifest.content_hash
        return params
