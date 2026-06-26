"""
Read and write DatasetManifest JSON artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from ..contracts.training import DatasetManifest

PathLike = Union[str, Path]


def write_manifest(manifest: DatasetManifest, path: PathLike) -> Path:
    """Serialize a dataset manifest to JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return output


def read_manifest(path: PathLike) -> DatasetManifest:
    """Load a dataset manifest from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(data)
