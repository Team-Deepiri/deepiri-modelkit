"""Dataset manifest I/O tests."""

import json
from pathlib import Path

from deepiri_modelkit.contracts.training import DatasetManifest
from deepiri_modelkit.training.manifest_io import read_manifest, write_manifest


def test_write_and_read_manifest(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        id="ds-io",
        version="0.1.0",
        path=str(tmp_path / "dataset.jsonl"),
        content_hash="hash-io",
        row_count=42,
        schema={"feature": "float"},
        produced_by="test-suite",
        metadata={"split": "train"},
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)

    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["id"] == "ds-io"

    restored = read_manifest(manifest_path)
    assert restored == manifest
