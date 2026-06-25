import json
from pathlib import Path

from deepiri_modelkit.contracts.training import DatasetManifest
from deepiri_modelkit.data.manifest_validator import validate_manifest_against_path
from deepiri_modelkit.training.pipeline_factory import create_run_context, persist_manifest_for_dataset


def test_validate_manifest_against_path(tmp_path: Path):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "hello"}\n', encoding="utf-8")
    import hashlib

    sha = hashlib.sha256()
    sha.update(data_file.read_bytes())
    manifest = DatasetManifest(
        id="test",
        version="1",
        path=str(data_file),
        content_hash=sha.hexdigest(),
        row_count=1,
        schema={"fields": {}},
        produced_by="test",
    )
    report = validate_manifest_against_path(manifest)
    assert report["valid"] is True


def test_create_run_context_with_manifest(tmp_path: Path):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "ds1",
                "version": "1",
                "path": "/tmp",
                "content_hash": "abc",
                "row_count": 0,
                "schema": {},
                "produced_by": "test",
            }
        ),
        encoding="utf-8",
    )
    ctx = create_run_context("exp1", "model-a", manifest_path=str(manifest_path))
    assert ctx.manifest is not None
    assert ctx.manifest.id == "ds1"
