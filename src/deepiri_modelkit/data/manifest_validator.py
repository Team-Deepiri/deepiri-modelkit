"""Validate DatasetManifest artifacts against on-disk datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.training import DatasetManifest


def _hash_path(path: Path) -> str:
    sha = hashlib.sha256()
    if path.is_file():
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                sha.update(str(child.relative_to(path)).encode())
                with open(child, "rb") as handle:
                    for chunk in iter(lambda: handle.read(8192), b""):
                        sha.update(chunk)
        return sha.hexdigest()
    raise FileNotFoundError(path)


def _count_jsonl_rows(path: Path) -> int:
    if path.is_file() and path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.is_dir():
        total = 0
        for file_path in path.rglob("*.jsonl"):
            with open(file_path, encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        return total
    return 0


def validate_manifest_against_path(
    manifest: DatasetManifest,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate manifest fields against filesystem state.

    Returns a report dict with ``valid`` bool and per-check results.
    """
    path = Path(dataset_path or manifest.path)
    report: Dict[str, Any] = {
        "valid": True,
        "checks": {},
        "manifest_id": manifest.id,
    }

    if not path.exists():
        report["valid"] = False
        report["checks"]["path_exists"] = False
        return report
    report["checks"]["path_exists"] = True

    try:
        actual_hash = _hash_path(path)
        hash_ok = actual_hash == manifest.content_hash
        report["checks"]["content_hash"] = hash_ok
        if not hash_ok:
            report["valid"] = False
            report["actual_hash"] = actual_hash
    except OSError as exc:
        report["valid"] = False
        report["checks"]["content_hash"] = str(exc)

    if manifest.row_count > 0:
        actual_rows = _count_jsonl_rows(path)
        rows_ok = actual_rows == manifest.row_count
        report["checks"]["row_count"] = rows_ok
        report["actual_row_count"] = actual_rows
        if not rows_ok:
            report["valid"] = False

    return report


def validate_manifest_file(manifest_path: str | Path) -> Dict[str, Any]:
    """Load manifest JSON and validate against its embedded path."""
    from ..training.manifest_io import read_manifest

    manifest = read_manifest(manifest_path)
    return validate_manifest_against_path(manifest)
