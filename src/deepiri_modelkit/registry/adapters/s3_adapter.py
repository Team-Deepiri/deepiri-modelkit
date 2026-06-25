"""S3/MinIO-backed model registry adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..model_registry import ModelRegistryClient


class S3Adapter:
    """Thin adapter over ModelRegistryClient for S3-compatible storage."""

    def __init__(
        self,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ) -> None:
        self._client = ModelRegistryClient(
            registry_type="s3",
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_bucket=s3_bucket,
        )

    @property
    def client(self) -> ModelRegistryClient:
        return self._client

    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any],
    ) -> bool:
        return self._client.register_model(model_name, version, model_path, metadata)

    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client.get_model(model_name, version)

    def download_model(self, model_name: str, version: str, destination: str) -> str:
        return self._client.download_model(model_name, version, destination)

    def list_models(self, model_name: Optional[str] = None) -> list:
        return self._client.list_models(model_name)
