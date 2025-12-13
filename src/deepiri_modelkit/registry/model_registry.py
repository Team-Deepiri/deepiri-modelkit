"""
Unified model registry client
Supports MLflow, S3/MinIO, and local storage
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import mlflow
import boto3
from botocore.exceptions import ClientError

from ..contracts.models import ModelMetadata


class ModelRegistryClient:
    """
    Unified client for model registry operations
    Supports MLflow, S3/MinIO, and local storage
    """
    
    def __init__(
        self,
        registry_type: str = "mlflow",  # mlflow, s3, local
        mlflow_tracking_uri: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        local_path: Optional[str] = None
    ):
        """
        Initialize model registry client
        
        Args:
            registry_type: Type of registry (mlflow, s3, local)
            mlflow_tracking_uri: MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env var or http://mlflow:5000)
            s3_endpoint: S3/MinIO endpoint
            s3_access_key: S3 access key
            s3_secret_key: S3 secret key
            s3_bucket: S3 bucket name
            local_path: Local storage path
        """
        self.registry_type = registry_type
        
        if registry_type == "mlflow":
            # Use provided URI, or environment variable, or default
            tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
            mlflow.set_tracking_uri(tracking_uri)
            self.client = mlflow
            self.tracking_uri = tracking_uri
        elif registry_type == "s3":
            self.s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key
            )
            self.s3_bucket = s3_bucket
        elif registry_type == "local":
            self.local_path = Path(local_path or "./models")
            self.local_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown registry type: {registry_type}")
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Register model in registry
        
        Args:
            model_name: Model name
            version: Model version
            model_path: Path to model file/directory
            metadata: Model metadata
        
        Returns:
            True if successful
        """
        try:
            if self.registry_type == "mlflow":
                # Register with MLflow
                model_uri = f"runs:/{metadata.get('run_id', 'latest')}/model"
                mlflow.register_model(model_uri, f"{model_name}-{version}")
                return True
            
            elif self.registry_type == "s3":
                # Upload to S3
                s3_key = f"models/{model_name}/{version}/model"
                self.s3_client.upload_file(model_path, self.s3_bucket, s3_key)
                
                # Upload metadata
                import json
                metadata_key = f"models/{model_name}/{version}/metadata.json"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=metadata_key,
                    Body=json.dumps(metadata)
                )
                return True
            
            elif self.registry_type == "local":
                # Copy to local storage
                model_dir = self.local_path / model_name / version
                model_dir.mkdir(parents=True, exist_ok=True)
                
                import shutil
                if os.path.isdir(model_path):
                    shutil.copytree(model_path, model_dir / "model", dirs_exist_ok=True)
                else:
                    shutil.copy2(model_path, model_dir / "model")
                
                # Save metadata
                import json
                with open(model_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f)
                
                return True
        
        except Exception as e:
            print(f"Error registering model: {e}")
            return False
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model information from registry
        
        Args:
            model_name: Model name
            version: Model version (optional, gets latest if not specified)
        
        Returns:
            Model information dict
        """
        try:
            if self.registry_type == "mlflow":
                if version:
                    model_uri = f"models:/{model_name}/{version}"
                else:
                    model_uri = f"models:/{model_name}/latest"
                
                model = mlflow.pyfunc.load_model(model_uri)
                return {
                    "model": model,
                    "uri": model_uri,
                    "type": "mlflow"
                }
            
            elif self.registry_type == "s3":
                if not version:
                    # List versions and get latest
                    prefix = f"models/{model_name}/"
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.s3_bucket,
                        Prefix=prefix,
                        Delimiter="/"
                    )
                    versions = [obj["Prefix"].split("/")[-2] for obj in response.get("CommonPrefixes", [])]
                    version = max(versions) if versions else None
                
                if not version:
                    raise ValueError(f"Model {model_name} not found")
                
                # Download metadata
                metadata_key = f"models/{model_name}/{version}/metadata.json"
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=metadata_key)
                import json
                metadata = json.loads(response["Body"].read())
                
                return {
                    "model_path": f"s3://{self.s3_bucket}/models/{model_name}/{version}/model",
                    "metadata": metadata,
                    "type": "s3"
                }
            
            elif self.registry_type == "local":
                if not version:
                    # Get latest version
                    model_dir = self.local_path / model_name
                    if not model_dir.exists():
                        raise ValueError(f"Model {model_name} not found")
                    
                    versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                    version = max(versions) if versions else None
                
                if not version:
                    raise ValueError(f"Model {model_name} not found")
                
                model_dir = self.local_path / model_name / version
                metadata_path = model_dir / "metadata.json"
                
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                return {
                    "model_path": str(model_dir / "model"),
                    "metadata": metadata,
                    "type": "local"
                }
        
        except Exception as e:
            print(f"Error getting model: {e}")
            raise
    
    def download_model(
        self,
        model_name: str,
        version: str,
        destination: str
    ) -> str:
        """
        Download model to destination
        
        Args:
            model_name: Model name
            version: Model version
            destination: Local destination path
        
        Returns:
            Local path to downloaded model
        """
        model_info = self.get_model(model_name, version)
        
        if self.registry_type == "s3":
            # Download from S3
            s3_key = f"models/{model_name}/{version}/model"
            os.makedirs(destination, exist_ok=True)
            
            # Check if it's a file or directory
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                # It's a file
                local_path = os.path.join(destination, "model")
                self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                return local_path
            except ClientError:
                # It's a directory, list and download all files
                prefix = f"{s3_key}/"
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        local_file = os.path.join(destination, key[len(prefix):])
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                        self.s3_client.download_file(self.s3_bucket, key, local_file)
                return destination
        
        elif self.registry_type == "local":
            # Copy from local
            source = model_info["model_path"]
            import shutil
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)
            return destination
        
        else:
            # MLflow handles loading directly
            return model_info["uri"]
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """
        List available models
        
        Args:
            model_name: Optional model name filter
        
        Returns:
            List of model information dicts
        """
        try:
            if self.registry_type == "mlflow":
                if model_name:
                    models = mlflow.search_registered_models(filter_string=f"name='{model_name}'")
                else:
                    models = mlflow.search_registered_models()
                
                return [
                    {
                        "name": m.name,
                        "versions": [v.version for v in m.latest_versions],
                        "latest_version": m.latest_versions[0].version if m.latest_versions else None
                    }
                    for m in models
                ]
            
            elif self.registry_type == "s3":
                prefix = "models/"
                if model_name:
                    prefix = f"models/{model_name}/"
                
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=prefix,
                    Delimiter="/"
                )
                
                models = []
                for prefix_obj in response.get("CommonPrefixes", []):
                    model_path = prefix_obj["Prefix"]
                    parts = model_path.strip("/").split("/")
                    if len(parts) >= 2:
                        models.append({
                            "name": parts[1],
                            "path": model_path
                        })
                
                return models
            
            elif self.registry_type == "local":
                models = []
                for model_dir in self.local_path.iterdir():
                    if model_dir.is_dir():
                        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                        models.append({
                            "name": model_dir.name,
                            "versions": versions,
                            "latest_version": max(versions) if versions else None
                        })
                return models
        
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

