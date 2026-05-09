"""
Service contracts and interfaces
"""
from typing import Protocol, Dict, Any, Optional
from pydantic import BaseModel


class ModelRegistryService(Protocol):
    """Interface for model registry operations"""
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Register a model in the registry"""
        ...
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model information from registry"""
        ...
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """List available models"""
        ...
    
    def download_model(
        self,
        model_name: str,
        version: str,
        destination: str
    ) -> str:
        """Download model to destination, returns local path"""
        ...


class StreamingService(Protocol):
    """Interface for streaming operations"""
    
    def publish(self, topic: str, event: Dict[str, Any]) -> bool:
        """Publish event to topic"""
        ...
    
    def subscribe(
        self,
        topic: str,
        callback: callable,
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to topic with callback"""
        ...

