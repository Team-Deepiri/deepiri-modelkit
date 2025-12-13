"""
Model contracts and interfaces
"""
from __future__ import annotations  # Defer annotation evaluation to prevent Pydantic from processing Protocol types

from typing import Protocol, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from datetime import datetime


class ModelInput(BaseModel):
    """Standard model input schema"""
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelOutput(BaseModel):
    """Standard model output schema"""
    prediction: Any
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelMetadata(BaseModel):
    """Model metadata schema"""
    name: str
    version: str
    description: Optional[str] = None
    architecture: Optional[str] = None
    accuracy: Optional[float] = None
    size_mb: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    trained_by: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class AIModel(Protocol):
    """
    Interface that all models must implement
    Used by both Helox (training) and Cyrex (runtime)
    
    Note: This is a Protocol (structural type). To use in Pydantic models,
    use AIModelPydantic instead, which has full Pydantic schema support.
    """
    
    def predict(self, input: ModelInput) -> ModelOutput:
        """Run inference on input"""
        ...
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        ...
    
    def validate(self) -> bool:
        """Validate model is ready for use"""
        ...
    
    def export(self, format: str = "onnx") -> str:
        """Export model to specified format, returns path"""
        ...


class AIModelPydantic:
    """
    Pydantic-compatible wrapper for AIModel Protocol.
    Implements __get_pydantic_core_schema__ to provide full schema support.
    
    Usage in Pydantic models:
        model: Optional[AIModelPydantic] = None
    
    Note: In practice, model instances should be loaded separately and referenced
    by ID/path rather than stored directly in serializable Pydantic models.
    """
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Pydantic Core Schema handler for AIModel Protocol.
        
        This allows Pydantic to process AIModel types in model fields.
        Since Protocols are structural types, we validate that the object
        has the required methods rather than checking exact type.
        """
        def validate_aimodel(value: Any) -> Any:
            """Validate that value implements AIModel Protocol interface"""
            if value is None:
                return None
            
            # Check for required Protocol methods
            required_methods = ['predict', 'get_metadata', 'validate', 'export']
            missing_methods = [m for m in required_methods if not hasattr(value, m)]
            
            if missing_methods:
                raise ValueError(
                    f"Object does not implement AIModel Protocol. "
                    f"Missing methods: {', '.join(missing_methods)}"
                )
            
            return value
        
        def serialize_aimodel(value: Any) -> Dict[str, Any]:
            """Serialize AIModel instance to dict"""
            if value is None:
                return None
            
            # Try to get metadata if available
            metadata = None
            if hasattr(value, 'get_metadata'):
                try:
                    metadata = value.get_metadata()
                    # Convert ModelMetadata to dict if it's a Pydantic model
                    if hasattr(metadata, 'model_dump'):
                        metadata = metadata.model_dump()
                    elif hasattr(metadata, 'dict'):
                        metadata = metadata.dict()
                except Exception:
                    pass
            
            return {
                "type": "AIModel",
                "metadata": metadata,
                "has_predict": hasattr(value, 'predict'),
                "has_validate": hasattr(value, 'validate'),
            }
        
        return core_schema.no_info_plain_validator_function(
            validate_aimodel,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_aimodel
            )
        )


# ModelContract moved to contract.py to avoid Pydantic Protocol conflicts
# Import it from .contract import ModelContract when needed

