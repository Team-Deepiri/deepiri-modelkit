"""
Model contract for registry (separated from models.py to avoid Pydantic Protocol conflicts)
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel

from .models import ModelMetadata


class ModelContract(BaseModel):
    """
    Complete model contract for registry.
    
    A contract is serializable metadata that describes a model's interface,
    input/output schemas, and validation requirements. It does NOT contain
    the actual model instance (which would be a Protocol type that Pydantic
    cannot serialize). The model instance should be loaded separately when needed.
    """
    metadata: ModelMetadata
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    validation_tests: Optional[list] = None
    model_path: Optional[str] = None  # Path/reference to where the model can be loaded from
    model_id: Optional[str] = None  # Unique identifier for the model instance

