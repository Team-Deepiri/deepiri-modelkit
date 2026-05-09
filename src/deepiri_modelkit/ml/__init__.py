"""ML utilities for Deepiri ModelKit"""

try:
    from .confidence import (
        ConfidenceLevel,
        ConfidenceSource,
        ConfidenceAttributes,
        ConfidenceCalculator,
        get_confidence_calculator,
    )
    _HAS_CONFIDENCE = True
except ImportError:
    _HAS_CONFIDENCE = False

try:
    from .semantic import SemanticAnalyzer, get_semantic_analyzer
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

__all__ = []

if _HAS_CONFIDENCE:
    __all__ += [
        "ConfidenceLevel",
        "ConfidenceSource",
        "ConfidenceAttributes",
        "ConfidenceCalculator",
        "get_confidence_calculator",
    ]

if _HAS_SEMANTIC:
    __all__ += ["SemanticAnalyzer", "get_semantic_analyzer"]
