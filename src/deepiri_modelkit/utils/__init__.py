"""Common utilities for Deepiri ModelKit"""

try:
    from .device import get_device, get_torch_device
    __all__ = ["get_device", "get_torch_device"]
except ImportError:
    __all__ = []
