"""
Shared logging utilities for all Deepiri services
Used by: Cyrex (runtime), Helox (training), and all microservices
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """JSON structured logger for all Deepiri services"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create console handler with JSON formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        
        self.logger.propagate = False
    
    def _log(self, level: int, event: str, **kwargs):
        """Internal log method with structured data"""
        extra = {"event": event, "timestamp": datetime.utcnow().isoformat() + "Z"}
        extra.update(kwargs)
        self.logger.log(level, json.dumps(extra))
    
    def debug(self, event: str, **kwargs):
        self._log(logging.DEBUG, event, **kwargs)
    
    def info(self, event: str, **kwargs):
        self._log(logging.INFO, event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        self._log(logging.WARNING, event, **kwargs)
    
    def error(self, event: str, **kwargs):
        self._log(logging.ERROR, event, **kwargs)
    
    def critical(self, event: str, **kwargs):
        self._log(logging.CRITICAL, event, **kwargs)


class JsonFormatter(logging.Formatter):
    """Format logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'event'):
            log_data['event'] = record.event
        
        # Add any custom fields from extra={}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'event', 'timestamp']:
                log_data[key] = value
        
        return json.dumps(log_data)


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """
    Get structured logger instance
    
    Usage:
        from deepiri_modelkit.logging import get_logger
        logger = get_logger("my_service")
        logger.info("service_started", port=8000, version="1.0")
    """
    return StructuredLogger(name, level)


class ErrorLogger:
    """Error logging with context"""
    
    def __init__(self):
        self.logger = get_logger("error_logger")
    
    def log_api_error(self, error: Exception, request_id: str, endpoint: str):
        """Log API errors with context"""
        self.logger.error(
            "api_error",
            error=str(error),
            error_type=type(error).__name__,
            request_id=request_id,
            endpoint=endpoint
        )
    
    def log_model_error(self, error: Exception, model_name: str, input_data: Optional[Dict] = None):
        """Log model inference errors"""
        self.logger.error(
            "model_error",
            error=str(error),
            error_type=type(error).__name__,
            model_name=model_name,
            input_sample=str(input_data)[:200] if input_data else None
        )
    
    def log_training_error(self, error: Exception, pipeline: str, config: Optional[Dict] = None):
        """Log training pipeline errors"""
        self.logger.error(
            "training_error",
            error=str(error),
            error_type=type(error).__name__,
            pipeline=pipeline,
            config=config
        )


# Singleton instances
_loggers: Dict[str, StructuredLogger] = {}
_error_logger: Optional[ErrorLogger] = None


def get_cached_logger(name: str) -> StructuredLogger:
    """Get or create cached logger instance"""
    if name not in _loggers:
        _loggers[name] = get_logger(name)
    return _loggers[name]


def get_error_logger() -> ErrorLogger:
    """Get singleton error logger"""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger

