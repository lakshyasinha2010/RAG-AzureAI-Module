"""
Logging configuration and utilities.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory

from ..config import get_settings


def configure_logging():
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.format == "json"
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.logging.level.upper()),
    )
    
    # Configure file logging if specified
    if settings.logging.file_path:
        file_handler = logging.FileHandler(settings.logging.file_path)
        file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get a logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(logger: structlog.stdlib.BoundLogger):
    """Decorator to log function calls with parameters and execution time."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            logger.info(
                "Function called",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            try:
                result = await func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(
                    "Function completed",
                    function=func.__name__,
                    execution_time=execution_time,
                )
                return result
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            logger.info(
                "Function called",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(
                    "Function completed",
                    function=func.__name__,
                    execution_time=execution_time,
                )
                return result
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def sanitize_log_data(data: Dict[str, Any], max_length: int = 1000) -> Dict[str, Any]:
    """Sanitize log data by truncating long strings and removing sensitive information."""
    sensitive_keys = {"password", "token", "key", "secret", "credential", "api_key"}
    
    def sanitize_value(value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length] + "... (truncated)"
            return value
        elif isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [sanitize_value(item) for item in value[:10]]  # Limit list size
        else:
            return value
    
    sanitized = {}
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = sanitize_value(value)
    
    return sanitized


def create_correlation_id() -> str:
    """Create a unique correlation ID for request tracking."""
    import uuid
    return str(uuid.uuid4())


class CorrelationContext:
    """Context manager for correlation IDs in logs."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or create_correlation_id()
        self.logger = get_logger("correlation")
        self._bound_logger = None
    
    def __enter__(self):
        self._bound_logger = self.logger.bind(correlation_id=self.correlation_id)
        return self._bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._bound_logger.error(
                "Operation failed",
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self._bound_logger.info("Operation completed successfully")


# Performance monitoring
class PerformanceTimer:
    """Context manager for measuring and logging execution time."""
    
    def __init__(self, operation_name: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger("performance")
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.debug("Operation started", operation=self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.warning(
                "Operation failed",
                operation=self.operation_name,
                duration=duration,
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self.logger.info(
                "Operation completed",
                operation=self.operation_name,
                duration=duration,
            )
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration in seconds if the timer has completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# Initialize logging when module is imported
configure_logging()