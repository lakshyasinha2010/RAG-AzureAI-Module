"""
Structured logging configuration for the multimodal RAG service.
Provides enterprise-grade logging with structured output, request tracking, and performance metrics.
"""

import sys
import logging
import logging.handlers
import json
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager
from functools import wraps
import structlog
from structlog.stdlib import LoggerFactory
import uuid

from .config import get_settings


def configure_logging():
    """Configure structured logging for the application."""
    settings = get_settings().logging
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.level),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_timestamp,
        add_correlation_id,
    ]
    
    if settings.format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup file logging if configured
    if settings.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.file_path,
            maxBytes=settings.max_file_size,
            backupCount=settings.backup_count
        )
        file_handler.setLevel(getattr(logging, settings.level))
        logging.getLogger().addHandler(file_handler)


def add_timestamp(logger, method_name, event_dict):
    """Add timestamp to log entries."""
    event_dict["timestamp"] = time.time()
    return event_dict


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to log entries if available in context."""
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


@contextmanager
def log_context(**kwargs):
    """Context manager for adding structured context to logs."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.clear_contextvars()


def with_logging(func_name: Optional[str] = None):
    """Decorator to add automatic logging to functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or func.__name__
            
            with log_context(function=name, operation_id=str(uuid.uuid4())):
                start_time = time.time()
                logger.info("Function started", function=name)
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(
                        "Function completed successfully", 
                        function=name, 
                        duration=duration
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        "Function failed",
                        function=name,
                        duration=duration,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or func.__name__
            
            with log_context(function=name, operation_id=str(uuid.uuid4())):
                start_time = time.time()
                logger.info("Function started", function=name)
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(
                        "Function completed successfully", 
                        function=name, 
                        duration=duration
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        "Function failed",
                        function=name,
                        duration=duration,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    
    return decorator


class MetricsLogger:
    """Logger for application metrics and performance data."""
    
    def __init__(self):
        self.logger = get_logger("metrics")
    
    def log_request_metrics(
        self, 
        method: str, 
        path: str, 
        status_code: int, 
        duration: float,
        **kwargs
    ):
        """Log HTTP request metrics."""
        self.logger.info(
            "HTTP request completed",
            metric_type="http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            **kwargs
        )
    
    def log_processing_metrics(
        self, 
        operation: str, 
        file_type: str, 
        file_size: int, 
        duration: float,
        success: bool = True,
        **kwargs
    ):
        """Log document processing metrics."""
        self.logger.info(
            "Document processing completed",
            metric_type="processing",
            operation=operation,
            file_type=file_type,
            file_size=file_size,
            duration=duration,
            success=success,
            **kwargs
        )
    
    def log_azure_api_metrics(
        self, 
        service: str, 
        operation: str, 
        duration: float, 
        tokens_used: Optional[int] = None,
        success: bool = True,
        **kwargs
    ):
        """Log Azure API call metrics."""
        self.logger.info(
            "Azure API call completed",
            metric_type="azure_api",
            service=service,
            operation=operation,
            duration=duration,
            tokens_used=tokens_used,
            success=success,
            **kwargs
        )
    
    def log_search_metrics(
        self, 
        query_type: str, 
        results_count: int, 
        duration: float,
        **kwargs
    ):
        """Log search operation metrics."""
        self.logger.info(
            "Search operation completed",
            metric_type="search",
            query_type=query_type,
            results_count=results_count,
            duration=duration,
            **kwargs
        )


class AuditLogger:
    """Logger for security and audit events."""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_authentication(
        self, 
        user_id: Optional[str], 
        ip_address: str, 
        success: bool,
        **kwargs
    ):
        """Log authentication attempts."""
        self.logger.info(
            "Authentication attempt",
            event_type="authentication",
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            **kwargs
        )
    
    def log_data_access(
        self, 
        user_id: Optional[str], 
        resource: str, 
        action: str,
        **kwargs
    ):
        """Log data access events."""
        self.logger.info(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def log_security_event(
        self, 
        event_type: str, 
        severity: str, 
        description: str,
        **kwargs
    ):
        """Log security events."""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            severity=severity,
            description=description,
            **kwargs
        )


# Global logger instances
metrics_logger = MetricsLogger()
audit_logger = AuditLogger()


def setup_logging():
    """Initialize logging configuration."""
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Logging configuration initialized")


# Initialize logging when module is imported
setup_logging()