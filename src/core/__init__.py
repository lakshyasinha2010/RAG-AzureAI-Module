"""Core modules for the multimodal RAG service."""

from .config import settings, get_settings, validate_configuration
from .logging import get_logger, log_context, with_logging, metrics_logger, audit_logger
from .exceptions import (
    BaseRAGException,
    ConfigurationError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    FileProcessingError,
    DocumentProcessingError,
    AzureServiceError,
    VectorStoreError,
    SearchError,
    EmbeddingError,
    RateLimitError,
    ResourceNotFoundError,
    ResourceExistsError,
    ProcessingTimeoutError,
    UnsupportedFileTypeError,
    FileSizeError,
    ContentExtractionError,
    IndexingError,
    QueryError,
    ResponseGenerationError,
    create_http_exception,
    exception_handler
)

__all__ = [
    # Config
    "settings",
    "get_settings", 
    "validate_configuration",
    
    # Logging
    "get_logger",
    "log_context",
    "with_logging",
    "metrics_logger",
    "audit_logger",
    
    # Exceptions
    "BaseRAGException",
    "ConfigurationError", 
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "FileProcessingError",
    "DocumentProcessingError",
    "AzureServiceError",
    "VectorStoreError",
    "SearchError",
    "EmbeddingError",
    "RateLimitError",
    "ResourceNotFoundError",
    "ResourceExistsError",
    "ProcessingTimeoutError",
    "UnsupportedFileTypeError",
    "FileSizeError",
    "ContentExtractionError",
    "IndexingError",
    "QueryError", 
    "ResponseGenerationError",
    "create_http_exception",
    "exception_handler"
]