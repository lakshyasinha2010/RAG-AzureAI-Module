"""
Custom exceptions for the multimodal RAG service.
Provides structured error handling with proper categorization and context.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException
from fastapi import status


class BaseRAGException(Exception):
    """Base exception for all RAG service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self.message)


class ConfigurationError(BaseRAGException):
    """Raised when there are configuration issues."""
    pass


class AuthenticationError(BaseRAGException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(BaseRAGException):
    """Raised when authorization fails."""
    pass


class ValidationError(BaseRAGException):
    """Raised when input validation fails."""
    pass


class FileProcessingError(BaseRAGException):
    """Raised when file processing fails."""
    pass


class DocumentProcessingError(BaseRAGException):
    """Raised when document processing fails."""
    pass


class AzureServiceError(BaseRAGException):
    """Raised when Azure service calls fail."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        operation: str,
        azure_error_code: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.operation = operation
        self.azure_error_code = azure_error_code


class VectorStoreError(BaseRAGException):
    """Raised when vector store operations fail."""
    pass


class SearchError(BaseRAGException):
    """Raised when search operations fail."""
    pass


class EmbeddingError(BaseRAGException):
    """Raised when embedding generation fails."""
    pass


class RateLimitError(BaseRAGException):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ResourceNotFoundError(BaseRAGException):
    """Raised when a requested resource is not found."""
    pass


class ResourceExistsError(BaseRAGException):
    """Raised when trying to create a resource that already exists."""
    pass


class ProcessingTimeoutError(BaseRAGException):
    """Raised when processing operations timeout."""
    pass


class UnsupportedFileTypeError(BaseRAGException):
    """Raised when an unsupported file type is encountered."""
    
    def __init__(
        self,
        message: str,
        file_type: str,
        supported_types: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_type = file_type
        self.supported_types = supported_types or []


class FileSizeError(BaseRAGException):
    """Raised when file size limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        file_size: int,
        max_size: int,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_size = file_size
        self.max_size = max_size


class ContentExtractionError(BaseRAGException):
    """Raised when content extraction from files fails."""
    pass


class IndexingError(BaseRAGException):
    """Raised when document indexing fails."""
    pass


class QueryError(BaseRAGException):
    """Raised when query processing fails."""
    pass


class ResponseGenerationError(BaseRAGException):
    """Raised when response generation fails."""
    pass


# HTTP Exception mappings for FastAPI
def create_http_exception(exc: BaseRAGException) -> HTTPException:
    """Convert a RAG exception to an HTTP exception."""
    
    error_mappings = {
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
        ResourceExistsError: status.HTTP_409_CONFLICT,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
        FileSizeError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        UnsupportedFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ProcessingTimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
    }
    
    status_code = error_mappings.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    detail = {
        "error_code": exc.error_code,
        "message": exc.message,
        "context": exc.context
    }
    
    # Add specific fields for certain exception types
    if isinstance(exc, RateLimitError) and exc.retry_after:
        detail["retry_after"] = exc.retry_after
    
    if isinstance(exc, UnsupportedFileTypeError):
        detail.update({
            "file_type": exc.file_type,
            "supported_types": exc.supported_types
        })
    
    if isinstance(exc, FileSizeError):
        detail.update({
            "file_size": exc.file_size,
            "max_size": exc.max_size
        })
    
    headers = {}
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers=headers if headers else None
    )


class ExceptionHandler:
    """Centralized exception handling utilities."""
    
    @staticmethod
    def handle_azure_exception(
        exc: Exception,
        service_name: str,
        operation: str
    ) -> AzureServiceError:
        """Convert Azure SDK exceptions to service exceptions."""
        # Handle specific Azure exception types
        if hasattr(exc, 'status_code'):
            if exc.status_code == 401:
                return AuthenticationError(
                    f"Authentication failed for {service_name}",
                    context={"service": service_name, "operation": operation}
                )
            elif exc.status_code == 403:
                return AuthorizationError(
                    f"Authorization failed for {service_name}",
                    context={"service": service_name, "operation": operation}
                )
            elif exc.status_code == 429:
                return RateLimitError(
                    f"Rate limit exceeded for {service_name}",
                    context={"service": service_name, "operation": operation}
                )
        
        # Generic Azure service error
        return AzureServiceError(
            f"Azure service error: {str(exc)}",
            service_name=service_name,
            operation=operation,
            context={"original_error": str(exc)}
        )
    
    @staticmethod
    def handle_file_exception(
        exc: Exception,
        file_path: str,
        operation: str
    ) -> BaseRAGException:
        """Convert file operation exceptions to service exceptions."""
        if isinstance(exc, FileNotFoundError):
            return ResourceNotFoundError(
                f"File not found: {file_path}",
                context={"file_path": file_path, "operation": operation}
            )
        elif isinstance(exc, PermissionError):
            return AuthorizationError(
                f"Permission denied for file: {file_path}",
                context={"file_path": file_path, "operation": operation}
            )
        elif isinstance(exc, OSError) and exc.errno == 28:  # No space left
            return FileProcessingError(
                "Insufficient disk space for file operation",
                context={"file_path": file_path, "operation": operation}
            )
        
        return FileProcessingError(
            f"File operation failed: {str(exc)}",
            context={"file_path": file_path, "operation": operation, "original_error": str(exc)}
        )
    
    @staticmethod
    def handle_processing_timeout(
        operation: str,
        timeout_seconds: int
    ) -> ProcessingTimeoutError:
        """Create a processing timeout exception."""
        return ProcessingTimeoutError(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            context={"operation": operation, "timeout": timeout_seconds}
        )


# Global exception handler instance
exception_handler = ExceptionHandler()