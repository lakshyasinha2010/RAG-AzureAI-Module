"""Data models for the multimodal RAG service."""

from .requests import (
    FileType,
    ProcessingMode,
    QueryType,
    DocumentMetadata,
    DocumentUploadRequest,
    BatchDocumentUploadRequest,
    QueryRequest,
    MultimodalQueryRequest,
    DocumentDeleteRequest,
    IndexManagementRequest,
    HealthCheckRequest,
    ConfigurationUpdateRequest
)

from .responses import (
    ProcessingStatus,
    ServiceStatus,
    DocumentChunk,
    ExtractedContent,
    DocumentProcessingResult,
    DocumentUploadResponse,
    BatchUploadResponse,
    SearchResult,
    QueryResponse,
    ServiceHealthResponse,
    IndexStatsResponse,
    ErrorResponse,
    ProcessingJobStatus,
    ConfigurationResponse,
    MetricsResponse,
    SuccessResponse
)

__all__ = [
    # Enums
    "FileType",
    "ProcessingMode", 
    "QueryType",
    "ProcessingStatus",
    "ServiceStatus",
    
    # Request Models
    "DocumentMetadata",
    "DocumentUploadRequest",
    "BatchDocumentUploadRequest", 
    "QueryRequest",
    "MultimodalQueryRequest",
    "DocumentDeleteRequest",
    "IndexManagementRequest",
    "HealthCheckRequest",
    "ConfigurationUpdateRequest",
    
    # Response Models
    "DocumentChunk",
    "ExtractedContent",
    "DocumentProcessingResult",
    "DocumentUploadResponse",
    "BatchUploadResponse",
    "SearchResult", 
    "QueryResponse",
    "ServiceHealthResponse",
    "IndexStatsResponse",
    "ErrorResponse",
    "ProcessingJobStatus",
    "ConfigurationResponse",
    "MetricsResponse",
    "SuccessResponse"
]