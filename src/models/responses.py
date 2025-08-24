"""
Response models for the multimodal RAG service API.
Defines Pydantic models for all API response payloads.
"""

from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class ProcessingStatus(str, Enum):
    """Status of document processing."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceStatus(str, Enum):
    """Status of service health checks."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DocumentChunk(BaseModel):
    """Represents a processed document chunk."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., description="Index of the chunk in the document")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    start_offset: Optional[int] = Field(None, description="Start character offset")
    end_offset: Optional[int] = Field(None, description="End character offset")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class ExtractedContent(BaseModel):
    """Content extracted from a document."""
    
    text: Optional[str] = Field(None, description="Extracted text content")
    images: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Extracted images with metadata"
    )
    tables: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Extracted tables with data"
    )
    audio_transcripts: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Audio transcription results"
    )
    video_analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Video analysis results"
    )
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Extracted entities"
    )
    key_phrases: List[str] = Field(
        default_factory=list, 
        description="Extracted key phrases"
    )


class DocumentProcessingResult(BaseModel):
    """Result of document processing."""
    
    document_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    status: ProcessingStatus = Field(..., description="Processing status")
    processed_at: datetime = Field(..., description="Processing completion time")
    chunks: List[DocumentChunk] = Field(
        default_factory=list, 
        description="Document chunks"
    )
    extracted_content: Optional[ExtractedContent] = Field(
        None, 
        description="Extracted content"
    )
    summary: Optional[str] = Field(None, description="Document summary")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Document metadata"
    )
    processing_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Processing performance metrics"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")


class DocumentUploadResponse(BaseModel):
    """Response for document upload operations."""
    
    document_id: str = Field(..., description="Unique document identifier")
    status: ProcessingStatus = Field(..., description="Initial processing status")
    upload_url: Optional[str] = Field(None, description="URL for file upload")
    processing_job_id: Optional[str] = Field(
        None, 
        description="Job ID for async processing"
    )
    estimated_completion_time: Optional[datetime] = Field(
        None, 
        description="Estimated completion time"
    )
    message: str = Field(..., description="Status message")


class BatchUploadResponse(BaseModel):
    """Response for batch document upload operations."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    total_documents: int = Field(..., description="Total number of documents")
    accepted_documents: int = Field(..., description="Number of accepted documents")
    rejected_documents: int = Field(..., description="Number of rejected documents")
    processing_job_ids: List[str] = Field(
        default_factory=list, 
        description="List of processing job IDs"
    )
    status: ProcessingStatus = Field(..., description="Batch processing status")
    estimated_completion_time: Optional[datetime] = Field(
        None, 
        description="Estimated completion time"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Relevant content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Document metadata"
    )
    highlights: List[str] = Field(
        default_factory=list, 
        description="Highlighted text snippets"
    )
    source_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="Source information"
    )


class QueryResponse(BaseModel):
    """Response for query operations."""
    
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    results: List[SearchResult] = Field(
        default_factory=list, 
        description="Search results"
    )
    total_results: int = Field(..., description="Total number of results found")
    processing_time: float = Field(..., description="Query processing time in seconds")
    model_info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Information about models used"
    )
    confidence_score: Optional[float] = Field(
        None, 
        description="Confidence score for the answer"
    )
    sources_used: List[str] = Field(
        default_factory=list, 
        description="List of source document IDs used"
    )


class ServiceHealthResponse(BaseModel):
    """Response for service health checks."""
    
    status: ServiceStatus = Field(..., description="Overall service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, ServiceStatus] = Field(
        default_factory=dict, 
        description="Status of dependent services"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Service metrics"
    )
    checks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Detailed health check results"
    )


class IndexStatsResponse(BaseModel):
    """Response for index statistics."""
    
    index_name: str = Field(..., description="Index name")
    document_count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Number of chunks")
    size_bytes: int = Field(..., description="Index size in bytes")
    last_updated: datetime = Field(..., description="Last update timestamp")
    schema_version: str = Field(..., description="Index schema version")
    fields: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Index field definitions"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional error details"
    )
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ProcessingJobStatus(BaseModel):
    """Status of a processing job."""
    
    job_id: str = Field(..., description="Job identifier")
    status: ProcessingStatus = Field(..., description="Job status")
    progress: float = Field(..., description="Job progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[DocumentProcessingResult] = Field(
        None, 
        description="Processing result if completed"
    )


class ConfigurationResponse(BaseModel):
    """Response for configuration queries."""
    
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    supported_file_types: List[str] = Field(
        default_factory=list, 
        description="Supported file types"
    )
    processing_limits: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Processing limits and quotas"
    )
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict, 
        description="Feature availability flags"
    )
    version_info: Dict[str, str] = Field(
        default_factory=dict, 
        description="Version information"
    )


class MetricsResponse(BaseModel):
    """Response for service metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    requests_total: int = Field(..., description="Total requests processed")
    requests_per_minute: float = Field(..., description="Requests per minute")
    average_response_time: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    documents_processed: int = Field(..., description="Total documents processed")
    storage_used: int = Field(..., description="Storage used in bytes")
    active_connections: int = Field(..., description="Active connections")
    resource_usage: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Resource usage statistics"
    )


class SuccessResponse(BaseModel):
    """Generic success response."""
    
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    timestamp: datetime = Field(..., description="Response timestamp")