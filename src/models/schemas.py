"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Supported content types for multimodal processing."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class ProcessingStatus(str, Enum):
    """Status of content processing jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchType(str, Enum):
    """Search types supported by the RAG service."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# Base models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Content models
class ContentChunk(BaseModel):
    """Represents a chunk of processed content."""
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The actual content text")
    content_type: ContentType = Field(..., description="Type of content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    source_id: str = Field(..., description="ID of the source document/image")
    chunk_index: int = Field(..., description="Index of this chunk within the source")


class ProcessedContent(BaseModel):
    """Represents processed content ready for indexing."""
    id: str = Field(..., description="Unique identifier for the content")
    title: str = Field(..., description="Title or name of the content")
    content_type: ContentType = Field(..., description="Type of content")
    chunks: List[ContentChunk] = Field(..., description="List of content chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_size: Optional[int] = Field(None, description="Size of original file in bytes")
    file_path: Optional[str] = Field(None, description="Path to original file")


# Request models
class IngestContentRequest(BaseModel):
    """Request model for content ingestion."""
    title: Optional[str] = Field(None, description="Title for the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    content_type: Optional[ContentType] = Field(None, description="Override content type detection")
    chunk_size: Optional[int] = Field(None, description="Override default chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Override default chunk overlap")


class TextIngestRequest(IngestContentRequest):
    """Request model for text content ingestion."""
    text: str = Field(..., description="Text content to ingest", min_length=1)


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="Query text", min_length=1)
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    temperature: Optional[float] = Field(None, description="Override temperature for generation", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Override max tokens for generation", ge=1, le=8000)


# Response models
class SearchResult(BaseModel):
    """Individual search result."""
    id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Relevant content")
    score: float = Field(..., description="Relevance score")
    source_id: str = Field(..., description="Source document ID")
    content_type: ContentType = Field(..., description="Type of content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")


class QueryResponse(BaseResponse):
    """Response model for RAG queries."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SearchResult] = Field(..., description="Source documents used")
    search_type: SearchType = Field(..., description="Search type used")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for generation")


class IngestJob(BaseModel):
    """Represents a content ingestion job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: ProcessingStatus = Field(..., description="Current status")
    content_type: ContentType = Field(..., description="Type of content being processed")
    title: str = Field(..., description="Title of the content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: float = Field(default=0.0, description="Progress percentage", ge=0.0, le=100.0)
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks to process")


class IngestResponse(BaseResponse):
    """Response model for content ingestion."""
    job: IngestJob = Field(..., description="Ingestion job details")


class JobStatusResponse(BaseResponse):
    """Response model for job status queries."""
    job: IngestJob = Field(..., description="Job status details")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content_type: ContentType = Field(..., description="Content type")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class ListDocumentsResponse(BaseResponse):
    """Response model for listing documents."""
    documents: List[DocumentInfo] = Field(..., description="List of indexed documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class HealthStatus(BaseModel):
    """Health check status model."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseResponse):
    """Health check response model."""
    overall_status: str = Field(..., description="Overall system status")
    services: List[HealthStatus] = Field(..., description="Individual service statuses")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")


# Validation models
class FileUploadLimits(BaseModel):
    """File upload limits and validation."""
    max_size_mb: int = Field(default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default=["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff", "gif"],
        description="Allowed file extensions"
    )
    
    @validator('allowed_extensions')
    def validate_extensions(cls, v):
        """Ensure extensions are lowercase."""
        return [ext.lower() for ext in v]


class MetricsResponse(BaseResponse):
    """Response model for system metrics."""
    total_documents: int = Field(..., description="Total number of indexed documents")
    total_chunks: int = Field(..., description="Total number of content chunks")
    active_jobs: int = Field(..., description="Number of active processing jobs")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    avg_query_time: float = Field(..., description="Average query processing time")
    system_uptime: float = Field(..., description="System uptime in seconds")