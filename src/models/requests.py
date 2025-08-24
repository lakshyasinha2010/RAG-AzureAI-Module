"""
Request models for the multimodal RAG service API.
Defines Pydantic models for all API request payloads with validation.
"""

from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
from datetime import datetime


class FileType(str, Enum):
    """Supported file types for processing."""
    
    # Text documents
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    
    # Images
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    
    # Audio
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    
    # Video
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WMV = "wmv"


class ProcessingMode(str, Enum):
    """Processing mode for documents."""
    
    SYNC = "sync"  # Synchronous processing
    ASYNC = "async"  # Asynchronous processing


class QueryType(str, Enum):
    """Types of queries supported."""
    
    TEXT = "text"
    MULTIMODAL = "multimodal"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    category: Optional[str] = Field(None, description="Document category")
    language: Optional[str] = Field("en", description="Document language")
    source: Optional[str] = Field(None, description="Document source")
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Custom metadata fields"
    )


class DocumentUploadRequest(BaseModel):
    """Request for uploading a document."""
    
    file_name: str = Field(..., description="Name of the file")
    file_type: FileType = Field(..., description="Type of the file")
    processing_mode: ProcessingMode = Field(
        ProcessingMode.SYNC, 
        description="Processing mode"
    )
    metadata: Optional[DocumentMetadata] = Field(
        None, 
        description="Document metadata"
    )
    extract_tables: bool = Field(
        True, 
        description="Whether to extract tables from documents"
    )
    extract_images: bool = Field(
        True, 
        description="Whether to extract images from documents"
    )
    generate_summary: bool = Field(
        False, 
        description="Whether to generate document summary"
    )
    chunk_strategy: Optional[str] = Field(
        "semantic", 
        description="Chunking strategy: 'fixed', 'semantic', 'document_structure'"
    )

    @validator('file_name')
    def validate_file_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('File name cannot be empty')
        if len(v) > 255:
            raise ValueError('File name too long')
        return v.strip()


class BatchDocumentUploadRequest(BaseModel):
    """Request for uploading multiple documents."""
    
    documents: List[DocumentUploadRequest] = Field(
        ..., 
        description="List of documents to upload",
        min_items=1,
        max_items=100
    )
    processing_mode: ProcessingMode = Field(
        ProcessingMode.ASYNC, 
        description="Processing mode for the batch"
    )
    notification_webhook: Optional[HttpUrl] = Field(
        None,
        description="Webhook URL for processing notifications"
    )


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    
    query: str = Field(..., description="The query text", min_length=1, max_length=2000)
    query_type: QueryType = Field(
        QueryType.HYBRID, 
        description="Type of query to perform"
    )
    max_results: int = Field(
        10, 
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    include_sources: bool = Field(
        True, 
        description="Whether to include source information"
    )
    include_metadata: bool = Field(
        False, 
        description="Whether to include document metadata"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Filters to apply to the search"
    )
    similarity_threshold: float = Field(
        0.7, 
        description="Minimum similarity threshold",
        ge=0.0,
        le=1.0
    )
    response_language: Optional[str] = Field(
        "en", 
        description="Preferred response language"
    )


class MultimodalQueryRequest(QueryRequest):
    """Request for multimodal queries with additional media."""
    
    image_urls: Optional[List[HttpUrl]] = Field(
        None, 
        description="URLs of images to include in the query"
    )
    audio_urls: Optional[List[HttpUrl]] = Field(
        None, 
        description="URLs of audio files to include in the query"
    )
    video_urls: Optional[List[HttpUrl]] = Field(
        None, 
        description="URLs of video files to include in the query"
    )


class DocumentDeleteRequest(BaseModel):
    """Request for deleting documents."""
    
    document_ids: List[str] = Field(
        ..., 
        description="List of document IDs to delete",
        min_items=1,
        max_items=100
    )
    delete_from_storage: bool = Field(
        True, 
        description="Whether to delete from storage as well"
    )


class IndexManagementRequest(BaseModel):
    """Request for index management operations."""
    
    operation: str = Field(
        ..., 
        description="Operation to perform: 'create', 'delete', 'rebuild', 'optimize'"
    )
    index_name: Optional[str] = Field(
        None, 
        description="Name of the index (for create operation)"
    )
    force: bool = Field(
        False, 
        description="Force the operation even if data loss may occur"
    )

    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ['create', 'delete', 'rebuild', 'optimize']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {valid_operations}')
        return v


class HealthCheckRequest(BaseModel):
    """Request for health check with specific service checks."""
    
    include_dependencies: bool = Field(
        True, 
        description="Whether to check dependent services"
    )
    check_azure_services: bool = Field(
        True, 
        description="Whether to check Azure services connectivity"
    )
    check_vector_store: bool = Field(
        True, 
        description="Whether to check vector store connectivity"
    )


class ConfigurationUpdateRequest(BaseModel):
    """Request for updating service configuration."""
    
    max_file_size: Optional[int] = Field(
        None, 
        description="Maximum file size in bytes",
        gt=0
    )
    chunk_size: Optional[int] = Field(
        None, 
        description="Text chunk size",
        gt=0,
        le=2000
    )
    chunk_overlap: Optional[int] = Field(
        None, 
        description="Text chunk overlap",
        ge=0,
        le=500
    )
    rate_limit_requests: Optional[int] = Field(
        None, 
        description="Rate limit requests per window",
        gt=0
    )
    rate_limit_window: Optional[int] = Field(
        None, 
        description="Rate limit window in seconds",
        gt=0
    )