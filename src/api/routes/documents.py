"""
Document upload and processing endpoints.
"""

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from datetime import datetime

from ...services.rag_service import RAGService
from ...services.vector_store import VectorStoreService
from ...services.document_processor import DocumentProcessor
from ...models.schemas import ContentType, ProcessingStatus
from ...core.exceptions import FileProcessingError, ValidationError
from ..main import get_rag_service, get_vector_store


router = APIRouter()


@router.post("/upload", summary="Upload and process document")
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Upload and process a document.
    
    Args:
        file: Document file to upload
        metadata: Optional metadata as JSON string
        
    Returns:
        Upload and processing results
    """
    try:
        # Validate file
        if not file.filename:
            raise ValidationError("Filename is required")
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise ValidationError("File is empty")
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            import json
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format for metadata")
        
        # Add upload metadata
        doc_metadata.update({
            "uploaded_at": datetime.utcnow().isoformat(),
            "original_filename": file.filename,
            "content_type": file.content_type or "application/octet-stream",
            "file_size": len(file_content)
        })
        
        # Process the document
        result = await rag_service.ingest_document(
            file_data=file_content,
            filename=file.filename,
            metadata=doc_metadata
        )
        
        return {
            "success": result["success"],
            "message": "Document uploaded and processed successfully" if result["success"] else "Document processing failed",
            "data": result
        }
        
    except (FileProcessingError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/upload/batch", summary="Upload and process multiple documents")
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Upload and process multiple documents in batch.
    
    Args:
        files: List of document files to upload
        
    Returns:
        Batch upload and processing results
    """
    try:
        if not files:
            raise ValidationError("No files provided")
        
        if len(files) > 10:  # Limit batch size
            raise ValidationError("Maximum 10 files allowed per batch")
        
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                if not file.filename:
                    results.append({
                        "filename": "unknown",
                        "success": False,
                        "error": "Missing filename"
                    })
                    failed += 1
                    continue
                
                # Read file content
                file_content = await file.read()
                
                if not file_content:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Empty file"
                    })
                    failed += 1
                    continue
                
                # Process the document
                metadata = {
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "original_filename": file.filename,
                    "content_type": file.content_type or "application/octet-stream",
                    "file_size": len(file_content),
                    "batch_upload": True
                }
                
                result = await rag_service.ingest_document(
                    file_data=file_content,
                    filename=file.filename,
                    metadata=metadata
                )
                
                results.append({
                    "filename": file.filename,
                    "success": result["success"],
                    "chunks_processed": result.get("chunks_processed", 0),
                    "chunks_indexed": result.get("chunks_indexed", 0),
                    "error": result.get("errors")
                })
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                failed += 1
        
        return {
            "success": failed == 0,
            "message": f"Batch processing completed: {successful} successful, {failed} failed",
            "summary": {
                "total_files": len(files),
                "successful": successful,
                "failed": failed
            },
            "results": results
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{source_id}", summary="Delete document")
async def delete_document(
    source_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Delete a document and all its chunks.
    
    Args:
        source_id: Source ID of the document to delete
        
    Returns:
        Deletion results
    """
    try:
        if not source_id.strip():
            raise ValidationError("Source ID is required")
        
        result = await rag_service.delete_document(source_id)
        
        return {
            "success": result["success"],
            "message": f"Document {source_id} deleted successfully" if result["success"] else f"Failed to delete document {source_id}",
            "data": result
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/supported-types", summary="Get supported file types")
async def get_supported_file_types(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get list of supported file types and formats.
    
    Returns:
        Supported file types and their descriptions
    """
    try:
        supported_types = await rag_service.document_processor.get_supported_types()
        
        return {
            "success": True,
            "message": "Supported file types retrieved successfully",
            "data": {
                "supported_types": supported_types,
                "descriptions": {
                    "text": "Plain text files and text-based documents",
                    "document": "Structured documents (PDF, DOCX, PPTX, Excel)",
                    "image": "Image files with OCR and visual analysis capabilities"
                },
                "max_file_size_mb": rag_service.document_processor.max_file_size // (1024 * 1024)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/validate", summary="Validate file before upload")
async def validate_file(
    filename: str = Query(..., description="Filename to validate"),
    file_size: int = Query(..., description="File size in bytes"),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Validate if a file can be processed before upload.
    
    Args:
        filename: Filename to validate
        file_size: File size in bytes
        
    Returns:
        Validation results
    """
    try:
        validation_result = await rag_service.document_processor.validate_file(filename, file_size)
        
        return {
            "success": True,
            "message": "File validation completed",
            "data": validation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stats", summary="Get document statistics")
async def get_document_statistics(
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Get statistics about stored documents.
    
    Returns:
        Document statistics and metrics
    """
    try:
        stats = await vector_store.get_index_statistics()
        
        return {
            "success": True,
            "message": "Document statistics retrieved successfully",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/search", summary="Search documents by metadata")
async def search_documents(
    query: str = Query(..., description="Search query"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    source_id: Optional[str] = Query(None, description="Filter by source ID"),
    top_k: int = Query(10, description="Number of results to return", ge=1, le=100),
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Search documents by content and metadata.
    
    Args:
        query: Search query text
        content_type: Optional content type filter
        source_id: Optional source ID filter
        top_k: Number of results to return
        
    Returns:
        Search results
    """
    try:
        # Build filters
        filters = []
        if content_type:
            filters.append(f"content_type eq '{content_type}'")
        if source_id:
            filters.append(f"source_id eq '{source_id}'")
        
        filter_string = " and ".join(filters) if filters else None
        
        # Perform search
        results = await vector_store.search_text(
            query_text=query,
            top_k=top_k,
            filters=filter_string,
            include_metadata=True
        )
        
        return {
            "success": True,
            "message": f"Found {len(results)} results",
            "data": {
                "query": query,
                "filters": {
                    "content_type": content_type,
                    "source_id": source_id
                },
                "results": results,
                "total_found": len(results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{source_id}/chunks", summary="Get document chunks")
async def get_document_chunks(
    source_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Get all chunks for a specific document.
    
    Args:
        source_id: Source ID of the document
        
    Returns:
        Document chunks
    """
    try:
        if not source_id.strip():
            raise ValidationError("Source ID is required")
        
        # Search for all chunks with the source ID
        results = await vector_store.search_text(
            query_text="*",
            filters=f"source_id eq '{source_id}'",
            top_k=1000,  # Assume max 1000 chunks per document
            include_metadata=True
        )
        
        # Sort by chunk index
        sorted_results = sorted(results, key=lambda x: x.get("chunk_index", 0))
        
        return {
            "success": True,
            "message": f"Found {len(sorted_results)} chunks for document {source_id}",
            "data": {
                "source_id": source_id,
                "chunk_count": len(sorted_results),
                "chunks": sorted_results
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")