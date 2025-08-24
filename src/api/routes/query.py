"""
Query and search endpoints for the RAG service.
"""

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import datetime

from ...services.rag_service import RAGService
from ...models.schemas import SearchType
from ...core.exceptions import SearchError, ValidationError
from ..main import get_rag_service


router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for text queries."""
    query: str = Field(..., description="Query text", min_length=1, max_length=2000)
    search_type: SearchType = Field(SearchType.HYBRID, description="Type of search to perform")
    top_k: int = Field(5, description="Number of context chunks to retrieve", ge=1, le=20)
    filters: Optional[str] = Field(None, description="Search filters (OData format)")
    include_sources: bool = Field(True, description="Whether to include source information")
    temperature: Optional[float] = Field(None, description="Temperature for response generation", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response", ge=1, le=4000)
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")


class MultimodalQueryRequest(BaseModel):
    """Request model for multimodal queries."""
    text_query: Optional[str] = Field(None, description="Text component of the query", max_length=2000)
    search_type: SearchType = Field(SearchType.HYBRID, description="Type of search to perform")
    top_k: int = Field(5, description="Number of context chunks to retrieve", ge=1, le=20)
    filters: Optional[str] = Field(None, description="Search filters (OData format)")
    include_sources: bool = Field(True, description="Whether to include source information")


@router.post("/text", summary="Process text query")
async def query_text(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Process a text-based query using the RAG system.
    
    Args:
        request: Query request parameters
        
    Returns:
        Query response with generated answer and sources
    """
    try:
        result = await rag_service.process_query(
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k,
            filters=request.filters,
            include_sources=request.include_sources,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        
        return {
            "success": True,
            "message": "Query processed successfully",
            "data": result
        }
        
    except (SearchError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/text", summary="Simple text query (GET)")
async def query_text_simple(
    q: str = Query(..., description="Query text", min_length=1, max_length=2000),
    search_type: SearchType = Query(SearchType.HYBRID, description="Type of search to perform"),
    top_k: int = Query(5, description="Number of context chunks to retrieve", ge=1, le=20),
    include_sources: bool = Query(True, description="Whether to include source information"),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Simple text query using GET request.
    
    Args:
        q: Query text
        search_type: Type of search to perform
        top_k: Number of context chunks to retrieve
        include_sources: Whether to include source information
        
    Returns:
        Query response with generated answer and sources
    """
    try:
        result = await rag_service.process_query(
            query=q,
            search_type=search_type,
            top_k=top_k,
            include_sources=include_sources
        )
        
        return {
            "success": True,
            "message": "Query processed successfully",
            "data": result
        }
        
    except (SearchError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/multimodal", summary="Process multimodal query")
async def query_multimodal(
    text_query: Optional[str] = Form(None),
    search_type: SearchType = Form(SearchType.HYBRID),
    top_k: int = Form(5),
    filters: Optional[str] = Form(None),
    include_sources: bool = Form(True),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Process a multimodal query combining text, image, and/or audio.
    
    Args:
        text_query: Text component of the query
        search_type: Type of search to perform
        top_k: Number of context chunks to retrieve
        filters: Search filters
        include_sources: Whether to include source information
        image_file: Optional image file
        audio_file: Optional audio file
        
    Returns:
        Query response with generated answer and sources
    """
    try:
        # Validate that at least one query component is provided
        if not any([text_query, image_file, audio_file]):
            raise ValidationError("At least one query component (text, image, or audio) must be provided")
        
        # Read file data if provided
        image_data = None
        audio_data = None
        
        if image_file:
            image_data = await image_file.read()
            if not image_data:
                raise ValidationError("Image file is empty")
        
        if audio_file:
            audio_data = await audio_file.read()
            if not audio_data:
                raise ValidationError("Audio file is empty")
        
        # Process the multimodal query
        result = await rag_service.process_multimodal_query(
            text_query=text_query,
            image_data=image_data,
            audio_data=audio_data,
            search_type=search_type,
            top_k=top_k,
            filters=filters,
            include_sources=include_sources
        )
        
        return {
            "success": True,
            "message": "Multimodal query processed successfully",
            "data": result
        }
        
    except (SearchError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/similar", summary="Find similar content")
async def find_similar_content(
    content: str = Query(..., description="Content to find similar items for", min_length=1),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    top_k: int = Query(10, description="Number of similar items to return", ge=1, le=50),
    similarity_threshold: float = Query(0.7, description="Minimum similarity score", ge=0.0, le=1.0),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Find content similar to the provided text.
    
    Args:
        content: Content to find similar items for
        content_type: Optional content type filter
        top_k: Number of similar items to return
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of similar content items
    """
    try:
        # Generate embedding for the content
        content_embedding = await rag_service.azure_ai.generate_text_embedding(content)
        
        # Build filters
        filters = None
        if content_type:
            filters = f"content_type eq '{content_type}'"
        
        # Search for similar content
        results = await rag_service.vector_store.search_vector(
            query_vector=content_embedding,
            top_k=top_k,
            filters=filters,
            include_metadata=True
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.get("score", 0) >= similarity_threshold
        ]
        
        return {
            "success": True,
            "message": f"Found {len(filtered_results)} similar content items",
            "data": {
                "query_content": content,
                "similarity_threshold": similarity_threshold,
                "results": filtered_results,
                "total_found": len(filtered_results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/context", summary="Get context for query without generation")
async def get_query_context(
    query: str = Query(..., description="Query to get context for", min_length=1, max_length=2000),
    search_type: SearchType = Query(SearchType.HYBRID, description="Type of search to perform"),
    top_k: int = Query(5, description="Number of context chunks to retrieve", ge=1, le=20),
    filters: Optional[str] = Query(None, description="Search filters (OData format)"),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get relevant context for a query without generating a response.
    
    Args:
        query: Query to get context for
        search_type: Type of search to perform
        top_k: Number of context chunks to retrieve
        filters: Search filters
        
    Returns:
        Retrieved context chunks
    """
    try:
        # Enhanced query processing
        processed_query = query.strip()
        
        # Retrieve context using the internal method
        if search_type == SearchType.VECTOR:
            query_embedding = await rag_service.azure_ai.generate_text_embedding(processed_query)
            context_chunks = await rag_service.vector_store.search_vector(
                query_embedding, top_k, filters
            )
        elif search_type == SearchType.KEYWORD:
            context_chunks = await rag_service.vector_store.search_text(
                processed_query, top_k, filters
            )
        elif search_type == SearchType.HYBRID:
            query_embedding = await rag_service.azure_ai.generate_text_embedding(processed_query)
            context_chunks = await rag_service.vector_store.search_hybrid(
                processed_query, query_embedding, top_k, filters
            )
        else:
            raise ValidationError(f"Unsupported search type: {search_type}")
        
        return {
            "success": True,
            "message": f"Retrieved {len(context_chunks)} context chunks",
            "data": {
                "query": query,
                "search_type": search_type.value,
                "context_chunks": context_chunks,
                "total_retrieved": len(context_chunks)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except (SearchError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze-image", summary="Analyze image content")
async def analyze_image(
    image_file: UploadFile = File(...),
    include_ocr: bool = Query(True, description="Whether to include OCR text extraction"),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Analyze image content using Azure Computer Vision.
    
    Args:
        image_file: Image file to analyze
        include_ocr: Whether to include OCR text extraction
        
    Returns:
        Image analysis results
    """
    try:
        # Validate file
        if not image_file.filename:
            raise ValidationError("Filename is required")
        
        # Check if it's an image file
        if not image_file.content_type or not image_file.content_type.startswith('image/'):
            raise ValidationError("File must be an image")
        
        # Read image data
        image_data = await image_file.read()
        
        if not image_data:
            raise ValidationError("Image file is empty")
        
        # Analyze image
        vision_analysis = await rag_service.azure_ai.analyze_image(image_data)
        
        result = {
            "filename": image_file.filename,
            "content_type": image_file.content_type,
            "file_size": len(image_data),
            "vision_analysis": vision_analysis
        }
        
        # Include OCR if requested
        if include_ocr:
            ocr_results = await rag_service.azure_ai.extract_text_from_image(image_data)
            result["ocr_results"] = ocr_results
        
        return {
            "success": True,
            "message": "Image analysis completed successfully",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/transcribe-audio", summary="Transcribe audio content")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Transcribe audio content using Azure Speech Services.
    
    Args:
        audio_file: Audio file to transcribe
        
    Returns:
        Audio transcription results
    """
    try:
        # Validate file
        if not audio_file.filename:
            raise ValidationError("Filename is required")
        
        # Check if it's an audio file
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise ValidationError("File must be an audio file")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        if not audio_data:
            raise ValidationError("Audio file is empty")
        
        # Save to temporary file for speech services
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            transcription = await rag_service.azure_ai.speech_to_text(temp_file_path)
            
            result = {
                "filename": audio_file.filename,
                "content_type": audio_file.content_type,
                "file_size": len(audio_data),
                "transcription": transcription
            }
            
            return {
                "success": True,
                "message": "Audio transcription completed successfully",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/history", summary="Get query history")
async def get_query_history(
    limit: int = Query(10, description="Number of recent queries to return", ge=1, le=100)
) -> Dict[str, Any]:
    """
    Get recent query history (placeholder - would need database implementation).
    
    Args:
        limit: Number of recent queries to return
        
    Returns:
        Query history
    """
    # This is a placeholder - in a real implementation, you would store and retrieve query history
    return {
        "success": True,
        "message": "Query history feature not implemented yet",
        "data": {
            "history": [],
            "note": "Query history tracking would require a database implementation"
        },
        "timestamp": datetime.utcnow().isoformat()
    }