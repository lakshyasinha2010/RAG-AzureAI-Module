"""
Main RAG orchestration service.
Handles query processing, context retrieval, ranking, and response generation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from ..config import get_settings
from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import SearchError, EmbeddingError, VectorStoreError
from ..models.schemas import ContentType, SearchType, ContentChunk
from .azure_ai_service import AzureAIService
from .vector_store import VectorStoreService
from .document_processor import DocumentProcessor


class RAGService(LoggingMixin):
    """Main RAG orchestration service for multimodal query processing and response generation."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.settings = get_settings()
        
        # Initialize component services
        self.azure_ai = AzureAIService()
        self.vector_store = VectorStoreService()
        self.document_processor = DocumentProcessor()
        
        # RAG configuration
        self.default_top_k = 5
        self.max_context_length = 8000  # Maximum tokens for context
        self.similarity_threshold = 0.7  # Minimum similarity score
        self.max_query_length = 2000  # Maximum query length
    
    @log_function_call
    async def process_query(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        top_k: Optional[int] = None,
        filters: Optional[str] = None,
        include_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query and generate a RAG response.
        
        Args:
            query: User query
            search_type: Type of search to perform
            top_k: Number of context chunks to retrieve
            filters: Search filters
            include_sources: Whether to include source information
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
            system_prompt: Custom system prompt
            
        Returns:
            Complete RAG response with context and sources
        """
        if len(query) > self.max_query_length:
            raise SearchError(
                f"Query too long ({len(query)} chars). Maximum allowed: {self.max_query_length}",
                context={"query_length": len(query)}
            )
        
        if top_k is None:
            top_k = self.default_top_k
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Enhanced query processing
            processed_query = await self._enhance_query(query)
            
            # Step 2: Retrieve relevant context
            context_chunks = await self._retrieve_context(
                processed_query, search_type, top_k, filters
            )
            
            # Step 3: Rank and filter context
            ranked_chunks = await self._rank_and_filter_context(
                processed_query, context_chunks
            )
            
            # Step 4: Generate response
            response = await self._generate_response(
                query, ranked_chunks, system_prompt, temperature, max_tokens
            )
            
            # Step 5: Prepare final result
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "query": query,
                "response": response,
                "context_used": len(ranked_chunks),
                "search_type": search_type.value,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if include_sources:
                result["sources"] = self._extract_source_information(ranked_chunks)
            
            self.logger.info(
                "RAG query processed successfully",
                query_length=len(query),
                context_chunks=len(ranked_chunks),
                processing_time=processing_time,
                search_type=search_type.value
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "RAG query processing failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e)
            )
            raise SearchError(
                f"Failed to process query: {str(e)}",
                context={"query": query, "search_type": search_type.value}
            )
    
    @log_function_call
    async def process_multimodal_query(
        self,
        text_query: Optional[str] = None,
        image_data: Optional[bytes] = None,
        audio_data: Optional[bytes] = None,
        search_type: SearchType = SearchType.HYBRID,
        top_k: Optional[int] = None,
        filters: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a multimodal query combining text, image, and/or audio.
        
        Args:
            text_query: Text component of the query
            image_data: Image data to analyze and query
            audio_data: Audio data to transcribe and query
            search_type: Type of search to perform
            top_k: Number of context chunks to retrieve
            filters: Search filters
            include_sources: Whether to include source information
            
        Returns:
            Complete RAG response with multimodal context
        """
        if not any([text_query, image_data, audio_data]):
            raise SearchError("At least one query component (text, image, or audio) must be provided")
        
        try:
            # Process different modalities
            query_components = []
            
            # Process text query
            if text_query:
                query_components.append(f"Text query: {text_query}")
            
            # Process image query
            if image_data:
                image_description = await self.azure_ai.generate_image_description(image_data)
                image_ocr = await self.azure_ai.extract_text_from_image(image_data)
                
                image_query = f"Image description: {image_description}"
                if image_ocr.get("text"):
                    image_query += f" Image text: {image_ocr['text']}"
                
                query_components.append(image_query)
            
            # Process audio query
            if audio_data:
                # Save audio to temporary file for processing
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                try:
                    transcription = await self.azure_ai.speech_to_text(temp_file_path)
                    query_components.append(f"Audio transcription: {transcription}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            # Combine all query components
            combined_query = " ".join(query_components)
            
            # Process the combined multimodal query
            result = await self.process_query(
                query=combined_query,
                search_type=search_type,
                top_k=top_k,
                filters=filters,
                include_sources=include_sources
            )
            
            # Add multimodal context
            result["multimodal"] = {
                "has_text": text_query is not None,
                "has_image": image_data is not None,
                "has_audio": audio_data is not None,
                "combined_query": combined_query
            }
            
            return result
            
        except Exception as e:
            self.logger.error("Multimodal query processing failed", error=str(e))
            raise SearchError(
                f"Failed to process multimodal query: {str(e)}",
                context={"has_text": text_query is not None, "has_image": image_data is not None, "has_audio": audio_data is not None}
            )
    
    @log_function_call
    async def ingest_document(
        self,
        file_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_data: Raw file data
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Ingestion results
        """
        try:
            # Process the document
            chunks = await self.document_processor.process_file(
                file_data, filename, metadata=metadata
            )
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content chunks were generated from the document",
                    "filename": filename
                }
            
            # Index the chunks in the vector store
            indexing_result = await self.vector_store.index_documents_batch(
                chunks, title=filename, file_path=filename, file_size=len(file_data)
            )
            
            result = {
                "success": indexing_result["success"],
                "filename": filename,
                "chunks_processed": len(chunks),
                "chunks_indexed": indexing_result["indexed"],
                "chunks_failed": indexing_result["failed"],
                "file_size": len(file_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if indexing_result["errors"]:
                result["errors"] = indexing_result["errors"]
            
            self.logger.info(
                "Document ingestion completed",
                filename=filename,
                chunks_processed=len(chunks),
                chunks_indexed=indexing_result["indexed"],
                success=indexing_result["success"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Document ingestion failed", filename=filename, error=str(e))
            raise VectorStoreError(
                f"Failed to ingest document {filename}: {str(e)}",
                context={"filename": filename}
            )
    
    @log_function_call
    async def delete_document(self, source_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its chunks from the RAG system.
        
        Args:
            source_id: Source ID of the document to delete
            
        Returns:
            Deletion results
        """
        try:
            deleted_count = await self.vector_store.delete_documents_by_source(source_id)
            
            result = {
                "success": deleted_count > 0,
                "source_id": source_id,
                "chunks_deleted": deleted_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(
                "Document deletion completed",
                source_id=source_id,
                chunks_deleted=deleted_count
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Document deletion failed", source_id=source_id, error=str(e))
            raise VectorStoreError(
                f"Failed to delete document {source_id}: {str(e)}",
                context={"source_id": source_id}
            )
    
    async def _enhance_query(self, query: str) -> str:
        """
        Enhance the query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
        # For now, return the original query
        # Future enhancements could include:
        # - Query expansion
        # - Spelling correction
        # - Intent detection
        # - Entity recognition
        return query.strip()
    
    async def _retrieve_context(
        self,
        query: str,
        search_type: SearchType,
        top_k: int,
        filters: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: Enhanced query
            search_type: Type of search to perform
            top_k: Number of results to retrieve
            filters: Search filters
            
        Returns:
            List of context chunks
        """
        try:
            if search_type == SearchType.VECTOR:
                # Vector-only search
                query_embedding = await self.azure_ai.generate_text_embedding(query)
                return await self.vector_store.search_vector(
                    query_embedding, top_k, filters
                )
            
            elif search_type == SearchType.KEYWORD:
                # Text-only search
                return await self.vector_store.search_text(
                    query, top_k, filters
                )
            
            elif search_type == SearchType.HYBRID:
                # Hybrid search
                query_embedding = await self.azure_ai.generate_text_embedding(query)
                return await self.vector_store.search_hybrid(
                    query, query_embedding, top_k, filters
                )
            
            else:
                raise SearchError(f"Unsupported search type: {search_type}")
            
        except Exception as e:
            self.logger.error("Context retrieval failed", query=query, search_type=search_type.value, error=str(e))
            raise SearchError(
                f"Failed to retrieve context: {str(e)}",
                context={"query": query, "search_type": search_type.value}
            )
    
    async def _rank_and_filter_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank and filter context chunks based on relevance.
        
        Args:
            query: Original query
            context_chunks: Retrieved context chunks
            
        Returns:
            Ranked and filtered context chunks
        """
        if not context_chunks:
            return []
        
        # Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in context_chunks 
            if chunk.get("score", 0) >= self.similarity_threshold
        ]
        
        # If no chunks meet threshold, take the top ones anyway
        if not filtered_chunks and context_chunks:
            filtered_chunks = context_chunks[:min(3, len(context_chunks))]
        
        # Ensure we don't exceed context length
        total_length = 0
        final_chunks = []
        
        for chunk in filtered_chunks:
            chunk_length = len(chunk.get("content", ""))
            if total_length + chunk_length <= self.max_context_length:
                final_chunks.append(chunk)
                total_length += chunk_length
            else:
                break
        
        self.logger.debug(
            "Context ranking completed",
            original_chunks=len(context_chunks),
            filtered_chunks=len(filtered_chunks),
            final_chunks=len(final_chunks),
            total_context_length=total_length
        )
        
        return final_chunks
    
    async def _generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the query and context.
        
        Args:
            query: Original query
            context_chunks: Ranked context chunks
            system_prompt: Custom system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response
        """
        try:
            # Prepare context text
            context_text = "\\n\\n".join([
                f"[Context {i+1}]: {chunk.get('content', '')}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Use default system prompt if none provided
            if not system_prompt:
                system_prompt = self._get_default_system_prompt()
            
            # Generate response
            response = await self.azure_ai.generate_rag_response(
                query=query,
                context_chunks=[chunk.get('content', '') for chunk in context_chunks],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
            
        except Exception as e:
            self.logger.error("Response generation failed", query=query, error=str(e))
            raise EmbeddingError(
                f"Failed to generate response: {str(e)}",
                context={"query": query, "context_chunks": len(context_chunks)}
            )
    
    def _extract_source_information(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from context chunks.
        
        Args:
            context_chunks: Context chunks with metadata
            
        Returns:
            List of source information
        """
        sources = []
        seen_sources = set()
        
        for chunk in context_chunks:
            source_id = chunk.get("source_id")
            if source_id and source_id not in seen_sources:
                source_info = {
                    "source_id": source_id,
                    "title": chunk.get("title", ""),
                    "file_path": chunk.get("file_path", ""),
                    "content_type": chunk.get("content_type", ""),
                    "relevance_score": chunk.get("score", 0)
                }
                sources.append(source_info)
                seen_sources.add(source_id)
        
        return sources
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for RAG responses."""
        return """You are a helpful AI assistant that answers questions based on the provided context. 
        
Guidelines:
- Use only the information provided in the context to answer questions
- If the context doesn't contain enough information to answer the question, say so clearly
- Be specific and cite relevant parts of the context when possible
- Maintain a professional and helpful tone
- If asked about topics not covered in the context, acknowledge the limitation
- Provide accurate and concise responses based on the available information"""
    
    @log_function_call
    async def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            System statistics and health information
        """
        try:
            # Get vector store statistics
            vector_stats = await self.vector_store.get_index_statistics()
            
            # Get component health checks
            ai_health = await self.azure_ai.health_check()
            vector_health = await self.vector_store.health_check()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "vector_store": {
                    "statistics": vector_stats,
                    "health": vector_health
                },
                "ai_services": {
                    "health": ai_health
                },
                "configuration": {
                    "default_top_k": self.default_top_k,
                    "max_context_length": self.max_context_length,
                    "similarity_threshold": self.similarity_threshold,
                    "max_query_length": self.max_query_length
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to get system statistics", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the RAG system.
        
        Returns:
            Health check results
        """
        try:
            # Test basic functionality
            test_query = "test health check query"
            test_embedding = await self.azure_ai.generate_text_embedding(test_query)
            
            # Check component health
            ai_health = await self.azure_ai.health_check()
            vector_health = await self.vector_store.health_check()
            
            overall_status = "healthy"
            if (ai_health.get("overall_status") != "healthy" or 
                vector_health.get("status") != "healthy"):
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "service": "RAGService",
                "components": {
                    "ai_services": ai_health,
                    "vector_store": vector_health
                },
                "test_results": {
                    "embedding_generation": len(test_embedding) == 1536,
                    "embedding_dimensions": len(test_embedding)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("RAG service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "RAGService",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass