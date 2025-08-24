"""
Vector Store Service using Azure Cognitive Search.
Handles vector index management, document embedding storage, and similarity search.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    VectorSearchAlgorithmConfiguration, SearchAnalyzerName
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

from ..config import get_settings
from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import ExceptionHandler, VectorStoreError
from ..models.schemas import ContentType, ContentChunk, SearchType


class VectorStoreService(LoggingMixin):
    """Vector store service using Azure Cognitive Search."""
    
    def __init__(self):
        """Initialize the Vector Store service."""
        self.settings = get_settings().azure_cognitive_search
        self._search_client = None
        self._index_client = None
        self._setup_clients()
        
        # Vector configuration
        self.vector_dimensions = 1536  # Ada-002 embedding dimensions
        self.index_name = self.settings.index_name
    
    def _setup_clients(self):
        """Setup Azure Cognitive Search clients."""
        try:
            credential = AzureKeyCredential(self.settings.api_key)
            
            self._search_client = SearchClient(
                endpoint=self.settings.endpoint,
                index_name=self.settings.index_name,
                credential=credential
            )
            
            self._index_client = SearchIndexClient(
                endpoint=self.settings.endpoint,
                credential=credential
            )
            
            self.logger.info("Azure Cognitive Search clients initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Azure Cognitive Search clients", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "CognitiveSearch", "client_initialization"
            )
    
    @log_function_call
    async def create_index_if_not_exists(self) -> bool:
        """
        Create the search index if it doesn't exist.
        
        Returns:
            True if index was created or already exists
        """
        try:
            # Check if index exists
            try:
                await self._index_client.get_index(self.index_name)
                self.logger.info("Search index already exists", index_name=self.index_name)
                return True
            except ResourceNotFoundError:
                pass  # Index doesn't exist, create it
            
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name=SearchAnalyzerName.STANDARD_LUCENE),
                SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="title", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="file_size", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.vector_dimensions,
                    vector_search_profile_name="default-vector-profile"
                ),
                SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=True)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw-config",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-config"
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            await self._index_client.create_index(index)
            self.logger.info("Search index created successfully", index_name=self.index_name)
            return True
            
        except ResourceExistsError:
            self.logger.info("Search index already exists", index_name=self.index_name)
            return True
        except Exception as e:
            self.logger.error("Failed to create search index", error=str(e))
            raise VectorStoreError(
                f"Failed to create index {self.index_name}: {str(e)}",
                context={"index_name": self.index_name}
            )
    
    @log_function_call
    async def index_document(
        self,
        chunk: ContentChunk,
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> bool:
        """
        Index a document chunk in the vector store.
        
        Args:
            chunk: Content chunk to index
            title: Document title
            file_path: Original file path
            file_size: File size in bytes
            
        Returns:
            True if successful
        """
        try:
            if not chunk.embedding:
                raise VectorStoreError(
                    "Content chunk must have embeddings before indexing",
                    context={"chunk_id": chunk.id}
                )
            
            # Prepare document for indexing
            now = datetime.utcnow()
            document = {
                "id": chunk.id,
                "content": chunk.content,
                "content_type": chunk.content_type.value,
                "source_id": chunk.source_id,
                "chunk_index": chunk.chunk_index,
                "title": title or chunk.metadata.get("filename", ""),
                "file_path": file_path or chunk.metadata.get("file_path", ""),
                "file_size": file_size or chunk.metadata.get("file_size", 0),
                "created_at": now,
                "updated_at": now,
                "content_vector": chunk.embedding,
                "metadata": str(chunk.metadata)  # Store as JSON string
            }
            
            # Upload to search index
            result = await self._search_client.upload_documents([document])
            
            if result[0].succeeded:
                self.logger.debug(
                    "Document indexed successfully",
                    chunk_id=chunk.id,
                    content_type=chunk.content_type.value,
                    source_id=chunk.source_id
                )
                return True
            else:
                self.logger.error(
                    "Failed to index document",
                    chunk_id=chunk.id,
                    error=result[0].error_message
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Exception while indexing document",
                chunk_id=chunk.id,
                error=str(e)
            )
            raise VectorStoreError(
                f"Failed to index document {chunk.id}: {str(e)}",
                context={"chunk_id": chunk.id}
            )
    
    @log_function_call
    async def index_documents_batch(
        self,
        chunks: List[ContentChunk],
        title: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Index multiple document chunks in batch.
        
        Args:
            chunks: List of content chunks to index
            title: Document title
            file_path: Original file path
            file_size: File size in bytes
            
        Returns:
            Batch indexing results
        """
        try:
            if not chunks:
                return {"success": True, "indexed": 0, "failed": 0, "errors": []}
            
            # Prepare documents for batch indexing
            documents = []
            now = datetime.utcnow()
            
            for chunk in chunks:
                if not chunk.embedding:
                    self.logger.warning("Skipping chunk without embeddings", chunk_id=chunk.id)
                    continue
                
                document = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "content_type": chunk.content_type.value,
                    "source_id": chunk.source_id,
                    "chunk_index": chunk.chunk_index,
                    "title": title or chunk.metadata.get("filename", ""),
                    "file_path": file_path or chunk.metadata.get("file_path", ""),
                    "file_size": file_size or chunk.metadata.get("file_size", 0),
                    "created_at": now,
                    "updated_at": now,
                    "content_vector": chunk.embedding,
                    "metadata": str(chunk.metadata)
                }
                documents.append(document)
            
            if not documents:
                return {"success": False, "indexed": 0, "failed": len(chunks), "errors": ["No valid chunks to index"]}
            
            # Upload batch to search index
            results = await self._search_client.upload_documents(documents)
            
            # Analyze results
            successful = sum(1 for result in results if result.succeeded)
            failed = len(results) - successful
            errors = [result.error_message for result in results if not result.succeeded]
            
            self.logger.info(
                "Batch indexing completed",
                total_chunks=len(chunks),
                documents_prepared=len(documents),
                successful=successful,
                failed=failed
            )
            
            return {
                "success": failed == 0,
                "indexed": successful,
                "failed": failed,
                "errors": errors
            }
            
        except Exception as e:
            self.logger.error("Batch indexing failed", error=str(e))
            raise VectorStoreError(
                f"Batch indexing failed: {str(e)}",
                context={"chunk_count": len(chunks)}
            )
    
    @log_function_call
    async def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: OData filter string
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        try:
            search_results = await self._search_client.search(
                search_text="",
                vector_queries=[{
                    "vector": query_vector,
                    "k_nearest_neighbors": top_k,
                    "fields": "content_vector"
                }],
                filter=filters,
                select=["id", "content", "content_type", "source_id", "chunk_index", "title", "file_path", "metadata"] if include_metadata else ["id", "content"],
                top=top_k
            )
            
            results = []
            async for result in search_results:
                search_result = {
                    "id": result["id"],
                    "content": result["content"],
                    "score": result["@search.score"],
                    "content_type": result.get("content_type"),
                    "source_id": result.get("source_id"),
                    "chunk_index": result.get("chunk_index")
                }
                
                if include_metadata:
                    search_result.update({
                        "title": result.get("title"),
                        "file_path": result.get("file_path"),
                        "metadata": result.get("metadata")
                    })
                
                results.append(search_result)
            
            self.logger.debug(
                "Vector search completed",
                query_vector_dims=len(query_vector),
                top_k=top_k,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Vector search failed", error=str(e))
            raise VectorStoreError(
                f"Vector search failed: {str(e)}",
                context={"top_k": top_k, "vector_dims": len(query_vector)}
            )
    
    @log_function_call
    async def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform text-based search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            filters: OData filter string
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        try:
            search_results = await self._search_client.search(
                search_text=query_text,
                filter=filters,
                select=["id", "content", "content_type", "source_id", "chunk_index", "title", "file_path", "metadata"] if include_metadata else ["id", "content"],
                top=top_k,
                highlight_fields=["content"]
            )
            
            results = []
            async for result in search_results:
                search_result = {
                    "id": result["id"],
                    "content": result["content"],
                    "score": result["@search.score"],
                    "content_type": result.get("content_type"),
                    "source_id": result.get("source_id"),
                    "chunk_index": result.get("chunk_index"),
                    "highlights": result.get("@search.highlights", {})
                }
                
                if include_metadata:
                    search_result.update({
                        "title": result.get("title"),
                        "file_path": result.get("file_path"),
                        "metadata": result.get("metadata")
                    })
                
                results.append(search_result)
            
            self.logger.debug(
                "Text search completed",
                query_text=query_text,
                top_k=top_k,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Text search failed", error=str(e))
            raise VectorStoreError(
                f"Text search failed: {str(e)}",
                context={"query_text": query_text, "top_k": top_k}
            )
    
    @log_function_call
    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True,
        text_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            query_text: Search query text
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: OData filter string
            include_metadata: Whether to include metadata in results
            text_weight: Weight for text search results
            vector_weight: Weight for vector search results
            
        Returns:
            List of search results
        """
        try:
            search_results = await self._search_client.search(
                search_text=query_text,
                vector_queries=[{
                    "vector": query_vector,
                    "k_nearest_neighbors": top_k,
                    "fields": "content_vector"
                }],
                filter=filters,
                select=["id", "content", "content_type", "source_id", "chunk_index", "title", "file_path", "metadata"] if include_metadata else ["id", "content"],
                top=top_k,
                highlight_fields=["content"]
            )
            
            results = []
            async for result in search_results:
                search_result = {
                    "id": result["id"],
                    "content": result["content"],
                    "score": result["@search.score"],
                    "content_type": result.get("content_type"),
                    "source_id": result.get("source_id"),
                    "chunk_index": result.get("chunk_index"),
                    "highlights": result.get("@search.highlights", {})
                }
                
                if include_metadata:
                    search_result.update({
                        "title": result.get("title"),
                        "file_path": result.get("file_path"),
                        "metadata": result.get("metadata")
                    })
                
                results.append(search_result)
            
            self.logger.debug(
                "Hybrid search completed",
                query_text=query_text,
                query_vector_dims=len(query_vector),
                top_k=top_k,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Hybrid search failed", error=str(e))
            raise VectorStoreError(
                f"Hybrid search failed: {str(e)}",
                context={"query_text": query_text, "top_k": top_k, "vector_dims": len(query_vector)}
            )
    
    @log_function_call
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        try:
            result = await self._search_client.delete_documents([{"id": document_id}])
            
            if result[0].succeeded:
                self.logger.debug("Document deleted successfully", document_id=document_id)
                return True
            else:
                self.logger.error("Failed to delete document", document_id=document_id, error=result[0].error_message)
                return False
                
        except Exception as e:
            self.logger.error("Exception while deleting document", document_id=document_id, error=str(e))
            raise VectorStoreError(
                f"Failed to delete document {document_id}: {str(e)}",
                context={"document_id": document_id}
            )
    
    @log_function_call
    async def delete_documents_by_source(self, source_id: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source_id: Source ID to delete documents for
            
        Returns:
            Number of documents deleted
        """
        try:
            # First, search for all documents with the source ID
            search_results = await self._search_client.search(
                search_text="*",
                filter=f"source_id eq '{source_id}'",
                select=["id"],
                top=1000  # Assume max 1000 chunks per document
            )
            
            document_ids = []
            async for result in search_results:
                document_ids.append({"id": result["id"]})
            
            if not document_ids:
                self.logger.debug("No documents found for source", source_id=source_id)
                return 0
            
            # Delete all found documents
            results = await self._search_client.delete_documents(document_ids)
            
            deleted_count = sum(1 for result in results if result.succeeded)
            
            self.logger.info(
                "Documents deleted by source",
                source_id=source_id,
                found=len(document_ids),
                deleted=deleted_count
            )
            
            return deleted_count
            
        except Exception as e:
            self.logger.error("Failed to delete documents by source", source_id=source_id, error=str(e))
            raise VectorStoreError(
                f"Failed to delete documents by source {source_id}: {str(e)}",
                context={"source_id": source_id}
            )
    
    @log_function_call
    async def get_document_count(self) -> int:
        """
        Get total number of documents in the index.
        
        Returns:
            Number of documents
        """
        try:
            # Use search to get document count
            search_results = await self._search_client.search(
                search_text="*",
                include_total_count=True,
                top=0  # We only want the count
            )
            
            # Access the count from search results
            count = search_results.get_count()
            
            self.logger.debug("Retrieved document count", count=count)
            return count or 0
            
        except Exception as e:
            self.logger.error("Failed to get document count", error=str(e))
            raise VectorStoreError(f"Failed to get document count: {str(e)}")
    
    @log_function_call
    async def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics
        """
        try:
            # Get basic statistics
            document_count = await self.get_document_count()
            
            # Get content type distribution
            search_results = await self._search_client.search(
                search_text="*",
                facets=["content_type,count:100", "source_id,count:100"],
                top=0
            )
            
            facets = {}
            async for result in search_results:
                if hasattr(result, '@search.facets'):
                    facets = result['@search.facets']
                break
            
            return {
                "total_documents": document_count,
                "content_type_distribution": facets.get("content_type", []),
                "source_distribution": facets.get("source_id", []),
                "index_name": self.index_name
            }
            
        except Exception as e:
            self.logger.error("Failed to get index statistics", error=str(e))
            return {
                "total_documents": 0,
                "error": str(e),
                "index_name": self.index_name
            }
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.
        
        Returns:
            Health check status
        """
        try:
            # Test basic connectivity
            document_count = await self.get_document_count()
            
            # Test index existence
            try:
                await self._index_client.get_index(self.index_name)
                index_exists = True
            except ResourceNotFoundError:
                index_exists = False
            
            return {
                "status": "healthy",
                "service": "VectorStore",
                "endpoint": self.settings.endpoint,
                "index_name": self.index_name,
                "index_exists": index_exists,
                "document_count": document_count,
                "vector_dimensions": self.vector_dimensions
            }
            
        except Exception as e:
            self.logger.error("Vector store health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "VectorStore",
                "error": str(e),
                "index_name": self.index_name
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._search_client:
            await self._search_client.close()
        if self._index_client:
            await self._index_client.close()