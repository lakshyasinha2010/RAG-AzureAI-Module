"""
Azure Cognitive Search service for vector storage and hybrid search.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField,
    VectorSearch, VectorSearchProfile, VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration, SemanticConfiguration, SemanticField,
    SemanticPrioritizedFields, SemanticSearch
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from ..config import get_settings
from ..models.schemas import ContentType, SearchResult
from ..utils.logging import LoggingMixin, log_function_call
from ..utils.helpers import generate_id


class AzureCognitiveSearchService(LoggingMixin):
    """Service for interacting with Azure Cognitive Search."""
    
    def __init__(self):
        self.settings = get_settings().azure_search
        self.credential = AzureKeyCredential(self.settings.api_key)
        self.search_client = SearchClient(
            endpoint=self.settings.endpoint,
            index_name=self.settings.index_name,
            credential=self.credential
        )
        self.index_client = SearchIndexClient(
            endpoint=self.settings.endpoint,
            credential=self.credential
        )
        self.embedding_dimension = 1536  # Default for OpenAI embeddings
    
    @log_function_call
    async def create_or_update_index(self, embedding_dimension: int = 1536) -> bool:
        """
        Create or update the search index with multimodal schema.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
            
        Returns:
            True if successful
        """
        try:
            self.embedding_dimension = embedding_dimension
            
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="file_size", type=SearchFieldDataType.Int64, filterable=True),
                SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=embedding_dimension,
                    vector_search_profile_name="vector-profile"
                ),
                SimpleField(name="metadata", type=SearchFieldDataType.String)  # JSON string
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="vector-config",
                        parameters={
                            "m": 4,
                            "ef_construction": 400,
                            "ef_search": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name=self.settings.semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")]
                )
            )
            
            semantic_search = SemanticSearch(
                configurations=[semantic_config]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.settings.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            result = self.index_client.create_or_update_index(index)
            
            self.logger.info(
                "Index created/updated successfully",
                index_name=self.settings.index_name,
                embedding_dimension=embedding_dimension
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to create/update index",
                error=str(e),
                index_name=self.settings.index_name
            )
            raise
    
    @log_function_call
    async def index_document(
        self,
        content_id: str,
        content: str,
        content_vector: List[float],
        content_type: ContentType,
        source_id: str,
        chunk_index: int,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_size: Optional[int] = None,
        file_path: Optional[str] = None
    ) -> bool:
        """
        Index a single document chunk.
        
        Args:
            content_id: Unique identifier for the content chunk
            content: Text content to index
            content_vector: Embedding vector for the content
            content_type: Type of content (text, image, document)
            source_id: ID of the source document
            chunk_index: Index of this chunk within the source
            title: Title of the content
            metadata: Additional metadata
            file_size: Size of original file
            file_path: Path to original file
            
        Returns:
            True if successful
        """
        try:
            document = {
                "id": content_id,
                "content": content,
                "content_vector": content_vector,
                "content_type": content_type.value,
                "source_id": source_id,
                "chunk_index": chunk_index,
                "title": title,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": json.dumps(metadata or {}),
                "file_size": file_size,
                "file_path": file_path
            }
            
            result = self.search_client.upload_documents([document])
            
            if result[0].succeeded:
                self.logger.debug(
                    "Document indexed successfully",
                    content_id=content_id,
                    content_type=content_type.value,
                    source_id=source_id
                )
                return True
            else:
                self.logger.error(
                    "Failed to index document",
                    content_id=content_id,
                    error=result[0].error_message
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Exception while indexing document",
                content_id=content_id,
                error=str(e)
            )
            raise
    
    @log_function_call
    async def index_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary with success and failure counts
        """
        try:
            # Prepare documents for upload
            prepared_docs = []
            for doc in documents:
                prepared_doc = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "content_vector": doc["content_vector"],
                    "content_type": doc["content_type"],
                    "source_id": doc["source_id"],
                    "chunk_index": doc["chunk_index"],
                    "title": doc["title"],
                    "created_at": doc.get("created_at", datetime.utcnow().isoformat()),
                    "metadata": json.dumps(doc.get("metadata", {})),
                    "file_size": doc.get("file_size"),
                    "file_path": doc.get("file_path")
                }
                prepared_docs.append(prepared_doc)
            
            results = self.search_client.upload_documents(prepared_docs)
            
            success_count = sum(1 for result in results if result.succeeded)
            failure_count = len(results) - success_count
            
            self.logger.info(
                "Batch indexing completed",
                total_documents=len(documents),
                successful=success_count,
                failed=failure_count
            )
            
            return {"successful": success_count, "failed": failure_count}
            
        except Exception as e:
            self.logger.error(
                "Exception during batch indexing",
                document_count=len(documents),
                error=str(e)
            )
            raise
    
    @log_function_call
    async def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Perform vector search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: OData filter expression
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                select=["id", "content", "content_type", "source_id", "title", "metadata"] if include_metadata 
                       else ["id", "content", "content_type", "source_id", "title"]
            )
            
            search_results = []
            for result in results:
                metadata = {}
                if include_metadata and result.get("metadata"):
                    try:
                        metadata = json.loads(result["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                search_result = SearchResult(
                    id=result["id"],
                    content=result["content"],
                    score=result["@search.score"],
                    source_id=result["source_id"],
                    content_type=ContentType(result["content_type"]),
                    metadata=metadata
                )
                search_results.append(search_result)
            
            self.logger.debug(
                "Vector search completed",
                results_count=len(search_results),
                top_k=top_k
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(
                "Vector search failed",
                error=str(e),
                top_k=top_k
            )
            raise
    
    @log_function_call
    async def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Perform text search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            filters: OData filter expression
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        try:
            results = self.search_client.search(
                search_text=query_text,
                filter=filters,
                top=top_k,
                select=["id", "content", "content_type", "source_id", "title", "metadata"] if include_metadata 
                       else ["id", "content", "content_type", "source_id", "title"]
            )
            
            search_results = []
            for result in results:
                metadata = {}
                if include_metadata and result.get("metadata"):
                    try:
                        metadata = json.loads(result["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                search_result = SearchResult(
                    id=result["id"],
                    content=result["content"],
                    score=result["@search.score"],
                    source_id=result["source_id"],
                    content_type=ContentType(result["content_type"]),
                    metadata=metadata
                )
                search_results.append(search_result)
            
            self.logger.debug(
                "Text search completed",
                query=query_text,
                results_count=len(search_results),
                top_k=top_k
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(
                "Text search failed",
                query=query_text,
                error=str(e)
            )
            raise
    
    @log_function_call
    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search (vector + text).
        
        Args:
            query_text: Search query text
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: OData filter expression
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                select=["id", "content", "content_type", "source_id", "title", "metadata"] if include_metadata 
                       else ["id", "content", "content_type", "source_id", "title"]
            )
            
            search_results = []
            for result in results:
                metadata = {}
                if include_metadata and result.get("metadata"):
                    try:
                        metadata = json.loads(result["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                search_result = SearchResult(
                    id=result["id"],
                    content=result["content"],
                    score=result["@search.score"],
                    source_id=result["source_id"],
                    content_type=ContentType(result["content_type"]),
                    metadata=metadata
                )
                search_results.append(search_result)
            
            self.logger.debug(
                "Hybrid search completed",
                query=query_text,
                results_count=len(search_results),
                top_k=top_k
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(
                "Hybrid search failed",
                query=query_text,
                error=str(e)
            )
            raise
    
    @log_function_call
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful
        """
        try:
            result = self.search_client.delete_documents([{"id": document_id}])
            
            if result[0].succeeded:
                self.logger.debug("Document deleted successfully", document_id=document_id)
                return True
            else:
                self.logger.error(
                    "Failed to delete document",
                    document_id=document_id,
                    error=result[0].error_message
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Exception while deleting document",
                document_id=document_id,
                error=str(e)
            )
            raise
    
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
            # Search for documents with the source_id
            results = self.search_client.search(
                search_text="*",
                filter=f"source_id eq '{source_id}'",
                select=["id"]
            )
            
            doc_ids = [{"id": result["id"]} for result in results]
            
            if doc_ids:
                delete_results = self.search_client.delete_documents(doc_ids)
                deleted_count = sum(1 for result in delete_results if result.succeeded)
                
                self.logger.info(
                    "Documents deleted by source",
                    source_id=source_id,
                    deleted_count=deleted_count,
                    total_found=len(doc_ids)
                )
                
                return deleted_count
            else:
                self.logger.info("No documents found for source", source_id=source_id)
                return 0
                
        except Exception as e:
            self.logger.error(
                "Exception while deleting documents by source",
                source_id=source_id,
                error=str(e)
            )
            raise
    
    @log_function_call
    async def get_document_count(self) -> int:
        """
        Get total number of documents in the index.
        
        Returns:
            Total document count
        """
        try:
            results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0
            )
            
            count = results.get_count()
            self.logger.debug("Retrieved document count", count=count)
            return count
            
        except Exception as e:
            self.logger.error("Failed to get document count", error=str(e))
            raise
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the search service.
        
        Returns:
            Health status information
        """
        try:
            # Try to get index statistics
            index_stats = self.index_client.get_index_statistics(self.settings.index_name)
            
            return {
                "status": "healthy",
                "service": "azure_cognitive_search",
                "index_name": self.settings.index_name,
                "document_count": index_stats.document_count,
                "storage_size": index_stats.storage_size
            }
            
        except Exception as e:
            self.logger.error("Azure Cognitive Search health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "azure_cognitive_search",
                "error": str(e)
            }