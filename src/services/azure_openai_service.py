"""
Azure OpenAI service integration for embeddings and completions.
"""

import asyncio
from typing import List, Optional, Dict, Any
import openai
from azure.identity import DefaultAzureCredential
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..utils.logging import LoggingMixin, log_function_call
from ..utils.helpers import async_retry


class AzureOpenAIService(LoggingMixin):
    """Service for interacting with Azure OpenAI."""
    
    def __init__(self):
        self.settings = get_settings().azure_openai
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Azure OpenAI client."""
        try:
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version
            )
            self.logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Azure OpenAI client", error=str(e))
            raise
    
    @log_function_call
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = await self._client.embeddings.create(
                model=self.settings.embedding_deployment_name,
                input=text.replace("\n", " ")  # OpenAI recommends replacing newlines
            )
            
            embedding = response.data[0].embedding
            self.logger.debug(
                "Generated embedding",
                text_length=len(text),
                embedding_dimension=len(embedding)
            )
            return embedding
            
        except Exception as e:
            self.logger.error(
                "Failed to generate embedding",
                error=str(e),
                text_length=len(text)
            )
            raise
    
    @log_function_call
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each batch for processing
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Process texts in batch, replacing newlines
                processed_batch = [text.replace("\n", " ") for text in batch]
                
                response = await self._client.embeddings.create(
                    model=self.settings.embedding_deployment_name,
                    input=processed_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                self.logger.debug(
                    "Generated batch embeddings",
                    batch_size=len(batch),
                    processed_count=len(embeddings)
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to generate batch embeddings",
                    batch_index=i // batch_size,
                    batch_size=len(batch),
                    error=str(e)
                )
                raise
        
        self.logger.info(
            "Completed batch embedding generation",
            total_texts=len(texts),
            total_embeddings=len(embeddings)
        )
        
        return embeddings
    
    @log_function_call
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate completion using Azure OpenAI chat completions.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated completion text
        """
        try:
            # Prepare messages
            chat_messages = []
            
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            
            chat_messages.extend(messages)
            
            # Use provided parameters or defaults
            temperature = temperature or self.settings.temperature
            max_tokens = max_tokens or self.settings.max_tokens
            
            response = await self._client.chat.completions.create(
                model=self.settings.deployment_name,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            completion = response.choices[0].message.content
            
            self.logger.debug(
                "Generated completion",
                messages_count=len(chat_messages),
                temperature=temperature,
                max_tokens=max_tokens,
                completion_length=len(completion) if completion else 0,
                usage=response.usage.dict() if response.usage else None
            )
            
            return completion
            
        except Exception as e:
            self.logger.error(
                "Failed to generate completion",
                error=str(e),
                messages_count=len(messages),
                temperature=temperature,
                max_tokens=max_tokens
            )
            raise
    
    @log_function_call
    async def generate_rag_response(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate RAG response using query and context chunks.
        
        Args:
            query: User query
            context_chunks: List of relevant context chunks
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated response
        """
        # Prepare context
        context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
        
        # Default system prompt for RAG
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use only the information provided in the context to answer the question. 
            If the context doesn't contain enough information to answer the question, say so clearly.
            Always cite the relevant context sections using the [number] references."""
        
        # Prepare user message
        user_message = f"""Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above."""
        
        messages = [{"role": "user", "content": user_message}]
        
        return await self.generate_completion(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Azure OpenAI service.
        
        Returns:
            Health status information
        """
        try:
            # Test with a simple embedding request
            test_response = await self.generate_embedding("health check")
            
            return {
                "status": "healthy",
                "service": "azure_openai",
                "embedding_dimension": len(test_response),
                "deployment_name": self.settings.deployment_name,
                "embedding_deployment_name": self.settings.embedding_deployment_name
            }
            
        except Exception as e:
            self.logger.error("Azure OpenAI health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "azure_openai",
                "error": str(e)
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _retry_request(self, request_func, *args, **kwargs):
        """Retry wrapper for API requests with exponential backoff."""
        return await request_func(*args, **kwargs)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the deployed models."""
        try:
            # This would require Azure Management API calls
            # For now, return configuration information
            return {
                "completion_model": self.settings.deployment_name,
                "embedding_model": self.settings.embedding_deployment_name,
                "api_version": self.settings.api_version,
                "endpoint": self.settings.endpoint,
                "max_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature
            }
        except Exception as e:
            self.logger.error("Failed to get model info", error=str(e))
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, '_client') and self._client:
            # Azure OpenAI client doesn't require explicit cleanup
            pass