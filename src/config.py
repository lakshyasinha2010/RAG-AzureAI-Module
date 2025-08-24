"""
Configuration management for the RAG Azure AI Module.
Uses environment variables with sensible defaults for production deployment.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration settings."""
    
    endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    api_version: str = Field(default="2023-12-01-preview", env="AZURE_OPENAI_API_VERSION")
    deployment_name: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT_NAME")
    embedding_deployment_name: str = Field(..., env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    max_tokens: int = Field(default=4000, env="AZURE_OPENAI_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="AZURE_OPENAI_TEMPERATURE")


class AzureCognitiveSearchSettings(BaseSettings):
    """Azure Cognitive Search configuration settings."""
    
    endpoint: str = Field(..., env="AZURE_SEARCH_ENDPOINT")
    api_key: str = Field(..., env="AZURE_SEARCH_API_KEY")
    index_name: str = Field(default="rag-multimodal-index", env="AZURE_SEARCH_INDEX_NAME")
    semantic_config_name: str = Field(default="semantic-config", env="AZURE_SEARCH_SEMANTIC_CONFIG")


class AzureDocumentIntelligenceSettings(BaseSettings):
    """Azure Document Intelligence configuration settings."""
    
    endpoint: str = Field(..., env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key: str = Field(..., env="AZURE_DOCUMENT_INTELLIGENCE_API_KEY")


class AzureComputerVisionSettings(BaseSettings):
    """Azure Computer Vision configuration settings."""
    
    endpoint: str = Field(..., env="AZURE_COMPUTER_VISION_ENDPOINT")
    api_key: str = Field(..., env="AZURE_COMPUTER_VISION_API_KEY")


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    ttl_seconds: int = Field(default=3600, env="REDIS_TTL_SECONDS")  # 1 hour default


class APISettings(BaseSettings):
    """API configuration settings."""
    
    title: str = Field(default="RAG Azure AI Module", env="API_TITLE")
    description: str = Field(default="Multimodal RAG service using Azure AI", env="API_DESCRIPTION")
    version: str = Field(default="1.0.0", env="API_VERSION")
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    max_upload_size: int = Field(default=50 * 1024 * 1024, env="API_MAX_UPLOAD_SIZE")  # 50MB
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    api_key: Optional[str] = Field(default=None, env="API_KEY")  # Optional API key for security


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration settings."""
    
    requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    burst_size: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE")


class ProcessingSettings(BaseSettings):
    """Content processing configuration settings."""
    
    chunk_size: int = Field(default=1000, env="PROCESSING_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="PROCESSING_CHUNK_OVERLAP")
    max_concurrent_jobs: int = Field(default=5, env="PROCESSING_MAX_CONCURRENT_JOBS")
    supported_image_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp", "tiff", "gif"],
        env="PROCESSING_SUPPORTED_IMAGE_FORMATS"
    )
    supported_document_formats: List[str] = Field(
        default=["pdf", "docx", "txt"],
        env="PROCESSING_SUPPORTED_DOCUMENT_FORMATS"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Azure services
    azure_openai: AzureOpenAISettings = AzureOpenAISettings()
    azure_search: AzureCognitiveSearchSettings = AzureCognitiveSearchSettings()
    azure_document_intelligence: AzureDocumentIntelligenceSettings = AzureDocumentIntelligenceSettings()
    azure_computer_vision: AzureComputerVisionSettings = AzureComputerVisionSettings()
    
    # Infrastructure
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    processing: ProcessingSettings = ProcessingSettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings