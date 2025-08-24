"""
Core configuration management for the multimodal RAG service.
Handles environment-based configuration with validation and secure credential management.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from azure.identity import DefaultAzureCredential, ClientSecretCredential


class AzureConfig(BaseSettings):
    """Azure service configuration."""
    
    # Azure OpenAI
    openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    openai_api_version: str = Field("2023-12-01-preview", env="AZURE_OPENAI_API_VERSION")
    openai_deployment_name: str = Field("gpt-4", env="AZURE_OPENAI_DEPLOYMENT_NAME")
    openai_embedding_deployment: str = Field("text-embedding-ada-002", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    # Azure Cognitive Search
    search_service_name: str = Field(..., env="AZURE_SEARCH_SERVICE_NAME")
    search_api_key: str = Field(..., env="AZURE_SEARCH_API_KEY")
    search_index_name: str = Field("multimodal-rag-index", env="AZURE_SEARCH_INDEX_NAME")
    
    # Azure Computer Vision
    computer_vision_endpoint: str = Field(..., env="AZURE_COMPUTER_VISION_ENDPOINT")
    computer_vision_key: str = Field(..., env="AZURE_COMPUTER_VISION_KEY")
    
    # Azure Speech Services
    speech_key: str = Field(..., env="AZURE_SPEECH_KEY")
    speech_region: str = Field(..., env="AZURE_SPEECH_REGION")
    
    # Azure Form Recognizer
    form_recognizer_endpoint: str = Field(..., env="AZURE_FORM_RECOGNIZER_ENDPOINT")
    form_recognizer_key: str = Field(..., env="AZURE_FORM_RECOGNIZER_KEY")
    
    # Azure Storage
    storage_account_name: str = Field(..., env="AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key: str = Field(..., env="AZURE_STORAGE_ACCOUNT_KEY")
    storage_container_name: str = Field("documents", env="AZURE_STORAGE_CONTAINER_NAME")
    
    # Azure AD (Optional)
    tenant_id: Optional[str] = Field(None, env="AZURE_TENANT_ID")
    client_id: Optional[str] = Field(None, env="AZURE_CLIENT_ID")
    client_secret: Optional[str] = Field(None, env="AZURE_CLIENT_SECRET")

    @validator('search_service_name')
    def validate_search_service_name(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Search service name must be alphanumeric with hyphens/underscores')
        return v

    @property
    def search_endpoint(self) -> str:
        return f"https://{self.search_service_name}.search.windows.net"

    def get_credential(self):
        """Get Azure credential based on available configuration."""
        if self.client_id and self.client_secret and self.tenant_id:
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        return DefaultAzureCredential()


class APIConfig(BaseSettings):
    """API service configuration."""
    
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Security
    api_key: str = Field(..., env="API_KEY")
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # File handling
    max_file_size: int = Field(100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    upload_dir: str = Field("/tmp/uploads", env="UPLOAD_DIR")
    
    # Processing
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")  # 5 minutes

    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field("json", env="LOG_FORMAT")  # json or text
    file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    max_file_size: int = Field(10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")

    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()


class ProcessingConfig(BaseSettings):
    """Document processing configuration."""
    
    # Text processing
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Image processing
    max_image_size: int = Field(4096, env="MAX_IMAGE_SIZE")  # pixels
    image_quality: int = Field(85, env="IMAGE_QUALITY")  # JPEG quality
    
    # Audio processing
    audio_sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    max_audio_duration: int = Field(3600, env="MAX_AUDIO_DURATION")  # seconds
    
    # Video processing
    video_frame_rate: int = Field(1, env="VIDEO_FRAME_RATE")  # frames per second for extraction
    max_video_duration: int = Field(7200, env="MAX_VIDEO_DURATION")  # seconds
    
    # Batch processing
    batch_size: int = Field(10, env="BATCH_SIZE")
    max_retries: int = Field(3, env="MAX_RETRIES")
    retry_delay: int = Field(5, env="RETRY_DELAY")  # seconds


class Settings(BaseSettings):
    """Main application settings."""
    
    azure: AzureConfig = AzureConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    processing: ProcessingConfig = ProcessingConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_configuration() -> bool:
    """Validate all configuration settings."""
    try:
        # Test Azure credential
        credential = settings.azure.get_credential()
        
        # Validate required directories
        os.makedirs(settings.api.upload_dir, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False