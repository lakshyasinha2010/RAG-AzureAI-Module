"""
Unified Azure AI service wrapper that integrates multiple Azure AI services.
Provides a single interface for Azure OpenAI, Computer Vision, Speech Services, and Document Intelligence.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, BinaryIO
from azure.cognitiveservices.vision.computervision.aio import ComputerVisionClient
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.vision.computervision.models as cv_models

from ..config import get_settings
from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import ExceptionHandler, AzureServiceError
from .azure_openai_service import AzureOpenAIService
from .document_intelligence import DocumentIntelligenceService


class AzureAIService(LoggingMixin):
    """Unified Azure AI service wrapper for multimodal AI operations."""
    
    def __init__(self):
        """Initialize the unified Azure AI service."""
        self.settings = get_settings()
        
        # Initialize individual services
        self.openai_service = AzureOpenAIService()
        self.document_intelligence = DocumentIntelligenceService()
        
        # Initialize other Azure AI services
        self._computer_vision_client = None
        self._speech_config = None
        
        self._setup_computer_vision()
        self._setup_speech_services()
    
    def _setup_computer_vision(self):
        """Setup Azure Computer Vision client."""
        try:
            cv_settings = self.settings.azure_computer_vision
            self._computer_vision_client = ComputerVisionClient(
                endpoint=cv_settings.endpoint,
                credentials=AzureKeyCredential(cv_settings.api_key)
            )
            self.logger.info("Computer Vision client initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Computer Vision client", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "ComputerVision", "client_initialization"
            )
    
    def _setup_speech_services(self):
        """Setup Azure Speech Services configuration."""
        try:
            speech_settings = self.settings.azure_speech if hasattr(self.settings, 'azure_speech') else None
            if speech_settings:
                self._speech_config = SpeechConfig(
                    subscription=speech_settings.api_key,
                    region=speech_settings.region
                )
                self.logger.info("Speech Services configured successfully")
            else:
                self.logger.warning("Speech Services configuration not found")
        except Exception as e:
            self.logger.error("Failed to configure Speech Services", error=str(e))
            # Don't raise exception as speech services might be optional
    
    # Azure OpenAI Service Methods
    @log_function_call
    async def generate_text_embedding(self, text: str) -> List[float]:
        """Generate text embedding using Azure OpenAI."""
        return await self.openai_service.generate_embedding(text)
    
    @log_function_call
    async def generate_text_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> List[List[float]]:
        """Generate text embeddings in batches."""
        return await self.openai_service.generate_embeddings_batch(texts, batch_size)
    
    @log_function_call
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text completion using Azure OpenAI."""
        return await self.openai_service.generate_completion(
            messages, temperature, max_tokens, system_prompt
        )
    
    @log_function_call
    async def generate_rag_response(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate RAG response using Azure OpenAI."""
        return await self.openai_service.generate_rag_response(
            query, context_chunks, system_prompt, temperature, max_tokens
        )
    
    # Computer Vision Service Methods
    @log_function_call
    async def analyze_image(
        self, 
        image_data: Union[bytes, str],
        visual_features: Optional[List[str]] = None,
        details: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze image using Azure Computer Vision.
        
        Args:
            image_data: Image data as bytes or URL
            visual_features: List of visual features to analyze
            details: Additional details to extract
            
        Returns:
            Analysis results from Computer Vision
        """
        try:
            if visual_features is None:
                visual_features = [
                    cv_models.VisualFeatureTypes.categories,
                    cv_models.VisualFeatureTypes.description,
                    cv_models.VisualFeatureTypes.tags,
                    cv_models.VisualFeatureTypes.objects,
                    cv_models.VisualFeatureTypes.faces
                ]
            
            if isinstance(image_data, str):
                # URL-based analysis
                analysis = await self._computer_vision_client.analyze_image(
                    url=image_data,
                    visual_features=visual_features,
                    details=details
                )
            else:
                # Bytes-based analysis
                analysis = await self._computer_vision_client.analyze_image_in_stream(
                    image=image_data,
                    visual_features=visual_features,
                    details=details
                )
            
            return self._process_computer_vision_result(analysis)
            
        except Exception as e:
            self.logger.error("Computer Vision analysis failed", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "ComputerVision", "analyze_image"
            )
    
    @log_function_call
    async def extract_text_from_image(self, image_data: Union[bytes, str]) -> Dict[str, Any]:
        """
        Extract text from image using OCR.
        
        Args:
            image_data: Image data as bytes or URL
            
        Returns:
            Extracted text and OCR results
        """
        try:
            if isinstance(image_data, str):
                # URL-based OCR
                ocr_result = await self._computer_vision_client.read(url=image_data)
            else:
                # Bytes-based OCR
                ocr_result = await self._computer_vision_client.read_in_stream(image=image_data)
            
            # Get the operation location
            operation_location = ocr_result.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]
            
            # Wait for the operation to complete
            while True:
                read_result = await self._computer_vision_client.get_read_result(operation_id)
                if read_result.status not in [cv_models.OperationStatusCodes.running]:
                    break
                await asyncio.sleep(1)
            
            return self._process_ocr_result(read_result)
            
        except Exception as e:
            self.logger.error("OCR analysis failed", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "ComputerVision", "extract_text"
            )
    
    @log_function_call
    async def generate_image_description(self, image_data: Union[bytes, str]) -> str:
        """
        Generate a natural language description of an image.
        
        Args:
            image_data: Image data as bytes or URL
            
        Returns:
            Generated image description
        """
        analysis = await self.analyze_image(
            image_data, 
            visual_features=[cv_models.VisualFeatureTypes.description]
        )
        
        descriptions = analysis.get("descriptions", [])
        if descriptions:
            return descriptions[0].get("text", "No description available")
        return "No description available"
    
    # Document Intelligence Service Methods
    @log_function_call
    async def analyze_document(
        self,
        document_bytes: bytes,
        model_id: str = "prebuilt-document"
    ) -> Dict[str, Any]:
        """Analyze document using Document Intelligence."""
        return await self.document_intelligence.analyze_document_from_bytes(
            document_bytes, model_id
        )
    
    @log_function_call
    async def extract_document_tables(self, document_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract tables from document."""
        return await self.document_intelligence.extract_tables(document_bytes)
    
    @log_function_call
    async def extract_document_key_values(self, document_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract key-value pairs from document."""
        return await self.document_intelligence.extract_key_value_pairs(document_bytes)
    
    # Speech Services Methods (if configured)
    @log_function_call
    async def speech_to_text(self, audio_file_path: str) -> str:
        """
        Convert speech to text using Azure Speech Services.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not self._speech_config:
            raise AzureServiceError(
                "Speech Services not configured",
                service_name="SpeechServices",
                operation="speech_to_text"
            )
        
        try:
            audio_config = AudioConfig(filename=audio_file_path)
            speech_recognizer = SpeechRecognizer(
                speech_config=self._speech_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            result = await speech_recognizer.recognize_once_async()
            
            if result.reason == result.reason.RecognizedSpeech:
                return result.text
            elif result.reason == result.reason.NoMatch:
                return "No speech could be recognized"
            else:
                raise AzureServiceError(
                    f"Speech recognition failed: {result.reason}",
                    service_name="SpeechServices",
                    operation="speech_to_text"
                )
                
        except Exception as e:
            self.logger.error("Speech to text conversion failed", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "SpeechServices", "speech_to_text"
            )
    
    # Multimodal Processing Methods
    @log_function_call
    async def process_multimodal_content(
        self,
        content_type: str,
        content_data: Union[bytes, str],
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multimodal content using appropriate Azure AI services.
        
        Args:
            content_type: Type of content (image, document, audio, text)
            content_data: Content data
            additional_context: Additional context for processing
            
        Returns:
            Processed content results
        """
        results = {
            "content_type": content_type,
            "processed_data": {},
            "embeddings": None,
            "extracted_text": None
        }
        
        try:
            if content_type.startswith("image/"):
                # Process image content
                vision_analysis = await self.analyze_image(content_data)
                ocr_results = await self.extract_text_from_image(content_data)
                
                results["processed_data"]["vision_analysis"] = vision_analysis
                results["processed_data"]["ocr_results"] = ocr_results
                results["extracted_text"] = ocr_results.get("text", "")
                
                # Generate embedding for image description
                description = await self.generate_image_description(content_data)
                if description:
                    results["embeddings"] = await self.generate_text_embedding(description)
            
            elif content_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # Process document content
                doc_analysis = await self.analyze_document(content_data)
                results["processed_data"]["document_analysis"] = doc_analysis
                results["extracted_text"] = doc_analysis.get("content", "")
                
                # Generate embedding for document content
                if results["extracted_text"]:
                    results["embeddings"] = await self.generate_text_embedding(results["extracted_text"])
            
            elif content_type.startswith("audio/"):
                # Process audio content (if speech services are available)
                if self._speech_config:
                    # Note: content_data should be a file path for audio
                    transcribed_text = await self.speech_to_text(content_data)
                    results["processed_data"]["transcription"] = transcribed_text
                    results["extracted_text"] = transcribed_text
                    
                    # Generate embedding for transcribed text
                    if transcribed_text:
                        results["embeddings"] = await self.generate_text_embedding(transcribed_text)
            
            elif content_type.startswith("text/"):
                # Process text content
                text_content = content_data if isinstance(content_data, str) else content_data.decode('utf-8')
                results["extracted_text"] = text_content
                results["embeddings"] = await self.generate_text_embedding(text_content)
            
            return results
            
        except Exception as e:
            self.logger.error("Multimodal content processing failed", 
                            content_type=content_type, error=str(e))
            raise
    
    def _process_computer_vision_result(self, analysis) -> Dict[str, Any]:
        """Process Computer Vision analysis result."""
        result = {}
        
        if hasattr(analysis, 'categories') and analysis.categories:
            result["categories"] = [
                {"name": cat.name, "score": cat.score} 
                for cat in analysis.categories
            ]
        
        if hasattr(analysis, 'description') and analysis.description:
            result["descriptions"] = [
                {"text": desc.text, "confidence": desc.confidence}
                for desc in analysis.description.captions
            ]
        
        if hasattr(analysis, 'tags') and analysis.tags:
            result["tags"] = [
                {"name": tag.name, "confidence": tag.confidence}
                for tag in analysis.tags
            ]
        
        if hasattr(analysis, 'objects') and analysis.objects:
            result["objects"] = [
                {
                    "object": obj.object_property,
                    "confidence": obj.confidence,
                    "rectangle": {
                        "x": obj.rectangle.x,
                        "y": obj.rectangle.y,
                        "w": obj.rectangle.w,
                        "h": obj.rectangle.h
                    }
                }
                for obj in analysis.objects
            ]
        
        if hasattr(analysis, 'faces') and analysis.faces:
            result["faces"] = [
                {
                    "age": face.age,
                    "gender": face.gender,
                    "rectangle": {
                        "left": face.face_rectangle.left,
                        "top": face.face_rectangle.top,
                        "width": face.face_rectangle.width,
                        "height": face.face_rectangle.height
                    }
                }
                for face in analysis.faces
            ]
        
        return result
    
    def _process_ocr_result(self, read_result) -> Dict[str, Any]:
        """Process OCR analysis result."""
        result = {
            "text": "",
            "lines": [],
            "words": []
        }
        
        if read_result.analyze_result:
            all_text = []
            for read_result_page in read_result.analyze_result.read_results:
                for line in read_result_page.lines:
                    all_text.append(line.text)
                    
                    line_info = {
                        "text": line.text,
                        "bounding_box": line.bounding_box
                    }
                    result["lines"].append(line_info)
                    
                    # Extract words
                    for word in line.words:
                        word_info = {
                            "text": word.text,
                            "confidence": word.confidence,
                            "bounding_box": word.bounding_box
                        }
                        result["words"].append(word_info)
            
            result["text"] = " ".join(all_text)
        
        return result
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all Azure AI services.
        
        Returns:
            Health status of all services
        """
        health_status = {
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check OpenAI service
        try:
            openai_health = await self.openai_service.health_check()
            health_status["services"]["openai"] = openai_health
        except Exception as e:
            health_status["services"]["openai"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check Document Intelligence service
        try:
            doc_health = await self.document_intelligence.health_check()
            health_status["services"]["document_intelligence"] = doc_health
        except Exception as e:
            health_status["services"]["document_intelligence"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check Computer Vision service
        try:
            # Simple health check for Computer Vision
            health_status["services"]["computer_vision"] = {
                "status": "healthy" if self._computer_vision_client else "not_configured",
                "endpoint": getattr(self.settings.azure_computer_vision, 'endpoint', 'not_configured')
            }
        except Exception as e:
            health_status["services"]["computer_vision"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "degraded"
        
        # Check Speech Services
        health_status["services"]["speech_services"] = {
            "status": "configured" if self._speech_config else "not_configured"
        }
        
        return health_status
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._computer_vision_client:
            await self._computer_vision_client.close()
        
        if hasattr(self.document_intelligence, '__aexit__'):
            await self.document_intelligence.__aexit__(exc_type, exc_val, exc_tb)