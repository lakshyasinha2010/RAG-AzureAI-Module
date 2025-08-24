"""
Azure Form Recognizer (Document Intelligence) service integration.
Handles document analysis, data extraction, and custom model support.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, BinaryIO
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.ai.formrecognizer import AnalyzeResult, DocumentTable, DocumentKeyValuePair
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from ..config import get_settings
from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import ExceptionHandler, DocumentProcessingError
from ..models.schemas import ContentType


class DocumentIntelligenceService(LoggingMixin):
    """Service for Azure Document Intelligence (Form Recognizer) operations."""
    
    def __init__(self):
        """Initialize the Document Intelligence service."""
        self.settings = get_settings().azure_document_intelligence
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Azure Document Intelligence client."""
        try:
            self._client = DocumentAnalysisClient(
                endpoint=self.settings.endpoint,
                credential=AzureKeyCredential(self.settings.api_key)
            )
            self.logger.info("Document Intelligence client initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Document Intelligence client", error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "DocumentIntelligence", "client_initialization"
            )
    
    @log_function_call
    async def analyze_document_from_bytes(
        self,
        document_bytes: bytes,
        model_id: str = "prebuilt-document",
        include_text_details: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze document from bytes using specified model.
        
        Args:
            document_bytes: Document content as bytes
            model_id: Model to use for analysis (prebuilt-document, prebuilt-invoice, etc.)
            include_text_details: Whether to include detailed text analysis
            
        Returns:
            Dict containing extracted document information
        """
        try:
            async with self._client:
                poller = await self._client.begin_analyze_document(
                    model_id=model_id,
                    document=document_bytes,
                    features=["keyValuePairs", "tables", "styles"] if include_text_details else None
                )
                result = await poller.result()
                
                return self._process_analysis_result(result, model_id)
                
        except ResourceNotFoundError as e:
            self.logger.error("Document analysis model not found", model_id=model_id, error=str(e))
            raise DocumentProcessingError(
                f"Analysis model '{model_id}' not found",
                context={"model_id": model_id}
            )
        except HttpResponseError as e:
            self.logger.error("Document analysis failed", model_id=model_id, error=str(e))
            raise ExceptionHandler.handle_azure_exception(
                e, "DocumentIntelligence", "analyze_document"
            )
    
    @log_function_call
    async def analyze_invoice(self, document_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze invoice document and extract structured data.
        
        Args:
            document_bytes: Invoice document content as bytes
            
        Returns:
            Dict containing extracted invoice information
        """
        return await self.analyze_document_from_bytes(
            document_bytes, 
            model_id="prebuilt-invoice"
        )
    
    @log_function_call
    async def analyze_receipt(self, document_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze receipt document and extract structured data.
        
        Args:
            document_bytes: Receipt document content as bytes
            
        Returns:
            Dict containing extracted receipt information
        """
        return await self.analyze_document_from_bytes(
            document_bytes, 
            model_id="prebuilt-receipt"
        )
    
    @log_function_call
    async def analyze_business_card(self, document_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze business card and extract contact information.
        
        Args:
            document_bytes: Business card content as bytes
            
        Returns:
            Dict containing extracted business card information
        """
        return await self.analyze_document_from_bytes(
            document_bytes, 
            model_id="prebuilt-businessCard"
        )
    
    @log_function_call
    async def analyze_layout(self, document_bytes: bytes) -> Dict[str, Any]:
        """
        Perform layout analysis to extract document structure.
        
        Args:
            document_bytes: Document content as bytes
            
        Returns:
            Dict containing layout analysis results
        """
        return await self.analyze_document_from_bytes(
            document_bytes, 
            model_id="prebuilt-layout"
        )
    
    @log_function_call
    async def extract_tables(self, document_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract tables from document.
        
        Args:
            document_bytes: Document content as bytes
            
        Returns:
            List of extracted tables with their content
        """
        result = await self.analyze_document_from_bytes(document_bytes, "prebuilt-layout")
        return result.get("tables", [])
    
    @log_function_call
    async def extract_key_value_pairs(self, document_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract key-value pairs from document.
        
        Args:
            document_bytes: Document content as bytes
            
        Returns:
            List of extracted key-value pairs
        """
        result = await self.analyze_document_from_bytes(document_bytes, "prebuilt-document")
        return result.get("key_value_pairs", [])
    
    def _process_analysis_result(self, result: AnalyzeResult, model_id: str) -> Dict[str, Any]:
        """
        Process the analysis result into a structured format.
        
        Args:
            result: Analysis result from Document Intelligence
            model_id: Model used for analysis
            
        Returns:
            Processed analysis result
        """
        processed_result = {
            "model_id": model_id,
            "content": result.content,
            "pages": [],
            "tables": [],
            "key_value_pairs": [],
            "documents": []
        }
        
        # Process pages
        for page in result.pages:
            page_info = {
                "page_number": page.page_number,
                "angle": page.angle,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": []
            }
            
            if page.lines:
                for line in page.lines:
                    page_info["lines"].append({
                        "content": line.content,
                        "bounding_box": self._extract_bounding_box(line.polygon) if line.polygon else None
                    })
            
            processed_result["pages"].append(page_info)
        
        # Process tables
        if result.tables:
            for table in result.tables:
                table_info = {
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": []
                }
                
                for cell in table.cells:
                    table_info["cells"].append({
                        "content": cell.content,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "row_span": cell.row_span,
                        "column_span": cell.column_span,
                        "kind": cell.kind
                    })
                
                processed_result["tables"].append(table_info)
        
        # Process key-value pairs
        if result.key_value_pairs:
            for kv_pair in result.key_value_pairs:
                kv_info = {
                    "key": kv_pair.key.content if kv_pair.key else None,
                    "value": kv_pair.value.content if kv_pair.value else None,
                    "confidence": kv_pair.confidence
                }
                processed_result["key_value_pairs"].append(kv_info)
        
        # Process documents (for prebuilt models)
        if result.documents:
            for document in result.documents:
                doc_info = {
                    "doc_type": document.doc_type,
                    "confidence": document.confidence,
                    "fields": {}
                }
                
                if document.fields:
                    for field_name, field_value in document.fields.items():
                        doc_info["fields"][field_name] = {
                            "value": field_value.value,
                            "content": field_value.content,
                            "confidence": field_value.confidence
                        }
                
                processed_result["documents"].append(doc_info)
        
        return processed_result
    
    def _extract_bounding_box(self, polygon: List) -> List[float]:
        """
        Extract bounding box coordinates from polygon.
        
        Args:
            polygon: Polygon points
            
        Returns:
            Bounding box coordinates [x1, y1, x2, y2]
        """
        if not polygon:
            return None
        
        x_coords = [point.x for point in polygon]
        y_coords = [point.y for point in polygon]
        
        return [
            min(x_coords),  # x1
            min(y_coords),  # y1
            max(x_coords),  # x2
            max(y_coords)   # y2
        ]
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the Document Intelligence service.
        
        Returns:
            Health check status
        """
        try:
            # Test with a simple document analysis
            test_content = b"Test document for health check"
            
            async with self._client:
                # Quick test using layout model
                poller = await self._client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    document=test_content
                )
                # Don't wait for completion, just check if the request was accepted
                await asyncio.sleep(1)  # Brief pause
                
            return {
                "status": "healthy",
                "service": "DocumentIntelligence",
                "endpoint": self.settings.endpoint,
                "models_available": [
                    "prebuilt-document",
                    "prebuilt-layout",
                    "prebuilt-invoice",
                    "prebuilt-receipt",
                    "prebuilt-businessCard"
                ]
            }
            
        except Exception as e:
            self.logger.error("Document Intelligence health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "DocumentIntelligence",
                "error": str(e)
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.close()