
"""Main package for the multimodal RAG service."""

__version__ = "1.0.0"
__title__ = "Multimodal RAG Service"
__description__ = "Production-ready multimodal Retrieval-Augmented Generation service using Azure AI"
__author__ = "Azure AI Team"

from .core import settings, get_settings, get_logger

__all__ = [
    "__version__",
    "__title__", 
    "__description__",
    "__author__",
    "settings",
    "get_settings",
    "get_logger"
]
