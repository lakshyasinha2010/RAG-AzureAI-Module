"""
FastAPI application setup for the multimodal RAG service.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from ..config import get_settings
from ..utils.logging import LoggingMixin, get_logger
from ..core.exceptions import BaseRAGException
from ..services.rag_service import RAGService
from ..services.vector_store import VectorStoreService


# Global service instances
rag_service: RAGService = None
vector_store: VectorStoreService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger = get_logger("FastAPI")
    
    # Startup
    logger.info("Starting up RAG service application")
    
    try:
        # Initialize global services
        global rag_service, vector_store
        
        rag_service = RAGService()
        vector_store = VectorStoreService()
        
        # Ensure vector store index exists
        await vector_store.create_index_if_not_exists()
        
        # Perform health check
        health_status = await rag_service.health_check()
        if health_status.get("status") != "healthy":
            logger.warning("Service health check shows degraded status", status=health_status)
        else:
            logger.info("Service startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down RAG service application")
        
        # Cleanup resources if needed
        if rag_service:
            try:
                await rag_service.__aexit__(None, None, None)
            except Exception as e:
                logger.error("Error during RAG service cleanup", error=str(e))
        
        if vector_store:
            try:
                await vector_store.__aexit__(None, None, None)
            except Exception as e:
                logger.error("Error during vector store cleanup", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        lifespan=lifespan,
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
    )
    
    # Configure middleware
    _configure_middleware(app, settings)
    
    # Configure exception handlers
    _configure_exception_handlers(app)
    
    # Include routers
    _include_routers(app)
    
    return app


def _configure_middleware(app: FastAPI, settings):
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (if not debug mode)
    if not settings.api.debug:
        allowed_hosts = ["localhost", "127.0.0.1"]
        if hasattr(settings.api, 'allowed_hosts'):
            allowed_hosts.extend(settings.api.allowed_hosts)
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )


def _configure_exception_handlers(app: FastAPI):
    """Configure custom exception handlers."""
    
    @app.exception_handler(BaseRAGException)
    async def rag_exception_handler(request: Request, exc: BaseRAGException):
        """Handle custom RAG exceptions."""
        logger = get_logger("ExceptionHandler")
        logger.error(
            "RAG exception occurred",
            error_code=exc.error_code,
            message=exc.message,
            context=exc.context,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "context": exc.context
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger = get_logger("ExceptionHandler")
        logger.error(
            "Validation error occurred",
            errors=exc.errors(),
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger = get_logger("ExceptionHandler")
        logger.error(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP{exc.status_code}",
                    "message": exc.detail
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger = get_logger("ExceptionHandler")
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "InternalServerError",
                    "message": "An internal server error occurred"
                }
            }
        )


def _include_routers(app: FastAPI):
    """Include API routers."""
    from .routes import health, documents, query
    
    # Include routers with prefixes
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(query.router, prefix="/query", tags=["Query"])
    
    # Root endpoint
    @app.get("/", summary="Root endpoint")
    async def root():
        """Root endpoint returning service information."""
        settings = get_settings()
        return {
            "service": settings.api.title,
            "description": settings.api.description,
            "version": settings.api.version,
            "status": "running",
            "docs_url": "/docs" if settings.api.debug else None
        }


def get_rag_service() -> RAGService:
    """Dependency to get the RAG service instance."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return rag_service


def get_vector_store() -> VectorStoreService:
    """Dependency to get the vector store service instance."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store service not initialized")
    return vector_store


# Create the application instance
app = create_app()


if __name__ == "__main__":
    """Run the application with uvicorn."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="debug" if settings.api.debug else "info"
    )