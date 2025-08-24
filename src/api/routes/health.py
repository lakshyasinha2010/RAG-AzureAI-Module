"""
Health check endpoints for the RAG service.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from ...services.rag_service import RAGService
from ...services.vector_store import VectorStoreService
from ..main import get_rag_service, get_vector_store


router = APIRouter()


@router.get("/", summary="Basic health check")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    return {
        "status": "healthy",
        "service": "RAG Azure AI Module",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Service is running"
    }


@router.get("/detailed", summary="Detailed health check")
async def detailed_health_check(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Detailed health check including all service components.
    
    Returns:
        Comprehensive health status of all components
    """
    try:
        health_status = await rag_service.health_check()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "detailed_status": health_status,
            "message": "Detailed health check completed"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/services", summary="Individual service health")
async def services_health_check(
    rag_service: RAGService = Depends(get_rag_service),
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Check health of individual services.
    
    Returns:
        Health status of each service component
    """
    try:
        # Get health status from individual services
        ai_health = await rag_service.azure_ai.health_check()
        vector_health = await vector_store.health_check()
        doc_intelligence_health = await rag_service.azure_ai.document_intelligence.health_check()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "azure_ai": ai_health,
                "vector_store": vector_health,
                "document_intelligence": doc_intelligence_health
            },
            "message": "Individual service health check completed"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service health check failed: {str(e)}"
        )


@router.get("/stats", summary="System statistics")
async def system_statistics(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get system statistics and metrics.
    
    Returns:
        System statistics including document counts, performance metrics, etc.
    """
    try:
        stats = await rag_service.get_system_statistics()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": stats,
            "message": "System statistics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system statistics: {str(e)}"
        )


@router.get("/readiness", summary="Readiness probe")
async def readiness_probe(
    rag_service: RAGService = Depends(get_rag_service),
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    
    Returns:
        Service readiness status
    """
    try:
        # Test critical components
        health_status = await rag_service.health_check()
        vector_stats = await vector_store.get_index_statistics()
        
        overall_ready = (
            health_status.get("status") in ["healthy", "degraded"] and
            vector_stats.get("total_documents", 0) >= 0  # Index is accessible
        )
        
        return {
            "ready": overall_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "rag_service": health_status.get("status"),
                "vector_store": vector_stats is not None
            }
        }
    except Exception as e:
        return {
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/liveness", summary="Liveness probe")
async def liveness_probe() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe.
    
    Returns:
        Service liveness status
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Service is alive"
    }