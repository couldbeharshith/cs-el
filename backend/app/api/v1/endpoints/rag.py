from fastapi import APIRouter, HTTPException
from app.models.request import Request
from app.models.response import Response, ProductionResponse, HealthResponse
from app.config.settings import settings
import time
import httpx
from typing import Union

router = APIRouter()

# RAG Server Configuration
RAG_SERVER_URL = "http://localhost:7860/api/v1/hackrx/run"
RAG_SERVER_TIMEOUT = 300  # 5 minutes timeout for RAG processing

@router.post("/run", response_model=Union[Response, ProductionResponse])
async def run_rag(request: Request):
    """
    RAG endpoint - calls the Insurance RAG API server.
    
    Expected request format:
    {
        "documents": "URL to document",
        "questions": ["question1", "question2"]
    }
    
    Calls the RAG server at http://localhost:7860/api/v1/hackrx/run
    """
    start_time = time.time()
    
    try:
        # Prepare request for RAG server
        rag_request = {
            "documents": request.documents,
            "questions": request.questions
        }
        
        # Get API key from settings
        api_key = getattr(settings, 'RAG_API_KEY', None)
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="RAG_API_KEY not configured in settings"
            )
        
        # Call RAG server
        async with httpx.AsyncClient(timeout=RAG_SERVER_TIMEOUT) as client:
            response = await client.post(
                RAG_SERVER_URL,
                json=rag_request,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"RAG server error: {response.text}"
                )
            
            rag_response = response.json()
            answers = rag_response.get("answers", [])
        
        processing_time = time.time() - start_time
        
        if settings.ENVIRONMENT.lower() == "production":
            return ProductionResponse(
                success=True,
                answers=answers
            )
        else:
            return Response(
                success=True,
                answers=answers,
                processing_time=processing_time,
                document_metadata={
                    "status": "processed",
                    "document_url": request.documents
                },
                raw_response=rag_response
            )
        
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="RAG server request timed out"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to RAG server: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns basic system status.
    """
    try:
        return HealthResponse(
            status="healthy",
            vector_store="not_configured",
            llm_provider=settings.DEFAULT_LLM_PROVIDER,
            document_count=0
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            vector_store="not_configured",
            llm_provider=settings.DEFAULT_LLM_PROVIDER
        )
