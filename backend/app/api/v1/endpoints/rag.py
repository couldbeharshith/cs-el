from time import perf_counter

from fastapi import APIRouter, HTTPException

from app.config.settings import settings
from app.models.request import Request
from app.models.response import HealthResponse, ProductionResponse, Response
from integrations.insurance_rag import InsuranceRagError, answer_questions

router = APIRouter()


@router.post("/run", response_model=ProductionResponse | Response)
async def run_rag(request: Request):
    started = perf_counter()
    try:
        result = await answer_questions(request.documents, request.questions)
    except InsuranceRagError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    if settings.ENVIRONMENT.lower() == "production":
        return ProductionResponse(success=True, answers=result["answers"])

    return Response(
        success=True,
        answers=result["answers"],
        processing_time=perf_counter() - started,
        document_metadata={"status": "processed", "document_url": request.documents},
        raw_response=result,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        vector_store="not_configured",
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        document_count=0,
    )
