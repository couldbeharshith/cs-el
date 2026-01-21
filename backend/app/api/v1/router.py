from fastapi import APIRouter
from app.api.v1.endpoints.rag import router as rag_router

api_router = APIRouter()

api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
