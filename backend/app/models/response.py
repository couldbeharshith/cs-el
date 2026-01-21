from pydantic import BaseModel
from typing import List, Dict, Optional

class Response(BaseModel):
    success: bool
    answers: List[str]
    processing_time: Optional[float] = None
    document_metadata: Optional[Dict] = None
    raw_response: Optional[Dict] = None  

class ProductionResponse(BaseModel):
    success: bool
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    llm_provider: str
    document_count: Optional[int] = None
