from pydantic import BaseModel, validator
from typing import List, Optional

class Request(BaseModel):
    documents: str 
    questions: List[str]
    k: Optional[int] = None  # None = dynamic k, int value = static k
    processing_mode: Optional[str] = "traditional"
    
    @validator('processing_mode')
    def validate_processing_mode(cls, v):
        from app.services.pipelines.pipeline_manager import PipelineManager
        
        if v is not None and not PipelineManager.is_supported_pipeline(v):
            supported = list(PipelineManager.get_supported_pipelines().keys())
            raise ValueError(f"Unsupported processing_mode '{v}'. Supported modes: {supported}")
        
        return v or PipelineManager.get_default_pipeline()