"""
Pydantic models for structured input/output between leader and worker agents.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TopicRequest(BaseModel):
    """Request model for a single decision tree topic."""
    
    topic: str = Field(description="The specific topic to generate a decision tree for")
    document_url: str = Field(description="URL of the document to analyze")
    focus_areas: Optional[List[str]] = Field(
        default=None, 
        description="Specific areas within the topic to focus on"
    )
    priority: int = Field(default=1, description="Priority level (1=highest, 5=lowest)")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "eligibility_criteria",
                "document_url": "https://example.com/insurance_doc.pdf",
                "focus_areas": ["age_requirements", "income_thresholds"],
                "priority": 1
            }
        }


class TopicBatchRequest(BaseModel):
    """Request model for multiple decision tree topics."""
    
    document_url: str = Field(description="URL of the document to analyze")
    topics: List[str] = Field(description="List of topics to process")
    batch_id: Optional[str] = Field(default=None, description="Unique identifier for this batch")
    
    class Config:
        schema_extra = {
            "example": {
                "document_url": "https://example.com/insurance_doc.pdf",
                "topics": [
                    "eligibility_criteria",
                    "coverage_benefits", 
                    "claim_process",
                    "exclusions_check"
                ],
                "batch_id": "batch_001"
            }
        }


class TopicResult(BaseModel):
    """Result model for a single decision tree topic."""
    
    topic: str = Field(description="The topic that was processed")
    success: bool = Field(description="Whether processing was successful")
    file_path: Optional[str] = Field(default=None, description="Path to the generated decision tree file")
    mermaid_path: Optional[str] = Field(default=None, description="Path to the generated Mermaid diagram")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "eligibility_criteria",
                "success": True,
                "file_path": "/path/to/eligibility_criteria.json",
                "mermaid_path": "/path/to/eligibility_criteria.mmd",
                "error_message": None,
                "processing_time": 15.3
            }
        }


class BatchResult(BaseModel):
    """Result model for batch processing of decision tree topics."""
    
    batch_id: Optional[str] = Field(default=None, description="Unique identifier for this batch")
    document_url: str = Field(description="URL of the document that was analyzed")
    total_topics: int = Field(description="Total number of topics requested")
    successful_topics: int = Field(description="Number of topics processed successfully")
    failed_topics: int = Field(description="Number of topics that failed processing")
    topic_results: List[TopicResult] = Field(description="Results for each individual topic")
    total_processing_time: Optional[float] = Field(default=None, description="Total time for batch processing")
    errors: List[str] = Field(default_factory=list, description="List of general errors encountered")
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_topics == 0:
            return 0.0
        return (self.successful_topics / self.total_topics) * 100
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_001",
                "document_url": "https://example.com/insurance_doc.pdf",
                "total_topics": 4,
                "successful_topics": 3,
                "failed_topics": 1,
                "topic_results": [],
                "total_processing_time": 45.7,
                "errors": []
            }
        }