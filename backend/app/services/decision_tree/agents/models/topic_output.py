from typing import List
from pydantic import BaseModel, Field


class TopicInfo(BaseModel):
    topic_name: str = Field(description="Clear, concise topic name suitable for decision trees")
    description: str = Field(description="Brief description of what this topic covers")
    decision_complexity: str = Field(description="Simple, Moderate, or Complex")
    key_decision_points: List[str] = Field(description="Main decision points identified for this topic")


class ExtractedTopics(BaseModel):
    document_analyzed: str = Field(description="The document URL that was analyzed")
    total_topics_found: int = Field(description="Total number of topics extracted")
    topics: List[TopicInfo] = Field(description="List of all extracted topics")
    analysis_summary: str = Field(description="Overall summary of the document analysis")