"""
Models module for agent communication.
"""
from .topic_models import TopicRequest, TopicBatchRequest, TopicResult, BatchResult

__all__ = [
    "TopicRequest",
    "TopicBatchRequest", 
    "TopicResult",
    "BatchResult"
]