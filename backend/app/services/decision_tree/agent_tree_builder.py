import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from .agents import (
    DecisionTreeLeaderTeam,
    process_topics_with_team,
    TopicBatchRequest,
    BatchResult
)


class AgentBasedDecisionTreeBuilder:
    def __init__(self):
        self.default_topics = [
            "eligibility_criteria",

        ]
    
    async def run(self, document_url: str, topics: Optional[List[str]] = None) -> BatchResult:
        if topics is None:
            topics = self.default_topics
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"agent_batch_{timestamp}"
        
        print(f"Starting Agent-Based Decision Tree Builder")
        print(f"Document: {document_url}")
        print(f"Topics: {', '.join(topics)}")
        print(f"Batch ID: {batch_id}")
        print(f"Processing Mode: Sequential")
        
        result = await process_topics_with_team(
            document_url=document_url,
            topics=topics,
            batch_id=batch_id
        )
        
        print(f"\nProcessing Complete:")
        print(f"  Successful: {result.successful_topics}/{result.total_topics}")
        print(f"  Success Rate: {result.success_rate:.1f}%")
        print(f"  Total Time: {result.total_processing_time:.2f}s")

        return result
    
    async def run_single_topic(self, document_url: str, topic: str):
        from .agents import process_single_topic_with_team
        
        print(f"Processing single topic: {topic}")
        result = await process_single_topic_with_team(document_url, topic)

        print(f"Result: {'✅ Success' if result.success else '❌ Failed'}")
        if result.error_message:
            print(f" Error: {result.error_message}")
        
        return result


async def run_with_agents(document_url: str, topics: Optional[List[str]] = None) -> BatchResult:
    builder = AgentBasedDecisionTreeBuilder()
    return await builder.run(document_url, topics)


class DirectTeamInterface:
    def __init__(self):
        self.team = DecisionTreeLeaderTeam()
    
    async def process_custom_batch(self, batch_request: TopicBatchRequest) -> BatchResult:
        return await self.team.process_batch(batch_request)
    
    def get_team_status(self) -> Dict[str, Any]:
        return self.team.get_team_status()
    
    async def process_with_priorities(self, document_url: str, topic_priorities: Dict[str, int]) -> BatchResult:
        sorted_topics = sorted(topic_priorities.items(), key=lambda x: x[1], reverse=True)
        topics = [topic for topic, _ in sorted_topics]
        
        print(f"Processing topics in priority order: {topics}")

        batch_request = TopicBatchRequest(
            document_url=document_url,
            topics=topics,
            batch_id=f"priority_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return await self.process_custom_batch(batch_request)