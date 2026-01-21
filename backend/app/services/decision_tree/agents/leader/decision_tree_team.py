import time
from typing import List, Dict, Any, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from ..models.topic_models import TopicRequest, TopicBatchRequest, TopicResult, BatchResult
from ..workers.decision_tree_worker import DecisionTreeWorkerAgent
from dotenv import load_dotenv
load_dotenv()



class DecisionTreeLeaderTeam:
    def __init__(self):
        self.worker_agent = DecisionTreeWorkerAgent()
        self.team = None
        self._initialize_team()
    
    def _initialize_team(self):
        # Create the team with proper delegation settings
        self.team = Team(
            name="DecisionTreeLeaderTeam",
            model=OpenAIChat(id="gpt-4.1-mini"),
            members=[self.worker_agent],
            instructions=[
                "You are the leader of a decision tree generation team.",
                "You coordinate a worker agent to process insurance document topics sequentially.",
                "The worker specializes in processing individual decision tree topics.",
                "Process topics one by one in the order they are provided.",
                "Ensure each topic is completed successfully before moving to the next.",
                "Collect results from the worker and provide comprehensive summaries."
            ],
            determine_input_for_members=False,  # Send structured input directly to worker
            delegate_task_to_all_members=False,  # Process one topic at a time
            show_members_responses=True,
            markdown=True
        )
    
    async def process_batch(self, batch_request: TopicBatchRequest) -> BatchResult:
        start_time = time.time()
        
        try:
            # Set up the worker for this document
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.worker_agent.setup_for_document(batch_request.document_url, timestamp)
            
            topic_results = []
            
            # Process topics sequentially using the team
            for topic in batch_request.topics:
                topic_request = TopicRequest(
                    topic=topic,
                    document_url=batch_request.document_url,
                    priority=1
                )
                
                try:
                    # Generate prompt for this specific topic and let the team handle it
                    prompt = self._generate_topic_prompt(topic, batch_request.document_url)
                    
                    # Use the team to process the topic
                    result = await self.team.arun(prompt)
                    
                    # Create a successful result (simplified for team processing)
                    topic_results.append(TopicResult(
                        topic=topic,
                        success=True,
                        file_path=None,  # Will be determined by the save tool
                        mermaid_path=None,
                        error_message=None,
                        processing_time=0.0  # Team handles timing
                    ))
                    
                except Exception as e:
                    topic_results.append(TopicResult(
                        topic=topic,
                        success=False,
                        file_path=None,
                        mermaid_path=None,
                        error_message=str(e),
                        processing_time=0.0
                    ))
            
            # Calculate summary statistics
            successful_topics = sum(1 for result in topic_results if result.success)
            failed_topics = len(topic_results) - successful_topics
            total_processing_time = time.time() - start_time
            
            # Collect any general errors
            errors = [result.error_message for result in topic_results if result.error_message]
            
            return BatchResult(
                batch_id=batch_request.batch_id,
                document_url=batch_request.document_url,
                total_topics=len(batch_request.topics),
                successful_topics=successful_topics,
                failed_topics=failed_topics,
                topic_results=topic_results,
                total_processing_time=total_processing_time,
                errors=errors
            )
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            return BatchResult(
                batch_id=batch_request.batch_id,
                document_url=batch_request.document_url,
                total_topics=len(batch_request.topics),
                successful_topics=0,
                failed_topics=len(batch_request.topics),
                topic_results=[],
                total_processing_time=total_processing_time,
                errors=[str(e)]
            )
    
    def _generate_topic_prompt(self, topic: str, document_url: str) -> str:
        """Generate prompt for a specific topic"""
        current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"""
        You are tasked with generating a decision tree for a specific insurance topic.
        
        Document URL: {document_url}
        Topic: {topic}
        Current Timestamp: {current_timestamp}
        
        IMPORTANT: Generate exactly ONE decision tree for the topic: {topic}
        
        Steps:
        1. Use the query_document tool to get information specifically about {topic}
        2. Create a comprehensive decision tree with multiple decision points if needed
        3. Use the save_decision_tree tool to save the tree as "{topic}.json"
        
        When creating the decision tree:
        - Replace {{topic}} with: {topic}
        - Replace {{document_url}} with: {document_url}  
        - Replace {{current_timestamp}} with: {current_timestamp}
        - Focus only on {topic} and provide detailed, accurate decision paths based on the document content
        - Ensure each decision node has clear yes/no paths
        - Include proper confidence scores and source references
        - Make sure leaf nodes provide actionable outcomes
        """
    
    async def process_single_topic(self, topic_request: TopicRequest) -> TopicResult:
        """Process a single topic using the team"""
        try:
            # Set up the worker for this document
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.worker_agent.setup_for_document(topic_request.document_url, timestamp)
            
            # Generate prompt and process with team
            prompt = self._generate_topic_prompt(topic_request.topic, topic_request.document_url)
            result = await self.team.arun(prompt)
            
            return TopicResult(
                topic=topic_request.topic,
                success=True,
                file_path=None,  # Will be determined by the save tool
                mermaid_path=None,
                error_message=None,
                processing_time=0.0
            )
            
        except Exception as e:
            return TopicResult(
                topic=topic_request.topic,
                success=False,
                file_path=None,
                mermaid_path=None,
                error_message=str(e),
                processing_time=0.0
            )
    
    def get_team_status(self) -> Dict[str, Any]:
        return {
            "team_name": self.team.name if self.team else "Not initialized",
            "worker_count": 1,
            "team_model": "gpt-4.1-mini",
            "processing_mode": "sequential_delegation"
        }


async def process_topics_with_team(document_url: str, topics: List[str], batch_id: Optional[str] = None) -> BatchResult:
    """Convenience function to process multiple topics sequentially with the team"""
    team = DecisionTreeLeaderTeam()
    
    batch_request = TopicBatchRequest(
        document_url=document_url,
        topics=topics,
        batch_id=batch_id
    )
    
    return await team.process_batch(batch_request)


async def process_single_topic_with_team(document_url: str, topic: str) -> TopicResult:
    """Convenience function to process a single topic with the team"""
    team = DecisionTreeLeaderTeam()
    
    topic_request = TopicRequest(
        topic=topic,
        document_url=document_url,
        priority=1
    )
    
    return await team.process_single_topic(topic_request)