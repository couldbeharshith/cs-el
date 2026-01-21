from agno.team.team import Team
from agno.models.openai import OpenAIChat
from ..workers.simple_worker import create_decision_tree_worker
from datetime import datetime


def create_decision_tree_team():
    worker_agent = create_decision_tree_worker()
    
    team = Team(
        name="DecisionTreeTeam",
        model=OpenAIChat(id="gpt-4.1-mini"),
        members=[worker_agent],
        instructions=[
            "You coordinate decision tree creation for specific topics from insurance documents.",
            "Use DecisionTreeWorker to create decision trees for provided topics.",
            "IMPORTANT: Always use the exact document URL provided in the prompt.",
            "Never use example URLs or placeholder URLs.",
            "Process each topic thoroughly and ensure all decision trees are properly saved."
        ]
    )
    return team, worker_agent


async def process_single_topic(document_url: str, topic: str, timestamp: str):
    team, worker_agent = create_decision_tree_team()
    
    worker_agent.setup_for_document(document_url, timestamp)
    
    prompt = f"""
    Create a decision tree for the topic: {topic}
    
    IMPORTANT: Use this EXACT document URL for all queries: {document_url}
    Do NOT use any example or placeholder URLs.
    
    Steps:
    1. Use query_document with document_url="{document_url}" to get detailed information about {topic}
    2. Create a comprehensive decision tree following the exact JSON structure
    3. Save the decision tree using save_decision_tree tool
    
    Focus specifically on: {topic}
    Document URL to use: {document_url}
    """
    
    result = await team.arun(prompt)
    return result

