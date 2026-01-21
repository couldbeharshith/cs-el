from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ...utils.rag_helpers import RAGQueryHelper
from ...tools.query_tool import create_query_tool
from ..models.topic_output import ExtractedTopics, TopicInfo


def create_topic_explorer():
    rag_helper = RAGQueryHelper()
    query_tool = create_query_tool(rag_helper)
    
    return Agent(
        name="TopicExplorer",
        model=OpenAIChat(id="gpt-4.1-mini"),
        output_schema=ExtractedTopics,
        role="Extract decision tree topics from insurance documents with structured analysis",
        instructions=[
            "You analyze insurance documents to extract potential decision tree topics.",
            "Use query_document tool to make 4-5 targeted queries to get comprehensive information.",
            "Make these specific queries:",
            "1. Query for eligibility criteria and requirements",
            "2. Query for claim processes and approval workflows", 
            "3. Query for coverage scenarios and benefit calculations",
            "4. Query for exclusions, limitations, and conditional policies",
            "5. Query for risk assessment and decision-making procedures",
            "Extract 6-10 high-quality topics suitable for decision tree creation.",
            "Focus on areas with clear decision points and conditional logic.",
            "Classify decision complexity as Simple, Moderate, or Complex.",
            "Provide structured output with all required fields filled."
        ],
        tools=[query_tool]
    )