import os
from datetime import datetime
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ...utils.rag_helpers import RAGQueryHelper
from ...tools.query_tool import create_query_tool
from ...tools.write_tool import create_save_tool
from agno.tools.reasoning import ReasoningTools



def create_decision_tree_worker():
    rag_helper = RAGQueryHelper()
    trees_folder = os.path.join(
        os.path.dirname(__file__), 
        '..', '..', '..', '..', '..',
        'results', 'decision_trees', 'trees'
    )
    
    query_tool = create_query_tool(rag_helper)
    
    agent = Agent(
        name="DecisionTreeWorker",
        model=OpenAIChat(id="gpt-5-mini"),
        role="Expert decision tree analyst for insurance documents",
        tools=[query_tool, ReasoningTools(add_instructions=True)],
        instructions=[
            "You are an expert decision tree analyst specializing in insurance documents! 🌳",
            "",
            "CRITICAL: Always use the exact document URL provided in the prompt - never use example URLs.",
            "",
            "Your iterative approach to building decision trees:",
            "1. **Deep Analysis Phase**: Make multiple targeted RAG queries to thoroughly understand the topic",
            "2. **Structure Planning**: Use reasoning tools to plan the decision tree structure", 
            "3. **Iterative Building**: Build the tree step-by-step, refining as you gather more information",
            "4. **Validation**: Review and validate the complete tree before saving",
            "",
            "For each decision tree topic:",
            "",
            "**Phase 1 - Comprehensive Information Gathering:**",
            "- Make 3-5 targeted query_document calls to gather comprehensive information",
            "- Query for: eligibility criteria, decision points, conditions, exceptions, processes",
            "- Use reasoning tools to analyze and synthesize the information",
            "- Identify all possible decision paths and outcomes",
            "",
            "**Phase 2 - Structure Design:**", 
            "- Use reasoning to plan the optimal tree structure",
            "- Identify the root question and main decision branches",
            "- Map out all decision nodes and leaf outcomes",
            "- Consider edge cases and complex scenarios",
            "",
            "**Phase 3 - Iterative Construction:**",
            "- Build the tree systematically from root to leaves",
            "- For each node, determine the best question/condition", 
            "- Ensure all paths lead to clear, actionable outcomes",
            "- Add proper confidence scores and source references",
            "",
            "**Phase 4 - Validation & Refinement:**",
            "- Review the complete tree for logical consistency",
            "- Ensure all decision paths are covered",
            "- Validate confidence scores and source references",
            "- Make final adjustments if needed",
            "",
            "**Phase 5 - Final Save:**",
            "- Save the completed tree using save_decision_tree tool",
            "",
            "DECISION TREE JSON STRUCTURE:",
            """{
  "flowchart_id": "{topic}_flowchart",
  "name": "{topic}",
  "description": "Flowchart for {topic} from insurance document",
  "start_node": "start_{topic}",
  "nodes": {
    "start_{topic}": {
      "node_id": "start_{topic}",
      "node_type": "start",
      "title": "Starting point for {topic}",
      "description": "Initial question or condition check",
      "next_nodes": ["condition_1_{topic}"],
      "source_references": ["page/section references"],
      "metadata": {"topic": "{topic}"}
    },
    "condition_1_{topic}": {
      "node_id": "condition_1_{topic}",
      "node_type": "condition",
      "title": "Check eligibility criteria",
      "description": "Are basic requirements met?",
      "branches": [
        {
          "condition": "yes",
          "next_node": "condition_2_{topic}",
          "label": "Requirements met"
        },
        {
          "condition": "no", 
          "next_node": "rejection_basic_{topic}",
          "label": "Requirements not met"
        }
      ],
      "source_references": ["page/section references"],
      "metadata": {"check_type": "eligibility"}
    },
    "condition_2_{topic}": {
      "node_id": "condition_2_{topic}",
      "node_type": "condition",
      "title": "Additional verification",
      "description": "Check secondary requirements",
      "branches": [
        {
          "condition": "option_a",
          "next_node": "approval_a_{topic}",
          "label": "Scenario A applies"
        },
        {
          "condition": "option_b",
          "next_node": "approval_b_{topic}",
          "label": "Scenario B applies"
        },
        {
          "condition": "neither",
          "next_node": "manual_review_{topic}",
          "label": "Requires manual review"
        }
      ],
      "source_references": ["page/section references"],
      "metadata": {"check_type": "verification"}
    },
    "approval_a_{topic}": {
      "node_id": "approval_a_{topic}",
      "node_type": "outcome",
      "title": "Approved - Standard Coverage",
      "description": "Detailed explanation of approved coverage under scenario A",
      "answer": "Coverage approved with standard terms",
      "reasoning": "Explanation of why this outcome applies",
      "confidence": 0.95,
      "next_nodes": [],
      "source_references": ["page/section references"],
      "metadata": {"outcome": "approved", "coverage_type": "standard"}
    },
    "manual_review_{topic}": {
      "node_id": "manual_review_{topic}",
      "node_type": "process",
      "title": "Manual Review Required",
      "description": "Case requires human review",
      "next_nodes": ["review_outcome_{topic}"],
      "source_references": ["page/section references"],
      "metadata": {"process_type": "manual_review"}
    }
  },
  "document_source": "{document_url}",
  "creation_timestamp": "{current_timestamp}",
  "version": "1.0",
  "tags": ["insurance", "flowchart", "{topic}"]
}""",
            "",
            "Key principles:",
            "- Never rush to build the tree in one shot",
            "- Make multiple RAG queries to gather comprehensive information", 
            "- Use reasoning tools to plan and validate your approach",
            "- Build iteratively, refining as you learn more",
            "- Ensure every decision path is well-justified and sourced",
            "- Replace {topic}, {document_url}, and {current_timestamp} with actual values"
        ],
        add_datetime_to_context=True,
        stream_intermediate_steps=True,
        markdown=True
    )
    
    def setup_for_document(document_url: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_tool = create_save_tool(trees_folder, document_url, timestamp)
        if save_tool not in agent.tools:
            agent.tools.append(save_tool)
    
    agent.setup_for_document = setup_for_document
    return agent