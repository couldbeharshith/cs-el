import os
from datetime import datetime
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ...utils.rag_helpers import RAGQueryHelper
from ...tools.query_tool import create_query_tool
from ...tools.write_tool import create_save_tool
from dotenv import load_dotenv
load_dotenv()


class DecisionTreeWorkerAgent(Agent):
    def __init__(self):
        self.rag_helper = RAGQueryHelper()
        self.trees_folder = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..', '..',
            'results', 'decision_trees', 'trees'
        )
        
        # Initialize as an Agno Agent
        super().__init__(
            name="DecisionTreeWorker",
            model=OpenAIChat(id="gpt-4.1-mini"),
            role="Specialist for generating decision trees from insurance documents",
            instructions=[
                "You are an expert at creating decision trees from insurance documents.",
                "You have access to query_document and save_decision_tree tools.",
                "Use query_document to get information about the topic from the insurance document.",
                "Create a comprehensive decision tree with clear decision points and outcomes.",
                "Each node should have confidence scores and source references.",
                "Save the decision tree using the save_decision_tree tool.",
                "Provide detailed reasoning for each decision point and outcome.",
                "Ensure all paths lead to clear, actionable conclusions.",
                "",
                "IMPORTANT: The decision tree must follow this exact JSON structure:",
                "{",
                '    "tree_id": "{topic}_tree",',
                '    "name": "{topic}",',
                '    "description": "Decision tree for {topic} from insurance document",',
                '    "root_node_id": "root_{topic}",',
                '    "nodes": {',
                '        "root_{topic}": {',
                '            "node_id": "root_{topic}",',
                '            "node_type": "root",',
                '            "question": "Main question for {topic}",',
                '            "left_child": null,',
                '            "right_child": "decision_1_{topic}",',
                '            "answer": null,',
                '            "reasoning": null,',
                '            "confidence": 0.95,',
                '            "source_references": ["page/section references"],',
                '            "metadata": {"topic": "{topic}"}',
                '        },',
                '        "decision_1_{topic}": {',
                '            "node_id": "decision_1_{topic}",',
                '            "node_type": "decision",',
                '            "question": "First decision point for {topic}",',
                '            "left_child": "no_path_{topic}",',
                '            "right_child": "yes_path_{topic}",',
                '            "answer": null,',
                '            "reasoning": "Why this decision point matters for {topic}",',
                '            "confidence": 0.9,',
                '            "source_references": ["page/section references"],',
                '            "metadata": {"branch": "decision_1"}',
                '        },',
                '        "yes_path_{topic}": {',
                '            "node_id": "yes_path_{topic}",',
                '            "node_type": "leaf",',
                '            "question": "Outcome when condition is met",',
                '            "left_child": null,',
                '            "right_child": null,',
                '            "answer": "Detailed answer for positive outcome in {topic}",',
                '            "reasoning": "Explanation of this positive outcome",',
                '            "confidence": 0.95,',
                '            "source_references": ["page/section references"],',
                '            "metadata": {"outcome": "approved"}',
                '        },',
                '        "no_path_{topic}": {',
                '            "node_id": "no_path_{topic}",',
                '            "node_type": "leaf",',
                '            "question": "Outcome when condition is not met",',
                '            "left_child": null,',
                '            "right_child": null,',
                '            "answer": "Detailed answer for negative outcome in {topic}",',
                '            "reasoning": "Explanation of this negative outcome",',
                '            "confidence": 0.95,',
                '            "source_references": ["page/section references"],',
                '            "metadata": {"outcome": "denied"}',
                '        }',
                '    },',
                '    "document_source": "{document_url}",',
                '    "creation_timestamp": "{current_timestamp}",',
                '    "version": "1.0",',
                '    "tags": ["insurance", "decision_tree", "{topic}"]',
                "}",
                "",
                "Replace {topic}, {document_url}, and {current_timestamp} with actual values.",
                "Focus only on the assigned topic and provide detailed, accurate decision paths."
            ],
            tools=self._get_tools()
        )
    
    def _get_tools(self):
        """Get the tools needed for decision tree processing"""
        query_tool = create_query_tool(self.rag_helper)
        # We'll set up the save tool when we get the document URL
        return [query_tool]
    
    def setup_for_document(self, document_url: str, timestamp: str = None):
        """Set up the save tool for a specific document"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_tool = create_save_tool(self.trees_folder, document_url, timestamp)
        # Add the save tool to existing tools
        if save_tool not in self.tools:
            self.tools.append(save_tool)