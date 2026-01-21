def get_decision_tree_prompt(document_url: str) -> str:
    return f"""
        You are tasked with generating decision trees for an insurance document.
        
        Document URL: {document_url}
        
        Generate decision trees for these topics:
        1. eligibility_criteria
        2. coverage_benefits
        3. claim_process
        4. exclusions_check
        5. waiting_periods
        6. pre_authorization
        7. network_providers
        8. deductibles
        
        IMPORTANT: Each decision tree must follow this exact JSON structure:
        
        {{
            "tree_id": "unique_id",
            "name": "topic_name",
            "description": "Brief description of what this tree covers",
            "root_node_id": "root_node_id",
            "nodes": {{
                "root_node_id": {{
                    "node_id": "root_node_id",
                    "node_type": "root",
                    "question": "Main question for this topic",
                    "left_child": null,
                    "right_child": "next_node_id",
                    "answer": null,
                    "reasoning": null,
                    "confidence": 0.95,
                    "source_references": ["page/section references"],
                    "metadata": {{"topic": "topic_name"}}
                }},
                "decision_node_id": {{
                    "node_id": "decision_node_id",
                    "node_type": "decision",
                    "question": "Yes/No decision question",
                    "left_child": "no_path_node",
                    "right_child": "yes_path_node",
                    "answer": null,
                    "reasoning": "Why this decision point matters",
                    "confidence": 0.9,
                    "source_references": ["page/section references"],
                    "metadata": {{"branch": "yes/no"}}
                }},
                "leaf_node_id": {{
                    "node_id": "leaf_node_id",
                    "node_type": "leaf",
                    "question": "Final outcome question",
                    "left_child": null,
                    "right_child": null,
                    "answer": "Detailed answer with policy details",
                    "reasoning": "Explanation of this outcome",
                    "confidence": 0.95,
                    "source_references": ["page/section references"],
                    "metadata": {{"outcome": "approved/denied/conditional"}}
                }}
            }},
            "document_source": "{document_url}",
            "creation_timestamp": "2025-09-13T00:00:00Z",
            "version": "1.0",
            "tags": ["insurance", "decision_tree", "topic_name"]
        }}
        
        For each topic:
        1. Use the query_document tool to get relevant information
        2. Use the save_decision_tree tool to create and save the decision tree
        
        Process all topics and provide a summary of results.
        """