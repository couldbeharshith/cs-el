"""
Query Processor Module
Handles policy-related queries and generates answers using OpenAI
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class PolicyQueryProcessor:
    """Process and answer policy-related queries"""
    
    def __init__(self, config, graph_builder):
        self.config = config
        self.graph_builder = graph_builder
        self.client = OpenAI(api_key=config.openai_api_key)
    
    async def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a policy-related query"""
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Search graph for relevant facts
            search_results = await self.graph_builder.search(query)
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Generate answer using OpenAI
            answer = self._generate_answer(query, context)
            
            return {
                "query": query,
                "answer": answer,
                "sources": search_results[:5],  # Top 5 sources
                "context_found": len(search_results) > 0
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_found": False
            }
    
    def _prepare_context(self, search_results: List[Dict]) -> str:
        """Prepare context from search results"""
        if not search_results:
            return "No relevant information found in the policy database."
        
        context_parts = []
        for i, result in enumerate(search_results[:10], 1):  # Use top 10 results
            fact = result.get("fact", "")
            if fact:
                context_parts.append(f"Source {i}: {fact}")
        
        return "\n\n".join(context_parts) if context_parts else "No relevant context available."
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using OpenAI"""
        prompt = f"""
You are an expert insurance policy analyst. Answer the following query based on the policy information provided.

Instructions:
- Be specific and cite relevant policy sections when possible
- If you find specific amounts, percentages, or limits, quote them exactly
- If information is not available in the context, clearly state that
- Focus on practical, actionable information
- Use clear, professional language

Context from Policy Documents:
{context}

Query: {query}

Please provide a comprehensive answer based on the available policy information:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert insurance policy analyst who provides accurate, detailed answers based on policy documents."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        
        for query in queries:
            result = await self.answer_query(query)
            results.append(result)
        
        return results
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries for policy exploration"""
        return [
            "What are the room rent limits?",
            "What is the sum insured amount?",
            "What are the waiting periods?",
            "What conditions are excluded?",
            "How do I file a claim?",
            "What are the co-payment requirements?",
            "What is covered under maternity benefits?",
            "What are the age limits for coverage?",
            "What is the policy period?",
            "What are the premium payment terms?"
        ]