from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.providers.base import BaseLLMProvider
from app.prompts.traditional_rag_prompt import TraditionalRagPrompt
from typing import List, Dict, Optional
import asyncio
import time
from langchain_community.retrievers import BM25Retriever

class RetrievalService:
    def __init__(self, vector_store: BaseVectorStore, llm_provider: BaseLLMProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
    def _find_optimal_k(self, scores: List[float], min_k: int = 4, max_k: int = 15, threshold: float = 0.1) -> tuple:
        """Find optimal k using dynamic similarity analysis"""
        if not scores or len(scores) <= 1:
            return min_k, "insufficient_data", [], []
        
        # Calculate relative drops between consecutive scores
        relative_drops = []
        for i in range(1, len(scores)):
            if scores[i-1] > 0:
                drop = (scores[i-1] - scores[i]) / scores[i-1]
                relative_drops.append(drop)
            else:
                relative_drops.append(1.0)
        
        # Adjust threshold based on score quality
        adaptive_threshold = threshold
        if scores[0] > 0.8:  # High-quality scores
            adaptive_threshold = max(0.08, threshold * 0.6)
        elif scores[0] > 0.6:  # Medium-quality scores  
            adaptive_threshold = max(0.1, threshold * 0.8)
        
        # Find significant drops
        significant_drops = []
        for i, drop in enumerate(relative_drops):
            if drop >= adaptive_threshold and (i + 2) >= min_k:
                significant_drops.append((i + 2, drop))
        
        # Quality maintenance - stop when quality drops below acceptable level
        quality_cutoff = None
        base_quality = scores[0]
        quality_threshold = base_quality * (0.75 if base_quality > 0.8 else 0.7)
        
        for i, score in enumerate(scores):
            if i >= min_k - 1 and score < quality_threshold:
                quality_cutoff = i + 1
                break
        
        # Gradual degradation detection
        gradual_cutoff = None
        cumulative_quality_loss = 0
        for i, drop in enumerate(relative_drops):
            cumulative_quality_loss += drop
            if cumulative_quality_loss > 0.25 and (i + 2) >= min_k:
                gradual_cutoff = i + 2
                break
        
        # Choose optimal k
        candidates = []
        if significant_drops:
            candidates.append((significant_drops[0][0], f"significant_drop_at_{significant_drops[0][0]}"))
        if quality_cutoff:
            candidates.append((quality_cutoff, f"quality_maintenance_at_{quality_cutoff}"))
        if gradual_cutoff:
            candidates.append((gradual_cutoff, f"gradual_degradation_at_{gradual_cutoff}"))
        
        if candidates:
            optimal_k, reason = min(candidates, key=lambda x: x[0])  # Choose earliest cutoff
        else:
            # Fallback based on average drop
            avg_drop = sum(relative_drops) / len(relative_drops) if relative_drops else 0
            if avg_drop < 0.03:
                optimal_k = min(len(scores), int(max_k * 0.8))
                reason = "stable_scores_using_80pct_max"
            elif avg_drop < 0.06:
                optimal_k = min_k + int((max_k - min_k) * 0.4)
                reason = "moderate_decline_middle_range"
            else:
                optimal_k = min_k + int((max_k - min_k) * 0.2)
                reason = "high_decline_conservative"
        
        # Apply guardrails
        optimal_k = max(min_k, min(optimal_k, max_k, len(scores)))
        
        return optimal_k, reason, scores, relative_drops
        
    async def process_document_queries(
        self, 
        document_id: str, 
        questions: List[str],
        k: Optional[int] = None  # If provided, uses static k; otherwise dynamic k
    ) -> Dict:
        
        async def process_single_question(i: int, question: str) -> Dict:
            try:
                # Step 1: Determine k (static or dynamic)
                if k is not None:
                    # Static k mode
                    final_k = k
                    retrieval_method = f"Hybrid Search (Static k={k})"
                    k_analysis = None
                else:
                    # Dynamic k mode - analyze scores to find optimal k
                    pre_search_results = await self.vector_store.asimilarity_search_with_score(
                        query=question, k=15, filter={"document_id": document_id}  # Get max for analysis
                    )
                    if not pre_search_results:
                        return {
                            "answer": "No relevant information found in the document.",
                            "debug_info": {
                                "question": question,
                                "answer": "No relevant information found in the document.",
                                "context_with_scores": [],
                                "chunks_count": 0,
                                "retrieval_method": "Hybrid Search (Dynamic)"
                            }
                        }
                    
                    scores = [score for _, score in pre_search_results]
                    final_k, reason, all_scores, relative_drops = self._find_optimal_k(scores)
                    retrieval_method = f"Hybrid Search (Dynamic k={final_k})"
                    k_analysis = {
                        "dynamic_k": final_k,
                        "selection_reason": reason,
                        "available_chunks": len(all_scores),
                        "top_scores": all_scores[:10],
                        "relative_drops": relative_drops[:5]  # First 5 drops for brevity
                    }
                
                # Step 2: Efficient Hybrid Retrieval (Question-Specific)
                # Vector retrieval - get semantically relevant candidates
                vector_candidates_size = max(50, final_k * 3) # Reasonable corpus
                vector_docs_with_scores = await self.vector_store.asimilarity_search_with_score(
                    query=question, k=vector_candidates_size, filter={"document_id": document_id}
                )
                vector_docs = [doc for doc, _ in vector_docs_with_scores]
                
                if not vector_docs:
                    return {
                        "answer": "No relevant information found in the document.",
                        "debug_info": {
                            "question": question,
                            "answer": "No relevant information found in the document.",
                            "context_with_scores": [],
                            "chunks_count": 0,
                            "retrieval_method": retrieval_method
                        }
                    }
                
                # BM25 retrieval - use only the semantically relevant candidates as corpus
                bm25_retriever = BM25Retriever.from_documents(vector_docs)
                bm25_retriever.k = final_k * 2  # Get top BM25 results from relevant corpus
                bm25_docs = bm25_retriever.invoke(question)
                
                # Step 3: Combine and deduplicate
                combined_docs = []
                seen_content = set()
                
                # Add top vector docs first (semantic relevance priority)
                top_vector_docs = vector_docs[:final_k * 2]  # Limit vector candidates
                for doc in top_vector_docs:
                    content_key = doc.page_content[:200]  # More robust dedup key
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        combined_docs.append(doc)
                
                # Add unique BM25 docs (keyword matching diversity)
                for doc in bm25_docs:
                    content_key = doc.page_content[:200]
                    if content_key not in seen_content and len(combined_docs) < final_k * 3:  # Limit total candidates
                        seen_content.add(content_key)
                        combined_docs.append(doc)
                
                # Step 4: Combine selected
                final_docs = combined_docs[:final_k]
                
                if not final_docs:
                    return {
                        "answer": "No relevant information found in the document.",
                        "debug_info": {
                            "question": question,
                            "answer": "No relevant information found in the document.",
                            "context_with_scores": [],
                            "chunks_count": 0,
                            "retrieval_method": retrieval_method
                        }
                    }
                
                # Step 5: Generate answer
                llm = self.llm_provider.get_langchain_llm()
                traditional_rag_prompt = TraditionalRagPrompt.get_traditional_rag_prompt()

                context = "\n\n".join([doc.page_content for doc in final_docs])
                prompt = traditional_rag_prompt.format(context=context)

                answer = await asyncio.to_thread(
                    lambda: llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": question}]).content
                )
                
                # Create context with scores (reranked positions as proxy scores)
                context_with_scores = []
                for idx, doc in enumerate(final_docs):
                    context_with_scores.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": 1.0 - (idx * 0.05)  # Decreasing score based on rerank position
                    })
                
                # Build debug info
                debug_info = {
                    "question": question,
                    "answer": answer,
                    "context_documents": [doc.page_content for doc in final_docs],
                    "context_with_scores": context_with_scores,
                    "chunks_count": len(final_docs),
                    "retrieval_method": retrieval_method,
                    "hybrid_stats": {
                        "vector_candidates_retrieved": len(vector_docs),
                        "bm25_docs_found": len(bm25_docs),
                        "combined_before_rerank": len(combined_docs),
                        "final_after_rerank": len(final_docs),
                        "efficiency_note": f"BM25 corpus size: {len(vector_docs)} (question-specific vs 1000 global)"
                    }
                }
                
                # Add dynamic k analysis if applicable
                if k_analysis:
                    debug_info.update(k_analysis)
                
                return {
                    "answer": answer,
                    "debug_info": debug_info
                }
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                result = {
                    "answer": error_msg,
                    "debug_info": {
                        "question": question,
                        "answer": error_msg,
                        "chunks_count": 0,
                        "error": str(e),
                        "retrieval_method": "Hybrid Search (Error)"
                    }
                }
                print(f"Error: {error_msg}")
                return result
        
        print(f"Processing {len(questions)} questions in parallel with hybrid search...")
        start_time = time.time()
        
        tasks = [process_single_question(i, question) for i, question in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        answers = []
        debug_info = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error processing question {i+1}: {str(result)}"
                answers.append(error_msg)
                debug_info.append({
                    "question": questions[i],
                    "answer": error_msg,
                    "chunks_count": 0,
                    "error": str(result),
                    "retrieval_method": "Hybrid Search + Reranker (Exception)"
                })
            else:
                answers.append(result["answer"])
                debug_info.append(result["debug_info"])
        
        return {
            "answers": answers,
            "debug_info": debug_info
        }
    
    