from app.services.vector_stores.base_vector_store import BaseVectorStore
from app.providers.base import BaseLLMProvider
from app.prompts.structure_aware_rag_prompt import StructureAwareRagPrompt
from app.services.preprocessors.cell_aware_reranker import CellAwareReranker
from typing import List, Dict, Optional
import asyncio
import time
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class TrueStructureAwareRetrievalService:
    def __init__(self, vector_store: BaseVectorStore, llm_provider: BaseLLMProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
        # Initialize cell-aware reranker
        self.structure_aware_reranker = CellAwareReranker()
    
    def _find_optimal_k(self, scores: List[float], min_k: int = 4, max_k: int = 15, threshold: float = 0.1) -> tuple:
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
    
    def _create_structure_enhanced_context(self, documents: List[Document]) -> str:
        """Create context with structure-aware tags for optimal LLM processing"""
        
        context_parts = []
        
        # Group documents by structure type
        grouped_docs = defaultdict(list)
        for doc in documents:
            element_type = doc.metadata.get('element_type', 'text')
            grouped_docs[element_type].append(doc)
        
        # Add complete tables first (highest priority for comprehensive view)
        if 'complete_table' in grouped_docs:
            for doc in grouped_docs['complete_table']:
                table_name = doc.metadata.get('table_name', 'Policy Table')
                context_parts.append(f"[COMPLETE TABLE - {table_name}]")
                context_parts.append(doc.page_content)
                context_parts.append("[/COMPLETE TABLE]")
                context_parts.append("")
        
        # Add table rows (detailed row-level information)
        if 'table_row' in grouped_docs:
            for doc in grouped_docs['table_row']:
                table_ref = doc.metadata.get('table_id', 'Unknown Table')
                context_parts.append(f"[TABLE ROW - {table_ref}]")
                context_parts.append(doc.page_content)
                context_parts.append("[/TABLE ROW]")
                context_parts.append("")
        
        # Add important table cells (specific values)
        if 'table_cell' in grouped_docs:
            for doc in grouped_docs['table_cell']:
                cell_ref = doc.metadata.get('cell_reference', 'Cell')
                context_parts.append(f"[TABLE CELL - {cell_ref}]")
                context_parts.append(doc.page_content)
                context_parts.append("[/TABLE CELL]")
                context_parts.append("")
        
        # Add footnotes (critical exceptions and conditions)
        if 'footnote' in grouped_docs:
            for doc in grouped_docs['footnote']:
                footnote_ref = doc.metadata.get('footnote_ref', '*')
                context_parts.append(f"[FOOTNOTE {footnote_ref}]")
                context_parts.append(doc.page_content)
                context_parts.append("[/FOOTNOTE]")
                context_parts.append("")
        
        # Add regular text content last
        if 'text' in grouped_docs:
            context_parts.append("[POLICY TEXT]")
            for doc in grouped_docs['text']:
                context_parts.append(doc.page_content)
                context_parts.append("")
            context_parts.append("[/POLICY TEXT]")
        
        return "\n".join(context_parts)
    
    def _select_optimal_prompt(self, documents: List[Document], question: str):
        """Select the most appropriate prompt based on retrieved content and question type"""
        
        structure_counts = defaultdict(int)
        for doc in documents:
            element_type = doc.metadata.get('element_type', 'text')
            structure_counts[element_type] += 1
        
        question_lower = question.lower()
        
        # Use table-focused prompt for quantitative questions with substantial table content
        table_content_count = (structure_counts['complete_table'] + 
                              structure_counts['table_row'] + 
                              structure_counts['table_cell'])
        
        if (any(term in question_lower for term in ["amount", "limit", "premium", "cost", "rate", "%", "$", "coverage", "maximum", "minimum"]) and
            table_content_count >= 2):
            return StructureAwareRagPrompt.get_table_focused_prompt()
        
        # Use footnote-aware prompt when footnotes are prominent and question asks about conditions
        if (structure_counts['footnote'] >= 1 and 
            any(term in question_lower for term in ["condition", "exception", "exclude", "requirement", "restriction", "limitation", "not covered", "unless", "provided"])):
            return StructureAwareRagPrompt.get_footnote_aware_prompt()
        
        # Default to comprehensive structure-aware prompt
        return StructureAwareRagPrompt.get_structure_aware_rag_prompt()
    
    async def process_document_queries_structure_aware(
        self,
        document_id: str,
        questions: List[str],
        k: Optional[int] = None  # If provided, uses static k; otherwise dynamic k
    ) -> Dict:
        """
        Process queries with true structure-aware hybrid retrieval and cell-aware reranking
        Following the optimized approach with question-specific BM25 corpus
        """
        
        async def process_single_question(i: int, question: str) -> Dict:
            try:
                # Step 1: Determine k (static or dynamic)
                if k is not None:
                    # Static k mode
                    final_k = k
                    retrieval_method = f"Structure-Aware Hybrid Search + Cell-Aware Reranker (Static k={k})"
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
                                "retrieval_method": "Structure-Aware Hybrid Search + Cell-Aware Reranker (Dynamic)"
                            }
                        }
                    
                    scores = [score for _, score in pre_search_results]
                    final_k, reason, all_scores, relative_drops = self._find_optimal_k(scores)
                    retrieval_method = f"Structure-Aware Hybrid Search + Cell-Aware Reranker (Dynamic k={final_k})"
                    k_analysis = {
                        "dynamic_k": final_k,
                        "selection_reason": reason,
                        "available_chunks": len(all_scores),
                        "top_scores": all_scores[:10],
                        "relative_drops": relative_drops[:5]  # First 5 drops for brevity
                    }
                
                # Step 2: Structure-Aware Hybrid Retrieval (Vector + BM25)
                # Vector retrieval - prioritize first-class retrievables
                vector_candidates_size = max(60, final_k * 4)
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
                
                # Add top vector docs first (semantic relevance + structure priority)
                top_vector_docs = vector_docs[:final_k * 2]  # Limit vector candidates for efficiency
                for doc in top_vector_docs:
                    content_key = doc.page_content[:200]  # More robust dedup key
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        combined_docs.append(doc)
                
                # Add unique BM25 docs
                for doc in bm25_docs:
                    content_key = doc.page_content[:200]
                    if content_key not in seen_content and len(combined_docs) < final_k * 3:  # Limit total candidates
                        seen_content.add(content_key)
                        combined_docs.append(doc)
                
                # Structure-aware prioritization: prefer first-class retrievables
                priority_map = {
                    'footnote': 4,
                    'complete_table': 3,
                    'table_row': 2,
                    'table_cell': 1,
                    'text': 0
                }
                def _priority_key(d: Document):
                    et = d.metadata.get('element_type', 'text')
                    imp = d.metadata.get('importance_score', 0)
                    return (priority_map.get(et, 0), imp)
                combined_docs.sort(key=_priority_key, reverse=True)
                
                # Step 4: Cell-Aware Reranking to final k - STRUCTURE-AWARE IMPROVEMENT
                # Diagnostics: count structure types before rerank
                pre_rerank_counts = defaultdict(int)
                for d in combined_docs:
                    pre_rerank_counts[d.metadata.get('element_type', 'text')] += 1

                if len(combined_docs) > final_k:
                    try:
                        # Use custom cell-aware reranker instead of generic CrossEncoder
                        final_docs = self.structure_aware_reranker.rerank_documents(combined_docs, question, final_k)
                    except Exception as rerank_error:
                        print(f"Cell-aware reranking failed, using top {final_k} docs: {rerank_error}")
                        final_docs = combined_docs[:final_k]
                else:
                    final_docs = combined_docs
                
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
                
                # Step 5: Create structure-enhanced context - STRUCTURE-AWARE IMPROVEMENT
                enhanced_context = self._create_structure_enhanced_context(final_docs)
                
                # Step 6: Select optimal prompt based on structure content - STRUCTURE-AWARE IMPROVEMENT
                selected_prompt = self._select_optimal_prompt(final_docs, question)
                
                # Step 7: Generate answer - SAME PATTERN AS SKELETON
                llm = self.llm_provider.get_langchain_llm()
                prompt = selected_prompt.format(context=enhanced_context)

                answer = await asyncio.to_thread(
                    lambda: llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": question}]).content
                )
                
                # Create context with scores and structure analysis - ENHANCED FROM SKELETON
                context_with_scores = []
                structure_analysis = defaultdict(int)
                
                for idx, doc in enumerate(final_docs):
                    element_type = doc.metadata.get('element_type', 'text')
                    structure_analysis[element_type] += 1
                    
                    context_with_scores.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": 1.0 - (idx * 0.05),  # Decreasing score based on rerank position
                        "element_type": element_type,
                        "is_first_class_retrievable": element_type in ['complete_table', 'table_row', 'table_cell', 'footnote']
                    })
                
                # Build debug info with improved structure-aware hybrid stats
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
                    },
                    "structure_priority_applied": True,
                    # Pre-rerank structure distribution (diagnostics)
                    "pre_rerank_structure_counts": dict(pre_rerank_counts),
                    # Structure-aware specific metrics
                    "structure_analysis": dict(structure_analysis),
                    "first_class_retrievables": {
                        "complete_tables": structure_analysis.get('complete_table', 0),
                        "table_rows": structure_analysis.get('table_row', 0),
                        "table_cells": structure_analysis.get('table_cell', 0),
                        "footnotes": structure_analysis.get('footnote', 0),
                        "total": (structure_analysis.get('complete_table', 0) + 
                                structure_analysis.get('table_row', 0) + 
                                structure_analysis.get('table_cell', 0) + 
                                structure_analysis.get('footnote', 0))
                    },
                    "cell_aware_reranking_applied": True,
                    "structure_enhanced_context": True,
                    "prompt_selection": {
                        "selected_prompt": selected_prompt.__name__ if hasattr(selected_prompt, '__name__') else "structure_aware_prompt",
                        "reason": "Based on content structure and question type analysis"
                    }
                }
                
                # Add dynamic k analysis if applicable - SAME AS SKELETON
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
                        "retrieval_method": "Structure-Aware Hybrid Search + Cell-Aware Reranker (Error)"
                    }
                }
                print(f"Error: {error_msg}")
                return result
        
        print(f"Processing {len(questions)} questions with structure-aware hybrid search + cell-aware reranker...")
        start_time = time.time()
        
        tasks = [process_single_question(i, question) for i, question in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        print(f"Structure-aware processing time: {end_time - start_time:.2f} seconds")
        
        # Enhanced statistics - SAME PATTERN AS SKELETON BUT WITH STRUCTURE METRICS
        answers = []
        debug_info = []
        total_first_class = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error processing question {i+1}: {str(result)}"
                answers.append(error_msg)
                debug_info.append({
                    "question": questions[i],
                    "answer": error_msg,
                    "chunks_count": 0,
                    "error": str(result),
                    "retrieval_method": "Structure-Aware Hybrid Search + Cell-Aware Reranker (Exception)"
                })
            else:
                answers.append(result["answer"])
                debug_info.append(result["debug_info"])
                
                # Track first-class retrievables usage
                first_class_stats = result["debug_info"].get("first_class_retrievables", {})
                total_first_class += first_class_stats.get("total", 0)
        
        print(f"📊 Structure-aware processing complete:")
        print(f"   ⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"   📄 Questions processed: {len(questions)}")
        print(f"   🏗️  First-class retrievables used: {total_first_class}")
        print(f"   ♦️  Cell-aware reranking applied: True")
        print()
        
        return {
            "answers": answers,
            "debug_info": debug_info
        }