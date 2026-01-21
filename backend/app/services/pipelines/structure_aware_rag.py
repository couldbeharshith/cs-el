from typing import List, Tuple, Dict, Any, Optional

async def structure_aware_rag(
    *,
    document_id: str,
    document_url: str,
    questions: List[str],
    k: Optional[int],
    vector_store,
    document_processor,
    retrieval_service,
    settings,
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """Run the structure-aware RAG flow optimized for insurance documents.
    
    This pipeline treats tables, table rows, and footnotes as first-class retrievables,
    using enhanced chunking with table stitching and cell-aware reranking.
    """

    answers: List[str] = []
    document_metadata: Dict[str, Any] = {}
    raw_response: Dict[str, Any] = {}

    # Process document - caching is handled internally by the vector store
    processing_result = await document_processor.process_document_from_url_async(
        url=document_url,
        metadata={"document_id": document_id}
    )

    if not processing_result["success"]:
        raise RuntimeError(processing_result["error"])

    # Check if this was a cache hit (no new chunks added)
    cache_used = processing_result.get("cache_used", False)
    chunks_count = processing_result["chunk_count"]

    # Use structure-aware retrieval if available, otherwise fall back to traditional
    if hasattr(retrieval_service, 'process_document_queries_structure_aware'):
        query_results = await retrieval_service.process_document_queries_structure_aware(
            document_id=document_id,
            questions=questions,
            k=k,  # Pass k directly. If it's None, dynamic mode is used.
        )
        retrieval_method = "True Structure-Aware RAG with Cell-Aware Scoring"
    else:
        # Fallback to traditional retrieval if structure-aware not available
        query_results = await retrieval_service.process_document_queries(
            document_id=document_id,
            questions=questions,
            k=k,  # Pass k directly. If it's None, dynamic mode is used.
        )
        retrieval_method = "Hybrid Search (Structure-Aware Fallback)"

    answers = query_results["answers"]
    debug_info = query_results["debug_info"]

    # Extract structure-aware metrics if available
    structure_metrics = {}
    if debug_info and len(debug_info) > 0:
        first_debug = debug_info[0]
        structure_metrics = {
            "first_class_retrievables_used": first_debug.get("first_class_retrievables", {}),
            "structure_analysis": first_debug.get("structure_analysis", {}),
            "cell_aware_reranking": first_debug.get("cell_aware_reranking_applied", False),
            "structure_enhanced_context": first_debug.get("structure_enhanced_context", False),
            "prompt_selection": first_debug.get("prompt_selection", {}),
        }

    # Determine retrieval method from the first valid debug info
    retrieval_method = "Structure-Aware RAG with Table Scoring"
    if debug_info and isinstance(debug_info[0], dict):
        retrieval_method = debug_info[0].get("retrieval_method", "Structure-Aware RAG with Table Scoring")
    
    # Override if fallback was used
    if not hasattr(retrieval_service, 'process_document_queries_structure_aware'):
        retrieval_method = "Hybrid Search (Structure-Aware Fallback)"

    raw_response = {
        "chunks_per_question": k if k is not None else "dynamic",
        "total_questions": len(questions),
        "retrieval_method": retrieval_method,
        "cache_used": cache_used,
        "processing_mode": "structure_aware",
        "debug_info": debug_info,
        "structure_metrics": structure_metrics,
    }

    document_metadata = {
        "document_id": document_id,
        "chunks_processed": chunks_count,
        "vector_store": processing_result.get("vector_store", vector_store.store_type),
        "cache_used": cache_used,
        "processing_mode": "structure_aware",
        "structure_metrics": structure_metrics,
    }

    return answers, document_metadata, raw_response