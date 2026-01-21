from typing import List, Tuple, Dict, Any, Optional

async def traditional_rag(
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
    """Run the traditional RAG flow used by the `/run` endpoint.
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

    query_results = await retrieval_service.process_document_queries(
        document_id=document_id,
        questions=questions,
        k=k,  # Pass k directly. If it's None, dynamic mode is used.
    )

    answers = query_results["answers"]
    debug_info = query_results["debug_info"]

    # Determine retrieval method from the first valid debug info
    retrieval_method = "Hybrid Search"
    if debug_info and isinstance(debug_info[0], dict):
        retrieval_method = debug_info[0].get("retrieval_method", "Hybrid Search")

    raw_response = {
        "chunks_per_question": k if k is not None else "dynamic",
        "total_questions": len(questions),
        "retrieval_method": retrieval_method,
        "cache_used": cache_used,
        "processing_mode": "traditional",
        "debug_info": debug_info,
    }

    document_metadata = {
        "document_id": document_id,
        "chunks_processed": chunks_count,
        "vector_store": processing_result.get("vector_store", vector_store.store_type),
        "cache_used": cache_used,
        "processing_mode": "traditional",
    }


    return answers, document_metadata, raw_response