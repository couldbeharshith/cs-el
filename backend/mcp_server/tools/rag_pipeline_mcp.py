import hashlib
from typing import Union, List, Dict, Any, Optional
import time
import uuid

from app.services.vector_stores.vector_store_factory import VectorStoreFactory
from app.services.preprocessors.unified_processor import UnifiedDocumentProcessor
from app.services.retrievers.retrieval_service import RetrievalService
from app.services.retrievers.structure_aware_retrieval_service import TrueStructureAwareRetrievalService
from app.services.pipelines.pipeline_manager import PipelineManager
from app.providers.factory import LLMProviderFactory
from app.config.settings import settings

vector_store = VectorStoreFactory.create_vector_store(settings)
llm_provider = LLMProviderFactory.create_provider(
    settings.DEFAULT_LLM_PROVIDER,
    settings
)
document_processor = UnifiedDocumentProcessor(
    vector_store=vector_store,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)
traditional_retrieval_service = RetrievalService(
    vector_store=vector_store,
    llm_provider=llm_provider
)

structure_aware_retrieval_service = TrueStructureAwareRetrievalService(
    vector_store=vector_store,
    llm_provider=llm_provider
)

async def rag_pipeline_mcp(
    document_url: str,
    questions: Union[str, List[str]],
    rag_mode: Optional[str],                        # 'traditional' | 'structure_aware'
    k_strategy: Optional[str],                      # 'dynamic' | 'static'
    k_value: Optional[int],                         # required when k_strategy == 'static'
) -> Dict[str, Any]:

    start_time = time.time()

    try:
        if isinstance(questions, str):
            questions = [questions]

        document_id = hashlib.sha256(document_url.encode()).hexdigest()[:16]

        if not rag_mode:
            raise ValueError("rag_mode is required: 'traditional' or 'structure_aware'")
        resolved_rag_mode = rag_mode.strip()
        if not PipelineManager.is_supported_pipeline(resolved_rag_mode):
            supported = list(PipelineManager.get_supported_pipelines().keys())
            raise ValueError(f"Unsupported rag_mode '{resolved_rag_mode}'. Supported: {supported}")

        if not k_strategy:
            raise ValueError("k_strategy is required: 'dynamic' or 'static'")
        resolved_k_strategy = k_strategy.strip().lower()

        if resolved_k_strategy not in {"dynamic", "static"}:
            raise ValueError("k_strategy must be either 'dynamic' or 'static'")

        # Resolve k value (only when static)
        resolved_k_value: Optional[int]
        if resolved_k_strategy == 'static':
            resolved_k_value = k_value
            if resolved_k_value is None:
                raise ValueError("k_value is required when k_strategy='static'")
            if not isinstance(resolved_k_value, int) or resolved_k_value <= 0:
                raise ValueError("k_value must be a positive integer when k_strategy='static'")
        else:
            # For dynamic, ignore any provided k_value to avoid ambiguity
            resolved_k_value = None

        print(f"Processing document: {document_url}")
        print(f"Using {resolved_rag_mode} RAG processing: {PipelineManager.get_pipeline_description(resolved_rag_mode)}")
        if resolved_k_strategy == 'static':
            print(f"Chunk selection: static k={resolved_k_value}")
        else:
            print("Chunk selection: dynamic k")

        # Execute selected pipeline (parity with FastAPI endpoint)
        answers, document_metadata, raw_response = await PipelineManager.execute_pipeline(
            pipeline_type=resolved_rag_mode,
            document_id=document_id,
            document_url=document_url,
            questions=questions,
            k=resolved_k_value,
            vector_store=vector_store,
            document_processor=document_processor,
            traditional_retrieval_service=traditional_retrieval_service,
            structure_aware_retrieval_service=structure_aware_retrieval_service,
            settings=settings,
        )

        processing_time = time.time() - start_time

        result: Dict[str, Any] = {
            "success": True,
            "document_url": document_url,
            "questions_asked": len(questions),
            "answers": answers,
            "processing_time": round(processing_time, 2),
            "document_metadata": document_metadata,
            "chunks_retrieved": resolved_k_value if resolved_k_strategy == 'static' else None,
            "rag_mode": resolved_rag_mode,
            "k_strategy": resolved_k_strategy,
            "k_value": resolved_k_value,
            # "raw_response": raw_response if settings.ENVIRONMENT.lower() != "production" else None
        }

        print(f"RAG processing completed in {processing_time:.2f}s | answers: {len(answers)}")

        return result

    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)

        print(f"RAG processing failed after {processing_time:.2f}s: {error_message}")

        return {
            "success": False,
            "document_url": document_url,
            "questions_asked": len(questions) if isinstance(questions, list) else 1,
            "answers": [],
            "processing_time": round(processing_time, 2),
            "error": error_message,
            "document_metadata": {},
            "chunks_retrieved": k_value if (k_strategy or '').lower() == 'static' else None,
            "rag_mode": rag_mode,
            "k_strategy": (k_strategy or ("static" if k_value is not None else "dynamic")),
            "k_value": k_value,
            "raw_response": None
        }
