from typing import List, Tuple, Dict, Any, Optional
from app.services.pipelines.traditional_rag import traditional_rag
from app.services.pipelines.structure_aware_rag import structure_aware_rag

class PipelineManager:
    """
    Centralized pipeline manager for routing requests to different RAG implementations.
    """

    SUPPORTED_PIPELINES = {
        "traditional": "Traditional RAG with Hybrid Search",
        "structure_aware": "Structure-Aware RAG with Table/Cell Enhancement"
    }
    
    # Pipeline execution mapping - makes it easy to add new pipelines
    PIPELINE_EXECUTORS = {
        "traditional": traditional_rag,
        "structure_aware": structure_aware_rag,
    }
    
    @classmethod
    def is_supported_pipeline(cls, pipeline_type: str) -> bool:
        """Check if a pipeline type is supported."""
        return pipeline_type in cls.SUPPORTED_PIPELINES
    
    @classmethod
    def get_default_pipeline(cls) -> str:
        """Get the default pipeline type."""
        return "traditional"
    
    @classmethod
    async def execute_pipeline(
        cls,
        pipeline_type: str,
        *,
        document_id: str,
        document_url: str,
        questions: List[str],
        k: Optional[int],
        vector_store,
        document_processor,
        traditional_retrieval_service,
        structure_aware_retrieval_service,
        settings,
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Execute the specified RAG pipeline with the given parameters.
        
        Args:
            pipeline_type: The pipeline type (must be in SUPPORTED_PIPELINES)
            document_id: Unique identifier for the document
            document_url: URL of the document to process
            questions: List of questions to answer
            k: Number of chunks to retrieve (None for dynamic k)
            vector_store: Vector store instance
            document_processor: Document processor instance
            traditional_retrieval_service: Traditional retrieval service instance
            structure_aware_retrieval_service: Structure-aware retrieval service instance
            settings: Application settings
            
        Returns:
            Tuple of (answers, document_metadata, raw_response)
            
        Raises:
            ValueError: If pipeline_type is not supported
        """
        
        # Validate pipeline type
        if not cls.is_supported_pipeline(pipeline_type):
            supported = ", ".join(cls.SUPPORTED_PIPELINES.keys())
            raise ValueError(f"Unsupported pipeline type '{pipeline_type}'. Supported types: {supported}")
        
        # Select the appropriate retrieval service based on pipeline type
        if pipeline_type == "structure_aware":
            retrieval_service = structure_aware_retrieval_service
        else:
            # Default to traditional for all other pipeline types
            retrieval_service = traditional_retrieval_service
        
        # Get and execute the pipeline
        pipeline_executor = cls.PIPELINE_EXECUTORS[pipeline_type]
        return await pipeline_executor(
            document_id=document_id,
            document_url=document_url,
            questions=questions,
            k=k,
            vector_store=vector_store,
            document_processor=document_processor,
            retrieval_service=retrieval_service,
            settings=settings,
        )
    
    @classmethod
    def register_pipeline(cls, pipeline_type: str, description: str, executor_function):
        """Register a new pipeline type dynamically.
        
        Args:
            pipeline_type: Unique identifier for the pipeline
            description: Human-readable description
            executor_function: Async function with the same signature as traditional_rag
        """
        cls.SUPPORTED_PIPELINES[pipeline_type] = description
        cls.PIPELINE_EXECUTORS[pipeline_type] = executor_function
        print(f"✅ Registered new pipeline: {pipeline_type} - {description}")
    
    @classmethod
    def get_pipeline_description(cls, pipeline_type: str) -> str:
        """Get a human-readable description of the pipeline."""
        return cls.SUPPORTED_PIPELINES.get(pipeline_type, f"Unknown pipeline: {pipeline_type}")
    
    @classmethod
    def get_supported_pipelines(cls) -> Dict[str, str]:
        """Get all supported pipelines and their descriptions."""
        return cls.SUPPORTED_PIPELINES.copy()