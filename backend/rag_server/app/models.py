"""
Consolidated data models for the Insurance RAG API.
All Pydantic models for API requests, responses, and internal data structures.

Note - some of these are deprecated and not used.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re
from urllib.parse import urlparse


# CORE DATA MODELS
class ChunkMetadata(BaseModel):
    """Metadata for document chunks with citation information."""
    document_id: str = Field(..., description="Unique identifier for the document")
    page_number: int = Field(..., description="Page number in the original document")
    line_number: int = Field(..., description="Line number on the page")
    chunk_number: int = Field(..., description="Sequential chunk number")
    document_type: str = Field(..., description="Type of document (pdf, docx, txt, pptx, xlsx, image)")
    file_name: str = Field(..., description="Original filename")
    slide_number: Optional[int] = Field(None, description="Slide number for PPTX documents")
    content_type: Optional[str] = Field(None, description="Content type (title, subtitle, text, table, image_ocr) for PPTX documents")
    sheet_name: Optional[str] = Field(None, description="Sheet name for XLSX documents")
    sheet_index: Optional[int] = Field(None, description="Sheet index for XLSX documents")


class DocumentChunk(BaseModel):
    """Represents a chunk of processed document with metadata."""
    content: str = Field(..., description="Text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata for the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    
    class Config:
        arbitrary_types_allowed = True


class StructuredQuery(BaseModel):
    """Structured representation of a parsed natural language query."""
    raw_query: str = Field(..., description="Original natural language query")
    age: Optional[int] = Field(None, description="Extracted age information")
    procedure: Optional[str] = Field(None, description="Extracted procedure information")
    location: Optional[str] = Field(None, description="Extracted location information")
    policy_duration: Optional[str] = Field(None, description="Extracted policy duration")
    additional_context: Dict[str, Any] = Field(default_factory=dict, description="Additional extracted context")
    completeness_score: float = Field(0.0, description="Score indicating query completeness")


class Citation(BaseModel):
    """Citation information for decision justification."""
    chunk_number: int = Field(..., description="Number of the cited chunk")
    page_number: int = Field(..., description="Page number of the citation")
    line_number: int = Field(..., description="Line number of the citation")
    relevant_text: str = Field(..., description="Relevant text from the citation")
    document_name: str = Field(..., description="Name of the source document")


class Decision(BaseModel):
    """Decision result with justification and citations."""
    status: str = Field(..., description="Decision status: approved, rejected, or needs_review")
    amount: Optional[float] = Field(None, description="Amount if applicable to the decision")
    brief_explanation: str = Field("", description="Short 1-2 line explanation of the decision")
    detailed_justification: str = Field("", description="Comprehensive reasoning with policy references")
    citations: List[Citation] = Field(default_factory=list, description="List of citations supporting the decision")
    confidence_score: float = Field(0.0, description="Confidence score for the decision")
    decision_factors: List[str] = Field(default_factory=list, description="Factors that influenced the decision")


class RelevantChunk(BaseModel):
    """Chunk with relevance score from vector search."""
    chunk: DocumentChunk = Field(..., description="The document chunk")
    relevance_score: float = Field(..., description="Relevance score from vector search")
    distance: float = Field(..., description="Vector distance from query")


# API REQUEST/RESPONSE MODELS

class QueryRequest(BaseModel):
    """Request model for query processing endpoint."""
    query: str = Field(..., description="Natural language query about insurance policy")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of specific document IDs to search")
    include_confidence: bool = Field(True, description="Whether to include confidence scores in response")
    use_thinking_mode: bool = Field(True, description="Toggle for Gemini thinking mode")


class CitationResponse(BaseModel):
    """Citation information in API response."""
    chunk_number: int = Field(..., description="Number of the cited chunk")
    page_number: int = Field(..., description="Page number of the citation")
    line_number: int = Field(..., description="Line number of the citation")
    relevant_text: str = Field(..., description="Relevant text from the citation")
    document_name: str = Field(..., description="Name of the source document")


class QueryResponse(BaseModel):
    """Response model for query processing endpoint."""
    decision: str = Field(..., description="Decision status: approved, rejected, or needs_review")
    amount: Optional[float] = Field(None, description="Amount if applicable to the decision")
    brief_explanation: str = Field(..., description="Short 1-2 line explanation of the decision")
    detailed_justification: str = Field(..., description="Comprehensive reasoning with policy references")
    citations: List[CitationResponse] = Field(default_factory=list, description="List of citations supporting the decision")
    confidence_score: float = Field(..., description="Confidence score for the decision")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during processing")
    thinking_process: Optional[str] = Field(None, description="Thinking process if thinking mode is enabled")


class UploadRequest(BaseModel):
    """Request model for document upload (metadata)."""
    document_type: str = Field(..., description="Type of document: pdf, docx, eml, txt")
    file_name: str = Field(..., description="Original filename")


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    file_name: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Type of document processed")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    processing_time: float = Field(..., description="Time taken to process the document in seconds")
    status: str = Field(..., description="Processing status: success, partial, failed")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during processing")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Current timestamp (ISO format)")
    services: Dict[str, Any] = Field(..., description="Status of individual services and metrics")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error_type: str = Field(..., description="Type of error that occurred")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="When the error occurred (ISO format)")
    request_id: str = Field(..., description="Unique identifier for the request")


# HACKRX API MODELS

class HackRXRequest(BaseModel):
    """Request model for HackRX API endpoint."""
    documents: str = Field(..., description="URL to the policy document", max_length=2048)
    questions: List[str] = Field(..., min_items=1, max_items=50, description="List of questions to answer")
    
    @validator('documents')
    def validate_url(cls, v):
        """Validate document URL with comprehensive security checks."""
        if not v or not isinstance(v, str):
            raise ValueError('Document URL is required')
        
        v = v.strip()
        
        # Length validation
        if len(v) > 2048:
            raise ValueError('URL exceeds maximum length of 2048 characters')
        if len(v) < 10:
            raise ValueError('URL is too short to be valid')
        
        # URL format validation
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid URL format')
            if parsed.scheme.lower() not in ['http', 'https']:
                raise ValueError('Only HTTP and HTTPS schemes are allowed')
        except Exception:
            raise ValueError('Invalid URL format')
        
        # Security checks for internal/private addresses
        hostname = parsed.hostname
        if hostname:
            hostname_lower = hostname.lower()
            # Block localhost and private IP patterns
            blocked_patterns = [
                'localhost', '127.0.0.1', '0.0.0.0', '192.168.', '10.', '172.16.',
                '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.',
                '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
                '172.29.', '172.30.', '172.31.'
            ]
            for pattern in blocked_patterns:
                if pattern in hostname_lower:
                    raise ValueError('URL points to internal/private address')
        
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list with security checks."""
        if not v or not isinstance(v, list):
            raise ValueError('Questions list is required')
        
        # Validate each question
        sanitized_questions = []
        for i, question in enumerate(v, 1):
            if not isinstance(question, str):
                raise ValueError(f'Question {i} must be a string')
            
            question = question.strip()
            if not question:
                raise ValueError(f'Question {i} cannot be empty')
            
            if len(question) > 1000:
                raise ValueError(f'Question {i} exceeds maximum length of 1000 characters')
            
            if len(question) < 3:
                raise ValueError(f'Question {i} is too short (minimum 3 characters)')
            
            # Basic security checks
            dangerous_patterns = [
                '<script', 'javascript:', 'vbscript:', '<iframe', r'on\w+\s*=',
                'drop table', 'insert into', 'delete from', 'union select'
            ]
            question_lower = question.lower()
            for pattern in dangerous_patterns:
                if re.search(pattern, question_lower):
                    raise ValueError(f'Question {i} contains potentially harmful content')
            
            sanitized_questions.append(question)
        
        return sanitized_questions


class HackRXResponse(BaseModel):
    """Response model for HackRX API endpoint."""
    answers: List[str] = Field(..., description="Complete answers including decision, amount, reasoning, and references")


class RelevanceResponse(BaseModel):
    """Response model for irrelevant queries."""
    answers: List[str] = Field(default=["the query is not relevant to use case of this API"])


# SERVICE INTERFACE MODELS

class DocumentProcessingRequest(BaseModel):
    """Request model for document processing service."""
    url: str = Field(..., description="URL of the document to process")
    document_type: Optional[str] = Field(None, description="Type of document (auto-detected if not provided)")
    chunk_size: int = Field(1000, description="Size of text chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")


class DocumentProcessingResult(BaseModel):
    """Result model for document processing service."""
    document_hash: str = Field(..., description="Hash of the document content")
    namespace: str = Field(..., description="Pinecone namespace for the document")
    chunks: List[DocumentChunk] = Field(..., description="Processed document chunks")
    processing_time: float = Field(..., description="Time taken to process the document")
    cached: bool = Field(False, description="Whether the document was retrieved from cache")
    direct_gemini_content: Optional[bytes] = Field(None, description="Raw content for direct Gemini upload")
    direct_gemini_content_type: Optional[str] = Field(None, description="Content type for direct Gemini upload")


class VectorSearchRequest(BaseModel):
    """Request model for vector search service."""
    query: str = Field(..., description="Search query")
    namespace: str = Field(..., description="Pinecone namespace to search")
    top_k: int = Field(5, description="Number of top results to return")
    include_metadata: bool = Field(True, description="Whether to include metadata in results")


class VectorSearchResult(BaseModel):
    """Result model for vector search service."""
    chunks: List[RelevantChunk] = Field(..., description="Relevant chunks with scores")
    search_time: float = Field(..., description="Time taken for the search")
    total_results: int = Field(..., description="Total number of results found")


class LLMRequest(BaseModel):
    """Request model for LLM service."""
    prompt_type: str = Field(..., description="Type of prompt to use")
    context: str = Field(..., description="Context for the prompt")
    query: str = Field(..., description="User query")
    temperature: float = Field(0.1, description="Temperature for LLM generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")


class LLMResponse(BaseModel):
    """Response model for LLM service."""
    response: str = Field(..., description="Generated response")
    tokens_used: int = Field(..., description="Number of tokens used")
    processing_time: float = Field(..., description="Time taken for generation")
    model_used: str = Field(..., description="Model used for generation")


class RelevanceCheckRequest(BaseModel):
    """Request model for relevance checking service."""
    query: str = Field(..., description="Query to check for relevance")
    threshold: float = Field(0.7, description="Relevance threshold")


class RelevanceCheckResult(BaseModel):
    """Result model for relevance checking service."""
    is_relevant: bool = Field(..., description="Whether the query is relevant")
    relevance_score: float = Field(..., description="Relevance score")
    matched_domains: List[str] = Field(default_factory=list, description="Domains that matched")
    processing_time: float = Field(..., description="Time taken for relevance check")


# BATCH PROCESSING MODELS

class BatchProcessingRequest(BaseModel):
    """Request model for batch processing operations."""
    items: List[Dict[str, Any]] = Field(..., description="Items to process in batch")
    batch_size: int = Field(90, description="Size of each batch")
    max_concurrent: int = Field(5, description="Maximum concurrent operations")


class BatchProcessingResult(BaseModel):
    """Result model for batch processing operations."""
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successfully processed items")
    failed_items: int = Field(..., description="Number of failed items")
    processing_time: float = Field(..., description="Total processing time")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")


# LLM KEY MANAGEMENT MODELS

class APIKeyHealthResponse(BaseModel):
    """API response model for API key health information."""
    key_id: str = Field(..., description="Identifier for the API key")
    service_type: str = Field(..., description="Type of service (gemini or groq)")
    status: str = Field(..., description="Current status of the key")
    success_count: int = Field(..., description="Number of successful requests")
    failure_count: int = Field(..., description="Number of failed requests")
    consecutive_failures: int = Field(..., description="Number of consecutive failures")
    is_healthy: bool = Field(..., description="Whether the key is currently healthy")
    last_success: Optional[str] = Field(None, description="Timestamp of last successful request")
    last_failure: Optional[str] = Field(None, description="Timestamp of last failed request")
    cooldown_until: Optional[str] = Field(None, description="Timestamp when cooldown period ends")


class LLMServiceHealthResponse(BaseModel):
    """API response model for LLM service health summary."""
    service_type: str = Field(..., description="Type of LLM service")
    total_keys: int = Field(..., description="Total number of API keys")
    healthy_keys: int = Field(..., description="Number of healthy keys")
    unhealthy_keys: int = Field(..., description="Number of unhealthy keys")
    health_percentage: float = Field(..., description="Percentage of healthy keys")
    total_successes: int = Field(..., description="Total successful requests across all keys")
    total_failures: int = Field(..., description="Total failed requests across all keys")
    success_rate: float = Field(..., description="Overall success rate percentage")


# PERFORMANCE MONITORING MODELS

class PerformanceMetrics(BaseModel):
    """Model for performance metrics tracking."""
    operation: str = Field(..., description="Name of the operation")
    duration: float = Field(..., description="Duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the operation occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    success: bool = Field(True, description="Whether the operation was successful")


class SystemHealth(BaseModel):
    """Model for system health status."""
    status: str = Field(..., description="Overall system status")
    services: Dict[str, bool] = Field(..., description="Status of individual services")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    uptime: float = Field(..., description="System uptime in seconds")