"""
Custom exceptions module for the Insurance RAG API.

This module provides a consolidated exception hierarchy for different error types
with structured error response formatting and consistent error handling across all services.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from fastapi import HTTPException
from app.models import ErrorResponse


# BASE EXCEPTION CLASSES
class InsuranceRAGException(Exception):
    """
    Base exception for the Insurance RAG API application.
    
    All custom exceptions should inherit from this base class to ensure
    consistent error handling and response formatting.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_error_response(self) -> ErrorResponse:
        """Convert exception to structured ErrorResponse model."""
        return ErrorResponse(
            error_type=self.__class__.__name__,
            message=self.message,
            details={
                "error_code": self.error_code,
                **self.details
            },
            timestamp=self.timestamp,
            request_id=self.request_id
        )
    
    def to_http_exception(self, status_code: int = 500) -> HTTPException:
        """Convert exception to FastAPI HTTPException with structured detail."""
        return HTTPException(
            status_code=status_code,
            detail=self.to_error_response().model_dump()
        )


# DOCUMENT PROCESSING EXCEPTIONS

class DocumentProcessingError(InsuranceRAGException):
    """
    Exception raised when document processing operations fail.
    
    This includes document fetching, parsing, content extraction,
    and chunking operations.
    """
    
    def __init__(
        self,
        message: str,
        document_url: Optional[str] = None,
        document_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        processing_details = details or {}
        if document_url:
            processing_details["document_url"] = document_url
        if document_type:
            processing_details["document_type"] = document_type
        
        super().__init__(
            message=message,
            error_code=error_code or "DOCUMENT_PROCESSING_ERROR",
            details=processing_details,
            request_id=request_id
        )


class DocumentFetchError(DocumentProcessingError):
    """Exception raised when document fetching from URL fails."""
    
    def __init__(
        self,
        message: str,
        document_url: str,
        http_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        fetch_details = details or {}
        if http_status:
            fetch_details["http_status"] = http_status
        
        super().__init__(
            message=message,
            document_url=document_url,
            error_code="DOCUMENT_FETCH_ERROR",
            details=fetch_details,
            request_id=request_id
        )


class DocumentParseError(DocumentProcessingError):
    """Exception raised when document parsing fails."""
    
    def __init__(
        self,
        message: str,
        document_type: str,
        document_url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            document_url=document_url,
            document_type=document_type,
            error_code="DOCUMENT_PARSE_ERROR",
            details=details,
            request_id=request_id
        )


class DocumentSizeError(DocumentProcessingError):
    """Exception raised when document exceeds size limits."""
    
    def __init__(
        self,
        message: str,
        document_size: int,
        max_size: int,
        document_url: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        size_details = details or {}
        size_details.update({
            "document_size_bytes": document_size,
            "max_size_bytes": max_size,
            "document_size_mb": round(document_size / (1024 * 1024), 2),
            "max_size_mb": round(max_size / (1024 * 1024), 2)
        })
        
        super().__init__(
            message=message,
            document_url=document_url,
            error_code="DOCUMENT_SIZE_ERROR",
            details=size_details,
            request_id=request_id
        )


# VECTOR DATABASE EXCEPTIONS

class VectorSearchError(InsuranceRAGException):
    """
    Exception raised when vector database operations fail.
    
    This includes Pinecone connection issues, search operations,
    upsert operations, and namespace management.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        namespace: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        vector_details = details or {}
        if operation:
            vector_details["operation"] = operation
        if namespace:
            vector_details["namespace"] = namespace
        
        super().__init__(
            message=message,
            error_code=error_code or "VECTOR_SEARCH_ERROR",
            details=vector_details,
            request_id=request_id
        )


class PineconeConnectionError(VectorSearchError):
    """Exception raised when Pinecone connection fails."""
    
    def __init__(
        self,
        message: str,
        index_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        connection_details = details or {}
        if index_name:
            connection_details["index_name"] = index_name
        
        super().__init__(
            message=message,
            operation="connection",
            error_code="PINECONE_CONNECTION_ERROR",
            details=connection_details,
            request_id=request_id
        )


class VectorUpsertError(VectorSearchError):
    """Exception raised when vector upsert operations fail."""
    
    def __init__(
        self,
        message: str,
        namespace: str,
        batch_size: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        upsert_details = details or {}
        if batch_size:
            upsert_details["batch_size"] = batch_size
        
        super().__init__(
            message=message,
            operation="upsert",
            namespace=namespace,
            error_code="VECTOR_UPSERT_ERROR",
            details=upsert_details,
            request_id=request_id
        )


class VectorQueryError(VectorSearchError):
    """Exception raised when vector query operations fail."""
    
    def __init__(
        self,
        message: str,
        namespace: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        query_details = details or {}
        if query:
            query_details["query"] = query[:100] + "..." if len(query) > 100 else query
        if top_k:
            query_details["top_k"] = top_k
        
        super().__init__(
            message=message,
            operation="query",
            namespace=namespace,
            error_code="VECTOR_QUERY_ERROR",
            details=query_details,
            request_id=request_id
        )


class NamespaceError(VectorSearchError):
    """Exception raised when namespace operations fail."""
    
    def __init__(
        self,
        message: str,
        namespace: str,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            operation=f"namespace_{operation}",
            namespace=namespace,
            error_code="NAMESPACE_ERROR",
            details=details,
            request_id=request_id
        )


# LLM PROCESSING EXCEPTIONS

class LLMProcessingError(InsuranceRAGException):
    """
    Exception raised when LLM API operations fail.
    
    This includes Gemini API calls, prompt processing,
    response parsing, and token limit issues.
    """
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        prompt_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        llm_details = details or {}
        if model:
            llm_details["model"] = model
        if prompt_type:
            llm_details["prompt_type"] = prompt_type
        
        super().__init__(
            message=message,
            error_code=error_code or "LLM_PROCESSING_ERROR",
            details=llm_details,
            request_id=request_id
        )


class GeminiAPIError(LLMProcessingError):
    """Exception raised when Gemini API calls fail."""
    
    def __init__(
        self,
        message: str,
        model: str,
        api_error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        api_details = details or {}
        if api_error_code:
            api_details["api_error_code"] = api_error_code
        
        super().__init__(
            message=message,
            model=model,
            error_code="GEMINI_API_ERROR",
            details=api_details,
            request_id=request_id
        )


class PromptProcessingError(LLMProcessingError):
    """Exception raised when prompt processing fails."""
    
    def __init__(
        self,
        message: str,
        prompt_type: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            prompt_type=prompt_type,
            error_code="PROMPT_PROCESSING_ERROR",
            details=details,
            request_id=request_id
        )


class TokenLimitError(LLMProcessingError):
    """Exception raised when token limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        token_count: int,
        token_limit: int,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        token_details = details or {}
        token_details.update({
            "token_count": token_count,
            "token_limit": token_limit,
            "tokens_over_limit": token_count - token_limit
        })
        
        super().__init__(
            message=message,
            model=model,
            error_code="TOKEN_LIMIT_ERROR",
            details=token_details,
            request_id=request_id
        )


# RELEVANCE FILTERING EXCEPTIONS

class RelevanceFilterError(InsuranceRAGException):
    """
    Exception raised when relevance filtering operations fail.
    
    This includes relevance checking, domain initialization,
    and relevance threshold processing.
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        relevance_details = details or {}
        if query:
            relevance_details["query"] = query[:100] + "..." if len(query) > 100 else query
        
        super().__init__(
            message=message,
            error_code=error_code or "RELEVANCE_FILTER_ERROR",
            details=relevance_details,
            request_id=request_id
        )


class DomainInitializationError(RelevanceFilterError):
    """Exception raised when relevance domain initialization fails."""
    
    def __init__(
        self,
        message: str,
        domain_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        domain_details = details or {}
        if domain_count:
            domain_details["domain_count"] = domain_count
        
        super().__init__(
            message=message,
            error_code="DOMAIN_INITIALIZATION_ERROR",
            details=domain_details,
            request_id=request_id
        )


class RelevanceThresholdError(RelevanceFilterError):
    """Exception raised when relevance threshold processing fails."""
    
    def __init__(
        self,
        message: str,
        threshold: float,
        score: Optional[float] = None,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        threshold_details = details or {}
        threshold_details["threshold"] = threshold
        if score is not None:
            threshold_details["relevance_score"] = score
        
        super().__init__(
            message=message,
            query=query,
            error_code="RELEVANCE_THRESHOLD_ERROR",
            details=threshold_details,
            request_id=request_id
        )


# AUTHENTICATION AND AUTHORIZATION EXCEPTIONS

class AuthenticationError(InsuranceRAGException):
    """Exception raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_scheme: str = "Bearer",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        auth_details = details or {}
        auth_details["auth_scheme"] = auth_scheme
        
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=auth_details,
            request_id=request_id
        )
    
    def to_http_exception(self, status_code: int = 401) -> HTTPException:
        """Override to return 401 by default for authentication errors."""
        return super().to_http_exception(status_code)


class AuthorizationError(InsuranceRAGException):
    """Exception raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        auth_details = details or {}
        if required_permission:
            auth_details["required_permission"] = required_permission
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=auth_details,
            request_id=request_id
        )
    
    def to_http_exception(self, status_code: int = 403) -> HTTPException:
        """Override to return 403 by default for authorization errors."""
        return super().to_http_exception(status_code)


# VALIDATION EXCEPTIONS

class ValidationError(InsuranceRAGException):
    """Exception raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        if value is not None:
            validation_details["invalid_value"] = str(value)[:100]
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=validation_details,
            request_id=request_id
        )
    
    def to_http_exception(self, status_code: int = 400) -> HTTPException:
        """Override to return 400 by default for validation errors."""
        return super().to_http_exception(status_code)


class ConfigurationError(InsuranceRAGException):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        config_details = details or {}
        if config_key:
            config_details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=config_details,
            request_id=request_id
        )


# RATE LIMITING EXCEPTIONS

class RateLimitError(InsuranceRAGException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        rate_details = details or {}
        if limit:
            rate_details["rate_limit"] = limit
        if window:
            rate_details["time_window"] = window
        if retry_after:
            rate_details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=rate_details,
            request_id=request_id
        )
    
    def to_http_exception(self, status_code: int = 429) -> HTTPException:
        """Override to return 429 by default for rate limit errors."""
        return super().to_http_exception(status_code)


# UTILITY FUNCTIONS

def handle_exception(
    exception: Exception,
    request_id: Optional[str] = None,
    default_message: str = "An unexpected error occurred"
) -> InsuranceRAGException:
    """
    Convert any exception to an InsuranceRAGException with proper formatting.
    
    Args:
        exception: The original exception
        request_id: Optional request ID for tracking
        default_message: Default message if exception message is empty
        
    Returns:
        InsuranceRAGException with proper formatting
    """
    if isinstance(exception, InsuranceRAGException):
        return exception
    
    message = str(exception) if str(exception) else default_message
    
    return InsuranceRAGException(
        message=message,
        error_code=type(exception).__name__,
        details={"original_exception": type(exception).__name__},
        request_id=request_id
    )


def create_http_error_response(
    exception: Union[Exception, InsuranceRAGException],
    status_code: int = 500,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Create a standardized HTTP error response from any exception.
    
    Args:
        exception: The exception to convert
        status_code: HTTP status code to return
        request_id: Optional request ID for tracking
        
    Returns:
        HTTPException with structured error detail
    """
    if isinstance(exception, InsuranceRAGException):
        return exception.to_http_exception(status_code)
    
    # Convert regular exception to InsuranceRAGException first
    rag_exception = handle_exception(exception, request_id)
    return rag_exception.to_http_exception(status_code)