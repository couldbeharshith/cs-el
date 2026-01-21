"""
Insurance RAG API - Main FastAPI Application
Simplified architecture with consolidated services
"""

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from datetime import datetime, timezone
import logging
import time
import uuid
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from app.config import Config
from app.models import HackRXRequest, HackRXResponse, ErrorResponse
from app.services import (
    DocumentService,
    VectorService,
    LLMService,
    RelevanceService,
    QueryExpansionService,
)
from app.utils import (
    sanitize_input,
    validate_bearer_token,
    log_performance_metrics,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL), format=Config.LOG_FORMAT
    )
logger = logging.getLogger(__name__)

# Get application settings from Config
app_name = Config.APP_NAME
app_version = Config.APP_VERSION
debug_mode = Config.DEBUG

# Initialize services at module level
document_service = DocumentService()
vector_service = VectorService()
llm_service = LLMService()
relevance_service = RelevanceService()
query_expansion_service = QueryExpansionService()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown
    """
    # Startup
    logger.info(f"Starting {app_name} v{app_version} on {Config.PLATFORM}")

    # Validate required configuration
    missing_configs = Config.validate_required_config()
    if missing_configs:
        logger.error(f"Missing required configuration: {missing_configs}")
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_configs)}"
        )

    # Log configuration summary (without sensitive data)
    config_summary = Config.get_config_summary()
    logger.info(f"Application configuration: {config_summary}")

    # Initialize services
    # Note: Relevance checking now uses document-based similarity only
    logger.info(
        "Services initialized - using document-based relevance checking and multi-query processing"
    )
    logger.info(
        f"Direct Gemini upload enabled for supported file types < {Config.DIRECT_GEMINI_UPLOAD_THRESHOLD_MB}MB (PDF, text, HTML, images, audio, video)"
    )

    logger.info("Application startup complete and ready to serve requests")

    yield

    logger.info("Application shutdown complete")


# Log startup information
logger.info(f"Starting {app_name} v{app_version}")
logger.info(f"Debug mode: {debug_mode}")

# Create FastAPI app with simplified configuration
app = FastAPI(
    title=app_name,
    description="A high-accuracy RAG-based API for processing insurance policy queries with URL-based document processing and Pinecone vector storage",
    version=app_version,
    docs_url="/docs" if debug_mode else None,
    redoc_url="/redoc" if debug_mode else None,
    debug=debug_mode,
    lifespan=lifespan,
    openapi_url="/openapi.json" if debug_mode else None,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


# Add security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for request validation and headers."""
    start_time = time.time()

    # No request size limits - removed size restrictions
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            content_length = int(content_length)
            logger.info(
                f"Processing request with content length: {content_length} bytes"
            )
        except ValueError:
            pass  # Invalid content-length header, let it pass

    # Process request
    response = await call_next(request)

    # Add security headers
    for header_name, header_value in Config.SECURITY_HEADERS.items():
        response.headers[header_name] = header_value

    # Add platform-specific headers
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Platform"] = Config.PLATFORM
    response.headers["X-Serverless"] = str(Config.is_serverless())

    # Log long-running requests for monitoring (no limits enforced)
    if process_time > 60:  # Log requests over 1 minute for monitoring
        logger.info(
            f"Long-running request: {process_time:.2f}s",
            extra={
                "event_type": "long_request",
                "process_time": process_time,
                "path": str(request.url.path),
                "method": request.method,
            },
        )

    return response


# Authentication dependency
async def verify_bearer_token(authorization: Optional[str] = Header(None)):
    """Verify Bearer token authentication."""
    if not validate_bearer_token(authorization):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error_type="AuthenticationError",
                message="Invalid or missing Bearer token",
                details={"auth_scheme": "Bearer"},
                timestamp=datetime.now(timezone.utc).isoformat(),
                request_id=str(uuid.uuid4()),
            ).model_dump(),
        )


# HackRX API endpoint
@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(
    request: HackRXRequest, authorization: Optional[str] = Header(None)
):
    """
    HackRX API endpoint for processing insurance policy documents from URLs.

    Requires Bearer token authentication and processes documents from URLs
    to answer insurance policy questions with high accuracy using consolidated services.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"HackRX request started: {request_id} with {len(request.questions)} questions"
    )

    # Log request data for analysis
    logger.info(f"REQUEST DATA - URL: {request.documents}")
    logger.info(f"REQUEST DATA - Questions: {request.questions}")

    try:
        # Verify authentication
        await verify_bearer_token(authorization)

        # Sanitize all questions for processing
        sanitized_questions = [sanitize_input(q) for q in request.questions]
        logger.info(f"Sanitized {len(sanitized_questions)} questions for processing")

        # Process document
        logger.info(f"Processing document from URL")
        document_result = await document_service.process_document_from_url(
            request.documents, vector_service
        )

        # Check if this is an image document and handle it specially
        if document_result.namespace.startswith("image_") or (
            document_result.chunks
            and len(document_result.chunks) > 0
            and document_result.chunks[0].metadata.document_type == "image"
        ):

            logger.info(
                f"Image document detected, using direct Gemini vision API (no vector storage)"
            )

            # For image documents, process questions directly with Gemini vision API
            async def process_image_question(question: str, question_index: int) -> str:
                """Process a single question about an image using Gemini vision API."""
                try:
                    logger.info(
                        f"Processing image question {question_index + 1}: {question[:100]}..."
                    )

                    # Fetch the image file again for vision processing
                    content, content_type = await document_service.fetch_document(
                        request.documents
                    )

                    # Save image to temporary file for processing
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as temp_file:
                        temp_file.write(content)
                        temp_image_path = temp_file.name

                    try:
                        # Use LLM service for image analysis
                        answer = await llm_service.generate_image_answer(
                            question, temp_image_path
                        )
                        return answer
                    finally:
                        # Clean up temporary file
                        import os

                        try:
                            os.unlink(temp_image_path)
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Failed to cleanup temporary image file: {cleanup_error}"
                            )

                except Exception as e:
                    logger.error(
                        f"Error processing image question {question_index + 1}: {e}"
                    )
                    return f"I apologize, but I encountered an error while analyzing the image: {str(e)}"

            # Process all image questions
            logger.info(f"Processing {len(sanitized_questions)} image questions")

            # Create tasks for parallel execution of image questions
            image_question_tasks = [
                process_image_question(question, i)
                for i, question in enumerate(sanitized_questions)
            ]

            # Execute all image questions in parallel
            processed_answers = await asyncio.gather(
                *image_question_tasks, return_exceptions=True
            )

            # Create final answers array
            final_answers = []
            successful_questions = 0

            # Fill in processed answers
            for i, answer in enumerate(processed_answers):
                if isinstance(answer, Exception):
                    logger.error(
                        f"Image question {i+1} failed with exception: {answer}"
                    )
                    final_answers.append(
                        f"I apologize, but I encountered an error while analyzing the image."
                    )
                else:
                    final_answers.append(answer)
                    successful_questions += 1

            processing_time = time.time() - start_time

            # Log performance metrics
            await log_performance_metrics(
                request_id=request_id,
                endpoint="/api/v1/hackrx/run",
                processing_time=processing_time,
                questions_count=len(request.questions),
                success=True,
            )

            logger.info(
                f"HackRX image request completed: {request_id} in {processing_time:.2f}s "
                f"({successful_questions}/{len(request.questions)} questions successful)"
            )

            # Log response data for analysis
            logger.info(f"RESPONSE DATA - Image Answers: {final_answers}")

            return HackRXResponse(answers=final_answers)

        # Check if this is a direct Gemini document and handle it specially
        if (
            document_result.namespace.startswith("direct_gemini_")
            and hasattr(document_result, "direct_gemini_content")
            and document_result.direct_gemini_content is not None
        ):

            logger.info(
                f"Direct Gemini document detected, using direct Gemini API (no vector storage)"
            )

            # For direct Gemini documents, process questions directly with Gemini document API
            async def process_direct_gemini_question(
                question: str, question_index: int
            ) -> str:
                """Process a single question about a document using direct Gemini API."""
                try:
                    logger.info(
                        f"Processing direct Gemini question {question_index + 1}: {question[:100]}..."
                    )

                    # Get filename from chunks metadata
                    filename = "document"
                    if document_result.chunks and len(document_result.chunks) > 0:
                        filename = document_result.chunks[0].metadata.file_name

                    # Use LLM service for direct document analysis
                    answer = await llm_service.generate_direct_gemini_answer(
                        question,
                        document_result.direct_gemini_content,
                        document_result.direct_gemini_content_type,
                        filename,
                    )
                    return answer

                except Exception as e:
                    logger.error(
                        f"Error processing direct Gemini question {question_index + 1}: {e}"
                    )
                    return f"I apologize, but I encountered an error while analyzing the document: {str(e)}"

            # Process all direct Gemini questions
            logger.info(
                f"Processing {len(sanitized_questions)} direct Gemini questions"
            )

            # Create tasks for parallel execution of direct Gemini questions
            direct_gemini_question_tasks = [
                process_direct_gemini_question(question, i)
                for i, question in enumerate(sanitized_questions)
            ]

            # Execute all direct Gemini questions in parallel
            processed_answers = await asyncio.gather(
                *direct_gemini_question_tasks, return_exceptions=True
            )

            # Create final answers array
            final_answers = []
            successful_questions = 0

            for i, answer in enumerate(processed_answers):
                if isinstance(answer, Exception):
                    logger.error(
                        f"Direct Gemini question {i+1} failed with exception: {answer}"
                    )
                    final_answers.append(
                        f"I apologize, but I encountered an error while analyzing the document."
                    )
                else:
                    final_answers.append(answer)
                    successful_questions += 1

            processing_time = time.time() - start_time

            # Log performance metrics
            await log_performance_metrics(
                request_id=request_id,
                endpoint="/api/v1/hackrx/run",
                processing_time=processing_time,
                questions_count=len(request.questions),
                success=True,
            )

            logger.info(
                f"HackRX direct Gemini request completed: {request_id} in {processing_time:.2f}s "
                f"({successful_questions}/{len(request.questions)} questions successful)"
            )

            # Log response data for analysis
            logger.info(f"RESPONSE DATA - Direct Gemini Answers: {final_answers}")

            return HackRXResponse(answers=final_answers)

        # Check if this is a ZIP file rejection
        if document_result.document_hash == "zip_rejected":
            logger.info(f"ZIP file rejected, returning standard response")
            zip_rejection_message = "ZIP files cannot be processed using this API."

            # Return ZIP rejection message for all questions
            final_answers = [zip_rejection_message] * len(request.questions)

            processing_time = time.time() - start_time

            # Log performance metrics
            await log_performance_metrics(
                request_id=request_id,
                endpoint="/api/v1/hackrx/run",
                processing_time=processing_time,
                questions_count=len(request.questions),
                success=True,
            )

            logger.info(
                f"HackRX request completed (ZIP rejected): {request_id} in {processing_time:.2f}s "
                f"({len(request.questions)} questions, ZIP file rejected)"
            )

            return HackRXResponse(answers=final_answers)

        # Check if file is too large
        if document_result.document_hash == "file_too_large":
            logger.info(f"File too large, returning error message for all questions")
            file_too_large_message = (
                "The file uploaded is too large and is over the file limit of this API."
            )

            # Return the same error message for ALL questions
            final_answers = [file_too_large_message] * len(request.questions)

            processing_time = time.time() - start_time

            # Log performance metrics
            await log_performance_metrics(
                request_id=request_id,
                endpoint="/api/v1/hackrx/run",
                processing_time=processing_time,
                questions_count=len(request.questions),
                success=True,
            )

            logger.info(
                f"HackRX request completed (file too large): {request_id} in {processing_time:.2f}s "
                f"({len(request.questions)} questions, all answered with file too large message)"
            )

            return HackRXResponse(answers=final_answers)

        if document_result:
            if document_result.cached:
                logger.info(
                    f"Using cached document from namespace: {document_result.namespace}"
                )
            else:
                logger.info(
                    f"Processed new document into namespace: {document_result.namespace} with {len(document_result.chunks)} chunks"
                )

        # Step 1: Batch process all relevance checks in parallel
        logger.info("Starting relevance checks for all questions...")
        relevance_tasks = [
            relevance_service.check_query_relevance(
                question, i + 1, document_result.namespace, use_semaphore=False
            )
            for i, question in enumerate(sanitized_questions)
        ]
        relevance_results = await asyncio.gather(
            *relevance_tasks, return_exceptions=True
        )

        # Step 2: Batch process query expansion for all relevant questions in parallel
        logger.info("Starting query expansion for all relevant questions...")
        expansion_tasks = []
        question_to_expansion_map = {}

        # Get available Groq keys for distribution
        available_groq_keys = []
        if (
            query_expansion_service.groq_key_manager
            and query_expansion_service.groq_key_manager.api_keys
        ):
            available_groq_keys = (
                query_expansion_service.groq_key_manager.api_keys.copy()
            )

        for i, (question, relevance_result) in enumerate(
            zip(sanitized_questions, relevance_results)
        ):
            if isinstance(relevance_result, Exception) or not relevance_result:
                # Skip expansion for irrelevant questions
                question_to_expansion_map[i] = [question]
            else:
                # Pre-assign API key to avoid lock contention
                assigned_key = (
                    available_groq_keys[i % len(available_groq_keys)]
                    if available_groq_keys
                    else None
                )

                # Create expansion task for relevant questions (without semaphore for batch processing)
                expansion_tasks.append(
                    (
                        i,
                        query_expansion_service._generate_with_groq_direct(
                            question, assigned_key
                        ),
                    )
                )

        # Execute all expansion tasks in parallel
        if expansion_tasks:
            expansion_indices, expansion_coroutines = zip(*expansion_tasks)
            expansion_results = await asyncio.gather(
                *expansion_coroutines, return_exceptions=True
            )

            # Map results back to questions
            for idx, result in zip(expansion_indices, expansion_results):
                if isinstance(result, Exception):
                    logger.debug(f"Query expansion failed for Q{idx + 1}: {result}")
                    question_to_expansion_map[idx] = [sanitized_questions[idx]]
                else:
                    question_to_expansion_map[idx] = result

        # Fill in any missing mappings
        for i in range(len(sanitized_questions)):
            if i not in question_to_expansion_map:
                question_to_expansion_map[i] = [sanitized_questions[i]]

        # Define async function to process individual questions
        async def process_single_question(question: str, question_index: int) -> str:
            """Enhanced question processing with pre-computed query expansion."""
            question_start_time = time.time()

            try:
                logger.debug(
                    f"Processing question {question_index + 1}: {question[:100]}..."
                )

                # Step 1: Use pre-computed relevance check
                relevance_result = relevance_results[question_index]
                if isinstance(relevance_result, Exception) or not relevance_result:
                    logger.info(f"Q{question_index + 1}: Not relevant to document")
                    return "This question is not relevant to the provided document content."

                logger.debug(f"Question {question_index + 1} passed relevance check")

                # Ensure namespace is ready for search (wait for indexing if needed)
                if not document_result.cached:
                    logger.debug(
                        f"Checking if namespace {document_result.namespace} is ready for search..."
                    )
                    is_ready = await vector_service.is_namespace_ready(
                        document_result.namespace,
                        min_vectors=max(
                            1, len(document_result.chunks) // 2
                        ),  # At least half the chunks should be indexed
                    )
                    if not is_ready:
                        logger.info(
                            f"Namespace {document_result.namespace} not ready, waiting for indexing..."
                        )
                        await asyncio.sleep(1)  # Reduced wait time

                # Step 2: Use pre-computed diverse queries
                logger.info(f"📝 Q{question_index + 1}: {question}")
                diverse_queries = question_to_expansion_map[question_index]

                # Step 3: Multi-query vector search with integrated reranking
                final_chunks = []
                if len(diverse_queries) > 1:
                    # Use multi-query search (includes deduplication and reranking)
                    try:
                        final_chunks = await vector_service.search_multiple_queries(
                            queries=diverse_queries,
                            namespace=document_result.namespace,
                            original_query=question,
                            top_k=Config.PINECONE_TOP_K,
                        )
                    except Exception as e:
                        logger.debug(
                            f"Multi-query search failed for Q{question_index + 1}: {e}"
                        )
                        final_chunks = []

                # Fallback to single query search if multi-query failed or returned no results
                if not final_chunks:
                    try:
                        search_result = await vector_service.search_documents(
                            query=question,
                            namespace=document_result.namespace,
                            use_semaphore=False,
                        )
                        # For single query, apply reranking if we have chunks
                        if search_result.chunks:
                            final_chunks = (
                                await vector_service.rerank_chunks_with_original_query(
                                    chunks=search_result.chunks, original_query=question
                                )
                            )
                        else:
                            final_chunks = []
                        logger.debug(
                            f"Single-query fallback: {len(final_chunks)} chunks for Q{question_index + 1}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Single-query search also failed for question {question_index + 1}: {e}"
                        )
                        final_chunks = []

                # Check if we found any relevant chunks and retry if needed
                if not final_chunks:
                    logger.warning(
                        f"No relevant chunks found for question {question_index + 1}, namespace may not be ready"
                    )

                    # If document was just processed (not cached), wait a bit more and retry once
                    if not document_result.cached:
                        logger.info(
                            "Retrying search after additional wait for indexing..."
                        )
                        await asyncio.sleep(1)  # Reduced retry wait time

                        # Retry with single query search
                        try:
                            search_result = await vector_service.search_documents(
                                query=question,
                                namespace=document_result.namespace,
                                use_semaphore=False,
                            )
                            if search_result.chunks:
                                final_chunks = await vector_service.rerank_chunks_with_original_query(
                                    chunks=search_result.chunks, original_query=question
                                )
                            else:
                                final_chunks = []
                            logger.debug(
                                f"Retry found {len(final_chunks)} relevant chunks"
                            )
                        except Exception as e:
                            logger.error(
                                f"Retry search failed for question {question_index + 1}: {e}"
                            )
                            final_chunks = []

                # Step 5: LLM response generation (without semaphore for batch processing)
                if final_chunks:
                    try:
                        answer = await llm_service.generate_answer(
                            question=question,
                            context_chunks=final_chunks,
                            use_semaphore=False,
                        )
                    except Exception as e:
                        logger.error(
                            f"LLM answer generation failed for question {question_index + 1}: {e}"
                        )
                        answer = "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question."
                else:
                    # Fallback when no chunks are found
                    answer = "Please provide the policy context.\n\nI cannot answer this question without the actual policy document or a summary of its relevant sections. The document may still be processing or the question may not be related to the provided document content."

                question_time = time.time() - question_start_time
                logger.debug(
                    f"Question {question_index + 1} processed in {question_time:.2f}s"
                )

                return answer

            except Exception as e:
                question_time = time.time() - question_start_time
                logger.error(
                    f"Error processing question {question_index + 1} after {question_time:.2f}s: {e}"
                )

                # Return a user-friendly error message instead of exposing internal errors
                return f"I apologize, but I encountered an error while processing this question. Please try rephrasing your question."

        # Process all questions concurrently using asyncio.gather
        logger.info(
            f"Starting parallel processing of {len(sanitized_questions)} questions"
        )

        # Create a semaphore to limit concurrent question processing
        question_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)

        async def process_question_with_semaphore(
            question: str, question_index: int
        ) -> str:
            """Wrapper to process question with semaphore control."""
            async with question_semaphore:
                return await process_single_question(question, question_index)

        # Create tasks for parallel execution of all questions
        question_tasks = [
            process_question_with_semaphore(question, i)
            for i, question in enumerate(sanitized_questions)
        ]

        # Execute all questions in parallel with proper exception handling
        processed_answers = await asyncio.gather(
            *question_tasks, return_exceptions=True
        )

        # Create final answers array
        final_answers = []
        successful_questions = 0

        # Fill in processed answers
        for i, answer in enumerate(processed_answers):
            if isinstance(answer, Exception):
                logger.error(f"Question {i+1} failed with exception: {answer}")
                final_answers.append(
                    f"I apologize, but I encountered an error while processing this question. Please try rephrasing your question."
                )
            else:
                final_answers.append(answer)
                successful_questions += 1

        processing_time = time.time() - start_time

        # Log comprehensive performance metrics
        await log_performance_metrics(
            request_id=request_id,
            endpoint="/api/v1/hackrx/run",
            processing_time=processing_time,
            questions_count=len(request.questions),
            success=True,
        )

        logger.info(
            f"HackRX request completed: {request_id} in {processing_time:.2f}s "
            f"({successful_questions}/{len(request.questions)} questions successful, "
            f"document cached: {document_result.cached if document_result else 'N/A'})"
        )

        # Log response data for analysis
        logger.info(f"RESPONSE DATA - Answers: {final_answers}")

        return HackRXResponse(answers=final_answers)

    except HTTPException:
        # Re-raise HTTP exceptions (like authentication failures)
        raise

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"HackRX request failed: {request_id} after {processing_time:.2f}s - {e}"
        )

        # Log performance metrics for failed request
        await log_performance_metrics(
            request_id=request_id,
            endpoint="/api/v1/hackrx/run",
            processing_time=processing_time,
            questions_count=(
                len(request.questions) if hasattr(request, "questions") else 0
            ),
            success=False,
        )

        # Return structured error response
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_type=type(e).__name__,
                message=str(e),
                details={
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "endpoint": "/api/v1/hackrx/run",
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                request_id=request_id,
            ).model_dump(),
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    response_data = {
        "message": app_name,
        "version": app_version,
        "platform": Config.PLATFORM,
        "serverless": Config.is_serverless(),
        "status": "operational",
        "endpoints": {"hackrx": "/api/v1/hackrx/run", "health": "/health"},
    }

    # Include docs URLs only in debug mode
    if debug_mode:
        response_data["docs"] = "/docs"
        response_data["redoc"] = "/redoc"
        response_data["openapi"] = "/openapi.json"

    return response_data


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    """
    try:
        # Basic health check
        config_summary = Config.get_config_summary()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": Config.PLATFORM,
            "version": app_version,
            "serverless": Config.is_serverless(),
            "config": {
                "gemini_model": config_summary.get("gemini_model"),
                "embedding": "Pinecone built-in",
                "pinecone_index": config_summary.get("pinecone_index"),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# Server configuration for local development
if __name__ == "__main__":
    import uvicorn

    if Config.PLATFORM == "huggingface":
        host = "0.0.0.0"
        port = 7860
        workers = 1
    else:
        # Local development
        host = Config.HOST
        port = Config.PORT
        workers = 1

    limit_concurrency = Config.MAX_CONCURRENT_REQUESTS

    logger.info(f"Starting uvicorn server:")
    logger.info(f"  Platform: {Config.PLATFORM}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Concurrency limit: {limit_concurrency}")
    logger.info(f"  Debug mode: {debug_mode}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        limit_concurrency=limit_concurrency,
        reload=debug_mode,
        access_log=debug_mode,
        log_level="info" if debug_mode else "warning",
    )
