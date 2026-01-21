"""
PyMuPDF Markdown Processor

This module provides fast PDF to markdown conversion using PyMuPDF4LLM with multiprocessing
for efficient processing of large documents.
"""

import os
import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

try:
    from pymupdf4llm import to_markdown
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    to_markdown = None

from app.models import DocumentChunk, ChunkMetadata
from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PyMuPDFMarkdownError(Exception):
    """Custom exception for PyMuPDF markdown processing errors"""
    message: str
    error_code: str = "PYMUPDF_MARKDOWN_ERROR"
    original_error: Optional[Exception] = None


def process_batch_to_markdown(vector):
    """
    Processes a range of pages from a PDF and returns the combined Markdown text.
    
    Args:
        vector: A list containing [process_index, total_cpus, filename].
    """
    try:
        idx, cpu, filename = vector
        
        # Open the document within this process
        doc = fitz.open(filename)
        num_pages = doc.page_count

        # Calculate the page range this process is responsible for
        pages_per_cpu = int(num_pages / cpu + 1)
        start_page = idx * pages_per_cpu
        end_page = min(start_page + pages_per_cpu, num_pages)

        # If this process has no pages to work on, return an empty string
        if start_page >= end_page:
            doc.close()
            return ""

        # Call to_markdown ONCE per process, giving it the doc and a list of pages
        pages_to_process = list(range(start_page, end_page))
        md_text = to_markdown(doc, pages=pages_to_process, show_progress=False)
        
        doc.close()
        
        logger.debug(f"Process {idx} successfully finished pages {start_page} through {end_page - 1}.")
        
        # Return the resulting markdown text for the entire batch
        return md_text
    except Exception as e:
        # Catch any other potential errors in the worker
        logger.error(f"An error occurred in process {idx}: {e}")
        return ""


class PyMuPDFMarkdownProcessor:
    """
    Fast PDF to markdown processor using PyMuPDF4LLM with multiprocessing support.
    
    Provides efficient markdown extraction from PDF documents with proper formatting
    and structure preservation.
    """
    
    def __init__(self):
        """Initialize PyMuPDF markdown processor."""
        self.supported_formats = ['.pdf']
        
        if not PYMUPDF4LLM_AVAILABLE:
            logger.warning("pymupdf4llm is not available. Markdown processing will not work.")
        
        logger.info(f"PyMuPDF markdown processor initialized (pymupdf4llm available: {PYMUPDF4LLM_AVAILABLE})")
    
    def can_process(self, doc_type: str) -> bool:
        """
        Check if PyMuPDF markdown processor can process the given document type.
        
        Args:
            doc_type: Document type/extension (with or without dot)
            
        Returns:
            True if processor can handle this document type
        """
        if not PYMUPDF4LLM_AVAILABLE:
            return False
        
        if not doc_type:
            return False
        
        # Normalize extension format
        if not doc_type.startswith('.'):
            doc_type = f'.{doc_type}'
        
        doc_type = doc_type.lower()
        
        can_process = doc_type in self.supported_formats
        logger.debug(f"Can process '{doc_type}' for markdown: {can_process}")
        
        return can_process
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats for markdown processing.
        
        Returns:
            List of supported file extensions
        """
        return self.supported_formats.copy()
    
    def extract_markdown_from_pdf(self, file_path: str, use_multiprocessing: bool = True) -> str:
        """
        Extract markdown from PDF document using PyMuPDF4LLM with optional multiprocessing.
        
        Args:
            file_path: Path to the PDF file
            use_multiprocessing: Whether to use multiprocessing for large PDFs
            
        Returns:
            Extracted markdown content
            
        Raises:
            PyMuPDFMarkdownError: If PDF processing fails
        """
        if not PYMUPDF4LLM_AVAILABLE:
            raise PyMuPDFMarkdownError(
                message="pymupdf4llm is not available. Please install it to use markdown processing.",
                error_code="PYMUPDF4LLM_NOT_AVAILABLE"
            )
        
        doc = None
        try:
            logger.debug(f"Opening PDF document for markdown extraction: {file_path}")
            doc = fitz.open(file_path)
            
            if doc.is_encrypted:
                logger.warning(f"PDF is encrypted: {file_path}")
                raise PyMuPDFMarkdownError(
                    message="PDF document is encrypted and cannot be processed",
                    error_code="ENCRYPTED_PDF"
                )
            
            page_count = len(doc)
            logger.debug(f"Processing {page_count} pages from PDF for markdown")
            
            # Decide whether to use multiprocessing based on page count and user preference
            if use_multiprocessing and page_count > 10:  # Use multiprocessing for larger PDFs
                logger.info(f"Using multiprocessing for PDF with {page_count} pages")
                doc.close()  # Close doc before multiprocessing
                doc = None
                
                cpu = cpu_count()
                vectors = [(i, cpu, file_path) for i in range(cpu)]
                
                with Pool(processes=cpu) as pool:
                    results_from_batches = pool.map(process_batch_to_markdown, vectors)
                
                # Join the Markdown text from all batches
                full_markdown_document = "\n".join(results_from_batches)
                
            else:
                # Single-threaded processing for smaller PDFs
                logger.info(f"Using single-threaded processing for PDF with {page_count} pages")
                full_markdown_document = to_markdown(doc, show_progress=False)
            
            if not full_markdown_document or not full_markdown_document.strip():
                raise PyMuPDFMarkdownError(
                    message="No readable markdown content found in PDF document",
                    error_code="NO_MARKDOWN_FOUND"
                )
            
            logger.info(f"Successfully extracted {len(full_markdown_document)} characters of markdown from PDF")
            
            return full_markdown_document.strip()
            
        except PyMuPDFMarkdownError:
            raise
        except Exception as e:
            logger.error(f"PyMuPDF markdown processing failed for {file_path}: {e}")
            raise PyMuPDFMarkdownError(
                message=f"Failed to process PDF to markdown: {str(e)}",
                error_code="MARKDOWN_PROCESSING_ERROR",
                original_error=e
            )
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close PDF document: {cleanup_error}")
    
    def process_document_to_markdown_chunks(self, file_path: str, document_id: str, use_multiprocessing: bool = True) -> List[DocumentChunk]:
        """
        Process PDF document and return structured chunks with markdown content.
        
        Args:
            file_path: Path to the PDF file
            document_id: Unique identifier for the document
            use_multiprocessing: Whether to use multiprocessing for processing
            
        Returns:
            List of DocumentChunk objects with markdown content and metadata
            
        Raises:
            PyMuPDFMarkdownError: If document processing fails
        """
        try:
            logger.info(f"Processing PDF document to markdown chunks: {file_path}")
            
            # Extract markdown using PyMuPDF4LLM
            full_markdown = self.extract_markdown_from_pdf(file_path, use_multiprocessing)
            
            # For now, create a single chunk with the full markdown content
            # This can be enhanced later with page-by-page processing if needed
            metadata = ChunkMetadata(
                document_id=document_id,
                page_number=1,
                line_number=1,
                chunk_number=1,
                document_type='pdf',
                file_name=Path(file_path).name
            )
            
            chunk = DocumentChunk(content=full_markdown, metadata=metadata)
            chunks = [chunk]
            
            logger.info(f"Successfully created {len(chunks)} markdown chunks from PDF document")
            return chunks
            
        except PyMuPDFMarkdownError:
            raise
        except Exception as e:
            logger.error(f"Failed to process PDF document to markdown chunks: {e}")
            raise PyMuPDFMarkdownError(
                message=f"Failed to process document to markdown chunks: {str(e)}",
                error_code="MARKDOWN_CHUNK_PROCESSING_ERROR",
                original_error=e
            )
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get document information for markdown processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing document information
        """
        doc = None
        try:
            doc = fitz.open(file_path)
            
            info = {
                'page_count': len(doc),
                'is_encrypted': doc.is_encrypted,
                'format': 'PDF',
                'metadata': doc.metadata,
                'file_size': os.path.getsize(file_path),
                'file_name': Path(file_path).name,
                'markdown_supported': PYMUPDF4LLM_AVAILABLE
            }
            
            # Add page dimensions for first page if available
            if len(doc) > 0:
                first_page = doc[0]
                rect = first_page.rect
                info['page_dimensions'] = {
                    'width': rect.width,
                    'height': rect.height
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get document info for {file_path}: {e}")
            return {
                'error': str(e),
                'file_name': Path(file_path).name,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'markdown_supported': PYMUPDF4LLM_AVAILABLE
            }
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close document: {cleanup_error}")
    
    def validate_document(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if PDF document can be processed for markdown extraction.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not PYMUPDF4LLM_AVAILABLE:
                return False, "pymupdf4llm is not available"
            
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            if os.path.getsize(file_path) == 0:
                return False, "File is empty"
            
            # Try to open the document
            doc = None
            try:
                doc = fitz.open(file_path)
                
                if doc.is_encrypted:
                    return False, "Document is encrypted"
                
                if len(doc) == 0:
                    return False, "Document has no pages"
                
                return True, "Document is valid for markdown processing"
                
            finally:
                if doc:
                    doc.close()
                    
        except Exception as e:
            return False, f"Document validation failed: {str(e)}"