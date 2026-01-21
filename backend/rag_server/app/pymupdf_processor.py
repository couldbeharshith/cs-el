"""
PyMuPDF Document Processor

This module provides fast PDF and multi-format document processing using PyMuPDF (fitz).
It includes document type detection, text extraction (plain text and markdown), and error handling
"""

import os
import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from app.models import DocumentChunk, ChunkMetadata
from app.config import Config

# Import markdown processor
try:
    from app.pymupdf_markdown_processor import PyMuPDFMarkdownProcessor, PyMuPDFMarkdownError
    MARKDOWN_PROCESSOR_AVAILABLE = True
except ImportError:
    MARKDOWN_PROCESSOR_AVAILABLE = False
    PyMuPDFMarkdownProcessor = None
    PyMuPDFMarkdownError = None

logger = logging.getLogger(__name__)


@dataclass
class PyMuPDFError(Exception):
    """Custom exception for PyMuPDF processing errors"""
    message: str
    error_code: str = "PYMUPDF_ERROR"
    original_error: Optional[Exception] = None


class PyMuPDFProcessor:
    """
    Fast document processor using PyMuPDF for PDF and multi-format document processing.
    
    Supports PDF, XPS, EPUB, MOBI, FB2, CBZ, and SVG formats with optimized text extraction
    (both plain text and markdown) and proper resource management.
    """
    
    def __init__(self, use_markdown: bool = True):
        """Initialize PyMuPDF processor with supported formats and configuration."""
        self.supported_formats = [
            '.pdf', '.xps', '.epub', '.mobi', '.fb2', '.cbz', '.svg'
        ]
        
        # Additional formats that PyMuPDF can handle
        self.extended_formats = [
            '.oxps', '.html', '.htm', '.xml', '.txt', '.pptx'
        ]
        
        self.all_supported_formats = self.supported_formats + self.extended_formats
        
        # Initialize markdown processor if available and requested
        self.use_markdown = use_markdown and MARKDOWN_PROCESSOR_AVAILABLE
        if self.use_markdown:
            self.markdown_processor = PyMuPDFMarkdownProcessor()
            logger.info("PyMuPDF processor initialized with markdown support enabled")
        else:
            self.markdown_processor = None
            if use_markdown and not MARKDOWN_PROCESSOR_AVAILABLE:
                logger.warning("Markdown support requested but not available, falling back to plain text")
            else:
                logger.info("PyMuPDF processor initialized with plain text extraction only")
        
        logger.info(f"PyMuPDF processor initialized with {len(self.all_supported_formats)} supported formats")
    
    def can_process(self, doc_type: str) -> bool:
        """
        Check if PyMuPDF can process the given document type.
        
        Args:
            doc_type: Document type/extension (with or without dot)
            
        Returns:
            True if PyMuPDF can process this document type
        """
        if not doc_type:
            return False
        
        # Normalize extension format
        if not doc_type.startswith('.'):
            doc_type = f'.{doc_type}'
        
        doc_type = doc_type.lower()
        
        can_process = doc_type in self.all_supported_formats
        logger.debug(f"Can process '{doc_type}': {can_process}")
        
        return can_process
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported document formats.
        
        Returns:
            List of supported file extensions
        """
        return self.all_supported_formats.copy()
    
    def detect_document_type_from_path(self, file_path: str) -> str:
        """
        Detect document type from file path extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document type/extension (with dot)
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if not extension:
                # Try to detect from content if no extension
                return self._detect_from_content(file_path)
            
            logger.debug(f"Detected document type '{extension}' from path: {file_path}")
            return extension
            
        except Exception as e:
            logger.warning(f"Failed to detect document type from path {file_path}: {e}")
            return '.pdf'  # Default fallback
    
    def _detect_from_content(self, file_path: str) -> str:
        """
        Detect document type from file content magic bytes.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Detected document type/extension
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
            
            # Check for common magic bytes
            if header.startswith(b'%PDF'):
                return '.pdf'
            elif header.startswith(b'PK\x03\x04'):
                # Could be EPUB or other ZIP-based format
                if b'mimetype' in header and b'epub' in header:
                    return '.epub'
                return '.pdf'  # Default for ZIP-based
            elif header.startswith(b'<?xml'):
                return '.xml'
            elif header.startswith(b'<html') or header.startswith(b'<!DOCTYPE html'):
                return '.html'
            else:
                return '.pdf'  # Default fallback
                
        except Exception as e:
            logger.warning(f"Failed to detect document type from content: {e}")
            return '.pdf'
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF document using PyMuPDF.
        Uses markdown extraction if available and PDF has <300 pages, otherwise falls back to plain text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content (markdown or plain text)
            
        Raises:
            PyMuPDFError: If PDF processing fails
        """
        # Check page count first to decide processing method
        doc = None
        page_count = 0
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            doc = None
        except Exception as e:
            logger.warning(f"Failed to check page count for {file_path}: {e}")
            page_count = 0
        
        # Try markdown extraction first if available and PDF has <300 pages
        if self.use_markdown and self.markdown_processor and page_count < 300:
            try:
                logger.debug(f"Attempting markdown extraction for PDF with {page_count} pages: {file_path}")
                markdown_content = self.markdown_processor.extract_markdown_from_pdf(file_path, use_multiprocessing=True)
                logger.info(f"Successfully extracted {len(markdown_content)} characters of markdown from PDF")
                return markdown_content
            except PyMuPDFMarkdownError as e:
                logger.warning(f"Markdown extraction failed: {e.message}, falling back to plain text")
            except Exception as e:
                logger.warning(f"Unexpected error in markdown extraction: {e}, falling back to plain text")
        elif self.use_markdown and self.markdown_processor and page_count >= 300:
            logger.info(f"PDF has {page_count} pages (≥300), using plain text extraction instead of markdown")
        
        # Fallback to plain text extraction
        doc = None
        try:
            logger.debug(f"Opening PDF document for plain text extraction: {file_path}")
            doc = fitz.open(file_path)
            
            if doc.is_encrypted:
                logger.warning(f"PDF is encrypted: {file_path}")
                raise PyMuPDFError(
                    message="PDF document is encrypted and cannot be processed",
                    error_code="ENCRYPTED_PDF"
                )
            
            text_content = []
            page_count = len(doc)
            
            logger.debug(f"Processing {page_count} pages from PDF")
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        # Clean the extracted text
                        clean_text = self._clean_extracted_text(page_text)
                        if clean_text:
                            text_content.append(clean_text)
                            
                except Exception as e:
                    logger.warning(f"Failed to extract text from PDF page {page_num + 1}: {e}")
                    continue
            
            if not text_content:
                raise PyMuPDFError(
                    message="No readable text found in PDF document",
                    error_code="NO_TEXT_FOUND"
                )
            
            full_text = '\n\n'.join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters of plain text from PDF")
            
            return full_text
            
        except PyMuPDFError:
            raise
        except Exception as e:
            logger.error(f"PyMuPDF PDF processing failed for {file_path}: {e}")
            raise PyMuPDFError(
                message=f"Failed to process PDF document: {str(e)}",
                error_code="PDF_PROCESSING_ERROR",
                original_error=e
            )
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close PDF document: {cleanup_error}")
    
    def extract_text_from_document(self, file_path: str, doc_type: str = None) -> str:
        """
        Extract text from various document formats using PyMuPDF.
        
        Args:
            file_path: Path to the document file
            doc_type: Optional document type hint
            
        Returns:
            Extracted text content
            
        Raises:
            PyMuPDFError: If document processing fails
        """
        if not doc_type:
            doc_type = self.detect_document_type_from_path(file_path)
        
        # Normalize doc_type
        if not doc_type.startswith('.'):
            doc_type = f'.{doc_type}'
        doc_type = doc_type.lower()
        
        if not self.can_process(doc_type):
            raise PyMuPDFError(
                message=f"Document type '{doc_type}' is not supported by PyMuPDF",
                error_code="UNSUPPORTED_FORMAT"
            )
        
        # For PDF files, use the specialized PDF method
        if doc_type == '.pdf':
            return self.extract_text_from_pdf(file_path)
        
        # For other formats, use generic PyMuPDF processing
        doc = None
        try:
            logger.debug(f"Opening {doc_type} document: {file_path}")
            doc = fitz.open(file_path)
            
            text_content = []
            page_count = len(doc)
            
            logger.debug(f"Processing {page_count} pages from {doc_type} document")
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        clean_text = self._clean_extracted_text(page_text)
                        if clean_text:
                            text_content.append(clean_text)
                            
                except Exception as e:
                    logger.warning(f"Failed to extract text from {doc_type} page {page_num + 1}: {e}")
                    continue
            
            if not text_content:
                raise PyMuPDFError(
                    message=f"No readable text found in {doc_type} document",
                    error_code="NO_TEXT_FOUND"
                )
            
            full_text = '\n\n'.join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from {doc_type} document")
            
            return full_text
            
        except PyMuPDFError:
            raise
        except Exception as e:
            logger.error(f"PyMuPDF processing failed for {doc_type} document {file_path}: {e}")
            raise PyMuPDFError(
                message=f"Failed to process {doc_type} document: {str(e)}",
                error_code="DOCUMENT_PROCESSING_ERROR",
                original_error=e
            )
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close {doc_type} document: {cleanup_error}")
    
    def process_document_to_chunks(self, file_path: str, document_id: str, doc_type: str = None) -> List[DocumentChunk]:
        """
        Process document and return structured chunks with metadata.
        Uses markdown extraction for PDFs when available.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            doc_type: Optional document type hint
            
        Returns:
            List of DocumentChunk objects with extracted content and metadata
            
        Raises:
            PyMuPDFError: If document processing fails
        """
        if not doc_type:
            doc_type = self.detect_document_type_from_path(file_path)
        
        # Normalize doc_type for processing
        normalized_type = doc_type.lstrip('.')
        
        try:
            logger.info(f"Processing {doc_type} document to chunks: {file_path}")
            
            chunks = []
            
            if doc_type == '.pdf':
                # Check page count first to decide processing method
                doc = None
                page_count = 0
                try:
                    doc = fitz.open(file_path)
                    page_count = len(doc)
                    doc.close()
                    doc = None
                except Exception as e:
                    logger.warning(f"Failed to check page count for {file_path}: {e}")
                    page_count = 0
                
                # For PDFs, try markdown extraction first if available and <300 pages
                if self.use_markdown and self.markdown_processor and page_count < 300:
                    try:
                        logger.debug(f"Attempting markdown chunk extraction for PDF with {page_count} pages: {file_path}")
                        chunks = self.markdown_processor.process_document_to_markdown_chunks(file_path, document_id, use_multiprocessing=True)
                        logger.info(f"Successfully created {len(chunks)} markdown chunks from PDF")
                        return chunks
                    except PyMuPDFMarkdownError as e:
                        logger.warning(f"Markdown chunk extraction failed: {e.message}, falling back to plain text chunks")
                    except Exception as e:
                        logger.warning(f"Unexpected error in markdown chunk extraction: {e}, falling back to plain text chunks")
                elif self.use_markdown and self.markdown_processor and page_count >= 300:
                    logger.info(f"PDF has {page_count} pages (≥300), using plain text extraction instead of markdown")
                
                # Fallback to plain text page-by-page processing for PDFs
                chunks = self._process_pdf_to_chunks(file_path, document_id)
            else:
                # For other formats, extract text and create a single chunk
                full_text = self.extract_text_from_document(file_path, doc_type)
                metadata = ChunkMetadata(
                    document_id=document_id,
                    page_number=1,
                    line_number=1,
                    chunk_number=1,
                    document_type=normalized_type,
                    file_name=Path(file_path).name
                )
                
                chunk = DocumentChunk(content=full_text, metadata=metadata)
                chunks = [chunk]
            
            logger.info(f"Successfully created {len(chunks)} chunks from {doc_type} document")
            return chunks
            
        except PyMuPDFError:
            raise
        except Exception as e:
            logger.error(f"Failed to process {doc_type} document to chunks: {e}")
            raise PyMuPDFError(
                message=f"Failed to process document to chunks: {str(e)}",
                error_code="CHUNK_PROCESSING_ERROR",
                original_error=e
            )
    
    def _process_pdf_to_chunks(self, file_path: str, document_id: str) -> List[DocumentChunk]:
        """
        Process PDF document page-by-page to create individual chunks.
        
        Args:
            file_path: Path to the PDF file
            document_id: Unique identifier for the document
            
        Returns:
            List of DocumentChunk objects, one per page
        """
        doc = None
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            
            if doc.is_encrypted:
                raise PyMuPDFError(
                    message="PDF document is encrypted and cannot be processed",
                    error_code="ENCRYPTED_PDF"
                )
            
            page_count = len(doc)
            chunk_number = 1
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        clean_text = self._clean_extracted_text(page_text)
                        if clean_text:
                            metadata = ChunkMetadata(
                                document_id=document_id,
                                page_number=page_num + 1,
                                line_number=1,
                                chunk_number=chunk_number,
                                document_type='pdf',
                                file_name=Path(file_path).name
                            )
                            
                            chunk = DocumentChunk(content=clean_text, metadata=metadata)
                            chunks.append(chunk)
                            chunk_number += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {page_num + 1}: {e}")
                    continue
            
            return chunks
            
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close PDF document: {cleanup_error}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing whitespace.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
        
        # Check for PDF metadata patterns that indicate corrupted extraction
        pdf_metadata_patterns = [
            '%PDF', 'endobj', '/Type/', 'obj\n', '<<', '>>', 'stream', 'endstream',
            '/Filter', '/Length', '/Root', '/Info', 'xref', 'trailer'
        ]
        
        # If text contains too many metadata patterns, it's likely corrupted
        metadata_count = sum(1 for pattern in pdf_metadata_patterns if pattern in text)
        if metadata_count > 3:  # Allow some false positives
            logger.warning("Text contains excessive PDF metadata, likely corrupted extraction")
            return ""
        
        # Clean up whitespace and formatting
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive whitespace
                line = ' '.join(line.split())
                clean_lines.append(line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(clean_lines)
        
        # Remove excessive newlines (more than 2 consecutive)
        import re
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get document information and metadata using PyMuPDF.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document information
        """
        doc = None
        try:
            doc = fitz.open(file_path)
            
            info = {
                'page_count': len(doc),
                'is_encrypted': doc.is_encrypted,
                'format': doc.name.split('.')[-1].upper() if '.' in doc.name else 'UNKNOWN',
                'metadata': doc.metadata,
                'file_size': os.path.getsize(file_path),
                'file_name': Path(file_path).name
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
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to close document: {cleanup_error}")
    
    def validate_document(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if document can be processed by PyMuPDF.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            if os.path.getsize(file_path) == 0:
                return False, "File is empty"
            
            doc_type = self.detect_document_type_from_path(file_path)
            if not self.can_process(doc_type):
                return False, f"Document type '{doc_type}' is not supported"
            
            # Try to open the document
            doc = None
            try:
                doc = fitz.open(file_path)
                
                if doc.is_encrypted:
                    return False, "Document is encrypted"
                
                if len(doc) == 0:
                    return False, "Document has no pages"
                
                # Try to extract text from first page
                first_page = doc[0]
                test_text = first_page.get_text()
                
                return True, "Document is valid"
                
            finally:
                if doc:
                    doc.close()
                    
        except Exception as e:
            return False, f"Document validation failed: {str(e)}"