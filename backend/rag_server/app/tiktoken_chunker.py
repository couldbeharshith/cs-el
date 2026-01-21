"""
TikToken-based text chunking service for the Insurance RAG API.

This module provides token-aware text chunking using TikToken with cl100k_base encoding,
designed to replace the existing manual chunking approach with proper token boundaries
and configurable overlap.
"""

import logging
import tiktoken
from typing import List, Tuple
from dataclasses import dataclass

from app.models import DocumentChunk, ChunkMetadata
from app.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TokenChunkInfo:
    """Information about a token-based chunk."""
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    token_count: int


class TikTokenError(Exception):
    """Custom exception for TikToken processing errors."""
    pass


class TikTokenChunker:
    """
    Token-aware text chunker using TikToken with cl100k_base encoding.
    
    This class provides intelligent text chunking that respects token boundaries
    and preserves word boundaries when possible, with configurable overlap
    between chunks for better context preservation.
    """
    
    def __init__(self, 
                 encoding_name: str = "cl100k_base",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the TikToken chunker.
        
        Args:
            encoding_name: TikToken encoding to use (default: cl100k_base)
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            
        Raises:
            TikTokenError: If encoding cannot be loaded
        """
        self.encoding_name = encoding_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate parameters
        if chunk_size <= 0:
            raise TikTokenError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise TikTokenError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise TikTokenError("chunk_overlap must be less than chunk_size")
        
        # Load TikToken encoding
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"Loaded TikToken encoding: {encoding_name}")
        except Exception as e:
            error_msg = f"Failed to load TikToken encoding '{encoding_name}': {e}"
            logger.error(error_msg)
            raise TikTokenError(error_msg)
    
    def chunk_text(self, text: str, metadata: ChunkMetadata) -> List[DocumentChunk]:
        """
        Chunk text using TikToken with word boundary preservation and overlap.
        
        Args:
            text: Text to chunk
            metadata: Base metadata for chunks
            
        Returns:
            List of DocumentChunk objects with token-aware chunking
            
        Raises:
            TikTokenError: If chunking fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Encode text to tokens
            tokens = self.encoding.encode(text)
            total_tokens = len(tokens)
            
            logger.debug(f"Text encoded to {total_tokens} tokens for chunking")
            
            # If text fits in one chunk, return as-is
            if total_tokens <= self.chunk_size:
                enhanced_metadata = self._enhance_metadata_with_tokens(
                    metadata, 0, total_tokens, 0, len(text), total_tokens
                )
                return [DocumentChunk(content=text, metadata=enhanced_metadata)]
            
            # Create overlapping chunks
            chunks = self._create_chunks_with_overlap(tokens, text, metadata)
            
            logger.info(f"Created {len(chunks)} chunks from {total_tokens} tokens")
            return chunks
            
        except Exception as e:
            error_msg = f"Failed to chunk text: {e}"
            logger.error(error_msg)
            raise TikTokenError(error_msg)
    
    def _create_chunks_with_overlap(self, tokens: List[int], text: str, 
                                   metadata: ChunkMetadata) -> List[DocumentChunk]:
        """
        Create overlapping chunks from token list with word boundary preservation.
        
        Args:
            tokens: List of token IDs
            text: Original text
            metadata: Base metadata for chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start_token = 0
        chunk_number = 0
        
        while start_token < len(tokens):
            # Calculate end token position
            end_token = min(start_token + self.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_token:end_token]
            
            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Find character positions in original text
            start_char, end_char = self._find_char_positions(
                tokens, text, start_token, end_token
            )
            
            # Apply word boundary preservation if not at text end
            if end_token < len(tokens):
                adjusted_end_char = self._find_word_boundary(text, end_char)
                if adjusted_end_char != end_char:
                    # Re-encode adjusted text to get proper token boundary
                    adjusted_text = text[start_char:adjusted_end_char]
                    adjusted_tokens = self.encoding.encode(adjusted_text)
                    chunk_text = adjusted_text
                    end_char = adjusted_end_char
                    end_token = start_token + len(adjusted_tokens)
            
            # Create chunk with enhanced metadata
            if chunk_text.strip():
                enhanced_metadata = self._enhance_metadata_with_tokens(
                    metadata, start_token, end_token, start_char, end_char, len(chunk_tokens)
                )
                enhanced_metadata.chunk_number = metadata.chunk_number * 100 + chunk_number
                
                chunk = DocumentChunk(content=chunk_text.strip(), metadata=enhanced_metadata)
                chunks.append(chunk)
                chunk_number += 1
            
            # Move to next chunk with overlap
            next_start = end_token - self.chunk_overlap
            if next_start <= start_token:
                # Prevent infinite loop
                next_start = start_token + max(1, self.chunk_size // 2)
            
            start_token = next_start
            
            # Safety check to prevent infinite loops
            if start_token >= len(tokens):
                break
        
        return chunks
    
    def _find_char_positions(self, tokens: List[int], text: str, 
                           start_token: int, end_token: int) -> Tuple[int, int]:
        """
        Find character positions corresponding to token positions.
        
        Args:
            tokens: Full token list
            text: Original text
            start_token: Starting token index
            end_token: Ending token index
            
        Returns:
            Tuple of (start_char, end_char) positions
        """
        try:
            # Decode tokens before start to find start character position
            if start_token == 0:
                start_char = 0
            else:
                prefix_text = self.encoding.decode(tokens[:start_token])
                start_char = len(prefix_text)
            
            # Decode tokens up to end to find end character position
            if end_token >= len(tokens):
                end_char = len(text)
            else:
                prefix_text = self.encoding.decode(tokens[:end_token])
                end_char = len(prefix_text)
            
            return start_char, end_char
            
        except Exception as e:
            logger.warning(f"Failed to find exact character positions: {e}")
            # Fallback to approximate positions
            chars_per_token = len(text) / len(tokens) if tokens else 1
            start_char = int(start_token * chars_per_token)
            end_char = int(end_token * chars_per_token)
            return start_char, min(end_char, len(text))
    
    def _find_word_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest word boundary to avoid cutting mid-word.
        
        Args:
            text: Text to search in
            position: Character position to adjust
            
        Returns:
            Adjusted position at word boundary
        """
        if position >= len(text):
            return len(text)
        
        # If we're already at a word boundary, return as-is
        if position == 0 or text[position].isspace():
            return position
        
        # Look backwards for a space within reasonable distance
        search_distance = min(100, position)  # Don't search too far back
        
        for i in range(position, max(0, position - search_distance), -1):
            if text[i].isspace():
                return i + 1  # Return position after the space
        
        # If no space found, look forward (less preferred)
        search_distance = min(100, len(text) - position)
        for i in range(position, min(len(text), position + search_distance)):
            if text[i].isspace():
                return i
        
        # If no word boundary found, return original position
        return position
    
    def _enhance_metadata_with_tokens(self, base_metadata: ChunkMetadata,
                                    start_token: int, end_token: int,
                                    start_char: int, end_char: int,
                                    token_count: int) -> ChunkMetadata:
        """
        Enhance chunk metadata with token information.
        
        Args:
            base_metadata: Original metadata
            start_token: Starting token index
            end_token: Ending token index
            start_char: Starting character position
            end_char: Ending character position
            token_count: Number of tokens in chunk
            
        Returns:
            Enhanced ChunkMetadata with token information
        """
        # Create new metadata with token information
        enhanced_metadata = ChunkMetadata(
            document_id=base_metadata.document_id,
            page_number=base_metadata.page_number,
            line_number=base_metadata.line_number,
            chunk_number=base_metadata.chunk_number,
            document_type=base_metadata.document_type,
            file_name=base_metadata.file_name
        )
        
        # Add token information as additional attributes
        # Note: These will be stored as extra fields in Pydantic model
        enhanced_metadata.__dict__.update({
            'token_count': token_count,
            'encoding_used': self.encoding_name,
            'chunk_start_token': start_token,
            'chunk_end_token': end_token,
            'chunk_start_char': start_char,
            'chunk_end_char': end_char
        })
        
        return enhanced_metadata
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the loaded encoding.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
            
        Raises:
            TikTokenError: If token counting fails
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            error_msg = f"Failed to count tokens: {e}"
            logger.error(error_msg)
            raise TikTokenError(error_msg)
    
    def get_encoding_info(self) -> dict:
        """
        Get information about the current encoding.
        
        Returns:
            Dictionary with encoding information
        """
        return {
            'encoding_name': self.encoding_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_token_value': getattr(self.encoding, 'max_token_value', 'unknown')
        }
    
    @classmethod
    def create_from_config(cls) -> 'TikTokenChunker':
        """
        Create TikTokenChunker instance using configuration values.
        
        Returns:
            Configured TikTokenChunker instance
            
        Raises:
            TikTokenError: If configuration is invalid or encoding fails
        """
        try:
            return cls(
                encoding_name=Config.TIKTOKEN_ENCODING,
                chunk_size=Config.TIKTOKEN_CHUNK_SIZE,
                chunk_overlap=Config.TIKTOKEN_CHUNK_OVERLAP
            )
        except Exception as e:
            error_msg = f"Failed to create TikTokenChunker from config: {e}"
            logger.error(error_msg)
            raise TikTokenError(error_msg)


def create_fallback_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> callable:
    """
    Create a fallback character-based chunker when TikToken fails.
    
    Args:
        chunk_size: Character-based chunk size
        chunk_overlap: Character-based overlap
        
    Returns:
        Fallback chunking function
    """
    def fallback_chunk_text(text: str, metadata: ChunkMetadata) -> List[DocumentChunk]:
        """
        Fallback character-based chunking when TikToken fails.
        
        Args:
            text: Text to chunk
            metadata: Base metadata for chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_number = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text) and not text[end].isspace():
                last_space = chunk_text.rfind(' ')
                if last_space > start + (chunk_size * 0.7):
                    end = start + last_space
                    chunk_text = text[start:end]
            
            if chunk_text.strip():
                # Create fallback metadata
                fallback_metadata = ChunkMetadata(
                    document_id=metadata.document_id,
                    page_number=metadata.page_number,
                    line_number=metadata.line_number,
                    chunk_number=metadata.chunk_number * 100 + chunk_number,
                    document_type=metadata.document_type,
                    file_name=metadata.file_name
                )
                
                chunk = DocumentChunk(content=chunk_text.strip(), metadata=fallback_metadata)
                chunks.append(chunk)
                chunk_number += 1
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
            if start <= 0:
                start = end
        
        logger.warning(f"Used fallback chunking, created {len(chunks)} chunks")
        return chunks
    
    return fallback_chunk_text