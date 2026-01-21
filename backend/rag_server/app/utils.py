"""
Essential utilities module for the Insurance RAG API.
Consolidated from multiple utility files to provide only essential functionality.
"""

import asyncio
import functools
import logging
import time
import re
import html
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Essential input sanitization utilities."""
    
    # Regex patterns for validation
    URL_PATTERN = re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    @staticmethod
    def sanitize_string(
        text: str, 
        max_length: int = 1000,
        remove_html: bool = True,
        normalize_whitespace: bool = True
    ) -> str:
        """
        Sanitize string input to prevent XSS and injection attacks.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            remove_html: Whether to remove/escape HTML
            normalize_whitespace: Whether to normalize whitespace
            
        Returns:
            Sanitized string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Initial cleanup
        sanitized = text.strip()
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Handle HTML
        if remove_html:
            # Remove script tags and their content
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            # Remove other HTML tags
            sanitized = re.sub(r'<[^>]*>', '', sanitized)
            # Escape remaining HTML entities
            sanitized = html.escape(sanitized)
        
        # Remove dangerous patterns
        dangerous_patterns = [
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',  # Event handlers
            r'union\s+select',  # SQL injection
            r'drop\s+table',
            r'insert\s+into',
            r'delete\s+from',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        if normalize_whitespace:
            sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip()
        
        return sanitized
    
    @staticmethod
    def sanitize_questions(questions: List[str]) -> List[str]:
        """
        Sanitize a list of questions.
        
        Args:
            questions: List of questions to sanitize
            
        Returns:
            List of sanitized questions
        """
        if not questions or not isinstance(questions, list):
            return []
        
        sanitized_questions = []
        
        for question in questions:
            if not isinstance(question, str):
                continue
            
            # Sanitize the question
            sanitized = InputSanitizer.sanitize_string(
                question,
                max_length=1000,
                remove_html=True,
                normalize_whitespace=True
            )
            
            # Only add non-empty questions with minimum length
            if sanitized and len(sanitized.strip()) >= 3:
                # Ensure question ends with appropriate punctuation
                sanitized = sanitized.rstrip()
                if sanitized and not sanitized[-1] in '.?!':
                    sanitized += '?'
                
                sanitized_questions.append(sanitized)
        
        return sanitized_questions


def validate_bearer_token(authorization_header: Optional[str]) -> bool:
    """
    Validate Bearer token against configured API key.
    
    Args:
        authorization_header: Authorization header value
        
    Returns:
        True if token is valid, False otherwise
    """
    from app.config import Config
    
    if not authorization_header or not isinstance(authorization_header, str):
        return False
    
    # Check Bearer token format
    if not authorization_header.startswith('Bearer '):
        return False
    
    # Extract token
    token = authorization_header[7:].strip()  # Remove 'Bearer ' prefix
    
    # Validate against configured API key
    return token == Config.API_KEY and bool(Config.API_KEY)


# PERFORMANCE LOGGING UTILITIES

class PerformanceLogger:
    """Essential performance logging utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
        self.operation_times = {}

    def log_api_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        processing_time: float,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Log API request performance metrics.
        
        Args:
            request_id: Unique request identifier
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            processing_time: Request processing time in seconds
            client_ip: Client IP address
            user_agent: Client user agent
        """
        log_data = {
            'request_id': request_id,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'processing_time_seconds': round(processing_time, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        if client_ip:
            log_data['client_ip'] = client_ip
        
        if user_agent:
            log_data['user_agent'] = user_agent
        
        # Determine log level based on status code and processing time
        if status_code >= 500:
            self.logger.error(f"API request error: {json.dumps(log_data)}")
        elif status_code >= 400:
            self.logger.warning(f"API request client error: {json.dumps(log_data)}")
        elif processing_time > 10.0:
            self.logger.warning(f"Slow API request: {json.dumps(log_data)}")
        else:
            self.logger.info(f"API request: {json.dumps(log_data)}")


# Global performance logger instance
performance_logger = PerformanceLogger()


# CONVENIENCE FUNCTIONS FOR MAIN.PY

def sanitize_input(text: str) -> str:
    """
    Convenience function for sanitizing input text.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    return InputSanitizer.sanitize_string(text)


async def log_performance_metrics(
    request_id: str,
    endpoint: str,
    processing_time: float,
    questions_count: int = 0,
    success: bool = True
) -> None:
    """
    Convenience function for logging performance metrics.
    
    Args:
        request_id: Unique request identifier
        endpoint: API endpoint
        processing_time: Processing time in seconds
        questions_count: Number of questions processed
        success: Whether the request was successful
    """
    additional_data = {
        'questions_count': questions_count,
        'avg_time_per_question': processing_time / max(questions_count, 1)
    }
    
    performance_logger.log_api_request(
        request_id=request_id,
        endpoint=endpoint,
        method="POST",
        status_code=200 if success else 500,
        processing_time=processing_time
    )