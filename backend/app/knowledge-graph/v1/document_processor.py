"""
Document Processing Module
Uses Azure Document Intelligence for OCR processing
"""

import os
import logging
from typing import Optional
from pathlib import Path

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class AzureDocumentProcessor:
    """Process documents using Azure Document Intelligence"""
    
    def __init__(self, endpoint: str, key: str, model: str = "prebuilt-layout"):
        self.endpoint = endpoint
        self.key = key
        self.model = model
        
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
    
    def process_file(self, file_path: str) -> str:
        """Process a file and return markdown content"""
        logger.info(f"Processing file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                self.model,
                document=f
            )
        
        result = poller.result()
        
        # Return content as markdown
        return result.content
    
    def process_url(self, url: str) -> str:
        """Process a document from URL"""
        logger.info(f"Processing URL: {url}")
        
        poller = self.client.begin_analyze_document_from_url(
            self.model,
            document_url=url
        )
        
        result = poller.result()
        return result.content
