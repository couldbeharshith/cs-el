from supabase import create_client, Client
from app.config.settings import settings
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class SupabaseLogger:
    def __init__(self):
        self.client: Optional[Client] = None
        self.enabled = settings.ENABLE_REQUEST_LOGGING
        
        if self.enabled and settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
            try:
                self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
                logger.info("Supabase logger initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                self.enabled = False
        else:
            logger.warning("Supabase logging disabled - missing configuration")
            self.enabled = False
    
    async def log_api_request(
        self,
        document_url: str,
        questions: List[str],
        answers: List[str],
        processing_time: float,
        document_metadata: Dict[str, Any],
        raw_response: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[str]:        
        if not self.enabled or not self.client:
            return None
        
        try:
            request_id = str(uuid.uuid4())
            
            log_entry = {
                "id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "document_url": document_url,
                "questions": questions,
                "answers": answers,
                "processing_time": processing_time,
                "document_metadata": document_metadata,
                "raw_response": raw_response,
                "success": success,
                "error_message": error_message,
                "questions_count": len(questions),
                "chunks_processed": document_metadata.get("chunks_processed", 0),
                "vector_store": document_metadata.get("vector_store", "unknown")
            }
            
            self.client.table("hackrx_requests").insert(log_entry).execute()
            return request_id
                
        except Exception as e:
            logger.error(f"Error logging API request: {str(e)}")
            return None
    

supabase_logger = SupabaseLogger()
