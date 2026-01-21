"""
LLM Key Manager for handling multiple API keys with rotation, health tracking, and automatic recovery.

This module provides the LLMKeyManager class that manages multiple API keys for LLM services,
implements key rotation, tracks key health, and provides automatic recovery mechanisms.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    """Status of an API key."""
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    COOLDOWN = "cooldown"


@dataclass
class APIKeyHealth:
    """Health tracking information for an API key."""
    key_id: str
    service_type: str  # "gemini" or "groq"
    status: KeyStatus = KeyStatus.HEALTHY
    failure_count: int = 0
    success_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Rate limiting tracking
    requests_this_minute: int = 0
    minute_window_start: Optional[datetime] = None
    
    def is_healthy(self) -> bool:
        """Check if the key is currently healthy and usable."""
        now = datetime.utcnow()
        
        # Check if in cooldown period
        if self.cooldown_until and now < self.cooldown_until:
            return False
        
        # Check if too many consecutive failures
        if self.consecutive_failures >= 3:
            return False
        
        # Check rate limiting (basic implementation)
        if self.minute_window_start:
            if now - self.minute_window_start < timedelta(minutes=1):
                if self.requests_this_minute >= 60:  # Conservative rate limit
                    return False
            else:
                # Reset rate limiting window
                self.requests_this_minute = 0
                self.minute_window_start = now
        
        return self.status in [KeyStatus.HEALTHY, KeyStatus.RATE_LIMITED]
    
    def mark_success(self):
        """Mark a successful API call."""
        self.status = KeyStatus.HEALTHY
        self.success_count += 1
        self.last_success = datetime.utcnow()
        self.consecutive_failures = 0
        
        # Update rate limiting
        now = datetime.utcnow()
        if not self.minute_window_start or now - self.minute_window_start >= timedelta(minutes=1):
            self.minute_window_start = now
            self.requests_this_minute = 1
        else:
            self.requests_this_minute += 1
    
    def mark_failure(self, error_type: str = "unknown"):
        """Mark a failed API call."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure = datetime.utcnow()
        
        # Determine status based on error type
        if "rate" in error_type.lower() or "quota" in error_type.lower():
            self.status = KeyStatus.RATE_LIMITED
            # Set cooldown for rate limiting
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=5)
        else:
            self.status = KeyStatus.FAILED
            # Set cooldown based on consecutive failures
            cooldown_minutes = min(self.consecutive_failures * 2, 30)  # Max 30 minutes
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
    
    def reset_health(self):
        """Reset the key health to healthy state."""
        self.status = KeyStatus.HEALTHY
        self.consecutive_failures = 0
        self.cooldown_until = None


class LLMKeyManager:
    """
    Manages multiple API keys with rotation, health tracking, and automatic recovery.
    
    This class handles:
    - Key rotation across multiple API keys
    - Health tracking and failure detection
    - Automatic recovery after cooldown periods
    - Rate limiting awareness
    - Service-specific key management
    """
    
    def __init__(self, service_type: str, api_keys: List[str]):
        """
        Initialize the key manager.
        
        Args:
            service_type: Type of service ("gemini" or "groq")
            api_keys: List of API keys to manage
        """
        self.service_type = service_type.lower()
        self.api_keys = api_keys
        self.current_key_index = 0
        self._lock = asyncio.Lock()
        
        # Initialize health tracking for each key
        self.key_health: Dict[str, APIKeyHealth] = {}
        for i, key in enumerate(api_keys):
            key_id = f"{service_type}_{i+1}"
            self.key_health[key] = APIKeyHealth(
                key_id=key_id,
                service_type=service_type
            )
        
        logger.info(f"Initialized {service_type} key manager with {len(api_keys)} keys")
    
    async def get_next_key(self) -> Optional[str]:
        """
        Get the next available healthy API key.
        
        Returns:
            Next available API key or None if no healthy keys available
        """
        async with self._lock:
            if not self.api_keys:
                return None
            
            # First, try to find a healthy key starting from current index
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_key_index]
                health = self.key_health[key]
                
                if health.is_healthy():
                    # Move to next key for next request
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    return key
                
                # Move to next key
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            # If no healthy keys found, check if any keys can be recovered
            await self._attempt_key_recovery()
            
            # Try again after recovery attempt
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_key_index]
                health = self.key_health[key]
                
                if health.is_healthy():
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    return key
                
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            logger.warning(f"No healthy {self.service_type} keys available")
            return None
    
    async def mark_key_success(self, key: str):
        """
        Mark a successful API call for the given key.
        
        Args:
            key: The API key that was used successfully
        """
        async with self._lock:
            if key in self.key_health:
                self.key_health[key].mark_success()
                logger.debug(f"Marked {self.service_type} key as successful")
    
    async def mark_key_failed(self, key: str, error: str):
        """
        Mark a failed API call for the given key.
        
        Args:
            key: The API key that failed
            error: Error message or type
        """
        async with self._lock:
            if key in self.key_health:
                self.key_health[key].mark_failure(error)
                logger.warning(f"Marked {self.service_type} key as failed: {error}")
    
    def get_healthy_keys(self) -> List[str]:
        """
        Get list of currently healthy keys.
        
        Returns:
            List of healthy API keys
        """
        healthy_keys = []
        for key, health in self.key_health.items():
            if health.is_healthy():
                healthy_keys.append(key)
        return healthy_keys
    
    def reset_key_health(self, key: str):
        """
        Reset the health status of a specific key.
        
        Args:
            key: The API key to reset
        """
        if key in self.key_health:
            self.key_health[key].reset_health()
            logger.info(f"Reset health for {self.service_type} key")
    
    def reset_all_keys_health(self):
        """Reset health status for all keys."""
        for key in self.key_health:
            self.key_health[key].reset_health()
        logger.info(f"Reset health for all {self.service_type} keys")
    
    async def _attempt_key_recovery(self):
        """
        Attempt to recover keys that may have passed their cooldown period.
        """
        now = datetime.utcnow()
        recovered_count = 0
        
        for key, health in self.key_health.items():
            if health.cooldown_until and now >= health.cooldown_until:
                health.status = KeyStatus.HEALTHY
                health.cooldown_until = None
                health.consecutive_failures = max(0, health.consecutive_failures - 1)
                recovered_count += 1
        
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} {self.service_type} keys from cooldown")
    
    def get_key_statistics(self) -> Dict[str, Dict[str, any]]:
        """
        Get statistics for all managed keys.
        
        Returns:
            Dictionary with key statistics
        """
        stats = {}
        for key, health in self.key_health.items():
            # Mask the key for security
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            stats[masked_key] = {
                "key_id": health.key_id,
                "status": health.status.value,
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "consecutive_failures": health.consecutive_failures,
                "is_healthy": health.is_healthy(),
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "cooldown_until": health.cooldown_until.isoformat() if health.cooldown_until else None,
            }
        return stats
    
    def get_service_health_summary(self) -> Dict[str, any]:
        """
        Get a summary of the service health.
        
        Returns:
            Dictionary with service health summary
        """
        total_keys = len(self.api_keys)
        healthy_keys = len(self.get_healthy_keys())
        
        total_successes = sum(health.success_count for health in self.key_health.values())
        total_failures = sum(health.failure_count for health in self.key_health.values())
        
        return {
            "service_type": self.service_type,
            "total_keys": total_keys,
            "healthy_keys": healthy_keys,
            "unhealthy_keys": total_keys - healthy_keys,
            "health_percentage": (healthy_keys / total_keys * 100) if total_keys > 0 else 0,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": (total_successes / (total_successes + total_failures) * 100) 
                          if (total_successes + total_failures) > 0 else 0,
        }