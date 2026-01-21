"""
Document chunking and entity extraction for insurance policies
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from config import PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """Represents a chunk of policy document"""
    content: str
    index: int
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)


class PolicyChunker:
    """Smart chunking for insurance policy documents"""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        
        # Policy section patterns
        self.section_patterns = {
            'coverage': r'(?i)(coverage|benefits?\s+covered|sum\s+insured)',
            'exclusions': r'(?i)(exclusions?|not\s+covered|exceptions?)',
            'definitions': r'(?i)(definitions?|meanings?|terms?\s+defined)',
            'claims': r'(?i)(claims?\s+procedure|notification|settlement)',
            'premium': r'(?i)(premium|payment|fees?)',
            'conditions': r'(?i)(conditions?|terms?\s+and\s+conditions?)',
            'waiting_period': r'(?i)(waiting\s+periods?|probation)',
            'limits': r'(?i)(limits?|sub-?limits?|maximum|ceiling)'
        }
    
    def chunk_document(self, content: str, metadata: Optional[Dict] = None) -> List[PolicyChunk]:
        """Chunk document into semantic sections"""
        logger.info("Starting document chunking")
        
        # Try to split by major sections first
        sections = self._split_by_sections(content)
        
        chunks = []
        chunk_index = 0
        
        for section_title, section_content in sections:
            # Further chunk large sections
            if len(section_content) > self.config.chunk_size:
                sub_chunks = self._split_text(section_content, self.config.chunk_size, self.config.chunk_overlap)
                for sub_chunk in sub_chunks:
                    chunks.append(PolicyChunk(
                        content=sub_chunk,
                        index=chunk_index,
                        section_title=section_title,
                        metadata=metadata or {}
                    ))
                    chunk_index += 1
            else:
                chunks.append(PolicyChunk(
                    content=section_content,
                    index=chunk_index,
                    section_title=section_title,
                    metadata=metadata or {}
                ))
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by major sections"""
        sections = []
        
        # Find section headers
        section_pattern = r'^(?:#{1,3}\s+)?(\d+\.?\s*)?([A-Z][A-Z\s]+)$'
        lines = content.split('\n')
        
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            # Check if this is a section header
            if re.match(section_pattern, line) and len(line) < 100:
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections if sections else [("Full Document", content)]
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + chunk_size * 0.5:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        return chunks


class PolicyEntityExtractor:
    """Extract insurance-specific entities from policy text"""
    
    def __init__(self):
        self.entity_patterns = {
            'sum_insured': r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:lakh|lac|crore))?',
            'percentage': r'\d+(?:\.\d+)?%',
            'waiting_period': r'\d+\s*(?:days?|months?|years?)\s*(?:waiting\s*period)?',
            'age_limit': r'\d+\s*years?\s*(?:of\s*age)?',
            'room_rent': r'(?:room\s*rent|room\s*charges?).*?(?:₹|Rs\.?|INR)?\s*[\d,]+',
            'co_payment': r'(?:co-?pay(?:ment)?|deductible).*?\d+%?',
            'policy_period': r'(?:policy\s*period|coverage\s*period).*?\d+\s*(?:days?|months?|years?)',
            'diseases': r'(?i)(?:diabetes|hypertension|cancer|heart\s*disease|stroke|kidney|liver|asthma|arthritis)',
            'procedures': r'(?i)(?:surgery|operation|transplant|dialysis|chemotherapy|radiation|angioplasty)',
            'benefits': r'(?i)(?:maternity|dental|optical|wellness|preventive|vaccination|mental\s*health)',
            'plans': r'(?i)(?:silver|gold|diamond|platinum)\s*(?:plan|variant)?'
        }
    
    def extract_entities(self, chunk: PolicyChunk) -> PolicyChunk:
        """Extract entities from chunk content"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, chunk.content, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        chunk.entities = entities
        return chunk