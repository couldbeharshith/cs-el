import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from app.config.settings import settings
from .table_stitcher import TableStitcher

@dataclass 
class ChunkMetadata:
    """Enhanced metadata for insurance document chunks"""
    chunk_id: str
    chunk_type: str 
    importance_score: float
    section_hierarchy: List[str] 
    has_tables: bool
    table_count: int
    has_monetary_values: bool
    monetary_amounts: List[str]
    has_exclusions: bool
    exclusion_phrases: List[str]
    policy_terms: List[str]
    cross_references: List[str]
    original_headers: List[str]
    chunk_position: int
    source_lines: Tuple[int, int]
    has_important_notes: bool
    important_notes: List[str]
    has_structured_lists: bool
    list_types: List[str]  # e.g., ['numbered', 'lettered', 'roman'] 
    # Structure-aware additions
    element_type: str = "text"  # one of: text, complete_table, table_row, table_cell, footnote
    table_id: Optional[str] = None
    table_headers: Optional[List[str]] = None
    row_index: Optional[int] = None
    column_data: Optional[Dict[str, str]] = None
    is_complete_table: bool = False
    footnote_ref: Optional[str] = None
    cell_reference: Optional[str] = None

class InsuranceDocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        min_chunk_size: int = 300,
        use_semantic_chunker: bool = True,
        embeddings_model: Optional[Any] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_semantic_chunker = use_semantic_chunker
        
        # Initialize embeddings for semantic chunker
        self.embeddings = embeddings_model or self._get_default_embeddings()
        
        # Initialize table stitcher
        self.table_stitcher = TableStitcher()
        
        # Azure OCR specific patterns for insurance documents
        self._compile_insurance_patterns()
        
        # Initialize LangChain splitters
        self._setup_splitters()
    
    def _get_default_embeddings(self):
        """Get default embeddings with proper API key configuration"""
        try:
            if not settings.OPENAI_API_KEY:
                print("Warning: OPENAI_API_KEY not found in settings. Semantic chunking disabled.")
                return None
            
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI embeddings: {e}. Semantic chunking disabled.")
            return None
    
    def _compile_insurance_patterns(self):
        """Compile patterns specific to insurance documents and Azure OCR output"""
        
        # Azure OCR often outputs XML-like tags
        self.table_patterns = [
            re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE),
            re.compile(r'\|[^|]+\|([^|]+\|)+', re.MULTILINE),  # Markdown tables
            re.compile(r'\+[-=]+\+.*?\+[-=]+\+', re.DOTALL),   # ASCII tables
            re.compile(r'<tr[^>]*>.*?</tr>', re.DOTALL | re.IGNORECASE),  # Table rows
        ]
        
        # Azure OCR page metadata patterns
        self.page_metadata_patterns = [
            re.compile(r'<!--\s*PageFooter\s*=\s*["\']?[^"\']*["\']?\s*-->', re.IGNORECASE),
            re.compile(r'<!--\s*PageNumber\s*=\s*["\']?[^"\']*["\']?\s*-->', re.IGNORECASE),
            re.compile(r'<!--\s*PageBreak\s*-->', re.IGNORECASE),
        ]
        
        # Bold/emphasis patterns from Azure OCR
        self.emphasis_patterns = [
            re.compile(r'<strong[^>]*>(.*?)</strong>', re.IGNORECASE),
            re.compile(r'<b[^>]*>(.*?)</b>', re.IGNORECASE),
            re.compile(r'\*\*(.*?)\*\*'),
            re.compile(r'__(.*?)__'),
        ]
        
        # List and numbering patterns from Azure OCR
        self.list_patterns = [
            # Escaped numbered lists: 1\. 2\. etc.
            re.compile(r'^(\d+\\\.)\s+(.+)', re.MULTILINE),
            # Dash/bullet points: \- or -
            re.compile(r'^\\?-\s+(.+)', re.MULTILINE),
            # Lettered points: a. b. c. or a) b) c)
            re.compile(r'^([a-z][\.\)])\s+(.+)', re.MULTILINE | re.IGNORECASE),
            # Roman numerals: i. ii. iii. or i) ii) iii)
            re.compile(r'^([ivxlcdm]+[\.\)])\s+(.+)', re.MULTILINE | re.IGNORECASE),
            # Subsection numbering: 2.1, 2.2, etc.
            re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s+(.+)', re.MULTILINE),
        ]
        
        # Important note patterns
        self.note_patterns = [
            re.compile(r'\b(Note\s*:\s*.+?)(?=\n\n|\n[A-Z]|\Z)', re.DOTALL | re.IGNORECASE),
            re.compile(r'\b(Important\s*:\s*.+?)(?=\n\n|\n[A-Z]|\Z)', re.DOTALL | re.IGNORECASE),
            re.compile(r'\b(Attention\s*:\s*.+?)(?=\n\n|\n[A-Z]|\Z)', re.DOTALL | re.IGNORECASE),
            re.compile(r'\b(Warning\s*:\s*.+?)(?=\n\n|\n[A-Z]|\Z)', re.DOTALL | re.IGNORECASE),
        ]
        
        # Insurance-specific monetary patterns
        self.monetary_patterns = [
            re.compile(r'₹\s*[\d,]+(?:\.\d{2})?(?:\s*(?:lakh|crore|thousand)s?)?'),
            re.compile(r'Rs\.?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:lakh|crore|thousand)s?)?'),
            re.compile(r'INR\s*[\d,]+(?:\.\d{2})?'),
            re.compile(r'[\d,]+\s*(?:lakh|crore|thousand)s?', re.IGNORECASE),
        ]
        
        # Insurance exclusion patterns
        self.exclusion_patterns = [
            re.compile(r'\b(?:not covered|excluded|limitation|restriction)\b', re.IGNORECASE),
            re.compile(r'\b(?:provided that|except|however|subject to|notwithstanding)\b', re.IGNORECASE),
            re.compile(r'\b(?:shall not|does not cover|will not pay|maximum limit)\b', re.IGNORECASE),
        ]
        
        # Policy terms patterns  
        self.policy_terms_patterns = [
            re.compile(r'\b(?:sum insured|premium|deductible|copay|coverage)\b', re.IGNORECASE),
            re.compile(r'\b(?:proportionate deduction|room rent|icu charges|daycare)\b', re.IGNORECASE),
            re.compile(r'\b(?:pre-hospitalization|post-hospitalization|waiting period)\b', re.IGNORECASE),
            re.compile(r'\b(?:cashless|reimbursement|ayush|ambulance|maternity)\b', re.IGNORECASE),
        ]
        
        # Cross-reference patterns
        self.cross_ref_patterns = [
            re.compile(r'(?:as mentioned in|refer to|as per|subject to|in accordance with)\s+(?:section|clause|table|point|paragraph)\s*[\d.]+', re.IGNORECASE),
            re.compile(r'(?:section|clause)\s*[\d.]+(?:\.\d+)*', re.IGNORECASE),
        ]
    
    def _setup_splitters(self):
        """Initialize LangChain text splitters"""
        
        # 1. Header-based splitter for hierarchical structure
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
            # Also handle numbered sections from Azure OCR
        ]
        
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        # 2. Recursive character splitter as fallback
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 3. Semantic chunker for complex sections
        if self.use_semantic_chunker and self.embeddings:
            try:
                self.semantic_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",  # or "standard_deviation"
                    breakpoint_threshold_amount=85  # Adjust based on your needs
                )
            except Exception as e:
                print(f"Warning: Could not initialize semantic chunker: {e}")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None
    
    def split_text(self, text: str) -> List[str]:
        """
        Main interface method compatible with DocumentEmbedder
        Returns list of chunk texts
        """
        chunks_with_metadata = self.chunk_with_metadata(text)
        return [chunk_text for chunk_text, _ in chunks_with_metadata]
    
    def chunk_with_metadata(self, text: str) -> List[Tuple[str, ChunkMetadata]]:
        """
        Advanced chunking with comprehensive metadata
        Returns list of (chunk_text, metadata) tuples
        """
        
        # Step 1: Preprocess Azure OCR text
        preprocessed_text = self._preprocess_azure_ocr_text(text)
        
        # Step 2: Extract and preserve tables 
        tables, text_without_tables = self._extract_and_preserve_tables(preprocessed_text)
        
        # Step 3: Use MarkdownHeaderTextSplitter for hierarchical structure
        header_chunks = self._split_by_headers(text_without_tables)
        
        # Step 4: Apply semantic chunking to complex sections
        refined_chunks = self._apply_semantic_refinement(header_chunks)
        
        # Step 5: Process tables as separate high-priority chunks
        table_chunks = self._process_table_for_structure_aware_rag_chunks(tables)
        
        # Step 6: Combine and optimize all chunks
        all_chunks = refined_chunks + table_chunks
        final_chunks = self._post_process_chunks(all_chunks)
        
        return final_chunks
    
    def _preprocess_azure_ocr_text(self, text: str) -> str:
        """Preprocess Azure OCR text to clean up XML tags and formatting"""
        
        # Step 1: Remove Azure OCR page metadata comments
        # Preserve PageBreak markers for downstream page index detection
        for pattern in self.page_metadata_patterns:
            if 'PageBreak' in getattr(pattern, 'pattern', ''):
                continue
            text = pattern.sub('', text)
        
        # Step 2: Convert XML-like bold tags to markdown
        for pattern in self.emphasis_patterns:
            text = pattern.sub(r'**\1**', text)
        
        # Step 3: Clean up common Azure OCR artifacts
        text = re.sub(r'</?p[^>]*>', '', text)  # Remove paragraph tags
        text = re.sub(r'</?div[^>]*>', '', text)  # Remove div tags
        text = re.sub(r'<br[^>]*/?>', '\n', text)  # Convert breaks to newlines
        
        # Step 4: Normalize different list formats to consistent markdown
        # Handle escaped numbered lists: 1\. 2\. -> 1. 2.
        text = re.sub(r'^(\d+)\\\.(\s+)', r'\1.\2', text, flags=re.MULTILINE)
        
        # Handle dash points: \- -> -
        text = re.sub(r'^\\-(\s+)', r'-\1', text, flags=re.MULTILINE)
        
        # Convert subsection numbering to markdown headers
        # 2.1 Some Title -> ### 2.1 Some Title
        text = re.sub(r'^(\d+\.\d+(?:\.\d+)*)\s+([A-Z][^.]*?)(?=\n|$)', r'### \1 \2', text, flags=re.MULTILINE)
        
        # Convert main section numbering to headers
        # 2. MAIN SECTION -> ## 2. MAIN SECTION
        text = re.sub(r'^(\d+\.\s+[A-Z\s]+):?\s*$', r'## \1', text, flags=re.MULTILINE)
        
        # Step 5: Handle lettered and roman numeral lists consistently
        # Convert lettered points to proper list format
        text = re.sub(r'^([a-z][\.\)])\s+', r'- **\1** ', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Convert roman numeral points to proper list format  
        text = re.sub(r'^([ivxlcdm]+[\.\)])\s+', r'- **\1** ', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Step 6: Enhance important notes formatting
        for pattern in self.note_patterns:
            text = pattern.sub(r'\n\n**\1**\n', text)
        
        # Step 7: Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = text.strip()  # Remove leading/trailing whitespace
        
        return text
    
    def _extract_and_preserve_tables(self, text: str) -> Tuple[List[Dict], str]:
        """Extract tables WITH surrounding context (page breaks)"""
        tables = []
        text_without_tables = text
        page_segments = re.split(r'(<!-- PageBreak -->)', text)
        
        for i, pattern in enumerate(self.table_patterns):
            matches = list(pattern.finditer(text_without_tables))
            
            for j, match in enumerate(reversed(matches)):
                table_text = match.group(0)
                # Map positions to original text (not the mutated text_without_tables)
                orig_start_pos = text.find(table_text)
                orig_end_pos = (orig_start_pos + len(table_text)) if orig_start_pos != -1 else match.end()
                page_index = self._find_page_segment(orig_start_pos if orig_start_pos != -1 else match.start(), page_segments)
                # Keep local positions for replacement in the current mutable buffer
                start_pos_local = match.start()
                end_pos_local = match.end()
                
                table_info = {
                    'id': f'table_{len(tables)}',
                    'text': table_text,
                    'start_pos': orig_start_pos if orig_start_pos != -1 else match.start(),
                    'end_pos': orig_end_pos,
                    'page_index': page_index,  
                    'has_pagebreak_after': self._has_pagebreak_after(orig_end_pos, text),  
                    'importance_score': self._calculate_table_importance(table_text),
                    'pattern_type': i,
                    'position': len(tables),
                    'headers': self._extract_table_headers(table_text)
                }
                
                tables.append(table_info)
                
                # Replace with placeholder
                placeholder = f"\n\n[TABLE_{table_info['id']}_PLACEHOLDER]\n\n"
                text_without_tables = (text_without_tables[:start_pos_local] + 
                                     placeholder + 
                                     text_without_tables[end_pos_local:])
        
        # Stitch multi-page tables
        stitched_tables = self.table_stitcher.stitch_tables(tables, text)
        
        return stitched_tables, text_without_tables

    def _find_page_segment(self, start_pos: int, page_segments: List[str]) -> int:
        """Return page index for a given start position using pre-split page segments.
        We reconstruct offsets by walking the segments in order.
        Even indices are content; odd indices are the literal PageBreak tokens.
        """
        if start_pos is None or start_pos < 0:
            return 0
        pos = 0
        page_idx = 0
        for i, seg in enumerate(page_segments):
            next_pos = pos + len(seg)
            if start_pos < next_pos:
                # Number of page breaks encountered is the count of odd indices before i
                page_idx = i // 2
                return page_idx
            pos = next_pos
        # If beyond last, return last page
        return max(0, (len(page_segments) - 1) // 2)

    def _has_pagebreak_after(self, end_pos: int, text: str) -> bool:
        """Check if a PageBreak marker occurs after the given end position in original text."""
        if end_pos is None or end_pos < 0 or end_pos >= len(text):
            return False
        return re.search(r'<!--\s*PageBreak\s*-->', text[end_pos:]) is not None
    
    def _calculate_table_importance(self, table_text: str) -> float:
        """Calculate importance score for tables"""
        score = 5.0  # Base score for being a table
        
        # Monetary values significantly increase importance
        monetary_matches = sum(len(p.findall(table_text)) for p in self.monetary_patterns)
        score += monetary_matches * 3.0
        
        # Policy terms add importance
        policy_matches = sum(len(p.findall(table_text)) for p in self.policy_terms_patterns)
        score += policy_matches * 2.0
        
        # Exclusion terms are critical
        exclusion_matches = sum(len(p.findall(table_text)) for p in self.exclusion_patterns)
        score += exclusion_matches * 4.0
        
        # Table complexity (rows/columns)
        row_count = table_text.count('\n') + table_text.count('<tr')
        score += min(row_count * 0.5, 5.0)  # Cap bonus at 5 points
        
        return score
    
    def _extract_table_headers(self, table_text: str) -> List[str]:
        """Extract table headers for multi-page matching"""
        headers = []
        
        if '<table' in table_text.lower():
            # HTML table headers
            header_cells = re.findall(r'<th[^>]*>(.*?)</th>', table_text, re.IGNORECASE | re.DOTALL)
            if not header_cells:  # Fallback to first row td
                first_row = re.search(r'<tr[^>]*>(.*?)</tr>', table_text, re.IGNORECASE | re.DOTALL)
                if first_row:
                    header_cells = re.findall(r'<td[^>]*>(.*?)</td>', first_row.group(1), re.IGNORECASE | re.DOTALL)
            headers = [re.sub(r'<[^>]+>', '', cell).strip().lower() for cell in header_cells]
            
        elif '|' in table_text:
            # Markdown table headers
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            # Skip separator lines like |---|: find the first non-separator pipe line as header
            header_line = next((line for line in lines if '|' in line and not re.match(r'^\|[\s\-:\|]+\|$', line)), None)
            if header_line:
                headers = [cell.strip().lower() for cell in header_line.split('|') if cell.strip()]
        
        return headers[:5]  # Limit to first 5 headers for comparison
    
    def _split_by_headers(self, text: str) -> List[Dict]:
        """Split text using MarkdownHeaderTextSplitter"""
        
        try:
            # Use LangChain's header splitter
            header_splits = self.header_splitter.split_text(text)
            
            chunks = []
            for i, doc in enumerate(header_splits):
                chunk_info = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'chunk_type': 'header_section',
                    'position': i,
                    'importance_score': self._calculate_section_importance(doc.page_content),
                    'headers': list(doc.metadata.values()) if doc.metadata else []
                }
                chunks.append(chunk_info)
            
            return chunks
            
        except Exception as e:
            print(f"Header splitting failed: {e}. Falling back to recursive splitter.")
            
            # Fallback to recursive splitter
            recursive_splits = self.recursive_splitter.split_text(text)
            chunks = []
            for i, chunk_text in enumerate(recursive_splits):
                chunk_info = {
                    'text': chunk_text,
                    'metadata': {},
                    'chunk_type': 'recursive_fallback',
                    'position': i,
                    'importance_score': self._calculate_section_importance(chunk_text),
                    'headers': []
                }
                chunks.append(chunk_info)
            
            return chunks
    
    def _apply_semantic_refinement(self, header_chunks: List[Dict]) -> List[Dict]:
        """Apply semantic chunking to complex sections"""
        
        if not self.semantic_splitter:
            return header_chunks
        
        refined_chunks = []
        
        for chunk in header_chunks:
            chunk_text = chunk['text']
            
            # Apply semantic chunking only to large, complex chunks
            if (len(chunk_text) > self.chunk_size * 1.5 and 
                chunk['importance_score'] > 5.0):
                
                try:
                    semantic_splits = self.semantic_splitter.split_text(chunk_text)
                    
                    for i, semantic_chunk in enumerate(semantic_splits):
                        refined_chunk = chunk.copy()
                        refined_chunk['text'] = semantic_chunk
                        refined_chunk['chunk_type'] = 'semantic_refined'
                        refined_chunk['position'] = f"{chunk['position']}.{i}"
                        refined_chunk['importance_score'] = self._calculate_section_importance(semantic_chunk)
                        refined_chunks.append(refined_chunk)
                
                except Exception as e:
                    print(f"Semantic chunking failed for chunk {chunk['position']}: {e}")
                    refined_chunks.append(chunk)  # Keep original
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks
    
    def _process_table_for_structure_aware_rag_chunks(self, tables: List[Dict]) -> List[Dict]:
        """Process all tables into chunks"""
        all_table_chunks = []
        for table_info in tables:
            table_chunks = self._process_table_for_structure_aware_rag(table_info)
            all_table_chunks.extend(table_chunks)
        return all_table_chunks
    
    def _process_table_for_structure_aware_rag(self, table_info: Dict) -> List[Dict]:
        """
        Process tables for structure-aware RAG:
        1. Extract individual rows as retrievable chunks
        2. Preserve table structure metadata
        3. Handle multi-page table stitching
        """
        table_chunks = []
        table_text = table_info['text']
        table_id = table_info['id']

        # If the table has already been normalized to LLM-friendly text by the stitcher,
        # emit a single complete_table chunk and do not attempt to re-parse into rows.
        if table_info.get('normalized_format'):
            whole_table_chunk = {
                'text': table_text,
                'chunk_type': 'table',
                'table_id': table_id,
                'table_headers': table_info.get('headers', []),
                'importance_score': table_info.get('importance_score', 5.0),
                'position': table_info.get('position', 0),
                'is_complete_table': True
            }
            return [whole_table_chunk]
        
        # Parse table structure (non-normalized tables only)
        if '<table' in table_text.lower():
            # HTML table parsing
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_text, re.IGNORECASE | re.DOTALL)
            headers = []
            
            if rows:
                # Extract headers from first row
                header_cells = re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.IGNORECASE | re.DOTALL)
                if not header_cells:  # Fallback to td if no th
                    header_cells = re.findall(r'<td[^>]*>(.*?)</td>', rows[0], re.IGNORECASE | re.DOTALL)
                headers = [re.sub(r'<[^>]+>', '', cell).strip() for cell in header_cells]
        
        elif '|' in table_text:
            # Markdown table parsing
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            rows = [line for line in lines if '|' in line and not re.match(r'\|[\s\-:]*\|', line)]
            
            if rows:
                # Extract headers
                header_row = rows[0]
                headers = [cell.strip() for cell in header_row.split('|') if cell.strip()]
        
        else:
            # Fallback: treat entire table as single chunk
            headers = []
            rows = [table_text]
        
        # Create chunks for table structure-aware RAG
        
        # 1. Whole table chunk (for complex queries)
        whole_table_chunk = {
            'text': table_text,
            'chunk_type': 'table',
            'table_id': table_id,
            'table_headers': headers,
            'importance_score': table_info['importance_score'],
            'position': table_info.get('position', 0),  # Include position from table_info
            'is_complete_table': True
        }
        table_chunks.append(whole_table_chunk)
        
        # 2. Individual row chunks (for specific data retrieval)
        for i, row in enumerate(rows[1:] if headers else rows):  # Skip header row
            if isinstance(row, str):
                if '<td>' in row:
                    # HTML row
                    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
                    cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                else:
                    # Markdown row
                    cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                
                # Create column mapping
                column_data = {}
                for j, cell in enumerate(cells):
                    header = headers[j] if j < len(headers) else f"Column_{j+1}"
                    column_data[header] = cell
                
                # Create row-specific text
                row_text = f"Table {table_id} Row {i+1}:\n"
                for header, value in column_data.items():
                    row_text += f"{header}: {value}\n"
                
                row_chunk = {
                    'text': row_text,
                    'chunk_type': 'table_row',
                    'table_id': table_id,
                    'table_headers': headers,
                    'row_index': i,
                    'column_data': column_data,
                    'importance_score': self._calculate_row_importance(column_data),
                    'position': table_info.get('position', 0) + i + 1,  # Add position for sorting
                    'is_complete_table': False
                }
                table_chunks.append(row_chunk)
        
        return table_chunks
    
    def _calculate_row_importance(self, column_data: Dict[str, str]) -> float:
        """Calculate importance score for individual table rows"""
        score = 2.0  # Base score for being a table row
        
        # Check if row contains monetary values
        row_text = ' '.join(column_data.values())
        monetary_matches = sum(len(p.findall(row_text)) for p in self.monetary_patterns)
        score += monetary_matches * 3.0
        
        # Check for domain keywords
        if hasattr(self, 'domain_keywords') and self.domain_keywords:
            for keyword in self.domain_keywords:
                if any(keyword.lower() in str(value).lower() for value in column_data.values()):
                    score += 2.0
        
        return score
    
    # Note: Footnote extraction is intentionally omitted at this time to keep the
    # chunker focused on table vs non-table handling. Reintroduce if needed.
    
    def _calculate_section_importance(self, text: str) -> float:
        """Calculate importance score for text sections"""
        
        score = 1.0  # Base score
        
        # Monetary values (high importance)
        monetary_count = sum(len(p.findall(text)) for p in self.monetary_patterns)
        score += monetary_count * 2.0
        
        # Policy terms (medium-high importance)
        policy_count = sum(len(p.findall(text)) for p in self.policy_terms_patterns)
        score += policy_count * 1.5
        
        # Exclusions (very high importance)
        exclusion_count = sum(len(p.findall(text)) for p in self.exclusion_patterns)
        score += exclusion_count * 3.0
        
        # Cross-references (medium importance)
        cross_ref_count = sum(len(p.findall(text)) for p in self.cross_ref_patterns)
        score += cross_ref_count * 1.0
        
        # Important notes (very high importance)
        note_count = sum(len(p.findall(text)) for p in self.note_patterns)
        score += note_count * 4.0
        
        # List structures (medium importance - indicates structured content)
        list_count = sum(len(p.findall(text)) for p in self.list_patterns)
        score += list_count * 0.5
        
        # Tables (very high importance)
        if any(pattern.search(text) for pattern in self.table_patterns):
            score += 5.0
        
        # Length factor (longer sections might be more comprehensive)
        length_factor = min(len(text) / 1000, 2.0)  # Cap at 2x multiplier
        score *= (1.0 + length_factor * 0.2)
        
        return score
    
    def _post_process_chunks(self, all_chunks: List[Dict]) -> List[Tuple[str, ChunkMetadata]]:
        """Final processing and metadata creation"""
        
        final_chunks = []
        
        # Sort by importance and position (handle missing position gracefully)
        sorted_chunks = sorted(all_chunks, key=lambda x: (x['importance_score'], x.get('position', 0)), reverse=True)
        
        for i, chunk_info in enumerate(sorted_chunks):
            chunk_text = chunk_info['text']
            
            # Skip tiny chunks unless they're tables
            if len(chunk_text.strip()) < self.min_chunk_size and chunk_info['chunk_type'] != 'table':
                continue
            
            # Extract detailed metadata
            monetary_amounts = []
            for pattern in self.monetary_patterns:
                monetary_amounts.extend(pattern.findall(chunk_text))
            
            exclusion_phrases = []
            for pattern in self.exclusion_patterns:
                exclusion_phrases.extend(pattern.findall(chunk_text))
            
            policy_terms = []
            for pattern in self.policy_terms_patterns:
                policy_terms.extend(pattern.findall(chunk_text))
            
            cross_references = []
            for pattern in self.cross_ref_patterns:
                cross_references.extend(pattern.findall(chunk_text))
            
            # Extract important notes
            important_notes = []
            for pattern in self.note_patterns:
                important_notes.extend([match.group(1) for match in pattern.finditer(chunk_text)])
            
            # Detect structured lists
            has_lists = any(pattern.search(chunk_text) for pattern in self.list_patterns)
            
            # Enhance chunk type classification
            enhanced_chunk_type = chunk_info['chunk_type']
            if important_notes:
                enhanced_chunk_type += '_with_notes'
            if has_lists:
                enhanced_chunk_type += '_with_lists'
            if any(pattern.search(chunk_text) for pattern in self.table_patterns):
                enhanced_chunk_type += '_with_tables'
            
            # Detect list types present in chunk
            list_types = []
            if re.search(r'\d+\\?\.\s+', chunk_text):  # Numbered lists
                list_types.append('numbered')
            if re.search(r'[a-z][\.\)]\s+', chunk_text, re.IGNORECASE):  # Lettered lists
                list_types.append('lettered') 
            if re.search(r'[ivxlcdm]+[\.\)]\s+', chunk_text, re.IGNORECASE):  # Roman numeral lists
                list_types.append('roman')
            if re.search(r'^\\?-\s+', chunk_text, re.MULTILINE):  # Dash/bullet lists
                list_types.append('bulleted')
            
            # Determine structure element type from original chunk_info
            orig_type = chunk_info.get('chunk_type', 'text')
            is_complete_table = bool(chunk_info.get('is_complete_table'))
            if orig_type == 'table' or is_complete_table:
                element_type = 'complete_table'
            elif orig_type == 'table_row':
                element_type = 'table_row'
            elif orig_type == 'table_cell':
                element_type = 'table_cell'
            else:
                element_type = 'text'

            # Create comprehensive metadata including structure-aware fields
            metadata = ChunkMetadata(
                chunk_id=f"chunk_{i}",
                chunk_type=enhanced_chunk_type,
                importance_score=chunk_info['importance_score'],
                section_hierarchy=chunk_info.get('headers', []),
                has_tables='[TABLE_' in chunk_text or any(pattern.search(chunk_text) for pattern in self.table_patterns),
                table_count=chunk_text.count('[TABLE_') + len([p for p in self.table_patterns if p.search(chunk_text)]),
                has_monetary_values=len(monetary_amounts) > 0,
                monetary_amounts=monetary_amounts[:10],  # Limit to first 10
                has_exclusions=len(exclusion_phrases) > 0,
                exclusion_phrases=exclusion_phrases[:5],  # Limit to first 5
                policy_terms=policy_terms[:10],  # Limit to first 10
                cross_references=cross_references[:5],  # Limit to first 5
                original_headers=chunk_info.get('headers', []),
                chunk_position=i,
                source_lines=(0, 0),  # You could track this if needed
                # New fields
                has_important_notes=len(important_notes) > 0,
                important_notes=important_notes[:3],  # Limit to first 3
                has_structured_lists=len(list_types) > 0,
                list_types=list_types,
                # Structure-aware fields
                element_type=element_type,
                table_id=chunk_info.get('table_id'),
                table_headers=chunk_info.get('table_headers'),
                row_index=chunk_info.get('row_index'),
                column_data=chunk_info.get('column_data'),
                is_complete_table=is_complete_table,
                footnote_ref=None,
                cell_reference=None
            )
            
            final_chunks.append((chunk_text, metadata))
        
        return final_chunks