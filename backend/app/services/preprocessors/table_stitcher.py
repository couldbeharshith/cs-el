"""
Multi-page table stitching and LLM-friendly normalization utility
Handles Azure OCR output with <!-- PageBreak --> detection
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class ParsedTable:
    """Structured representation of a table"""
    headers: List[str]
    rows: List[List[str]]
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    row_labels: List[str] = None  # For first column (e.g., "i.", "ii.")


class TableStitcher:
    """
    Advanced table processor that:
    1. Detects PageBreak-separated table continuations
    2. Stitches multi-page tables together
    3. Converts tables to LLM-friendly text format
    """
    
    def __init__(self):
        # Patterns for detecting table structure
        self.html_header_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.IGNORECASE | re.DOTALL)
        self.html_row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
        self.html_cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)
        
        # Azure OCR page break pattern
        self.pagebreak_pattern = re.compile(r'<!--\s*PageBreak\s*-->', re.IGNORECASE)
        
        # Row label detector (roman/digit/alpha with optional punctuation/paren)
        self.row_label_pattern = re.compile(
            r'^\s*\(?\s*(?:([ivxlcdm]{1,8})|([A-Za-z])|(\d{1,3}))\s*\)?\s*(?:\(\s*[a-zA-Z]\s*\))?\s*[\.)]?\s*$',
            re.IGNORECASE
        )
    
    def stitch_tables(self, tables: List[Dict], original_text: str = "") -> List[Dict]:
        """
        Main entry point: detect and stitch PageBreak-separated tables
        
        Args:
            tables: List of table dictionaries with metadata
            original_text: Full document text (to check for PageBreaks)
        
        Returns:
            List of processed tables (stitched + normalized)
        """
        if len(tables) < 2:
            # Single table - just normalize it
            return [self._normalize_single_table(tables[0])] if tables else []

        # Sort by source order; prefer start_pos for reliable ordering, fallback to position
        sorted_tables = sorted(tables, key=lambda t: t.get('start_pos', t.get('position', 0)))
        
        # Detect PageBreak-separated table groups
        table_groups = self._group_pagebreak_tables(sorted_tables, original_text)
        
        # Process each group
        processed_tables = []
        for group in table_groups:
            if len(group) > 1:
                # Multi-page table - stitch and normalize
                stitched = self._stitch_and_normalize_group(group)
                processed_tables.append(stitched)
            else:
                # Standalone table - just normalize
                normalized = self._normalize_single_table(group[0])
                processed_tables.append(normalized)
        
        return processed_tables
    
    def _group_pagebreak_tables(self, tables: List[Dict], original_text: str) -> List[List[Dict]]:
        """
        Group consecutive tables that are separated by PageBreak tags OR continuation tables
        with no headers that follow a table with headers.
        
        This handles Azure OCR output where multi-page tables may have:
        1. PageBreak tags between fragments
        2. Text content between fragments (notes, disclaimers)
        3. All fragments using <th> tags for data rows (not just headers)
        """
        if not tables:
            return []
        
        groups = []
        current_group = [tables[0]]
        first_table_has_headers = bool(self._extract_column_headers(tables[0]['text']))
        
        for i in range(1, len(tables)):
            prev_table = tables[i - 1]
            curr_table = tables[i]
            
            # Check if there's a PageBreak between these tables
            has_pagebreak = self._has_pagebreak_between(
                prev_table, curr_table, original_text
            )
            
            # Check if tables have similar structure
            similar_structure = self._have_similar_structure(prev_table, curr_table)
            
            # Check if current table is a continuation table (no headers)
            curr_has_headers = bool(self._extract_column_headers(curr_table['text']))
            is_continuation = not curr_has_headers and similar_structure
            
            # Group if: (PageBreak + similar structure) OR (continuation table with similar structure)
            if (has_pagebreak and similar_structure) or (first_table_has_headers and is_continuation):
                current_group.append(curr_table)
            else:
                # Different table - start new group
                groups.append(current_group)
                current_group = [curr_table]
                first_table_has_headers = curr_has_headers
        
        # Add last group
        groups.append(current_group)
        
        return groups
    
    def _has_pagebreak_between(self, table1: Dict, table2: Dict, original_text: str) -> bool:
        """Check if there's a PageBreak comment between two tables"""
        if not original_text:
            # Fallback: check metadata if available
            return table1.get('has_pagebreak_after', False)
        
        # Get positions
        end_pos1 = table1.get('end_pos', 0)
        start_pos2 = table2.get('start_pos', len(original_text))
        
        # Extract text between tables
        between_text = original_text[end_pos1:start_pos2]
        
        # Check for PageBreak
        return bool(self.pagebreak_pattern.search(between_text))
    
    def _have_similar_structure(self, table1: Dict, table2: Dict) -> bool:
        """
        Check if two tables have similar column structure.
        Handles continuation tables that may have no headers (only data rows after PageBreak).
        """
        headers1 = self._extract_column_headers(table1['text'])
        headers2 = self._extract_column_headers(table2['text'])
        
        # Special case: One or both tables have no headers (continuation tables with only data rows)
        if not headers1 or not headers2:
            # Check if both have similar column counts by inspecting first data row
            cols1 = self._count_table_columns(table1['text'])
            cols2 = self._count_table_columns(table2['text'])
            col_diff = abs(cols1 - cols2)
            if col_diff <= 1:
                return True
            else:
                return False
        
        # Allow ±1 column difference for rowspan/colspan scenarios
        col_diff = abs(len(headers1) - len(headers2))
        if col_diff > 1:
            return False
        
        # If column counts match exactly, check header similarity
        if col_diff == 0:
            # Check header similarity (case-insensitive)
            headers1_clean = [h.lower().strip() for h in headers1]
            headers2_clean = [h.lower().strip() for h in headers2]
            
            # Calculate similarity
            matches = sum(1 for h1, h2 in zip(headers1_clean, headers2_clean) 
                         if h1 == h2 or self._headers_are_similar(h1, h2))
            
            similarity = matches / len(headers1)
            
            # Require 75%+ similarity
            return similarity >= 0.75
        else:
            # Column count differs by 1 - likely rowspan/colspan continuation
            return True  # Accept as potential continuation
    
    def _count_table_columns(self, table_text: str) -> int:
        """Count columns more robustly using spans across first few rows."""
        raw_rows = self._extract_rows_with_spans(table_text)
        if not raw_rows:
            return 0
        consider = raw_rows[:5]
        return max(sum(cell['colspan'] for cell in row) for row in consider) if consider else 0
    
    def _headers_are_similar(self, h1: str, h2: str) -> bool:
        """Check header similarity using token overlap and small synonym buckets."""
        synonyms = {
            'limits of covered expenses': {'covered', 'expenses', 'expense', 'limits', 'covered expenses'},
            'sum insured': {'sum insured', 'insured amount', 'sum assured'},
            'silver': {'silver', 'silver plan', 'silver variant', 'silver option'},
            'gold': {'gold', 'gold plan', 'gold plus', 'gold variant', 'gold option'},
            'diamond': {'diamond', 'diamond plan'},
            'platinum': {'platinum', 'platinum plan', 'platinum plus', 'platinum variant'},
            'bronze': {'bronze', 'bronze plan'},
            'elite': {'elite', 'elite plan'},
            'supreme': {'supreme', 'supreme plan'},
            'premium': {'premium', 'premium plan'},
            'ultimate': {'ultimate', 'ultimate plan'},
            'classic': {'classic', 'classic plan'},
            'standard': {'standard plan'},
            'basic': {'basic plan'},
            'prime': {'prime', 'prime plan'},
            'deluxe': {'deluxe', 'deluxe plan'},
        }
        def tokens(s: str) -> Set[str]:
            s = re.sub(r'[\W_]+', ' ', s.lower()).strip()
            return {t for t in s.split() if t}
        t1, t2 = tokens(h1), tokens(h2)
        if not t1 or not t2:
            return False
        # Direct substring fast-path
        if ' '.join(sorted(t1)) in ' '.join(sorted(t2)) or ' '.join(sorted(t2)) in ' '.join(sorted(t1)):
            return True
        # Synonym buckets
        for bucket in synonyms.values():
            if (any(b in h1.lower() for b in bucket) and any(b in h2.lower() for b in bucket)):
                return True
        # Jaccard
        inter = len(t1 & t2)
        union = len(t1 | t2)
        return (inter / union) >= 0.5 if union else False
    
    def _extract_column_headers(self, table_text: str) -> List[str]:
        """
        Extract column headers from table HTML, handling continuation tables with <th> data rows.
        Skips rows that start with row labels (i., ii., A., etc.) which are data rows, not headers.
        """
        headers = []
        
        if '<table' in table_text.lower():
            # Extract all rows to check for continuation table pattern
            row_matches = self.html_row_pattern.findall(table_text)
            
            for row_html in row_matches:
                # Try to find <th> tags
                th_cells = self.html_header_pattern.findall(row_html)
                
                if th_cells:
                    # Check if first cell is a row label (i., ii., A., etc.)
                    first_cell_clean = self._clean_html(th_cells[0]).strip()
                    if self._is_row_label(first_cell_clean):
                        # Likely a data row mis-tagged as header by OCR
                        continue
                    
                    # This is a real header row
                    headers = [self._clean_html(cell) for cell in th_cells]
                    break  # Use first valid header row
            
            # Fallback: use first row with <td> tags if no headers found
            if not headers:
                first_row_match = self.html_row_pattern.search(table_text)
                if first_row_match:
                    first_row = first_row_match.group(1)
                    td_cells = self.html_cell_pattern.findall(first_row)
                    headers = [self._clean_html(cell) for cell in td_cells]
        
        return headers
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags and clean text"""
        clean = re.sub(r'<[^>]+>', '', html_text)
        clean = re.sub(r'&amp;', '&', clean)
        clean = re.sub(r'&nbsp;', ' ', clean)
        return clean.strip()
    
    def _stitch_and_normalize_group(self, group: List[Dict]) -> Dict:
        """
        Stitch multiple tables together and convert to LLM-friendly format
        """
        if len(group) == 1:
            return self._normalize_single_table(group[0])
        
        # Parse all tables in group
        parsed_tables = [self._parse_html_table(t['text']) for t in group]
        
        # Merge tables (use headers from first table)
        merged_table = self._merge_parsed_tables(parsed_tables)
        
        # Convert to LLM-friendly text
        llm_friendly_text = self._convert_to_llm_friendly_format(merged_table)
        
        # Update metadata
        base_table = group[0].copy()
        base_table['text'] = llm_friendly_text
        base_table['id'] = f"stitched_{base_table['id']}"
        base_table['is_multipage'] = True
        base_table['page_count'] = len(group)
        base_table['importance_score'] = base_table.get('importance_score', 5.0) + 2.0
        base_table['headers'] = merged_table.headers
        base_table['normalized_format'] = True
        
        return base_table
    
    def _normalize_single_table(self, table: Dict) -> Dict:
        """Normalize a single table to LLM-friendly format"""
        parsed = self._parse_html_table(table['text'])
        llm_friendly_text = self._convert_to_llm_friendly_format(parsed)
        
        table = table.copy()
        table['text'] = llm_friendly_text
        table['headers'] = parsed.headers
        table['normalized_format'] = True
        
        return table
    
    def _extract_rows_with_spans(self, table_html: str):
        """Extract rows with rowspan/colspan information"""
        rows = []
        row_matches = self.html_row_pattern.findall(table_html)
        
        for row_html in row_matches:
            cells = []
            
            # Find all cells (th or td) with their attributes
            cell_pattern = re.compile(r'<(th|td)([^>]*)>(.*?)</\1>', re.DOTALL | re.IGNORECASE)
            for match in cell_pattern.finditer(row_html):
                tag_type = match.group(1).lower()
                attributes = match.group(2)
                content = match.group(3)
                
                # Extract rowspan and colspan
                rowspan = 1
                colspan = 1
                rowspan_match = re.search(r'rowspan\s*=\s*["\']?(\d+)["\']?', attributes, re.IGNORECASE)
                colspan_match = re.search(r'colspan\s*=\s*["\']?(\d+)["\']?', attributes, re.IGNORECASE)
                
                if rowspan_match:
                    rowspan = int(rowspan_match.group(1))
                if colspan_match:
                    colspan = int(colspan_match.group(1))
                
                cells.append({
                    'content': self._clean_html(content),
                    'rowspan': rowspan,
                    'colspan': colspan,
                    'is_header': tag_type == 'th'
                })
            
            if cells:
                rows.append(cells)
        
        return rows
    
    def _resolve_spans(self, raw_rows):
        """Resolve rowspan/colspan to create a proper grid, merging fragmented cells"""
        if not raw_rows:
            return []
        
        # Build a grid to track cell placement
        max_cols = max(sum(cell['colspan'] for cell in row) for row in raw_rows) if raw_rows else 0
        grid = []
        pending_spans = []  # Track cells that span multiple rows: (row_idx, col_idx, content, remaining_rows)
        
        for row_idx, row_cells in enumerate(raw_rows):
            grid_row = [None] * max_cols
            col_idx = 0
            cell_input_idx = 0
            
            while col_idx < max_cols:
                # Check if there's a pending rowspan cell for this position
                pending_cell = None
                for i, (span_row, span_col, span_content, remaining) in enumerate(pending_spans):
                    if span_col == col_idx:
                        pending_cell = (i, span_content, remaining)
                        break
                
                if pending_cell:
                    # Use the spanned cell content
                    idx, content, remaining = pending_cell
                    grid_row[col_idx] = content
                    
                    # Update or remove from pending
                    if remaining > 1:
                        pending_spans[idx] = (row_idx + 1, col_idx, content, remaining - 1)
                    else:
                        pending_spans.pop(idx)
                    
                    col_idx += 1
                elif cell_input_idx < len(row_cells):
                    # Place new cell
                    cell = row_cells[cell_input_idx]
                    content = cell['content']
                    
                    # Place the cell
                    grid_row[col_idx] = content
                    
                    # Handle rowspan - add to pending
                    if cell['rowspan'] > 1:
                        pending_spans.append((row_idx + 1, col_idx, content, cell['rowspan'] - 1))
                    
                    # Handle colspan
                    for span_offset in range(1, cell['colspan']):
                        if col_idx + span_offset < max_cols:
                            grid_row[col_idx + span_offset] = content
                    
                    col_idx += cell['colspan']
                    cell_input_idx += 1
                else:
                    # No more cells in this row
                    col_idx += 1
            
            grid.append(grid_row)
        
        return grid
    
    def _parse_html_table(self, table_html: str) -> ParsedTable:
        """Parse HTML table into structured format with rowspan/colspan handling"""
        headers = []
        rows = []
        section_title = None
        row_labels = []
        
        # First pass: extract raw rows with rowspan/colspan info
        raw_rows = self._extract_rows_with_spans(table_html)
        
        # Second pass: resolve rowspan/colspan to create final grid
        resolved_rows = self._resolve_spans(raw_rows)
        
        # Process resolved rows
        for i, grid_row in enumerate(resolved_rows):
            # Filter out None cells
            clean_row = [cell if cell is not None else "" for cell in grid_row]
            
            # Check for section header row (single content spanning all columns)
            if len(set(clean_row)) == 1 and clean_row[0] and not section_title:
                section_title = clean_row[0]
                continue
            
            # Check if this row should be headers
            if not headers and i < len(raw_rows):
                # Check if original row had <th> tags
                has_th_tags = any(cell['is_header'] for cell in raw_rows[i])
                
                # Check if first cell is a row label
                first_cell_clean = clean_row[0].strip() if clean_row else ""
                is_data_row_with_label = self._is_row_label(first_cell_clean)
                
                # Check if it's a continuation row (empty first cell but has content elsewhere)
                first_cell_empty = not clean_row[0].strip() if clean_row else False
                has_content = any(cell.strip() for cell in clean_row[1:]) if len(clean_row) > 1 else False
                is_continuation_row = first_cell_empty and has_content
                
                if has_th_tags and not is_data_row_with_label and not is_continuation_row:
                    headers = clean_row
                    continue
            
            # Check if first cell is a row label
            if clean_row:
                first_cell = clean_row[0].strip()
                if self._is_row_label(first_cell):
                    row_labels.append(first_cell)
            
            # Add as data row
            rows.append(clean_row)
        
        # If no headers found, use first row (unless it's a data row with a row label OR continuation row)
        if not headers and rows:
            first_row = rows[0]
            is_data_row_with_label = first_row and self._is_row_label(first_row[0].strip())
            
            # Check if it's a continuation row (empty first cell but has content elsewhere)
            first_cell_empty = not first_row[0].strip() if first_row else False
            has_content = any(cell.strip() for cell in first_row[1:]) if len(first_row) > 1 else False
            is_continuation_row = first_cell_empty and has_content
            
            if not is_data_row_with_label and not is_continuation_row:
                # First row looks like headers, use it
                headers = rows[0]
                rows = rows[1:]
            # else: First row is a data row or continuation row, keep it in rows
        
        # Post-process: Merge rows with duplicate labels/descriptions (from rowspan fragments)
        rows = self._merge_duplicate_rows(rows)
        
        return ParsedTable(
            headers=headers,
            rows=rows,
            section_title=section_title,
            row_labels=row_labels if row_labels else None
        )
    
    def _merge_duplicate_rows(self, rows):
        """Merge rows where first column (label) is identical (rowspan fragments)"""
        if not rows or len(rows) < 2:
            return rows
        
        merged = []
        i = 0
        
        while i < len(rows):
            current_row = rows[i]
            
            # Look ahead for rows with same label (first column)
            if len(current_row) >= 1:
                label = current_row[0].strip() if current_row[0] else ""
                
                # Collect all consecutive rows with same label
                matching_rows = [current_row]
                j = i + 1
                
                while j < len(rows):
                    next_row = rows[j]
                    if len(next_row) >= 1:
                        next_label = next_row[0].strip() if next_row[0] else ""
                        
                        # Match if: same label OR (current label exists and next is empty)
                        if next_label == label:
                            matching_rows.append(next_row)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # Merge all columns if multiple rows found with same label
                if len(matching_rows) > 1:
                    merged_row = current_row[:]
                    
                    # Merge each column
                    for col_idx in range(1, len(merged_row)):
                        # Collect all unique non-empty values from matching rows
                        seen = set()
                        values = []
                        for row in matching_rows:
                            if col_idx < len(row) and row[col_idx]:
                                val = row[col_idx].strip()
                                if val and val not in seen:
                                    values.append(val)
                                    seen.add(val)
                        
                        # Merge with space
                        if values:
                            merged_row[col_idx] = " ".join(values)
                    
                    merged.append(merged_row)
                    i = j
                else:
                    merged.append(current_row)
                    i += 1
            else:
                merged.append(current_row)
                i += 1
        
        return merged
    
    def _concat_cell(self, left: str, right: str) -> str:
        """Safely concatenate two cell texts with a single space and basic normalization."""
        l = (left or '').strip()
        r = (right or '').strip()
        if not l:
            return r
        if not r:
            return l
        # Simple join with space and collapse any double spaces later in pipeline
        return f"{l} {r}".strip()

    def _merge_parsed_tables(self, tables: List[ParsedTable]) -> ParsedTable:
        """Merge multiple parsed tables (from different pages)"""
        if not tables:
            return ParsedTable(headers=[], rows=[])
        
        # Use headers from first table
        base_table = tables[0]
        all_rows = base_table.rows.copy()
        all_row_labels = base_table.row_labels.copy() if base_table.row_labels else []
        
        # Append rows from continuation tables, merging split cells across fragments
        for idx, table in enumerate(tables[1:], 1):
            labels_extended = False
            if table.rows and all_rows:
                # Check if first row of current fragment continues last row of previous fragment
                first_row = table.rows[0]
                last_row = all_rows[-1]

                if base_table.headers and first_row and last_row:
                    desc_idx = self._identify_description_column(base_table.headers)
                    plan_map = self._identify_plan_columns(base_table.headers)
                    plan_idxs = list(plan_map.values()) if plan_map else [
                        i for i in range(len(base_table.headers)) if i not in (0, desc_idx if desc_idx is not None else -1)
                    ]
                    # Determine if description in first row is empty/short and any plan columns have content
                    first_desc = (
                        first_row[desc_idx].strip() if desc_idx is not None and desc_idx < len(first_row) else ""
                    )
                    plan_nonempty_count = sum(1 for i in plan_idxs if i < len(first_row) and bool(first_row[i].strip()))
                    # Tighten: require multiple plan cells and very short/empty desc
                    looks_like_suffix = (plan_nonempty_count >= 2) and (not first_desc or len(first_desc) <= 2)

                    if looks_like_suffix:
                        # Merge plan suffixes into the previous row's plan cells
                        for i in plan_idxs:
                            if i < len(first_row) and first_row[i].strip():
                                if i < len(last_row) and last_row[i].strip():
                                    last_row[i] = self._concat_cell(last_row[i], first_row[i])
                                elif i < len(last_row):
                                    last_row[i] = first_row[i].strip()
                        # Skip the first row (it was just a suffix carrier)
                        all_rows.extend(table.rows[1:])
                        if table.row_labels:
                            all_row_labels.extend(table.row_labels[1:])
                            labels_extended = True
                        continue
                
                # Check if first row has empty/rowspan-only first columns followed by content
                if len(first_row) >= 2 and len(last_row) >= 2:
                    # Check if first column is empty or a label equal to previous row (row fragments)
                    first_col_text = first_row[0].strip() if first_row else ""
                    last_label = (last_row[0].strip() if last_row and last_row[0] else "")
                    first_col_empty = not first_col_text
                    first_col_same_label = (self._is_row_label(first_col_text) and last_label == first_col_text)
                    # Use description column awareness when available
                    desc_idx = self._identify_description_column(base_table.headers)
                    if desc_idx is not None and desc_idx < len(first_row):
                        desc_empty = first_row[desc_idx].strip() == ''
                    else:
                        desc_empty = False
                    has_content_other_cols = any(
                        (ci != desc_idx and ci < len(first_row) and first_row[ci].strip())
                        for ci in range(len(first_row))
                    )
                    # Merge only for true continuation (empty/label) and other columns having content
                    should_merge = (first_col_empty or first_col_same_label or desc_empty) and has_content_other_cols
                    
                    if should_merge:
                        # Merge this row's cells with last row's cells
                        for col_idx in range(len(first_row)):
                            if col_idx < len(last_row) and first_row[col_idx].strip():
                                # Append content to previous row's cell
                                if last_row[col_idx].strip():
                                    last_row[col_idx] = self._concat_cell(last_row[col_idx], first_row[col_idx])
                                else:
                                    last_row[col_idx] = first_row[col_idx].strip()
                        
                        # Skip first row since we merged it
                        all_rows.extend(table.rows[1:])
                        if table.row_labels:
                            all_row_labels.extend(table.row_labels[1:])
                            labels_extended = True
                    else:
                        # Normal append
                        all_rows.extend(table.rows)
                else:
                    all_rows.extend(table.rows)
            else:
                all_rows.extend(table.rows)
            
            if table.row_labels and not labels_extended:
                all_row_labels.extend(table.row_labels)
        
        return ParsedTable(
            headers=base_table.headers,
            rows=all_rows,
            section_title=base_table.section_title,
            row_labels=all_row_labels if all_row_labels else None
        )
    
    def _convert_to_llm_friendly_format(self, parsed: ParsedTable) -> str:
        """
        Convert parsed table to explicit LLM-friendly text format
        """
        output_lines = []
        
        # Add section title if present
        if parsed.section_title:
            output_lines.append(f"Section: {parsed.section_title}")
            output_lines.append("")
        
        # Identify plan columns (SILVER, GOLD, DIAMOND, PLATINUM)
        plan_columns = self._identify_plan_columns(parsed.headers)
        
        # Identify benefit/expense description column
        description_col_idx = self._identify_description_column(parsed.headers)
        
        # Process each row
        for i, row in enumerate(parsed.rows):
            if len(row) <= 1:
                continue  # Skip empty/malformed rows
            
            # Check if first cell is a row label (i., ii., A., etc.)
            row_label = None
            first_cell = row[0].strip() if row else ""
            if self._is_row_label(first_cell):
                row_label = first_cell
            
            # Get benefit description (skip first column if it's a row label)
            start_col = 1 if row_label else 0
            if description_col_idx is not None and description_col_idx < len(row):
                description = row[description_col_idx]
            else:
                # Fallback: use second column if first is row label, otherwise first
                description = row[start_col] if len(row) > start_col else row[0]
            
            # Skip if description is empty or just whitespace
            if not description or description.strip() in ['', '-', 'N/A']:
                continue
            
            # Add row label if available
            if row_label:
                output_lines.append(f"{row_label} {description}:")
            else:
                output_lines.append(f"{description}:")
            
            # Add plan-specific values (skip row label column)
            if plan_columns:
                for plan_name, col_idx in plan_columns.items():
                    if col_idx < len(row):
                        value = row[col_idx]
                        if value and value.strip():
                            output_lines.append(f"  - {plan_name}: {value}")
            else:
                # No plan columns identified - just list all values
                for j, cell in enumerate(row):
                    if j == 0 and row_label:  # Skip row label column
                        continue
                    if j != description_col_idx and cell.strip():
                        header = parsed.headers[j] if j < len(parsed.headers) else f"Column {j+1}"
                        output_lines.append(f"  - {header}: {cell}")
            
            output_lines.append("")  # Blank line between rows
        
        return "\n".join(output_lines)
    
    def _identify_plan_columns(self, headers: List[str]) -> Dict[str, int]:
        """Identify plan columns; fallback to header names when unknown plans."""
        plan_names = ['SILVER', 'GOLD', 'DIAMOND', 'PLATINUM']
        plan_columns: Dict[str, int] = {}
        for i, header in enumerate(headers):
            header_upper = header.upper().strip()
            for plan in plan_names:
                if plan in header_upper:
                    plan_columns[plan] = i
                    break
        if plan_columns:
            return plan_columns
        # Fallback: treat columns beyond description as plan-like
        desc_idx = self._identify_description_column(headers)
        for i, header in enumerate(headers):
            if i == 0 or i == desc_idx:
                continue
            name = header.strip() or f"Column {i+1}"
            plan_columns[name.upper()] = i
        return plan_columns
    
    def _identify_description_column(self, headers: List[str]) -> Optional[int]:
        """Identify the description column with broader keywords and heuristics."""
        description_keywords = [
            'expense', 'benefit', 'coverage', 'covered', 'description', 'item', 'particulars',
            'feature', 'features', 'service', 'services', 'treatment', 'procedure', 'head', 'details', 'clause'
        ]
        for i, header in enumerate(headers):
            header_lower = header.lower().strip()
            if any(keyword in header_lower for keyword in description_keywords):
                return i
        # Fallback: first non-numeric-looking header after first column
        def is_numeric_like(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            digits = sum(ch.isdigit() for ch in s)
            letters = sum(ch.isalpha() for ch in s)
            return digits > letters
        for i in range(1, len(headers)):
            if not is_numeric_like(headers[i]):
                return i
        return 1 if len(headers) > 1 else None

    def _is_row_label(self, text: str) -> bool:
        """Return True if text looks like a row label (roman/digit/alpha with optional suffixes)."""
        if not text:
            return False
        t = text.strip()
        # Avoid misclassifying long tokens
        if len(t) > 10:
            return False
        return bool(self.row_label_pattern.match(t))