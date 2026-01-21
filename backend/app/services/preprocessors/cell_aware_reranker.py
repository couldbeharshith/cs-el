"""
Cell-Aware Reranker for structure-aware retrieval
"""
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from collections import defaultdict
import re
import hashlib

class CellAwareReranker:
    """Custom reranker that understands table structure and relationships"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.base_reranker = None
        
        # Structure-aware scoring weights
        self.structure_weights = {
            'complete_table': 1.0,  # Base weight
            'table_row': 0.9,       # Slightly lower than complete table
            'table_cell': 0.8,     # Individual cells lower priority
            'footnote': 1.2,       # High importance for exceptions
            'text': 0.7           # Lowest priority for general text
        }
        
        # Question-type specific boosts
        self.question_boosts = {
            'monetary': {
                'complete_table': 1.3,
                'table_row': 1.4,
                'table_cell': 1.5,  # Highest for specific values
                'footnote': 1.1,
                'text': 0.8
            },
            'condition': {
                'complete_table': 1.1,
                'table_row': 1.1,
                'table_cell': 0.9,
                'footnote': 1.6,    # Highest for conditions/exceptions
                'text': 1.0
            },
            'coverage': {
                'complete_table': 1.4,  # Tables show coverage details
                'table_row': 1.3,
                'table_cell': 1.2,
                'footnote': 1.2,
                'text': 0.9
            },
            'general': {
                'complete_table': 1.1,
                'table_row': 1.0,
                'table_cell': 0.9,
                'footnote': 1.1,
                'text': 1.0
            }
        }
    
    def classify_question_type(self, question: str) -> str:
        """Classify question to apply appropriate boosts"""
        question_lower = question.lower()
        
        # Monetary questions
        monetary_keywords = [
            'amount', 'cost', 'premium', 'price', 'limit', 'maximum', 'minimum',
            'deductible', 'co-pay', 'benefit', '$', '₹', 'rs', 'inr', 'rupee', '%', 'percentage',
            'lakh', 'crore'
        ]
        if any(keyword in question_lower for keyword in monetary_keywords):
            return 'monetary'
        
        # Condition/exception questions
        condition_keywords = [
            'condition', 'requirement', 'restriction', 'limitation', 'exclusion',
            'exception', 'not covered', 'exclude', 'except', 'unless', 'provided'
        ]
        if any(keyword in question_lower for keyword in condition_keywords):
            return 'condition'
        
        # Coverage questions
        coverage_keywords = [
            'cover', 'covered', 'coverage', 'include', 'included', 'benefit',
            'eligible', 'qualify', 'scope', 'apply', 'applicable'
        ]
        if any(keyword in question_lower for keyword in coverage_keywords):
            return 'coverage'
        
        return 'general'
    
    def calculate_structure_relationships(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze relationships between structured elements"""
        relationships = {
            'table_families': defaultdict(list),  # Group related table elements
            'parent_child_map': {},
            'sibling_groups': defaultdict(list)
        }
        
        for doc in documents:
            metadata = doc.metadata
            element_type = metadata.get('element_type', 'text')
            
            # Group table family members
            if element_type in ['complete_table', 'table_row', 'table_cell']:
                table_id = metadata.get('table_id')
                if table_id:
                    relationships['table_families'][table_id].append(doc)
            
            # Map parent-child relationships
            parent_id = metadata.get('parent_id')
            if parent_id:
                relationships['parent_child_map'][self._get_doc_key(doc)] = parent_id
            
            # Group siblings (same parent)
            if parent_id:
                relationships['sibling_groups'][parent_id].append(doc)
        
        return relationships
    
    def apply_relationship_boosts(
        self, 
        doc: Document, 
        base_score: float, 
        relationships: Dict[str, Any],
        question: str
    ) -> float:
        """Apply boosts based on structural relationships"""
        
        boosted_score = base_score
        metadata = doc.metadata
        element_type = metadata.get('element_type', 'text')
        
        # Table family coherence boost
        if element_type in ['table_row', 'table_cell']:
            table_id = metadata.get('table_id')
            if table_id and table_id in relationships['table_families']:
                family_size = len(relationships['table_families'][table_id])
                if family_size > 3:  # Substantial table
                    boosted_score *= 1.1
                
                # Check if multiple family members are relevant
                family_docs = relationships['table_families'][table_id]
                relevant_family_count = sum(1 for fam_doc in family_docs 
                                          if self._is_potentially_relevant(fam_doc, question))
                
                if relevant_family_count > 1:
                    boosted_score *= 1.15  # Multiple related elements boost
        
        # Parent-child relationship boost
        doc_key = self._get_doc_key(doc)
        if doc_key in relationships['parent_child_map']:
            parent_id = relationships['parent_child_map'][doc_key]
            sibling_docs = relationships['sibling_groups'].get(parent_id, [])
            
            if len(sibling_docs) > 1:
                # Check if siblings are also relevant
                relevant_siblings = sum(1 for sibling in sibling_docs 
                                      if self._is_potentially_relevant(sibling, question))
                
                if relevant_siblings > 1:
                    boosted_score *= 1.1  # Sibling relevance boost
        
        # Monetary specificity boost: adapt to our chunker metadata
        # Our chunker sets has_monetary_values (plural) at chunk level; percentages can be detected via content.
        if self.classify_question_type(question) == 'monetary':
            has_money_meta = bool(metadata.get('has_monetary_values'))
            has_percent = bool(re.search(r'\b\d+\s*%\b', doc.page_content))
            if element_type == 'table_cell' and (has_money_meta or has_percent):
                boosted_score *= 1.2
            elif element_type in ('table_row', 'complete_table') and (has_money_meta or has_percent):
                boosted_score *= 1.05
        
        return boosted_score
    
    def _is_potentially_relevant(self, doc: Document, question: str) -> bool:
        """Quick relevance check for relationship analysis"""
        question_words = set(question.lower().split())
        doc_words = set(doc.page_content.lower().split())
        
        # Simple word overlap check
        overlap = len(question_words.intersection(doc_words))
        return overlap >= 2
    
    def rerank_documents(
        self, 
        documents: List[Document], 
        question: str, 
        top_k: int
    ) -> List[Document]:
        """
        Cell-aware reranking that considers structure and relationships
        """
        
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        # Step 1: Get base relevance scores
        base_scores = [self._simple_similarity(question, doc.page_content) 
                       for doc in documents]
        
        # Step 2: Classify question type
        question_type = self.classify_question_type(question)
        
        # Step 3: Calculate structure relationships
        relationships = self.calculate_structure_relationships(documents)
        
        # Step 4: Apply structure-aware scoring
        final_scores = []
        for i, (doc, base_score) in enumerate(zip(documents, base_scores)):
            metadata = doc.metadata
            element_type = metadata.get('element_type', 'text')
            
            # Apply base structure weight
            structure_weight = self.structure_weights.get(element_type, 1.0)
            
            # Apply question-type specific boost
            question_boost = self.question_boosts[question_type].get(element_type, 1.0)
            
            # Calculate structure-weighted score
            structure_score = base_score * structure_weight * question_boost
            
            # Apply relationship boosts
            final_score = self.apply_relationship_boosts(
                doc, structure_score, relationships, question
            )
            
            # Add diversity penalty for similar elements
            diversity_penalty = self._calculate_diversity_penalty(doc, documents[:i])
            final_score *= (1 - diversity_penalty)
            
            final_scores.append((doc, base_score, final_score, {
                'element_type': element_type,
                'structure_weight': structure_weight,
                'question_boost': question_boost,
                'diversity_penalty': diversity_penalty
            }))
        
        # Step 5: Sort by final scores and return top_k
        final_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Ensure we maintain some diversity in element types
        reranked_docs = self._apply_diversity_filter(final_scores[:top_k * 2], top_k)
        
        return reranked_docs
    
    def _simple_similarity(self, query: str, text: str) -> float:
        """Fallback similarity calculation"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_diversity_penalty(self, current_doc: Document, previous_docs: List[Document]) -> float:
        """Calculate penalty for similar documents to promote diversity"""
        if not previous_docs:
            return 0.0
        
        current_type = current_doc.metadata.get('element_type', 'text')
        current_table = current_doc.metadata.get('table_id', 'none')
        
        penalty = 0.0
        for prev_doc in previous_docs:
            prev_type = prev_doc.metadata.get('element_type', 'text')
            prev_table = prev_doc.metadata.get('table_id', 'none')
            
            # Penalty for same element type (but allow some duplication)
            if current_type == prev_type:
                penalty += 0.02
            
            # Higher penalty for same table elements
            if current_table != 'none' and current_table == prev_table:
                penalty += 0.05
            
            # Content similarity penalty
            content_similarity = self._simple_similarity(
                current_doc.page_content[:200], 
                prev_doc.page_content[:200]
            )
            penalty += content_similarity * 0.1
        
        return min(penalty, 0.3)  # Cap penalty at 30%
    
    def _apply_diversity_filter(self, scored_docs: List[Tuple], target_count: int) -> List[Document]:
        """Apply final diversity filtering to maintain element type balance"""
        
        # Group by element type
        type_groups = defaultdict(list)
        for doc_tuple in scored_docs:
            doc = doc_tuple[0]
            element_type = doc.metadata.get('element_type', 'text')
            type_groups[element_type].append(doc_tuple)
        
        # Ensure representation from different types
        final_docs = []
        remaining_slots = target_count
        
        # Priority order for element types
        type_priority = ['footnote', 'complete_table', 'table_row', 'table_cell', 'text']
        
        # First pass: ensure at least one from each important type
        for element_type in type_priority:
            if element_type in type_groups and remaining_slots > 0:
                if element_type in ['footnote', 'complete_table']:
                    # Take up to 2 from high-priority types
                    take_count = min(2, len(type_groups[element_type]), remaining_slots)
                else:
                    # Take 1 from other types
                    take_count = min(1, len(type_groups[element_type]), remaining_slots)
                
                # Add the best documents of this type
                for doc_tuple in type_groups[element_type][:take_count]:
                    final_docs.append(doc_tuple[0])
                    remaining_slots -= 1
        
        # Second pass: fill remaining slots with highest scoring documents
        all_remaining = []
        for element_type, docs in type_groups.items():
            # Skip documents already added
            already_added_keys = {self._get_doc_key(d) for d in final_docs}
            for doc_tuple in docs:
                if self._get_doc_key(doc_tuple[0]) not in already_added_keys:
                    all_remaining.append(doc_tuple)
        
        # Sort remaining by score and take what we need
        all_remaining.sort(key=lambda x: x[2], reverse=True)
        for doc_tuple in all_remaining[:remaining_slots]:
            final_docs.append(doc_tuple[0])
        
        return final_docs

    def _get_doc_key(self, doc: Document) -> str:
        """Create a stable key for a document to use in dicts/sets.

        Prefer metadata IDs if present; otherwise use a content-based hash.
        """
        md = getattr(doc, 'metadata', {}) or {}
        for k in ('chunk_id', 'id', 'doc_id', 'node_id', 'uuid', 'element_id'):
            v = md.get(k)
            if v is not None:
                return f"{k}:{v}"

        # Fallback: hash content + some metadata to keep it reasonably stable
        basis = (
            (doc.page_content[:512] if getattr(doc, 'page_content', None) else ''),
            str(md.get('element_type', '')),
            str(md.get('table_id', '')),
            str(md.get('parent_id', '')),
        )
        h = hashlib.sha1("|".join(basis).encode('utf-8', 'ignore')).hexdigest()
        return f"sha1:{h}"