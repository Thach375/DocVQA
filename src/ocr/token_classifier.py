"""
Token Classifier: Phân loại OCR tokens theo semantic type.

Phân loại trước khi grouping để:
- Detect form fields (key: value)
- Detect dates
- Detect numbers, codes, etc.
"""

import re
from typing import List, Dict, Any


class TokenClassifier:
    """Classifier để gán nhãn semantic cho OCR tokens."""
    
    def __init__(self):
        # Date patterns (nhiều format)
        self.date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # DD/MM/YYYY, DD-MM-YY
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',    # YYYY-MM-DD
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # DD Month YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        ]
        
        # Form field patterns (key: value)
        self.form_key_patterns = [
            r'^[A-Z][A-Za-z\s]{1,30}:$',  # "Name:", "Date:", etc.
            r'^[A-Z][A-Za-z\s]{1,30}\s*:',  # With optional space before colon
        ]
        
        # Common form field keywords
        self.form_keywords = [
            'name', 'address', 'phone', 'email', 'date', 'signature',
            'company', 'title', 'department', 'position', 'age', 'gender',
            'id', 'number', 'code', 'reference', 'account', 'customer',
            'total', 'amount', 'quantity', 'price', 'description'
        ]
        
        # Number patterns
        self.number_patterns = [
            r'^\d+$',  # Pure integer
            r'^\d+\.\d+$',  # Decimal
            r'^\d{1,3}(,\d{3})*(\.\d+)?$',  # Number with thousand separators
            r'^[$€£¥]\s*\d+',  # Currency
        ]
        
        # Code patterns (ID, serial number, etc.)
        self.code_patterns = [
            r'^[A-Z]{2,}\d{4,}$',  # ABC12345
            r'^\d{4,}-[A-Z\d]+$',  # 1234-ABC
            r'^[A-Z\d]{8,}$',  # Long alphanumeric
        ]
    
    def classify_token(self, token: Dict[str, Any]) -> str:
        """
        Phân loại 1 token OCR.
        
        Args:
            token: OCR token với 'text' field
            
        Returns:
            str: Token type ('date', 'form_key', 'form_value', 'number', 'code', 'text')
        """
        text = token.get('text', '').strip()
        
        if not text:
            return 'empty'
        
        # Check date
        if self._is_date(text):
            return 'date'
        
        # Check form key (label)
        if self._is_form_key(text):
            return 'form_key'
        
        # Check code/ID
        if self._is_code(text):
            return 'code'
        
        # Check number
        if self._is_number(text):
            return 'number'
        
        # Default
        return 'text'
    
    def _is_date(self, text: str) -> bool:
        """Check if text is a date."""
        for pattern in self.date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _is_form_key(self, text: str) -> bool:
        """Check if text is a form field label (key)."""
        # Pattern-based check
        for pattern in self.form_key_patterns:
            if re.match(pattern, text):
                return True
        
        # Keyword-based check (without colon)
        text_lower = text.lower().rstrip(':')
        if text_lower in self.form_keywords:
            return True
        
        return False
    
    def _is_number(self, text: str) -> bool:
        """Check if text is a number."""
        for pattern in self.number_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _is_code(self, text: str) -> bool:
        """Check if text is a code/ID."""
        for pattern in self.code_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def classify_tokens(self, tokens: List[Dict]) -> List[Dict]:
        """
        Phân loại tất cả tokens trong list.
        
        Args:
            tokens: List of OCR tokens
            
        Returns:
            List of tokens with added 'token_type' field
        """
        classified_tokens = []
        
        for i, token in enumerate(tokens):
            token_copy = token.copy()
            token_type = self.classify_token(token)
            token_copy['token_type'] = token_type
            
            # Detect form value (follows form key)
            if i > 0 and classified_tokens[i-1].get('token_type') == 'form_key':
                # If previous token is form_key and current is close spatially
                prev_bbox = classified_tokens[i-1].get('bbox', [])
                curr_bbox = token.get('bbox', [])
                
                if self._is_spatially_close(prev_bbox, curr_bbox):
                    token_copy['token_type'] = 'form_value'
            
            classified_tokens.append(token_copy)
        
        return classified_tokens
    
    def _is_spatially_close(self, bbox1: List, bbox2: List, threshold: float = 100) -> bool:
        """Check if two bboxes are close (same line or nearby)."""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        # Calculate center points
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        # Euclidean distance
        distance = ((x2_center - x1_center)**2 + (y2_center - y1_center)**2) ** 0.5
        
        return distance < threshold
    
    def get_statistics(self, tokens: List[Dict]) -> Dict[str, int]:
        """
        Thống kê phân loại tokens.
        
        Returns:
            Dict with counts for each token type
        """
        from collections import Counter
        
        types = [t.get('token_type', 'unknown') for t in tokens]
        return dict(Counter(types))
