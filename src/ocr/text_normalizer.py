"""
Text Normalization Utilities
Normalize OCR text for answer matching and comparison
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalize OCR text for robust answer matching
    Handles: case, punctuation, whitespace, numbers, unicode, OCR errors
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        normalize_whitespace: bool = True,
        normalize_numbers: bool = False,
        remove_accents: bool = False,
        fix_common_ocr_errors: bool = True
    ):
        """
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove all punctuation
            normalize_whitespace: Normalize all whitespace to single space
            normalize_numbers: Normalize number formats (1,000 -> 1000, 1.5K -> 1500)
            remove_accents: Remove accent marks (é -> e)
            fix_common_ocr_errors: Fix common OCR mistakes (0/O, 1/l, etc.)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.normalize_numbers = normalize_numbers
        self.remove_accents = remove_accents
        self.fix_common_ocr_errors = fix_common_ocr_errors
        
        # Common OCR error mappings
        self.ocr_error_map = {
            # Number/Letter confusions
            'O': '0',  # Letter O -> Zero (in numeric contexts)
            'l': '1',  # Lowercase L -> One (in numeric contexts)
            'I': '1',  # Uppercase I -> One (in numeric contexts)
            'S': '5',  # S -> 5 (in numeric contexts)
            'Z': '2',  # Z -> 2 (in numeric contexts)
            # Common character confusions
            '|': 'I',
            '¢': 'c',
            '§': 's',
        }
    
    def normalize(self, text: str, preserve_case: bool = False) -> str:
        """
        Apply full normalization pipeline
        
        Args:
            text: Input text
            preserve_case: Override lowercase setting for this call
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # 1. Unicode normalization (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # 2. Remove accents if needed
        if self.remove_accents:
            text = self._remove_accents(text)
        
        # 3. Fix common OCR errors
        if self.fix_common_ocr_errors:
            text = self._fix_ocr_errors(text)
        
        # 4. Normalize numbers
        if self.normalize_numbers:
            text = self._normalize_numbers(text)
        
        # 5. Lowercase
        if self.lowercase and not preserve_case:
            text = text.lower()
        
        # 6. Remove punctuation
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # 7. Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def normalize_for_matching(self, text: str) -> str:
        """
        Aggressive normalization for answer matching
        Removes punctuation, normalizes case, whitespace, numbers
        """
        return self.normalize(
            text,
            preserve_case=False
        )
    
    def normalize_for_display(self, text: str) -> str:
        """
        Light normalization for display (preserve more structure)
        Only fixes whitespace and OCR errors
        """
        normalizer = TextNormalizer(
            lowercase=False,
            remove_punctuation=False,
            normalize_whitespace=True,
            normalize_numbers=False,
            remove_accents=False,
            fix_common_ocr_errors=True
        )
        return normalizer.normalize(text)
    
    # Private normalization methods
    
    def _remove_accents(self, text: str) -> str:
        """Remove accent marks from characters"""
        # Decompose unicode (e.g., é -> e + ́)
        nfd = unicodedata.normalize('NFD', text)
        # Remove combining marks (accents)
        without_accents = ''.join(
            char for char in nfd
            if unicodedata.category(char) != 'Mn'
        )
        return without_accents
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR character confusions
        
        Strategy:
        - In numeric contexts (surrounded by digits), fix O->0, l->1, etc.
        - Otherwise, keep as-is
        """
        # Fix common character substitutions
        for wrong, correct in self.ocr_error_map.items():
            text = text.replace(wrong, correct)
        
        # Context-aware fixes (number/letter confusion)
        # Replace O with 0 when surrounded by digits
        text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)  # 1O5 -> 105
        text = re.sub(r'^O(\d)', r'0\g<1>', text)          # O123 -> 0123
        text = re.sub(r'(\d)O$', r'\g<1>0', text)          # 10O -> 100
        
        # Replace l/I with 1 when surrounded by digits
        text = re.sub(r'(\d)[lI](\d)', r'\g<1>1\g<2>', text)
        text = re.sub(r'^[lI](\d)', r'1\g<1>', text)
        text = re.sub(r'(\d)[lI]$', r'\g<1>1', text)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """
        Normalize number formats for consistent matching
        
        Handles:
        - Thousand separators: 1,000 -> 1000
        - Decimal separators: normalize to .
        - Currency symbols: $1,234.56 -> 1234.56
        - Abbreviations: 1.5K -> 1500, 2M -> 2000000
        - Percentages: 50% -> 50
        """
        # Remove currency symbols
        text = re.sub(r'[$€£¥₹]', '', text)
        
        # Handle abbreviations (K, M, B)
        def expand_abbreviation(match):
            num_str = match.group(1)
            abbr = match.group(2).upper()
            try:
                num = float(num_str.replace(',', ''))
                multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
                result = int(num * multipliers.get(abbr, 1))
                return str(result)
            except:
                return match.group(0)
        
        text = re.sub(r'([\d,\.]+)\s*([KMB])\b', expand_abbreviation, text, flags=re.IGNORECASE)
        
        # Remove thousand separators (1,000 -> 1000)
        # But be careful not to remove decimal separators
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # Handle 1,000
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # Handle 1,000,000 (repeat)
        
        # Remove percentage signs
        text = re.sub(r'(\d)%', r'\1', text)
        
        return text
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation while preserving structure
        Keep: apostrophes in contractions, hyphens in words
        """
        # Keep apostrophes in words (don't -> dont is OK, but we keep it)
        # Keep hyphens in words (e-mail, co-worker)
        # Remove other punctuation
        
        # Replace punctuation with spaces (except apostrophe and hyphen in words)
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        
        # Remove apostrophes not in words (e.g., trailing quotes)
        text = re.sub(r"(?<!\w)'|'(?!\w)", '', text)
        
        # Remove hyphens not in words
        text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces"""
        # Replace all whitespace (spaces, tabs, newlines) with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Utility methods for token/line normalization
    
    def normalize_tokens(self, tokens: List[Dict]) -> List[Dict]:
        """
        Normalize text in token list
        Returns new list with normalized 'text' field
        """
        normalized_tokens = []
        for token in tokens:
            normalized_token = token.copy()
            normalized_token['text'] = self.normalize(token['text'])
            normalized_token['original_text'] = token['text']  # Preserve original
            normalized_tokens.append(normalized_token)
        
        return normalized_tokens
    
    def normalize_lines(self, lines: List[Dict]) -> List[Dict]:
        """
        Normalize text in line list
        Returns new list with normalized 'text' field
        """
        normalized_lines = []
        for line in lines:
            normalized_line = line.copy()
            normalized_line['text'] = self.normalize(line['text'])
            normalized_line['original_text'] = line['text']
            normalized_lines.append(normalized_line)
        
        return normalized_lines


# Fuzzy matching utilities

def compute_similarity(text1: str, text2: str, normalizer: TextNormalizer = None) -> float:
    """
    Compute similarity between two text strings
    
    Args:
        text1, text2: Texts to compare
        normalizer: TextNormalizer instance (or use default)
        
    Returns:
        Similarity score [0, 1]
    """
    if normalizer is None:
        normalizer = TextNormalizer(
            lowercase=True,
            remove_punctuation=True,
            normalize_whitespace=True,
            normalize_numbers=True
        )
    
    # Normalize both texts
    norm1 = normalizer.normalize(text1)
    norm2 = normalizer.normalize(text2)
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Token-based Jaccard similarity
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    jaccard = intersection / union if union > 0 else 0.0
    
    return jaccard


def find_answer_span(
    answer: str,
    context: str,
    normalizer: TextNormalizer = None,
    threshold: float = 0.8
) -> Tuple[int, int, float]:
    """
    Find answer span in context using fuzzy matching
    
    Args:
        answer: Answer text to find
        context: Context text to search in
        normalizer: TextNormalizer instance
        threshold: Minimum similarity threshold
        
    Returns:
        (start_char, end_char, similarity) or (-1, -1, 0.0) if not found
    """
    if normalizer is None:
        normalizer = TextNormalizer()
    
    norm_answer = normalizer.normalize(answer)
    norm_context = normalizer.normalize(context)
    
    # Exact match
    start = norm_context.find(norm_answer)
    if start != -1:
        return (start, start + len(norm_answer), 1.0)
    
    # Fuzzy match: sliding window
    answer_tokens = norm_answer.split()
    context_tokens = norm_context.split()
    window_size = len(answer_tokens)
    
    best_start = -1
    best_end = -1
    best_similarity = 0.0
    
    for i in range(len(context_tokens) - window_size + 1):
        window = ' '.join(context_tokens[i:i + window_size])
        similarity = compute_similarity(norm_answer, window, normalizer)
        
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            # Map back to character positions (approximate)
            best_start = norm_context.find(window)
            best_end = best_start + len(window)
    
    return (best_start, best_end, best_similarity)


# Convenience function

def normalize_text(
    text: str,
    mode: str = 'matching'
) -> str:
    """
    Convenience function for text normalization
    
    Args:
        text: Input text
        mode: 'matching' (aggressive) or 'display' (light)
        
    Returns:
        Normalized text
    """
    normalizer = TextNormalizer()
    
    if mode == 'matching':
        return normalizer.normalize_for_matching(text)
    elif mode == 'display':
        return normalizer.normalize_for_display(text)
    else:
        return normalizer.normalize(text)


# Example usage
if __name__ == "__main__":
    # Test normalization
    test_texts = [
        "Invoice Number: INV-2024-001",
        "Total: $1,234.56 (10% tax)",
        "Date: O3/15/2O24",  # OCR errors: O instead of 0
        "Amount: 1.5K",
        "Café résumé",
    ]
    
    normalizer = TextNormalizer(
        lowercase=True,
        remove_punctuation=True,
        normalize_whitespace=True,
        normalize_numbers=True,
        remove_accents=True,
        fix_common_ocr_errors=True
    )
    
    print(f"\n{'='*60}")
    print("Text Normalization Examples")
    print(f"{'='*60}\n")
    
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()
    
    # Test similarity
    print(f"\n{'='*60}")
    print("Similarity Matching")
    print(f"{'='*60}\n")
    
    pairs = [
        ("INV-2024-001", "Invoice Number: INV-2024-001"),
        ("$1,234.56", "1234.56"),
        ("March 15, 2024", "03/15/2024"),
    ]
    
    for text1, text2 in pairs:
        similarity = compute_similarity(text1, text2, normalizer)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity: {similarity:.2%}\n")
