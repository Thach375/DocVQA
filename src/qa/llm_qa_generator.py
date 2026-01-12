"""
LLM-based QA Generator using Graph Edges.

Generates questions that require linking >=2 regions with verifiable answers.
Includes self-verification to reject hallucinations and unanswerable questions.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ReasoningType(Enum):
    """Types of reasoning required to answer the question."""
    EXTRACTION = "extraction"           # Direct extraction from text
    COMPARISON = "comparison"           # Compare values across regions
    AGGREGATION = "aggregation"         # Aggregate info from multiple regions
    INFERENCE = "inference"             # Logical inference from facts
    VALIDATION = "validation"           # Verify claim against evidence
    COREFERENCE = "coreference"         # Resolve references across regions
    CALCULATION = "calculation"         # Numerical computation
    TEMPORAL = "temporal"               # Time-based reasoning
    CAUSAL = "causal"                   # Cause-effect reasoning


class RejectReason(Enum):
    """Reasons for rejecting a generated QA pair."""
    HALLUCINATION = "hallucination"             # Answer not in evidence
    VAGUE_QUESTION = "vague_question"           # Question too ambiguous
    UNANSWERABLE = "unanswerable"               # Cannot derive answer from regions
    SINGLE_REGION = "single_region"             # Only uses 1 region (need >=2)
    NO_EVIDENCE = "no_evidence"                 # Missing evidence quotes
    TRIVIAL = "trivial"                         # Too simple/obvious
    INCONSISTENT = "inconsistent"               # Answer contradicts evidence


@dataclass
class Evidence:
    """Evidence supporting an answer."""
    region_id: int
    region_type: str
    quote: str
    relevance: str  # How this evidence supports the answer


@dataclass
class GeneratedQA:
    """A generated QA pair with evidence."""
    question: str
    answer: str
    evidence_region_ids: List[int]
    evidence_quotes: List[str]
    reasoning_type: str
    confidence: float
    relation_used: str
    metadata: Dict[str, Any]


@dataclass 
class VerificationResult:
    """Result of self-verification."""
    is_valid: bool
    reject_reason: Optional[str]
    consistency_score: float
    issues: List[str]


class LLMQAPromptBuilder:
    """
    Builds prompts for LLM-based QA generation.
    
    Framework:
    1. Input: nodeA, nodeB, relation, optional extra nodes
    2. Constraints: verifiable answer, cite evidence
    3. Output: structured JSON with question, answer, evidence
    """
    
    # System prompt defining the task
    SYSTEM_PROMPT = """You are an expert Question-Answer generator for document understanding.

Your task: Generate a question that REQUIRES information from MULTIPLE document regions to answer.

CRITICAL RULES:
1. The answer MUST be derivable from the provided text excerpts
2. The question MUST require linking >=2 regions (not answerable from single region)
3. You MUST cite exact evidence (region IDs + quote spans)
4. NO hallucination - every fact in answer must appear in evidence
5. Questions should be natural, as a human would ask about the document

AVOID:
- Questions answerable from just one region
- Vague or ambiguous questions
- Questions requiring external knowledge
- Yes/No questions without requiring evidence synthesis
- Trivial questions (e.g., "What is written in region 1?")
"""

    # Base prompt template
    BASE_TEMPLATE = """## DOCUMENT REGIONS

{regions_text}

## RELATION INFORMATION
- Primary Relation: {relation_type}
- Source Region: #{source_id} ({source_type})
- Target Region: #{target_id} ({target_type})
- Relation Score: {relation_score:.2f}

## YOUR TASK
Generate a question-answer pair that:
1. Requires synthesizing information from Region #{source_id} AND Region #{target_id}
2. Has an answer that can be VERIFIED from the quoted text
3. Tests {reasoning_description}

## OUTPUT FORMAT (JSON)
```json
{{
    "question": "Your natural question here",
    "answer": "Concise, factual answer",
    "evidence_region_ids": [list of region IDs used],
    "evidence_quotes": ["exact quote 1", "exact quote 2"],
    "reasoning_type": "{reasoning_type}",
    "reasoning_explanation": "How the answer is derived from evidence"
}}
```

Generate the QA pair:"""

    # Relation-specific prompt additions
    RELATION_PROMPTS = {
        # Text ↔ Table: Validate text claims against table data
        "text_table_validation": {
            "reasoning_type": "validation",
            "reasoning_description": "validating a text claim against tabular data",
            "extra_instructions": """
FOCUS: Generate a question that asks to verify/validate information mentioned in the text 
using data from the table, or vice versa.

GOOD EXAMPLES:
- "According to the table, is the claim about [X] in the text accurate?"
- "What value in the table corresponds to [entity] mentioned in paragraph [Y]?"
- "Does the table support the statement that [claim]?"

BAD EXAMPLES:
- "What does the table show?" (single region)
- "What is mentioned in the text?" (single region)
"""
        },
        
        # Figure ↔ Caption: Map visual descriptions to captions
        "figure_caption_mapping": {
            "reasoning_type": "coreference",
            "reasoning_description": "connecting figure content with its caption description",
            "extra_instructions": """
FOCUS: Generate a question that requires understanding both the figure description 
and its caption to answer completely.

GOOD EXAMPLES:
- "What does Figure [X] illustrate according to its caption?"
- "Which element in the figure corresponds to [description in caption]?"
- "How does the caption explain the [specific feature] shown in the figure?"

BAD EXAMPLES:
- "What is the figure about?" (too vague)
- "Read the caption" (not a question)
"""
        },
        
        # Text ↔ Text: Coreference resolution
        "text_text_coreference": {
            "reasoning_type": "coreference",
            "reasoning_description": "resolving references and connecting information across text blocks",
            "extra_instructions": """
FOCUS: Generate a question that requires connecting pronouns, abbreviations, 
or references across different text regions.

GOOD EXAMPLES:
- "What does '[pronoun/abbreviation]' in Region [X] refer to based on Region [Y]?"
- "How does the information in paragraph [X] relate to [entity] mentioned in paragraph [Y]?"
- "What additional details about [entity from Region X] are provided in Region [Y]?"

BAD EXAMPLES:
- "What is in Region X?" (single region)
- Questions that don't require cross-referencing
"""
        },
        
        # Table ↔ Table: Cross-check data
        "table_table_crosscheck": {
            "reasoning_type": "comparison",
            "reasoning_description": "comparing or cross-checking data across multiple tables",
            "extra_instructions": """
FOCUS: Generate a question that requires comparing, aggregating, or cross-referencing 
data from multiple tables.

GOOD EXAMPLES:
- "What is the difference in [metric] between Table [X] and Table [Y]?"
- "Which [entity] appears in both tables and what are its values in each?"
- "How do the [category] totals compare across the two tables?"

BAD EXAMPLES:
- "What does Table 1 show?" (single table)
- Questions answerable from one table alone
"""
        },
        
        # Form ↔ Conclusion: Connect form data to conclusions
        "form_conclusion": {
            "reasoning_type": "inference",
            "reasoning_description": "connecting form field values to conclusions or summaries",
            "extra_instructions": """
FOCUS: Generate a question that requires using form field values to understand 
or verify conclusions/summaries in the document.

GOOD EXAMPLES:
- "Based on the [field] value in the form, what conclusion is drawn in the summary?"
- "How does the form data support the statement that [conclusion]?"
- "What form field provides evidence for the claim about [topic]?"

BAD EXAMPLES:
- "What is the value of [field]?" (single region extraction)
- Questions not requiring both form and conclusion
"""
        },
        
        # Generic spatial relations
        "spatial_above_below": {
            "reasoning_type": "extraction",
            "reasoning_description": "connecting spatially related content (above/below)",
            "extra_instructions": """
FOCUS: Generate a question where the spatial relationship (above/below) is meaningful 
for understanding the document structure.

GOOD EXAMPLES:
- "What heading appears above the [content description]?"
- "What details are provided below the [section title]?"
- "How does the content above relate to the data below?"
"""
        },
        
        # Nearest neighbor (contextual)
        "nearest_neighbor": {
            "reasoning_type": "extraction",
            "reasoning_description": "connecting contextually related nearby regions",
            "extra_instructions": """
FOCUS: Generate a question that requires understanding the contextual relationship 
between nearby document elements.

GOOD EXAMPLES:
- "What context does Region [X] provide for understanding Region [Y]?"
- "How are these adjacent sections related?"
"""
        }
    }

    @classmethod
    def build_prompt(
        cls,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        relation: str,
        relation_score: float,
        extra_nodes: Optional[List[Dict[str, Any]]] = None,
        relation_category: Optional[str] = None
    ) -> str:
        """
        Build a complete prompt for QA generation.
        
        Args:
            source_node: Source region node with 'node_id', 'region_type', 'text', 'bbox'
            target_node: Target region node
            relation: Relation type (e.g., 'above', 'is_caption_of')
            relation_score: Confidence score of the relation
            extra_nodes: Optional additional context nodes
            relation_category: Category for selecting appropriate prompt template
        
        Returns:
            Complete prompt string
        """
        # Build regions text
        regions_text = cls._format_regions(source_node, target_node, extra_nodes)
        
        # Determine relation category if not provided
        if relation_category is None:
            relation_category = cls._infer_relation_category(
                source_node['region_type'],
                target_node['region_type'],
                relation
            )
        
        # Get relation-specific prompt additions
        relation_config = cls.RELATION_PROMPTS.get(
            relation_category,
            cls.RELATION_PROMPTS['nearest_neighbor']  # Default fallback
        )
        
        # Build the prompt
        prompt = cls.BASE_TEMPLATE.format(
            regions_text=regions_text,
            relation_type=relation,
            source_id=source_node['node_id'],
            source_type=source_node['region_type'],
            target_id=target_node['node_id'],
            target_type=target_node['region_type'],
            relation_score=relation_score,
            reasoning_description=relation_config['reasoning_description'],
            reasoning_type=relation_config['reasoning_type']
        )
        
        # Add relation-specific instructions
        prompt += "\n\n## SPECIFIC GUIDANCE\n" + relation_config['extra_instructions']
        
        return prompt
    
    @classmethod
    def _format_regions(
        cls,
        source_node: Dict,
        target_node: Dict,
        extra_nodes: Optional[List[Dict]] = None
    ) -> str:
        """Format region information for the prompt."""
        regions = [source_node, target_node]
        if extra_nodes:
            regions.extend(extra_nodes)
        
        formatted = []
        for node in regions:
            region_str = f"""### Region #{node['node_id']} ({node['region_type']})
```
{node['text']}
```
"""
            formatted.append(region_str)
        
        return "\n".join(formatted)
    
    @classmethod
    def _infer_relation_category(
        cls,
        source_type: str,
        target_type: str,
        relation: str
    ) -> str:
        """Infer the relation category from region types and relation."""
        source_type = source_type.lower()
        target_type = target_type.lower()
        
        # Text ↔ Table
        if ('text' in source_type and 'table' in target_type) or \
           ('table' in source_type and 'text' in target_type):
            return 'text_table_validation'
        
        # Figure ↔ Caption
        if 'figure' in source_type or 'figure' in target_type:
            if 'caption' in relation.lower() or 'text' in source_type or 'text' in target_type:
                return 'figure_caption_mapping'
        
        # Table ↔ Table
        if 'table' in source_type and 'table' in target_type:
            return 'table_table_crosscheck'
        
        # Form relations
        if 'form' in source_type or 'form' in target_type:
            return 'form_conclusion'
        
        # Text ↔ Text
        if 'text' in source_type and 'text' in target_type:
            return 'text_text_coreference'
        
        # Spatial relations
        if relation in ['above', 'below', 'left_of', 'right_of']:
            return 'spatial_above_below'
        
        return 'nearest_neighbor'


class QAVerifier:
    """
    Self-verification system for generated QA pairs.
    
    Checks:
    1. Consistency: Answer derivable from evidence
    2. Multi-region: Requires >=2 regions
    3. No hallucination: All facts grounded in evidence
    4. Not trivial: Requires actual reasoning
    """
    
    @classmethod
    def verify(
        cls,
        qa: GeneratedQA,
        source_node: Dict,
        target_node: Dict,
        extra_nodes: Optional[List[Dict]] = None
    ) -> VerificationResult:
        """
        Verify a generated QA pair.
        
        Returns:
            VerificationResult with validity status and issues
        """
        issues = []
        
        # Check 1: Multiple regions used
        if len(qa.evidence_region_ids) < 2:
            issues.append("Only uses single region - need >=2")
            return VerificationResult(
                is_valid=False,
                reject_reason=RejectReason.SINGLE_REGION.value,
                consistency_score=0.0,
                issues=issues
            )
        
        # Check 2: Evidence quotes provided
        if not qa.evidence_quotes or len(qa.evidence_quotes) < 2:
            issues.append("Missing or insufficient evidence quotes")
            return VerificationResult(
                is_valid=False,
                reject_reason=RejectReason.NO_EVIDENCE.value,
                consistency_score=0.0,
                issues=issues
            )
        
        # Check 3: Evidence quotes exist in source text
        all_text = cls._collect_all_text(source_node, target_node, extra_nodes)
        quote_validity = cls._verify_quotes(qa.evidence_quotes, all_text)
        
        if quote_validity['invalid_count'] > 0:
            issues.append(f"{quote_validity['invalid_count']} quote(s) not found in source text")
            if quote_validity['invalid_count'] == len(qa.evidence_quotes):
                return VerificationResult(
                    is_valid=False,
                    reject_reason=RejectReason.HALLUCINATION.value,
                    consistency_score=0.0,
                    issues=issues
                )
        
        # Check 4: Answer consistency with evidence
        consistency_score = cls._check_answer_consistency(qa.answer, qa.evidence_quotes, all_text)
        
        if consistency_score < 0.3:
            issues.append(f"Low answer-evidence consistency: {consistency_score:.2f}")
            return VerificationResult(
                is_valid=False,
                reject_reason=RejectReason.HALLUCINATION.value,
                consistency_score=consistency_score,
                issues=issues
            )
        
        # Check 5: Question quality
        question_issues = cls._check_question_quality(qa.question)
        issues.extend(question_issues)
        
        if any('vague' in issue.lower() for issue in question_issues):
            return VerificationResult(
                is_valid=False,
                reject_reason=RejectReason.VAGUE_QUESTION.value,
                consistency_score=consistency_score,
                issues=issues
            )
        
        if any('trivial' in issue.lower() for issue in question_issues):
            return VerificationResult(
                is_valid=False,
                reject_reason=RejectReason.TRIVIAL.value,
                consistency_score=consistency_score,
                issues=issues
            )
        
        # All checks passed
        return VerificationResult(
            is_valid=True,
            reject_reason=None,
            consistency_score=consistency_score,
            issues=issues if issues else []
        )
    
    @classmethod
    def _collect_all_text(
        cls,
        source_node: Dict,
        target_node: Dict,
        extra_nodes: Optional[List[Dict]]
    ) -> str:
        """Collect all text from nodes."""
        texts = [source_node.get('text', ''), target_node.get('text', '')]
        if extra_nodes:
            texts.extend([n.get('text', '') for n in extra_nodes])
        return ' '.join(texts).lower()
    
    @classmethod
    def _verify_quotes(cls, quotes: List[str], all_text: str) -> Dict:
        """Verify that quotes exist in source text."""
        valid_count = 0
        invalid_quotes = []
        
        for quote in quotes:
            # Normalize for comparison
            normalized_quote = quote.lower().strip()
            
            # Check exact match or fuzzy match
            if normalized_quote in all_text:
                valid_count += 1
            elif cls._fuzzy_match(normalized_quote, all_text):
                valid_count += 1
            else:
                invalid_quotes.append(quote)
        
        return {
            'valid_count': valid_count,
            'invalid_count': len(invalid_quotes),
            'invalid_quotes': invalid_quotes
        }
    
    @classmethod
    def _fuzzy_match(cls, quote: str, text: str, threshold: float = 0.8) -> bool:
        """Check if quote has fuzzy match in text."""
        # Simple word overlap check
        quote_words = set(quote.split())
        text_words = set(text.split())
        
        if not quote_words:
            return False
        
        overlap = len(quote_words & text_words) / len(quote_words)
        return overlap >= threshold
    
    @classmethod
    def _check_answer_consistency(
        cls,
        answer: str,
        evidence_quotes: List[str],
        all_text: str
    ) -> float:
        """
        Check if answer is consistent with evidence.
        
        Returns score 0-1 indicating consistency.
        """
        answer_lower = answer.lower()
        evidence_text = ' '.join(evidence_quotes).lower()
        
        # Extract key terms from answer
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                      'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                      'very', 'just', 'also'}
        
        answer_words = set(answer_lower.split()) - stop_words
        
        if not answer_words:
            return 0.5  # Can't verify empty/trivial answers
        
        # Check how many answer key terms appear in evidence
        found_in_evidence = sum(1 for word in answer_words 
                                if word in evidence_text or word in all_text)
        
        consistency = found_in_evidence / len(answer_words)
        return consistency
    
    @classmethod
    def _check_question_quality(cls, question: str) -> List[str]:
        """Check question quality and return issues."""
        issues = []
        
        question_lower = question.lower()
        
        # Check for vague questions
        vague_patterns = [
            r'^what is this',
            r'^what does .+ show$',
            r'^describe',
            r'^explain',
            r'^tell me about',
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, question_lower):
                issues.append(f"Vague question pattern: {pattern}")
                break
        
        # Check for trivial questions
        trivial_patterns = [
            r'^what is (written|shown|displayed) in region',
            r'^read region',
            r'^what does region \d+ (say|contain)',
        ]
        
        for pattern in trivial_patterns:
            if re.search(pattern, question_lower):
                issues.append(f"Trivial question pattern: {pattern}")
                break
        
        # Check minimum length
        if len(question.split()) < 5:
            issues.append("Question too short (< 5 words)")
        
        # Check for question mark
        if not question.strip().endswith('?'):
            issues.append("Question doesn't end with '?'")
        
        return issues


class LLMQAGenerator:
    """
    Main class for generating QA pairs using LLM.
    
    Workflow:
    1. Select relevant graph edges
    2. Build prompts using LLMQAPromptBuilder
    3. Call LLM to generate QA
    4. Parse and verify using QAVerifier
    5. Return valid QA pairs
    """
    
    def __init__(self, llm_client=None, model_name: str = "gpt-4"):
        """
        Initialize generator.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            model_name: Model to use for generation
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.prompt_builder = LLMQAPromptBuilder()
        self.verifier = QAVerifier()
    
    def generate_from_edge(
        self,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        edge: Dict[str, Any],
        extra_nodes: Optional[List[Dict]] = None,
        max_retries: int = 3
    ) -> Optional[GeneratedQA]:
        """
        Generate a QA pair from a graph edge.
        
        Args:
            source_node: Source region node
            target_node: Target region node
            edge: Edge dict with 'relation', 'score', etc.
            extra_nodes: Additional context nodes
            max_retries: Max generation attempts
        
        Returns:
            GeneratedQA if valid, None if rejected
        """
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            source_node=source_node,
            target_node=target_node,
            relation=edge['relation'],
            relation_score=edge['score'],
            extra_nodes=extra_nodes
        )
        
        for attempt in range(max_retries):
            try:
                # Call LLM (placeholder - implement with actual client)
                response = self._call_llm(prompt)
                
                # Parse response
                qa = self._parse_response(response, edge)
                
                if qa is None:
                    continue
                
                # Verify
                verification = self.verifier.verify(
                    qa, source_node, target_node, extra_nodes
                )
                
                if verification.is_valid:
                    qa.confidence = verification.consistency_score
                    return qa
                else:
                    # Log rejection for debugging
                    print(f"QA rejected (attempt {attempt+1}): {verification.reject_reason}")
                    print(f"  Issues: {verification.issues}")
                    
            except Exception as e:
                print(f"Generation error (attempt {attempt+1}): {e}")
        
        return None
    
    def generate_batch(
        self,
        graph_data: Dict[str, Any],
        max_qa_per_document: int = 10,
        min_edge_score: float = 0.3
    ) -> List[GeneratedQA]:
        """
        Generate QA pairs from a document graph.
        
        Args:
            graph_data: Dict with 'nodes', 'edges', 'adjacency'
            max_qa_per_document: Maximum QA pairs to generate
            min_edge_score: Minimum edge score to consider
        
        Returns:
            List of valid GeneratedQA objects
        """
        nodes = {n['node_id']: n for n in graph_data['nodes']}
        edges = graph_data['edges']
        
        # Filter and sort edges by score
        valid_edges = [e for e in edges if e['score'] >= min_edge_score]
        valid_edges.sort(key=lambda x: x['score'], reverse=True)
        
        generated_qa = []
        
        for edge in valid_edges:
            if len(generated_qa) >= max_qa_per_document:
                break
            
            source_node = nodes.get(edge['source'])
            target_node = nodes.get(edge['target'])
            
            if not source_node or not target_node:
                continue
            
            qa = self.generate_from_edge(source_node, target_node, edge)
            
            if qa:
                generated_qa.append(qa)
        
        return generated_qa
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt.
        
        Override this method with actual LLM client implementation.
        """
        if self.llm_client is None:
            raise NotImplementedError(
                "LLM client not configured. Set self.llm_client or override _call_llm()"
            )
        
        # Example for OpenAI
        # response = self.llm_client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "system", "content": LLMQAPromptBuilder.SYSTEM_PROMPT},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.7,
        #     max_tokens=1000
        # )
        # return response.choices[0].message.content
        
        raise NotImplementedError("Implement _call_llm with your LLM client")
    
    def _parse_response(self, response: str, edge: Dict) -> Optional[GeneratedQA]:
        """Parse LLM response into GeneratedQA object."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try parsing entire response as JSON
                json_str = response
            
            data = json.loads(json_str)
            
            return GeneratedQA(
                question=data['question'],
                answer=data['answer'],
                evidence_region_ids=data['evidence_region_ids'],
                evidence_quotes=data['evidence_quotes'],
                reasoning_type=data.get('reasoning_type', 'extraction'),
                confidence=0.0,  # Will be set after verification
                relation_used=edge['relation'],
                metadata={
                    'reasoning_explanation': data.get('reasoning_explanation', ''),
                    'edge_score': edge['score'],
                    'edge_category': edge.get('category', '')
                }
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            return None


# Export QA to standard format
def export_qa_dataset(
    qa_list: List[GeneratedQA],
    output_path: str,
    document_id: str
) -> None:
    """Export generated QA pairs to JSON format."""
    output_data = {
        'document_id': document_id,
        'qa_pairs': [asdict(qa) for qa in qa_list],
        'statistics': {
            'total': len(qa_list),
            'by_reasoning_type': {},
            'by_relation': {}
        }
    }
    
    # Compute statistics
    for qa in qa_list:
        rt = qa.reasoning_type
        output_data['statistics']['by_reasoning_type'][rt] = \
            output_data['statistics']['by_reasoning_type'].get(rt, 0) + 1
        
        rel = qa.relation_used
        output_data['statistics']['by_relation'][rel] = \
            output_data['statistics']['by_relation'].get(rel, 0) + 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
