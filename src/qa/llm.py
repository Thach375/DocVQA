from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# -----------------------------
# Utilities
# -----------------------------
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_get_text(node: Dict[str, Any]) -> str:
    t = node.get("text") or ""
    return t if isinstance(t, str) else str(t)

def region_category(region_type: str) -> str:
    """Map node region_type to our coarse categories."""
    rt = (region_type or "").lower()
    if rt in {"form"}:
        return "form"
    if rt in {"table"}:
        return "table"
    if rt in {"figure", "chart", "plot", "graph"}:
        return "figure"
    # common variants
    if "table" in rt:
        return "table"
    if "form" in rt:
        return "form"
    if "fig" in rt or "chart" in rt or "plot" in rt:
        return "figure"
    return "text"

def is_long_text(node_text: str, min_words: int = 40) -> bool:
    return len(normalize_space(node_text).split()) >= min_words

def build_adjacency(nodes: Dict[int, Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    adj = {nid: [] for nid in nodes}
    for e in edges or []:
        src = e.get("source")
        tgt = e.get("target")
        if src is None or tgt is None:
            continue
        if src in adj:
            adj[src].append(tgt)
        if tgt in adj:
            adj[tgt].append(src)
    return adj

# -----------------------------
# Rule-based generators
# -----------------------------
def extract_form_fields(text: str) -> List[Tuple[str, str]]:
    """
    Trích xuất tất cả form fields (key: value) từ text.
    Xử lý cả trường hợp nhiều fields trên cùng 1 line.
    
    Args:
        text: Text của node (có thể nhiều lines hoặc 1 line ghép)
        
    Returns:
        List of (key, value) tuples
    """
    fields = []
    
    # Sentence starters to filter out (these are unlikely to be form keys)
    sentence_starters = {'this', 'that', 'the', 'it', 'there', 'here', 
                         'what', 'when', 'where', 'which', 'who', 'how',
                         'please', 'if', 'as', 'we', 'they', 'you', 'i', 'he', 'she',
                         'our', 'your', 'his', 'her', 'my', 'a', 'an', 'and', 'or', 'but'}
    
    def is_valid_key(key: str) -> bool:
        """Check if key is valid form field label."""
        if not key:
            return False
        if len(key) > 30:
            return False
        if key.count(' ') > 4:
            return False
        if '  ' in key:  # Double space = likely sentence
            return False
        
        first_word = key.split()[0].lower() if key else ''
        if first_word in sentence_starters:
            return False
        
        return True
    
    # Process each line
    for line in text.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        # Find all colon positions
        colon_indices = [i for i, c in enumerate(line) if c == ':']
        
        if len(colon_indices) == 1:
            # Simple case: only one colon
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ''
            if is_valid_key(key) and value:
                fields.append((key, value))
        else:
            # Multiple colons: need to find key boundaries
            # Strategy: for each colon, look backwards for the start of key
            # Key starts at: beginning of line, or after a value (lowercase followed by space + capital)
            
            extracted = []
            prev_value_end = 0  # Track where previous value ended
            
            for idx, colon_pos in enumerate(colon_indices):
                # Get text between previous value end and this colon
                text_segment = line[prev_value_end:colon_pos]
                
                # The key is typically the last capitalized phrase in this segment
                # Find where the key starts by looking for capital letter after space
                key_start_in_segment = 0
                
                # If this is not the first field, find where key starts
                if prev_value_end > 0:
                    # Look for pattern: lowercase/digit followed by space + capital
                    for j in range(len(text_segment) - 1):
                        if (text_segment[j].islower() or text_segment[j].isdigit() or text_segment[j] in '.,;') \
                           and j + 1 < len(text_segment) and text_segment[j + 1] == ' ':
                            # Check if there's a capital letter after spaces
                            for k in range(j + 2, len(text_segment)):
                                if text_segment[k] == ' ':
                                    continue
                                if text_segment[k].isupper():
                                    key_start_in_segment = k
                                    break
                                break
                
                key = text_segment[key_start_in_segment:].strip()
                
                if is_valid_key(key):
                    extracted.append((colon_pos, key, prev_value_end + key_start_in_segment))
                    # Update prev_value_end to after this colon
                    prev_value_end = colon_pos + 1
            
            # Now extract values based on key positions
            for i, (colon_pos, key, key_start) in enumerate(extracted):
                value_start = colon_pos + 1
                
                # Value ends at the start of next key or end of line
                if i + 1 < len(extracted):
                    value_end = extracted[i + 1][2]  # Start of next key
                else:
                    value_end = len(line)
                
                value = line[value_start:value_end].strip()
                if value:
                    fields.append((key, value))
    
    return fields


def gen_form_kv_qas(node: Dict[str, Any], max_qas: int = 5) -> List[Dict[str, Any]]:
    """
    Generate simple Key-Value questions from form-like text.
    Đã cải thiện: tách đúng form fields, không bị ghép nhiều fields vào 1 answer.
    """
    nid = node["node_id"]
    text = safe_get_text(node)
    qas: List[Dict[str, Any]] = []
    
    # Sử dụng hàm extract mới
    fields = extract_form_fields(text)
    
    for key, value in fields:
        # Tạo câu hỏi
        qas.append({
            "question": f"What is the {key}?",
            "answer": value,
            "evidence_region_ids": [nid],
            "evidence_quotes": [f"{key}: {value}"],
            "reasoning_type": "lookup",
            "reasoning_explanation": f"Extracted value for '{key}' from form field."
        })
        
        if len(qas) >= max_qas:
            break
    
    return qas

def _split_row(row: str) -> List[str]:
    row = row.strip().strip("|")
    if "|" in row:
        cells = [c.strip() for c in row.split("|")]
        return [c for c in cells if c]
    if "\t" in row:
        cells = [c.strip() for c in row.split("\t")]
        return [c for c in cells if c]
    # split on 2+ spaces
    cells = [c.strip() for c in re.split(r"\s{2,}", row)]
    return [c for c in cells if c]

def parse_table_text(table_text: str) -> List[List[str]]:
    lines = [l for l in table_text.splitlines() if l.strip()]
    rows = [_split_row(l) for l in lines]
    # keep rows with at least 2 cells
    rows = [r for r in rows if len(r) >= 2]
    return rows

def gen_table_qas(node: Dict[str, Any], max_qas: int = 5) -> List[Dict[str, Any]]:
    nid = node["node_id"]
    text = safe_get_text(node)
    rows = parse_table_text(text)
    if not rows:
        return []
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    qas: List[Dict[str, Any]] = []

    # Q1: column headers
    qas.append({
        "question": "What are the column headers in the table?",
        "answer": ", ".join(header),
        "evidence_region_ids": [nid],
        "evidence_quotes": [" | ".join(header)],
        "reasoning_type": "lookup",
        "reasoning_explanation": "The header row contains the table's column names."
    })

    # Q2: row count (excluding header if present)
    qas.append({
        "question": "How many data rows are in the table (excluding the header)?",
        "answer": str(max(0, len(data))),
        "evidence_region_ids": [nid],
        "evidence_quotes": [f"Data rows detected: {len(data)}"],
        "reasoning_type": "lookup",
        "reasoning_explanation": "Counted the number of non-header rows parsed from the table."
    })

    # Q3+: cell lookup questions
    if data and len(header) >= 2 and len(data[0]) >= 2:
        # choose up to 3 rows
        for r in data[: min(3, len(data))]:
            row_id = r[0]
            col_idx = 1
            col_name = header[col_idx] if col_idx < len(header) else f"column {col_idx+1}"
            cell_val = r[col_idx] if col_idx < len(r) else ""
            if not cell_val:
                continue
            qas.append({
                "question": f"What is the {col_name} for {row_id}?",
                "answer": cell_val,
                "evidence_region_ids": [nid],
                "evidence_quotes": [" | ".join(r)],
                "reasoning_type": "lookup",
                "reasoning_explanation": "Looked up the cell at the intersection of the selected row and column."
            })
            if len(qas) >= max_qas:
                break

    return qas[:max_qas]

def gen_figure_qas(node: Dict[str, Any], max_qas: int = 3) -> List[Dict[str, Any]]:
    """Very light figure QA based on its textual description (OCR/alt-text)."""
    nid = node["node_id"]
    text = normalize_space(safe_get_text(node))
    if not text:
        return []
    qas: List[Dict[str, Any]] = []
    qas.append({
        "question": "What does the figure show?",
        "answer": text,
        "evidence_region_ids": [nid],
        "evidence_quotes": [safe_get_text(node)[:300]],
        "reasoning_type": "lookup",
        "reasoning_explanation": "Answered using the figure's extracted description text."
    })
    return qas[:max_qas]

# -----------------------------
# LLM-based generator (with fallback)
# -----------------------------
@dataclass
class LLMBackend:
    name: str = "flan-t5-small-local"
    model_id: str = "google/flan-t5-small"
    max_new_tokens: int = 384
    num_beams: int = 4
    available: bool = False
    _tokenizer: Any = None
    _model: Any = None

    def try_load(self) -> None:
        """Try to load a small model from local cache only."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, local_files_only=True)
            self.available = True
        except Exception:
            self.available = False
            self._tokenizer = None
            self._model = None

    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("LLM backend not available locally.")
        import torch  # type: ignore
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

def _json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON object embedded in a string."""
    s = s.strip()
    # if the model returns fenced block
    if "```" in s:
        # take the largest json-like block
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        for b in blocks:
            try:
                return json.loads(b)
            except Exception:
                continue
    # otherwise try raw JSON
    try:
        return json.loads(s)
    except Exception:
        # try locate first {...}
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

def deterministic_single_region_qa(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tạo QA từ single region (cho long text).
    Thay thế multi-region fallback không hợp lý.
    """
    nid = node["node_id"]
    text = normalize_space(safe_get_text(node))
    
    # Lấy câu đầu tiên hoặc đoạn đầu
    sentences = re.split(r'[.!?]', text)
    first_sentence = sentences[0].strip() if sentences else text[:100]
    
    # Tạo câu hỏi về nội dung chính
    words = text.split()
    summary = ' '.join(words[:30]) + ('...' if len(words) > 30 else '')
    
    return {
        "question": "What is the main topic or content discussed in this section?",
        "answer": summary,
        "evidence_region_ids": [nid],
        "evidence_quotes": [text[:300]],
        "reasoning_type": "comprehension",
        "reasoning_explanation": "Summarized the main content from the text region."
    }


def find_title_or_header_for_region(
    target_node: Dict[str, Any],
    nodes: Dict[int, Dict[str, Any]],
    adjacency: Dict[int, List[int]]
) -> Optional[Dict[str, Any]]:
    """
    Tìm title/header/caption cho một region (table/figure).
    Title thường là text ngắn, nằm trên hoặc dưới region.
    
    Args:
        target_node: Node cần tìm title (table/figure)
        nodes: Dict of all nodes
        adjacency: Adjacency list
        
    Returns:
        Title node nếu tìm được, None nếu không
    """
    target_id = target_node["node_id"]
    target_bbox = target_node.get("bbox", [])
    
    if not target_bbox:
        return None
    
    # Lấy y_center của target
    target_y_min = min(p[1] for p in target_bbox)
    target_y_max = max(p[1] for p in target_bbox)
    
    candidates = []
    
    # Tìm trong neighbors trước
    neighbor_ids = adjacency.get(target_id, [])
    
    for nid in neighbor_ids:
        if nid == target_id or nid not in nodes:
            continue
        
        n = nodes[nid]
        n_cat = region_category(n.get("region_type", ""))
        n_text = normalize_space(safe_get_text(n))
        
        # Title/caption thường là text ngắn
        word_count = len(n_text.split())
        
        if n_cat == "text" and 2 <= word_count <= 25:
            n_bbox = n.get("bbox", [])
            if n_bbox:
                n_y_min = min(p[1] for p in n_bbox)
                n_y_max = max(p[1] for p in n_bbox)
                
                # Title nằm trên target (y nhỏ hơn)
                if n_y_max < target_y_min:
                    distance = target_y_min - n_y_max
                    # Kiểm tra có từ khóa title/figure/table không
                    has_keyword = any(kw in n_text.lower() for kw in 
                                     ['table', 'figure', 'fig.', 'tab.', 'chart', 'exhibit'])
                    candidates.append((n, distance, has_keyword, 'above'))
                
                # Caption nằm dưới target (y lớn hơn)
                elif n_y_min > target_y_max:
                    distance = n_y_min - target_y_max
                    has_keyword = any(kw in n_text.lower() for kw in 
                                     ['table', 'figure', 'fig.', 'tab.', 'chart', 'source', 'note'])
                    candidates.append((n, distance, has_keyword, 'below'))
    
    if not candidates:
        return None
    
    # Ưu tiên: có keyword > gần hơn > nằm trên
    candidates.sort(key=lambda x: (not x[2], x[1], x[3] != 'above'))
    
    return candidates[0][0]


def gen_title_content_qa(
    title_node: Dict[str, Any],
    content_node: Dict[str, Any],
    content_type: str
) -> Dict[str, Any]:
    """
    Tạo QA liên kết title/caption với content (table/figure).
    Đây là cross-region QA có ý nghĩa vì người dùng có thể thấy title.
    
    Args:
        title_node: Node chứa title/caption
        content_node: Node chứa table/figure
        content_type: "table" hoặc "figure"
        
    Returns:
        QA dict
    """
    title_text = normalize_space(safe_get_text(title_node))
    content_text = normalize_space(safe_get_text(content_node))
    
    title_id = title_node["node_id"]
    content_id = content_node["node_id"]
    
    if content_type == "table":
        # Hỏi về nội dung bảng dựa trên title
        question = f"What information does '{title_text}' present?"
        
        # Trích xuất headers từ table
        rows = parse_table_text(safe_get_text(content_node))
        if rows:
            headers = rows[0]
            answer = f"The table shows data with columns: {', '.join(headers)}."
            if len(rows) > 1:
                answer += f" It contains {len(rows)-1} data rows."
        else:
            answer = content_text[:200]
        
    elif content_type == "figure":
        question = f"What does '{title_text}' illustrate?"
        answer = content_text if content_text else "The figure illustrates the referenced data."
    
    else:
        question = f"What is the content related to '{title_text}'?"
        answer = content_text[:200]
    
    return {
        "question": question,
        "answer": answer,
        "evidence_region_ids": [title_id, content_id],
        "evidence_quotes": [title_text, content_text[:300]],
        "reasoning_type": "coreference",
        "reasoning_explanation": f"Connected the title/caption with its {content_type} content."
    }

def build_prompt_from_template(template_str: str, a: Dict[str, Any], b: Dict[str, Any], a_type: str, b_type: str) -> str:
    """Build prompt from template, handling JSON braces in template safely."""
    # Use manual replacement instead of .format() to avoid issues with JSON braces
    replacements = {
        "{source_id}": str(a["node_id"]),
        "{source_type}": a_type,
        "{source_text}": safe_get_text(a),
        "{target_id}": str(b["node_id"]),
        "{target_type}": b_type,
        "{target_text}": safe_get_text(b),
        "{figure_id}": str(a["node_id"]),
        "{caption_id}": str(b["node_id"]),
        "{text1_id}": str(a["node_id"]),
        "{text2_id}": str(b["node_id"]),
        "{table1_id}": str(a["node_id"]),
        "{table2_id}": str(b["node_id"]),
        "{form_id}": str(a["node_id"]),
        "{conclusion_id}": str(b["node_id"]),
    }
    
    result = template_str
    for key, value in replacements.items():
        result = result.replace(key, value)
    
    return result

def choose_llm_template(llm_templates: Dict[str, Any], a_cat: str, b_cat: str) -> Optional[Dict[str, Any]]:
    pts = llm_templates.get("prompt_templates", {})
    # map category pairs to template keys
    if (a_cat, b_cat) in {("text", "table"), ("table", "text")} and "text_table_validation" in pts:
        return pts["text_table_validation"]
    if (a_cat, b_cat) in {("figure", "text"), ("text", "figure")} and "figure_caption_mapping" in pts:
        return pts["figure_caption_mapping"]
    if (a_cat, b_cat) == ("text", "text") and "text_text_coreference" in pts:
        return pts["text_text_coreference"]
    if (a_cat, b_cat) == ("table", "table") and "table_table_crosscheck" in pts:
        return pts["table_table_crosscheck"]
    if (a_cat, b_cat) in {("form", "text"), ("text", "form")} and "form_conclusion" in pts:
        return pts["form_conclusion"]
    return None

def generate_llm_qas(
    nodes: Dict[int, Dict[str, Any]],
    adjacency: Dict[int, List[int]],
    llm_templates: Dict[str, Any],
    llm_backend: LLMBackend,
    max_qas: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate QAs for document regions.
    
    LOGIC MỚI:
    1. Single-region QA cho long text (không hỏi về "region" vì user không thấy)
    2. Cross-region QA CHỈ cho title/caption + table/figure (vì user thấy được title)
    3. Không dùng fallback kiểu "what appears in region 1 and region 2"
    """
    qas: List[Dict[str, Any]] = []
    
    # === PHẦN 1: Title-Content QA cho Table/Figure ===
    # Đây là cross-region QA hợp lý vì user có thể thấy title/caption
    
    for nid, node in nodes.items():
        if len(qas) >= max_qas:
            break
            
        cat = region_category(node.get("region_type", ""))
        
        if cat in ("table", "figure"):
            # Tìm title/caption cho table/figure này
            title_node = find_title_or_header_for_region(node, nodes, adjacency)
            
            if title_node:
                qa = gen_title_content_qa(title_node, node, cat)
                qas.append(qa)
    
    # === PHẦN 2: Single-region QA cho Long Text ===
    # Chỉ hỏi về nội dung trong 1 region, không reference "region 1, region 2"
    
    long_text_nodes = [
        (nid, n) for nid, n in nodes.items() 
        if region_category(n.get("region_type", "")) == "text" 
        and is_long_text(safe_get_text(n), min_words=30)
    ]
    
    # Sort by length, lấy longest texts
    long_text_nodes.sort(key=lambda x: len(safe_get_text(x[1])), reverse=True)
    
    for nid, node in long_text_nodes[:3]:  # Tối đa 3 long text QAs
        if len(qas) >= max_qas:
            break
        
        qa = deterministic_single_region_qa(node)
        qas.append(qa)
    
    # === PHẦN 3: LLM-based QA cho title-content pairs (nếu có LLM) ===
    # Chỉ dùng LLM cho các pairs có ý nghĩa: text-table, figure-caption
    
    if llm_backend.available and len(qas) < max_qas:
        system_prompt = llm_templates.get("system_prompt", "").strip()
        
        # Tìm meaningful pairs: text gần table, figure gần text (caption)
        meaningful_pairs = []
        
        for nid, node in nodes.items():
            cat = region_category(node.get("region_type", ""))
            
            if cat == "table":
                # Tìm text node adjacent có thể là description/analysis của table
                for nb_id in adjacency.get(nid, []):
                    if nb_id in nodes:
                        nb_cat = region_category(nodes[nb_id].get("region_type", ""))
                        if nb_cat == "text" and is_long_text(safe_get_text(nodes[nb_id]), min_words=20):
                            meaningful_pairs.append((nb_id, nid, "text_table_validation"))
                            break
            
            elif cat == "figure":
                # Tìm caption cho figure
                for nb_id in adjacency.get(nid, []):
                    if nb_id in nodes:
                        nb_cat = region_category(nodes[nb_id].get("region_type", ""))
                        nb_text = safe_get_text(nodes[nb_id])
                        # Caption thường ngắn
                        if nb_cat == "text" and 5 <= len(nb_text.split()) <= 50:
                            meaningful_pairs.append((nid, nb_id, "figure_caption_mapping"))
                            break
        
        # Generate LLM QAs for meaningful pairs only
        prompt_templates = llm_templates.get("prompt_templates", {})
        
        for (a_id, b_id, template_key) in meaningful_pairs:
            if len(qas) >= max_qas:
                break
            
            if template_key not in prompt_templates:
                continue
            
            a, b = nodes[a_id], nodes[b_id]
            tmpl_obj = prompt_templates[template_key]
            tmpl_str = tmpl_obj["prompt_template"]
            
            prompt = system_prompt + "\n\n" + build_prompt_from_template(
                tmpl_str,
                a=a, b=b,
                a_type=a.get("region_type", "TextBlock"),
                b_type=b.get("region_type", "TextBlock"),
            )
            
            try:
                out_text = llm_backend.generate(prompt)
                out_json = _json_from_text(out_text) or {}
                
                if "evidence_region_ids" not in out_json:
                    out_json["evidence_region_ids"] = [a_id, b_id]
                if "evidence_quotes" not in out_json:
                    out_json["evidence_quotes"] = [safe_get_text(a)[:200], safe_get_text(b)[:200]]
                if "reasoning_type" not in out_json:
                    out_json["reasoning_type"] = tmpl_obj.get("reasoning_type", "validation")
                
                qas.append(out_json)
            except Exception:
                # Nếu LLM fail, không dùng fallback vô nghĩa
                pass
    
    return qas[:max_qas]

# -----------------------------
# Orchestrator
# -----------------------------
def generate_qas(
    doc: Dict[str, Any],
    rule_templates: Dict[str, Any],
    llm_templates: Dict[str, Any],
    max_qas: int = 10
) -> List[Dict[str, Any]]:
    nodes_list: List[Dict[str, Any]] = doc.get("nodes", [])
    edges = doc.get("edges", [])
    nodes = {int(n["node_id"]): n for n in nodes_list}
    adjacency = doc.get("adjacency") or build_adjacency(nodes, edges)

    # rule-based for form/table/figure
    rule_qas: List[Dict[str, Any]] = []
    for nid, node in nodes.items():
        cat = region_category(node.get("region_type",""))
        if cat == "form":
            rule_qas.extend(gen_form_kv_qas(node, max_qas=5))
        elif cat == "table":
            rule_qas.extend(gen_table_qas(node, max_qas=5))
        elif cat == "figure":
            rule_qas.extend(gen_figure_qas(node, max_qas=3))

    # LLM-based for long text (and multi-region)
    llm_backend = LLMBackend()
    llm_backend.try_load()
    llm_qas = generate_llm_qas(nodes, adjacency, llm_templates, llm_backend, max_qas=5)

    # merge with simple dedupe by question text
    seen = set()
    merged: List[Dict[str, Any]] = []
    for qa in rule_qas + llm_qas:
        q = normalize_space(str(qa.get("question", "")))
        if not q or q in seen:
            continue
        seen.add(q)
        merged.append(qa)
        if len(merged) >= max_qas:
            break
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=str, required=True, help="Path to document graph JSON (nodes+edges).")
    ap.add_argument("--rule_templates", type=str, required=True, help="Path to rule_based_templates.json")
    ap.add_argument("--llm_templates", type=str, required=True, help="Path to llm_prompt_templates.json")
    ap.add_argument("--out", type=str, default="qas_out.json", help="Output QA JSON path")
    ap.add_argument("--max_qas", type=int, default=10)
    args = ap.parse_args()

    doc = load_json(args.doc)
    rule_templates = load_json(args.rule_templates)
    llm_templates = load_json(args.llm_templates)

    qas = generate_qas(doc, rule_templates, llm_templates, max_qas=args.max_qas)
    save_json(qas, args.out)
    print(f"Wrote {len(qas)} QA pairs -> {args.out}")

if __name__ == "__main__":
    main()