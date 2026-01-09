"""
Module phân tích layout của document từ kết quả OCR.
Bao gồm:
- Group token → line → block
- Phát hiện region types (table, form, figure, text, layout)
- Trích xuất cấu trúc theo từng loại region
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN
import re


class DocumentLayoutAnalyzer:
    """Class phân tích layout document từ OCR results."""
    
    def __init__(
        self,
        y_overlap_threshold: float = 0.5,
        line_height_tolerance: float = 0.3,
        max_x_gap_ratio: float = 3.0,  # Tỉ lệ so với median char width
        block_vertical_gap: float = 20,  # Hard gap pixels (thay vì adaptive)
        block_x_overlap_threshold: float = 0.3
    ):
        """
        Khởi tạo analyzer.
        
        Args:
            y_overlap_threshold: Ngưỡng overlap theo y để gom token vào line
            line_height_tolerance: Độ sai lệch y_center so với median height
            max_x_gap_ratio: Tỉ lệ khoảng cách x tối đa (x median char width)
            block_vertical_gap: Khoảng cách vertical tối đa (pixels) để gom lines vào block
            block_x_overlap_threshold: Ngưỡng overlap x để gom lines vào block
        """
        self.y_overlap_threshold = y_overlap_threshold
        self.line_height_tolerance = line_height_tolerance
        self.max_x_gap_ratio = max_x_gap_ratio
        self.block_vertical_gap = block_vertical_gap
        self.block_x_overlap_threshold = block_x_overlap_threshold
    
    @staticmethod
    def bbox_overlap_y(bbox1: List, bbox2: List) -> float:
        """
        Tính overlap ratio theo trục y của 2 bbox.
        
        Args:
            bbox1, bbox2: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Overlap ratio (0-1)
        """
        # Lấy y_min, y_max
        y1_min = min(p[1] for p in bbox1)
        y1_max = max(p[1] for p in bbox1)
        y2_min = min(p[1] for p in bbox2)
        y2_max = max(p[1] for p in bbox2)
        
        # Tính overlap
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        overlap = max(0, overlap_end - overlap_start)
        
        # Height của mỗi bbox
        h1 = y1_max - y1_min
        h2 = y2_max - y2_min
        
        if h1 == 0 or h2 == 0:
            return 0.0
        
        return overlap / min(h1, h2)
    
    @staticmethod
    def bbox_center(bbox: List) -> Tuple[float, float]:
        """Tính center (x, y) của bbox."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (sum(x_coords) / 4, sum(y_coords) / 4)
    
    @staticmethod
    def bbox_bounds(bbox: List) -> Tuple[float, float, float, float]:
        """Trả về (x_min, y_min, x_max, y_max) của bbox."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    @staticmethod
    def estimate_char_width(tokens: List[Dict]) -> float:
        """
        Ước tính độ rộng trung bình của một ký tự.
        
        Args:
            tokens: List of tokens với 'box' và 'text'
            
        Returns:
            Median char width
        """
        char_widths = []
        for tok in tokens:
            if tok['box'] and tok['text']:
                x_min, _, x_max, _ = DocumentLayoutAnalyzer.bbox_bounds(tok['box'])
                width = x_max - x_min
                num_chars = len(tok['text'])
                if num_chars > 0:
                    char_widths.append(width / num_chars)
        
        return np.median(char_widths) if char_widths else 10.0
    
    @staticmethod
    def estimate_line_height(tokens: List[Dict]) -> float:
        """
        Ước tính chiều cao trung bình của line (từ tokens).
        
        Args:
            tokens: List of tokens với 'box'
            
        Returns:
            Median line height
        """
        heights = []
        for tok in tokens:
            if tok['box']:
                _, y_min, _, y_max = DocumentLayoutAnalyzer.bbox_bounds(tok['box'])
                heights.append(y_max - y_min)
        
        return np.median(heights) if heights else 20.0
    
    @staticmethod
    def detect_keyvalue_pattern(lines: List[Dict]) -> Dict:
        """
        Phát hiện pattern "key: value" trong lines.
        
        Args:
            lines: List of lines với 'text'
            
        Returns:
            {
                'has_pattern': bool,
                'keyvalue_ratio': float,  # Tỉ lệ lines có pattern
                'avg_key_length': float,  # Độ dài trung bình của key
                'keyvalue_lines': List[str]  # Danh sách các dòng có pattern
            }
        """
        if not lines:
            return {'has_pattern': False, 'keyvalue_ratio': 0.0}
        
        # Pattern: "key: value" hoặc "key : value"
        # Key thường ngắn (< 50 chars), value có thể dài hoặc ngắn
        keyvalue_pattern = re.compile(r'^([^:]{1,50}):\s*(.+)$')
        
        keyvalue_lines = []
        key_lengths = []
        
        for line in lines:
            text = line.get('text', '').strip()
            match = keyvalue_pattern.match(text)
            
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Filter: key không quá dài, value có nội dung
                if len(key) > 0 and len(key) < 50 and len(value) > 0:
                    keyvalue_lines.append(text)
                    key_lengths.append(len(key))
        
        keyvalue_ratio = len(keyvalue_lines) / len(lines) if lines else 0.0
        avg_key_length = np.mean(key_lengths) if key_lengths else 0.0
        
        # Có pattern nếu >= 40% lines là key:value
        has_pattern = (keyvalue_ratio >= 0.4 and len(keyvalue_lines) >= 2)
        
        return {
            'has_pattern': has_pattern,
            'keyvalue_ratio': keyvalue_ratio,
            'avg_key_length': avg_key_length,
            'keyvalue_lines': keyvalue_lines
        }
    
    def group_tokens_to_lines(self, ocr_result: Dict[str, Any]) -> List[Dict]:
        """
        Gom các token thành lines dựa trên y-overlap và x-distance.
        
        Args:
            ocr_result: Kết quả OCR với 'details' chứa tokens
            
        Returns:
            List of lines, mỗi line có {text, bbox, tokens, token_ids}
        """
        if not ocr_result['success'] or not ocr_result['details']:
            return []
        
        tokens = ocr_result['details']
        n_tokens = len(tokens)
        
        # Tính median height và char width để làm reference
        median_height = self.estimate_line_height(tokens)
        median_char_width = self.estimate_char_width(tokens)
        
        # Tính max_x_gap dựa trên char width
        max_x_gap = self.max_x_gap_ratio * median_char_width
        
        # Dùng union-find để gom tokens
        parent = list(range(n_tokens))
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj
        
        # Merge tokens if they satisfy line conditions
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if not tokens[i]['box'] or not tokens[j]['box']:
                    continue
                
                # Check y-overlap
                y_overlap = self.bbox_overlap_y(tokens[i]['box'], tokens[j]['box'])
                
                # Check y-center difference
                cx_i, cy_i = self.bbox_center(tokens[i]['box'])
                cx_j, cy_j = self.bbox_center(tokens[j]['box'])
                y_diff = abs(cy_i - cy_j)
                
                # Check x-distance
                x_i_min, _, x_i_max, _ = self.bbox_bounds(tokens[i]['box'])
                x_j_min, _, x_j_max, _ = self.bbox_bounds(tokens[j]['box'])
                x_gap = min(abs(x_i_max - x_j_min), abs(x_j_max - x_i_min))
                
                # Merge condition
                if (y_overlap >= self.y_overlap_threshold or 
                    y_diff < self.line_height_tolerance * median_height) and \
                   x_gap < max_x_gap:
                    union(i, j)
        
        # Group tokens by component
        groups = defaultdict(list)
        for i in range(n_tokens):
            groups[find(i)].append(i)
        
        # Create lines
        lines = []
        for group_indices in groups.values():
            line_tokens = [tokens[i] for i in sorted(group_indices, 
                                                     key=lambda idx: self.bbox_center(tokens[idx]['box'])[0])]
            
            # Merge text
            line_text = ' '.join(tok['text'] for tok in line_tokens)
            
            # Merge bbox
            all_coords = []
            for tok in line_tokens:
                if tok['box']:
                    all_coords.extend(tok['box'])
            
            if all_coords:
                x_coords = [p[0] for p in all_coords]
                y_coords = [p[1] for p in all_coords]
                line_bbox = [
                    [min(x_coords), min(y_coords)],
                    [max(x_coords), min(y_coords)],
                    [max(x_coords), max(y_coords)],
                    [min(x_coords), max(y_coords)]
                ]
            else:
                line_bbox = None
            
            lines.append({
                'text': line_text,
                'bbox': line_bbox,
                'tokens': line_tokens,
                'token_ids': group_indices
            })
        
        # Sort lines top to bottom
        lines.sort(key=lambda line: self.bbox_center(line['bbox'])[1] if line['bbox'] else 0)
        
        return lines
    
    def group_lines_to_blocks(self, lines: List[Dict]) -> List[Dict]:
        """
        Gom các lines thành blocks dựa trên vertical gap và x-overlap.
        
        Args:
            lines: List of lines từ group_tokens_to_lines
            
        Returns:
            List of blocks, mỗi block có {lines, bbox, line_ids}
        """
        if not lines:
            return []
        
        n_lines = len(lines)
        
        # Sử dụng hard gap thay vì adaptive gap (tránh nhiễu từ bbox kích thước khác nhau)
        vertical_gap_threshold = self.block_vertical_gap  # Fixed 20 pixels
        
        # Tính median line width để xác định x1_close threshold
        line_widths = []
        for line in lines:
            if line['bbox']:
                x_min, _, x_max, _ = self.bbox_bounds(line['bbox'])
                line_widths.append(x_max - x_min)
        
        median_line_width = np.median(line_widths) if line_widths else 100.0
        x1_close_threshold = 0.05 * median_line_width  # 5% of median width
        
        # Build adjacency graph
        adj = defaultdict(list)
        
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                if not lines[i]['bbox'] or not lines[j]['bbox']:
                    continue
                
                # Check vertical distance
                xi_min, yi_min, xi_max, yi_max = self.bbox_bounds(lines[i]['bbox'])
                xj_min, yj_min, xj_max, yj_max = self.bbox_bounds(lines[j]['bbox'])
                
                v_gap = min(abs(yi_max - yj_min), abs(yj_max - yi_min))
                
                # Check x-overlap
                x_overlap_start = max(xi_min, xj_min)
                x_overlap_end = min(xi_max, xj_max)
                x_overlap = max(0, x_overlap_end - x_overlap_start)
                x_overlap_ratio = x_overlap / min(xi_max - xi_min, xj_max - xj_min)
                
                # Also check if x1 positions are close (for aligned text)
                x1_close = abs(xi_min - xj_min) < x1_close_threshold
                
                # Connect if close vertically and have x-overlap or aligned
                if v_gap < vertical_gap_threshold and \
                   (x_overlap_ratio >= self.block_x_overlap_threshold or x1_close):
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find connected components using DFS
        visited = [False] * n_lines
        blocks = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n_lines):
            if not visited[i]:
                component = []
                dfs(i, component)
                
                block_lines = [lines[idx] for idx in sorted(component, 
                                                            key=lambda idx: self.bbox_center(lines[idx]['bbox'])[1])]
                
                # Merge bbox
                all_coords = []
                for line in block_lines:
                    if line['bbox']:
                        all_coords.extend(line['bbox'])
                
                if all_coords:
                    x_coords = [p[0] for p in all_coords]
                    y_coords = [p[1] for p in all_coords]
                    block_bbox = [
                        [min(x_coords), min(y_coords)],
                        [max(x_coords), min(y_coords)],
                        [max(x_coords), max(y_coords)],
                        [min(x_coords), max(y_coords)]
                    ]
                else:
                    block_bbox = None
                
                # Phát hiện key:value pattern trong block
                keyvalue_info = self.detect_keyvalue_pattern(block_lines)
                
                blocks.append({
                    'lines': block_lines,
                    'bbox': block_bbox,
                    'line_ids': component,
                    'keyvalue_pattern': keyvalue_info  # Metadata cho form detection
                })
        
        return blocks
    
    def detect_table_region(self, block: Dict) -> Dict:
        """
        Phát hiện table region từ block.
        
        Returns:
            {
                'is_table': bool,
                'score': float,
                'num_cols': int,
                'col_stability': float,
                'row_spacing_var': float
            }
        """
        lines = block['lines']
        if len(lines) < 2:
            return {'is_table': False, 'score': 0.0}
        
        # Collect all token x_centers
        x_centers = []
        for line in lines:
            for tok in line['tokens']:
                if tok['box']:
                    cx, _ = self.bbox_center(tok['box'])
                    x_centers.append(cx)
        
        if len(x_centers) < 5:
            return {'is_table': False, 'score': 0.0}
        
        # Tính median char width để xác định eps cho clustering
        all_tokens = []
        for line in lines:
            all_tokens.extend(line['tokens'])
        median_char_width = self.estimate_char_width(all_tokens)
        clustering_eps = max(30, 3 * median_char_width)  # Adaptive eps
        
        # Cluster x_centers to find columns
        X = np.array(x_centers).reshape(-1, 1)
        clustering = DBSCAN(eps=clustering_eps, min_samples=2).fit(X)
        labels = clustering.labels_
        num_cols_est = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate col_stability: % of lines where tokens align with column clusters
        col_centers = []
        for label in set(labels):
            if label != -1:
                cluster_points = X[labels == label]
                col_centers.append(np.mean(cluster_points))
        
        col_centers = sorted(col_centers)
        
        # Tính token alignment threshold dựa trên char width
        alignment_threshold = max(30, 3 * median_char_width)
        
        aligned_lines = 0
        for line in lines:
            token_cols = []
            for tok in line['tokens']:
                if tok['box']:
                    cx, _ = self.bbox_center(tok['box'])
                    # Find closest column
                    if col_centers:
                        closest_dist = min(abs(cx - cc) for cc in col_centers)
                        if closest_dist < alignment_threshold:
                            token_cols.append(True)
            
            if len(token_cols) >= 2:  # At least 2 tokens aligned
                aligned_lines += 1
        
        col_stability = aligned_lines / len(lines) if lines else 0.0
        
        # Calculate row spacing variance
        y_centers = [self.bbox_center(line['bbox'])[1] for line in lines if line['bbox']]
        if len(y_centers) >= 2:
            y_gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
            row_spacing_var = np.std(y_gaps) / np.mean(y_gaps) if np.mean(y_gaps) > 0 else 1.0
        else:
            row_spacing_var = 1.0
        
        # Table score
        is_table = (num_cols_est >= 3 and col_stability >= 0.6 and row_spacing_var < 0.3)
        score = (num_cols_est / 10.0) * col_stability * (1 - row_spacing_var)
        
        return {
            'is_table': is_table,
            'score': score,
            'num_cols': num_cols_est,
            'col_stability': col_stability,
            'row_spacing_var': row_spacing_var
        }
    
    def detect_form_region(self, block: Dict) -> Dict:
        """
        Phát hiện form region (key:value pairs) từ block.
        
        Returns:
            {
                'is_form': bool,
                'score': float,
                'colon_ratio': float,
                'left_alignment': float,
                'right_alignment': float,
                'avg_line_length': float,
                'line_length_variance': float
            }
        """
        lines = block['lines']
        if len(lines) < 2:
            return {'is_form': False, 'score': 0.0}
        
        # ===== SỬ DỤNG KEY:VALUE PATTERN TỪ METADATA =====
        keyvalue_info = block.get('keyvalue_pattern', {})
        has_keyvalue_pattern = keyvalue_info.get('has_pattern', False)
        keyvalue_ratio = keyvalue_info.get('keyvalue_ratio', 0.0)
        
        # Nếu phát hiện pattern rõ ràng → XÁC ĐỊNH FORM NGAY
        if has_keyvalue_pattern:
            return {
                'is_form': True,
                'score': 0.9 + keyvalue_ratio * 0.1,  # High score
                'text_boost_score': -0.3,  # Penalty cho text
                'keyvalue_ratio': keyvalue_ratio,
                'has_keyvalue_pattern': True,
                'colon_ratio': keyvalue_ratio,
                'left_alignment': 1.0,  # Assumed
                'right_alignment': 0.0,
                'avg_line_length': keyvalue_info.get('avg_key_length', 0.0),
                'line_length_variance': 0.0,
                'short_line_ratio': 0.0,
                'long_line_ratio': 0.0,
                'is_justified_text': False
            }
        
        # ===== LOGIC CŨ: Phân tích alignment và statistics =====
        # Tận dụng keyvalue_ratio từ metadata thay vì đếm lại
        colon_ratio = keyvalue_ratio  # Đã được tính trong detect_keyvalue_pattern
        
        # Check left alignment (keys)
        left_x = [self.bbox_bounds(line['bbox'])[0] for line in lines if line['bbox']]
        left_alignment = 0.0
        if len(left_x) >= 2:
            left_std = np.std(left_x)
            left_mean = np.mean(left_x)
            left_alignment = 1.0 - min(1.0, left_std / max(left_mean, 1.0))
        
        # Check right alignment (values)
        right_x = [self.bbox_bounds(line['bbox'])[2] for line in lines if line['bbox']]
        right_alignment = 0.0
        if len(right_x) >= 2:
            right_std = np.std(right_x)
            right_mean = np.mean(right_x)
            right_alignment = 1.0 - min(1.0, right_std / max(right_mean, 1.0))
        
        # ===== PHÂN BIỆT FORM vs JUSTIFIED TEXT =====
        # 1. Đo độ dài dòng (characters)
        line_lengths = [len(line['text']) for line in lines]
        avg_line_length = np.mean(line_lengths)
        line_length_variance = np.std(line_lengths) / max(np.mean(line_lengths), 1.0)
        
        # 2. Đo độ rộng bbox của dòng (pixels)
        line_widths = []
        for line in lines:
            if line['bbox']:
                x_min, _, x_max, _ = self.bbox_bounds(line['bbox'])
                line_widths.append(x_max - x_min)
        
        avg_line_width = np.mean(line_widths) if line_widths else 0
        line_width_variance = np.std(line_widths) / max(np.mean(line_widths), 1.0) if line_widths else 0
        
        # 3. Kiểm tra pattern của form: nhiều dòng ngắn
        short_lines = sum(1 for l in line_lengths if l < 40)  # Dòng < 40 ký tự
        short_line_ratio = short_lines / len(lines) if lines else 0
        
        # 4. Kiểm tra có nhiều dòng dài (văn bản thông thường)
        long_lines = sum(1 for l in line_lengths if l > 60)  # Dòng > 60 ký tự
        long_line_ratio = long_lines / len(lines) if lines else 0
        
        # ===== LOGIC PHÂN LOẠI =====
        # Nếu cả left và right alignment cao + nhiều dòng dài → Justified Text, không phải Form
        is_justified_text = (
            left_alignment >= 0.8 and 
            right_alignment >= 0.7 and 
            long_line_ratio >= 0.5 and  # Hơn 50% dòng dài
            avg_line_length > 50  # Dòng trung bình dài
        )
        
        # Form thật: ngắn, đứt gãy, variance cao (key ngắn, value dài bất đối xứng)
        # QUAN TRỌNG: Form phải có ít nhất một chút dấu ":" để tránh nhầm với địa chỉ/contact info
        is_true_form = (
            colon_ratio >= 0.25 and  # Nhiều dấu ":" (giảm từ 0.3)
            short_line_ratio >= 0.3 and  # Nhiều dòng ngắn (giảm từ 0.4)
            line_length_variance > 0.25  # Độ dài không đều (giảm từ 0.3)
        ) or (
            colon_ratio >= 0.1 and  # Ít nhất 10% lines có ":" (giảm từ 0.15 để bắt form nhỏ)
            left_alignment >= 0.65 and  # Giảm từ 0.7
            right_alignment < 0.65 and  # Right KHÔNG thẳng (khác justified)
            short_line_ratio >= 0.25  # Giảm từ 0.3
        )
        
        # Form score và Text boost
        text_boost_score = 0.0  # Điểm cộng cho text region
        
        if is_justified_text:
            # Penalty for form, boost for text
            is_form = False
            score = 0.0
            # Cộng điểm cho text region
            text_boost_score = (
                (left_alignment + right_alignment) / 2 * 0.4 +
                long_line_ratio * 0.3 +
                min(1.0, avg_line_length / 100.0) * 0.3
            )
        elif is_true_form:
            is_form = True
            # Score tăng với colon_ratio, short_line_ratio, line_length_variance
            score = (
                colon_ratio * 0.4 + 
                short_line_ratio * 0.3 + 
                line_length_variance * 0.2 +
                left_alignment * 0.1
            )
            # Giảm điểm cho text region
            text_boost_score = -0.2
        else:
            is_form = False
            score = colon_ratio * 0.3 + left_alignment * 0.2
            text_boost_score = 0.0
        
        return {
            'is_form': is_form,
            'score': score,
            'text_boost_score': text_boost_score,  # Điểm cộng cho text
            'has_keyvalue_pattern': False,  # Không có pattern rõ ràng
            'keyvalue_ratio': colon_ratio,  # Fallback to colon_ratio
            'colon_ratio': colon_ratio,
            'left_alignment': left_alignment,
            'right_alignment': right_alignment,
            'avg_line_length': avg_line_length,
            'line_length_variance': line_length_variance,
            'short_line_ratio': short_line_ratio,
            'long_line_ratio': long_line_ratio,
            'is_justified_text': is_justified_text
        }
    
    def detect_figure_region(self, block: Dict) -> Dict:
        """
        Phát hiện figure region (chart/plot) từ block.
        
        Returns:
            {
                'is_figure': bool,
                'score': float,
                'empty_center_ratio': float,
                'tick_like_numbers': int,
                'legend_cluster': bool
            }
        """
        lines = block['lines']
        if not block['bbox']:
            return {'is_figure': False, 'score': 0.0}
        
        x_min, y_min, x_max, y_max = self.bbox_bounds(block['bbox'])
        width = x_max - x_min
        height = y_max - y_min
        
        if width == 0 or height == 0:
            return {'is_figure': False, 'score': 0.0}
        
        # Create grid to check empty center
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))
        
        for line in lines:
            for tok in line['tokens']:
                if tok['box']:
                    tx_min, ty_min, tx_max, ty_max = self.bbox_bounds(tok['box'])
                    # Map to grid
                    gx1 = int((tx_min - x_min) / width * grid_size)
                    gy1 = int((ty_min - y_min) / height * grid_size)
                    gx2 = int((tx_max - x_min) / width * grid_size)
                    gy2 = int((ty_max - y_min) / height * grid_size)
                    
                    for gx in range(max(0, gx1), min(grid_size, gx2+1)):
                        for gy in range(max(0, gy1), min(grid_size, gy2+1)):
                            grid[gy, gx] = 1
        
        # Check center area
        center_start = grid_size // 4
        center_end = 3 * grid_size // 4
        center_area = grid[center_start:center_end, center_start:center_end]
        empty_center_ratio = 1.0 - (np.sum(center_area) / center_area.size)
        
        # Count tick-like numbers (short numeric tokens)
        tick_like_numbers = 0
        for line in lines:
            for tok in line['tokens']:
                if re.match(r'^-?\d+\.?\d*$', tok['text'].strip()) and len(tok['text'].strip()) <= 6:
                    tick_like_numbers += 1
        
        # Check for legend cluster (tokens stacked vertically on right)
        right_tokens = []
        for line in lines:
            for tok in line['tokens']:
                if tok['box']:
                    tx_min, _, _, _ = self.bbox_bounds(tok['box'])
                    if tx_min > x_min + 0.7 * width:
                        right_tokens.append(tok)
        
        legend_cluster = len(right_tokens) >= 3
        
        # Figure score
        is_figure = (empty_center_ratio >= 0.3 and (tick_like_numbers >= 5 or legend_cluster))
        score = empty_center_ratio * 0.5 + min(1.0, tick_like_numbers / 10.0) * 0.3 + (0.2 if legend_cluster else 0.0)
        
        return {
            'is_figure': is_figure,
            'score': score,
            'empty_center_ratio': empty_center_ratio,
            'tick_like_numbers': tick_like_numbers,
            'legend_cluster': legend_cluster
        }
    
    def detect_text_region(self, block: Dict, form_boost: float = 0.0) -> Dict:
        """
        Phát hiện running text region từ block.
        
        Args:
            block: Block để phân tích
            form_boost: Điểm cộng từ form detection (nếu phát hiện justified text)
        
        Returns:
            {
                'is_text': bool,
                'score': float,
                'avg_line_length': float,
                'spacing_uniformity': float,
                'justified_boost': float
            }
        """
        lines = block['lines']
        if len(lines) < 2:
            return {'is_text': False, 'score': 0.0}
        
        # Average line length (in characters)
        line_lengths = [len(line['text']) for line in lines]
        avg_line_length = np.mean(line_lengths)
        
        # Check spacing uniformity
        y_centers = [self.bbox_center(line['bbox'])[1] for line in lines if line['bbox']]
        if len(y_centers) >= 2:
            y_gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
            spacing_uniformity = 1.0 - min(1.0, np.std(y_gaps) / max(np.mean(y_gaps), 1.0))
        else:
            spacing_uniformity = 0.0
        
        # Check for few columns (not table-like)
        x_centers = []
        for line in lines:
            for tok in line['tokens']:
                if tok['box']:
                    cx, _ = self.bbox_center(tok['box'])
                    x_centers.append(cx)
        
        if x_centers:
            X = np.array(x_centers).reshape(-1, 1)
            # Tính adaptive eps dựa trên char width
            all_tokens = []
            for line in lines:
                all_tokens.extend(line['tokens'])
            median_char_width = self.estimate_char_width(all_tokens)
            clustering_eps = max(30, 3 * median_char_width)
            
            clustering = DBSCAN(eps=clustering_eps, min_samples=2).fit(X)
            num_cols = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            few_columns = (num_cols <= 2)
        else:
            few_columns = True
        
        # ===== DETECT JUSTIFIED TEXT PATTERN =====
        # Check alignment (từ form detection logic)
        left_x = [self.bbox_bounds(line['bbox'])[0] for line in lines if line['bbox']]
        right_x = [self.bbox_bounds(line['bbox'])[2] for line in lines if line['bbox']]
        
        left_alignment = 0.0
        if len(left_x) >= 2:
            left_std = np.std(left_x)
            left_mean = np.mean(left_x)
            left_alignment = 1.0 - min(1.0, left_std / max(left_mean, 1.0))
        
        right_alignment = 0.0
        if len(right_x) >= 2:
            right_std = np.std(right_x)
            right_mean = np.mean(right_x)
            right_alignment = 1.0 - min(1.0, right_std / max(right_mean, 1.0))
        
        long_lines = sum(1 for l in line_lengths if l > 60)
        long_line_ratio = long_lines / len(lines) if lines else 0
        
        # Justified text boost
        justified_boost = 0.0
        if (left_alignment >= 0.8 and right_alignment >= 0.7 and 
            long_line_ratio >= 0.5 and avg_line_length > 50):
            justified_boost = 0.3  # Cộng điểm cho text
        
        # Text score (base + justified boost + form boost)
        base_score = min(1.0, avg_line_length / 100.0) * 0.5 + spacing_uniformity * 0.5
        score = base_score + justified_boost + form_boost
        
        is_text = (avg_line_length >= 30 and spacing_uniformity >= 0.6 and few_columns) or \
                  (justified_boost > 0) or (form_boost > 0)
        
        return {
            'is_text': is_text,
            'score': min(1.0, score),  # Cap at 1.0
            'avg_line_length': avg_line_length,
            'spacing_uniformity': spacing_uniformity,
            'justified_boost': justified_boost,
            'form_boost': form_boost
        }
    
    def analyze_layout(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích toàn bộ layout của document.
        
        Args:
            ocr_result: Kết quả OCR
            
        Returns:
            {
                'lines': List[Dict],
                'blocks': List[Dict],
                'regions': List[Dict] - với region type và metadata
            }
        """
        # Stage I: Neutral grouping
        lines = self.group_tokens_to_lines(ocr_result)
        blocks = self.group_lines_to_blocks(lines)
        
        # Stage II: Region detection
        regions = []
        
        for block_idx, block in enumerate(blocks):
            # Detect all hypotheses
            table_result = self.detect_table_region(block)
            form_result = self.detect_form_region(block)
            figure_result = self.detect_figure_region(block)
            
            # Truyền form boost vào text detection
            form_boost = form_result.get('text_boost_score', 0.0)
            text_result = self.detect_text_region(block, form_boost=form_boost)
            
            # Create region hypotheses
            hypotheses = [
                {'type': 'table', 'score': table_result['score'], 'metadata': table_result},
                {'type': 'form', 'score': form_result['score'], 'metadata': form_result},
                {'type': 'figure', 'score': figure_result['score'], 'metadata': figure_result},
                {'type': 'text', 'score': text_result['score'], 'metadata': text_result},
            ]
            
            # Sort by score and keep top-2
            hypotheses.sort(key=lambda h: h['score'], reverse=True)
            top_hypotheses = hypotheses[:2]
            
            # Minimum score thresholds cho từng loại region
            min_scores = {
                'table': 0.25,   # Table cần độ tin cậy cao (nhiều columns, alignment)
                'form': 0.20,    # Form giảm từ 0.25 → 0.20 để bắt form nhỏ hơn
                'figure': 0.25,  # Figure cần empty center hoặc ticks
                'text': 0.65     # Text dễ phát hiện hơn, threshold thấp hơn
            }
            
            # Add to regions
            for hyp in top_hypotheses:
                min_score = min_scores.get(hyp['type'], 0.2)  # Default 0.2
                if hyp['score'] >= min_score:
                    regions.append({
                        'block_id': block_idx,
                        'block': block,
                        'region_type': hyp['type'],
                        'score': hyp['score'],
                        'metadata': hyp['metadata']
                    })
        
        return {
            'lines': lines,
            'blocks': blocks,
            'regions': regions
        }
