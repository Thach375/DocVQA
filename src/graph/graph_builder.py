"""
Module xây dựng Semantic Layout Graph từ OCR regions.

Input: Regions với bounding boxes từ layout analysis
Output: Graph edges với spatial và semantic relations

Relations:
- Spatial: left_of, right_of, above, below, inside, contains
- Proximity: nearest_neighbor
- Semantic: caption_of, explains (Table↔Caption, Figure↔Caption, Text↔Table)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class GraphBuilder:
    """
    Xây dựng semantic layout graph từ regions.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.1,
        distance_threshold: float = 200.0,
        projection_threshold: float = 0.3,
        max_neighbors: int = 5,
        min_edge_score: float = 0.2
    ):
        """
        Args:
            iou_threshold: Ngưỡng IoU để xác định overlap
            distance_threshold: Khoảng cách tối đa (pixels) cho nearest_neighbor
            projection_threshold: Ngưỡng overlap projection theo trục x/y
            max_neighbors: Số lượng neighbors tối đa cho mỗi node (top-k)
            min_edge_score: Score tối thiểu để giữ edge (filtering)
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.projection_threshold = projection_threshold
        self.max_neighbors = max_neighbors
        self.min_edge_score = min_edge_score
    
    @staticmethod
    def bbox_to_xyxy(bbox: List) -> Tuple[float, float, float, float]:
        """
        Convert bbox [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> (xmin, ymin, xmax, ymax).
        """
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    @staticmethod
    def bbox_center(bbox: List) -> Tuple[float, float]:
        """Tính center (cx, cy) của bbox."""
        xmin, ymin, xmax, ymax = GraphBuilder.bbox_to_xyxy(bbox)
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)
    
    @staticmethod
    def compute_iou(bbox1: List, bbox2: List) -> float:
        """
        Tính Intersection over Union (IoU) của 2 bboxes.
        
        Returns:
            IoU score (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = GraphBuilder.bbox_to_xyxy(bbox1)
        x2_min, y2_min, x2_max, y2_max = GraphBuilder.bbox_to_xyxy(bbox2)
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def compute_projection_overlap_x(bbox1: List, bbox2: List) -> float:
        """
        Tính overlap ratio của projection lên trục X.
        
        Returns:
            Overlap ratio (0-1)
        """
        x1_min, _, x1_max, _ = GraphBuilder.bbox_to_xyxy(bbox1)
        x2_min, _, x2_max, _ = GraphBuilder.bbox_to_xyxy(bbox2)
        
        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        overlap = max(0, overlap_end - overlap_start)
        
        width1 = x1_max - x1_min
        width2 = x2_max - x2_min
        
        if width1 == 0 or width2 == 0:
            return 0.0
        
        return overlap / min(width1, width2)
    
    @staticmethod
    def compute_projection_overlap_y(bbox1: List, bbox2: List) -> float:
        """
        Tính overlap ratio của projection lên trục Y.
        
        Returns:
            Overlap ratio (0-1)
        """
        _, y1_min, _, y1_max = GraphBuilder.bbox_to_xyxy(bbox1)
        _, y2_min, _, y2_max = GraphBuilder.bbox_to_xyxy(bbox2)
        
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        overlap = max(0, overlap_end - overlap_start)
        
        height1 = y1_max - y1_min
        height2 = y2_max - y2_min
        
        if height1 == 0 or height2 == 0:
            return 0.0
        
        return overlap / min(height1, height2)
    
    @staticmethod
    def compute_center_distance(bbox1: List, bbox2: List) -> float:
        """
        Tính Euclidean distance giữa centers của 2 bboxes.
        
        Returns:
            Distance (pixels)
        """
        cx1, cy1 = GraphBuilder.bbox_center(bbox1)
        cx2, cy2 = GraphBuilder.bbox_center(bbox2)
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def compute_relation_left_of(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 nằm bên trái bbox2.
        
        Logic:
        - Center của bbox1 phải ở bên trái center của bbox2
        - Có projection overlap theo trục Y (cùng hàng ngang)
        
        Returns:
            (is_left_of, score)
        """
        cx1, cy1 = self.bbox_center(bbox1)
        cx2, cy2 = self.bbox_center(bbox2)
        
        x1_min, y1_min, x1_max, y1_max = self.bbox_to_xyxy(bbox1)
        x2_min, y2_min, x2_max, y2_max = self.bbox_to_xyxy(bbox2)
        
        # Check center position
        if cx1 >= cx2:
            return (False, 0.0)
        
        # Check projection overlap Y
        proj_overlap_y = self.compute_projection_overlap_y(bbox1, bbox2)
        
        if proj_overlap_y < self.projection_threshold:
            return (False, 0.0)
        
        # Score based on:
        # - Horizontal gap (smaller is better)
        # - Y projection overlap (higher is better)
        horizontal_gap = x2_min - x1_max
        if horizontal_gap < 0:
            horizontal_gap = 0  # Overlap case
        
        # Normalize gap (assume max gap = 500px)
        gap_score = max(0, 1.0 - horizontal_gap / 500.0)
        
        score = 0.6 * proj_overlap_y + 0.4 * gap_score
        
        return (True, score)
    
    def compute_relation_right_of(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 nằm bên phải bbox2.
        
        Returns:
            (is_right_of, score)
        """
        # Symmetric với left_of
        return self.compute_relation_left_of(bbox2, bbox1)
    
    def compute_relation_above(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 nằm phía trên bbox2.
        
        Logic:
        - Center của bbox1 phải ở phía trên center của bbox2
        - Có projection overlap theo trục X (cùng cột dọc)
        
        Returns:
            (is_above, score)
        """
        cx1, cy1 = self.bbox_center(bbox1)
        cx2, cy2 = self.bbox_center(bbox2)
        
        x1_min, y1_min, x1_max, y1_max = self.bbox_to_xyxy(bbox1)
        x2_min, y2_min, x2_max, y2_max = self.bbox_to_xyxy(bbox2)
        
        # Check center position
        if cy1 >= cy2:
            return (False, 0.0)
        
        # Check projection overlap X
        proj_overlap_x = self.compute_projection_overlap_x(bbox1, bbox2)
        
        if proj_overlap_x < self.projection_threshold:
            return (False, 0.0)
        
        # Score based on vertical gap and X projection overlap
        vertical_gap = y2_min - y1_max
        if vertical_gap < 0:
            vertical_gap = 0
        
        gap_score = max(0, 1.0 - vertical_gap / 500.0)
        
        score = 0.6 * proj_overlap_x + 0.4 * gap_score
        
        return (True, score)
    
    def compute_relation_below(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 nằm phía dưới bbox2.
        
        Returns:
            (is_below, score)
        """
        # Symmetric với above
        return self.compute_relation_above(bbox2, bbox1)
    
    def compute_relation_inside(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 nằm bên trong bbox2.
        
        Logic:
        - bbox1 phải nằm hoàn toàn trong bbox2
        - Dùng IoU và containment check
        
        Returns:
            (is_inside, score)
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox_to_xyxy(bbox1)
        x2_min, y2_min, x2_max, y2_max = self.bbox_to_xyxy(bbox2)
        
        # Check containment
        is_contained = (
            x1_min >= x2_min and
            y1_min >= y2_min and
            x1_max <= x2_max and
            y1_max <= y2_max
        )
        
        if not is_contained:
            return (False, 0.0)
        
        # Score based on area ratio
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        if area2 == 0:
            return (False, 0.0)
        
        area_ratio = area1 / area2
        
        # Smaller bbox inside larger → higher score
        score = 1.0 - area_ratio
        
        return (True, score)
    
    def compute_relation_contains(self, bbox1: List, bbox2: List) -> Tuple[bool, float]:
        """
        Kiểm tra bbox1 chứa bbox2.
        
        Returns:
            (is_contains, score)
        """
        # Symmetric với inside
        return self.compute_relation_inside(bbox2, bbox1)
    
    def compute_relation_nearest_neighbor(
        self, 
        region1: Dict, 
        region2: Dict,
        all_regions: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Kiểm tra region2 có phải nearest neighbor của region1 không.
        
        Logic:
        - Tính center distance
        - So sánh với distance đến các regions khác
        - Chỉ giữ top-k nearest
        
        Returns:
            (is_nearest, score)
        """
        bbox1 = region1['block']['bbox']
        bbox2 = region2['block']['bbox']
        
        distance = self.compute_center_distance(bbox1, bbox2)
        
        if distance > self.distance_threshold:
            return (False, 0.0)
        
        # Score inversely proportional to distance
        # Normalize by threshold
        score = max(0, 1.0 - distance / self.distance_threshold)
        
        return (True, score)
    
    def detect_caption_relation(
        self,
        region1: Dict,
        region2: Dict
    ) -> Tuple[str, float]:
        """
        Phát hiện relation caption giữa 2 regions.
        
        Logic:
        - Figure/Table + Text (short, above/below) → caption_of
        - Text (short) + Figure/Table (below/above) → is_caption_of
        
        Returns:
            (relation_type, score) hoặc (None, 0.0) nếu không phải caption
        """
        type1 = region1['region_type']
        type2 = region2['region_type']
        
        bbox1 = region1['block']['bbox']
        bbox2 = region2['block']['bbox']
        
        # Check if region is short text (potential caption)
        def is_short_text(region):
            if region['region_type'] != 'text':
                return False
            lines = region['block']['lines']
            if len(lines) > 3:  # Caption thường ngắn
                return False
            total_chars = sum(len(line['text']) for line in lines)
            return total_chars < 150  # < 150 chars
        
        # Case 1: Text → Figure/Table (text là caption)
        if is_short_text(region1) and type2 in ['figure', 'table']:
            is_above, above_score = self.compute_relation_above(bbox1, bbox2)
            is_below, below_score = self.compute_relation_below(bbox1, bbox2)
            
            if is_above:
                return ('is_caption_of', above_score * 0.9)
            elif is_below:
                return ('is_caption_of', below_score * 0.8)  # Above caption thường hơn
        
        # Case 2: Figure/Table → Text (text là caption)
        if type1 in ['figure', 'table'] and is_short_text(region2):
            is_above, above_score = self.compute_relation_above(bbox2, bbox1)
            is_below, below_score = self.compute_relation_below(bbox2, bbox1)
            
            if is_above:
                return ('has_caption', above_score * 0.9)
            elif is_below:
                return ('has_caption', below_score * 0.8)
        
        # Case 3: Table ↔ Text explanation (longer text, nearby)
        if type1 == 'table' and type2 == 'text' and not is_short_text(region2):
            distance = self.compute_center_distance(bbox1, bbox2)
            if distance < self.distance_threshold:
                score = max(0, 1.0 - distance / self.distance_threshold) * 0.7
                return ('has_explanation', score)
        
        if type1 == 'text' and type2 == 'table' and not is_short_text(region1):
            distance = self.compute_center_distance(bbox1, bbox2)
            if distance < self.distance_threshold:
                score = max(0, 1.0 - distance / self.distance_threshold) * 0.7
                return ('explains', score)
        
        return (None, 0.0)
    
    def build_graph(self, regions: List[Dict]) -> Dict[str, Any]:
        """
        Xây dựng semantic layout graph từ regions.
        
        Args:
            regions: List of regions từ layout analysis
            
        Returns:
            {
                'nodes': List[Dict],  # Regions với node_id
                'edges': List[Dict],  # Edges với relation và score
                'adjacency': Dict[int, List[int]]  # Node adjacency list
            }
        """
        # Create nodes
        nodes = []
        for idx, region in enumerate(regions):
            nodes.append({
                'node_id': idx,
                'region_type': region['region_type'],
                'bbox': region['block']['bbox'],
                'score': region['score'],
                'text': ' '.join(line['text'] for line in region['block']['lines'])[:200]
            })
        
        # Compute all potential edges
        all_edges = []
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                
                region1 = regions[i]
                region2 = regions[j]
                bbox1 = region1['block']['bbox']
                bbox2 = region2['block']['bbox']
                
                # Spatial relations
                relations_to_check = [
                    ('left_of', self.compute_relation_left_of(bbox1, bbox2)),
                    ('right_of', self.compute_relation_right_of(bbox1, bbox2)),
                    ('above', self.compute_relation_above(bbox1, bbox2)),
                    ('below', self.compute_relation_below(bbox1, bbox2)),
                    ('inside', self.compute_relation_inside(bbox1, bbox2)),
                    ('contains', self.compute_relation_contains(bbox1, bbox2)),
                ]
                
                # Add spatial edges
                for rel_type, (has_rel, score) in relations_to_check:
                    if has_rel and score >= self.min_edge_score:
                        all_edges.append({
                            'source': i,
                            'target': j,
                            'relation': rel_type,
                            'score': score,
                            'category': 'spatial'
                        })
                
                # Nearest neighbor (bidirectional)
                is_nearest, nn_score = self.compute_relation_nearest_neighbor(
                    region1, region2, regions
                )
                if is_nearest and nn_score >= self.min_edge_score:
                    all_edges.append({
                        'source': i,
                        'target': j,
                        'relation': 'nearest_neighbor',
                        'score': nn_score,
                        'category': 'proximity'
                    })
                
                # Semantic relations (caption)
                caption_rel, caption_score = self.detect_caption_relation(region1, region2)
                if caption_rel and caption_score >= self.min_edge_score:
                    all_edges.append({
                        'source': i,
                        'target': j,
                        'relation': caption_rel,
                        'score': caption_score,
                        'category': 'semantic'
                    })
        
        # Pruning: top-k neighbors per node
        edges_by_source = defaultdict(list)
        for edge in all_edges:
            edges_by_source[edge['source']].append(edge)
        
        # Keep only top-k highest scoring edges per source
        pruned_edges = []
        for source, edges in edges_by_source.items():
            # Sort by score descending
            sorted_edges = sorted(edges, key=lambda e: e['score'], reverse=True)
            # Keep top-k
            top_edges = sorted_edges[:self.max_neighbors]
            pruned_edges.extend(top_edges)
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in pruned_edges:
            adjacency[edge['source']].append(edge['target'])
        
        return {
            'nodes': nodes,
            'edges': pruned_edges,
            'adjacency': dict(adjacency)
        }
