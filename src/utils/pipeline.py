"""
Full pipeline processor: OCR → Layout → Graph → JSON Export
"""

import cv2
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class FullPipelineProcessor:
    """
    Processor chạy toàn bộ pipeline cho 1 image:
    OCR → Layout Analysis → Graph Building → JSON Export
    """
    
    def __init__(self, ocr_processor, layout_analyzer, graph_builder):
        """
        Args:
            ocr_processor: PaddleOCRProcessor instance
            layout_analyzer: LayoutAnalyzer instance
            graph_builder: GraphBuilder instance
        """
        self.ocr_processor = ocr_processor
        self.layout_analyzer = layout_analyzer
        self.graph_builder = graph_builder
    
    def process_image(
        self, 
        image_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process một image qua toàn bộ pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Path to save JSON output (optional)
        
        Returns:
            dict: Complete pipeline result với OCR, Layout, Graph data
        """
        try:
            # 1. Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    'success': False,
                    'error': 'Cannot load image',
                    'image_path': str(image_path)
                }
            
            image_id = Path(image_path).stem
            
            # 2. Run OCR
            ocr_result = self.ocr_processor.run_ocr_with_layout(image)
            
            if not ocr_result.get('success', False):
                return {
                    'success': False,
                    'error': 'OCR failed',
                    'image_id': image_id,
                    'image_path': str(image_path)
                }
            
            # 3. Layout Analysis
            layout_result = self.layout_analyzer.analyze_layout(ocr_result)
            
            regions = layout_result.get('regions', [])
            
            # 4. Build Graph (only if we have regions)
            if len(regions) > 0:
                graph_result = self.graph_builder.build_graph(regions)
            else:
                graph_result = {
                    'nodes': [],
                    'edges': [],
                    'adjacency': {}
                }
            
            # 5. Prepare output data
            output_data = {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'image_id': image_id,
                'image_path': str(image_path),
                'ocr': {
                    'success': ocr_result.get('success', False),
                    'num_tokens': len(ocr_result.get('details', [])),
                    'processing_time_ms': ocr_result.get('processing_time_ms', 0),
                    'tokens': ocr_result.get('details', [])
                },
                'layout': {
                    'num_lines': len(layout_result.get('lines', [])),
                    'num_blocks': len(layout_result.get('blocks', [])),
                    'num_regions': len(regions),
                    'lines': layout_result.get('lines', []),
                    'blocks': layout_result.get('blocks', []),
                    'regions': regions
                },
                'graph': {
                    'num_nodes': len(graph_result['nodes']),
                    'num_edges': len(graph_result['edges']),
                    'nodes': graph_result['nodes'],
                    'edges': graph_result['edges'],
                    'adjacency': graph_result['adjacency']
                }
            }
            
            # 6. Prepare result (BEFORE saving to catch JSON errors)
            result = {
                'success': True,
                'image_id': image_id,
                'num_tokens': output_data['ocr']['num_tokens'],
                'num_regions': output_data['layout']['num_regions'],
                'num_edges': output_data['graph']['num_edges'],
                'data': output_data
            }
            
            # 7. Save to JSON if output path provided
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                except Exception as save_error:
                    # If save fails, still return the error
                    return {
                        'success': False,
                        'error': f'Failed to save JSON: {str(save_error)}',
                        'image_id': image_id,
                        'image_path': str(image_path)
                    }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_path': str(image_path)
            }
