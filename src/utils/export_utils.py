"""
Utilities để export dữ liệu OCR và layout analysis ra JSON.
"""

import json
from typing import Dict, Any
from datetime import datetime


def export_ocr_to_json(
    ocr_result: Dict[str, Any],
    layout_result: Dict[str, Any],
    graph_result: Dict[str, Any],
    output_path: str,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Export OCR results, layout analysis, và graph ra file JSON.
    
    Args:
        ocr_result: Kết quả từ OCR engine
        layout_result: Kết quả từ layout analyzer (lines, blocks, regions)
        graph_result: Kết quả từ graph builder (nodes, edges)
        output_path: Đường dẫn file output
        metadata: Metadata bổ sung (image info, timestamps, etc.)
    """
    output_data = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'metadata': metadata or {},
        'ocr': {
            'success': ocr_result.get('success', False),
            'num_tokens': len(ocr_result.get('details', [])),
            'processing_time_ms': ocr_result.get('processing_time_ms', 0),
            'tokens': ocr_result.get('details', [])
        },
        'layout': {
            'num_lines': len(layout_result.get('lines', [])),
            'num_blocks': len(layout_result.get('blocks', [])),
            'num_regions': len(layout_result.get('regions', [])),
            'lines': layout_result.get('lines', []),
            'blocks': layout_result.get('blocks', []),
            'regions': layout_result.get('regions', [])
        },
        'graph': {
            'num_nodes': len(graph_result.get('nodes', [])),
            'num_edges': len(graph_result.get('edges', [])),
            'nodes': graph_result.get('nodes', []),
            'edges': graph_result.get('edges', []),
            'adjacency': graph_result.get('adjacency', {})
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported to: {output_path}")
    print(f"   - OCR tokens: {output_data['ocr']['num_tokens']}")
    print(f"   - Layout lines: {output_data['layout']['num_lines']}")
    print(f"   - Layout blocks: {output_data['layout']['num_blocks']}")
    print(f"   - Layout regions: {output_data['layout']['num_regions']}")
    print(f"   - Graph nodes: {output_data['graph']['num_nodes']}")
    print(f"   - Graph edges: {output_data['graph']['num_edges']}")


def export_graph_only(
    graph_result: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export chỉ graph ra file JSON (lightweight).
    
    Args:
        graph_result: Kết quả từ graph builder
        output_path: Đường dẫn file output
    """
    output_data = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'nodes': graph_result.get('nodes', []),
        'edges': graph_result.get('edges', []),
        'adjacency': graph_result.get('adjacency', {})
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported graph to: {output_path}")
    print(f"   - Nodes: {len(output_data['nodes'])}")
    print(f"   - Edges: {len(output_data['edges'])}")
