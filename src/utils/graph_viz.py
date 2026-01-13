"""
Graph visualization utilities.
Provides functions to visualize semantic layout graph from DocVQA pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
from typing import Dict, List, Any, Optional
import numpy as np
import json
from pathlib import Path
from datetime import datetime


# ==================== COLOR SCHEMES ====================

REGION_COLORS = {
    'TextBlock': '#3498db',  # Blue
    'Table': '#e74c3c',      # Red
    'Figure': '#2ecc71',     # Green
    'Form': '#f39c12'        # Orange
}

RELATION_COLORS = {
    'left_of': '#9b59b6',
    'right_of': '#e91e63',
    'above': '#3f51b5',
    'below': '#00bcd4',
    'inside': '#4caf50',
    'contains': '#8bc34a',
    'nearest_neighbor': '#ff9800',
    'is_caption_of': '#ff5722',
    'has_caption': '#f44336',
    'explains': '#795548',
    'has_explanation': '#607d8b'
}


# ==================== DATA HELPERS ====================

def load_regions_from_json(json_path: str) -> List[Dict]:
    """
    Load regions from OCR output JSON.
    
    Args:
        json_path: Path to JSON file with OCR + layout results
        
    Returns:
        List of region dicts
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    regions = []
    if 'layout' in data and 'regions' in data['layout']:
        regions = data['layout']['regions']
    elif 'regions' in data:
        regions = data['regions']
    
    print(f"✅ Loaded {len(regions)} regions from {json_path}")
    return regions


def create_demo_regions() -> List[Dict]:
    """
    Create sample regions for testing visualization.
    
    Returns:
        List of demo region dicts matching GraphBuilder format
    """
    regions = [
        {
            'region_type': 'TextBlock',
            'block': {
                'bbox': [[50, 50], [300, 50], [300, 100], [50, 100]],
                'lines': [{'text': 'Document Title: Annual Report 2024'}]
            },
            'score': 0.95
        },
        {
            'region_type': 'TextBlock',
            'block': {
                'bbox': [[50, 120], [200, 120], [200, 160], [50, 160]],
                'lines': [{'text': 'Table 1: Quarterly Sales Figures'}]
            },
            'score': 0.92
        },
        {
            'region_type': 'Table',
            'block': {
                'bbox': [[50, 180], [350, 180], [350, 350], [50, 350]],
                'lines': [{'text': 'Q1: $1.2M | Q2: $1.5M | Q3: $1.8M | Q4: $2.1M'}]
            },
            'score': 0.88
        },
        {
            'region_type': 'TextBlock',
            'block': {
                'bbox': [[400, 50], [550, 50], [550, 100], [400, 100]],
                'lines': [{'text': 'Figure 1: Sales Growth Chart'}]
            },
            'score': 0.90
        },
        {
            'region_type': 'Figure',
            'block': {
                'bbox': [[400, 120], [650, 120], [650, 350], [400, 350]],
                'lines': [{'text': '[Chart Image]'}]
            },
            'score': 0.85
        },
        {
            'region_type': 'TextBlock',
            'block': {
                'bbox': [[50, 380], [650, 380], [650, 450], [50, 450]],
                'lines': [{'text': 'The sales figures show consistent growth across all quarters...'}]
            },
            'score': 0.93
        }
    ]
    
    print(f"✅ Created {len(regions)} demo regions")
    return regions


# ==================== VISUALIZATION FUNCTIONS ====================

def visualize_graph_basic(nodes: List[Dict], edges: List[Dict], figsize: tuple = (14, 10)):
    """
    Visualize the graph with all nodes (bboxes) and edges (arrows).
    
    Args:
        nodes: List of node dicts from graph_result['nodes']
        edges: List of edge dicts from graph_result['edges']
        figsize: Figure size tuple (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate canvas size from node bboxes
    max_x = max_y = 0
    for node in nodes:
        bbox = node['bbox']  # 4-point format [[x1,y1], ...]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        max_x = max(max_x, max(x_coords))
        max_y = max(max_y, max(y_coords))
    
    ax.set_xlim(0, max_x + 50)
    ax.set_ylim(max_y + 50, 0)  # Invert y-axis (image coordinates)
    
    # Draw nodes (regions)
    for node in nodes:
        bbox = node['bbox']  # 4-point format
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        width = x2 - x1
        height = y2 - y1
        
        color = REGION_COLORS.get(node['region_type'], '#95a5a6')
        
        # Draw rectangle
        rect = Rectangle((x1, y1), width, height, 
                         linewidth=2, edgecolor=color, 
                         facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Add node label
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ax.text(center_x, center_y, f"#{node['node_id']}\n{node['region_type']}", 
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw edges
    for edge in edges:
        src_idx = edge['source']
        tgt_idx = edge['target']
        relation = edge['relation']
        score = edge['score']
        
        # Get node bboxes
        src_node = nodes[src_idx]
        tgt_node = nodes[tgt_idx]
        
        src_bbox = src_node['bbox']
        tgt_bbox = tgt_node['bbox']
        
        # Calculate centers
        src_x = sum(p[0] for p in src_bbox) / 4
        src_y = sum(p[1] for p in src_bbox) / 4
        tgt_x = sum(p[0] for p in tgt_bbox) / 4
        tgt_y = sum(p[1] for p in tgt_bbox) / 4
        
        color = RELATION_COLORS.get(relation, '#34495e')
        
        # Draw arrow
        arrow = FancyArrowPatch((src_x, src_y), (tgt_x, tgt_y),
                               arrowstyle='->', mutation_scale=15,
                               color=color, alpha=0.6, linewidth=1.5,
                               linestyle='--' if score < 0.5 else '-')
        ax.add_patch(arrow)
    
    ax.set_aspect('equal')
    ax.set_title('Semantic Layout Graph - All Nodes and Edges', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Create legend for region types
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor=color, label=rtype, alpha=0.6)
                      for rtype, color in REGION_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def visualize_by_relation_type(
    nodes: List[Dict], 
    edges: List[Dict], 
    relation_types: List[str], 
    figsize: tuple = (14, 10)
):
    """
    Visualize only specific relation types with edge labels.
    
    Args:
        nodes: List of node dicts from graph_result['nodes']
        edges: List of edge dicts from graph_result['edges']
        relation_types: List of relation types to display
        figsize: Figure size tuple (width, height)
    """
    filtered_edges = [e for e in edges if e['relation'] in relation_types]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate canvas size
    max_x = max_y = 0
    for node in nodes:
        bbox = node['bbox']
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        max_x = max(max_x, max(x_coords))
        max_y = max(max_y, max(y_coords))
    
    ax.set_xlim(0, max_x + 50)
    ax.set_ylim(max_y + 50, 0)
    
    # Draw nodes
    for node in nodes:
        bbox = node['bbox']
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        width = x2 - x1
        height = y2 - y1
        
        color = REGION_COLORS.get(node['region_type'], '#95a5a6')
        
        rect = Rectangle((x1, y1), width, height, 
                         linewidth=2, edgecolor=color, 
                         facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ax.text(center_x, center_y, f"#{node['node_id']}\n{node['region_type']}", 
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw filtered edges with labels
    for edge in filtered_edges:
        src_idx = edge['source']
        tgt_idx = edge['target']
        relation = edge['relation']
        score = edge['score']
        
        src_bbox = nodes[src_idx]['bbox']
        tgt_bbox = nodes[tgt_idx]['bbox']
        
        src_x = sum(p[0] for p in src_bbox) / 4
        src_y = sum(p[1] for p in src_bbox) / 4
        tgt_x = sum(p[0] for p in tgt_bbox) / 4
        tgt_y = sum(p[1] for p in tgt_bbox) / 4
        
        color = RELATION_COLORS.get(relation, '#34495e')
        
        arrow = FancyArrowPatch((src_x, src_y), (tgt_x, tgt_y),
                               arrowstyle='->', mutation_scale=20,
                               color=color, alpha=0.8, linewidth=2.5)
        ax.add_patch(arrow)
        
        # Add edge label
        mid_x = (src_x + tgt_x) / 2
        mid_y = (src_y + tgt_y) / 2
        ax.text(mid_x, mid_y, f"{relation}\n{score:.2f}", 
               fontsize=7, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax.set_aspect('equal')
    title = f'Graph Visualization - Relations: {", ".join(relation_types)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Showing {len(filtered_edges)} edges of type(s): {relation_types}")


# ==================== ANALYSIS FUNCTIONS ====================

def print_edge_details(nodes: List[Dict], edges: List[Dict], top_n: int = 20):
    """
    Print detailed information about top edges sorted by score.
    
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        top_n: Number of top edges to display
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_n} EDGES BY SCORE")
    print(f"{'='*80}\n")
    
    # Sort edges by score
    sorted_edges = sorted(edges, key=lambda x: x['score'], reverse=True)[:top_n]
    
    for i, edge in enumerate(sorted_edges, 1):
        src_idx = edge['source']
        tgt_idx = edge['target']
        relation = edge['relation']
        score = edge['score']
        category = edge.get('category', 'unknown')
        
        src_node = nodes[src_idx]
        tgt_node = nodes[tgt_idx]
        
        print(f"{i:2d}. [{relation}] Score: {score:.3f} ({category})")
        print(f"    Source #{src_idx} ({src_node['region_type']}): {src_node['text'][:50]}...")
        print(f"    Target #{tgt_idx} ({tgt_node['region_type']}): {tgt_node['text'][:50]}...")
        print()


def analyze_graph_statistics(nodes: List[Dict], edges: List[Dict]):
    """
    Compute and display comprehensive graph statistics.
    
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
    """
    print(f"\n{'='*60}")
    print(f"GRAPH STATISTICS")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Number of Nodes: {len(nodes)}")
    print(f"Number of Edges: {len(edges)}")
    
    if len(edges) == 0:
        print("\n⚠️ No edges were generated. Check region data format.")
        return
    
    # Node degree statistics
    in_degree = {i: 0 for i in range(len(nodes))}
    out_degree = {i: 0 for i in range(len(nodes))}
    
    for edge in edges:
        out_degree[edge['source']] += 1
        in_degree[edge['target']] += 1
    
    avg_out_degree = sum(out_degree.values()) / len(nodes) if nodes else 0
    avg_in_degree = sum(in_degree.values()) / len(nodes) if nodes else 0
    
    print(f"\nDegree Statistics:")
    print(f"  Average Out-Degree: {avg_out_degree:.2f}")
    print(f"  Average In-Degree: {avg_in_degree:.2f}")
    print(f"  Max Out-Degree: {max(out_degree.values()) if out_degree else 0}")
    print(f"  Max In-Degree: {max(in_degree.values()) if in_degree else 0}")
    
    # Region type distribution
    type_counts = {}
    for node in nodes:
        rtype = node['region_type']
        type_counts[rtype] = type_counts.get(rtype, 0) + 1
    
    print(f"\nRegion Type Distribution:")
    for rtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rtype:15s}: {count:3d} ({count/len(nodes)*100:.1f}%)")
    
    # Relation type distribution
    relation_counts = {}
    for edge in edges:
        rel = edge['relation']
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print(f"\nRelation Type Distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel:20s}: {count:3d} ({count/len(edges)*100:.1f}%)")
    
    # Score statistics
    scores = [e['score'] for e in edges]
    print(f"\nEdge Score Statistics:")
    print(f"  Mean Score: {np.mean(scores):.3f}")
    print(f"  Median Score: {np.median(scores):.3f}")
    print(f"  Min Score: {np.min(scores):.3f}")
    print(f"  Max Score: {np.max(scores):.3f}")
    print(f"  Std Dev: {np.std(scores):.3f}")
    
    print(f"\n{'='*60}\n")


def plot_score_distributions(edges: List[Dict]):
    """
    Plot score distributions for different relation types.
    
    Args:
        edges: List of edge dicts
    """
    if len(edges) == 0:
        print("⚠️ No edges to visualize.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall score distribution
    scores = [e['score'] for e in edges]
    axes[0].hist(scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(scores):.3f}')
    axes[0].axvline(np.median(scores), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(scores):.3f}')
    axes[0].set_xlabel('Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Edge Score Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Score distribution by relation type
    relation_scores = {}
    for edge in edges:
        rel = edge['relation']
        if rel not in relation_scores:
            relation_scores[rel] = []
        relation_scores[rel].append(edge['score'])
    
    # Box plot (sorted by mean score)
    sorted_items = sorted(relation_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    box_data = [scores for rel, scores in sorted_items]
    box_labels = [rel for rel, scores in sorted_items]
    
    if box_data:
        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
    
    axes[1].set_xlabel('Relation Type', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Score Distribution by Relation Type', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# ==================== EXPORT FUNCTIONS ====================

def export_graph_to_json(
    graph_result: Dict[str, Any], 
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Export graph structure to JSON file.
    
    Args:
        graph_result: Graph result dict with nodes, edges, adjacency
        output_path: Path to save JSON file
        metadata: Optional metadata dict to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    graph_data = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'nodes': graph_result['nodes'],
        'edges': graph_result['edges'],
        'adjacency': graph_result['adjacency']
    }
    
    if metadata:
        graph_data['metadata'] = metadata
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Graph exported to: {output_path}")
    print(f"\nExported data structure:")
    print(f"  - Nodes: {len(graph_data['nodes'])}")
    print(f"  - Edges: {len(graph_data['edges'])}")
    print(f"  - Adjacency entries: {len(graph_data['adjacency'])}")
