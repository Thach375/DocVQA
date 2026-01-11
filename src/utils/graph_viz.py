"""
Graph visualization utilities.
Provides functions to visualize semantic layout graph.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any
import numpy as np


def visualize_graph(
    graph: Dict[str, Any],
    regions: List[Dict],
    image_width: int = 600,
    image_height: int = 800,
    show_all_edges: bool = False,
    edge_categories: List[str] = None
):
    """
    Visualize semantic layout graph on document canvas.
    
    Args:
        graph: Graph result từ GraphBuilder
        regions: Original regions với bbox
        image_width: Canvas width (default: 600)
        image_height: Canvas height (default: 800)
        show_all_edges: Show tất cả edges (mặc định: chỉ semantic)
        edge_categories: List categories để show ['spatial', 'proximity', 'semantic']
    """
    if edge_categories is None:
        edge_categories = ['semantic']  # Default: chỉ semantic edges
    
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()  # Y tăng xuống dưới
    ax.set_aspect('equal')
    
    # Draw nodes (regions)
    node_colors = {
        'text': 'lightblue',
        'table': 'lightgreen',
        'form': 'lightyellow',
        'figure': 'lightcoral'
    }
    
    node_positions = {}  # Store center positions for edge drawing
    
    for idx, node in enumerate(graph['nodes']):
        bbox = node['bbox']
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        
        # Store center
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        node_positions[idx] = (cx, cy)
        
        # Draw bbox
        region_type = node['region_type']
        color = node_colors.get(region_type, 'lightgray')
        
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # Draw label
        label = f"{idx}: {region_type}"
        ax.text(
            cx, cy,
            label,
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Draw edges
    edge_colors = {
        'spatial': 'blue',
        'proximity': 'orange',
        'semantic': 'red'
    }
    
    edge_styles = {
        'spatial': '-',
        'proximity': '--',
        'semantic': '-'
    }
    
    edge_widths = {
        'spatial': 1,
        'proximity': 1.5,
        'semantic': 2.5
    }
    
    # Filter edges by category
    edges_to_draw = [
        e for e in graph['edges'] 
        if e['category'] in edge_categories
    ]
    
    for edge in edges_to_draw:
        source = edge['source']
        target = edge['target']
        
        if source not in node_positions or target not in node_positions:
            continue
        
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]
        
        category = edge['category']
        color = edge_colors.get(category, 'gray')
        style = edge_styles.get(category, '-')
        width = edge_widths.get(category, 1)
        
        # Draw arrow
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=width,
                linestyle=style,
                alpha=0.6
            )
        )
        
        # Draw edge label (relation + score)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        label = f"{edge['relation']}\n{edge['score']:.2f}"
        ax.text(
            mid_x, mid_y,
            label,
            fontsize=7,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            color=color
        )
    
    # Legend
    legend_elements = []
    for cat in edge_categories:
        if cat in edge_colors:
            legend_elements.append(
                plt.Line2D(
                    [0], [0],
                    color=edge_colors[cat],
                    lw=edge_widths[cat],
                    linestyle=edge_styles[cat],
                    label=cat.capitalize()
                )
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Semantic Layout Graph', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.show()


def print_graph_stats(graph: Dict[str, Any]):
    """
    In thống kê về graph.
    
    Args:
        graph: Graph result từ GraphBuilder
    """
    print("="*80)
    print("GRAPH STATISTICS")
    print("="*80)
    
    # Node stats
    print(f"\n📊 Nodes: {len(graph['nodes'])}")
    node_types = {}
    for node in graph['nodes']:
        region_type = node['region_type']
        node_types[region_type] = node_types.get(region_type, 0) + 1
    
    print("   By type:")
    for rtype, count in sorted(node_types.items()):
        print(f"     - {rtype}: {count}")
    
    # Edge stats
    print(f"\n🔗 Edges: {len(graph['edges'])}")
    
    edge_categories = {}
    edge_relations = {}
    
    for edge in graph['edges']:
        category = edge['category']
        relation = edge['relation']
        
        edge_categories[category] = edge_categories.get(category, 0) + 1
        edge_relations[relation] = edge_relations.get(relation, 0) + 1
    
    print("   By category:")
    for cat, count in sorted(edge_categories.items()):
        print(f"     - {cat}: {count}")
    
    print("   By relation:")
    for rel, count in sorted(edge_relations.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {rel}: {count}")
    
    # Score stats
    scores = [edge['score'] for edge in graph['edges']]
    if scores:
        print(f"\n📈 Edge Scores:")
        print(f"     - Min: {min(scores):.3f}")
        print(f"     - Max: {max(scores):.3f}")
        print(f"     - Mean: {np.mean(scores):.3f}")
        print(f"     - Median: {np.median(scores):.3f}")
    
    # Adjacency stats
    adjacency = graph['adjacency']
    if adjacency:
        out_degrees = [len(neighbors) for neighbors in adjacency.values()]
        print(f"\n🔢 Node Degrees:")
        print(f"     - Min out-degree: {min(out_degrees)}")
        print(f"     - Max out-degree: {max(out_degrees)}")
        print(f"     - Avg out-degree: {np.mean(out_degrees):.2f}")
    
    print("="*80)
