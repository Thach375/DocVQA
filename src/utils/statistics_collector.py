"""
Statistics collector cho processed dataset.
"""

import json
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


class StatisticsCollector:
    """
    Thu thập và phân tích thống kê từ processed JSON files.
    """
    
    @staticmethod
    def collect_from_folder(output_folder: Path) -> Dict[str, Any]:
        """
        Thu thập thống kê từ tất cả JSON files trong folder.
        
        Args:
            output_folder: Folder chứa JSON outputs
        
        Returns:
            dict: Complete statistics
        """
        stats = {
            'total_files': 0,
            'total_tokens': 0,
            'total_regions': 0,
            'total_edges': 0,
            'avg_tokens_per_image': 0,
            'avg_regions_per_image': 0,
            'avg_edges_per_image': 0,
            'region_type_counts': {},
            'relation_type_counts': {},
            'edge_category_counts': {}
        }
        
        json_files = list(output_folder.glob('**/*.json'))
        
        # Exclude dataset_statistics.json if exists
        json_files = [f for f in json_files if f.name != 'dataset_statistics.json']
        
        stats['total_files'] = len(json_files)
        
        if stats['total_files'] == 0:
            return stats
        
        for json_file in tqdm(json_files, desc="Collecting statistics"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # OCR stats
                stats['total_tokens'] += data['ocr']['num_tokens']
                
                # Layout stats
                stats['total_regions'] += data['layout']['num_regions']
                
                # Count region types
                for region in data['layout']['regions']:
                    rtype = region['region_type']
                    stats['region_type_counts'][rtype] = stats['region_type_counts'].get(rtype, 0) + 1
                
                # Graph stats
                stats['total_edges'] += data['graph']['num_edges']
                
                # Count relation types
                for edge in data['graph']['edges']:
                    rel = edge['relation']
                    stats['relation_type_counts'][rel] = stats['relation_type_counts'].get(rel, 0) + 1
                    
                    # Count by category (spatial, semantic, proximity)
                    cat = edge.get('category', 'unknown')
                    stats['edge_category_counts'][cat] = stats['edge_category_counts'].get(cat, 0) + 1
                    
            except Exception as e:
                print(f"⚠️ Error processing {json_file.name}: {e}")
                continue
        
        # Calculate averages
        if stats['total_files'] > 0:
            stats['avg_tokens_per_image'] = stats['total_tokens'] / stats['total_files']
            stats['avg_regions_per_image'] = stats['total_regions'] / stats['total_files']
            stats['avg_edges_per_image'] = stats['total_edges'] / stats['total_files']
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict[str, Any]) -> None:
        """
        In ra thống kê dưới dạng đẹp.
        
        Args:
            stats: Statistics dict
        """
        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print(f"{'='*70}")
        print(f"Total Images: {stats['total_files']:,}")
        print(f"Total OCR Tokens: {stats['total_tokens']:,}")
        print(f"Total Layout Regions: {stats['total_regions']:,}")
        print(f"Total Graph Edges: {stats['total_edges']:,}")
        
        print(f"\nAverages per Image:")
        print(f"  Tokens: {stats['avg_tokens_per_image']:.1f}")
        print(f"  Regions: {stats['avg_regions_per_image']:.1f}")
        print(f"  Edges: {stats['avg_edges_per_image']:.1f}")
        
        if stats['region_type_counts']:
            print(f"\nRegion Type Distribution:")
            for rtype, count in sorted(stats['region_type_counts'].items(), 
                                      key=lambda x: x[1], reverse=True):
                pct = (count / stats['total_regions'] * 100) if stats['total_regions'] > 0 else 0
                print(f"  {rtype:15}: {count:,} ({pct:.1f}%)")
        
        if stats['edge_category_counts']:
            print(f"\nEdge Category Distribution:")
            for cat, count in sorted(stats['edge_category_counts'].items(), 
                                    key=lambda x: x[1], reverse=True):
                pct = (count / stats['total_edges'] * 100) if stats['total_edges'] > 0 else 0
                print(f"  {cat:15}: {count:,} ({pct:.1f}%)")
        
        if stats['relation_type_counts']:
            print(f"\nTop 10 Relation Types:")
            sorted_relations = sorted(stats['relation_type_counts'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
            for rel, count in sorted_relations:
                pct = (count / stats['total_edges'] * 100) if stats['total_edges'] > 0 else 0
                print(f"  {rel:20}: {count:,} ({pct:.1f}%)")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def save_statistics(stats: Dict[str, Any], output_path: Path) -> None:
        """
        Lưu statistics ra JSON file.
        
        Args:
            stats: Statistics dict
            output_path: Path to save JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Statistics saved to: {output_path}")
