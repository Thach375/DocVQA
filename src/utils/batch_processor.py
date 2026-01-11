"""
Batch processor cho DocVQA dataset.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm


class BatchProcessor:
    """
    Xử lý batch images từ dataset.
    """
    
    def __init__(self, pipeline_processor):
        """
        Args:
            pipeline_processor: FullPipelineProcessor instance
        """
        self.pipeline = pipeline_processor
    
    def process_dataset(
        self,
        images_folder: Path,
        output_folder: Path,
        subsets: List[str] = ['train', 'validation', 'test'],
        max_images_per_subset: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Process toàn bộ dataset theo batches.
        
        Args:
            images_folder: Root folder chứa images
            output_folder: Root folder để lưu JSON outputs
            subsets: List các subsets cần process (train, val, test)
            max_images_per_subset: Giới hạn số lượng images mỗi subset
            skip_existing: Skip files đã được process
        
        Returns:
            dict: Overall statistics
        """
        overall_stats = {
            'total_processed': 0,
            'total_success': 0,
            'total_failed': 0,
            'by_subset': {}
        }
        
        for subset in subsets:
            print(f"\n{'='*70}")
            print(f"Processing subset: {subset.upper()}")
            print(f"{'='*70}")
            
            # Get image files
            subset_folder = images_folder / subset
            if not subset_folder.exists():
                print(f"⚠️ Folder not found: {subset_folder}")
                continue
            
            image_files = sorted(list(subset_folder.glob('*.png')))
            
            if max_images_per_subset:
                image_files = image_files[:max_images_per_subset]
            
            print(f"Found {len(image_files)} images")
            
            # Create output subfolder
            subset_output = output_folder / subset
            subset_output.mkdir(parents=True, exist_ok=True)
            
            # Process each image with progress bar
            subset_stats = self._process_subset(
                image_files,
                subset_output,
                subset,
                skip_existing
            )
            
            # Print subset summary
            print(f"\n{subset.upper()} Summary:")
            print(f"  ✅ Success: {subset_stats['success']}")
            print(f"  ❌ Failed: {subset_stats['failed']}")
            print(f"  📊 Total: {subset_stats['processed']}")
            
            # Update overall stats
            overall_stats['total_processed'] += subset_stats['processed']
            overall_stats['total_success'] += subset_stats['success']
            overall_stats['total_failed'] += subset_stats['failed']
            overall_stats['by_subset'][subset] = subset_stats
        
        return overall_stats
    
    def _process_subset(
        self,
        image_files: List[Path],
        output_folder: Path,
        subset_name: str,
        skip_existing: bool
    ) -> Dict[str, Any]:
        """
        Process một subset.
        
        Returns:
            dict: Subset statistics
        """
        stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for img_path in tqdm(image_files, desc=f"Processing {subset_name}"):
            image_id = img_path.stem
            output_path = output_folder / f"{image_id}.json"
            
            # Skip if already processed AND valid JSON
            if skip_existing and output_path.exists():
                # Validate JSON file
                if self._validate_json_file(output_path):
                    stats['processed'] += 1
                    stats['success'] += 1
                    continue
                else:
                    # File exists but corrupt, re-process
                    print(f"\n⚠️ Corrupt JSON detected: {image_id}, re-processing...")
            
            # Process image
            result = self.pipeline.process_image(
                image_path=img_path,
                output_path=output_path
            )
            
            stats['processed'] += 1
            
            if result['success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['errors'].append({
                    'image_id': image_id,
                    'error': result.get('error', 'Unknown error')
                })
        
        return stats
    
    def _validate_json_file(self, json_path: Path) -> bool:
        """
        Validate if JSON file is valid and complete.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required keys
            required_keys = ['version', 'image_id', 'ocr', 'layout', 'graph']
            if not all(key in data for key in required_keys):
                return False
            
            # Check OCR success
            if not data['ocr'].get('success', False):
                return False
            
            return True
            
        except (json.JSONDecodeError, IOError, KeyError):
            return False
