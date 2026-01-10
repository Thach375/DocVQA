import sys
sys.path.insert(0, '..')

import os
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image

from src.ocr.ocr_processor import PaddleOCRProcessor
from src.utils.constant import * 

def sample_layouts(sample_image, ocr_processor):
    

    if sample_image:
        print(f"Ảnh mẫu: {sample_image}")
        
        # Chạy OCR
        print("\nĐang chạy OCR...")
        result = ocr_processor.run_ocr(sample_image)
        
        if result['success']:
            print(f"\n✓ Phát hiện {result['num_lines']} dòng văn bản")
            print(f"✓ Độ tin cậy TB: {result['avg_confidence']:.2%}")
        else:
            print(f"\n✗ Lỗi: {result['error']}")
    else:
        print("Không tìm thấy ảnh mẫu. Hãy chạy notebook 1_Download_data.ipynb trước.")
    
    return sample_image
        
def preprocess(sample_image, ocr_processor):
    if sample_image:
        original_img = Image.open(sample_image)
        img_step1 = ocr_processor.load_and_fix_exif(sample_image)
        img_step2 = ocr_processor.resize_image(img_step1, max_size=2500)
        img_step3 = ocr_processor.remove_padding(img_step2)
        img_step4 = ocr_processor.perspective_correction(img_step3)
        
        print("\n✓ Hoàn thành preprocessing!")
    else:
        print("Không tìm thấy ảnh mẫu")
        
    result_no_preprocess = ocr_processor.run_ocr(sample_image, use_preprocessing=False)
    result_with_preprocess = ocr_processor.run_ocr(sample_image, use_preprocessing=True)
    
    # So sánh kết quả
    print(f"\n{'='*80}")
    print("SO SÁNH KẾT QUẢ")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Không Preprocess':>20} {'Có Preprocess':>20}")
    print(f"{'-'*80}")
    print(f"{'Số dòng phát hiện':<30} {result_no_preprocess['num_lines']:>20} {result_with_preprocess['num_lines']:>20}")
    print(f"{'Độ tin cậy trung bình':<30} {result_no_preprocess['avg_confidence']:>19.2%} {result_with_preprocess['avg_confidence']:>19.2%}")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Ảnh gốc
    axes[0, 0].imshow(Image.open(sample_image))
    axes[0, 0].set_title('Ảnh gốc', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ảnh sau preprocessing
    preprocessed_img = ocr_processor.preprocess_image(sample_image, max_size=2500)
    axes[0, 1].imshow(preprocessed_img)
    axes[0, 1].set_title('Ảnh sau Preprocessing', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # OCR không preprocessing
    img_no_preprocess = ocr_processor.draw_bounding_boxes(sample_image, result_no_preprocess, show_text=False)
    axes[1, 0].imshow(img_no_preprocess)
    axes[1, 0].set_title(
        f'OCR KHÔNG Preprocessing\n{result_no_preprocess["num_lines"]} dòng | Độ tin cậy: {result_no_preprocess["avg_confidence"]:.1%}',
        fontsize=12
    )
    axes[1, 0].axis('off')
    
    # OCR có preprocessing - cần vẽ trên ảnh đã preprocess
    # Tạo temporary result cho ảnh preprocessed
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        preprocessed_img.save(tmp.name)
        img_with_preprocess = ocr_processor.draw_bounding_boxes(tmp.name, result_with_preprocess, show_text=False)
    
    axes[1, 1].imshow(img_with_preprocess)
    axes[1, 1].set_title(
        f'OCR CÓ Preprocessing\n{result_with_preprocess["num_lines"]} dòng | Độ tin cậy: {result_with_preprocess["avg_confidence"]:.1%}',
        fontsize=12
    )
    axes[1, 1].axis('off')
    
    plt.suptitle('So sánh OCR: Với và Không có Preprocessing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return result_no_preprocess, result_with_preprocess


def visualize_stage1_groups(sample_image, ocr_processor):
    """
    Vẽ visualization cho các group ở Stage I: tokens → lines → blocks
    """
    if not sample_image:
        print("Không có ảnh mẫu")
        return
    
    print("Chạy OCR với Layout Analysis...")
    result_with_layout = ocr_processor.run_ocr_with_layout(
        sample_image, 
        use_preprocessing=True, 
        max_size=2500
    )
    
    if not result_with_layout['success'] or not result_with_layout['layout']:
        print("Không có kết quả layout")
        return
    
    layout = result_with_layout['layout']
    
    # Load ảnh gốc
    img = Image.open(sample_image)
    
    # Tạo figure với 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. Tokens (OCR bounding boxes)
    img_tokens = img.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img_tokens)
    
    # Vẽ từng token từ OCR details
    for detail in result_with_layout['details']:
        if detail.get('box'):
            box = detail['box']
            # Convert box coordinates to bbox format [x1, y1, x2, y2]
            if len(box) == 4:  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                draw.rectangle(bbox, outline='red', width=2)
    
    axes[0].imshow(img_tokens)
    axes[0].set_title(f'Stage I-A: Tokens\n({len(result_with_layout["details"])} tokens)', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Lines
    img_lines = img.copy()
    draw_lines = ImageDraw.Draw(img_lines)
    
    import numpy as np
    colors = plt.cm.rainbow(np.linspace(0, 1, len(layout['lines'])))
    
    for idx, line in enumerate(layout['lines']):
        bbox = line['bbox']
        if bbox:
            # Convert [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [x1, y1, x2, y2]
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            rect_bbox = [min(xs), min(ys), max(xs), max(ys)]
            color = tuple([int(c * 255) for c in colors[idx][:3]])
            draw_lines.rectangle(rect_bbox, outline=color, width=3)
    
    axes[1].imshow(img_lines)
    axes[1].set_title(f'Stage I-B: Lines\n({len(layout["lines"])} lines)', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Blocks
    img_blocks = img.copy()
    draw_blocks = ImageDraw.Draw(img_blocks)
    
    block_colors = plt.cm.Set3(np.linspace(0, 1, len(layout['blocks'])))
    
    for idx, block in enumerate(layout['blocks']):
        bbox = block['bbox']
        if bbox:
            # Convert [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [x1, y1, x2, y2]
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            rect_bbox = [min(xs), min(ys), max(xs), max(ys)]
            color = tuple([int(c * 255) for c in block_colors[idx][:3]])
            draw_blocks.rectangle(rect_bbox, outline=color, width=4)
            
            # Thêm label số block
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw_blocks.text((rect_bbox[0], rect_bbox[1]-25), f'Block {idx+1}', fill=color, font=font)
    
    axes[2].imshow(img_blocks)
    axes[2].set_title(f'Stage I-C: Blocks\n({len(layout["blocks"])} blocks)', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Stage I: Token Grouping (Token → Line → Block)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # In thống kê
    print(f"\n{'='*80}")
    print("STAGE I - TOKEN GROUPING STATISTICS")
    print(f"{'='*80}")
    print(f"Tokens: {len(result_with_layout['details'])}")
    print(f"Lines:  {len(layout['lines'])}")
    print(f"Blocks: {len(layout['blocks'])}")
    print(f"{'='*80}\n")
    
    return result_with_layout


def group(sample_image, ocr_processor):
    if sample_image:
        print("Chạy OCR với Layout Analysis...")
        result_with_layout = ocr_processor.run_ocr_with_layout(
            sample_image, 
            use_preprocessing=True, 
            max_size=2500
        )
        
        if result_with_layout['success'] and result_with_layout['layout']:
            layout = result_with_layout['layout']
            
            print(f"\n{'='*80}")
            print("KẾT QUẢ LAYOUT ANALYSIS")
            print(f"{'='*80}")
            print(f"Số lines: {len(layout['lines'])}")
            print(f"Số blocks: {len(layout['blocks'])}")
            print(f"Số regions: {len(layout['regions'])}")
            
            print(f"\n{'Regions phát hiện:'}")
            print(f"{'-'*80}")
            
            for idx, region in enumerate(layout['regions'], 1):
                print(f"\n{idx}. Type: {region['region_type'].upper()} | Score: {region['score']:.3f}")
                
                # Print metadata
                metadata = region['metadata']
                if region['region_type'] == 'table':
                    print(f"   - Columns: {metadata.get('num_cols', 0)}")
                    print(f"   - Col stability: {metadata.get('col_stability', 0):.2f}")
                    print(f"   - Row spacing var: {metadata.get('row_spacing_var', 0):.2f}")
                
                elif region['region_type'] == 'form':
                    print(f"   - Colon ratio: {metadata.get('colon_ratio', 0):.2f}")
                    print(f"   - Left alignment: {metadata.get('left_alignment', 0):.2f}")
                    print(f"   - Right alignment: {metadata.get('right_alignment', 0):.2f}")
                
                elif region['region_type'] == 'figure':
                    print(f"   - Empty center ratio: {metadata.get('empty_center_ratio', 0):.2f}")
                    print(f"   - Tick-like numbers: {metadata.get('tick_like_numbers', 0)}")
                    print(f"   - Legend cluster: {metadata.get('legend_cluster', False)}")
                
                elif region['region_type'] == 'text':
                    print(f"   - Avg line length: {metadata.get('avg_line_length', 0):.1f}")
                    print(f"   - Spacing uniformity: {metadata.get('spacing_uniformity', 0):.2f}")
            
            print(f"{'='*80}")
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
            # Ảnh gốc
            axes[0].imshow(Image.open(sample_image))
            axes[0].set_title('Ảnh gốc', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # OCR bounding boxes
            img_with_boxes = ocr_processor.draw_bounding_boxes(sample_image, result_with_layout, show_text=False)
            axes[1].imshow(img_with_boxes)
            axes[1].set_title('OCR Token Boxes', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Layout regions
            img_with_layout = ocr_processor.draw_layout_regions(sample_image, result_with_layout['layout'])
            axes[2].imshow(img_with_layout)
            axes[2].set_title('Layout Regions\n(Red=Table, Blue=Form, Green=Figure, Orange=Text)', 
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Hiển thị legend
            print("\n" + "="*80)
            print("LEGEND - Region Colors:")
            print("="*80)
            print("  🔴 RED    = Table (bảng)")
            print("  🔵 BLUE   = Form (key:value)")
            print("  🟢 GREEN  = Figure (chart/plot)")
            print("  🟠 ORANGE = Text (đoạn văn)")
            print("  🟣 PURPLE = Layout (multi-column/heading)")
            print("="*80)
        
            layout = result_with_layout['layout']
    
            print("\n" + "="*80)
            print("TEXT CONTENT THEO TỪNG REGION")
            print("="*80)
            
            # Group regions by type
            regions_by_type = {}
            for region in layout['regions']:
                region_type = region['region_type']
                if region_type not in regions_by_type:
                    regions_by_type[region_type] = []
                regions_by_type[region_type].append(region)
            
            # Print text for each region type
            for region_type in ['table', 'form', 'figure', 'text']:
                if region_type in regions_by_type:
                    regions = regions_by_type[region_type]
                    
                    print(f"\n{'─'*80}")
                    print(f"📋 {region_type.upper()} REGIONS ({len(regions)} phát hiện)")
                    print(f"{'─'*80}")
                    
                    for idx, region in enumerate(regions, 1):
                        block = region['block']
                        score = region['score']
                        
                        print(f"\n[{region_type.upper()} #{idx}] Score: {score:.3f}")
                        print(f"{'─'*40}")
                        
                        # Print metadata nếu là form
                        if region_type == 'form':
                            metadata = region['metadata']
                            has_kv = metadata.get('has_keyvalue_pattern', False)
                            kv_ratio = metadata.get('keyvalue_ratio', 0.0)
                            
                            if has_kv:
                                print(f"✓ Key:Value Pattern Detected ({kv_ratio:.1%})")
                                print(f"{'─'*40}")
                        
                        # Print all lines in block
                        for line_idx, line in enumerate(block['lines'], 1):
                            text = line['text'].strip()
                            if text:
                                print(f"{line_idx:2d}. {text}")
                        
                        if not block['lines'] or not any(line['text'].strip() for line in block['lines']):
                            print("   (No text content)")
            
            
        