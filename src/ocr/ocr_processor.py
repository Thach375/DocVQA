"""
Module xử lý OCR sử dụng PaddleOCR.
Bao gồm:
- Preprocessing ảnh (resize, remove padding, perspective, denoise)
- Trích xuất text từ ảnh
- Vẽ bounding box
- Lưu kết quả vào JSON
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import glob

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
from scipy.ndimage import rotate as scipy_rotate

from .layout_analyzer import DocumentLayoutAnalyzer


class PaddleOCRProcessor:
    """Class xử lý OCR sử dụng PaddleOCR."""
    
    def __init__(
        self,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
    ):
        """
        Khởi tạo PaddleOCR engine.
        
        Args:
            use_doc_orientation_classify: Sử dụng phân loại hướng document
            use_doc_unwarping: Sử dụng làm phẳng document
            use_textline_orientation: Sử dụng phát hiện hướng textline
        """
        from paddleocr import PaddleOCR
        
        print("Đang khởi tạo PaddleOCR engine...")
        self.ocr_engine = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
        )
        print("-> PaddleOCR đã sẵn sàng!")
        
        # Màu sắc cho bounding box
        self.box_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Khởi tạo layout analyzer
        self.layout_analyzer = DocumentLayoutAnalyzer()
    
    @staticmethod
    def load_and_fix_exif(image_path: str) -> Image.Image:
        """
        Load ảnh và tự động xoay theo EXIF orientation.
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            PIL Image đã được xoay đúng
        """
        img = Image.open(image_path)
        # Tự động xoay theo EXIF orientation tag
        img = ImageOps.exif_transpose(img)
        return img
    
    @staticmethod
    def resize_image(img: Image.Image, max_size: int = 2500) -> Image.Image:
        """
        Resize ảnh sao cho cạnh lớn nhất <= max_size, giữ nguyên tỷ lệ.
        
        Args:
            img: PIL Image
            max_size: Kích thước tối đa của cạnh lớn nhất
            
        Returns:
            PIL Image đã resize
        """
        width, height = img.size
        if max(width, height) <= max_size:
            return img
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    @staticmethod
    def remove_padding(img: Image.Image, threshold: int = 240) -> Image.Image:
        """
        Loại bỏ padding trắng xung quanh ảnh.
        
        Args:
            img: PIL Image
            threshold: Ngưỡng để xác định pixel trắng (0-255)
            
        Returns:
            PIL Image đã crop
        """
        # Convert sang numpy array
        img_array = np.array(img.convert('L'))
        
        # Tìm vùng không phải background
        mask = img_array < threshold
        coords = np.argwhere(mask)
        
        if len(coords) == 0:
            return img
        
        # Tìm bounding box
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        # Crop ảnh
        return img.crop((x0, y0, x1, y1))
    
    @staticmethod
    def denoise_image(img: Image.Image, strength: int = 10) -> Image.Image:
        """
        Khử nhiễu ảnh sử dụng Non-local Means Denoising.
        
        Args:
            img: PIL Image
            strength: Độ mạnh của denoise (1-20)
            
        Returns:
            PIL Image đã denoise
        """
        img_array = np.array(img)
        
        # Nếu là ảnh grayscale
        if len(img_array.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(img_array, None, strength, 7, 21)
        else:
            # Ảnh màu
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
        
        return Image.fromarray(denoised)
    
    @staticmethod
    def perspective_correction(img: Image.Image) -> Image.Image:
        """
        Tự động sửa perspective (góc chụp nghiêng) của document.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image đã sửa perspective
        """
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Blur và threshold
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Tìm contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Tìm contour lớn nhất (giả sử là document)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tính diện tích
        area = cv2.contourArea(largest_contour)
        img_area = img_array.shape[0] * img_array.shape[1]
        
        # Chỉ áp dụng nếu contour chiếm > 50% ảnh
        if area < img_area * 0.5:
            return img
        
        # Approximation để tìm 4 góc
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Nếu không tìm được 4 góc, thử với epsilon nhỏ hơn
        if len(approx) != 4:
            epsilon = 0.05 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            return img
        
        # Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        # Tính kích thước output
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Định nghĩa destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Tính perspective transform matrix và warp
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_array, M, (maxWidth, maxHeight))
        
        return Image.fromarray(warped)
    
    def preprocess_image(
        self,
        image_path: str,
        max_size: int = 2500,
        apply_denoise: bool = True,
        apply_perspective: bool = True,
        apply_remove_padding: bool = True
    ) -> Image.Image:
        """
        Áp dụng toàn bộ preprocessing pipeline cho ảnh.
        
        Args:
            image_path: Đường dẫn đến ảnh
            max_size: Kích thước tối đa
            apply_denoise: Có áp dụng denoise không
            apply_perspective: Có áp dụng perspective correction không
            apply_remove_padding: Có loại bỏ padding không
            
        Returns:
            PIL Image đã được preprocess
        """
        # 1. Load và fix EXIF
        img = self.load_and_fix_exif(image_path)
        
        # 2. Resize
        img = self.resize_image(img, max_size)
        
        # 3. Remove padding
        if apply_remove_padding:
            img = self.remove_padding(img)
        
        # 4. Perspective correction
        if apply_perspective:
            try:
                img = self.perspective_correction(img)
            except Exception:
                pass  # Nếu lỗi, giữ ảnh gốc
        
        # 5. Denoise
        if apply_denoise:
            try:
                img = self.denoise_image(img, strength=10)
            except Exception:
                pass
        
        return img
    
    def run_ocr(
        self, 
        image_path: Union[str, Image.Image],
        use_preprocessing: bool = False,
        max_size: int = 2500
    ) -> Dict[str, Any]:
        """
        Chạy OCR trên một ảnh.
        
        Args:
            image_path: Đường dẫn đến ảnh hoặc PIL Image
            use_preprocessing: Có áp dụng preprocessing không
            max_size: Kích thước tối đa nếu dùng preprocessing
            
        Returns:
            Dict chứa kết quả OCR với các key:
            - text: Toàn bộ text ghép lại
            - lines: List các dòng text
            - num_lines: Số dòng
            - details: Chi tiết từng vùng text (text, confidence, box)
            - avg_confidence: Độ tin cậy trung bình
            - success: True/False
            - error: Message lỗi nếu có
        """
        try:
            # Nếu use_preprocessing, preprocess ảnh trước
            if use_preprocessing:
                if isinstance(image_path, str):
                    preprocessed_img = self.preprocess_image(image_path, max_size=max_size)
                    # Lưu tạm để OCR engine có thể đọc
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        preprocessed_img.save(tmp.name)
                        ocr_input = tmp.name
                else:
                    # Nếu đã là PIL Image
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        image_path.save(tmp.name)
                        ocr_input = tmp.name
            else:
                ocr_input = image_path if isinstance(image_path, str) else image_path
            
            out = self.ocr_engine.predict(ocr_input)
            
            if not out:
                return self._empty_result("Không có output từ predict()")
            
            lines, details = [], []
            
            # Parse kết quả từ PaddleOCR
            for res in out if isinstance(out, (list, tuple)) else [out]:
                j = res.json if hasattr(res, "json") else res
                r = j.get("res", {}) if isinstance(j, dict) else {}
                
                texts = r.get("rec_texts", []) or []
                scores = r.get("rec_scores", []) or []
                polys = r.get("rec_polys", None)
                
                for i, text in enumerate(texts):
                    if not text or not str(text).strip():
                        continue
                        
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    box = None
                    
                    if polys is not None and i < len(polys):
                        box = polys[i]
                        if hasattr(box, "tolist"):
                            box = box.tolist()
                    
                    lines.append(text)
                    details.append({
                        "text": text,
                        "confidence": conf,
                        "box": box
                    })
            
            if not lines:
                return self._empty_result("Không phát hiện văn bản")
            
            avg_conf = sum(d["confidence"] for d in details) / len(details) if details else 0.0
            
            return {
                "text": " ".join(lines),
                "lines": lines,
                "num_lines": len(lines),
                "details": details,
                "avg_confidence": avg_conf,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return self._empty_result(str(e))
    
    @staticmethod
    def _empty_result(error_msg: str) -> Dict[str, Any]:
        """Tạo kết quả rỗng với message lỗi."""
        return {
            "text": "",
            "lines": [],
            "num_lines": 0,
            "details": [],
            "avg_confidence": 0.0,
            "success": False,
            "error": error_msg
        }
    
    def run_ocr_with_layout(
        self, 
        image_path: Union[str, Image.Image],
        use_preprocessing: bool = False,
        max_size: int = 2500
    ) -> Dict[str, Any]:
        """
        Chạy OCR và phân tích layout.
        
        Args:
            image_path: Đường dẫn đến ảnh hoặc PIL Image
            use_preprocessing: Có áp dụng preprocessing không
            max_size: Kích thước tối đa nếu dùng preprocessing
            
        Returns:
            Dict chứa kết quả OCR + layout analysis với các key:
            - Tất cả key từ run_ocr()
            - layout: {lines, blocks, regions}
        """
        # Run OCR first
        ocr_result = self.run_ocr(image_path, use_preprocessing, max_size)
        
        if not ocr_result['success']:
            ocr_result['layout'] = None
            return ocr_result
        
        # Analyze layout
        try:
            layout_result = self.layout_analyzer.analyze_layout(ocr_result)
            ocr_result['layout'] = layout_result
        except Exception as e:
            print(f"Layout analysis error: {e}")
            ocr_result['layout'] = None
        
        return ocr_result
    
    def draw_bounding_boxes(
        self,
        image_path: str,
        ocr_result: Dict[str, Any],
        show_text: bool = True
    ) -> Image.Image:
        """
        Vẽ bounding box lên ảnh.
        
        Args:
            image_path: Đường dẫn đến ảnh
            ocr_result: Kết quả từ run_ocr()
            show_text: Có hiển thị text label không
            
        Returns:
            PIL Image với bounding boxes
        """
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        if not ocr_result['success'] or not ocr_result['details']:
            return img
        
        for idx, detail in enumerate(ocr_result['details']):
            box = detail['box']
            conf = detail['confidence']
            color = self.box_colors[idx % len(self.box_colors)]
            
            if box:
                # Vẽ polygon (4 điểm)
                points = [(int(p[0]), int(p[1])) for p in box]
                draw.polygon(points, outline=color, width=3)
                
                # Vẽ label nếu được yêu cầu
                if show_text:
                    text_pos = (int(box[0][0]), int(box[0][1]) - 20)
                    label = f"{idx+1}: {conf:.2f}"
                    draw.text(text_pos, label, fill=color)
        
        return img
    
    def draw_layout_regions(
        self,
        image_path: str,
        layout_result: Dict[str, Any]
    ) -> Image.Image:
        """
        Vẽ layout regions lên ảnh.
        
        Args:
            image_path: Đường dẫn đến ảnh
            layout_result: Kết quả từ layout analysis
            
        Returns:
            PIL Image với layout regions
        """
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        region_colors = {
            'table': 'red',
            'form': 'blue',
            'figure': 'green',
            'text': 'orange',
            'layout': 'purple'
        }
        
        if not layout_result or 'regions' not in layout_result:
            return img
        
        for region in layout_result['regions']:
            bbox = region['block']['bbox']
            region_type = region['region_type']
            score = region['score']
            
            if bbox:
                color = region_colors.get(region_type, 'gray')
                points = [(int(p[0]), int(p[1])) for p in bbox]
                draw.polygon(points, outline=color, width=5)
                
                # Draw label
                text_pos = (int(bbox[0][0]), int(bbox[0][1]) - 30)
                label = f"{region_type.upper()} ({score:.2f})"
                draw.text(text_pos, label, fill=color)
        
        return img
    
    @staticmethod
    def save_result_to_json(
        ocr_result: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Lưu kết quả OCR vào file JSON.
        
        Args:
            ocr_result: Kết quả từ run_ocr()
            output_path: Đường dẫn file JSON output
        """
        data = {
            'success': ocr_result['success'],
            'full_text': ocr_result['text'],
            'num_lines': ocr_result['num_lines'],
            'avg_confidence': ocr_result.get('avg_confidence', 0.0),
            'error': ocr_result['error'],
            'text_regions': []
        }
        
        for detail in ocr_result['details']:
            data['text_regions'].append({
                'text': detail['text'],
                'confidence': detail['confidence'],
                'bounding_box': detail['box']
            })
        
        # Add layout if available
        if 'layout' in ocr_result and ocr_result['layout']:
            layout = ocr_result['layout']
            data['layout'] = {
                'num_lines': len(layout.get('lines', [])),
                'num_blocks': len(layout.get('blocks', [])),
                'num_regions': len(layout.get('regions', [])),
                'regions': []
            }
            
            for region in layout.get('regions', []):
                data['layout']['regions'].append({
                    'type': region['region_type'],
                    'score': region['score'],
                    'metadata': region['metadata']
                })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            import numpy as np
            
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        data = convert_numpy_types(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def process_batch(
        self,
        image_folder: str,
        output_folder: str,
        image_extensions: tuple = ('.png', '.jpg', '.jpeg'),
        save_json: bool = True,
        max_images: Optional[int] = None,
        use_preprocessing: bool = True,
        max_size: int = 2500
    ) -> Dict[str, Any]:
        """
        Xử lý OCR hàng loạt cho tất cả ảnh trong folder.
        
        Args:
            image_folder: Thư mục chứa ảnh
            output_folder: Thư mục lưu kết quả JSON
            image_extensions: Các định dạng ảnh hỗ trợ
            save_json: Có lưu file JSON không
            max_images: Số ảnh tối đa cần xử lý (None = xử lý tất cả)
            use_preprocessing: Có áp dụng preprocessing không
            max_size: Kích thước tối đa cho preprocessing
            
        Returns:
            Dict chứa thống kê xử lý
        """
        # Thu thập file ảnh
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, f"**/*{ext}"), recursive=True))
        
        # Giới hạn số lượng ảnh nếu cần
        if max_images is not None:
            image_files = image_files[:max_images]
        
        print(f"Tìm thấy {len(image_files)} ảnh")
        if use_preprocessing:
            print("Preprocessing: Bật (resize, remove padding, perspective, denoise)")
        else:
            print("Preprocessing: Tắt")
        
        # Thống kê
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'results': []
        }
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Xử lý từng ảnh
        for image_path in tqdm(image_files, desc="Đang xử lý OCR"):
            ocr_result = self.run_ocr(image_path, use_preprocessing=use_preprocessing, max_size=max_size)
            
            # Tạo tên file JSON
            rel_path = os.path.relpath(image_path, image_folder)
            stem = Path(rel_path).stem
            json_filename = f"{stem}.json"
            json_path = os.path.join(output_folder, json_filename)
            
            if save_json:
                self.save_result_to_json(ocr_result, json_path)
            
            # Cập nhật thống kê
            if ocr_result['success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            stats['results'].append({
                'image_path': image_path,
                'json_path': json_path,
                'success': ocr_result['success'],
                'num_lines': ocr_result['num_lines']
            })
        
        # In thống kê
        print(f"\n{'='*60}")
        print("THỐNG KÊ XỬ LÝ OCR")
        print(f"{'='*60}")
        print(f"Tổng số ảnh: {stats['total']}")
        print(f"  ✓ Thành công: {stats['success']} ({stats['success']/max(stats['total'],1)*100:.1f}%)")
        print(f"  ✗ Thất bại: {stats['failed']} ({stats['failed']/max(stats['total'],1)*100:.1f}%)")
        print(f"{'='*60}")
        
        return stats
    
    def process_docvqa_images(
        self,
        images_folder: str,
        output_folder: str,
        subsets: List[str] = ["train", "test", "validation"],
        max_images_per_subset: Optional[int] = None,
        use_preprocessing: bool = True,
        max_size: int = 2500
    ) -> Dict[str, Any]:
        """
        Xử lý OCR cho dataset DocVQA.
        
        Args:
            images_folder: Thư mục gốc chứa ảnh DocVQA
            output_folder: Thư mục lưu kết quả
            subsets: Các tập dữ liệu cần xử lý
            max_images_per_subset: Số ảnh tối đa mỗi subset (None = xử lý tất cả)
            use_preprocessing: Có áp dụng preprocessing không
            max_size: Kích thước tối đa cho preprocessing
            
        Returns:
            Dict chứa thống kê theo từng subset
        """
        all_stats = {}
        
        for subset in subsets:
            subset_folder = os.path.join(images_folder, subset)
            if not os.path.exists(subset_folder):
                print(f"Không tìm thấy: {subset_folder}")
                continue
            
            print(f"\n{'='*60}")
            print(f"XỬ LÝ TẬP: {subset.upper()}")
            print(f"{'='*60}")
            
            output_subset = os.path.join(output_folder, subset)
            stats = self.process_batch(
                subset_folder, 
                output_subset, 
                max_images=max_images_per_subset,
                use_preprocessing=use_preprocessing,
                max_size=max_size
            )
            all_stats[subset] = stats
        
        return all_stats


# Chạy trực tiếp để test
if __name__ == "__main__":
    # Khởi tạo processor
    processor = PaddleOCRProcessor()
    
    # Test với 1 ảnh
    # result = processor.run_ocr("path/to/image.png")
    # print(result)
    
    # Xử lý batch cho DocVQA
    # processor.process_docvqa_images(
    #     images_folder="dataset/DocVQA_Images",
    #     output_folder="dataset/DocVQA_OCR"
    # )
