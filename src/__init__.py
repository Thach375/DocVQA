# VQA Dataset Pipeline
# Các module xử lý pipeline tạo dataset VQA

from .Dataset.data_downloader import DocVQADownloader
from .ocr.ocr_processor import PaddleOCRProcessor

__all__ = ['DocVQADownloader', 'PaddleOCRProcessor']
