import sys
sys.path.insert(0, '..')

import os
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image

from src.ocr.ocr_processor import PaddleOCRProcessor

# Đường dẫn input (ảnh DocVQA đã trích xuất)
IMAGES_FOLDER = "../dataset/DocVQA_Images"

# Đường dẫn output (kết quả OCR)
OUTPUT_FOLDER = "../dataset/DocVQA_OCR"

# Các tập dữ liệu
SUBSETS = ["train", "test", "validation"]

print(f"Input folder: {os.path.abspath(IMAGES_FOLDER)}")
print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")