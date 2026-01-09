"""
Module tải và xử lý dữ liệu DocVQA từ HuggingFace.
Bao gồm:
- Tải dataset từ HuggingFace
- Trích xuất ảnh từ file parquet
- Tạo file CSV labels
"""

import os
import io
import glob
from typing import Optional, List
import pandas as pd
from PIL import Image
from tqdm import tqdm


class DocVQADownloader:
    """Class xử lý tải và tiền xử lý dữ liệu DocVQA."""
    
    def __init__(
        self,
        repo_id: str = "lmms-lab/DocVQA",
        local_dir: str = "dataset",
        raw_folder: str = "dataset/DocVQA_raw",
        images_folder: str = "dataset/DocVQA_Images",
        labels_folder: str = "dataset/DocVQA_Labels"
    ):
        """
        Khởi tạo DocVQADownloader.
        
        Args:
            repo_id: ID repo trên HuggingFace
            local_dir: Thư mục lưu dataset
            raw_folder: Thư mục chứa file parquet gốc
            images_folder: Thư mục lưu ảnh đã trích xuất
            labels_folder: Thư mục lưu file CSV labels
        """
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.raw_folder = raw_folder
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        
    def download_from_huggingface(self, force_download: bool = False) -> str:
        """
        Tải dataset DocVQA từ HuggingFace.
        
        Args:
            force_download: Nếu True, tải lại ngay cả khi đã tồn tại
        
        Returns:
            str: Đường dẫn tuyệt đối đến thư mục dataset
        """
        dataset_path = os.path.abspath(os.path.join(self.local_dir, "DocVQA"))
        
        # Kiểm tra xem dataset đã tồn tại chưa
        if os.path.exists(dataset_path) and not force_download:
            # Kiểm tra xem có file parquet trong thư mục không
            parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
            if parquet_files:
                print(f"Dataset đã tồn tại tại: {dataset_path}")
                print(f"Tìm thấy {len(parquet_files)} file parquet. Bỏ qua tải xuống.")
                print("Để tải lại, sử dụng: download_from_huggingface(force_download=True)")
                return dataset_path
        
        print("Đang tải dataset từ HuggingFace...")
        from huggingface_hub import snapshot_download
        
        os.makedirs(self.local_dir, exist_ok=True)
        
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=self.local_dir,
            allow_patterns=["DocVQA/**"],
            local_dir_use_symlinks=False,
            resume_download=True
        )  # type: ignore
        
        print(f"Done! Dataset root: {dataset_path}")
        return dataset_path
    
    @staticmethod
    def _get_subset_name(filename: str) -> str:
        """Xác định tập dữ liệu dựa trên tên file."""
        if "train" in filename:
            return "train"
        elif "test" in filename:
            return "test"
        elif "val" in filename:
            return "validation"
        return "others"
    
    @staticmethod
    def _extract_image_bytes(img_data) -> Optional[bytes]:
        """Trích xuất bytes từ dữ liệu ảnh."""
        if isinstance(img_data, dict) and 'bytes' in img_data:
            return img_data['bytes']
        elif isinstance(img_data, bytes):
            return img_data
        return None
    
    @staticmethod
    def _generate_image_name(row: pd.Series, index: int, subset: str) -> str:
        """Tạo tên file ảnh từ row data."""
        if 'questionId' in row and pd.notna(row.get('questionId')):
            return f"{row['questionId']}.png"
        elif 'docId' in row and pd.notna(row.get('docId')):
            return f"{row['docId']}_{index}.png"
        return f"{subset}_{index}.png"
    
    def extract_images_from_parquet(self, force_extract: bool = False) -> None:
        """
        Trích xuất ảnh từ các file parquet và lưu thành file PNG.
        
        Args:
            force_extract: Nếu True, trích xuất lại ngay cả khi đã tồn tại
        """
        # Kiểm tra xem đã trích xuất ảnh chưa
        existing_images = 0
        for subset in ["train", "test", "validation"]:
            folder = os.path.join(self.images_folder, subset)
            if os.path.exists(folder):
                existing_images += len(glob.glob(os.path.join(folder, "*.png")))
        
        if existing_images > 0 and not force_extract:
            print(f"Đã tìm thấy {existing_images:,} ảnh trong {self.images_folder}")
            print("Bỏ qua trích xuất. Để trích xuất lại, dùng: extract_images_from_parquet(force_extract=True)")
            return
        
        parquet_files = glob.glob(os.path.join(self.raw_folder, "*.parquet"))
        print(f"Tìm thấy {len(parquet_files)} file parquet.")
        
        for file_path in parquet_files:
            file_name = os.path.basename(file_path)
            print(f"Đang xử lý: {file_name}...")
            
            subset = self._get_subset_name(file_name)
            output_dir = os.path.join(self.images_folder, subset)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Lỗi đọc file {file_name}: {e}")
                continue
            
            if 'image' not in df.columns:
                print(f"Không tìm thấy cột 'image' trong file {file_name}. Các cột có: {df.columns.tolist()}")
                continue
            
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Extracting {subset}"):
                try:
                    image_bytes = self._extract_image_bytes(row['image'])
                    
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        img_name = self._generate_image_name(row, index, subset)
                        save_path = os.path.join(output_dir, img_name)
                        image.convert("RGB").save(save_path, "PNG")
                        
                except Exception as e:
                    print(f"Lỗi lưu ảnh tại dòng {index}: {e}")
        
        print("\nHoàn tất quá trình trích xuất ảnh!")
    
    def generate_labels_csv(self, force_generate: bool = False) -> None:
        """
        Tạo file CSV chứa labels (question, answer, image_path) từ file parquet.
        
        Args:
            force_generate: Nếu True, tạo lại ngay cả khi đã tồn tại
        """
        # Kiểm tra xem đã có file CSV chưa
        existing_csvs = glob.glob(os.path.join(self.labels_folder, "*_labels.csv"))
        if existing_csvs and not force_generate:
            print(f"Đã tìm thấy {len(existing_csvs)} file labels CSV trong {self.labels_folder}:")
            for csv_file in existing_csvs:
                print(f"  - {os.path.basename(csv_file)}")
            print("Bỏ qua tạo labels. Để tạo lại, dùng: generate_labels_csv(force_generate=True)")
            return
        
        parquet_folder = os.path.join(self.local_dir, "DocVQA")
        parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
        
        # Gom nhóm file theo tập
        files_by_subset = {}
        for f in parquet_files:
            key = self._get_subset_name(f)
            if key not in files_by_subset:
                files_by_subset[key] = []
            files_by_subset[key].append(f)
        
        # Xử lý từng tập
        for subset, files in files_by_subset.items():
            print(f"\n--- Đang tạo CSV cho tập: {subset.upper()} ---")
            metadata_list = []
            
            for file_path in files:
                print(f"Đọc metadata từ: {os.path.basename(file_path)}")
                
                try:
                    df = pd.read_parquet(file_path)
                    if 'image' in df.columns:
                        df = df.drop(columns=['image'])
                except Exception as e:
                    print(f"Lỗi đọc file {file_path}: {e}")
                    continue
                
                for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Mapping"):
                    img_name = self._generate_image_name(row, index, subset)
                    relative_path = os.path.join(subset, img_name)
                    
                    record = {
                        'questionId': row.get('questionId', ''),
                        'question': row.get('question', ''),
                        'docId': row.get('docId', ''),
                        'image_filename': img_name,
                        'image_path': relative_path
                    }
                    
                    # Xử lý answers
                    answers = row.get('answers', [])
                    record['answers'] = str(list(answers)) if isinstance(answers, (list, tuple)) else str(answers)
                    
                    metadata_list.append(record)
            
            # Lưu file CSV
            if metadata_list:
                os.makedirs(self.labels_folder, exist_ok=True)
                csv_filename = f"{subset}_labels.csv"
                save_path = os.path.join(self.labels_folder, csv_filename)
                
                df_out = pd.DataFrame(metadata_list)
                df_out.to_csv(save_path, index=False, encoding='utf-8')
                print(f"-> Đã lưu: {save_path} ({len(df_out)} dòng)")
        
        print("\nHOÀN TẤT TẠO LABEL!")
    
    def get_sample_images(self, subset: str = "train", n_samples: int = 5) -> List[str]:
        """
        Lấy danh sách đường dẫn ảnh mẫu để hiển thị.
        
        Args:
            subset: Tập dữ liệu (train/test/validation)
            n_samples: Số lượng ảnh mẫu
            
        Returns:
            List đường dẫn ảnh
        """
        image_folder = os.path.join(self.images_folder, subset)
        if not os.path.exists(image_folder):
            print(f"Thư mục {image_folder} không tồn tại!")
            return []
        
        image_files = glob.glob(os.path.join(image_folder, "*.png"))[:n_samples]
        return image_files
    
    def get_dataset_stats(self) -> dict:
        """
        Lấy thống kê về dataset.
        
        Returns:
            dict chứa số lượng ảnh theo từng subset
        """
        stats = {}
        for subset in ["train", "test", "validation"]:
            folder = os.path.join(self.images_folder, subset)
            if os.path.exists(folder):
                count = len(glob.glob(os.path.join(folder, "*.png")))
                stats[subset] = count
            else:
                stats[subset] = 0
        return stats


# Chạy trực tiếp để test
if __name__ == "__main__":
    downloader = DocVQADownloader()
    
    # Bước 1: Tải từ HuggingFace (bỏ comment nếu cần)
    # downloader.download_from_huggingface()
    
    # Bước 2: Trích xuất ảnh
    # downloader.extract_images_from_parquet()
    
    # Bước 3: Tạo labels CSV
    # downloader.generate_labels_csv()
    
    # Kiểm tra stats
    stats = downloader.get_dataset_stats()
    print("Dataset statistics:", stats)
