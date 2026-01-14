# DocVQA Semantic Layout Graph Pipeline

Dự án xây dựng pipeline sinh dữ liệu cho bài toán Document VQA (Visual Question Answering), tập trung vào việc mô hình hóa tài liệu dưới dạng **Semantic Layout Graph** để tạo ra các câu hỏi phức tạp đòi hỏi suy luận đa thành phần (cross-element reasoning).

## 🚀 Tính năng chính

*   **Advanced OCR**: Tích hợp **PaddleOCR** với các bước tiền xử lý ảnh (deskew, denoise) và gán nhãn ngữ nghĩa cho token (date, form key-values).
*   **Semantic Layout Analysis**: Phân tích cấu trúc tài liệu, phân loại vùng (Text, Table, Form, Figure) và nhóm các thành phần ngữ nghĩa.
*   **Graph Construction**: Xây dựng đồ thị ngữ nghĩa $G=(V, E)$ kết nối các thành phần tài liệu thông qua quan hệ không gian (spatial) và ngữ nghĩa (semantic).
*   **Hybrid QA Generation**: Sinh câu hỏi/đáp án tự động kết hợp:
    *   **Rule-based**: Cho các thông tin trích xuất trực tiếp (Form keys, Dates).
    *   **LLM-based**: Cho các câu hỏi suy luận phức tạp (Text ↔ Table, Figure ↔ Caption).

## 📂 Cấu trúc dự án

*   **`pipeline/`**: Các notebook thực thi từng bước của quy trình (Download -> OCR -> Batch Process -> QA Gen -> Full Pipeline).
*   **`src/ocr/`**: Module xử lý OCR, phân tích layout và token classification.
*   **`src/graph/`**: Module xây dựng đồ thị quan hệ giữa các thành phần tài liệu.
*   **`src/qa/`**: Module sinh câu hỏi sử dụng template và LLM.
*   **`dataset/`**: Chứa dữ liệu DocVQA (raw images, labels).
*   **`output/`**: Kết quả đầu ra (JSON graph, CSV dataset).

## 🛠️ Cài đặt & Sử dụng

1.  Cài đặt thư viện: `pip install -r requirements.txt`
2.  Chạy pipeline theo thứ tự trong thư mục `pipeline/`:
    *   `1_Download_data.ipynb`: Tải dữ liệu.
    *   `2_PaddleOCR.ipynb`: Chạy thử nghiệm OCR.
    *   `5_full_pipeline.ipynb`: Chạy toàn bộ luồng xử lý và sinh QA.

## 📊 Kết quả (Output)

Dữ liệu đầu ra được lưu dưới dạng JSON chứa thông tin chi tiết về OCR, Layout và Graph của từng tài liệu, kèm theo file CSV tổng hợp các cặp câu hỏi-đáp án đã sinh ra.
