# GraphRAG Web Interface

## 🚀 Cài đặt và Khởi chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc chỉ cài Gradio nếu đã có các package khác:

```bash
pip install gradio>=4.0.0
```

### 2. Khởi chạy GUI

**Trên Windows:**
```bash
python app.py
```

Hoặc double-click file `run_gui.bat`

**Trên Linux/Mac:**
```bash
python3 app.py
```

### 3. Truy cập Web Interface

Mở trình duyệt và truy cập:
```
http://localhost:7860
```

---

## 📖 Hướng dẫn sử dụng

### Bước 1: Khởi tạo hệ thống

1. Click nút **"🚀 Initialize System"** để load toàn bộ dữ liệu
2. Đợi thông báo "✅ System initialized successfully!"

### Bước 2: Sử dụng Local Search

**Local Search** phù hợp cho:
- Câu hỏi cụ thể về entities, relationships
- Tìm kiếm thông tin chi tiết
- Ví dụ: "Who are the key people?", "What is the relationship between X and Y?"

**Các tùy chọn:**
- **Query**: Nhập câu hỏi của bạn
- **Number of Results (top-k)**: Số lượng kết quả tìm kiếm (5-50)
- **Generate Answer**: Bật/tắt tự động generate câu trả lời
- **Ground Truth**: (Tùy chọn) Câu trả lời chuẩn để đánh giá chính xác hơn

### Bước 3: Sử dụng Global Search

**Global Search** phù hợp cho:
- Câu hỏi tổng quan, high-level
- Tìm kiếm themes, patterns
- Ví dụ: "What are the main topics?", "What are the overall themes?"

**Các tùy chọn:**
- **Query**: Nhập câu hỏi của bạn
- **Number of Communities (top-k)**: Số lượng communities tìm kiếm (3-20)
- **Generate Answer**: Bật/tắt tự động generate câu trả lời
- **Ground Truth**: (Tùy chọn) Câu trả lời chuẩn để đánh giá chính xác hơn

### Bước 4: Xem kết quả

Mỗi search sẽ trả về 3 phần:

1. **Search Results**: Danh sách kết quả tìm kiếm với scores
2. **Generated Answer**: Câu trả lời được tạo ra từ context (nếu bật)
3. **Evaluation Metrics**: 4 metrics đánh giá chất lượng
   - Relevance Score
   - Coverage Score
   - Answer Quality
   - Faithfulness
   - Overall Score

---

## 📊 Các Metrics Đánh Giá

### 1. Relevance Score (0-1)
Đo lường mức độ liên quan giữa query và kết quả tìm kiếm.
- **Cao (>0.8)**: Kết quả rất liên quan đến câu hỏi
- **Trung bình (0.5-0.8)**: Kết quả khá liên quan
- **Thấp (<0.5)**: Kết quả ít liên quan

### 2. Coverage Score (0-1)
Đo lường độ đa dạng và toàn diện của thông tin.
- **Cao (>0.8)**: Thông tin đa dạng, bao phủ nhiều khía cạnh
- **Trung bình (0.5-0.8)**: Thông tin khá đa dạng
- **Thấp (<0.5)**: Thông tin thiếu đa dạng

### 3. Answer Quality (0-1)
Đánh giá chất lượng câu trả lời (completeness, coherence, informativeness).
- **Cao (>0.8)**: Câu trả lời đầy đủ, mạch lạc, nhiều thông tin
- **Trung bình (0.5-0.8)**: Câu trả lời chấp nhận được
- **Thấp (<0.5)**: Câu trả lời kém chất lượng

### 4. Faithfulness (0-1)
Đo lường mức độ câu trả lời dựa trên context (không hallucinate).
- **Cao (>0.8)**: Câu trả lời trung thực, dựa trên context
- **Trung bình (0.5-0.8)**: Có một số thông tin ngoài context
- **Thấp (<0.5)**: Nhiều hallucination

---

## 🎨 Tùy chỉnh

### Thay đổi cổng (port)

Mở file `app.py` và thay đổi dòng cuối:

```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,  # ← Đổi port ở đây
    share=False
)
```

### Chia sẻ public link

Để tạo public link (qua Gradio):

```python
app.launch(
    share=True  # ← Đổi thành True
)
```

---

## ❓ Troubleshooting

### Lỗi: "Please initialize the system first!"
- Click nút "🚀 Initialize System" trước khi search

### Lỗi: "Module 'gradio' not found"
- Chạy: `pip install gradio>=4.0.0`

### Lỗi khi load dữ liệu
- Kiểm tra đã chạy các script build graph chưa:
  - `python scripts/1_build_graph.py`
  - `python scripts/2_detect_communities.py`
  - `python scripts/3_embed_chunks.py`
  - `python scripts/4_embed_entities.py`
  - `python scripts/5_embed_communities.py`

### Port 7860 đã được sử dụng
- Đổi port trong `app.py` hoặc tắt ứng dụng đang dùng port đó

---

## 🚀 Features

✅ Giao diện web thân thiện, dễ sử dụng
✅ Hỗ trợ cả Local và Global Search
✅ Tự động đánh giá với 4 metrics chuyên nghiệp
✅ Real-time search và generation
✅ Hỗ trợ ground truth để đánh giá chính xác hơn
✅ Responsive design, hoạt động tốt trên mọi thiết bị

---

## 📝 Ví dụ Queries

### Local Search Examples:
- "Who are the key people mentioned in the documents?"
- "What is the relationship between AI and Machine Learning?"
- "What technologies are used in the project?"
- "Who worked on deep learning?"

### Global Search Examples:
- "What are the main topics discussed?"
- "What are the overall themes in the dataset?"
- "Summarize the key findings"
- "What are the major research areas?"

---

Enjoy using GraphRAG! 🎉
