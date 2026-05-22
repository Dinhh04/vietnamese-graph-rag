# Vietnamese Graph RAG for E-commerce 🛒
### Hệ thống Hỏi đáp Thông minh cho Thương mại Điện tử Tiếng Việt

> NLP Final Project — ĐH Khoa học Tự nhiên, ĐHQG-HCM

## 👥 Nhóm thực hiện
| Thành viên | MSSV | Vai trò |
|---|---|---|
| Lê Phạm Hồng Hiên | 22110059 | Nhóm trưởng — Kiến trúc, RAG Pipeline |
| Dương Bùi Phương Đình | 22110049 | Data Engineering, NER |
| Phạm Xuân Huyên | 22110076 | Embedding, Knowledge Graph |

## 📋 Mô tả dự án

Xây dựng hệ thống **Graph RAG** (Retrieval-Augmented Generation) cho domain **Thương mại Điện tử tiếng Việt**. Hệ thống cho phép người dùng đặt câu hỏi về sản phẩm (ví dụ: *"Camera iPhone 15 chụp đêm có tốt không?"*) và trả lời dựa trên đánh giá thực tế từ Shopee/Lazada, kết hợp Knowledge Graph để tăng chất lượng truy xuất.

### Kiến thức NLP áp dụng
- **Week 1**: TF-IDF — Baseline retrieval
- **Week 2**: Word2Vec — Document embedding
- **Week 3**: GloVe — Co-occurrence embedding  
- **Week 4**: Attention — Re-ranking documents
- **Week 5**: PhoBERT/Transformer — SOTA embedding + NER

## 📦 Dataset

| Dataset | Kích thước | Nội dung | Nguồn |
|---|---|---|---|
| **UIT-ViSFD** | 11,122 comments (train 7,786) | Review smartphone, **10 aspects** (gồm SER&ACC), 3 sentiments | Shopee |
| **PhoNER_COVID19** | — | NER tiếng Việt (domain COVID — tải về tham khảo, **không dùng** để gán nhãn sản phẩm) | HuggingFace |

> Ghi chú: 10 aspect = `SCREEN, CAMERA, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, FEATURES, SER&ACC`.
> NER thương hiệu/cửa hàng dùng `underthesea` (NER tổng quát PER/LOC/ORG) + gazetteer thương hiệu.

## 🏗️ Cấu trúc dự án (thực tế)
```
vietnamese-graph-rag/
├── README.md
├── requirements.txt
├── build_notebooks.py                       # sinh lại 2 notebook bên dưới
├── download_data.py
├── notebooks/
│   ├── kaggle_part1_embedding_ner.ipynb     # Data · Preprocess · NER · So sánh Embedding
│   └── kaggle_part2_graph_rag.ipynb         # Knowledge Graph · Attention rerank · Graph RAG
├── data/
│   └── raw/                                  # UIT-ViSFD (Train/Dev/Test.csv), PhoNER, stopwords
└── report/
    ├── NLP_Final_Report.tex
    └── figures/
```

## 🚀 Chạy trên Kaggle
1. Tạo Kaggle Notebook mới, upload `notebooks/kaggle_part1_embedding_ner.ipynb`.
2. **Settings**: bật **Internet ON** và **Accelerator = GPU T4**.
3. (Part 2) Thêm OpenAI API key: **Add-ons → Secrets → New secret**, tên `OPENAI_API_KEY`
   (lấy tại https://platform.openai.com/api-keys). Mặc định model `gpt-4o-mini` — đổi `LLM_MODEL` trong notebook nếu muốn `gpt-4o`/`gpt-3.5-turbo`. Không có key vẫn chạy được phần Retrieval + Graph.
4. Run All. Part 1 → `data/train_processed.csv` + `results_part1.json`; Part 2 tự chứa (không cần output Part 1).

Chạy local:
```bash
pip install -r requirements.txt
```

## 🔧 Các lỗi đã sửa so với bản đầu
| # | Lỗi | Cách sửa |
|---|---|---|
| 1 | Regex `\{(\w+)#(\w+)\}` làm **mất aspect SER&ACC** (1.995 lần) | Đổi thành `\{([\w&]+)#(\w+)\}` |
| 2 | Attention reranker `nn.Linear` **chưa train** → nhiễu ngẫu nhiên | Thay bằng **late-interaction MaxSim** (ColBERT-style, không cần train) |
| 3 | **Rò rỉ metric**: boost theo đúng nhãn đang chấm điểm | Boost theo aspect **quan sát trong nội dung**; chấm điểm bằng **nhãn vàng** (độc lập) |
| 4 | Aspect query chỉ khớp tiếng Anh | Thêm **bảng ánh xạ keyword tiếng Việt → aspect** |
| 5 | So sánh embedding bằng "avg cosine" (vô nghĩa) | Thay bằng **Precision@k & MRR** |
| 6 | NER tuyên bố sai domain | Dùng `underthesea` + gazetteer, **nối brand vào KG**, mô tả trung thực |
| 7 | README/báo cáo lệch thực tế | Đồng bộ cấu trúc + điền số liệu từ notebook |

## 📄 License
Academic use only — NLP Final Project 2025-2026.
