# Vietnamese Graph RAG — NLP Final Project
# Hệ thống Hỏi đáp Tiếng Việt dựa trên Đồ thị Tri thức

## Nhóm thực hiện
| Thành viên | MSSV | Vai trò |
|---|---|---|
| Lê Phạm Hồng Hiên | 22110059 | Nhóm trưởng |
| Dương Bùi Phương Đình | 22110049 | Thành viên |
| Phạm Xuân Huyên | 22110076 | Thành viên |

## Mô tả dự án
Xây dựng hệ thống **Graph RAG** (Retrieval-Augmented Generation) cho **tiếng Việt**, kết hợp Knowledge Graph với các phương pháp NLP cốt lõi: TF-IDF, Word2Vec, GloVe, Attention, PhoBERT.

## Cấu trúc dự án
```
final_project/
├── README.md                            ← File này
├── requirements.txt                     ← Dependencies
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      ← Đình
│   ├── 02_embedding_comparison.ipynb    ← Huyên
│   ├── 03_ner_extraction.ipynb          ← Đình
│   ├── 04_knowledge_graph.ipynb         ← Huyên
│   ├── 05_retrieval_evaluation.ipynb    ← Hiên
│   └── 06_full_pipeline_demo.ipynb      ← Hiên
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── ner_model.py
│   ├── knowledge_graph.py
│   ├── retrieval.py
│   └── rag_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
│   └── figures/
└── report/
    └── NLP_Final_Report.tex
```

## Cài đặt
```bash
pip install -r requirements.txt
```

## Dataset
- **UIT-ViQuAD 2.0**: `taidng/UIT-ViQuAD2.0`
- **PhoNER_COVID19**: `SEACrowd/pho_ner_covid`
- **Wikipedia tiếng Việt**: `tdtunlp/wikipedia_vi`
