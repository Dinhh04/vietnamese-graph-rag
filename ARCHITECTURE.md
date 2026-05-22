# Kiến trúc — Vietnamese Graph RAG (LLMOps)

Tài liệu này mô tả cách hệ thống được tổ chức để đọc **top-down**: tổng quan → luồng dữ liệu →
trách nhiệm từng module → hợp đồng dữ liệu (data contracts) → quyết định thiết kế → điểm mở rộng.

---

## 1. Tổng quan

Hệ thống hỏi–đáp e-commerce tiếng Việt theo kiến trúc **Graph RAG**: truy xuất review thật
(UIT-ViSFD smartphone + Shopee đa sản phẩm), bổ sung ngữ cảnh từ **Knowledge Graph**, rồi sinh
câu trả lời bằng LLM. Có 2 pha tách biệt:

- **Offline (build)** — encode corpus bằng PhoBERT, dựng index + KG, lưu thành *artifacts có version*.
- **Online (serve)** — nạp artifacts, mỗi truy vấn: retrieve → graph context → generate, có log đo đạc.

---

## 2. Luồng dữ liệu

```
                          ┌──────────────── OFFLINE (build_index / notebook §7) ───────────────┐
   data/raw/              │                                                                     │
   ├ UIT-ViSFD  ─┐        │   data.build_records ─► embeddings.encode_mean ─► index.save ──┐    │
   └ Shopee     ─┴──────► │                          (PhoBERT, GPU)                        ├──► artifacts/
                          │   data.{load_*}      ─► kg.build_kg ───────────► kg.save ──────┘    │  ├ doc_vectors.npy
                          └─────────────────────────────────────────────────────────────────────┤  ├ records.json
                                                                                                 │  ├ manifest.json (version)
                          ┌──────────────────── ONLINE (FastAPI / Gradio) ──────────────────────┤  └ kg.pkl
                          │                                                                      │
   user question ──► pipeline.answer():                                                          │
                          │   retrieval.retrieve  (bi-encoder → MaxSim rerank → graph boost)     │◄── nạp lại
                          │   kg.graph_query + product_context     (ngữ cảnh KG)                 │   (index.load/kg.load)
                          │   generate.generate   (OpenAI, grounded prompt)                      │
                          │   observability.log_query (latency / tokens / cost) ──► logs/queries.jsonl
                          └──────────────────────────────────────────────────────────────────────┘
                                          ▲
   evaluate.run_eval ── P@k / MRR ablation ── regression gate (CI) ──► artifacts/metrics.json
   UI 👍/👎 ─────────────────────────────────────────────────────────► logs/feedback.jsonl
```

---

## 3. Trách nhiệm từng module

Bố cục theo tầng (chiều phụ thuộc: `cli → rag → core → hạ tầng`):

```
src/vngraphrag/
├── config.py · observability.py     ⚙️ hạ tầng dùng chung
├── core/   data · embeddings · index · kg        📚 dữ liệu + biểu diễn
├── rag/    retrieval · generate · pipeline        🔗 suy luận
└── cli/    build_index · evaluate · import_artifacts   ⌨️ entrypoints
```

| Tầng | Module | Trách nhiệm | Phụ thuộc nặng |
|---|---|---|---|
| Hạ tầng | [`config.py`](src/vngraphrag/config.py) | Cấu hình từ YAML + env; secret chỉ qua env | — |
| Hạ tầng | [`observability.py`](src/vngraphrag/observability.py) | Log latency/token/cost + feedback → JSONL | — |
| core | [`core/data.py`](src/vngraphrag/core/data.py) | Load dữ liệu, parse nhãn (regex `[\w&]`), keyword→aspect, gazetteer, `build_records` | pandas |
| core | [`core/embeddings.py`](src/vngraphrag/core/embeddings.py) | `PhoBERTEncoder` (mean-pool + token-level), `maxsim` | torch* |
| core | [`core/index.py`](src/vngraphrag/core/index.py) | `DocumentIndex`: vector + meta, save/load **có version** | numpy |
| core | [`core/kg.py`](src/vngraphrag/core/kg.py) | Dựng/lưu/nạp Knowledge Graph, `graph_query`, `product_context` | networkx* |
| rag | [`rag/retrieval.py`](src/vngraphrag/rag/retrieval.py) | `HybridRetriever`: bi-encoder → MaxSim → graph boost | numpy |
| rag | [`rag/generate.py`](src/vngraphrag/rag/generate.py) | `Generator` (OpenAI) + `build_prompt` grounding | openai* |
| rag | [`rag/pipeline.py`](src/vngraphrag/rag/pipeline.py) | `GraphRAGPipeline.answer()`: orchestrate + observability | — |
| cli | [`cli/build_index.py`](src/vngraphrag/cli/build_index.py) | Build & persist index + KG | — |
| cli | [`cli/evaluate.py`](src/vngraphrag/cli/evaluate.py) | P@k/MRR ablation + regression gate | numpy |
| cli | [`cli/import_artifacts.py`](src/vngraphrag/cli/import_artifacts.py) | Xác nhận artifacts xuất từ notebook | — |
| serving | [`app/api.py`](app/api.py) | FastAPI `/health` `/query` `/feedback` | fastapi |
| serving | [`app/ui.py`](app/ui.py) | Gradio UI + feedback 👍/👎 | gradio |

`*` = **import lười** (chỉ nạp khi dùng) → import package rất nhẹ, CI và API khởi động không cần GPU stack.

---

## 4. Hợp đồng dữ liệu (data contracts)

**Record** (1 review, dùng xuyên suốt index/retrieval):
```python
{"raw": str, "source": "UIT-ViSFD"|"Shopee", "gold": set[str],
 "product": str|None, "shop": str|None, "rating": float|None}
```

**artifacts/** (định dạng `DocumentIndex` — notebook §7 và `build_index` cùng tạo ra):
```
doc_vectors.npy   float32 (N, dim)  — đã L2-normalize (cosine = dot)
records.json      list[record]      — gold lưu dưới dạng list đã sort
manifest.json     {model, n_records, version}   version = sha256(f"{model}|{N}")[:12]
kg.pkl            networkx.DiGraph (pickle)
```
`DocumentIndex.load()` trả `None` nếu `version` không khớp → buộc rebuild. Đây là cơ chế chống
"index cũ phục vụ model mới".

**logs/queries.jsonl** (mỗi dòng 1 truy vấn):
```python
{"id","ts","question","model","latency_ms","n_retrieved",
 "prompt_tokens","completion_tokens","cost_usd","sources"}
```

---

## 5. Vòng đời một truy vấn `/query`

```
POST /query {question}
  └► GraphRAGPipeline.answer()
       ├ retriever.retrieve(question)
       │    ├ encoder.encode_mean(q)         → cosine toàn corpus → Top-50 candidate
       │    ├ encoder.encode_tokens(q,doc)   → MaxSim rerank (attention late-interaction)
       │    └ aspects_from_text(doc)         → graph boost (tín hiệu nội dung, KHÔNG dùng nhãn vàng)
       ├ _graph_context(question)            → graph_query(aspect) + product_context()
       ├ generate.build_prompt + Generator   → OpenAI (hoặc fallback nếu thiếu key)
       └ observability.log_query             → latency + token + cost → JSONL
  ◄ {id, answer, graph_context, retrieved, latency_ms, cost_usd}
```

---

## 6. Quyết định thiết kế (vì sao như vậy)

- **MaxSim late-interaction thay vì reranker huấn luyện**: cơ chế attention có ý nghĩa, *không tham
  số ngẫu nhiên*, không cần train — tránh đưa nhiễu vào xếp hạng.
- **Graph boost theo nội dung, không theo nhãn vàng**: nhãn vàng chỉ dùng để *chấm điểm* (eval);
  nếu boost theo chính nhãn đang chấm thì rò rỉ metric → Graph RAG "thắng giả".
- **Index có version**: artifacts tái dùng được giữa notebook (GPU) và serving (CPU); version chặn
  việc nạp nhầm index cũ.
- **Import lười cho torch/networkx/openai**: import package nhẹ → CI chạy unit test không cần GPU.
- **Secret chỉ qua env**: không bao giờ ghi key vào YAML/repo.

---

## 7. Điểm mở rộng

| Muốn đổi | Sửa ở đâu |
|---|---|
| Thay vector store (FAISS/Qdrant) | `index.py` (giữ API `search`) |
| Thêm dataset mới | `data.load_*` + `build_records` |
| Đổi LLM (provider khác) | `generate.Generator` |
| Đổi trọng số retrieval | `config.yaml › retrieval` |
| Thêm metric đánh giá | `evaluate.py` |
| Thêm endpoint | `app/api.py` |
