"""Gradio frontend. Talks to the FastAPI backend if VNGR_API_URL is set, otherwise
loads the pipeline in-process. Includes 👍/👎 feedback that is logged for the LLMOps loop.

    python -m app.ui            # in-process
    VNGR_API_URL=http://localhost:8000 python -m app.ui   # via API
"""

from __future__ import annotations

import os

import gradio as gr

API_URL = os.environ.get("VNGR_API_URL")
_PIPE = None


def _answer(question: str) -> dict:
    global _PIPE
    if API_URL:
        import requests

        return requests.post(f"{API_URL}/query", json={"question": question}, timeout=120).json()
    if _PIPE is None:
        from vngraphrag.config import Config
        from vngraphrag.rag.pipeline import GraphRAGPipeline

        _PIPE = GraphRAGPipeline.from_artifacts(Config.load())
    return _PIPE.answer(question)


def _send_feedback(query_id: str, rating: int):
    if not query_id:
        return "Chưa có câu hỏi nào."
    if API_URL:
        import requests

        requests.post(f"{API_URL}/feedback", json={"query_id": query_id, "rating": rating}, timeout=30)
    elif _PIPE is not None:
        _PIPE.feedback(query_id, rating)
    return "Đã ghi nhận phản hồi 🙏"


def build_ui():
    with gr.Blocks(title="Vietnamese Graph RAG — E-commerce") as demo:
        gr.Markdown("# 🛒 Vietnamese Graph RAG — E-commerce QA\nUIT-ViSFD (smartphone) + Shopee (đa sản phẩm).")
        qid = gr.State("")
        q = gr.Textbox(label="Câu hỏi", value="Camera chụp đêm có tốt không?")
        btn = gr.Button("Hỏi", variant="primary")
        ans = gr.Textbox(label="💬 Trả lời")
        with gr.Row():
            kg = gr.Textbox(label="🕸️ Knowledge Graph context")
            refs = gr.Textbox(label="📄 Review trích dẫn")
        meta = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍 Hữu ích")
            down = gr.Button("👎 Chưa tốt")
        fb = gr.Markdown()

        gr.Examples(
            [
                ["Pin dùng được bao lâu?"],
                ["Áo hoodie chất vải thế nào?"],
                ["Shop giao hàng có nhanh không?"],
                ["Máy nào giá rẻ mà camera tốt?"],
            ],
            inputs=q,
        )

        def run(question):
            r = _answer(question)
            refs_txt = "\n\n".join(
                f"[{d.get('source')}{(' · ' + str(d['product'])) if d.get('product') else ''}] {d['text'][:140]}"
                for d in r.get("retrieved", [])
            )
            m = f"⏱️ {r.get('latency_ms', 0)} ms · 💵 ${r.get('cost_usd', 0)}"
            return r.get("answer", ""), r.get("graph_context", ""), refs_txt, m, r.get("id", "")

        btn.click(run, inputs=q, outputs=[ans, kg, refs, meta, qid])
        up.click(lambda i: _send_feedback(i, 1), inputs=qid, outputs=fb)
        down.click(lambda i: _send_feedback(i, -1), inputs=qid, outputs=fb)
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
