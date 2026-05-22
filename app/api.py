"""FastAPI serving layer.

    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /query    {"question": "..."}        -> answer + retrieved + latency + cost
    POST /feedback {"query_id": "...", "rating": 1|-1, "note": "..."}
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from vngraphrag.config import Config
from vngraphrag.rag.pipeline import GraphRAGPipeline

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.load()
    STATE["pipeline"] = GraphRAGPipeline.from_artifacts(cfg)
    STATE["llm_available"] = STATE["pipeline"].generator.available
    yield
    STATE.clear()


app = FastAPI(title="Vietnamese Graph RAG", version="0.1.0", lifespan=lifespan)


class QueryIn(BaseModel):
    question: str
    top_k: int | None = None


class FeedbackIn(BaseModel):
    query_id: str
    rating: int
    note: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "llm_available": STATE.get("llm_available", False)}


@app.post("/query")
def query(body: QueryIn):
    return STATE["pipeline"].answer(body.question, top_k=body.top_k)


@app.post("/feedback")
def feedback(body: FeedbackIn):
    STATE["pipeline"].feedback(body.query_id, body.rating, body.note)
    return {"ok": True}
