"""Observability: append one structured record per query (latency, tokens, cost, sources)
to a JSONL file. Also a helper to estimate USD cost from token usage."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, prices: dict) -> float:
    p = prices.get(model)
    if not p:
        return 0.0
    in_rate, out_rate = p
    return round(prompt_tokens / 1000 * in_rate + completion_tokens / 1000 * out_rate, 6)


class QueryLogger:
    def __init__(self, logs_dir: str = "logs"):
        self.dir = Path(logs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.query_log = self.dir / "queries.jsonl"
        self.feedback_log = self.dir / "feedback.jsonl"

    def log_query(self, record: dict) -> str:
        record = {"id": record.get("id") or uuid.uuid4().hex[:12], "ts": time.time(), **record}
        with self.query_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record["id"]

    def log_feedback(self, query_id: str, rating: int, note: str = ""):
        with self.feedback_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"query_id": query_id, "rating": int(rating), "note": note, "ts": time.time()}, ensure_ascii=False
                )
                + "\n"
            )


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = round((time.perf_counter() - self._t0) * 1000, 1)
