"""Versioned document index: normalized PhoBERT vectors + record metadata.

Persisted to disk so serving/eval don't recompute embeddings every run.
A manifest records a version hash; load() returns None on mismatch so the
caller knows to rebuild.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _version(model: str, n_records: int) -> str:
    h = hashlib.sha256(f"{model}|{n_records}".encode()).hexdigest()[:12]
    return h


class DocumentIndex:
    def __init__(self, vectors: np.ndarray, records: list[dict], model: str):
        self.vectors = vectors  # (N, dim) L2-normalized
        self.records = records  # aligned metadata
        self.model = model
        self.version = _version(model, len(records))

    # ---- build ----
    @classmethod
    def build(cls, records: list[dict], encoder, model: str) -> DocumentIndex:
        texts = [r["raw"] for r in records]
        vectors = encoder.encode_mean(texts)
        return cls(vectors, records, model)

    # ---- search ----
    def search(self, query_vec: np.ndarray, n: int = 50):
        sims = (self.vectors @ query_vec.reshape(-1)).astype("float32")
        idx = np.argsort(sims)[::-1][:n]
        return idx, sims

    # ---- persistence ----
    def save(self, artifacts_dir: str | Path):
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "doc_vectors.npy", self.vectors)
        # gold sets -> lists for JSON
        meta = [{**r, "gold": sorted(r.get("gold", set()))} for r in self.records]
        (d / "records.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (d / "manifest.json").write_text(
            json.dumps(
                {"model": self.model, "n_records": len(self.records), "version": self.version},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, artifacts_dir: str | Path, model: str) -> DocumentIndex | None:
        d = Path(artifacts_dir)
        man = d / "manifest.json"
        if not man.exists():
            return None
        info = json.loads(man.read_text(encoding="utf-8"))
        vectors = np.load(d / "doc_vectors.npy")
        meta = json.loads((d / "records.json").read_text(encoding="utf-8"))
        for r in meta:
            r["gold"] = set(r.get("gold", []))
        obj = cls(vectors, meta, info["model"])
        if info.get("version") != _version(model, len(meta)):
            # stale: model or data changed
            return None
        return obj
