"""Hybrid retrieval: PhoBERT bi-encoder -> top-N candidates -> attention (MaxSim) rerank
+ graph boost (aspect quan sát trong nội dung, KHÔNG dùng nhãn vàng -> không rò rỉ metric)."""

from __future__ import annotations

import numpy as np

from ..core import aspect_from_query, aspects_from_text, maxsim


def _nz(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else x * 0.0


class HybridRetriever:
    def __init__(self, index, encoder, cfg):
        self.index = index
        self.encoder = encoder
        self.cfg = cfg

    def retrieve(
        self, query: str, top_k: int | None = None, weights: tuple[float, float, float] | None = None
    ) -> list[dict]:
        c = self.cfg.retrieval
        top_k = top_k or c.top_k
        w_bi, w_attn, w_graph = weights or (c.w_bi, c.w_attn, c.w_graph)

        q_mean = self.encoder.encode_mean([query])[0]
        cand, sims = self.index.search(q_mean, c.n_candidates)
        bi = sims[cand]

        q_tok = self.encoder.encode_tokens(query)
        attn = np.array([maxsim(q_tok, self.encoder.encode_tokens(self.index.records[i]["raw"])) for i in cand])

        q_asp = aspect_from_query(query)
        graph = np.array(
            [1.0 if (q_asp and q_asp in aspects_from_text(self.index.records[i]["raw"])) else 0.0 for i in cand]
        )

        combined = w_bi * _nz(bi) + w_attn * _nz(attn) + w_graph * graph
        order = combined.argsort()[::-1][:top_k]

        results = []
        for o in order:
            i = int(cand[o])
            r = self.index.records[i]
            results.append(
                {
                    "idx": i,
                    "text": r["raw"],
                    "source": r.get("source"),
                    "product": r.get("product"),
                    "shop": r.get("shop"),
                    "rating": r.get("rating"),
                    "score": float(combined[o]),
                }
            )
        return results
