"""PhoBERT encoder: mean-pooled doc vectors + token-level vectors for late interaction.

torch / transformers / underthesea are imported lazily so that lightweight modules
(data, config, observability) can be used in CI without a GPU stack.
"""

from __future__ import annotations

import numpy as np


class PhoBERTEncoder:
    def __init__(self, model_name: str = "vinai/phobert-base-v2", max_seq_len: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from underthesea import word_tokenize

        self._torch = torch
        self._seg = lambda t: word_tokenize(str(t)[:256], format="text")
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def seg(self, text: str) -> str:
        return self._seg(text)

    def encode_mean(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    [self.seg(t) for t in texts[i : i + batch_size]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1)
                vecs = (h * m).sum(1) / m.sum(1)
                out.append(vecs.cpu().numpy())
        arr = np.vstack(out).astype("float32")
        # L2 normalize -> cosine == dot product
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norm, 1e-9, None)

    def encode_tokens(self, text: str) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(
                [self.seg(text)], padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(self.device)
            h = self.model(**enc).last_hidden_state[0]
            mask = enc["attention_mask"][0].bool()
            h = h[mask].cpu().numpy().astype("float32")
        norm = np.linalg.norm(h, axis=1, keepdims=True)
        return h / np.clip(norm, 1e-9, None)


def maxsim(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    """Late-interaction (ColBERT-style) score: trung bình max-cosine của token query với token doc."""
    if len(q_tokens) == 0 or len(d_tokens) == 0:
        return 0.0
    sim = q_tokens @ d_tokens.T  # (Lq, Ld) cosine vì đã normalize
    return float(sim.max(axis=1).mean())
