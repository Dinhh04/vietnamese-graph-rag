"""Unit tests cho phần lõi không cần GPU (parsing, keyword mapping, cost, config, index IO).
Chạy được trong CI mà không cần torch/transformers."""

import numpy as np

from vngraphrag.config import Config
from vngraphrag.core import (
    DocumentIndex,
    aspect_from_query,
    aspects_from_text,
    build_records,
    detect_brand,
    parse_label,
)
from vngraphrag.observability import estimate_cost


def test_parse_label_keeps_serracc():
    pairs = parse_label("{CAMERA#Positive};{SER&ACC#Negative};{OTHERS};")
    assert ("SER&ACC", "Negative") in pairs
    assert ("CAMERA", "Positive") in pairs


def test_aspect_from_query_vietnamese():
    assert aspect_from_query("pin dùng được bao lâu") == "BATTERY"
    assert aspect_from_query("màn hình có sắc nét không") == "SCREEN"
    assert aspect_from_query("nhân viên tư vấn nhiệt tình") == "SER&ACC"


def test_aspects_from_text():
    found = aspects_from_text("giao hàng nhanh, giá rẻ, vải đẹp")
    assert "PRICE" in found and "SER&ACC" in found


def test_detect_brand():
    assert detect_brand("mua iPhone 15 rất ngon") == "Apple"
    assert detect_brand("không có gì") == "Unknown"


def test_estimate_cost():
    prices = {"gpt-4o-mini": [0.00015, 0.0006]}
    c = estimate_cost("gpt-4o-mini", 1000, 1000, prices)
    assert abs(c - 0.00075) < 1e-9


def test_config_load_defaults():
    cfg = Config()
    assert cfg.retrieval.top_k == 5
    assert cfg.llm.model.startswith("gpt")


def test_index_save_load(tmp_path):
    import pandas as pd

    visfd = pd.DataFrame(
        {
            "comment": ["camera đẹp pin trâu", "giao hàng nhanh shop tốt"],
            "label": ["{CAMERA#Positive};{BATTERY#Positive};", "{SER&ACC#Positive};"],
        }
    )
    visfd["parsed_labels"] = visfd["label"].apply(parse_label)
    visfd["aspects"] = visfd["parsed_labels"].apply(lambda x: [a for a, _ in x])
    recs = build_records(visfd, pd.DataFrame(columns=["comment", "product_name", "shop_name", "rating_star"]))
    assert len(recs) == 2
    idx = DocumentIndex(np.random.rand(2, 8).astype("float32"), recs, "dummy-model")
    idx.save(tmp_path)
    loaded = DocumentIndex.load(tmp_path, "dummy-model")
    assert loaded is not None and len(loaded.records) == 2
    # version mismatch on different model
    assert DocumentIndex.load(tmp_path, "other-model") is None
